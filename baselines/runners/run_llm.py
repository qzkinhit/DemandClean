"""Replay the released foundation-model cleaning results without an API call.

    python baselines/runners/run_llm.py --dataset mercedes --check
    python baselines/runners/run_llm.py --dataset mercedes --out /tmp/mercedes.csv

The default mode uses captured input batches and structured responses. It reads
only CSV/JSON files and does not fit a downstream model. --check writes nothing.
The output preserves the full indexed table, with edits restricted to training
feature cells. The evaluator takes its training rows from that table.

Optional --live mode makes NEW paid Anthropic Messages API calls, using the
captured dirty inputs, cleaning prompt and response schemas. It provides Read and
StructuredOutput tools in a standalone adapter, rather than the original Claude
Agent SDK runtime. Its results and usage can differ from the captured runs.
ANTHROPIC_API_KEY is required only for --live. No API is called without that flag.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / 'results' / 'llm'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_csv(text):
    rows = [row for row in csv.reader(io.StringIO(text, newline='')) if row]
    if not rows:
        raise ValueError('empty CSV')
    header = [v.strip().strip('\ufeff') for v in rows[0]]
    if len(header) != len(set(header)) or 'index' not in header:
        raise ValueError('CSV needs unique column names and an index column')
    if any(len(row) != len(header) for row in rows[1:]):
        raise ValueError('CSV row width differs from its header')
    return header, rows[1:]


def csv_bytes(header, rows):
    out = io.StringIO(newline='')
    writer = csv.writer(out, lineterminator='\n')
    writer.writerow(header)
    writer.writerows(rows)
    return out.getvalue().encode('utf-8')


def edits_of(response):
    """Accept the two captured structured-output schemas without changing values."""
    if not isinstance(response, dict) or not isinstance(response.get('edits'), list):
        raise ValueError('response must contain an edits list')
    edits = []
    for item in response['edits']:
        if isinstance(item, dict) and all(k in item for k in ('index', 'column', 'value')):
            edits.append(item)
        elif isinstance(item, list) and len(item) == 3:
            edits.append(dict(index=item[0], column=item[1], value=item[2]))
        else:
            raise ValueError('malformed structured edit')
    return edits


def validate_inputs(header, rows, records, label):
    """Check that the captured inputs contain only unchanged dirty feature values."""
    if label not in header:
        raise ValueError('label column is absent from the dirty table')
    index_col = header.index('index')
    positions = {row[index_col]: i for i, row in enumerate(rows)}
    if len(positions) != len(rows):
        raise ValueError('duplicate row index in dirty table')
    features = [c for c in header if c not in ('index', label)]
    expected_header = ['index'] + features
    columns = [header.index(c) for c in expected_header]
    training = set()
    for record in records:
        inp = record['input']
        if sha256(inp['csv'].encode('utf-8')) != inp['sha256']:
            raise ValueError('captured input hash mismatch: ' + record['batch'])
        batch_header, batch_rows = read_csv(inp['csv'])
        if batch_header != expected_header:
            raise ValueError('model input includes a label or has different feature columns')
        for row in batch_rows:
            index = row[0]
            if index in training or index not in positions:
                raise ValueError('duplicate or unknown training row: ' + index)
            expected = [rows[positions[index]][c] for c in columns]
            if row != expected:
                raise ValueError('model input differs from the dirty table at index ' + index)
            training.add(index)
    return training


def replay(header, dirty_rows, records, label):
    """Apply responses as the recorded exporter did, without re-scoring."""
    training = validate_inputs(header, dirty_rows, records, label)
    rows = [row[:] for row in dirty_rows]
    positions = {row[header.index('index')]: i for i, row in enumerate(rows)}
    features = {c: header.index(c) for c in header if c not in ('index', label)}
    for record in records:
        for edit in edits_of(record['response']):
            index = str(edit['index']).strip()
            column = str(edit['column']).strip()
            # The recorded exporter ignores held-out rows, labels and unknown
            # fields. It applies the remaining edits in their recorded order.
            if index in training and column in features:
                rows[positions[index]][features[column]] = str(edit['value'])
    return csv_bytes(header, rows), training


def load_dataset(dataset):
    manifest = json.loads((BUNDLE / 'manifest.json').read_text())
    try:
        entry = manifest['datasets'][dataset]
    except KeyError:
        raise ValueError('unknown dataset; choose from ' + ', '.join(sorted(manifest['datasets'])))
    record_bytes = gzip.decompress((ROOT / entry['records']).read_bytes())
    if sha256(record_bytes) != entry['records_sha256']:
        raise ValueError('record file hash mismatch')
    records = [json.loads(line) for line in record_bytes.splitlines() if line.strip()]
    if [record['batch'] for record in records] != entry['batches']:
        raise ValueError('captured batch order or membership differs from the manifest')
    dirty_bytes = (ROOT / entry['data']).read_bytes()
    if sha256(dirty_bytes) != entry['data_sha256']:
        raise ValueError('dirty dataset hash mismatch')
    header, rows = read_csv(dirty_bytes.decode('utf-8-sig'))
    return entry, records, header, rows


def read_tool(input_csv, request, filename):
    """A live model can read only its own captured feature-only input file."""
    if request.get('file_path') not in (filename, 'input/' + filename):
        raise ValueError('Read is restricted to input/' + filename)
    offset = int(request.get('offset', 1))
    limit = int(request.get('limit', 2000))
    if offset < 1 or limit < 1:
        raise ValueError('Read offset and limit must be positive')
    lines = input_csv.splitlines()
    end = min(offset - 1 + limit, len(lines))
    return '\n'.join(str(i + 1) + '\t' + lines[i] for i in range(offset - 1, end))


def live_batch(record, system_prompt, api_key, base_url, model, max_tokens):
    """Make a new request using the recorded dirty input and a local Read tool."""
    messages = [{'role': 'user', 'content': record['request']['task']}]
    tools = [
        {'name': 'Read', 'description': 'Read lines of the one CSV input file.',
         'input_schema': {'type': 'object', 'properties': {
             'file_path': {'type': 'string'}, 'offset': {'type': 'integer', 'minimum': 1},
             'limit': {'type': 'integer', 'minimum': 1}}, 'required': ['file_path']}},
        {'name': 'StructuredOutput', 'description': 'Return the final cell edits.',
         'input_schema': record['request']['output_schema']},
    ]
    captured = {'batch': record['batch'], 'input': record['input'],
                'request': dict(record['request'], model=model, max_tokens=max_tokens,
                                runtime='standalone Anthropic Messages adapter'),
                'turns': [], 'reads': []}
    for _ in range(200):
        body = {'model': model, 'system': system_prompt, 'messages': messages,
                'tools': tools, 'max_tokens': max_tokens,
                'thinking': {'type': 'adaptive'},
                'output_config': {'effort': record['request']['effort']}}
        req = urllib.request.Request(base_url.rstrip('/') + '/v1/messages',
                                     data=json.dumps(body).encode('utf-8'), method='POST',
                                     headers={'x-api-key': api_key, 'anthropic-version': '2023-06-01',
                                              'content-type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError('Anthropic API returned HTTP ' + str(error.code)) from error
        captured['turns'].append({'model': result.get('model'), 'usage': result.get('usage', {})})
        content = result.get('content', [])
        messages.append({'role': 'assistant', 'content': content})
        replies = []
        for part in content:
            if part.get('type') != 'tool_use':
                continue
            if part['name'] == 'StructuredOutput':
                edits_of(part['input'])
                captured['response'] = part['input']
                return captured
            try:
                if part['name'] != 'Read':
                    raise ValueError('only Read and StructuredOutput are permitted')
                text = read_tool(record['input']['csv'], part['input'], record['input']['filename'])
                reply = {'type': 'tool_result', 'tool_use_id': part['id'], 'content': text}
            except (ValueError, TypeError) as error:
                reply = {'type': 'tool_result', 'tool_use_id': part['id'],
                         'content': str(error), 'is_error': True}
            replies.append(reply)
            captured['reads'].append({'request': part.get('input'), 'result': reply['content'],
                                      'is_error': reply.get('is_error', False)})
        messages.append({'role': 'user', 'content': replies or
                         'Return your final edits using StructuredOutput. Read only the specified input file.'})
    raise RuntimeError('live request exceeded the tool-turn limit')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dataset', required=True, help='dataset name, or all for an offline check')
    ap.add_argument('--check', action='store_true', help='replay in memory and compare with the shipped CSV; write nothing')
    ap.add_argument('--out', type=Path, help='CSV output path; default results/runs/llm/<dataset>/cleaned.csv')
    ap.add_argument('--live', action='store_true', help='explicitly make NEW paid API calls instead of replaying')
    ap.add_argument('--model', default=None, help='live mode only; defaults to the recorded model')
    ap.add_argument('--base-url', default=os.environ.get('ANTHROPIC_BASE_URL', 'https://api.anthropic.com'),
                    help='Anthropic Messages API endpoint, used only with --live')
    ap.add_argument('--max-tokens', type=int, default=16384, help='maximum tokens per NEW live API turn')
    args = ap.parse_args(argv)
    if args.check and (args.out or args.live):
        ap.error('--check cannot be combined with --out or --live')
    if args.dataset == 'all' and not args.check:
        ap.error('--dataset all requires --check')
    if args.model and not args.live:
        ap.error('--model is only for new live calls; replay uses the recorded model')
    if args.max_tokens <= 0:
        ap.error('--max-tokens must be positive')
    names = sorted(json.loads((BUNDLE / 'manifest.json').read_text())['datasets']) if args.dataset == 'all' else [args.dataset]
    for dataset in names:
        entry, records, header, rows = load_dataset(dataset)
        # Validate the partition before any optional network request.
        training = validate_inputs(header, rows, records, entry['label_column'])
        ids = [row[header.index('index')] for row in rows if row[header.index('index')] in training]
        if len(training) != entry['training_rows'] or sha256(('\n'.join(ids) + '\n').encode()) != entry['training_index_sha256']:
            raise ValueError('training index set differs from the recorded split')
        output = args.out or ROOT / 'results/runs/llm' / dataset / 'cleaned.csv'
        if not args.check and output.resolve() in {(ROOT / entry['result']).resolve(), (ROOT / entry['data']).resolve()}:
            ap.error('choose an output path that does not overwrite a shipped input or result')
        if args.live:
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                ap.error('--live requires ANTHROPIC_API_KEY')
            system_prompt = (BUNDLE / 'prompt_template.txt').read_text()
            output.parent.mkdir(parents=True, exist_ok=True)
            log = output.with_suffix('.requests.jsonl')
            with log.open('w') as log_file:
                for record in records:
                    captured = live_batch(record, system_prompt, api_key, args.base_url,
                                          args.model or record['request']['model'], args.max_tokens)
                    log_file.write(json.dumps(captured, ensure_ascii=False) + '\n')
                    log_file.flush()
                    record['response'] = captured['response']
        replayed, _ = replay(header, rows, records, entry['label_column'])
        if not args.live:
            if sha256(replayed) != entry['result_csv_sha256']:
                raise ValueError('replayed CSV differs from the recorded result')
            shipped = gzip.decompress((ROOT / entry['result']).read_bytes())
            if replayed != shipped:
                raise ValueError('shipped CSV differs from the replayed result')
        if args.check:
            print(dataset + ': exact CSV match, ' + str(len(records)) + ' input files, ' + str(len(training)) + ' training rows')
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(replayed)
            print(dataset + ': wrote ' + str(output))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print('error: ' + str(exc), file=sys.stderr)
        sys.exit(1)
