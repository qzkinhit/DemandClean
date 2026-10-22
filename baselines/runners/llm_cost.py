"""Recompute the LLM list-price cost from the captured per-turn usage.

    python baselines/runners/llm_cost.py --check
    python baselines/runners/llm_cost.py --out /tmp/llm_cost.json

This reads the released request records. It makes no API calls and writes nothing
unless --out is supplied. Each recorded input file and assistant turn is counted
once, using the list prices in results/llm/manifest.json.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / 'results/llm'


def compute_costs(bundle_dir=BUNDLE):
    bundle_dir = Path(bundle_dir)
    manifest = json.loads((bundle_dir / 'manifest.json').read_text())
    prices = manifest['prices_usd_per_million_tokens']
    costs = {}
    for dataset, entry in sorted(manifest['datasets'].items()):
        raw = gzip.decompress((bundle_dir / 'records' / (dataset + '.jsonl.gz')).read_bytes())
        if hashlib.sha256(raw).hexdigest() != entry['records_sha256']:
            raise ValueError('request-record hash mismatch: ' + dataset)
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if [r['batch'] for r in records] != entry['batches']:
            raise ValueError('request membership differs from manifest: ' + dataset)
        totals = dict(input=0, cache_write_5m=0, cache_write_1h=0, cache_read=0, output=0)
        turns = 0
        models = set()
        for record in records:
            for turn in record['turns']:
                usage = turn['usage']
                write_1h = (usage.get('cache_creation') or {}).get('ephemeral_1h_input_tokens') or 0
                totals['input'] += usage.get('input_tokens') or 0
                totals['cache_write_1h'] += write_1h
                totals['cache_write_5m'] += (usage.get('cache_creation_input_tokens') or 0) - write_1h
                totals['cache_read'] += usage.get('cache_read_input_tokens') or 0
                totals['output'] += usage.get('output_tokens') or 0
                turns += 1
                if turn.get('model'):
                    models.add(turn['model'])
        write = totals['cache_write_5m'] + totals['cache_write_1h']
        costs[dataset] = {
            'files': len(records), 'turns': turns,
            'input_tokens': totals['input'] + write + totals['cache_read'],
            'cache_read_tokens': totals['cache_read'], 'cache_write_tokens': write,
            'output_tokens': totals['output'],
            'usd': round(sum(totals[key] * prices[key] for key in prices) / 1e6, 2),
            'models': sorted(models),
        }
    return costs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--check', action='store_true', help='compare with results/llm/llm_cost.json; write nothing')
    ap.add_argument('--out', type=Path, help='write the recomputed cost JSON to this path')
    args = ap.parse_args(argv)
    if args.check and args.out:
        ap.error('--check and --out are mutually exclusive')
    costs = compute_costs()
    if args.check:
        expected = json.loads((BUNDLE / 'llm_cost.json').read_text())
        if costs != expected:
            raise ValueError('computed usage or costs differ from llm_cost.json')
        print('LLM costs match: %d datasets, %d input files, $%.2f total' %
              (len(costs), sum(v['files'] for v in costs.values()), sum(v['usd'] for v in costs.values())))
    elif args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(costs, ensure_ascii=False, indent=2) + '\n')
    else:
        print(json.dumps(costs, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as exc:
        print('error: ' + str(exc), file=sys.stderr)
        sys.exit(1)
