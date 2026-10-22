"""Headless adapter of qcri/RetClean (data lake of 1.65% of the table cells)'s semantic-retrieval + Llama 3.1 path.

Uses published dirty-only domain/FD checks and missing markers to select target cells.
The reference lake contains a fixed random half of training rows, fully GT-accounted.
Source search, prompt, model wrapper and output parser are called from upstream.

    RETCLEAN_SOURCE=<clone of qcri/RetClean> OLLAMA_URL=<Ollama server with llama3.1:8b> \
        SCORER_PYTHON=<python with scikit-learn 1.3.2> python baselines/runners/run_retclean/run_retclean_gt165.py --dataset beers
"""
import argparse
import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import types
import signal
import threading
from retclean_resources import EndpointPool, CONTEXT, VERSION, acquire_dataset_writer, load_fixed_reference

STOP = threading.Event()

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[3]  # repository root
os.environ.setdefault('OLLAMA_URL', 'http://127.0.0.1:11434')
sys.path.insert(0, str(ROOT / 'baselines' / 'runners'))
import common
import numpy as np
import pandas as pd

SOURCE = Path(os.environ.get('RETCLEAN_SOURCE', str(ROOT / 'baselines/Methods/RetClean/RetClean'))) / 'backend'
SCORER = os.environ.get('SCORER_PYTHON', sys.executable)


def atom(value):
    if pd.isna(value):
        return None
    return str(value)


def targets(bundle):
    """No clean references are consulted by target selection."""
    df = bundle['dirty_train']
    cols = bundle['features']
    t = pd.DataFrame(False, index=df.index, columns=cols)
    for c in cols:
        t[c] = df[c].isna() | df[c].astype(str).str.strip().str.lower().isin(
            ['empty', 'nan', 'null', 'none', '?', ''])
        if c not in bundle['categorical_cols']:
            numeric = pd.to_numeric(df[c], errors='coerce')
            t[c] |= numeric.isna()
    p = common.DATA_ROOT / bundle['dataset'] / 'rules.txt'
    section = ''
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line.startswith('['):
                section = line.upper()
                continue
            if section == '[DOMAIN]' and ':' in line:
                c, rule = line.split(':', 1)
                c = c.strip()
                if c not in t:
                    continue
                match = re.search(r'(?:FLOAT|INT|INTEGER)\s*\[\s*([^,]+),\s*([^\]]+)\]', rule)
                if match:
                    try:
                        lo, hi = map(float, match.groups())
                        values = pd.to_numeric(df[c], errors='coerce')
                        t[c] |= values.isna() | (values < lo) | (values > hi)
                    except ValueError:
                        pass
                elif 'ENUM' in rule and '{' in rule:
                    allowed = {x.strip().strip('"\'') for x in rule.split('{', 1)[1].split('}', 1)[0].split(',')}
                    t[c] |= ~df[c].astype(str).str.strip().isin(allowed)
            if section == '[FD]' and '=>' in line:
                left, right = line.split('=>', 1)
                determinants = [x.strip() for x in left.split(',')]
                c = right.strip()
                if c not in t or not all(x in df for x in determinants):
                    continue
                for _, group in df.groupby(determinants, dropna=False, sort=False):
                    if len(group) < 2:
                        continue
                    s = group[c].dropna().astype(str)
                    if len(s) and s.nunique() > 1:
                        counts = s.value_counts()
                        if counts.iloc[0] > 1:
                            t.loc[group.index, c] |= group[c].astype(str) != counts.index[0]
    return t


def setup_upstream(out):
    from urllib.request import urlopen
    while True:
        try:
            with urlopen(os.environ['OLLAMA_URL'].rstrip('/') + '/api/tags', timeout=10) as response:
                names = [m['name'] for m in json.load(response).get('models', [])]
            if 'llama3.1:8b' in names:
                break
        except Exception:
            pass
        print(json.dumps(dict(event='waiting_model', model='llama3.1:8b')), flush=True)
        time.sleep(20)
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
    import torch
    torch.set_num_threads(2)
    # Initialize only the chosen local model; upstream package initialization
    # otherwise eagerly initializes cloud providers that this run does not use.
    models_pkg = types.ModuleType('language_models')
    models_pkg.__path__ = [str(SOURCE / 'language_models')]
    sys.modules['language_models'] = models_pkg
    from language_models.language_model_llama3_1 import Llama3_1
    import language_models.base as model_base
    model_base.eval = ast.literal_eval
    model = Llama3_1()
    model.model = 'llama3.1:8b'
    endpoint_pool = EndpointPool()
    from threading import Lock
    log_lock = Lock()

    class RecordedClient:
        def chat(self, **kwargs):
            kwargs['options'] = dict(seed=42, temperature=0, num_predict=160, num_ctx=8192)
            with endpoint_pool.claim() as (endpoint, client, slot):
                response = client.chat(**kwargs)
                raw = response.model_dump() if hasattr(response, 'model_dump') else dict(response)
                with log_lock, (out / 'llm_requests.jsonl').open('a') as f:
                    f.write(json.dumps(dict(request=kwargs, response=raw, resource=dict(
                        version=VERSION, endpoint=endpoint, slot=slot,
                        cell=getattr(CONTEXT, 'cell', None))), default=str) + '\n')
                return response

    model.client = RecordedClient()
    vector = QdrantClient(':memory:')
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda:0')
    core = types.ModuleType('core')
    core.__path__ = [str(SOURCE / 'core')]
    core.qdrant_client = vector
    core.sentence_model = sentence_model
    # Semantic-only search never invokes ES. Its upstream existence guard is
    # tied to the semantic collection instead of requiring an unused ES server.
    class IndexPresence:
        def exists(self, index):
            return vector.collection_exists(index)
    core.es_client = types.SimpleNamespace(indices=IndexPresence())
    core.initialized_models = {'Llama 3.1': model}
    sys.modules['core'] = core
    reranking = types.ModuleType('reranking')
    reranking.RERANKER_MAP = {}
    sys.modules['reranking'] = reranking
    from core.repair import repair_data
    return vector, sentence_model, repair_data


def run(dataset):
    b = common.load_bundle(dataset)
    out = ROOT / 'results/runs/retclean_gt165' / dataset / 'seed42'
    out.mkdir(parents=True, exist_ok=True)
    writer_lock = acquire_dataset_writer(out)
    if (out / 'scores.json').exists():
        print(json.dumps(dict(event='already_completed', dataset=dataset)), flush=True)
        return
    started = time.time()
    dirty = b['dirty_train'].copy()
    mask = targets(b)
    counts = {c: int(mask[c].sum()) for c in mask if mask[c].any()}
    (out / 'targets.json').write_text(json.dumps(counts, indent=2))
    print(json.dumps(dict(event='targets', dataset=dataset, count=int(mask.values.sum()), columns=len(counts))), flush=True)
    meta = dict(source='https://github.com/qcri/RetClean',
                source_archive_sha256='b16c99599589d05b8d570706c66a725e8761ce06a97c1f3601f57078a6301102',
                model='llama3.1:8b (upstream Ollama local backend)', seed=42,
                retrieval='upstream semantic cosine top-3; Qdrant local storage',
                reranker=None, target_selection='missing/type/published domain/FD, dirty-only',
                reference_lake='fixed random training rows holding 1.65% of the table cells, 1.1 times the mean GT use of DemandClean; all non-index cells charged',
                pivot_selection='first eight other non-label input features, fixed column order',
                val_used_for_selection=False, api_cost_usd=0,
                source_parser_adaptation='ast.literal_eval instead of eval; same response format',
                inference_endpoint=os.environ['OLLAMA_URL'],
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                resource_version=VERSION,
                inference_endpoints=EndpointPool().endpoints,
                request_concurrency_per_endpoint=6,
                dataset_workers=int(os.environ.get('RETCLEAN_WORKERS', '12')))
    if not counts:
        common.write_cleaned(dataset, out, dirty, 'retclean', 0, meta)
    else:
        rng = np.random.RandomState(42)
        n_lake = min(len(dirty) // 2, max(1, int(0.0165 * b['gt_denominator_full_table']) // len(b['non_index_columns'])))
        positions = rng.choice(len(dirty), size=n_lake, replace=False)
        rows = dirty.iloc[positions]['index'].astype(int).tolist()
        frozen = load_fixed_reference(b, out, rows)
        if frozen is None:
            oracle = common.GTOracle(dataset, out)
            reference = oracle.query(rows, b['non_index_columns'])
            reference['index'] = rows
            reference.to_csv(out / 'reference_lake.csv', index=False)
            gt_cells = oracle.used
        else:
            reference, gt_cells, frozen_hashes = frozen
            meta.update(reference_reused_without_oracle=True, frozen_gt_hashes=frozen_hashes)
        print(json.dumps(dict(event='frozen_reference', dataset=dataset, gt_cells=gt_cells, reused=frozen is not None)), flush=True)
        vector, sentence, repair = setup_upstream(out)
        from qdrant_client import models
        edits_path = out / 'edits.jsonl'
        previous = {}
        if edits_path.exists():
            for line in edits_path.read_text().splitlines():
                try:
                    e = json.loads(line)
                    if e.get('status') == 'success' and e.get('result', {}).get('status') != 'fail':
                        key = (int(e['row_id']), e['column'])
                        if key in previous and previous[key] != e:
                            raise RuntimeError('Conflicting successful edits for one cell')
                        previous[key] = e
                except (ValueError, TypeError, KeyError):
                    continue
        id_to_position = {int(r): i for i, r in enumerate(dirty['index'])}
        completed, failed = 0, 0
        resumed_successes = 0
        for column in counts:
            pending = []
            for pos in np.where(mask[column])[0]:
                key = (int(dirty.iloc[pos]['index']), column)
                if key not in previous:
                    pending.append(int(pos))
                    continue
                e = previous[key]
                value = e.get('result', {}).get('value')
                if value is not None and str(value).strip().lower() not in ('none', 'nan', 'null', 'unknown', ''):
                    dirty.at[pos, column] = value
                completed += 1
                resumed_successes += 1
            if not pending:
                continue
            if STOP.is_set():
                break
            pivots = [c for c in b['features'] if c != column][:8]
            collection = 'ret_' + hashlib.sha1((dataset + ':' + column).encode()).hexdigest()[:20]
            records = []
            for rowid, row in reference.iterrows():
                values = {c: atom(row[c]) for c in pivots + [column]}
                records.append(json.dumps(values, ensure_ascii=False))
            embeddings = sentence.encode(records, batch_size=256, show_progress_bar=False)
            vector.recreate_collection(collection, vectors_config=models.VectorParams(size=embeddings.shape[1], distance=models.Distance.COSINE))
            vector.upsert(collection_name=collection, points=[models.PointStruct(
                id=i, vector=emb.tolist(), payload=dict(values=record,
                table_name=dataset + '_training_reference', row_number=int(rows[i])))
                for i, (emb, record) in enumerate(zip(embeddings, records))])

            def one(position):
                if STOP.is_set():
                    return None
                row = dirty.iloc[position]
                row_id = int(row['index'])
                key = (row_id, column)
                CONTEXT.cell = [row_id, column]
                target = [{'id': row_id, 'value': atom(row[column])}]
                pivot = [{'id': row_id, 'values': [atom(row[c]) or '' for c in pivots]}]
                result = asyncio.run(repair('A record from the ' + dataset + ' dataset.',
                    column, target, pivots, pivot, 'Llama 3.1', collection, 'semantic', None))
                response = (result.get('results') or [{}])[0]
                return dict(row_id=row_id, column=column, original=atom(row[column]),
                            result=response, status=result.get('status'))

            with ThreadPoolExecutor(max_workers=int(os.environ.get('RETCLEAN_WORKERS', '12'))) as pool:
                futures = {pool.submit(one, int(i)): int(i) for i in pending}
                for future in as_completed(futures):
                    pos = futures[future]
                    try:
                        e = future.result()
                        if e is None:
                            continue
                        value = e.get('result', {}).get('value')
                        if value is not None and str(value).strip().lower() not in ('none', 'nan', 'null', 'unknown', ''):
                            dirty.at[pos, column] = value
                        if e.get('status') != 'success' or e.get('result', {}).get('status') == 'fail':
                            failed += 1
                    except Exception as error:
                        failed += 1
                        e = dict(row_id=int(dirty.iloc[pos]['index']), column=column, error=repr(error))
                    with edits_path.open('a') as f:
                        f.write(json.dumps(e, default=str) + '\n')
                    completed += 1
                    if completed % 20 == 0:
                        dirty.to_csv(out / 'cleaned.partial.csv', index=False)
                        print(json.dumps(dict(event='progress', dataset=dataset, completed=completed,
                            total=int(mask.values.sum()), failed=failed, elapsed_s=time.time()-started)), flush=True)
            vector.delete_collection(collection)
            dirty.to_csv(out / 'cleaned.partial.csv', index=False)
        meta.update(queries=completed, failed_queries=failed, elapsed_s=time.time()-started,
                    resumed_successful_cells=resumed_successes, new_requests=completed-resumed_successes)
        if STOP.is_set():
            print(json.dumps(dict(event='interrupted_after_draining', dataset=dataset, completed=completed)), flush=True)
            return
        if completed and failed > completed * 0.2:
            (out / 'failure.json').write_text(json.dumps(meta, indent=2))
            raise RuntimeError('RetClean backend failed on more than 20% of requests; output retained as partial only')
        common.write_cleaned(dataset, out, dirty, 'retclean', gt_cells, meta)
    subprocess.run([SCORER, str(ROOT / 'baselines/runners/common.py'), '--dataset', dataset, '--out', str(out)], check=True)
    print(json.dumps(dict(event='complete', dataset=dataset, out=str(out), elapsed_s=time.time()-started)), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda signum, frame: STOP.set())
    signal.signal(signal.SIGINT, lambda signum, frame: STOP.set())
    run(common.canonical_name(a.dataset))
