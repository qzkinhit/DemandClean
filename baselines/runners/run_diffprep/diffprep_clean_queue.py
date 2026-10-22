#!/usr/bin/env python3
"""Run restricted DiffPrep on every dataset and grid point in parallel, then finalize each dataset.

    DIFFPREP_PYTHON=<python of the DiffPrep environment> python baselines/runners/run_diffprep/diffprep_clean_queue.py
"""
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # repository root
PY = os.environ.get('DIFFPREP_PYTHON', sys.executable)
AD = str(Path(__file__).resolve().parent / 'diffprep_clean.py')
RES = ROOT / 'results/runs/diffprep_clean'
LOG = ROOT / 'results/runs/logs/diffprep_clean'
LOG.mkdir(parents=True, exist_ok=True)
DATASETS = ['adult', 'beers', 'bike', 'breast_cancer', 'har', 'mercedes', 'nasa', 'smartfactory',
            'soilmoisture', 'flights', 'soccer', 'hospitals']
STAB = {'mercedes', 'nasa', 'soilmoisture'}
WORKERS = int(os.environ.get('WORKERS', '24'))
ENV = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
           NUMEXPR_NUM_THREADS='1', PYTHONHASHSEED='42')
lock = threading.Lock()


def emit(**row):
    row['time'] = time.time()
    with lock:
        with open(LOG / 'events.jsonl', 'a') as f:
            f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def call(args, log):
    with open(log, 'a') as f:
        return subprocess.call([PY, '-u', AD] + args, stdout=f, stderr=subprocess.STDOUT, env=ENV)


def main():
    jobs = []
    for d in DATASETS:
        lrs = [0.1, 0.01, 0.001] + ([1e-4, 1e-5] if d in STAB else [])
        jobs += [(d, lr) for lr in lrs]
    left = {}
    for d, _ in jobs:
        left[d] = left.get(d, 0) + 1

    def one(job):
        d, lr = job
        rc = call(['run', '--dataset', d, '--lr', repr(lr)], LOG / f'{d}_lr{lr:g}.log')
        emit(dataset=d, lr=lr, rc=rc)
        with lock:
            left[d] -= 1
            last = left[d] == 0
        if last:
            frc = call(['finalize', '--dataset', d], LOG / f'{d}_finalize.log')
            sp = RES / d / 'scores.json'
            emit(dataset=d, status='finalized', rc=frc,
                 test=json.loads(sp.read_text()).get('test') if sp.exists() else None)

    order = sorted(jobs, key=lambda j: (j[0] not in ('mercedes', 'har', 'adult', 'smartfactory', 'bike'), j))
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(one, order))
    emit(status='all_done')


if __name__ == '__main__':
    main()
