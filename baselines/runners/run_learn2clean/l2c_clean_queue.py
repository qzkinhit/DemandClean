#!/usr/bin/env python3
"""Run restricted Learn2Clean on all datasets in parallel: plans, then every candidate, then finalize.

    LEARN2CLEAN_PYTHON=<python of the Learn2Clean environment> python baselines/runners/run_learn2clean/l2c_clean_queue.py
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
PY = os.environ.get('LEARN2CLEAN_PYTHON', sys.executable)
AD = str(Path(__file__).resolve().parent / 'l2c_clean.py')
RES = ROOT / 'results/runs/learn2clean_clean'
LOG = ROOT / 'results/runs/logs/l2c_clean'
LOG.mkdir(parents=True, exist_ok=True)
DATASETS = ['adult', 'beers', 'bike', 'breast_cancer', 'har', 'mercedes', 'nasa', 'smartfactory',
            'soilmoisture', 'flights', 'soccer', 'hospitals']
CAP = {'har': 4}                 # agglomerative validation on ~40k rows needs memory
WORKERS = 48
CAND_TIMEOUT = 12 * 3600
ENV = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
           NUMEXPR_NUM_THREADS='1', PYTHONHASHSEED='42')
lock = threading.Lock()


def emit(**row):
    row['time'] = time.time()
    with lock:
        with open(LOG / 'events.jsonl', 'a') as f:
            f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def call(args, log, timeout):
    with open(log, 'a') as f:
        p = subprocess.Popen([PY, '-u', AD] + args, stdout=f, stderr=subprocess.STDOUT, env=ENV)
        try:
            return p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
            return 124


def main():
    with ThreadPoolExecutor(max_workers=len(DATASETS)) as pool:
        rcs = dict(zip(DATASETS, pool.map(lambda d: call(['plan', '--dataset', d], LOG / f'{d}_plan.log', 3600), DATASETS)))
    jobs = []
    for d in DATASETS:
        if rcs[d] != 0 or not (RES / d / 'plan.json').exists():
            emit(dataset=d, status='plan_failed', rc=rcs[d])
            continue
        pl = json.loads((RES / d / 'plan.json').read_text())
        emit(dataset=d, status='planned', candidates=len(pl['candidates']))
        jobs += [(d, c['candidate']) for c in pl['candidates']]
    left = {}
    for d, _ in jobs:
        left[d] = left.get(d, 0) + 1
    sems = {d: threading.Semaphore(CAP.get(d, WORKERS)) for d in left}

    def run_candidate(job):
        d, cid = job
        with sems[d]:
            rc = call(['candidate', '--dataset', d, '--candidate', str(cid)], LOG / f'{d}_cand_{cid:02d}.log', CAND_TIMEOUT)
        res = RES / d / 'candidates' / f'cand_{cid:02d}' / 'result.json'
        if rc == 124:
            res.parent.mkdir(parents=True, exist_ok=True)
            res.write_text(json.dumps(dict(dataset=d, candidate=cid, status='timeout', test_accessed=False)))
        emit(dataset=d, candidate=cid, rc=rc)
        with lock:
            left[d] -= 1
            last = left[d] == 0
        if last:
            frc = call(['finalize', '--dataset', d], LOG / f'{d}_finalize.log', 3600)
            score = None
            sp = RES / d / 'scores.json'
            if sp.exists():
                score = json.loads(sp.read_text()).get('test')
            emit(dataset=d, status='finalized', rc=frc, test=score)

    # Heavy datasets first so that their long candidates start early.
    order = sorted(jobs, key=lambda j: (j[0] not in ('mercedes', 'har', 'adult'), j))
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(run_candidate, order))
    emit(status='all_done')


if __name__ == '__main__':
    main()
