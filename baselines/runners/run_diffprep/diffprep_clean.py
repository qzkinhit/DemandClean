#!/usr/bin/env python
"""DiffPrep restricted to its cleaning steps, under the untouched-test rule.

DiffPrep-Fix (baselines/Methods/DiffPrep/diffprep_api.py, upstream commit 6499edf) searches the
default space without the normalization and discretization steps, so the pipeline holds the
missing-value imputation step and the outlier-cleaning step only. Its input is the dirty
training matrix in the canonical encoding with missing cells kept, the same input that the
Learn2Clean and DIANA adapters receive, and the clean validation split drives early stopping and
the learning-rate choice (smallest validation loss, the upstream grid rule). After the search the
soft pipeline output of the training rows (the probability-weighted transformations on which
DiffPrep's own end model is trained, as in upstream and in the release scoring) is written in the
canonical encoding. The test split never enters the method and is scored
untouched.

    SCORER_PYTHON=<python with scikit-learn 1.3.2> python baselines/runners/run_diffprep/diffprep_clean.py run --dataset beers --lr 0.01
    python baselines/runners/run_diffprep/diffprep_clean.py finalize --dataset beers

diffprep_clean_queue.py runs the learning-rate grid of every dataset and finalizes it.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]  # repository root
sys.path.insert(0, str(ROOT / 'baselines' / 'runners'))
import common  # noqa: E402

CANONICAL = os.environ.get('SCORER_PYTHON', sys.executable)
RESULTS = ROOT / 'results/runs/diffprep_clean'
METHOD = 'DiffPrep_Fix_cleaning_steps'
STAB = {'mercedes', 'nasa', 'soilmoisture'}  # stabilization of the release grid (grad clip 1, sanitize alpha)
KEEP_STEPS = ('missing_value_imputation', 'cleaning_outliers')


def run(ds, lr):
    import torch
    sys.path.insert(0, str(ROOT / 'baselines'))
    from Methods.DiffPrep import diffprep_api as A
    torch.set_num_threads(1)
    A.DEFAULT_SPACE = [s for s in A.DEFAULT_SPACE if s['name'] in KEEP_STEPS]
    out = RESULTS / ds / ('lr_%g' % lr)
    out.mkdir(parents=True, exist_ok=True)
    b = common.load_bundle(ds)
    cols = ['f%d' % i for i in range(len(b['features']))]
    ytr = np.asarray(b['y_train'], float)
    ok = np.isfinite(ytr)
    Xtr = pd.DataFrame(np.asarray(b['X_train_missing'], float)[ok], columns=cols)
    idx = np.asarray(b['idx_train'])[ok]
    yva = np.asarray(b['y_val'], float)
    okv = np.isfinite(yva)
    Xva = pd.DataFrame(np.asarray(b['X_val'], float)[okv], columns=cols)
    task = 'regression' if b['task'] == 'regression' else 'classification'
    if task == 'classification':  # har: activity label as the supervised proxy, as in the release runner
        ytr_l, yva_l = ytr[ok].astype(np.int64).astype(str), yva[okv].astype(np.int64).astype(str)
    else:
        ytr_l, yva_l = ytr[ok], yva[okv]
    params = dict(model_lr=[lr])
    if ds in STAB:
        params.update(grad_clip=1.0, sanitize_alpha=True)
    t0 = time.time()
    res = A.run_diffprep(Xtr, ytr_l, Xva, yva_l, task=task, params=params, log=lambda m: print(m, flush=True))
    res.params['input_clip'] = None  # export the cleaned rows without any clamp
    soft, hard = res.features(Xtr, 'train')
    pos = [cols.index(c) for c in res.columns]
    if soft.shape[1] != len(pos):
        raise ValueError('restricted pipeline changed the feature count')
    for name, arr in (('cleaned.npz', soft), ('cleaned_hard.npz', hard)):
        full = np.zeros((len(Xtr), len(cols)), dtype=float)  # all-missing training columns stay at the encoder fill 0
        full[:, pos] = arr
        np.savez_compressed(out / name, X=full, y=ytr[ok], idx=idx)
    rec = dict(dataset=ds, lr=lr, status='completed', best=res.best, runs=res.runs, columns=res.columns,
               dropped=res.dropped_columns, pipeline=res.describe_pipeline(), deviations=res.deviations,
               stabilization=ds in STAB, seconds=time.time() - t0, test_accessed=False, gt_cells=0)
    (out / 'result.json').write_text(json.dumps(rec, indent=2, default=str))
    print('DIFFPREP_DONE', ds, lr, res.best.get('best_val_loss'), flush=True)


def finalize(ds):
    rows = []
    for d in sorted((RESULTS / ds).glob('lr_*')):
        f = d / 'result.json'
        if f.exists():
            r = json.loads(f.read_text())
            if r.get('status') == 'completed' and np.isfinite(float(r['best'].get('best_val_loss', np.nan))):
                rows.append((float(r['best']['best_val_loss']), d, r))
    if not rows:
        raise RuntimeError('no completed grid point')
    loss, d, r = min(rows, key=lambda t: t[0])
    z = np.load(d / 'cleaned.npz')
    b = common.load_bundle(ds)
    meta = dict(selected_lr=r['lr'], best_val_loss=loss, grid=[dict(lr=x[2]['lr'], val_loss=x[0]) for x in rows],
                pipeline=r['pipeline'], deviations=r['deviations'] + [
                    'search space restricted to missing-value imputation and outlier cleaning',
                    'input: dirty training matrix in the canonical encoding with missing cells kept',
                    'soft pipeline output of the training rows exported without input clamp; test split untouched'],
                stabilization=r['stabilization'], gt_cells=0, gt_cap_cells=b['gt_cap_cells'])
    out = RESULTS / ds
    common.write_cleaned_encoded(ds, out, z['X'], z['y'], z['idx'], METHOD, gt_cells=0, metadata=meta)
    p = subprocess.run([CANONICAL, str(ROOT / 'baselines/runners/common.py'), '--dataset', ds, '--out', str(out)],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    print(p.stdout, flush=True)
    if p.returncode:
        raise RuntimeError('canonical final scoring failed')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['run', 'finalize'])
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--lr', type=float)
    a = ap.parse_args()
    try:
        run(a.dataset, a.lr) if a.mode == 'run' else finalize(a.dataset)
    except Exception as e:
        traceback.print_exc()
        if a.mode == 'run':
            out = RESULTS / a.dataset / ('lr_%g' % a.lr)
            out.mkdir(parents=True, exist_ok=True)
            (out / 'result.json').write_text(json.dumps(dict(dataset=a.dataset, lr=a.lr, status='failed',
                                                         error=repr(e)[:2000], test_accessed=False)))
        sys.exit(1)
