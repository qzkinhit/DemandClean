#!/usr/bin/env python
"""Two-player Shapley attribution of a DemandClean run that reads no clean training value.

Players are defined from the run's output and the dirty training split alone:
  write  kept training cell whose output value differs from its dirty value (GT-repair and
         VE-replace together, since telling them apart needs the clean value or a per-cell
         operation log that the runs do not store)
  del    training row absent from the output (Delete)
A coalition S starts from the dirty training split, writes the output value of every write cell
when write is in S, and removes the deleted rows when del is in S. v(S) is the main-table score of
that training split on the clean test split, the evaluation every main-table number uses. v of the
empty coalition is NoFix and v of the full coalition is the run's own score, so the two Shapley
values sum to the run's gain over NoFix. The script never reads the clean training rows.
The organically dirty datasets are scored with the nearest-category encoding, like their runs.

usage: attribution_operation.py <run_dir> [--data-root DIR]
  the run directory is a shipped DemandClean run (results/.../demandclean) with cleaned.csv,
  cleaned_encoded.npz and summary.json
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def load_scorer(ds):
    spec = importlib.util.spec_from_file_location('paper_score', os.path.join(HERE, 'paper_score.py'))
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    # the runs on the organically dirty datasets encode unseen categories by edit distance
    P.USE_NEAREST = ds in ('flights', 'soccer', 'hospitals')
    return P


def _same(a, b):
    """Bitwise equality with NaN == NaN, as in attribution_outcome.py."""
    return (np.isnan(a) & np.isnan(b)) | (a == b)


def attribute(P, run_dir, root, ds):
    task = P.DATASETS[ds]['task']
    model, metric = P.PAPER_MODEL[ds]
    ctx = P.dataset_context(root, ds)
    dirty, tr, enc = ctx['dirty'], ctx['tr'], ctx['enc']
    z = np.load(os.path.join(run_dir, 'cleaned_encoded.npz'), allow_pickle=True)
    out_idx = pd.read_csv(os.path.join(run_dir, 'cleaned.csv'))['index'].values
    Xr, yr = np.asarray(z['X_result'], dtype=float), np.asarray(z['y_result'], dtype=float)
    assert len(out_idx) == len(Xr), 'output index rows differ from the npz rows'
    pos = {v: i for i, v in enumerate(dirty['index'].values[tr])}
    kept = np.array([pos[v] for v in out_idx])
    assert (np.diff(kept) > 0).all(), 'output rows are not in training-split order'
    Xd, yd = enc._raw(P._clean_cols(dirty.iloc[tr].reset_index(drop=True)))
    Fd = np.column_stack([enc.scaler.transform(Xd), yd])
    Fr = np.column_stack([Xr, yr])
    written = ~_same(Fr, Fd[kept])
    keep_mask = np.zeros(len(tr), dtype=bool)
    keep_mask[kept] = True
    evals = {'test': ctx['test'], 'val': ctx['val']}

    def value(S):
        F = Fd.copy()
        if 'write' in S:
            rr, cc = np.nonzero(written)
            F[kept[rr], cc] = Fr[rr, cc]
        rows = keep_mask if 'del' in S else np.ones(len(tr), dtype=bool)
        X = F[rows, :-1]
        X[:, np.isnan(X).all(axis=0)] = 0.0
        res, _, _ = P.score_encoded(task, X, F[rows, -1], evals, [model], metric)
        return res[model]['test']

    values = {k: value(S) for k, S in (('none', ()), ('write', ('write',)), ('del', ('del',)), ('write+del', ('write', 'del')))}
    cells = {'write': int(written.sum()), 'del_rows': int((~keep_mask).sum())}
    return model, values, cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--data-root')
    a = ap.parse_args()
    run_dir = os.path.abspath(a.run_dir.rstrip('/'))
    with open(os.path.join(run_dir, 'summary.json')) as f:
        ds = json.load(f)['dataset']
    P = load_scorer(ds)
    root = P.data_root(a.data_root)
    score = P.score_run(run_dir, root, references=False)['test']
    model, v, cells = attribute(P, run_dir, root, ds)
    phi = {'write': 0.5 * (v['write'] - v['none']) + 0.5 * (v['write+del'] - v['del']),
           'del': 0.5 * (v['del'] - v['none']) + 0.5 * (v['write+del'] - v['write'])}
    total = sum(abs(x) for x in phi.values())
    print(json.dumps({
        'dataset': ds, 'run': os.path.relpath(run_dir, os.path.dirname(HERE)), 'model': model,
        'reads_clean_training_rows': False,
        'v_empty': v['none'], 'v_full': v['write+del'], 'run_score': score,
        'full_equals_run': abs(v['write+del'] - score) < 1e-9,
        'phi': phi, 'phi_hat': {p: (round(x / total, 4) if total else None) for p, x in phi.items()},
        'coalitions': v, 'cells': cells,
    }))


if __name__ == '__main__':
    main()
