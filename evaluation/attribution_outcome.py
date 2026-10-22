#!/usr/bin/env python
"""Shapley attribution of a DemandClean run (Table tab:shapley), computed from the run's output.

    python evaluation/attribution_outcome.py results/main_table/beers/demandclean

The attribution explains the downstream gain of the run over NoFix on the model, scorer and clean
test split of the main table (evaluation/paper_score.py).

Cells of the training split are classified by comparing the cleaned output of the run with the
dirty and the clean training rows:
  gt   kept cell whose value changed to its clean value (T in the paper: written with its true
       value, by GT-repair, by a calibration seed cell or by VE-replace)
  ve   kept cell whose value changed to a value other than the clean one (E: written with an
       estimate)
  del  training row absent from the output (D: deleted)
  noop every other cell (left as it is)
A coalition S of {gt, ve, del} starts from the dirty training split, writes the output value of
every cell whose class is in S, and removes the deleted rows when del is in S. v(S) is the
main-table score of that training split. The value of the full coalition is the score of the run
and the value of the empty coalition is NoFix, so the three Shapley values sum to the gain of
the run over NoFix, and no-op is a null player (phi_noop = 0). The Shapley values are exact over
the 3! orders, and the table normalizes them by sum |phi|.
"""
import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_score as P  # noqa: E402

PLAYERS = ('gt', 've', 'del')
TOL = 1e-6


def _close(a, b):
    both_nan = np.isnan(a) & np.isnan(b)
    with np.errstate(invalid='ignore'):
        near = np.abs(a - b) <= TOL * np.maximum(1.0, np.abs(b))
    return both_nan | near


def _same(a, b):
    """Bitwise equality with NaN == NaN. A changed cell is any cell the output differs in, so
    the full coalition rebuilds the output exactly (some downstream fits react to 1e-7 changes)."""
    return (np.isnan(a) & np.isnan(b)) | (a == b)


def _encoded_rows(enc, df):
    """Scaled matrix with missing cells kept as NaN (the in-run representation)."""
    X, y = enc._raw(P._clean_cols(df))
    return enc.scaler.transform(X), y


def shapley(values):
    phi = {p: 0.0 for p in PLAYERS}
    perms = list(itertools.permutations(PLAYERS))
    for perm in perms:
        cur = set()
        for p in perm:
            before = values[frozenset(cur)]
            cur.add(p)
            phi[p] += values[frozenset(cur)] - before
    return {p: v / len(perms) for p, v in phi.items()}


def attribute_npz(run_dir, root, ds, rep):
    task = P.DATASETS[ds]['task']
    model, metric = P.PAPER_MODEL[ds]
    ctx = P.dataset_context(root, ds)
    dirty, clean, tr, enc = ctx['dirty'], ctx['clean'], ctx['tr'], ctx['enc']
    z = np.load(os.path.join(run_dir, 'cleaned_encoded.npz'), allow_pickle=True)
    Xr, yr = np.asarray(z['X_result'], dtype=float), np.asarray(z['y_result'], dtype=float)
    out_idx = pd.read_csv(os.path.join(run_dir, 'cleaned.csv'))['index'].values
    assert len(out_idx) == len(Xr), 'cleaned.csv rows differ from the npz rows'
    tr_index = dirty['index'].values[tr]
    pos = {v: i for i, v in enumerate(tr_index)}
    kept = np.array([pos[v] for v in out_idx])
    assert (np.diff(kept) > 0).all(), 'output rows are not in training-split order'
    Xd, yd = _encoded_rows(enc, dirty.iloc[tr].reset_index(drop=True))
    Xc, yc = _encoded_rows(enc, clean.iloc[tr].reset_index(drop=True))
    # cell classes on the kept rows (feature cells and the label cell)
    Fd, Fc = np.column_stack([Xd, yd]), np.column_stack([Xc, yc])
    Fr = np.column_stack([Xr, yr])
    changed = ~_same(Fr, Fd[kept])
    restored = changed & _close(Fr, Fc[kept])
    estimated = changed & ~restored
    keep_mask = np.zeros(len(tr), dtype=bool)
    keep_mask[kept] = True
    evals = {'test': ctx['test'], 'val': ctx['val']}

    def value(S):
        F = Fd.copy()
        for cls, mask in (('gt', restored), ('ve', estimated)):
            if cls in S:
                rr, cc = np.nonzero(mask)
                F[kept[rr], cc] = Fr[rr, cc]
        rows = keep_mask if 'del' in S else np.ones(len(tr), dtype=bool)
        X = F[rows, :-1]
        # a column that is missing in every row (for example unparseable unit strings) is the
        # scaled column mean 0 until an operation writes it
        X[:, np.isnan(X).all(axis=0)] = 0.0
        res, _, _ = P.score_encoded(task, X, F[rows, -1], evals, [model], metric)
        return res[model]['test']

    values = {frozenset(S): value(S) for r in range(4) for S in itertools.combinations(PLAYERS, r)}
    cells = {'gt': int(restored.sum()), 've': int(estimated.sum()), 'del_rows': int((~keep_mask).sum())}
    return values, cells


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('run_dir')
    ap.add_argument('--data-root', default=None)
    a = ap.parse_args()
    run_dir = os.path.abspath(a.run_dir.rstrip('/'))
    summary = P.read_json(os.path.join(run_dir, 'summary.json'))
    ds = summary['dataset']
    root = P.data_root(a.data_root)
    values, cells = attribute_npz(run_dir, root, ds, summary)
    phi = shapley(values)
    total = sum(abs(v) for v in phi.values())
    phi_hat = {p: (round(v / total, 4) if total else 0.0) for p, v in phi.items()}
    full = values[frozenset(PLAYERS)]
    score = P.score_run(run_dir, root, references=False)['test']
    ops = summary.get('operations') or {}
    print(json.dumps({
        'dataset': ds, 'run': os.path.relpath(run_dir, P._ROOT), 'model': P.PAPER_MODEL[ds][0],
        'v_empty': values[frozenset()], 'v_full': full, 'run_score': score,
        'full_equals_run': abs(full - score) < 1e-9,
        'phi': phi, 'phi_hat': {'noop': 0.0, **phi_hat},
        'coalitions': {'+'.join(sorted(k)) or 'none': v for k, v in values.items()},
        'outcome_cells': cells,
        'operation_counts': {'noop': ops.get('no_op'), 'gt': ops.get('gt_repair'),
                             'del': ops.get('delete'), 've': ops.get('ve_replace')},
    }))


if __name__ == '__main__':
    main()
