#!/usr/bin/env python
"""REDR and detection F1 of the baseline outputs of DIANA, Learn2Clean, RetClean and DiffPrep.

    python evaluation/encoded_redr.py
    python evaluation/encoded_redr.py --table beers:diana:results/main_table/beers/baselines/diana/cleaned_encoded.npz

Detection F1: ground-truth error cells are feature cells whose dirty value differs from the clean
value, detected cells are feature cells whose output differs from the dirty value. REDR
(evaluation/table_redr.py): (R_before - R_after) / R_before over erroneous records, where after
cleaning only retained rows count and a retained row is erroneous if any of its cells differs from
the clean value.

Table outputs (cleaned.csv or cleaned.csv.gz with the index column) use the canonical cell form of
table_redr.py. Outputs in the canonical encoding (cleaned_encoded.npz with idx_train) are compared in
that encoding: an oracle candidate cell (canonical dirty != canonical clean) is detected when the
output differs from the dirty encoding, and it is restored when the output equals the clean
encoding. A non-candidate cell is an introduced error when the output differs from the dirty
encoding. Without --table the command evaluates the shipped outputs of the four methods on the eight
datasets of Figure 4 and prints one JSON object per output. The redr.json of each run holds its REDR.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

TOL = 1e-6
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
FIGURE_DATASETS = ['beers', 'breast_cancer', 'smartfactory', 'bike', 'mercedes', 'nasa', 'soilmoisture', 'har']
SHIPPED = [('diana', 'cleaned_encoded.npz'), ('learn2clean', 'cleaned_encoded.npz'), ('retclean', 'cleaned.csv.gz'),
           ('diffprep', 'cleaned_encoded.npz')]


def setup(root):
    sys.path.insert(0, os.path.join(root, 'evaluation'))
    import paper_score as P
    from table_redr import canon, read_str
    return P, canon, read_str


def oracle(P, canon, read_str, ds):
    root = P.data_root(None)
    ctx = P.dataset_context(root, ds)
    d = read_str(os.path.join(root, ds, 'dirty_index.csv'))
    c = read_str(os.path.join(root, ds, 'clean_index.csv'))
    cols = [x for x in d.columns if x != 'index']
    D = d[cols].apply(lambda s: s.map(canon)).values[ctx['tr']]
    Cm = c[cols].apply(lambda s: s.map(canon)).values[ctx['tr']]
    return ctx, d.iloc[ctx['tr']].reset_index(drop=True), cols, D, Cm, D != Cm


def f1(tp, ndet, ngt):
    p = tp / ndet if ndet else 0.0
    r = tp / ngt if ngt else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def table_metrics(P, canon, read_str, ds, path):
    ctx, dtr, cols, D, Cm, err = oracle(P, canon, read_str, ds)
    out = read_str(path)
    pos = {v: i for i, v in enumerate(dtr['index'].astype(str).values)}
    out = out[out['index'].astype(str).isin(pos)].reset_index(drop=True)  # full tables: training rows only
    rows = np.array([pos[v] for v in out['index'].astype(str).values])
    O = out[cols].apply(lambda s: s.map(canon)).values
    feats = [i for i, c in enumerate(cols) if c in ctx['enc'].features]
    det = np.zeros_like(err)
    det[rows] = O != D[rows]
    tp = int((det & err)[:, feats].sum())
    det_f1 = f1(tp, int(det[:, feats].sum()), int(err[:, feats].sum()))
    before = int(err.any(axis=1).sum())
    after = int((O != Cm[rows]).any(axis=1).sum())
    return dict(det_f1=det_f1, redr=(before - after) / before if before else 0.0, kept=len(rows))


def encoded_metrics(P, canon, read_str, ds, path):
    ctx, dtr, cols, D, Cm, err = oracle(P, canon, read_str, ds)
    enc = ctx['enc']
    ctr = ctx['clean'].iloc[ctx['tr']].reset_index(drop=True)
    Xd = enc.scaler.transform(enc._raw(P._clean_cols(dtr))[0])
    yd = enc._raw(P._clean_cols(dtr))[1]
    Xc, yc = enc.encode(ctr)
    z = np.load(path)
    X, y, idx = z['X_result'], z['y_result'], z['idx_train']
    pos = {int(v): i for i, v in enumerate(pd.to_numeric(dtr['index']).astype(int).values)}
    rows = np.array([pos[int(v)] for v in idx])
    fpos = {f: j for j, f in enumerate(enc.features)}
    fcols = [(i, fpos[c]) for i, c in enumerate(cols) if c in fpos]
    lab = [i for i, c in enumerate(cols) if c == enc.label]

    def differs(a, b):
        return ~((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= TOL * np.maximum(1.0, np.abs(b))))
    det = np.zeros((len(dtr), len(cols)), dtype=bool)
    bad = np.zeros((len(dtr), len(cols)), dtype=bool)  # erroneous after cleaning (retained rows only)
    for i, j in fcols:
        ch = differs(X[:, j], Xd[rows, j])
        det[rows, i] = ch
        restored = ~differs(X[:, j], Xc[rows, j])
        bad[rows, i] = np.where(err[rows, i], ~restored, ch)
    for i in lab:
        chy = differs(y, yd[rows])
        det[rows, i] = chy
        bad[rows, i] = np.where(err[rows, i], differs(y, yc[rows]), chy)
    feats = [i for i, _ in fcols]
    tp = int((det & err)[:, feats].sum())
    det_f1 = f1(tp, int(det[:, feats].sum()), int(err[:, feats].sum()))
    mapped = feats + lab
    before = int(err[:, mapped].any(axis=1).sum())
    after = int(bad[rows][:, mapped].any(axis=1).sum())
    return dict(det_f1=det_f1, redr=(before - after) / before if before else 0.0, kept=len(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--table', action='append', default=[], help='dataset:name:path')
    a = ap.parse_args()
    P, canon, read_str = setup(ROOT)
    items = [tuple(spec.split(':', 2)) for spec in a.table]
    if not items:
        items = [(ds, method, os.path.join(ROOT, 'results', 'main_table', ds, 'baselines', method, artifact))
                 for ds in FIGURE_DATASETS for method, artifact in SHIPPED]
    for ds, name, path in items:
        fn = encoded_metrics if path.endswith('.npz') else table_metrics
        m = fn(P, canon, read_str, ds, path)
        m.update(dataset=ds, method=name)
        print(json.dumps(m), flush=True)


if __name__ == '__main__':
    main()
