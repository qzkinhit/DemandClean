#!/usr/bin/env python
"""Build the ReplaceAll reference table of the main table.

    python evaluation/build_replaceall.py --dataset beers --out replaceall.csv.gz

The oracle error mask marks every cell whose dirty value differs from the clean value (numeric
tolerance 1e-6). Every marked cell, features and label, is replaced by the mean (numeric column)
or the mode (categorical column, and the label of a classification or clustering task) of the
unmarked cells of the training rows. A column without an unmarked training cell takes the mean
or the mode of its parseable dirty training cells. The table keeps all rows, the scorer uses its
training rows (evaluation/rescore_tables.py).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_score as P  # noqa: E402


def error_mask(d, c, cols):
    """Oracle error mask: a cell is an error when its dirty and clean values differ (numeric tolerance 1e-6, a missing value differs from a present one)."""
    M = np.zeros((len(d), len(cols)), bool)
    for j, k in enumerate(cols):
        dv, cv = d[k], c[k]
        dn = pd.to_numeric(dv, errors='coerce').values.astype(float)
        cn = pd.to_numeric(cv, errors='coerce').values.astype(float)
        dna, cna = dv.isna().values, cv.isna().values
        both = ~np.isnan(dn) & ~np.isnan(cn)
        with np.errstate(invalid='ignore'):
            num = np.abs(dn - cn) > 1e-6
        s = dv.astype(str).str.strip().values != cv.astype(str).str.strip().values
        M[:, j] = np.where(dna & cna, False, np.where(dna | cna, True, np.where(both, num, s)))
    return M


def numeric_cols(c_tr, cols):
    out = []
    for k in cols:
        v = c_tr[k]
        nn = v.notna().sum()
        if pd.to_numeric(v, errors='coerce').notna().sum() / max(nn, 1) > 0.5:
            out.append(k)
    return out


def replaceall_table(ds, d, c, tr):
    """ReplaceAll table of dataset ds from the dirty table d, the clean table c and the
    training positions tr."""
    lab = P.DATASETS[ds]['label']
    task = P.DATASETS[ds]['task']
    cols = [k for k in d.columns if k != 'index']
    M = error_mask(d, c, cols)
    ci = {k: j for j, k in enumerate(cols)}
    c_tr = c.iloc[tr]
    num = set(numeric_cols(c_tr, cols))
    cat_label = task in ('classification', 'clustering')
    stats = {}
    for k in cols:
        j = ci[k]
        if not M[:, j].any():
            continue
        ok = ~M[tr, j]
        vals = c_tr[k].values[ok]
        is_num = (k in num) and not (k == lab and cat_label)
        if is_num:
            v = pd.to_numeric(pd.Series(vals), errors='coerce')
            if v.notna().sum() == 0:
                v = pd.to_numeric(d[k].iloc[tr], errors='coerce')
            stats[k] = float(v.mean()) if v.notna().sum() else 0.0
        else:
            v = pd.Series(vals).dropna()
            if len(v) == 0:
                v = d[k].iloc[tr].dropna()
            stats[k] = v.mode().iloc[0] if len(v) else ''
    x = d.copy()
    for k in cols:
        if k not in stats:
            continue
        m = M[:, ci[k]]
        if k in num and not (k == lab and cat_label):
            x[k] = x[k].astype(object)
        x.loc[m, k] = stats[k]
    return x, int(M.sum()), int(M[tr].sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    root = P.data_root(a.data_root)
    d = pd.read_csv(os.path.join(root, a.dataset, 'dirty_index.csv'), low_memory=False)
    c = pd.read_csv(os.path.join(root, a.dataset, 'clean_index.csv'), low_memory=False)
    tr, _, _ = P.split_idx(len(d))
    x, n_err, n_err_train = replaceall_table(a.dataset, d, c, tr)
    x.to_csv(a.out, index=False)
    print(json.dumps(dict(dataset=a.dataset, table=a.out, error_cells=n_err,
                          error_cells_train=n_err_train)))


if __name__ == '__main__':
    main()
