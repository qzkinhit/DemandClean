#!/usr/bin/env python
"""Rebuild the cleaned table of a DemandClean run from its encoded output.

    python evaluation/redecode_npz.py results/main_table/adult/demandclean --out adult_cleaned.csv
    python evaluation/redecode_npz.py results/main_table/adult/demandclean --check

The categorical feature columns are decoded from X_result of cleaned_encoded.npz:
scaler.inverse_transform, round, clip to [0, n_classes - 1], LabelEncoder.inverse_transform, with
the scaler and the LabelEncoders fitted on the dirty training split. Non-finite codes stay missing.
The index, label and numeric columns are kept from cleaned.csv, whose rows are aligned with
X_result. The cleaned.csv files of the adult, beers and mercedes runs are built this way.
--check compares the rebuilt table with the cleaned.csv of the run.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_score as P  # noqa: E402


def redecode(run_dir, data_root):
    run_dir = os.path.abspath(run_dir.rstrip('/'))
    ds = P.read_json(os.path.join(run_dir, 'summary.json'))['dataset']
    npz = [os.path.join(run_dir, 'cleaned_encoded.npz')]
    csv = [os.path.join(run_dir, 'cleaned.csv')]
    z = np.load(npz[0], allow_pickle=True)
    ctx = P.dataset_context(data_root, ds)
    enc = ctx['enc']
    cols = [str(c) for c in z['column_names']]
    if cols != list(enc.features):
        raise ValueError(f'column mismatch {cols[:4]} vs {enc.features[:4]}')
    old = pd.read_csv(csv[0], dtype=str, keep_default_na=False)
    X = np.asarray(z['X_result'], dtype=float)
    if len(old) != len(X):
        raise ValueError(f'row mismatch csv {len(old)} npz {len(X)}')
    Xi = enc.scaler.inverse_transform(X) if len(X) else X
    n_nonfinite = 0
    for j, col in enumerate(cols):
        if col in enc.cat and col in enc.le:
            le = enc.le[col]
            v = Xi[:, j]
            fin = np.isfinite(v)
            n_nonfinite += int((~fin).sum())
            code = np.clip(np.round(np.where(fin, v, 0.0)).astype(int), 0, len(le.classes_) - 1)
            dec = le.inverse_transform(code).astype(object)
            dec[~fin] = ''
            old[col] = dec
    return ds, old, dict(rows=len(old), nonfinite_cat_cells=n_nonfinite,
                         n_cat=len(enc.cat & set(cols)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('run_dir')
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--out', default=None, help='write the rebuilt table to this CSV file')
    ap.add_argument('--check', action='store_true', help='compare with cleaned.csv of the run')
    a = ap.parse_args()
    ds, df, meta = redecode(a.run_dir, P.data_root(a.data_root))
    rec = dict(dataset=ds, **meta)
    if a.out:
        df.to_csv(a.out, index=False)
        rec['table'] = a.out
    if a.check:
        old = pd.read_csv(os.path.join(a.run_dir, 'cleaned.csv'), dtype=str, keep_default_na=False)
        new = df.astype(str).replace('nan', '')
        rec['equals_cleaned_csv'] = bool(list(old.columns) == list(new.columns)
                                         and (old.values == new.values).all())
    print(json.dumps(rec))


if __name__ == '__main__':
    main()
