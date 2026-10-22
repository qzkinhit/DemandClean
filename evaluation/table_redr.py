#!/usr/bin/env python
"""Record-level error drop rate (REDR) of cleaned tables on the training split.

    python evaluation/table_redr.py                           # every DemandClean run and table in results/
    python evaluation/table_redr.py --table beers:demandclean:results/main_table/beers/demandclean/cleaned.csv

  REDR = (R_before - R_after) / R_before
  R_before  erroneous records (rows with any cell whose dirty value != clean value, over all
            columns except index, features and label) of the training split
  R_after   erroneous records among the training rows the table keeps. A deleted row leaves the
            training data, so its errors no longer count, and a row that was clean but now holds
            a wrong value counts as erroneous.
REDR is 1 when every erroneous record is repaired or deleted and negative when a method writes
more wrong values than it fixes. Values are compared in a canonical form (numbers rounded to 6
decimals, missing tokens as empty, strings lower-cased). One JSON object per table is printed
(results/scores/redr.jsonl holds the output for the shipped tables).
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

MISS = {'', 'empty', 'Empty', 'EMPTY', 'nan', 'NaN', 'NULL', 'null', 'None', '__NULL__'}
DS = ['adult', 'beers', 'breast_cancer', 'smartfactory', 'bike', 'mercedes', 'nasa', 'soilmoisture', 'har']


def canon(v):
    s = str(v).strip()
    if s in MISS:
        return ''
    try:
        return repr(round(float(s), 6))
    except ValueError:
        return s.lower()


def read_str(p):
    df = pd.read_csv(p, dtype=str, keep_default_na=False, low_memory=False)
    df.columns = [c.strip().strip('﻿') for c in df.columns]
    if 'index' in df.columns:
        # some method outputs write the index as 1395.0; align on the integer value
        num = pd.to_numeric(df['index'], errors='coerce')
        ok = num.notna() & (num == num.round())
        df.loc[ok, 'index'] = num[ok].astype('int64').astype(str)
    return df


_CTX = {}


def context(root, ds):
    if ds not in _CTX:
        d = read_str(os.path.join(root, ds, 'dirty_index.csv'))
        c = read_str(os.path.join(root, ds, 'clean_index.csv'))
        tr, _, _ = P.split_idx(len(d))
        cols = [x for x in d.columns if x != 'index']
        D = d.set_index('index')[cols].apply(lambda s: s.map(canon))
        C = c.set_index('index')[cols].apply(lambda s: s.map(canon))
        _CTX[ds] = dict(D=D, C=C, cols=cols, tr_index=d['index'].values[tr], all_index=d['index'].values)
    return _CTX[ds]


def redr(root, ds, path, full_rows=False):
    ctx = context(root, ds)
    t = read_str(path)
    if 'index' not in t.columns:
        raise ValueError('no index column')
    t = t.drop_duplicates('index').set_index('index')
    scope = ctx['all_index'] if full_rows else ctx['tr_index']
    D, C = ctx['D'].loc[scope], ctx['C'].loc[scope]
    cols = [x for x in ctx['cols'] if x in t.columns]
    missing_cols = [x for x in ctx['cols'] if x not in t.columns]
    r_before = int((D[ctx['cols']] != C[ctx['cols']]).any(axis=1).sum())
    kept = [i for i in scope if i in t.index]
    T = t.loc[kept, cols].apply(lambda s: s.map(canon))
    Ck = C.loc[kept]
    err_after = (T != Ck[cols]).any(axis=1)
    # a column the table dropped keeps its dirty values in the kept rows
    if missing_cols:
        err_after = err_after | (D.loc[kept, missing_cols] != Ck[missing_cols]).any(axis=1)
    r_after = int(err_after.sum())
    Dk = D.loc[kept, cols]
    changed = (T != Dk)
    return dict(scope_rows=len(scope), kept_rows=len(kept), r_before=r_before, r_after=r_after,
                redr=(r_before - r_after) / r_before if r_before else None,
                changed=int(changed.values.sum()),
                changed_to_clean=int((changed & (T == Ck[cols])).values.sum()),
                changed_dirty=int((changed & (Dk != Ck[cols])).values.sum()),
                dirty_kept=int((Dk != Ck[cols]).values.sum()),
                missing_cols=missing_cols)


def shipped_tables(ds):
    """(name, path) of the DemandClean run and of every cleaned table of dataset ds in results/."""
    out = []
    run = os.path.join(P._ROOT, 'results', 'main_table', ds, 'demandclean', 'cleaned.csv')
    if os.path.exists(run):
        out.append(('demandclean', run))
    for path in sorted(glob.glob(os.path.join(P._ROOT, 'results', '*', ds, '*', '*', 'cleaned.csv.gz'))):
        out.append((os.path.basename(os.path.dirname(path)), path))
    for path in sorted(glob.glob(os.path.join(P._ROOT, 'results', 'budget', ds, '*', 'cleaned.csv.gz'))):
        out.append((os.path.basename(os.path.dirname(path)), path))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--datasets', default=','.join(DS))
    ap.add_argument('--table', action='append', default=[], help='dataset:name:path')
    a = ap.parse_args()
    root = P.data_root(a.data_root)
    items = [tuple(spec.split(':', 2)) for spec in a.table]
    if not items:
        for ds in a.datasets.split(','):
            items += [(ds, name, path) for name, path in shipped_tables(ds)]
    for ds, name, path in items:
        rec = dict(dataset=ds, method=name, table=os.path.relpath(os.path.abspath(path), P._ROOT))
        try:
            rec.update(redr(root, ds, path))
        except Exception as e:  # keep going
            rec['error'] = f'{type(e).__name__}: {e}'
        print(json.dumps(rec), flush=True)


if __name__ == '__main__':
    main()
