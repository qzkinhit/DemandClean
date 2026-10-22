"""Contract check for the SAGA cleaned tables.

For every dataset given on the command line it verifies that
<results>/saga/<ds>_saga/<ds>_saga_cleaned.csv has the original
columns (index column included), that its index values are a subset of the
60% train split of evaluation/paper_score.py:split_idx and contain no val/test index,
and reports rows kept, rows deleted and the number of feature cells that
differ from the dirty table, split into cells moved to the clean value and
cells moved elsewhere.

usage: python saga_check_outputs.py <repo> <results_root> [--method=<name>] <ds> [<ds> ...]
(--method selects the result folder, default saga, e.g. saga_lblsur)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

repo, root = sys.argv[1], sys.argv[2]
args = sys.argv[3:]
method = 'saga'
if args and args[0].startswith('--method='):
    method = args.pop(0).split('=', 1)[1]
sys.path.insert(0, os.path.join(repo, 'baselines', 'runners'))
from protocol import load_protocol, read_table as _read  # noqa: E402

MISS = {'', 'empty', 'Empty', 'EMPTY', 'nan', 'NaN', 'NULL', 'null', 'None', '__NULL__'}


def same(a, b):
    a = np.where(pd.Series(a).isin(MISS), 'empty', a)
    b = np.where(pd.Series(b).isin(MISS), 'empty', b)
    fa = pd.to_numeric(pd.Series(a), errors='coerce').values
    fb = pd.to_numeric(pd.Series(b), errors='coerce').values
    num = ~np.isnan(fa) & ~np.isnan(fb)
    return np.where(num, np.isclose(fa, fb, rtol=1e-6, atol=1e-9), a == b)


for ds in args:
    p = os.path.join(root, method, f'{ds}_{method}', f'{ds}_{method}_cleaned.csv')
    if not os.path.exists(p):
        print(json.dumps({'dataset': ds, 'ok': False, 'reason': 'missing'}))
        continue
    DATASETS, _, _, _, split_idx = load_protocol(repo, ds)
    d = _read(os.path.join(repo, 'data', ds, 'dirty_index.csv'))
    c = _read(os.path.join(repo, 'data', ds, 'clean_index.csv'))
    tr, va, te = split_idx(len(c))
    s = pd.read_csv(p, dtype=str, keep_default_na=False)
    tr_ids = set(d['index'].astype(str).values[tr])
    other = set(d['index'].astype(str).values[va]) | set(d['index'].astype(str).values[te])
    sid = s['index'].astype(str)
    label = DATASETS[ds]['label_col']
    feats = [col for col in d.columns if col not in ('index', label)]
    dt = d.set_index(d['index'].astype(str))
    ct = c.set_index(c['index'].astype(str))
    changed = to_clean = 0
    for col in feats:
        a = dt.loc[sid, col].astype(str).values
        b = s[col].astype(str).values
        g = ct.loc[sid, col].astype(str).values
        ch = ~same(a, b)
        changed += int(ch.sum())
        to_clean += int((ch & same(b, g)).sum())
    print(json.dumps({
        'dataset': ds, 'method': method, 'ok': list(s.columns) == list(d.columns) and set(sid) <= tr_ids and not (set(sid) & other)
        and sid.is_unique,
        'columns_match': list(s.columns) == list(d.columns), 'rows': int(len(s)), 'train_rows': int(len(tr)),
        'rows_deleted': int(len(tr_ids - set(sid))), 'val_or_test_rows': int(len(set(sid) & other)),
        'changed_cells': changed, 'changed_to_clean_value': to_clean}))
