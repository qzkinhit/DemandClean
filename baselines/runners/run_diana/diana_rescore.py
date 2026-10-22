#!/usr/bin/env python
"""Rescore DIANA under the untouched-test rule.

DIANA's official operators (per-column imputation and outlier handling chosen by its knowledge
base) change only the dirty training matrix in the canonical encoding and keep every row. The
earlier run stored that matrix together with the canonical clean test encoding as
transformed.npz. This script verifies that the stored test features equal the canonical test
encoding exactly, stores the cleaned training matrix as cleaned_encoded.npz and scores it with
the untouched test split.

    python baselines/runners/run_diana/diana_rescore.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]  # repository root
sys.path.insert(0, str(ROOT / 'baselines' / 'runners'))
BASE = ROOT / 'results' / 'runs'
import common  # noqa: E402

for ds in common.DATASETS:
    src = BASE / 'diana' / ds
    z = np.load(src / 'transformed.npz')
    ctx = common._context(ds)
    same = bool(np.array_equal(np.asarray(z['X_test'], float), np.asarray(ctx['test'][0], float), equal_nan=True))
    idx_ok = bool(np.array_equal(z['idx_test'], common._ids(ctx['dirty'], ctx['te'])))
    if not (same and idx_ok):
        print(ds, 'TEST FEATURES DIFFER, not rescored', flush=True)
        continue
    meta = json.loads((src / 'manifest.json').read_text()).get('metadata', {})
    meta = dict(meta, rescored_from=str((src / 'transformed.npz').relative_to(ROOT)),
                test_features_check='stored test features equal the canonical clean test encoding')
    out = BASE / 'diana_clean' / ds
    common.write_cleaned_encoded(ds, out, z['X_train'], z['y_train'], z['idx_train'], 'DIANA', gt_cells=0, metadata=meta)
    r = common.score_output(ds, out)
    old = json.loads((src / 'scores.json').read_text())['test']
    print(ds, 'old %.6f new %.6f rows %d' % (old, r['test'], r['n_train_rows']), flush=True)
