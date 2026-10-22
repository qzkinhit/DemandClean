"""Fixed main-table protocol. Baseline adapters receive dirty train and clean validation only.

The runners of DIANA, Learn2Clean, DiffPrep, RetClean and DAGPipe in baselines/runners/ use this
protocol. It reads data/ and evaluation/paper_score.py of this repository.

Test features may be loaded only after model/pipeline selection is frozen.
Final scoring runs in the canonical sklearn 1.3.2 environment via this file's CLI.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # repository root
sys.path.insert(0, str(ROOT / 'evaluation'))
import paper_score as P

DATA_ROOT = ROOT / 'data'
DATASETS = ['adult', 'beers', 'bike', 'breast_cancer', 'har', 'mercedes',
            'nasa', 'smartfactory', 'soilmoisture', 'flights', 'soccer', 'hospitals']


def canonical_name(dataset):
    return 'hospitals' if dataset == 'hospital' else dataset


def _context(dataset):
    return P.dataset_context(str(DATA_ROOT), canonical_name(dataset))


def _ids(df, positions):
    return (df.iloc[positions]['index'].to_numpy() if 'index' in df
            else np.asarray(positions)).astype(np.int64)


def load_bundle(dataset):
    """No clean training values and no test features/targets are returned."""
    ds = canonical_name(dataset)
    ctx = _context(ds)
    dirty = ctx['dirty'].iloc[ctx['tr']].reset_index(drop=True).copy()
    valid = ctx['clean'].iloc[ctx['va']].reset_index(drop=True).copy()
    enc = ctx['enc']
    X, y = enc.encode(dirty)
    Xm, _ = enc._raw(P._clean_cols(dirty))
    Xm = enc.scaler.transform(Xm)
    Xv, yv = ctx['val']
    task = P.DATASETS[ds]['task']
    model, metric = P.PAPER_MODEL[ds]
    non_index = [c for c in dirty.columns if c not in ('index', 'id')]
    return dict(dataset=ds, task=task, model=model, metric=metric,
                label=P.DATASETS[ds]['label'], features=list(enc.features),
                categorical_cols=sorted(enc.cat), dirty_train=dirty,
                clean_val=valid, X_train=X, X_train_missing=Xm, y_train=y,
                X_val=Xv, y_val=yv, idx_train=_ids(ctx['dirty'], ctx['tr']),
                idx_val=_ids(ctx['dirty'], ctx['va']),
                train_positions=ctx['tr'].copy(), val_positions=ctx['va'].copy(),
                total_rows=len(ctx['dirty']), n_train=len(dirty),
                non_index_columns=non_index,
                gt_cap_cells=int(math.floor(0.5 * len(dirty) * len(non_index))),
                gt_denominator_full_table=P.DS_TOTAL_CELLS.get(ds),
                split_seed=42, encoding='canonical_dirty_train_fitted')


def load_frozen_test(dataset):
    """Only call after selecting and freezing the pipeline. Never returns clean test targets."""
    ds = canonical_name(dataset)
    ctx = _context(ds)
    raw = ctx['clean'].iloc[ctx['te']].reset_index(drop=True).copy()
    return dict(X_test=ctx['test'][0].copy(),
                raw_features=raw.drop(columns=[P.DATASETS[ds]['label']], errors='ignore'),
                idx_test=_ids(ctx['dirty'], ctx['te']))


def validation_score(dataset, X_train, y_train, X_val):
    """Common main-table model, dirty training labels, clean validation labels only."""
    ds = canonical_name(dataset)
    b = load_bundle(ds)
    Xt = np.nan_to_num(np.asarray(X_train, float), nan=0., posinf=0., neginf=0.)
    yt = np.asarray(y_train, float)
    ok = np.isfinite(yt)
    Xt, yt = Xt[ok], yt[ok]
    if len(Xt) < 3 or Xt.shape[1] == 0:
        return float('-inf')
    if b['task'] == 'clustering':
        from sklearn.metrics import silhouette_score
        nc = len(np.unique(b['y_train'][np.isfinite(b['y_train'])]))
        pred = P.make_model('clustering', b['model'], nc).fit_predict(Xt)
        if not 1 < len(np.unique(pred)) < len(Xt):
            return float('-inf')
        return float(silhouette_score(Xt, pred, sample_size=min(len(Xt), 10000), random_state=42))
    model = P.make_model(b['task'], b['model']).fit(Xt, yt)
    Xv = np.nan_to_num(np.asarray(X_val, float), nan=0., posinf=0., neginf=0.)
    yv = np.asarray(b['y_val'], float)
    valid = np.isfinite(yv)
    pred = model.predict(Xv[valid])
    return P._metric(b['task'], b['metric'], yv[valid], pred)


class GTOracle:
    """Count every requested clean training cell; never exposes the clean table."""
    def __init__(self, dataset, output_dir=None):
        self.dataset = canonical_name(dataset)
        self.bundle = load_bundle(self.dataset)
        self.cap = self.bundle['gt_cap_cells']
        self.requests = []
        self.output_dir = Path(output_dir) if output_dir else None

    @property
    def used(self):
        return len(self.requests)

    def query(self, row_ids, columns):
        rows, cols = list(row_ids), list(columns)
        if not set(rows).issubset(set(self.bundle['idx_train'])):
            raise ValueError('GT request contains non-training rows')
        if not set(cols).issubset(set(self.bundle['non_index_columns'])):
            raise ValueError('GT request contains index or unknown columns')
        cost = len(rows) * len(cols)
        if self.used + cost > self.cap:
            raise ValueError('GT request exceeds the 50% training-cell cap')
        ctx = _context(self.dataset)
        clean = ctx['clean'].set_index('index')
        out = clean.loc[rows, cols].copy()
        self.requests.extend([(int(r), str(c)) for r in rows for c in cols])
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'gt_access.json').write_text(json.dumps(self.requests))
        return out


def write_cleaned(dataset, out_dir, cleaned_df, method, gt_cells=0, metadata=None):
    ds = canonical_name(dataset)
    ctx = _context(ds)
    frame = cleaned_df.copy()
    if 'index' not in frame:
        raise ValueError('cleaned output must preserve original row index')
    if frame['index'].duplicated().any():
        raise ValueError('duplicate output row IDs')
    valid = set(_ids(ctx['dirty'], ctx['tr']))
    if not set(frame['index']).issubset(valid):
        raise ValueError('cleaned output contains a non-training row')
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / 'cleaned.csv', index=False)
    X, y = ctx['enc'].encode(frame)
    np.savez_compressed(out / 'cleaned_encoded.npz', X_result=X, y_result=y,
                        idx_train=np.asarray(frame['index'], dtype=np.int64))
    return _write_manifest(ds, out, method, gt_cells, metadata, 'cleaned_table')


def write_transformed(dataset, out_dir, X_train, y_train, X_test, idx_train, idx_test,
                      method, gt_cells=0, metadata=None, X_val=None):
    ds = canonical_name(dataset)
    ctx = _context(ds)
    itr, ite = np.asarray(idx_train, np.int64), np.asarray(idx_test, np.int64)
    if not np.isin(itr, _ids(ctx['dirty'], ctx['tr'])).all() or len(np.unique(itr)) != len(itr):
        raise ValueError('transformed train row IDs violate the split')
    if not np.array_equal(ite, _ids(ctx['dirty'], ctx['te'])):
        raise ValueError('transformed test row IDs/order violate the split')
    Xtr, yt, Xte = np.asarray(X_train, float), np.asarray(y_train, float), np.asarray(X_test, float)
    if len(Xtr) != len(itr) or len(yt) != len(itr) or len(Xte) != len(ite):
        raise ValueError('transformed arrays and row IDs do not align')
    if Xtr.ndim != 2 or Xte.ndim != 2 or Xtr.shape[1] != Xte.shape[1]:
        raise ValueError('transformed feature dimensions do not align')
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays = dict(X_train=Xtr, y_train=yt, X_test=Xte, idx_train=itr, idx_test=ite)
    if X_val is not None:
        arrays['X_val'] = np.asarray(X_val, float)
    np.savez_compressed(out / 'transformed.npz', **arrays)
    return _write_manifest(ds, out, method, gt_cells, metadata, 'transformed_features')


def write_cleaned_encoded(dataset, out_dir, X_train, y_train, idx_train, method, gt_cells=0, metadata=None):
    """Cleaned training rows in the canonical encoding. The test split is never transformed."""
    ds = canonical_name(dataset)
    ctx = _context(ds)
    itr = np.asarray(idx_train, np.int64)
    if not np.isin(itr, _ids(ctx['dirty'], ctx['tr'])).all() or len(np.unique(itr)) != len(itr):
        raise ValueError('cleaned train row IDs violate the split')
    X, y = np.asarray(X_train, float), np.asarray(y_train, float)
    if X.ndim != 2 or X.shape[1] != ctx['test'][0].shape[1] or len(X) != len(itr) or len(y) != len(itr):
        raise ValueError('cleaned arrays do not match the canonical encoding')
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'cleaned_encoded.npz', X_result=X, y_result=y, idx_train=itr)
    return _write_manifest(ds, out, method, gt_cells, metadata, 'cleaned_encoded')


def _write_manifest(dataset, out, method, gt_cells, metadata, representation):
    b = load_bundle(dataset)
    if not 0 <= int(gt_cells) <= b['gt_cap_cells']:
        raise ValueError('GT accounting exceeds the allowed cap')
    record = dict(dataset=dataset, method=method, representation=representation,
                  gt_cells=int(gt_cells), gt_cap_cells=b['gt_cap_cells'],
                  gt_pct_full_table=100. * gt_cells / b['gt_denominator_full_table'],
                  split_seed=42, train_fraction=0.6, val_fraction=0.2, test_fraction=0.2,
                  train_labels='dirty', model=b['model'], metric=b['metric'],
                  test_used_for_selection=False, recorded_at=time.time(), metadata=metadata or {})
    (out / 'manifest.json').write_text(json.dumps(record, indent=2))
    return record


def score_output(dataset, out_dir):
    import sklearn
    if sklearn.__version__ != '1.3.2':
        raise RuntimeError('Final scores require sklearn 1.3.2, found ' + sklearn.__version__)
    ds = canonical_name(dataset)
    out = Path(out_dir)
    ctx = _context(ds)
    rec = json.loads((out / 'manifest.json').read_text())
    if rec['representation'] in ('cleaned_table', 'cleaned_encoded'):
        z = np.load(out / 'cleaned_encoded.npz', allow_pickle=False)
        X, y, test = z['X_result'], z['y_result'], ctx['test']
    else:
        z = np.load(out / 'transformed.npz', allow_pickle=False)
        X, y = z['X_train'], z['y_train']
        test = (np.nan_to_num(z['X_test'], nan=0., posinf=0., neginf=0.), ctx['test'][1])
        X = np.nan_to_num(X, nan=0., posinf=0., neginf=0.)
    model, metric = P.PAPER_MODEL[ds]
    scores, nr, ne = P.score_encoded(P.DATASETS[ds]['task'], X, y, {'test': test}, [model], metric)
    result = dict(dataset=ds, method=rec['method'], model=model, metric=metric,
                  test=scores[model]['test'], n_train_rows=nr, n_test=ne['test'],
                  gt_cells=rec['gt_cells'], gt_pct_full_table=rec['gt_pct_full_table'],
                  sklearn=sklearn.__version__, representation=rec['representation'])
    (out / 'scores.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--out')
    ap.add_argument('--inspect', action='store_true')
    args = ap.parse_args()
    if args.inspect:
        b = load_bundle(args.dataset)
        print(json.dumps({k: b[k] for k in ['dataset', 'task', 'model', 'metric', 'n_train', 'gt_cap_cells']}))
    else:
        print(json.dumps(score_output(args.dataset, args.out)))
