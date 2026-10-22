#!/usr/bin/env python
"""Scorer of every downstream number of the main table (tab:main_results).

    python evaluation/paper_score.py results/main_table/beers/demandclean
    python evaluation/paper_score.py --dataset beers --table data/beers/dirty_index.csv

A DemandClean run is scored on its cleaned training split, a table (a reference or the output of
a baseline) on its rows of the training split. Both are scored the same way.

    train rows : the cleaned training split (cleaned_encoded.npz of a run, deleted rows absent)
    test rows  : clean_index.csv rows of the 60/20/20 test split, encoded with the LabelEncoders
                 and the StandardScaler fitted on the dirty training split
    split      : np.arange(n) -> train_test_split(test_size=0.4, random_state=42)
                 -> train_test_split(test_size=0.5, random_state=42)
    NaN        : rows with a NaN label are dropped, remaining NaN features are filled with the
                 column mean of the same array
    models     : RF(100, random_state=42), LR(max_iter=1000), LinearRegression, GB(random_state=42), ...
    clustering : fit_predict on the cleaned training rows, silhouette on the same rows
                 (sample_size=min(n, 10000), random_state=42), no test split involved
    Categories absent from the dirty training split are encoded as missing (--nearest maps them to
    the nearest known category by edit distance instead).

GT% is the number of ground-truth cells (calibration seed set plus ground-truth repairs) divided
by the cell count of the full table (total_cells in the dataset config), in percent.

Tol_rel = (P_m - P_NF) / (P_Repa - P_NF), each performance clipped to [0, 1] first, where P_NF and
P_Repa score the dirty and the clean training split on the same path.

The reported numbers were produced with scikit-learn 1.3.2. RF, GB and the ill-conditioned nasa
LinearRegression differ in the third decimal under other versions.
"""

import argparse
import glob
import json
import os
import sys
import warnings
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, r2_score, silhouette_score

warnings.filterwarnings("ignore")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def _load_dataset_configs():
    out = {}
    for path in sorted(glob.glob(os.path.join(_ROOT, 'configs', 'datasets', '*.yaml'))):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        out[cfg['dataset']] = cfg
    return out


CONFIGS = _load_dataset_configs()
DATASETS = {ds: dict(task=c['task'], label=c['label_col'], cat=set(c.get('categorical_cols') or []))
            for ds, c in CONFIGS.items()}
PAPER_MODEL = {ds: (c['downstream_model'], c['metric']) for ds, c in CONFIGS.items()}
DS_TOTAL_CELLS = {ds: c['total_cells'] for ds, c in CONFIGS.items() if c.get('total_cells')}

MODELS = {
    'classification': ['rf', 'lr', 'svm', 'knn', 'dt', 'gb'],
    'regression': ['rf', 'lr', 'ridge', 'lasso', 'knn', 'gb'],
    'clustering': ['kmeans', 'agglomerative', 'spectral'],
}

NA_TOKENS = ['empty', 'Empty', 'EMPTY', 'nan', 'NaN', 'NULL', 'null']

USE_NEAREST = None


def _use_nearest(ds):
    """Use the archived dataset encoding unless the caller explicitly overrides it."""
    if USE_NEAREST is not None:
        return bool(USE_NEAREST)
    return CONFIGS[ds].get('evaluation', {}).get('unseen_categories', 'nan') == 'nearest'


def make_model(task, name, n_clusters=None):
    if task == 'classification':
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        return {
            'rf': lambda: RandomForestClassifier(n_estimators=100, random_state=42),
            'lr': lambda: LogisticRegression(max_iter=1000, random_state=42),
            'svm': lambda: SVC(random_state=42),
            'knn': lambda: KNeighborsClassifier(),
            'dt': lambda: DecisionTreeClassifier(random_state=42),
            'gb': lambda: GradientBoostingClassifier(random_state=42),
        }[name]()
    if task == 'regression':
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.neighbors import KNeighborsRegressor
        return {
            'rf': lambda: RandomForestRegressor(n_estimators=100, random_state=42),
            'lr': lambda: LinearRegression(),
            'ridge': lambda: Ridge(solver='lsqr', random_state=42),
            'lasso': lambda: Lasso(random_state=42),
            'knn': lambda: KNeighborsRegressor(),
            'gb': lambda: GradientBoostingRegressor(random_state=42),
        }[name]()
    from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
    return {
        'kmeans': lambda: KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
        'agglomerative': lambda: AgglomerativeClustering(n_clusters=n_clusters),
        'spectral': lambda: SpectralClustering(n_clusters=n_clusters, random_state=42,
                                               affinity='nearest_neighbors'),
    }[name]()


def find_nearest_known(value, known_values, threshold=0.6):
    """Nearest known value by difflib ratio (demandclean.utils.edit_distance.find_nearest_known)."""
    if not value or not known_values:
        return None
    best, best_ratio = None, -1.0
    for k in known_values:
        r = SequenceMatcher(None, value, k).ratio()
        if r > best_ratio:
            best_ratio, best = r, k
    return best if best_ratio >= threshold else None


def _clean_cols(df):
    df = df.copy()
    df.columns = [c.strip().strip('﻿') for c in df.columns]
    df.replace(NA_TOKENS, np.nan, inplace=True)
    return df


class TrainEncoder:
    """LabelEncoders and StandardScaler fitted on the dirty 60% training split."""

    def __init__(self, ds, dirty_train_df):
        cfg = DATASETS[ds]
        self.use_nearest = _use_nearest(ds)
        self.label = cfg['label']
        d = _clean_cols(dirty_train_df)
        drop = [c for c in ['index', 'id', self.label] if c in d.columns]
        self.features = [c for c in d.columns if c not in drop]
        self.cat = set(cfg['cat']) & set(self.features)
        self.le = {}
        for col in self.features:
            if col in self.cat:
                le = LabelEncoder()
                le.fit(d[col].dropna().astype(str).unique())
                self.le[col] = le
        try:
            pd.to_numeric(d[self.label].dropna(), errors='raise')
        except (ValueError, TypeError):
            le = LabelEncoder()
            le.fit(d[self.label].dropna().astype(str).unique())
            self.le[self.label] = le
        X, _ = self._raw(d)
        Xf = X.copy()
        m = np.nanmean(Xf, axis=0)
        for j in range(Xf.shape[1]):
            Xf[np.isnan(Xf[:, j]), j] = m[j] if not np.isnan(m[j]) else 0.0
        self.scaler = StandardScaler().fit(Xf)

    def _raw(self, df):
        X_df = df[self.features].copy()
        y = df[self.label].copy()
        for col in self.features:
            if col in self.cat and col in self.le:
                le = self.le[col]
                ks, kl = set(le.classes_), list(le.classes_)
                lut = {}

                def enc(v, _le=le, _ks=ks, _kl=kl, _lut=lut):
                    if v in _lut:
                        return _lut[v]
                    if v in _ks:
                        out = _le.transform([v])[0]
                    elif not self.use_nearest:
                        out = np.nan
                    else:
                        near = find_nearest_known(v, _kl, threshold=0.3)
                        out = _le.transform([near])[0] if near is not None else np.nan
                    _lut[v] = out
                    return out

                nan = X_df[col].isna()
                X_df.loc[~nan, col] = X_df.loc[~nan, col].astype(str).map(enc)
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
        if self.label in self.le:
            le = self.le[self.label]
            ks, kl = set(le.classes_), list(le.classes_)

            def enc_y(v, _le=le, _ks=ks, _kl=kl):
                s = str(v)
                if s in _ks:
                    return _le.transform([s])[0]
                if not self.use_nearest:
                    return np.nan
                near = find_nearest_known(s, _kl, threshold=0.3)
                return _le.transform([near])[0] if near is not None else np.nan

            nan = y.isna()
            ye = y.copy()
            ye.loc[~nan] = y.loc[~nan].astype(str).map(enc_y)
            y = pd.to_numeric(ye, errors='coerce')
        else:
            y = pd.to_numeric(y, errors='coerce')
        return X_df.values.astype(float), y.values.astype(float)

    def encode(self, df):
        """Column-mean fill before scaling, returns (X_scaled, y)."""
        X, y = self._raw(_clean_cols(df))
        m = np.nanmean(X, axis=0)
        for j in range(X.shape[1]):
            X[np.isnan(X[:, j]), j] = m[j] if not np.isnan(m[j]) else 0.0
        return self.scaler.transform(X), y


def split_idx(n):
    all_idx = np.arange(n)
    tr, tmp = train_test_split(all_idx, test_size=0.4, random_state=42)
    va, te = train_test_split(tmp, test_size=0.5, random_state=42)
    return tr, va, te


def _nan_prep(X, y):
    X = np.array(X, dtype=float, copy=True)
    y = np.array(y, dtype=float, copy=True)
    ok = ~np.isnan(y)
    X, y = X[ok], y[ok]
    m = np.nanmean(X, axis=0)
    nan = np.isnan(X)
    if nan.any():
        for j in range(X.shape[1]):
            X[nan[:, j], j] = m[j]
    return X, y


def _metric(task, metric, y_true, y_pred):
    if metric == 'accuracy':
        return float(accuracy_score(y_true, y_pred))
    return float(r2_score(y_true, y_pred))


def score_encoded(task, X_res, y_res, evals, models, metric):
    """Encoded branch. evals: {'test': (X, y), 'val': (X, y)}. Returns {model: {split: v}}."""
    Xtr, ytr = _nan_prep(X_res, y_res)
    prepared = {k: _nan_prep(*v) for k, v in evals.items()}
    out = {}
    if task == 'clustering':
        n_clusters = len(np.unique(prepared['test'][1]))
        for name in models:
            yp = make_model(task, name, n_clusters).fit_predict(Xtr)
            s = float(silhouette_score(Xtr, yp, sample_size=min(len(Xtr), 10000),
                                       random_state=42))
            out[name] = {'test': s, 'val': None}
        return out, len(Xtr), {k: len(v[1]) for k, v in prepared.items()}
    for name in models:
        res = {}
        if len(prepared['test'][1]) == len(Xtr):
            # Internal 80/20 split when the row counts coincide (not reached with the
            # 60/20/20 layout).
            tri, tei = train_test_split(np.arange(len(Xtr)), test_size=0.2, random_state=42)
            mdl = make_model(task, name).fit(Xtr[tri], ytr[tri])
            res['test'] = _metric(task, metric, prepared['test'][1][tei], mdl.predict(Xtr[tei]))
            res['val'] = None
        else:
            mdl = make_model(task, name).fit(Xtr, ytr)
            for k, (Xe, ye) in prepared.items():
                res[k] = _metric(task, metric, ye, mdl.predict(Xe))
        out[name] = res
    return out, len(Xtr), {k: len(v[1]) for k, v in prepared.items()}


def load_dataset(data_root, ds):
    d = os.path.join(data_root, ds)
    dp = os.path.join(d, 'dirty_index.csv')
    cp = os.path.join(d, 'clean_index.csv')
    if not os.path.exists(dp):
        dp = os.path.join(d, 'dirty_with_index.csv')
    if not os.path.exists(cp):
        cp = os.path.join(d, 'clean_with_index.csv')
    return pd.read_csv(dp), pd.read_csv(cp)


_CACHE = {}


def dataset_context(data_root, ds):
    key = (data_root, ds, _use_nearest(ds))
    if key not in _CACHE:
        dirty, clean = load_dataset(data_root, ds)
        tr, va, te = split_idx(len(dirty))
        enc = TrainEncoder(ds, dirty.iloc[tr].reset_index(drop=True))
        ctx = dict(dirty=dirty, clean=clean, tr=tr, va=va, te=te, enc=enc)
        ctx['test'] = enc.encode(clean.iloc[te].reset_index(drop=True))
        ctx['val'] = enc.encode(clean.iloc[va].reset_index(drop=True))
        _CACHE[key] = ctx
    return _CACHE[key]


def data_root(explicit=None):
    return os.path.abspath(explicit or os.path.join(_ROOT, 'data'))


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def gt_info(summary, ds):
    g = (summary or {}).get('ground_truth_cells') or {}
    total = g.get('total')
    if total is None:
        return {}
    return {
        'gt_cells': total,
        'gt_repairs': g.get('repairs'),
        'gt_seed_set': g.get('seed_set'),
        'gt_pct': round(100.0 * total / DS_TOTAL_CELLS[ds], 4) if ds in DS_TOTAL_CELLS else None,
    }


def _clip01(v):
    return None if v is None else min(1.0, max(0.0, float(v)))


def tol_rel(p_m, p_nf, p_repa):
    p_m, p_nf, p_repa = _clip01(p_m), _clip01(p_nf), _clip01(p_repa)
    if None in (p_m, p_nf, p_repa) or abs(p_repa - p_nf) < 1e-12:
        return None
    return (p_m - p_nf) / (p_repa - p_nf)


def score_run(run_dir, root=None, all_models=False, references=True):
    """Downstream score, GT% and Tol_rel of one DemandClean run directory."""
    run_dir = os.path.abspath(run_dir.rstrip('/'))
    root = data_root(root)
    summary = read_json(os.path.join(run_dir, 'summary.json'))
    ds = summary['dataset']
    task = DATASETS[ds]['task']
    model, metric = PAPER_MODEL[ds]
    models = MODELS[task] if all_models else [model]
    out = dict(dataset=ds, variant=summary.get('variant'), model=model, metric=metric, nearest=_use_nearest(ds))
    z = np.load(os.path.join(run_dir, 'cleaned_encoded.npz'), allow_pickle=True)
    ctx = dataset_context(root, ds)
    res, n_tr, n_eval = score_encoded(task, z['X_result'], z['y_result'],
                                      {'test': ctx['test']}, models, metric)
    out.update(n_train_rows=n_tr, n_test=n_eval['test'], test=res[model]['test'])
    if all_models:
        out['all_models'] = {m: r['test'] for m, r in res.items()}
    out.update(gt_info(summary, ds))
    if references:
        refs = {name: score_table(ds, os.path.join(root, ds, fname), root)['test']
                for name, fname in (('NoFix', 'dirty_index.csv'), ('RepairAll', 'clean_index.csv'))}
        out['references'] = refs
        out['tol_rel'] = tol_rel(out['test'], refs['NoFix'], refs['RepairAll'])
    return out


def score_table(ds, table_path, root=None, all_models=False):
    """Score a table with the full row count (for example NoFix = dirty_index.csv, RepairAll =
    clean_index.csv) on its rows of the training split. Tables that hold the training rows
    only, such as the output of a baseline, are scored by evaluation/rescore_tables.py."""
    root = data_root(root)
    ctx = dataset_context(root, ds)
    task = DATASETS[ds]['task']
    model, metric = PAPER_MODEL[ds]
    models = MODELS[task] if all_models else [model]
    df = pd.read_csv(table_path)
    if len(df) == len(ctx['dirty']):
        df = df.iloc[ctx['tr']].reset_index(drop=True)
    X, y = ctx['enc'].encode(df)
    res, n_tr, n_eval = score_encoded(task, X, y, {'test': ctx['test']}, models, metric)
    out = dict(dataset=ds, table=os.path.basename(table_path), model=model, metric=metric,
               n_train_rows=n_tr, n_test=n_eval['test'], test=res[model]['test'])
    if all_models:
        out['all_models'] = {m: r['test'] for m, r in res.items()}
    return out


def write_scores(run_dir, scores):
    import sklearn
    scores = dict(scores)
    scores['scikit_learn'] = sklearn.__version__
    with open(os.path.join(run_dir, 'scores.json'), 'w') as f:
        json.dump(scores, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description='Scorer of the downstream numbers of the main table.')
    ap.add_argument('run_dirs', nargs='*')
    ap.add_argument('--data-root', default=None, help='directory holding <dataset>/dirty_index.csv and clean_index.csv')
    ap.add_argument('--all-models', action='store_true', help='score every candidate downstream model')
    ap.add_argument('--no-references', action='store_true', help='skip NoFix/RepairAll and Tol_rel')
    ap.add_argument('--write', action='store_true', help='write scores.json into each run directory')
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--table', default=None, help='score a table with the full row count (needs --dataset)')
    ap.add_argument('--nearest', action='store_true', default=None,
                    help='map categories absent from the training split by edit distance')
    ap.add_argument('--encoding', choices=['auto', 'nan', 'nearest'], default='auto',
                    help='auto uses the encoding recorded in the dataset configuration')
    a = ap.parse_args()
    global USE_NEAREST
    USE_NEAREST = True if a.nearest else {'auto': None, 'nan': False, 'nearest': True}[a.encoding]
    if a.table:
        print(json.dumps(score_table(a.dataset, a.table, a.data_root, a.all_models)))
        return
    if not a.run_dirs:
        ap.error('give at least one run directory or --table')
    rc = 0
    for rd in a.run_dirs:
        try:
            s = score_run(rd, a.data_root, a.all_models, references=not a.no_references)
            if a.write:
                write_scores(rd, s)
            print(json.dumps(s), flush=True)
        except Exception as e:
            rc = 1
            print(json.dumps({'run_dir': os.path.basename(rd.rstrip('/')),
                              'error': f'{type(e).__name__}: {e}'}), flush=True)
    sys.exit(rc)


if __name__ == '__main__':
    main()
