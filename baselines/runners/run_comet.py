"""Runner of the Comet baseline under the 60/20/20 protocol of the DemandClean runs.

Comet (Mohammed, Naumann, Harmouch, EDBT 2025) recommends, step by step, which
feature to clean next. This runner does all IO and protocol work, the method
itself lives in baselines/Methods/Comet.

Protocol
  split       the 60/20/20 split of evaluation/paper_score.py (seed 42, over the rows of
              clean_index.csv). The test rows are dropped right after the split and never
              read further.
  encoding    preprocess_data of scripts/run_demandclean.py fitted on the dirty training
              split, encode_subset for every matrix (the encoder of the scorer)
  training    dirty training rows, labels are the clean labels by default (--labels), the
              labels the scorer attaches to every training row
  selection   clean 20% validation split (Comet's internal score): macro F1 for
              classification (paper), R^2 for regression, silhouette of the nearest-centroid
              labels of an Agglomerative clustering for clustering (extensions)
  Cleaner     ground-truth repairs of dirty training cells, counted as GT cells
  output      the training split with the repaired cells, original columns plus
              index, written to
              <output_root>/<method>/<ds>_<method>/<ds>_<method>_cleaned.csv

Example (the budgets of Table tab:main_results are listed in results/README.md)
  python baselines/runners/run_comet.py --dataset beers --method_name comet
  python baselines/runners/run_comet.py --dataset beers --method_name comet_dcbudget --gt_budget 248 --labels dirty
"""

import argparse
import functools
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.dirname(HERE)
REPO_DIR = os.path.dirname(BENCH_DIR)

# Downstream model of each dataset in the paper's main table, used as Comet's utility model.
# The regression model 'linreg' of bike and nasa is sklearn LinearRegression, 'ridge' is Ridge(alpha=1).
FIRST_MODEL = {'adult': 'rf', 'beers': 'rf', 'breast_cancer': 'lr', 'smartfactory': 'rf',
               'bike': 'linreg', 'mercedes': 'rf', 'nasa': 'linreg', 'soilmoisture': 'gb',
               'har': 'agglo', 'flights': 'rf', 'hospitals': 'rf', 'soccer': 'rf'}
# Why a run replaces the downstream model by another utility model (measured with --dry_run, one process).
MODEL_SUBSTITUTION = {
    'soilmoisture': 'one GradientBoosting fit and score takes 3.33 s and an iteration needs 6081 fits '
                    '(304 feature/error pairs x 20 pollution fits + 1), about 5.6 h per iteration and about 170 h '
                    'for the 50-unit run; Ridge(alpha=1) is the linear model used instead',
    'mercedes': 'one RandomForest fit and score takes 129.8 s and an iteration needs 16161 fits (808 pairs), '
                'about 24 days per iteration; Ridge(alpha=1) is the linear model used instead, gramridge '
                'computes it from an updated Gram matrix (max |R^2 difference| to sklearn 8.6e-13)',
}
# Utility metric per task (Comet maximizes it on the clean validation split).
DEFAULT_METRIC = {'classification': 'macro_f1', 'regression': 'r2', 'clustering': 'silhouette'}
# Missing-value tokens of the benchmark tables.
MISSING_TOKENS = frozenset(['', 'empty', 'Empty', 'EMPTY', 'nan', 'NaN', 'NULL', 'null', 'None', '__NULL__'])
# Tokens that encode_subset replaces by NaN before encoding.
ENCODER_NA = ['empty', 'Empty', 'EMPTY', 'nan', 'NaN', 'NULL', 'null']


def parse_args():
    ap = argparse.ArgumentParser(description='Run Comet (EDBT 2025) on one dataset.')
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--method_name', default='comet')
    ap.add_argument('--cost_budget', type=float, default=None,
                    help='budget in cost units (paper default 50 when no --gt_budget is given)')
    ap.add_argument('--gt_budget', type=int, default=None, help='budget in ground-truth cells')
    ap.add_argument('--cost_model', default='constant', choices=['constant', 'multi_error'])
    ap.add_argument('--step_fraction', type=float, default=0.01)
    ap.add_argument('--model', default='auto',
                    help='auto (downstream model of the main table) or rf/lr/svm/knn/dt/gb, ridge/linreg (regression), '
                         'agglo (clustering)')
    ap.add_argument('--metric', default='auto', choices=['auto', 'macro_f1', 'accuracy', 'r2', 'silhouette'])
    ap.add_argument('--labels', default='clean', choices=['clean', 'dirty'],
                    help='training labels; clean = the label the common evaluator attaches to every training row')
    ap.add_argument('--cluster_sample', type=int, default=3000,
                    help='clustering only: training rows Comet works on and validation rows it scores, '
                         'each a fixed random sample (seed 42) to keep Agglomerative clustering tractable')
    ap.add_argument('--n_clusters', type=int, default=None,
                    help='clustering only: fixed number of clusters (har: 3, the distinct gt strings sit/stand/empty '
                         'of the clean training split, the k of the scorer); the label column is then '
                         'not read at all, the fit uses every sampled training row and the silhouette every sampled '
                         'validation row.  Default: number of encoded labels, rows without a label left out')
    ap.add_argument('--dry_run', action='store_true', help='print the dirty-cell statistics and one timed fit, then exit')
    ap.add_argument('--no_select_positives', action='store_true',
                    help="rank all pairs as in the authors' code instead of the paper's step (A)")
    ap.add_argument('--n_jobs', type=int, default=1)
    ap.add_argument('--project_root', default=REPO_DIR,
                    help='root of this repository (data/, configs/, scripts/, evaluation/)')
    ap.add_argument('--output_root', default=os.path.join(REPO_DIR, 'results', 'runs', 'baselines'))
    ap.add_argument('--records_path', default=None, help='optional JSONL file that receives the run summary')
    ap.add_argument('--tag', default='')
    return ap.parse_args()


class AggloNearestCentroid(object):
    """Agglomerative clustering (Ward, the sklearn default linkage) fitted on the training rows.

    predict assigns rows to the nearest cluster centroid of the training fit, so that a
    clustering learned from (possibly dirty) training rows can be scored on the clean
    validation rows.  y is ignored in fit.
    """

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        from sklearn.cluster import AgglomerativeClustering
        lab = AgglomerativeClustering(n_clusters=self.n_clusters).fit_predict(X)
        self.centroids_ = np.vstack([X[lab == k].mean(axis=0) for k in range(self.n_clusters)])
        return self

    def predict(self, X):
        d = ((X[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
        return d.argmin(axis=1)


class GramRidge(object):
    """Ridge regression (alpha=1, intercept) from an updated Gram matrix.

    Computes the same estimator as sklearn's Ridge(alpha=1.0) with its cholesky solver:
    coef = (Xc^T Xc + alpha I)^-1 Xc^T yc on the column-centered matrix.  Comet refits the
    model on matrices that differ from the previous fit in one or two columns, so the
    uncentered Gram matrix X^T X of the previous fit (kept per process) is updated in the
    changed columns only, which is exact up to floating-point rounding.  Used on mercedes
    (376 features, 16161 fits per Comet iteration).
    """

    _last = {}

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        from scipy import linalg
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        last = GramRidge._last
        if last.get('shape') == X.shape:
            changed = np.flatnonzero((X != last['X']).any(axis=0))
        else:
            changed = None
        if changed is None or len(changed) > d // 4:
            S = X.T.dot(X)
        else:
            S = last['S'].copy()
            if len(changed):
                block = X.T.dot(X[:, changed])
                S[:, changed] = block
                S[changed, :] = block.T
        # score_matrix passes a fresh array to every fit and never writes to it afterwards
        GramRidge._last = {'shape': X.shape, 'X': X, 'S': S}
        mu = X.mean(axis=0)
        ybar = y.mean()
        A = S - n * np.outer(mu, mu)
        A.flat[::d + 1] += self.alpha
        b = X.T.dot(y) - n * mu * ybar
        self.coef_ = linalg.solve(A, b, assume_a='pos', overwrite_a=True)
        self.intercept_ = ybar - mu.dot(self.coef_)
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float).dot(self.coef_) + self.intercept_


def model_factory(name, task='classification', n_clusters=None):
    from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                                  RandomForestClassifier, RandomForestRegressor)
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    if task == 'clustering':
        if name != 'agglo':
            raise SystemExit('clustering supports --model agglo only')
        return functools.partial(AggloNearestCentroid, n_clusters=n_clusters)
    if task == 'regression':
        # hyperparameters of the scorer (evaluation/paper_score.py:make_model)
        table = {
            'rf': functools.partial(RandomForestRegressor, n_estimators=100, random_state=42, n_jobs=1),
            'ridge': functools.partial(Ridge, random_state=42),
            'gramridge': functools.partial(GramRidge, alpha=1.0),  # same estimator as 'ridge'
            'linreg': functools.partial(LinearRegression),
            'gb': functools.partial(GradientBoostingRegressor, random_state=42),
        }
        return table[name]
    # hyperparameters of the scorer (evaluation/paper_score.py:make_model)
    table = {
        'rf': functools.partial(RandomForestClassifier, n_estimators=100, random_state=42, n_jobs=1),
        'lr': functools.partial(LogisticRegression, max_iter=1000, random_state=42),
        'svm': functools.partial(SVC, random_state=42),
        'knn': functools.partial(KNeighborsClassifier),
        'dt': functools.partial(DecisionTreeClassifier, random_state=42),
        'gb': functools.partial(GradientBoostingClassifier, random_state=42),
    }
    return table[name]


class SilhouetteOn(object):
    """Silhouette of predicted cluster labels on a fixed matrix (picklable score_fn)."""

    def __init__(self, X):
        self.X = X

    def __call__(self, y_true, y_pred):
        from sklearn.metrics import silhouette_score
        if len(np.unique(y_pred)) < 2:
            return -1.0
        return float(silhouette_score(self.X, y_pred))


def metric_fn(name, X_val=None):
    from sklearn.metrics import accuracy_score, f1_score, r2_score
    if name == 'macro_f1':
        return functools.partial(f1_score, average='macro', zero_division=0)
    if name == 'r2':
        return r2_score
    if name == 'silhouette':
        return SilhouetteOn(X_val)
    return accuracy_score


def make_report(task, X_val=None):
    from sklearn.metrics import accuracy_score, f1_score, r2_score
    if task == 'regression':
        return lambda y_true, y_pred: {'r2': float(r2_score(y_true, y_pred))}
    if task == 'clustering':
        sil = SilhouetteOn(X_val)
        return lambda y_true, y_pred: {'silhouette': sil(y_true, y_pred),
                                       'n_clusters_used': int(len(np.unique(y_pred)))}
    return lambda y_true, y_pred: {'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
                                   'accuracy': float(accuracy_score(y_true, y_pred))}


def read_csv(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip().strip('﻿') for c in df.columns]
    return df


def norm_cell(value):
    """Cell normalization for dirty-cell detection (integers without a decimal point, missing tokens as None)."""
    s = str(value).strip()
    if s in MISSING_TOKENS:
        return None
    try:
        f = float(s)
    except ValueError:
        return s
    if f != f:
        return None
    if f in (float('inf'), float('-inf')):
        return s
    return str(int(f)) if f.is_integer() else str(f)


def encode_raw(df, feature_cols, label_encoders, categorical_cols, find_nearest):
    """encode_subset's cell encoding without imputation and scaling (NaN kept)."""
    df = df.copy()
    df.columns = [c.strip().strip('﻿') for c in df.columns]
    df.replace(ENCODER_NA, np.nan, inplace=True)
    X = np.full((len(df), len(feature_cols)), np.nan)
    for j, col in enumerate(feature_cols):
        s = df[col]
        if col in categorical_cols and col in label_encoders:
            le = label_encoders[col]
            classes = list(le.classes_)
            pos = {v: i for i, v in enumerate(classes)}
            cache = {}

            def enc(v):
                if v in cache:
                    return cache[v]
                if v in pos:
                    out = float(pos[v])
                else:
                    near = find_nearest(v, classes, threshold=0.3) if find_nearest is not None else None
                    out = float(pos[near]) if near is not None else np.nan
                cache[v] = out
                return out

            mask = s.isna().values
            vals = s[~mask].astype(str).map(enc).values
            X[~mask, j] = vals
        else:
            X[:, j] = pd.to_numeric(s, errors='coerce').values.astype(float)
    return X


def deviations(task, args, model_name, n_comet_rows, n_train):
    """Deviations of this run from the paper's setting (recorded in every summary)."""
    downstream = FIRST_MODEL.get(args.dataset, 'rf')
    dev = ['split, encoder and validation set given by the benchmark (evaluation/paper_score.split_idx, '
           'preprocess_data fitted on the dirty training split), not the paper\'s own split',
           'utility model %s with the common evaluator\'s hyperparameters (downstream model of the main table: '
           '%s), not the paper\'s model list' % (model_name, downstream),
           'error types of the real dirty cells are assigned by classify_errors (missing, scaling by 10^k, '
           'Gaussian noise otherwise, categorical shift), the paper injects known error types',
           'Estimator posterior in closed form instead of NUTS sampling (same Gaussian posterior)']
    if args.labels == 'clean' and not (task == 'clustering' and getattr(args, 'n_clusters', None)):
        dev.append('training labels are the clean labels, as the common evaluator uses them')
    if args.gt_budget is not None:
        dev.append('budget in ground-truth cells (%d) instead of 50 cost units; the last step is truncated '
                   'to the remaining cells' % args.gt_budget)
    if model_name != downstream and not (model_name == 'gramridge' and downstream == 'ridge'):
        dev.append('utility model %s instead of the downstream model %s: %s'
                   % (model_name, downstream, MODEL_SUBSTITUTION.get(args.dataset, 'set by --model')))
    if args.step_fraction != 0.01:
        dev.append('cleaning and pollution step %.4g of the training rows instead of the paper\'s 0.01, so that '
                   'the ground-truth budget is spent in about 50 steps' % args.step_fraction)
    if task == 'regression':
        dev.append('regression extension: utility R^2 of the regressor on the clean validation split '
                   '(paper: macro F1 of a classifier), targets kept real-valued')
    if task == 'clustering' and getattr(args, 'n_clusters', None):
        dev.append('clustering extension: utility = silhouette on %d clean validation rows of the labels given '
                   'by the nearest centroid of an Agglomerative (Ward) clustering fitted on the training rows, '
                   'number of clusters fixed to %d, the k of the scorer (distinct gt strings of the '
                   'clean training split); the label column is not read'
                   % (args.cluster_sample, args.n_clusters))
    elif task == 'clustering':
        dev.append('clustering extension: utility = silhouette on %d clean validation rows of the labels given '
                   'by the nearest centroid of an Agglomerative (Ward) clustering fitted on the training rows, '
                   'number of clusters = number of activity labels (clustering protocol of the scorer, '
                   'which fits Ward on 15000 rows)' % args.cluster_sample)
    if task == 'clustering' and n_comet_rows < n_train:
        dev.append('Comet works on a fixed random sample of %d of the %d training rows (seed 42); steps are 1%% '
                   'of that sample, cells outside it stay dirty' % (n_comet_rows, n_train))
    return dev


def main():
    args = parse_args()
    t0 = time.time()
    root = os.path.abspath(os.path.expanduser(args.project_root))
    sys.path.insert(0, root)
    sys.path.insert(0, BENCH_DIR)
    from protocol import load_protocol
    DATASETS, preprocess_data, encode_subset, _find_nearest_known, split_idx = load_protocol(root, args.dataset)
    from Methods.Comet import Comet, classify_errors, prepare_matrix, ERROR_NAMES

    ds = args.dataset
    cfg = DATASETS[ds]
    task = cfg.get('task_type')
    # The paper evaluates classification only (Sec. 4.4).  Regression and clustering use the
    # same loop with another utility (R^2, silhouette); see the deviations in the summary.
    metric = DEFAULT_METRIC[task] if args.metric == 'auto' else args.metric
    if args.gt_budget is None and args.cost_budget is None:
        args.cost_budget = 50.0
    label = cfg['label_col']
    ddir = os.path.join(root, 'data', ds)
    dirty = read_csv(os.path.join(ddir, 'dirty_index.csv'))
    clean = read_csv(os.path.join(ddir, 'clean_index.csv'))
    if len(dirty) != len(clean) or not (dirty['index'].values == clean['index'].values).all():
        raise SystemExit('dirty and clean tables are not row-aligned on index')
    tr, va, te = split_idx(len(clean))
    n_test = int(len(te))
    dirty_tr = dirty.iloc[tr].reset_index(drop=True)
    clean_tr = clean.iloc[tr].reset_index(drop=True)
    clean_va = clean.iloc[va].reset_index(drop=True)
    del dirty, clean, te  # the 20% test rows are not used beyond this point

    out = preprocess_data(ds, dirty_df=dirty_tr.copy(), clean_df=None)
    feature_cols, scaler, label_encoders, categorical_cols = out[4], out[10], out[11], out[12]
    is_cat = np.array([c in categorical_cols for c in feature_cols])

    X_dirty = encode_raw(dirty_tr, feature_cols, label_encoders, categorical_cols, _find_nearest_known)
    X_clean = encode_raw(clean_tr, feature_cols, label_encoders, categorical_cols, _find_nearest_known)
    fit_df = dirty_tr.copy()
    if args.labels == 'clean':
        # the scorer attaches the clean label to every training row
        fit_df[label] = clean_tr[label].values
    n_label_diff = int((dirty_tr[label].map(norm_cell) != clean_tr[label].map(norm_cell)).sum())
    ref_dirty, y_train = encode_subset(fit_df, feature_cols, label, label_encoders, scaler, categorical_cols)
    ref_clean, _ = encode_subset(clean_tr.copy(), feature_cols, label, label_encoders, scaler, categorical_cols)
    X_val, y_val = encode_subset(clean_va.copy(), feature_cols, label, label_encoders, scaler, categorical_cols)
    center, scale = scaler.mean_.astype(float), scaler.scale_.astype(float)
    for name, mine, ref in (('dirty', X_dirty, ref_dirty), ('clean', X_clean, ref_clean)):
        if not np.allclose(prepare_matrix(mine, center, scale), ref, rtol=1e-9, atol=1e-9, equal_nan=False):
            raise SystemExit('internal encoding of the %s training split differs from encode_subset' % name)
    # as the scorer: drop unlabeled rows, then fill missing cells with the column mean
    # (with a fixed --n_clusters the clustering utility does not read labels, every validation row is kept)
    ok = ~np.isnan(y_val)
    if task == 'clustering' and args.n_clusters:
        ok = np.ones(len(y_val), dtype=bool)
        y_val = np.zeros(len(y_val))
    X_val = X_val[ok]
    y_val = y_val[ok] if task == 'regression' else y_val[ok].astype(int)
    col_m = np.nanmean(np.where(np.isnan(X_val), np.nan, X_val), axis=0) if np.isnan(X_val).any() else None
    if col_m is not None:
        r, c = np.nonzero(np.isnan(X_val))
        X_val[r, c] = np.where(np.isnan(col_m[c]), 0.0, col_m[c])

    dirty_mask = np.zeros((len(dirty_tr), len(feature_cols)), dtype=bool)
    for j, col in enumerate(feature_cols):
        dn = dirty_tr[col].map(norm_cell)
        cn = clean_tr[col].map(norm_cell)
        dirty_mask[:, j] = (dn != cn).values & ~(dn.isna() & cn.isna()).values
    codes = classify_errors(dirty_mask, X_dirty, X_clean, is_cat)
    by_type, by_pair = {}, {}
    for j, col in enumerate(feature_cols):
        for code in np.unique(codes[:, j]):
            if code == 0:
                continue
            k = int((codes[:, j] == code).sum())
            by_type[ERROR_NAMES[int(code)]] = by_type.get(ERROR_NAMES[int(code)], 0) + k
            by_pair['%s/%s' % (col, ERROR_NAMES[int(code)])] = k

    model_name = FIRST_MODEL.get(ds, 'rf') if args.model == 'auto' else args.model
    n_train_full = len(dirty_tr)
    comet_rows = np.arange(n_train_full)
    n_clusters = None
    if task == 'clustering':
        # Comet's loop fits the utility model hundreds of times per iteration; Ward clustering is
        # quadratic in the rows, so Comet works on a fixed sample of the training rows and scores
        # the silhouette on a fixed sample of the clean validation rows
        n_clusters = int(len(np.unique(y_train[~np.isnan(y_train)])))
        if args.n_clusters:
            # the Ward fit uses every training row Comet works on, the labels are not read
            n_clusters = int(args.n_clusters)
            y_train = np.zeros(len(y_train))
        if args.cluster_sample and args.cluster_sample < n_train_full:
            comet_rows = np.sort(np.random.RandomState(42).choice(n_train_full, args.cluster_sample, replace=False))
        if args.cluster_sample and args.cluster_sample < len(X_val):
            vs = np.sort(np.random.RandomState(42).choice(len(X_val), args.cluster_sample, replace=False))
            X_val, y_val = X_val[vs], y_val[vs]
        # as in the clustering protocol of the scorer: k = number of activity labels, rows without a
        # label are left out of the fit; the clustering itself ignores the label values
    score_fn = metric_fn(metric, X_val)
    report = make_report(task, X_val)
    X_dirty_c, X_clean_c = X_dirty[comet_rows], X_clean[comet_rows]
    y_train_c, codes_c = y_train[comet_rows], codes[comet_rows]

    def log(msg):
        print('[%s] %s' % (datetime.now().strftime('%H:%M:%S'), msg), flush=True)

    log('dataset=%s task=%s method=%s n_train=%d comet_rows=%d n_val=%d features=%d dirty_feature_cells=%d '
        'label_diff=%d labels=%s model=%s metric=%s cost_budget=%s gt_budget=%s'
        % (ds, task, args.method_name, n_train_full, len(comet_rows), len(y_val), len(feature_cols),
           int(dirty_mask.sum()), n_label_diff, args.labels, model_name, metric, args.cost_budget, args.gt_budget))
    log('dirty cells by pair: %s' % json.dumps(by_pair))
    if args.dry_run:
        from Methods.Comet.comet import score_matrix
        y_fit = y_train_c.astype(float if task == 'regression' else int)
        fr = np.flatnonzero(~np.isnan(y_train_c))
        ctx = {'model_factory': model_factory(model_name, task, n_clusters), 'score_fn': score_fn,
               'center': center, 'scale': scale, 'fit_rows': fr, 'y_fit': y_fit[fr], 'X_val': X_val,
               'y_val': y_val}
        t1 = time.time()
        sc = score_matrix(X_dirty_c, ctx)[0]
        dt = time.time() - t1
        n_pairs = len(by_pair) if task != 'clustering' else len(
            set((j, int(c)) for j in range(codes_c.shape[1]) for c in np.unique(codes_c[:, j]) if c))
        log('dry_run: dirty %s %.4f  one fit+score %.2fs  pairs=%d  fits/iteration=%d  est %.0fs/iteration  '
            'step_cells=%d' % (metric, sc, dt, n_pairs, n_pairs * 20 + 1, dt * (n_pairs * 20 + 2),
                               int(np.ceil(round(len(comet_rows) * args.step_fraction, 9)))))
        return

    comet = Comet(model_factory(model_name, task, n_clusters), score_fn, report_fn=report,
                  step_fraction=args.step_fraction,
                  cost_budget=args.cost_budget, gt_budget=args.gt_budget, cost_model=args.cost_model,
                  select_positives=not args.no_select_positives, n_jobs=args.n_jobs, log=log,
                  target_dtype=float if task == 'regression' else int)
    res = comet.run(X_dirty_c, X_clean_c, y_train_c, X_val, y_val, codes_c, is_cat, center, scale,
                    feature_names=feature_cols)
    # map Comet's row positions back to the training split
    res.cleaned_cells = [(int(comet_rows[r]), j) for r, j in res.cleaned_cells]
    res.consumed_cells = [(int(comet_rows[r]), j) for r, j in res.consumed_cells]

    # cleaned training split: dirty strings with the repaired cells replaced by ground truth
    out_df = dirty_tr.copy()
    for r, j in res.cleaned_cells:
        col = feature_cols[j]
        out_df.at[r, col] = clean_tr.at[r, col]
    out_df['_pos'] = pd.to_numeric(out_df['index'], errors='coerce')
    out_df = out_df.sort_values('_pos', kind='mergesort').drop(columns=['_pos'])
    tag = '_'.join(p for p in (ds, args.method_name, args.tag) if p)
    out_dir = os.path.join(os.path.expanduser(args.output_root), args.method_name, tag)
    os.makedirs(out_dir, exist_ok=True)
    cleaned_path = os.path.join(out_dir, tag + '_cleaned.csv')
    out_df.to_csv(cleaned_path, index=False)

    with open(os.path.join(out_dir, tag + '_trace.jsonl'), 'w') as f:
        for row in res.trace:
            f.write(json.dumps(row, default=float) + '\n')
    cleaned_by_pair = {}
    for s in res.steps:
        cleaned_by_pair[s['pair']] = cleaned_by_pair.get(s['pair'], 0) + s['cells']
    import sklearn
    summary = {
        'dataset': ds, 'method': args.method_name, 'paper': 'Comet, Mohammed/Naumann/Harmouch, EDBT 2025',
        'budget_mode': 'gt_cells' if args.gt_budget is not None else 'cost_units',
        'cost_budget': args.cost_budget, 'gt_budget': args.gt_budget, 'cost_model': args.cost_model,
        'n_train': int(len(dirty_tr)), 'n_val': int(len(y_val)), 'n_test': n_test,
        'n_features': len(feature_cols), 'model': model_name, 'metric': metric, 'task': task,
        'labels': args.labels, 'label_diff_train': n_label_diff, 'comet_rows': int(len(comet_rows)),
        'n_clusters': n_clusters, 'deviations': deviations(task, args, model_name, len(comet_rows), n_train_full),
        'step_fraction': args.step_fraction,
        'step_cells': int(np.ceil(round(len(comet_rows) * args.step_fraction, 9))),
        'select_positives': not args.no_select_positives,
        'dirty_feature_cells_train': int(dirty_mask.sum()), 'dirty_cells_by_type': by_type,
        'dirty_cells_by_pair': by_pair,
        'gt_cells': len(res.consumed_cells), 'gt_cells_applied': len(res.cleaned_cells),
        'gt_cells_reverted_never_applied': len(set(res.consumed_cells) - set(res.cleaned_cells)),
        'cleaned_cells_by_pair': cleaned_by_pair, 'cost_used': res.cost_used,
        'iterations': res.iterations, 'attempts': res.attempts, 'applied_steps': len(res.steps),
        'step_modes': {m: sum(1 for s in res.steps if s['mode'] == m) for m in set(s['mode'] for s in res.steps)},
        'stop_reason': res.stop_reason,
        'val_score_initial': res.initial_score, 'val_score_final': res.final_score,
        'val_report_final': res.final_report,
        'secs_comet': round(res.secs, 2), 'secs': round(time.time() - t0, 2),
        'cleaned_path': cleaned_path, 'steps': res.steps,
        'command': ' '.join([os.path.basename(sys.executable)] + sys.argv),
        'host': socket.gethostname(), 'python': platform.python_version(),
        'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__,
        'finished': datetime.now().isoformat(timespec='seconds'),
    }
    with open(os.path.join(out_dir, tag + '_summary.json'), 'w') as f:
        json.dump(summary, f, indent=1, default=float)
    if args.records_path:
        rec = {k: v for k, v in summary.items() if k != 'steps'}
        os.makedirs(os.path.dirname(os.path.abspath(os.path.expanduser(args.records_path))), exist_ok=True)
        with open(os.path.expanduser(args.records_path), 'a') as f:
            f.write(json.dumps(rec, default=float) + '\n')
    log('done: gt_cells=%d applied=%d cost_used=%s iterations=%d val %s %.4f -> %.4f  %.1fs  -> %s'
        % (summary['gt_cells'], summary['gt_cells_applied'], res.cost_used, res.iterations, metric,
           res.initial_score, res.final_score, summary['secs'], cleaned_path))


if __name__ == '__main__':
    main()
