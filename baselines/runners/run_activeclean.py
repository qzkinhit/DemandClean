"""Runner of the ActiveClean baseline under the 60/20/20 protocol of the DemandClean runs.

    python baselines/runners/run_activeclean.py --dataset beers --data_root data --budget dcbudget --results_root results/runs/baselines --records_dir results/runs/baselines/records

This script does argument parsing and IO only. The algorithm lives in
Methods/ActiveClean/activeclean_core.py.

Protocol
  * 60/20/20 split of the rows of clean_index.csv, identical to the split of
    evaluation/paper_score.py (train_test_split, seed 42).
  * ActiveClean sees the dirty 60% train rows, may query the clean version of a train
    record through a counting oracle, and uses the clean 20% validation rows for model
    selection. The label column of the test rows is dropped right after loading, and the
    selected model predicts the test rows only after the cleaning loop has finished.
  * Labels: the scorer takes the train labels from the clean table, and DemandClean never
    spends ground truth on the target. With --label_source clean_protocol (default)
    ActiveClean gets the same clean train labels, the cleaner repairs feature cells only, and
    one cleaned record costs len(feature_cols) cells. With --label_source dirty the dirty
    labels are kept, the cleaner also repairs the label, and one record costs
    len(feature_cols) + 1 cells.
  * --budget dcbudget : rows = floor(GT cells of the DemandClean run of the main table / cells
                       per record) (Figure 3 and the DemandClean budget of the appendix).
    --budget default : rows = 500, batch size 50 (the default budget of Table tab:budget).
    Batch size: 50 for the default budget and for any budget of at least 500 rows, otherwise
    ceil(rows / 10) (ten updates, the budget-to-batch ratio of the default).
    When the train split has fewer rows than the budget, every train record is cleaned.
  * har (clustering): ActiveClean needs a convex supervised loss, so the activity label gt is
    used as a supervised surrogate for the loss, the sampler and the detector. The labels are
    the dirty ones (no ground truth is read outside the oracle), a cleaned record also gets its
    clean label, and one record costs len(feature_cols) + 1 cells. The scorer clusters the
    exported cleaned train table.
  * flights: the four clock-time columns are featurised as minutes after midnight when they are
    spelled "hh:mm AM/PM" and as missing otherwise (ActiveClean's featuriser F(r) only).

Outputs
  <results_root>/<method>/<ds>_<method>/<ds>_<method>_cleaned.csv
      all train rows, original columns, cleaned records replaced by their clean rows
  <results_root>/<method>/<ds>_<method>/<ds>_<method>_run.json
      configuration, per-iteration history, cleaned positions and statistics
  <records_dir>/<method>_<ds>_{test,val}_{pred,index}.npy
      predictions of ActiveClean's own model and the matching values of the index column
  <records_dir>/parts/<method>_<ds>.json
      one record of the run
"""
import argparse
import json
import math
import os
import platform
import socket
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_BENCH_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _BENCH_ROOT)

from Methods.ActiveClean.activeclean_core import ActiveCleanConfig, run_activeclean  # noqa: E402

# Label, task and categorical columns of configs/datasets/<dataset>.yaml
DATASETS = {
    'adult': ('income', 'classification',
              {'workclass', 'education', 'marital_status', 'occupation', 'relationship',
               'race', 'gender', 'native_country'}),
    'beers': ('style', 'classification', {'beer_name', 'brewery_name', 'city', 'state'}),
    'breast_cancer': ('class', 'classification', set()),
    'smartfactory': ('labels', 'classification', set()),
    'bike': ('cnt', 'regression', set()),
    'mercedes': ('y', 'regression', {'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8'}),
    'nasa': ('sound_pressure_level', 'regression', set()),
    'soilmoisture': ('soil_moisture', 'regression', set()),
    'har': ('gt', 'clustering', set()),
    # organic datasets
    'hospitals': ('Condition', 'classification',
                  {'ProviderNumber', 'HospitalName', 'Address1', 'Address2', 'Address3', 'City',
                   'State', 'ZipCode', 'CountyName', 'PhoneNumber', 'HospitalType',
                   'HospitalOwner', 'EmergencyService', 'MeasureCode', 'MeasureName', 'Score',
                   'Sample', 'Stateavg'}),
    'flights': ('arrival_delay_bucket', 'classification',
                {'src', 'flight', 'sched_dep_time', 'act_dep_time', 'sched_arr_time',
                 'act_arr_time'}),
    'soccer': ('manager', 'classification',
               {'name', 'surname', 'birthplace', 'position', 'team', 'city', 'stadium'}),
}
# clustering datasets: label column used as the supervised surrogate
SURROGATE_LABEL = {'har': 'gt'}
# columns featurised as clock times (minutes after midnight)
CLOCK_COLS = {'flights': ('sched_dep_time', 'act_dep_time', 'sched_arr_time', 'act_arr_time')}
# GT cells (calibration seed set and repairs) that the DemandClean run of the main table spends
# on the train split. The main table gives the GT-consuming baselines this budget.
DC_GT_CELLS_MAIN = {'adult': 60, 'beers': 248, 'breast_cancer': 25, 'smartfactory': 311,
                    'bike': 2129, 'mercedes': 25231, 'nasa': 484, 'soilmoisture': 140, 'har': 7134}
SEED_SPLIT = 42


GENERIC_DEVIATIONS = [
    'reimplementation of the paper (the authors\' Python 2 code is tied to IMDB text data); '
    'gradient-norm importance sampling mixed with a 0.1 uniform floor',
    'featuriser: numeric robust z-score (median, IQR/1.349, clip +-10) with missing indicators, '
    'categorical one-hot over at most 100 frequent dirty-train categories, fit on dirty train only',
    'step-size constant {0.1,0.3,1,3}, dirty-model regularisation and stopping iterate chosen on the '
    'clean validation rows instead of a fixed learning rate',
    'record-level cost: every feature cell of a record sent to the oracle is charged',
    'single seed 42',
]


def split_idx(n):
    """Same as evaluation/paper_score.py:split_idx."""
    idx = np.arange(n)
    tr, tmp = train_test_split(idx, test_size=0.4, random_state=SEED_SPLIT)
    va, te = train_test_split(tmp, test_size=0.5, random_state=SEED_SPLIT)
    return tr, va, te


def read_table(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip().strip('﻿') for c in df.columns]
    return df


class CountingOracle(object):
    """Returns clean train records by position and counts every distinct record served."""

    def __init__(self, clean_train, columns):
        self._clean = clean_train
        self._cols = list(columns)
        self.served = []
        self._seen = set()
        self.calls = 0

    def __call__(self, positions):
        self.calls += 1
        for p in positions:
            if p in self._seen:
                raise RuntimeError('oracle asked twice for record %d' % p)
            self._seen.add(p)
            self.served.append(p)
        return self._clean.iloc[list(positions)][self._cols].reset_index(drop=True)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def main():
    ap = argparse.ArgumentParser(description='Budget-matched ActiveClean run on one dataset.')
    ap.add_argument('--dataset', required=True, choices=sorted(DATASETS))
    ap.add_argument('--data_root', required=True, help='directory with <ds>/{dirty,clean}_index.csv')
    ap.add_argument('--budget', required=True, choices=['dcbudget', 'default'])
    ap.add_argument('--results_root', required=True, help='output directory of the cleaned tables, e.g. results/runs/baselines')
    ap.add_argument('--records_dir', required=True, help='output directory of the run records and predictions')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--default_rows', type=int, default=500)
    ap.add_argument('--default_batch', type=int, default=50)
    ap.add_argument('--label_source', default='clean_protocol', choices=['clean_protocol', 'dirty'])
    ap.add_argument('--method_suffix', default='', help='appended to the method name')
    args = ap.parse_args()

    ds = args.dataset
    label_col, task, cat_cols = DATASETS[ds]
    method = 'activeclean_' + args.budget + args.method_suffix
    command = ' '.join([os.path.basename(sys.executable)] + sys.argv)
    eval_task = task
    deviations = list(GENERIC_DEVIATIONS)
    if task not in ('classification', 'regression'):
        if ds not in SURROGATE_LABEL:
            print('[run_activeclean] %s: task %s has no supervised convex loss and no surrogate '
                  'label; nothing written' % (ds, task))
            return 2
        label_col = SURROGATE_LABEL[ds]
        task = 'classification'
        if args.label_source != 'dirty':
            print('[run_activeclean] %s: surrogate labels are taken from the dirty table'
                  % ds)
        args.label_source = 'dirty'
        deviations.append(
            '%s is a clustering task; the activity label %s serves as the supervised surrogate for '
            'the convex loss (linear SVM, hinge), the importance sampler and the detector; the '
            'dirty labels are used and a cleaned record also receives its clean label, so one '
            'record costs len(feature_cols)+1 cells; validation selection uses surrogate accuracy '
            'on the clean validation rows; the evaluator clusters the exported cleaned train table'
            % (ds, label_col))
    clock_cols = CLOCK_COLS.get(ds, ())
    if clock_cols:
        deviations.append(
            'featuriser: %s parsed as minutes after midnight when spelled hh:mm AM/PM, any other '
            'spelling is missing (z = 0 plus a missing indicator); the benchmark schema lists '
            'them as categorical, and the exported table keeps the raw strings' % ', '.join(clock_cols))

    t_wall = time.time()
    dirty = read_table(os.path.join(args.data_root, ds, 'dirty_index.csv'))
    clean = read_table(os.path.join(args.data_root, ds, 'clean_index.csv'))
    if list(dirty.columns) != list(clean.columns) or len(dirty) != len(clean):
        raise RuntimeError('dirty/clean tables are not aligned')
    if not (dirty['index'].values == clean['index'].values).all():
        raise RuntimeError('index columns differ between dirty and clean')
    tr, va, te = split_idx(len(clean))
    # the test labels are never read: drop them before anything else
    feature_cols = [c for c in clean.columns if c not in ('index', 'id', label_col)]
    test_features = clean.iloc[te][['index'] + feature_cols].reset_index(drop=True)
    clean_train = clean.iloc[tr].reset_index(drop=True)
    val_clean = clean.iloc[va].reset_index(drop=True)
    dirty_train = dirty.iloc[tr].reset_index(drop=True)
    del clean, dirty
    dirty_train_raw = dirty_train.copy()
    n_label_fixed = 0
    if args.label_source == 'clean_protocol':
        n_label_fixed = int((dirty_train[label_col].values != clean_train[label_col].values).sum())
        dirty_train[label_col] = clean_train[label_col].values
        cells_per_row = len(feature_cols)
        deviations.append('train labels taken from the clean table as in the unified evaluator, '
                          'the cleaner repairs feature cells only')
    else:
        cells_per_row = len(feature_cols) + 1
    if args.budget == 'dcbudget':
        budget_cells = DC_GT_CELLS_MAIN[ds]
        rows = budget_cells // cells_per_row
    else:
        rows = args.default_rows
        budget_cells = rows * cells_per_row
    rows = min(rows, len(tr))
    if args.budget == 'default' or rows >= args.default_rows:
        batch = args.default_batch
    else:
        batch = max(1, int(math.ceil(rows / 10.0)))

    cfg = ActiveCleanConfig(feature_cols=feature_cols, categorical_cols=cat_cols,
                            label_col=label_col, task=task, budget_rows=rows,
                            batch_size=batch, seed=args.seed, clock_cols=clock_cols)
    oracle = CountingOracle(clean_train, feature_cols + [label_col])
    print('[run_activeclean] %s %s: train=%d val=%d test=%d cells/row=%d budget_cells=%d '
          'rows=%d batch=%d' % (ds, method, len(tr), len(va), len(te), cells_per_row,
                                budget_cells, rows, batch))
    sys.stdout.flush()

    t0 = time.time()
    res = run_activeclean(dirty_train, val_clean, oracle, cfg,
                          log=lambda m: (print(m), sys.stdout.flush()))
    secs = time.time() - t0
    if sorted(oracle.served) != sorted(res.cleaned_positions):
        raise RuntimeError('oracle log and cleaned positions disagree')

    # model predictions, made after the cleaning loop finished
    test_pred = res.predict(test_features)
    val_pred = res.predict(val_clean)
    os.makedirs(args.records_dir, exist_ok=True)
    os.makedirs(os.path.join(args.records_dir, 'parts'), exist_ok=True)
    stem = os.path.join(args.records_dir, '%s_%s' % (method, ds))
    if task == 'classification':
        test_pred = np.asarray(test_pred).astype(str)
        val_pred = np.asarray(val_pred).astype(str)
    else:
        test_pred = np.asarray(test_pred, dtype=float)
        val_pred = np.asarray(val_pred, dtype=float)
    np.save(stem + '_test_pred.npy', test_pred)
    np.save(stem + '_val_pred.npy', val_pred)
    np.save(stem + '_test_index.npy', pd.to_numeric(test_features['index']).values.astype(np.int64))
    np.save(stem + '_val_index.npy', pd.to_numeric(val_clean['index']).values.astype(np.int64))

    # cleaned train table: cleaned records replaced by their clean rows, others untouched
    cleaned_pos = np.array(res.cleaned_positions, dtype=int)
    out = dirty_train_raw.copy()
    if len(cleaned_pos):
        out.iloc[cleaned_pos] = clean_train.iloc[cleaned_pos].values
    cols_all = list(dirty_train.columns)
    attr_cols = feature_cols + [label_col]
    changed_raw = int((dirty_train_raw.iloc[cleaned_pos][attr_cols].values
                       != clean_train.iloc[cleaned_pos][attr_cols].values).sum()) if len(cleaned_pos) else 0
    res_dir = os.path.join(args.results_root, method, '%s_%s' % (ds, method))
    os.makedirs(res_dir, exist_ok=True)
    cleaned_csv = os.path.join(res_dir, '%s_%s_cleaned.csv' % (ds, method))
    out[cols_all].to_csv(cleaned_csv, index=False)

    gt_cells = len(oracle.served) * cells_per_row
    record = {
        'dataset': ds, 'method': method, 'status': 'ok',
        'n_train': int(len(tr)), 'n_val': int(len(va)), 'n_test': int(len(te)),
        'gt_cells': int(gt_cells), 'secs': round(secs, 3),
        'rows_cleaned': int(len(oracle.served)), 'cells_per_row': int(cells_per_row),
        'budget_cells': int(budget_cells), 'budget_rows': int(rows), 'batch_size': int(batch),
        'rows_cleaned_dirty': res.info['rows_cleaned_dirty'],
        'raw_cells_changed_in_cleaned_rows': changed_raw,
        'iterations': res.info['iterations'],
        'model': ('linear_svm_hinge' if task == 'classification' else 'linear_regression'),
        'model_regularisation': {k: v for k, v in res.info['dirty_model'].items()
                                 if k in ('C', 'alpha', 'lam')},
        'selected_step_constant': res.info['selected_step_constant'],
        'selected_iterate': res.info['selected_iterate'],
        'val_score_selected': res.info['val_score_selected'],
        'val_score_dirty_model': res.info['val_score_dirty_model'],
        'val_metric': (('surrogate_accuracy' if eval_task == 'clustering' else 'accuracy')
                       if task == 'classification' else 'r2'),
        'deviations': deviations,
        'detector_active_at_end': res.info['detector_active_at_end'],
        'seed': args.seed, 'task': task, 'eval_task': eval_task, 'label_col': label_col,
        'label_source': args.label_source,
        'train_labels_replaced_by_protocol': n_label_fixed,
        'pred_space': ('raw label strings' if task == 'classification'
                       else 'raw target scale'),
        'test_pred': stem + '_test_pred.npy', 'val_pred': stem + '_val_pred.npy',
        'test_index': stem + '_test_index.npy', 'val_index': stem + '_val_index.npy',
        'test_features': 'clean test split (label column dropped)',
        'cleaned_csv': cleaned_csv,
        'command': command, 'cwd': os.getcwd(), 'host': socket.gethostname(),
        'python': platform.python_version(),
        'finished': time.strftime('%Y-%m-%d %H:%M:%S'),
        'wall_secs': round(time.time() - t_wall, 3),
    }
    with open(os.path.join(args.records_dir, 'parts', '%s_%s.json' % (method, ds)), 'w') as f:
        json.dump(record, f, default=_json_default)
    run_log = {'record': record, 'config': cfg.as_dict(), 'info': res.info,
               'history': res.history,
               'cleaned_positions': [int(p) for p in res.cleaned_positions],
               'cleaned_index_values': [int(v) for v in
                                        pd.to_numeric(dirty_train['index']).values[cleaned_pos]],
               'cleaned_step': [int(s) for s in res.cleaned_step]}
    with open(os.path.join(res_dir, '%s_%s_run.json' % (ds, method)), 'w') as f:
        json.dump(run_log, f, default=_json_default, indent=1)
    print('[run_activeclean] done %s %s: rows=%d gt_cells=%d secs=%.1f selected c=%s t=%s '
          'val=%.4f (dirty model %.4f)'
          % (ds, method, len(oracle.served), gt_cells, secs, res.info['selected_step_constant'],
             res.info['selected_iterate'], res.info['val_score_selected'],
             res.info['val_score_dirty_model']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
