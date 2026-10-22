#!/usr/bin/env python
"""Learn2Clean V1 restricted to its cleaning operators, under the untouched-test rule.

The native Q-learning graph, traversal and operators of Learn2Clean V1 run with the
implementation fixes of baselines/Methods/Learn2Clean/learn2clean_v1.patch. Transitions into the normalization and
feature-selection states are removed from the reward matrix and traversals that start in those
states are discarded, so every candidate pipeline consists of imputation, outlier detection,
deduplication and consistency checking only. Each candidate is scored by the main-table model on
the clean validation split (clustering: training silhouette, as before). The best candidate's
cleaned training matrix, in the canonical encoding, is scored on the untouched test split.

    LEARN2CLEAN_SOURCE=<patched clone of LaureBerti/Learn2Clean> SCORER_PYTHON=<python with scikit-learn 1.3.2> \
        python baselines/runners/run_learn2clean/l2c_clean.py plan --dataset beers
    python baselines/runners/run_learn2clean/l2c_clean.py candidate --dataset beers --candidate 3
    python baselines/runners/run_learn2clean/l2c_clean.py finalize --dataset beers

l2c_clean_queue.py runs the three steps for every dataset.
"""
import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]  # repository root
sys.path.insert(0, str(ROOT / 'baselines' / 'runners'))
import common  # noqa: E402

CANONICAL = os.environ.get('SCORER_PYTHON', sys.executable)
RESULTS = ROOT / 'results/runs/learn2clean_clean'
SRC = Path(os.environ.get('LEARN2CLEAN_SOURCE', str(ROOT / 'baselines/Methods/Learn2Clean/Learn2Clean')))
UPSTREAM = dict(repository='https://github.com/LaureBerti/Learn2Clean',
                commit='169bac3236632d738dbe8b67e9373fd2bd19be16')
METHOD = 'Learn2Clean_V1_cleaning_operators'
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int
if 'bool' not in np.__dict__:
    np.bool = bool


def forbidden(check_missing):
    # check_missing: 0-3 imputation, 4-6 normalization, 7-10 feature selection, 11-13 outliers,
    # 14-15 consistency, 16-17 deduplication. Otherwise 0-2 normalization, 3-5 feature selection.
    return set(range(4, 11)) if check_missing else set(range(0, 6))


def build(ds, out):
    """Learner on dirty train (canonical encoding, missing values kept) and clean validation."""
    b = common.load_bundle(ds)
    sys.path.insert(0, str(SRC / 'Learn2Clean_V1/python-package'))
    from learn2clean.qlearning.qlearner import Qlearner
    from learn2clean.normalization.normalizer import Normalizer
    from learn2clean.imputation.imputer import Imputer
    from learn2clean.feature_selection.feature_selector import Feature_selector
    from learn2clean.outlier_detection.outlier_detector import Outlier_detector
    from learn2clean.duplicate_detection.duplicate_detector import Duplicate_detector
    from learn2clean.consistency_checking.consistency_checker import Consistency_checker
    from learn2clean.classification.classifier import Classifier
    from learn2clean.regression.regressor import Regressor
    from learn2clean.clustering.clusterer import Clusterer
    columns = ['feature_' + str(i) for i in range(len(b['features']))]
    Xt = pd.DataFrame(b['X_train_missing'], columns=columns, index=b['idx_train'])
    Xv = pd.DataFrame(b['X_val'], columns=columns, index=b['idx_val'])
    yt = pd.Series(b['y_train'], index=b['idx_train'], name='target')
    valid = np.isfinite(np.asarray(yt, float))
    Xt, yt = Xt.loc[valid], yt.loc[valid]
    data = dict(train=Xt, test=Xv, target=yt,
                target_test=pd.Series(b['y_val'], index=b['idx_val'], name='target'))
    state = dict(captured=None, score=None)

    def wrap_transform(cls):
        native = cls.transform

        def transform(obj):
            # Operators fit and act on the training rows only; validation rows are preserved.
            validation = obj.dataset.get('test', {})
            obj.dataset['test'] = {}
            result = native(obj)
            if isinstance(validation, dict):
                result['test'] = validation
                return result
            cols = [c for c in result['train'].columns if c in validation.columns]
            result['test'] = validation[cols].copy()
            return result
        cls.transform = transform
    for cls in [Normalizer, Imputer, Feature_selector, Outlier_detector, Duplicate_detector, Consistency_checker]:
        wrap_transform(cls)

    def terminal(obj):
        train, test = obj.dataset['train'], obj.dataset['test']
        target = obj.dataset['target'].reindex(train.index)
        if isinstance(test, dict):
            raise ValueError('missing evaluation features')
        if not train.columns.equals(test.columns):
            test = test.reindex(columns=train.columns)
        X, y, Xval = np.asarray(train, float), np.asarray(target, float), np.asarray(test, float)
        state['captured'] = dict(X=X, y=y, Xv=Xval, idx=np.asarray(train.index, dtype=np.int64),
                                 columns=list(train.columns))
        path = out / 'candidate_validation.npz'
        np.savez_compressed(path, X=X, y=y, Xv=Xval)
        p = subprocess.run([CANONICAL, str(ROOT / 'baselines/runners/validation_bridge.py'), ds, str(path)],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        if p.returncode:
            raise RuntimeError('canonical validation failed: ' + p.stdout[-2000:])
        line = [x for x in p.stdout.splitlines() if x.startswith('CANONICAL_VALIDATION ')][-1]
        state['score'] = float(line.split(' ', 1)[1])
        return {'quality_metric': -state['score'] if b['task'] == 'regression' else state['score']}
    Classifier.transform = terminal
    Regressor.transform = terminal
    Clusterer.transform = terminal
    goal = {'classification': 'NB', 'regression': 'OLS', 'clustering': 'HCA'}[b['task']]
    learner = Qlearner(dataset=data, goal=goal, target_goal='target', target_prepare=None,
                       verbose=False, file_name=ds)
    native_matrix = learner.Initialization_Reward_Matrix

    def restricted_matrix(dataset):
        q, r, n_actions, n_states, check_missing = native_matrix(dataset)
        r = r.copy()
        r[:, sorted(forbidden(check_missing))] = -1
        return q, r, n_actions, n_states, check_missing
    learner.Initialization_Reward_Matrix = restricted_matrix
    return b, data, learner, state, goal, columns


def plan(ds):
    out = RESULTS / ds
    (out / 'save').mkdir(parents=True, exist_ok=True)
    os.chdir(str(out))
    np.random.seed(42)
    random.seed(42)
    b, data, learner, state, goal, columns = build(ds, out)
    seen = []

    def recorder(dataset, actions_list, target_goal, target_prepare, check_missing):
        seen.append([int(a) for a in actions_list])
        return None, {'quality_metric': 0.}, 0.
    learner.pipeline = recorder
    learner.learn2clean()
    check = learner.Initialization_Reward_Matrix(data)[-1]
    bad = forbidden(check)
    cands = []
    for acts in seen:
        if any(a in bad for a in acts) or acts in cands:
            continue
        cands.append(acts)
    rec = dict(dataset=ds, method=METHOD, goal=goal, check_missing=bool(check),
               forbidden_actions=sorted(bad), traversals=seen,
               candidates=[dict(candidate=i + 1, actions=a) for i, a in enumerate(cands)],
               native_graph_seed=1999, primary_seed=42, created=time.time())
    (out / 'plan.json').write_text(json.dumps(rec, indent=2))
    print('PLAN', ds, len(seen), 'traversals', len(cands), 'candidates', flush=True)


def candidate(ds, cid):
    out = RESULTS / ds
    pl = json.loads((out / 'plan.json').read_text())
    acts = [c for c in pl['candidates'] if c['candidate'] == cid][0]['actions']
    cdir = out / 'candidates' / ('cand_%02d' % cid)
    (cdir / 'save').mkdir(parents=True, exist_ok=True)
    os.chdir(str(cdir))
    np.random.seed(42)
    random.seed(42)
    b, data, learner, state, goal, columns = build(ds, cdir)
    check = learner.Initialization_Reward_Matrix(data)[-1]
    if any(a in forbidden(check) for a in acts):
        raise ValueError('candidate holds a forbidden action')
    t0 = time.time()
    learner.pipeline(copy.deepcopy(data), acts, 'target', None, check)
    cap = state['captured']
    if cap is None or state['score'] is None or not np.isfinite(state['score']):
        raise RuntimeError('native pipeline did not reach the terminal model')
    if cap['columns'] != columns:
        raise ValueError('a cleaning-only pipeline changed the feature columns')
    np.savez_compressed(cdir / 'cleaned.npz', X=cap['X'], y=cap['y'], idx=cap['idx'])
    res = dict(dataset=ds, candidate=cid, actions=acts, status='completed', validation_score=state['score'],
               n_rows=int(len(cap['idx'])), seconds=time.time() - t0, test_accessed=False, gt_cells=0)
    (cdir / 'result.json').write_text(json.dumps(res, indent=2))
    print('CANDIDATE', json.dumps(res), flush=True)


def finalize(ds):
    out = RESULTS / ds
    pl = json.loads((out / 'plan.json').read_text())
    rows = []
    for c in pl['candidates']:
        f = out / 'candidates' / ('cand_%02d' % c['candidate']) / 'result.json'
        r = json.loads(f.read_text()) if f.exists() else dict(candidate=c['candidate'], status='missing')
        rows.append(r)
    good = [r for r in rows if r.get('status') == 'completed' and np.isfinite(r.get('validation_score', np.nan))]
    if not good:
        raise RuntimeError('no completed candidate')
    best = max(good, key=lambda r: r['validation_score'])  # first maximum in plan order
    frozen = dict(best, frozen_at=time.time(), candidates=[dict(candidate=r['candidate'], status=r.get('status'),
                  validation_score=r.get('validation_score'), actions=r.get('actions')) for r in rows])
    (out / 'frozen_plan.json').write_text(json.dumps(frozen, indent=2))
    sha = hashlib.sha256((out / 'frozen_plan.json').read_bytes()).hexdigest()
    z = np.load(out / 'candidates' / ('cand_%02d' % best['candidate']) / 'cleaned.npz')
    b = common.load_bundle(ds)
    meta = dict(upstream=dict(UPSTREAM), goal=pl['goal'],
                selected_actions=best['actions'], validation_score=best['validation_score'],
                frozen_plan_sha256=sha, forbidden_actions=pl['forbidden_actions'],
                candidate_count=len(rows), completed_candidates=len(good),
                adaptations=['official V1 Q-learning graph, greedy traversal and native operators with the implementation fixes of learn2clean_v1.patch',
                             'transitions into normalization and feature-selection states removed; traversals starting there discarded',
                             'operators fit and act on training rows only; validation and test rows untouched',
                             'selection on the clean validation split with the main-table model; test split never passed to Learn2Clean'],
                gt_cells=0, gt_cap_cells=b['gt_cap_cells'])
    common.write_cleaned_encoded(ds, out, z['X'], z['y'], z['idx'], METHOD, gt_cells=0, metadata=meta)
    p = subprocess.run([CANONICAL, str(ROOT / 'baselines/runners/common.py'), '--dataset', ds, '--out', str(out)],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    print(p.stdout, flush=True)
    if p.returncode:
        raise RuntimeError('canonical final scoring failed')
    (out / 'status.json').write_text(json.dumps(dict(dataset=ds, method=METHOD, status='completed',
                                                     finished=time.time()), indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['plan', 'candidate', 'finalize'])
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--candidate', type=int)
    a = ap.parse_args()
    try:
        {'plan': lambda: plan(a.dataset), 'candidate': lambda: candidate(a.dataset, a.candidate),
         'finalize': lambda: finalize(a.dataset)}[a.mode]()
    except Exception as e:
        traceback.print_exc()
        if a.mode == 'candidate':
            cdir = RESULTS / a.dataset / 'candidates' / ('cand_%02d' % a.candidate)
            cdir.mkdir(parents=True, exist_ok=True)
            (cdir / 'result.json').write_text(json.dumps(dict(dataset=a.dataset, candidate=a.candidate,
                                              status='failed', error=repr(e), test_accessed=False), indent=2))
        sys.exit(1)
