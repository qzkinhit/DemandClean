#!/usr/bin/env python
"""Fiber oscillation of the quality state (A, V) on the training split.

    python evaluation/fiber_oscillation.py --quick                 # beers + bike, 4 random policies
    python evaluation/fiber_oscillation.py --datasets beers,bike   # named datasets, full policy set
    python evaluation/fiber_oscillation.py                         # every dataset of configs/datasets/

The paper maps a cleaning policy to the two-dimensional quality state (A, V) (accuracy,
diversity) and bounds the planning error by the oscillation omega of the downstream
performance inside one (A, V) fiber. This script measures omega empirically: on every
dataset it builds many cleaning policies that share (almost) the same quality state and
records the spread of their downstream performance, then compares that spread with the
gain DemandClean achieves over NoFix on the same dataset.

Repair fibers (k in {0.25, 0.5, 0.75}): every policy repairs exactly round(k*|E|) of the
oracle candidate cells E (training-split cells whose dirty value differs from the clean
value in the canonical form of evaluation/table_redr.py, index column excluded, label
column included) by writing the clean value. Policies: 16 random subsets (seeds 42..57)
plus 4 structured variants -- column-concentrated / column-dispersed (columns ordered by
mean absolute correlation with the other columns of the encoded dirty training matrix),
row-concentrated (candidates packed into the fewest rows), row-dispersed (one candidate
per row before any row gets a second). Every policy repairs the same number of cells, so
A is identical inside a fiber; the per-column variance differs slightly, so the (A, V) of
every policy is recorded and the within-fiber spreads are reported.

Delete fibers (k in {0.25, 0.5}): every policy deletes exactly round(k*|E_rows|) of the
training rows that hold at least one candidate. Policies: 16 random subsets plus 2
structured variants -- rows whose removal changes the column covariance the most / the
least, approximated by the squared distance to the column means of the encoded dirty
training matrix. The number of candidate cells removed is not forced to be equal; the
actual A of every policy is recorded and the A spread of the fiber is reported.

Quality-state coordinates: A = 1 - (remaining erroneous cells)/(cells of the cleaned
training table); V = mean of the per-column variance of the encoded (scaled) training
matrix. eps_K = C*sqrt(log(4/delta)/K) with delta = 0.05 and C = 1 (the quality space is
the unit scale: A in [0, 1], V the variance of standardized columns) is reported per
fiber together with the observed spreads, so the reader can check that every policy of a
fiber lands inside the same eps_K grid cell.

References: k=0 (NoFix, dirty training split) and k=1 (RepairAll, clean training split)
are scored on the same path. The DemandClean gain over NoFix is read from
results/main_table/<ds>/demandclean/scores.json.

Scoring uses evaluation/paper_score.py: the TrainEncoder fitted on the dirty training
split, the clean test split as the only evaluation set, and for every dataset its
downstream_model and second_model. Clustering (har) fit_predicts on a <=10000-row
subsample of the cleaned training rows drawn with seed 42 (the same seed for every
policy, so all repair policies share the sampled row subset; delete policies draw the
subsample inside their kept rows) and reports the silhouette on that subsample.

Everything uses seed 42, nothing is tuned, and every constructed policy is reported.
Output: results/robustness/fiber_oscillation.json
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_score as P  # noqa: E402
from table_redr import canon, read_str  # noqa: E402

OUT = os.path.join(P._ROOT, 'results', 'robustness', 'fiber_oscillation.json')
K_REPAIR = [0.25, 0.5, 0.75]
K_DELETE = [0.25, 0.5]
N_SUB = 10000  # clustering subsample cap (paper_score uses the same for the silhouette)
DELTA = 0.05
C_GRID = 1.0  # quality space unit scale: A in [0, 1], V = variance of standardized columns


def candidates(ctx, root, ds):
    """Oracle candidate cells of the training split.

    Returns (cand, cols): cand is an (n, 2) int array of (row within the training split,
    position in cols) where the canonical dirty value differs from the canonical clean
    value; cols is every column except index (the label column included).
    """
    d = read_str(os.path.join(root, ds, 'dirty_index.csv'))
    c = read_str(os.path.join(root, ds, 'clean_index.csv'))
    cols = [x for x in d.columns if x != 'index']
    D = d[cols].apply(lambda s: s.map(canon))
    C = c[cols].apply(lambda s: s.map(canon))
    neq = (D != C).values
    return np.argwhere(neq[ctx['tr']]), cols


def column_priority(ctx, X0, cols):
    """Mean absolute correlation of every column with the others, on the encoded dirty
    training matrix. Columns the encoder drops (index/id/label) get priority 0."""
    prio = {c: 0.0 for c in cols}
    feats = ctx['enc'].features
    if X0.shape[1] > 1:
        with np.errstate(invalid='ignore'):
            corr = np.abs(np.corrcoef(X0, rowvar=False))
        corr = np.nan_to_num(corr)
        np.fill_diagonal(corr, 0.0)
        mean_corr = corr.sum(axis=1) / (corr.shape[1] - 1)
        for j, f in enumerate(feats):
            if f in prio:
                prio[f] = float(mean_corr[j])
    return np.array([prio[c] for c in cols])


def structured_repairs(cand, n_fix, col_pr_all):
    """The 4 structured repair policies, as index arrays into cand."""
    col_pr = col_pr_all[cand[:, 1]]
    col_conc = np.argsort(-col_pr, kind='stable')[:n_fix]
    col_disp = np.argsort(col_pr, kind='stable')[:n_fix]
    rows = cand[:, 0]
    uniq, counts = np.unique(rows, return_counts=True)
    cnt = dict(zip(uniq.tolist(), counts.tolist()))
    row_conc = np.array(sorted(range(len(cand)), key=lambda i: (-cnt[rows[i]], rows[i])))[:n_fix]
    rank = np.empty(len(cand), dtype=int)
    seen = {}
    for i in np.argsort(rows, kind='stable'):
        r = int(rows[i])
        rank[i] = seen.get(r, 0)
        seen[r] = rank[i] + 1
    row_disp = np.array(sorted(range(len(cand)), key=lambda i: (rank[i], rows[i])))[:n_fix]
    return [('col_concentrated', col_conc), ('col_dispersed', col_disp),
            ('row_concentrated', row_conc), ('row_dispersed', row_disp)]


def repair_policies(cand, n_fix, col_pr_all, n_random):
    """n_random random subsets (seeds 42+i) plus the 4 structured variants."""
    pol = []
    for i in range(n_random):
        rng = np.random.RandomState(42 + i)
        pol.append((f'random_{i}', rng.choice(len(cand), size=n_fix, replace=False)))
    return pol + structured_repairs(cand, n_fix, col_pr_all)


def delete_policies(cand_rows, n_del, impact, n_random):
    """n_random random row subsets (seeds 42+i) plus the 2 covariance-impact variants."""
    pol = []
    for i in range(n_random):
        rng = np.random.RandomState(42 + i)
        pol.append((f'random_{i}', np.sort(rng.choice(cand_rows, size=n_del, replace=False))))
    order = np.argsort(-impact)  # positions in cand_rows, highest removal impact first
    pol.append(('impact_high', np.sort(cand_rows[order[:n_del]])))
    pol.append(('impact_low', np.sort(cand_rows[order[::-1][:n_del]])))
    return pol


def apply_repair(dtr, ctr, cand, sel, cols):
    """Copy of the dirty training split with the selected candidate cells set to clean."""
    df = dtr.copy()
    sub = cand[sel]
    for cpos in np.unique(sub[:, 1]):
        rows = sub[sub[:, 1] == cpos, 0]
        col = cols[cpos]
        j = df.columns.get_loc(col)
        df[col] = df[col].astype(object)  # safe in-place value assignment below
        df.iloc[rows, j] = ctr.iloc[rows, j].values
    return df


def score_models(task, X, y, ctx, models, metric):
    """{model: performance} of one cleaned training table on the clean test split."""
    if task == 'clustering':
        n = len(X)
        m = min(n, N_SUB)
        sub = (np.random.RandomState(42).choice(n, size=m, replace=False)
               if m < n else np.arange(n))
        Xs = X[sub]
        n_clusters = len(np.unique(ctx['test'][1]))
        out = {}
        for name in models:
            yp = P.make_model(task, name, n_clusters).fit_predict(Xs)
            out[name] = float(silhouette_score(Xs, yp, sample_size=min(m, N_SUB),
                                               random_state=42))
        return out
    res, _, _ = P.score_encoded(task, X, y, {'test': ctx['test']}, models, metric)
    return {name: res[name]['test'] for name in models}


def _f(v):
    return None if v is None or not np.isfinite(v) else float(v)


def fiber_record(entries, model, delta=DELTA, c=C_GRID):
    """Oscillation statistics of one fiber for one downstream model."""
    perfs = np.array([e['perf'][model] for e in entries], dtype=float)
    a = np.array([e['A'] for e in entries])
    v = np.array([e['V'] for e in entries])
    a_spread = float(a.max() - a.min())
    v_spread = float(v.max() - v.min())
    k = len(entries)
    eps = c * math.sqrt(math.log(4.0 / delta) / k)
    return dict(n_policies=k,
                perf_min=_f(perfs.min()), perf_max=_f(perfs.max()),
                perf_mean=_f(perfs.mean()), perf_std=_f(perfs.std()),
                omega=_f(perfs.max() - perfs.min()),
                A_spread=a_spread, V_spread=v_spread,
                eps_K=eps, within_eps_K=bool(max(a_spread, v_spread) <= eps),
                policies=[dict(name=e['name'], perf=_f(e['perf'][model]),
                               A=float(e['A']), V=float(e['V'])) for e in entries])


def run_dataset(ds, root, n_random):
    cfg = P.CONFIGS[ds]
    task, metric = cfg['task'], cfg['metric']
    model, second = cfg['downstream_model'], cfg.get('second_model')
    models = [m for m in (model, second) if m]
    t0 = time.time()
    ctx = P.dataset_context(root, ds)
    tr = ctx['tr']
    dtr = ctx['dirty'].iloc[tr].reset_index(drop=True)
    ctr = ctx['clean'].iloc[tr].reset_index(drop=True)
    cand, cols = candidates(ctx, root, ds)
    n_rows, n_cols = len(dtr), len(cols)
    rec = dict(dataset=ds, task=task, model=model, second_model=second, metric=metric,
               n_train_rows=n_rows, n_cells=n_rows * n_cols,
               n_candidates=int(len(cand)), seed=42,
               grid=dict(delta=DELTA, C=C_GRID))
    sc = P.read_json(os.path.join(P._ROOT, 'results', 'main_table', ds, 'demandclean',
                                  'scores.json')) or {}
    refs = sc.get('references') or {}
    rec['demandclean_test'] = sc.get('test')
    rec['demandclean_nofix'] = refs.get('NoFix')
    rec['demandclean_gain'] = (_f(sc.get('test') - refs.get('NoFix'))
                               if sc.get('test') is not None and refs.get('NoFix') is not None
                               else None)
    X0, y0 = ctx['enc'].encode(dtr)
    nofix = score_models(task, X0, y0, ctx, models, metric)
    Xc, yc = ctx['enc'].encode(ctr)
    repairall = score_models(task, Xc, yc, ctx, models, metric)
    rec['nofix'] = _f(nofix[model])
    rec['repairall'] = _f(repairall[model])
    rec['nofix_all_models'] = {k: _f(v) for k, v in nofix.items()}
    rec['repairall_all_models'] = {k: _f(v) for k, v in repairall.items()}
    repair_fibers, delete_fibers = {}, {}
    if len(cand) == 0:
        rec['note'] = 'no oracle candidate cells in the training split'
    else:
        col_pr = column_priority(ctx, X0, cols)
        row_cnt = pd.Series(cand[:, 0]).value_counts()
        cand_rows = np.sort(row_cnt.index.values)
        mu = X0.mean(axis=0)
        impact = ((X0[cand_rows] - mu) ** 2).sum(axis=1)  # covariance-change proxy
        for k in K_REPAIR:
            n_fix = int(round(k * len(cand)))
            entries = []
            for name, sel in repair_policies(cand, n_fix, col_pr, n_random):
                df = apply_repair(dtr, ctr, cand, sel, cols)
                X, y = ctx['enc'].encode(df)
                entries.append(dict(name=name, perf=score_models(task, X, y, ctx, models, metric),
                                    A=1.0 - (len(cand) - len(sel)) / (n_rows * n_cols),
                                    V=float(np.var(X, axis=0).mean())))
            repair_fibers[str(k)] = entries
            print(f'{ds} repair k={k}: omega={max(e["perf"][model] for e in entries) - min(e["perf"][model] for e in entries):.6f}',
                  flush=True)
        for k in K_DELETE:
            n_del = int(round(k * len(cand_rows)))
            entries = []
            for name, rows in delete_policies(cand_rows, n_del, impact, n_random):
                keep = np.ones(n_rows, dtype=bool)
                keep[rows] = False
                X, y = ctx['enc'].encode(dtr.iloc[keep].reset_index(drop=True))
                resolved = int(row_cnt.loc[rows].sum())
                entries.append(dict(name=name, perf=score_models(task, X, y, ctx, models, metric),
                                    A=1.0 - (len(cand) - resolved) / (keep.sum() * n_cols),
                                    V=float(np.var(X, axis=0).mean())))
            delete_fibers[str(k)] = entries
            print(f'{ds} delete k={k}: omega={max(e["perf"][model] for e in entries) - min(e["perf"][model] for e in entries):.6f}',
                  flush=True)
    rec['repair_fibers'] = {k: fiber_record(e, model) for k, e in repair_fibers.items()}
    rec['delete_fibers'] = {k: fiber_record(e, model) for k, e in delete_fibers.items()}
    if second:
        rec['second'] = dict(
            model=second, nofix=_f(nofix[second]), repairall=_f(repairall[second]),
            repair_fibers={k: fiber_record(e, second) for k, e in repair_fibers.items()},
            delete_fibers={k: fiber_record(e, second) for k, e in delete_fibers.items()})
    rec['runtime_sec'] = round(time.time() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--datasets', default=None, help='comma list; default every configured dataset')
    ap.add_argument('--quick', action='store_true',
                    help='beers+bike only with 4 random policies per fiber (default 16)')
    ap.add_argument('--n-random', type=int, default=None)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    root = P.data_root(a.data_root)
    datasets = a.datasets.split(',') if a.datasets else sorted(P.CONFIGS)
    n_random = a.n_random if a.n_random is not None else (4 if a.quick else 16)
    if a.quick and not a.datasets:
        datasets = ['beers', 'bike']
    results = {}
    for ds in datasets:
        print(f'=== {ds} ===', flush=True)
        try:
            results[ds] = run_dataset(ds, root, n_random)
        except Exception as e:  # keep going with the other datasets
            results[ds] = dict(dataset=ds, error=f'{type(e).__name__}: {e}')
            print(f'{ds} ERROR: {e}', flush=True)
        tmp = a.out + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(results, f, indent=2)
        os.replace(tmp, a.out)
    print(json.dumps({ds: (r.get('error') or 'ok') for ds, r in results.items()}))


if __name__ == '__main__':
    main()
