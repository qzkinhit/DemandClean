#!/usr/bin/env python
"""How much of the downstream performance of a cleaning policy its quality state (A, V) determines.

    python evaluation/quality_state_sufficiency.py --datasets beers --n 600 --workers 8
    python evaluation/quality_state_sufficiency.py --analyze

Policies. A random policy draws operation probabilities p ~ Dirichlet(1, 1, 1, 1) over
{no-op, GT-repair, delete, VE-replace} with its own seed. It deletes every training row that holds
an oracle candidate cell (a training cell whose canonical dirty value, evaluation/table_redr.py,
differs from the canonical clean value) with probability p_delete, and assigns every candidate
cell of the kept rows no-op, GT-repair or VE-replace with probabilities proportional to the other
three components. GT-repair writes the clean value and VE-replace writes the value of the ReplaceAll
reference
(results/main_table/<ds>/references/replaceall/cleaned.csv.gz). NoFix, RepairAll, ReplaceAll and
the DemandClean output are measured the same way. For DemandClean, A compares its cleaned table
(results/main_table/<ds>/demandclean/cleaned.csv) with the dirty training split, and the other
statistics use the encoded matrix that the main table scores (cleaned_encoded.npz).

Statistics of a cleaned training table D' that keeps n_r of the n training rows, with d columns and
X' its encoding by the TrainEncoder of evaluation/paper_score.py:
    A     1 - U / (n_r * d), U the candidates left unresolved in the kept rows (authenticity)
    V     mean over the encoded columns that vary in X_dirty of (n_r / n) * var(X') / var(X_dirty)
          (diversity)
    corr  ||C(X') - C(X_clean)||_F / ||C(X_dirty) - C(X_clean)||_F, C the column correlation matrix
    ltv   total-variation distance between the label distributions of D' and of the clean
          training split (classification only)
    sw    sliced Wasserstein-1 distance between X' and X_clean, 50 directions, <= 2000 rows each
    perf  first-choice model on the clean test split, scored as evaluation/paper_score.py does
          (clustering: silhouette on a <= 10000-row subsample of the kept rows, seed 42)

Analysis (--analyze). Scores are clipped to [0, 1]. The (A, V) range of the random policies is cut
into a 20 x 20 grid, and the empirical fiber oscillation omega(q) of a grid cell q holding at least
three policies is the range of their scores. The 5-fold 10-NN R^2 of the score is reported from
(A, V), from (A, V, corr, sw, ltv) and from (corr, sw, ltv). The DemandClean output is compared with
the 20 random policies nearest to it in the standardized (A, V) plane and with all random policies.
Output: results/robustness/quality_state/<ds>_policies.csv and summary.csv
"""
import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_score as P  # noqa: E402
from table_redr import canon, read_str  # noqa: E402

OUT = os.path.join(P._ROOT, 'results', 'robustness', 'quality_state')
DATASETS = ['adult', 'beers', 'breast_cancer', 'smartfactory', 'bike', 'mercedes', 'nasa',
            'soilmoisture', 'har']
NOOP, GT, DEL, VE = 0, 1, 2, 3
N_SUB = 10000
SW_DIRS, SW_ROWS = 50, 2000
GRID, MIN_CELL, K_NEAR = 20, 3, 20

_G = {}


def candidates(ctx, root, ds):
    """Oracle candidate cells of the training split, as (row within the training split, position
    in cols), where the canonical dirty value differs from the canonical clean value. cols holds
    every column except index, the label column included."""
    d = read_str(os.path.join(root, ds, 'dirty_index.csv'))
    c = read_str(os.path.join(root, ds, 'clean_index.csv'))
    cols = [x for x in d.columns if x != 'index']
    D = d[cols].apply(lambda s: s.map(canon))
    C = c[cols].apply(lambda s: s.map(canon))
    return np.argwhere((D != C).values[ctx['tr']]), cols


def _corr(X):
    with np.errstate(invalid='ignore', divide='ignore'):
        C = np.corrcoef(X, rowvar=False)
    return np.nan_to_num(np.atleast_2d(C))


def _rows(n, m, seed):
    return np.random.RandomState(seed).choice(n, size=m, replace=False) if n > m else np.arange(n)


def _setup(ds, root):
    cfg = P.CONFIGS[ds]
    ctx = P.dataset_context(root, ds)
    tr = ctx['tr']
    dtr = ctx['dirty'].iloc[tr].reset_index(drop=True)
    ctr = ctx['clean'].iloc[tr].reset_index(drop=True)
    rep = pd.read_csv(os.path.join(P._ROOT, 'results', 'main_table', ds, 'references', 'replaceall',
                                   'cleaned.csv.gz'))
    if len(rep) == len(ctx['dirty']):
        vtr = rep.iloc[tr].reset_index(drop=True)
    else:
        vtr = rep.set_index('index').loc[dtr['index'].values].reset_index()
    if not np.array_equal(pd.to_numeric(vtr['index']).values, pd.to_numeric(dtr['index']).values):
        raise ValueError(f'{ds}: ReplaceAll rows do not align with the training split')
    cand, cols = candidates(ctx, root, ds)
    enc = ctx['enc']
    Xd, _ = enc.encode(dtr)
    Xc, yc = enc.encode(ctr)
    var_d = Xd.var(axis=0)
    rng = np.random.RandomState(7)
    dirs = rng.normal(size=(SW_DIRS, Xd.shape[1]))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pc = (Xc[_rows(len(Xc), SW_ROWS, 11)] @ dirs.T)
    Cc, Cd = _corr(Xc), _corr(Xd)
    task = cfg['task']
    ytest = ctx['test'][1]
    g = dict(ds=ds, task=task, model=cfg['downstream_model'], metric=cfg['metric'], ctx=ctx,
             dtr=dtr, ctr=ctr, vtr=vtr, cand=cand, cols=cols, d=len(cols), n=len(dtr), enc=enc,
             var_d=var_d, vmask=var_d > 1e-12, dirs=dirs, pc=pc, Cc=Cc,
             corr_den=float(np.linalg.norm(Cd - Cc)) or 1.0,
             n_clusters=len(np.unique(ytest[~np.isnan(ytest)])))
    if task == 'classification':
        yy = yc[~np.isnan(yc)]
        k, c = np.unique(yy, return_counts=True)
        g['pc_label'] = dict(zip(k.tolist(), (c / c.sum()).tolist()))
    return g


def _init(ds, root):
    _G.update(_setup(ds, root))


def _perf(g, X, y):
    if g['task'] == 'clustering':
        sub = _rows(len(X), N_SUB, 42)
        Xs = X[sub]
        yp = P.make_model('clustering', g['model'], g['n_clusters']).fit_predict(Xs)
        return float(silhouette_score(Xs, yp, sample_size=min(len(Xs), N_SUB), random_state=42))
    res, _, _ = P.score_encoded(g['task'], X, y, {'test': g['ctx']['test']}, [g['model']],
                                g['metric'])
    return float(res[g['model']]['test'])


def _stats(g, df_kept, n_kept, unresolved, seed):
    X, y = g['enc'].encode(df_kept)
    return _stats_encoded(g, X, y, n_kept, unresolved, seed)


def _stats_encoded(g, X, y, n_kept, unresolved, seed):
    r = n_kept / g['n']
    m = g['vmask']
    out = dict(n_kept=int(n_kept), A=1.0 - unresolved / (n_kept * g['d']),
               V=float(np.mean(r * X[:, m].var(axis=0) / g['var_d'][m])),
               corr=float(np.linalg.norm(_corr(X) - g['Cc']) / g['corr_den']))
    if g['task'] == 'classification':
        yy = y[~np.isnan(y)]
        k, c = np.unique(yy, return_counts=True)
        p = dict(zip(k.tolist(), (c / c.sum()).tolist()))
        keys = set(p) | set(g['pc_label'])
        out['ltv'] = 0.5 * sum(abs(p.get(t, 0.0) - g['pc_label'].get(t, 0.0)) for t in keys)
    else:
        out['ltv'] = np.nan
    px = X[_rows(len(X), SW_ROWS, seed)] @ g['dirs'].T
    out['sw'] = float(np.mean([wasserstein_distance(px[:, j], g['pc'][:, j])
                               for j in range(px.shape[1])]))
    out['perf'] = _perf(g, X, y)
    return out


def _apply(g, ops, del_rows):
    """Training split with GT-repair and VE-replace written and the deleted rows removed."""
    df = g['dtr'].copy()
    cand = g['cand']
    for code, src in ((GT, g['ctr']), (VE, g['vtr'])):
        sel = cand[ops == code]
        for cpos in np.unique(sel[:, 1]):
            rows = sel[sel[:, 1] == cpos, 0]
            col = g['cols'][cpos]
            if df[col].dtype != object:
                df[col] = df[col].astype(object)
            df.iloc[rows, df.columns.get_loc(col)] = src[col].values[rows]
    keep = np.ones(g['n'], dtype=bool)
    keep[del_rows] = False
    unresolved = int(np.sum((ops == NOOP) & keep[cand[:, 0]]))
    return df.iloc[keep].reset_index(drop=True), int(keep.sum()), unresolved


def _policy(i):
    g = _G
    cand = g['cand']
    rng = np.random.RandomState(1000 + i)
    p = rng.dirichlet(np.ones(4))
    cand_rows = np.unique(cand[:, 0])
    del_rows = cand_rows[rng.rand(len(cand_rows)) < p[DEL]]
    q = p[[NOOP, GT, VE]] / p[[NOOP, GT, VE]].sum()
    ops = np.array([NOOP, GT, VE])[rng.choice(3, size=len(cand), p=q)]
    df, n_kept, unresolved = _apply(g, ops, del_rows)
    rec = dict(policy='random', seed=1000 + i, p_noop=p[NOOP], p_gt=p[GT], p_del=p[DEL], p_ve=p[VE],
               gt_cells=int(np.sum((ops == GT) & np.isin(cand[:, 0], del_rows, invert=True))))
    rec.update(_stats(g, df, n_kept, unresolved, seed=1000 + i))
    return rec


def _references(g):
    cand = g['cand']
    out = []
    for name, code in (('NoFix', NOOP), ('RepairAll', GT), ('ReplaceAll', VE)):
        df, n_kept, unresolved = _apply(g, np.full(len(cand), code), np.array([], dtype=int))
        rec = dict(policy=name, seed=-1, gt_cells=int(len(cand)) if code == GT else 0)
        rec.update(_stats(g, df, n_kept, unresolved, seed=0))
        out.append(rec)
    return out


def _demandclean(g, root):
    """The DemandClean output on the same statistics, rows aligned by the index column."""
    run = os.path.join(P._ROOT, 'results', 'main_table', g['ds'], 'demandclean')
    path = os.path.join(run, 'cleaned.csv')
    if not os.path.exists(path):
        path += '.gz'
    dc_str = read_str(path)
    d_str = read_str(os.path.join(root, g['ds'], 'dirty_index.csv')).iloc[g['ctx']['tr']]
    d_str = d_str.reset_index(drop=True)
    pos = {v: i for i, v in enumerate(d_str['index'].values)}
    kept_pos = np.array([pos[v] for v in dc_str['index'].values])
    keep = np.zeros(g['n'], dtype=bool)
    keep[kept_pos] = True
    cand = g['cand']
    in_kept = keep[cand[:, 0]]
    row_of = np.full(g['n'], -1)
    row_of[kept_pos] = np.arange(len(kept_pos))
    unresolved = 0
    for cpos in np.unique(cand[:, 1]):
        sel = cand[(cand[:, 1] == cpos) & in_kept]
        if len(sel) == 0:
            continue
        col = g['cols'][cpos]
        dv = d_str[col].values[sel[:, 0]]
        ov = dc_str[col].values[row_of[sel[:, 0]]]
        unresolved += int(sum(canon(a) == canon(b) for a, b in zip(dv, ov)))
    z = np.load(os.path.join(run, 'cleaned_encoded.npz'), allow_pickle=True)
    rec = dict(policy='DemandClean', seed=-1)
    sc = P.read_json(os.path.join(run, 'scores.json')) or {}
    rec['perf_paper'] = sc.get('test')
    rec.update(_stats_encoded(g, np.asarray(z['X_result'], dtype=float),
                              np.asarray(z['y_result'], dtype=float), int(keep.sum()), unresolved, seed=0))
    return rec


def run(ds, n, workers, root, out):
    t0 = time.time()
    g = _setup(ds, root)
    rows = _references(g)
    rows.append(_demandclean(g, root))
    print(f'{ds}: references done in {time.time() - t0:.0f}s', flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(ds, root)) as ex:
        for k, rec in enumerate(ex.map(_policy, range(n), chunksize=4)):
            rows.append(rec)
            if (k + 1) % 100 == 0:
                print(f'{ds}: {k + 1}/{n} policies, {time.time() - t0:.0f}s', flush=True)
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, f'{ds}_policies.csv'), index=False)
    print(f'{ds}: done in {time.time() - t0:.0f}s', flush=True)


def _r2(df, cols, y):
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    mdl = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10))
    cv = KFold(5, shuffle=True, random_state=0)
    return float(cross_val_score(mdl, df[cols].values, y, cv=cv, scoring='r2').mean())


def analyze(out):
    rows = []
    for path in sorted(glob.glob(os.path.join(out, '*_policies.csv'))):
        ds = os.path.basename(path)[:-len('_policies.csv')]
        t = pd.read_csv(path)
        t['y'] = t['perf'].clip(0, 1)
        ref = {p: float(t.loc[t.policy == p, 'y'].iloc[0]) for p in ('NoFix', 'RepairAll', 'DemandClean')}
        r = t[t.policy == 'random'].reset_index(drop=True)
        a_e = np.linspace(r.A.min(), r.A.max() + 1e-12, GRID + 1)
        v_e = np.linspace(r.V.min(), r.V.max() + 1e-12, GRID + 1)
        r['cell'] = np.digitize(r.A, a_e) * (GRID + 2) + np.digitize(r.V, v_e)
        grp = r.groupby('cell')['y']
        om = (grp.max() - grp.min())[grp.size() >= MIN_CELL]
        extra = ['corr', 'sw'] + (['ltv'] if r['ltv'].notna().any() else [])
        dc = t[t.policy == 'DemandClean'].iloc[0]
        za = (r.A - r.A.mean()) / (r.A.std() or 1.0)
        zv = (r.V - r.V.mean()) / (r.V.std() or 1.0)
        dist = np.hypot(za - (dc.A - r.A.mean()) / (r.A.std() or 1.0),
                        zv - (dc.V - r.V.mean()) / (r.V.std() or 1.0))
        near = r.loc[np.argsort(dist.values)[:K_NEAR], 'y']
        summ = P.read_json(os.path.join(P._ROOT, 'results', 'main_table', ds, 'demandclean',
                                        'summary.json')) or {}
        dc_gt = float((summ.get('ground_truth_cells') or {}).get('total') or np.nan)
        gt_ge = r[r.gt_cells >= dc_gt]
        rows.append(dict(
            dataset=ds, n_random=len(r), cells=int(len(om)),
            nofix=ref['NoFix'], repairall=ref['RepairAll'], demandclean=ref['DemandClean'],
            demandclean_paper=dc.get('perf_paper'), dc_A=float(dc.A), dc_V=float(dc.V),
            random_max=float(r.y.max()), random_span=float(r.y.max() - r.y.min()),
            omega_median=float(om.median()), omega_p90=float(om.quantile(0.9)), omega_max=float(om.max()),
            r2_AV=_r2(r, ['A', 'V'], r.y.values), r2_AV_plus=_r2(r, ['A', 'V'] + extra, r.y.values),
            r2_plus_only=_r2(r, extra, r.y.values),
            dc_beats_near=float((near < ref['DemandClean']).mean()),
            dc_beats_all=float((r.y < ref['DemandClean']).mean()),
            near_dist=float(np.sort(dist.values)[K_NEAR - 1]),
            gain_clean=ref['RepairAll'] - ref['NoFix'],
            omega_median_over_gain=(float(om.median()) / (ref['RepairAll'] - ref['NoFix'])
                                    if ref['RepairAll'] > ref['NoFix'] else np.nan),
            dc_gt=dc_gt, random_gt_mean=float(r.gt_cells.mean()),
            share_random_gt_ge_dc=float(len(gt_ge) / len(r)),
            dc_beats_gt_ge=float((gt_ge.y < ref['DemandClean']).mean()) if len(gt_ge) else np.nan))
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(out, 'summary.csv'), index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        print(s.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--datasets', default=','.join(DATASETS))
    ap.add_argument('--n', type=int, default=600)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--analyze', action='store_true')
    a = ap.parse_args()
    if a.analyze:
        analyze(a.out)
        return
    root = P.data_root(a.data_root)
    for ds in a.datasets.split(','):
        run(ds, a.n, a.workers, root, a.out)


if __name__ == '__main__':
    main()
