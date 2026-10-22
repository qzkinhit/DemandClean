"""ActiveClean core algorithm (Krishnan, Wang, Wu, Franklin, Goldberg, PVLDB 9(12), 2016).

Faithful re-implementation of the paper's progressive cleaning loop for tabular data:

* Featurizer F(r) -> (x, y), fit on the dirty training records only.
* Model: convex regularised loss. Classification uses a linear SVM (hinge loss,
  one-versus-all for more than two classes), regression uses linear regression
  (squared loss). Both carry the L2 term (lam / 2) * ||W||^2 (intercept excluded).
* Initialisation theta(0) = theta(d), the model trained on the dirty records.
* Sampler: importance sampling with p(i) proportional to the estimated norm of the
  clean-record gradient (Section 5 / 6.3), mixed with a uniform floor so that p(i) > 0.
* Cleaner: an oracle supplied by the caller returns the clean version of each sampled
  record. Every record is requested at most once.
* Updater: theta(t+1) = theta(t) - gamma_t * (|R_dirty| / |R| * g_S + |R_clean| / |R| * g_C)
  with the importance-weighted sample gradient g_S and the exact gradient g_C over the
  records known (or detected) to be clean.
* Detector: the adaptive detector of Section 6.2, an all-versus-one linear SVM over the
  classes {clean, error classes} trained on the records cleaned so far. A record is
  removed from R_dirty only when it lies beyond the margin of the clean class.
* Estimator: the linearised estimate grad(x, y) + M_x * dx_u + M_y * dy_u with the
  average feature and label change dx_u, dy_u of error class u (Section 6.3, Appendix).

The module contains no argument parsing, no file paths and no test-set evaluation.
The clean validation records are used only for model selection (regularisation of the
dirty model, step-size constant and the stopping iterate), as allowed by the protocol.
"""

import math
import re
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC

MISSING_TOKENS = frozenset(['', 'empty', 'Empty', 'EMPTY', 'nan', 'NaN', 'NULL', 'null',
                            'None', '__NULL__'])
MISSING_CATEGORY = '__missing__'
CLEAN_TAG = 'clean'
# 12-hour clock times such as "07:10 AM"; any other spelling is treated as unparsable
_CLOCK_RE = re.compile(r'^(0?[1-9]|1[0-2]):([0-5][0-9]) ?([AaPp])\.?[Mm]\.?$')


# ---------------------------------------------------------------------------
# Configuration and result containers
# ---------------------------------------------------------------------------
class ActiveCleanConfig(object):
    """Hyper-parameters of one ActiveClean run."""

    def __init__(self, feature_cols, categorical_cols, label_col, task,
                 budget_rows, batch_size=50, seed=42,
                 step_grid=(0.1, 0.3, 1.0, 3.0), primary_step=1.0,
                 uniform_mix=0.1, detector_min_per_class=5, detector_margin=1.0,
                 detector_rare_class=3, svm_c_grid=(0.01, 0.1, 1.0, 10.0),
                 ridge_alpha_grid=(0.1, 1.0, 10.0, 100.0), max_categories=100,
                 min_category_count=2, z_clip=10.0, svm_max_iter=20000, clock_cols=()):
        if task not in ('classification', 'regression'):
            raise ValueError('ActiveClean needs a supervised convex loss, got task=%r' % task)
        self.feature_cols = list(feature_cols)
        self.categorical_cols = set(categorical_cols)
        self.clock_cols = set(clock_cols)
        self.label_col = label_col
        self.task = task
        self.budget_rows = int(budget_rows)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.step_grid = tuple(float(c) for c in step_grid)
        self.primary_step = float(primary_step)
        if self.primary_step not in self.step_grid:
            self.step_grid = tuple(sorted(set(self.step_grid) | {self.primary_step}))
        self.uniform_mix = float(uniform_mix)
        self.detector_min_per_class = int(detector_min_per_class)
        self.detector_margin = float(detector_margin)
        self.detector_rare_class = int(detector_rare_class)
        self.svm_c_grid = tuple(svm_c_grid)
        self.ridge_alpha_grid = tuple(ridge_alpha_grid)
        self.max_categories = int(max_categories)
        self.min_category_count = int(min_category_count)
        self.z_clip = float(z_clip)
        self.svm_max_iter = int(svm_max_iter)

    def as_dict(self):
        d = dict(self.__dict__)
        d['categorical_cols'] = sorted(self.categorical_cols)
        d['clock_cols'] = sorted(self.clock_cols)
        return d


class ActiveCleanResult(object):
    """Output of run_activeclean: the selected model and the cleaned record positions."""

    def __init__(self, featurizer, model, cleaned_positions, cleaned_step, history, info):
        self.featurizer = featurizer
        self.model = model
        self.cleaned_positions = cleaned_positions
        self.cleaned_step = cleaned_step
        self.history = history
        self.info = info

    def predict(self, df):
        """Predictions of the selected model in the original label space."""
        Xa = _augment(self.featurizer.transform_X(df))
        if self.featurizer.task == 'classification':
            return self.featurizer.classes_[self.model.predict_index(Xa)]
        return self.featurizer.decode_regression(self.model.scores(Xa)[:, 0])


# ---------------------------------------------------------------------------
# Featurizer F(r)
# ---------------------------------------------------------------------------
class TabularFeaturizer(object):
    """Numeric columns: parsed value, robust z-score with the dirty-train median and
    inter-quartile range, clipped to +-z_clip, missing or unparsable cells set to the centre
    (z = 0) plus a missing indicator for columns that have such cells in the dirty train.
    Categorical columns: one-hot over the frequent dirty-train categories.
    Clock columns (optional): a 12-hour time "hh:mm AM" becomes minutes after midnight and is
    then treated as a numeric column; every other spelling counts as unparsable.
    Labels: class index (classification) or z-scored target (regression)."""

    def __init__(self, feature_cols, categorical_cols, label_col, task,
                 max_categories=100, min_count=2, z_clip=10.0, clock_cols=()):
        self.feature_cols = list(feature_cols)
        self.clock_cols = set(clock_cols)
        self.categorical_cols = set(categorical_cols) - self.clock_cols
        self.label_col = label_col
        self.task = task
        self.max_categories = max_categories
        self.min_count = min_count
        self.z_clip = float(z_clip)

    @staticmethod
    def _parse_num(series):
        s = series.astype(str).str.strip()
        s = s.where(~s.isin(MISSING_TOKENS), None)
        v = pd.to_numeric(s, errors='coerce').astype(float).values
        v[~np.isfinite(v)] = np.nan
        return v

    @staticmethod
    def _parse_clock(series):
        out = np.full(len(series), np.nan)
        for i, v in enumerate(series.astype(str).str.strip().values):
            m = _CLOCK_RE.match(v)
            if m:
                h = int(m.group(1)) % 12 + (12 if m.group(3).lower() == 'p' else 0)
                out[i] = 60.0 * h + int(m.group(2))
        return out

    def _num(self, c, series):
        return self._parse_clock(series) if c in self.clock_cols else self._parse_num(series)

    @staticmethod
    def _norm_cat(series):
        s = series.astype(str).str.strip()
        return s.where(~s.isin(MISSING_TOKENS), MISSING_CATEGORY)

    def fit(self, df):
        self.num_cols = [c for c in self.feature_cols if c not in self.categorical_cols]
        self.cat_cols = [c for c in self.feature_cols if c in self.categorical_cols]
        self.num_stats = {}
        self.blocks = []
        pos = 0
        for c in self.feature_cols:
            if c in self.categorical_cols:
                vc = self._norm_cat(df[c]).value_counts()
                vc = vc[vc >= self.min_count]
                cats = [str(v) for v in vc.index[:self.max_categories]]
                width = len(cats)
                self.num_stats[c] = {'vocab': dict((v, i) for i, v in enumerate(cats))}
            else:
                v = self._num(c, df[c])
                ok = ~np.isnan(v)
                mu, sd = _robust_location_scale(v[ok])
                has_missing = bool((~ok).any())
                width = 2 if has_missing else 1
                self.num_stats[c] = {'mean': mu, 'std': sd, 'indicator': has_missing}
            self.blocks.append((c, pos, pos + width))
            pos += width
        self.n_features = pos
        if self.task == 'classification':
            lab = df[self.label_col].astype(str).str.strip()
            lab = lab[~lab.isin(MISSING_TOKENS)]
            self.classes_ = np.array(sorted(lab.unique()))
            self.class_index = dict((v, i) for i, v in enumerate(self.classes_))
        else:
            y = self._parse_num(df[self.label_col])
            ok = ~np.isnan(y)
            self.y_mean = float(np.mean(y[ok]))
            sd = float(np.std(y[ok]))
            self.y_std = sd if sd > 1e-12 else 1.0
        return self

    def transform_X(self, df):
        n = len(df)
        X = np.zeros((n, self.n_features), dtype=float)
        for c, s, e in self.blocks:
            st = self.num_stats[c]
            if 'vocab' in st:
                codes = self._norm_cat(df[c]).map(st['vocab']).values
                known = pd.notna(codes)
                rows = np.where(known)[0]
                X[rows, s + codes[known].astype(int)] = 1.0
            else:
                v = self._num(c, df[c])
                miss = np.isnan(v)
                z = np.clip((v - st['mean']) / st['std'], -self.z_clip, self.z_clip)
                z[miss] = 0.0
                X[:, s] = z
                if st['indicator']:
                    X[:, s + 1] = miss.astype(float)
        return X

    def transform_y(self, df):
        if self.task == 'classification':
            lab = df[self.label_col].astype(str).str.strip()
            return np.array([self.class_index.get(v, -1) for v in lab], dtype=int)
        y = self._parse_num(df[self.label_col])
        return (y - self.y_mean) / self.y_std

    def valid_y(self, y):
        if self.task == 'classification':
            return y >= 0
        return np.isfinite(y)

    def decode_regression(self, z):
        return z * self.y_std + self.y_mean

    def changed_blocks(self, xc, xd, yc, yd):
        """Source columns whose featurised value changed by cleaning (for error-class tags)."""
        out = []
        for c, s, e in self.blocks:
            if not np.allclose(xc[s:e], xd[s:e], rtol=1e-9, atol=1e-9):
                out.append(c)
        if self.task == 'classification':
            if yc != yd:
                out.append(self.label_col)
        elif not (np.isfinite(yc) and np.isfinite(yd) and abs(yc - yd) <= 1e-9):
            out.append(self.label_col)
        return out


def _robust_location_scale(v):
    """Median and IQR / 1.349; for near-constant columns the std of the values winsorised
    at the 0.5% and 99.5% quantiles. Injected outliers do not inflate the scale."""
    if len(v) == 0:
        return 0.0, 1.0
    med = float(np.median(v))
    q1, q3 = np.percentile(v, [25, 75])
    sd = float(q3 - q1) / 1.349
    if not np.isfinite(sd) or sd < 1e-12:
        lo, hi = np.percentile(v, [0.5, 99.5])
        sd = float(np.std(np.clip(v, lo, hi)))
    if not np.isfinite(sd) or sd < 1e-12:
        sd = 1.0
    return med, sd


def _augment(X):
    return np.hstack([X, np.ones((X.shape[0], 1))])


# ---------------------------------------------------------------------------
# Convex models: W has shape (d + 1, K); the last row is the intercept
# ---------------------------------------------------------------------------
class _LinearModel(object):
    def __init__(self, W, lam):
        self.W = np.array(W, dtype=float)
        self.lam = float(lam)

    def copy_with(self, W):
        m = self.__class__.__new__(self.__class__)
        m.__dict__.update(self.__dict__)
        m.W = np.array(W, dtype=float)
        return m

    def scores(self, Xa, W=None):
        return Xa.dot(self.W if W is None else W)

    def reg_grad(self, W):
        G = self.lam * W
        G[-1, :] = 0.0
        return G


class HingeOvR(_LinearModel):
    """Linear SVM, phi = sum_k max(0, 1 - y_k * w_k^T x) + (lam / 2) ||W||^2, y_k in {-1, +1}.
    K = 1 for binary problems (class index 1 is the positive class)."""
    loss = 'hinge'

    def __init__(self, W, lam, n_classes):
        _LinearModel.__init__(self, W, lam)
        self.n_classes = int(n_classes)

    def targets(self, yidx):
        yidx = np.asarray(yidx, dtype=int)
        n = len(yidx)
        if self.n_classes == 2:
            T = np.where(yidx == 1, 1.0, -1.0)[:, None]
        else:
            T = -np.ones((n, self.n_classes))
            ok = yidx >= 0
            T[np.where(ok)[0], yidx[ok]] = 1.0
        T[yidx < 0] = 0.0
        return T

    def coeffs(self, Xa, T, W=None):
        """Per-record gradient of the data term is x a^T; returns a and the active mask."""
        S = self.scores(Xa, W)
        active = (T * S < 1.0) & (T != 0.0)
        return -T * active, active

    def label_coeffs(self, active, DX, DT, W=None):
        # M_y * dy for the hinge: -x * dy_k on the active columns
        return -DT * active

    def predict_index(self, Xa, W=None):
        S = self.scores(Xa, W)
        if self.n_classes == 2:
            return (S[:, 0] > 0).astype(int)
        return np.argmax(S, axis=1)


class SquaredLoss(_LinearModel):
    """Linear regression, phi = 0.5 * (w^T x - y)^2 + (lam / 2) ||w||^2 on the z-scored target."""
    loss = 'squared'
    n_classes = 1

    def targets(self, y):
        T = np.asarray(y, dtype=float)[:, None].copy()
        T[~np.isfinite(T)] = 0.0
        return T

    def coeffs(self, Xa, T, W=None):
        return self.scores(Xa, W) - T, None

    def label_coeffs(self, active, DX, DT, W=None):
        # M_x * dx = r * dx + x * (w^T dx) and M_y * dy = -x * dy, so c = w^T dx - dy
        return DX.dot(self.W if W is None else W) - DT

    def predict_index(self, Xa, W=None):
        raise TypeError('regression model has no class index')


# ---------------------------------------------------------------------------
# Dirty model theta(d) with validation-based regularisation choice
# ---------------------------------------------------------------------------
def _val_score(model, W, Xva, yva, task):
    if task == 'classification':
        return float(np.mean(model.predict_index(Xva, W) == yva)) if len(yva) else float('nan')
    ok = np.isfinite(yva)
    pred = model.scores(Xva[ok], W)[:, 0]
    y = yva[ok]
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return float('nan')
    return 1.0 - float(np.sum((y - pred) ** 2)) / ss_tot


def _fit_dirty_model(cfg, feat, X, y, Xva_a, yva):
    n = X.shape[0]
    trials = []
    best = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        if cfg.task == 'classification':
            K = len(feat.classes_)
            for C in cfg.svm_c_grid:
                clf = LinearSVC(C=C, loss='hinge', dual=True, max_iter=cfg.svm_max_iter,
                                random_state=cfg.seed)
                clf.fit(X, y)
                W = np.vstack([clf.coef_.T, clf.intercept_[None, :]])
                if K > 2 and W.shape[1] != K:
                    # a class absent from the fit: give it a column that never wins
                    full = np.zeros((W.shape[0], K))
                    full[-1, :] = -1e6
                    full[:, clf.classes_] = W
                    W = full
                model = HingeOvR(W, 1.0 / (C * n), K)
                s = _val_score(model, model.W, Xva_a, yva, cfg.task)
                trials.append({'C': C, 'val': s})
                if best is None or s > best[0] + 1e-12:
                    best = (s, model, {'C': C, 'lam': model.lam})
        else:
            for a in cfg.ridge_alpha_grid:
                reg = Ridge(alpha=a)
                reg.fit(X, y)
                W = np.concatenate([reg.coef_.ravel(), [float(reg.intercept_)]])[:, None]
                model = SquaredLoss(W, a / n)
                s = _val_score(model, model.W, Xva_a, yva, cfg.task)
                trials.append({'alpha': a, 'val': s})
                if best is None or s > best[0] + 1e-12:
                    best = (s, model, {'alpha': a, 'lam': model.lam})
    info = dict(best[2])
    info['val_score_dirty_model'] = best[0]
    info['grid'] = trials
    info['convergence_warnings'] = sum(1 for w in caught
                                       if issubclass(w.category, ConvergenceWarning))
    return best[1], info


# ---------------------------------------------------------------------------
# Adaptive detector (Section 6.2)
# ---------------------------------------------------------------------------
class AdaptiveDetector(object):
    """All-versus-one linear SVM over {clean, error classes}; records beyond the clean-class
    margin are treated as clean, every other record stays in R_dirty."""

    def __init__(self, margin, seed):
        self.margin = margin
        self.seed = seed
        self.clf = None

    def fit(self, Z, tags):
        clf = LinearSVC(C=1.0, class_weight='balanced', dual=True, max_iter=5000,
                        random_state=self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            clf.fit(Z, tags)
        self.clf = clf
        self.classes_ = list(clf.classes_)
        return self

    def predict(self, Z):
        """Returns (confidently_clean mask, predicted error class or None per record)."""
        dec = self.clf.decision_function(Z)
        n = Z.shape[0]
        err = np.array([None] * n, dtype=object)
        if len(self.classes_) == 2:
            pos_is_clean = self.classes_[1] == CLEAN_TAG
            f_clean = dec if pos_is_clean else -dec
            other = self.classes_[0] if pos_is_clean else self.classes_[1]
            err[f_clean < 0] = other
            return f_clean >= self.margin, err
        ci = self.classes_.index(CLEAN_TAG)
        f_clean = dec[:, ci]
        arg = np.argmax(dec, axis=1)
        for i in np.where(arg != ci)[0]:
            err[i] = self.classes_[arg[i]]
        return f_clean >= self.margin, err


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def _smoothness(Xa):
    G = Xa.T.dot(Xa) / max(1, Xa.shape[0])
    return float(np.linalg.eigvalsh(G)[-1])


def _estimated_norms(Xa, DX, A, C):
    """|| (x + dx) a^T + x c^T ||_F for every candidate record."""
    XD = Xa + DX
    n1 = np.einsum('ij,ij->i', XD, XD)
    n0 = np.einsum('ij,ij->i', Xa, Xa)
    cross = np.einsum('ij,ij->i', XD, Xa)
    a2 = np.einsum('ij,ij->i', A, A)
    c2 = np.einsum('ij,ij->i', C, C)
    ac = np.einsum('ij,ij->i', A, C)
    return np.sqrt(np.maximum(n1 * a2 + n0 * c2 + 2.0 * cross * ac, 0.0))


def _step_gradient(model, W, step, Xa_fin, T_fin, Xa_d, T_d):
    """Gradient of the update rule at W for one recorded iteration."""
    g = np.zeros_like(W)
    Nd, Nc = step['Nd'], step['Nc']
    Ntot = float(Nd + Nc)
    if len(step['draw_pos']):
        rows = step['draw_pos']
        A, _ = model.coeffs(Xa_fin[rows], T_fin[rows], W)
        g += (Nd / Ntot) * Xa_fin[rows].T.dot(A * step['draw_coef'][:, None])
    if Nc > 0:
        acc = np.zeros_like(W)
        rc = step['clean_cleaned']
        if len(rc):
            A, _ = model.coeffs(Xa_fin[rc], T_fin[rc], W)
            acc += Xa_fin[rc].T.dot(A)
        rd = step['clean_detected']
        if len(rd):
            A, _ = model.coeffs(Xa_d[rd], T_d[rd], W)
            acc += Xa_d[rd].T.dot(A)
        g += (Nc / Ntot) * acc / Nc
    return g + model.reg_grad(W)


def run_activeclean(dirty_train, val_clean, oracle, cfg, log=None):
    """Run ActiveClean on the dirty training records.

    dirty_train : DataFrame of raw dirty training records (strings).
    val_clean   : DataFrame of raw clean validation records (model selection only).
    oracle      : callable(list of row positions) -> DataFrame of the clean records,
                  the only access to ground truth. Each position is requested once.
    cfg         : ActiveCleanConfig.
    """
    t_start = time.time()
    say = log if log is not None else (lambda msg: None)
    rng = np.random.RandomState(cfg.seed)
    feat = TabularFeaturizer(cfg.feature_cols, cfg.categorical_cols, cfg.label_col, cfg.task,
                             cfg.max_categories, cfg.min_category_count,
                             cfg.z_clip, cfg.clock_cols).fit(dirty_train)
    Xd = feat.transform_X(dirty_train)
    Xa_d = _augment(Xd)
    yd = feat.transform_y(dirty_train)
    valid_d = feat.valid_y(yd)
    Xva = _augment(feat.transform_X(val_clean))
    yva = feat.transform_y(val_clean)
    n = Xd.shape[0]

    model, fit_info = _fit_dirty_model(cfg, feat, Xd[valid_d], yd[valid_d], Xva, yva)
    W0 = model.W.copy()
    T_d = model.targets(yd)
    K = T_d.shape[1]
    say('[ActiveClean] featurised d=%d, K=%d, n_train=%d, dirty-model %s, val=%.4f'
        % (feat.n_features, K, n, {k: v for k, v in fit_info.items() if k in ('C', 'alpha')},
           fit_info['val_score_dirty_model']))

    L = _smoothness(Xa_d[valid_d]) + model.lam
    Xa_cur = Xa_d.copy()
    T_cur = T_d.copy()
    valid_cur = valid_d.copy()
    cleaned = np.zeros(n, dtype=bool)
    order = []
    cleaned_step = []
    p_at_clean = np.zeros(n)
    tags = {}
    deltas = {}
    detector = None
    Zd = np.hstack([Xd, T_d])
    steps = []
    history = []
    W = W0.copy()
    used = 0
    t = 0
    budget = max(0, cfg.budget_rows)
    while used < budget:
        cand = (~cleaned) & valid_d
        det_clean = np.zeros(n, dtype=bool)
        err_class = np.array([None] * n, dtype=object)
        if detector is not None and cand.any():
            cpos_all = np.where(cand)[0]
            cc, ec = detector.predict(Zd[cpos_all])
            det_clean[cpos_all[cc]] = True
            err_class[cpos_all] = ec
            cand &= ~det_clean
        Nd = int(cand.sum())
        if Nd == 0:
            say('[ActiveClean] R_dirty is empty, stop')
            break
        clean_cleaned = np.where(cleaned & valid_cur)[0]
        clean_detected = np.where(det_clean)[0]
        Nc = len(clean_cleaned) + len(clean_detected)
        t += 1
        cpos = np.where(cand)[0]

        # Sampler with the linearised estimator
        DX = np.zeros((Nd, Xa_d.shape[1]))
        DT = np.zeros((Nd, K))
        if deltas:
            ecs = err_class[cpos]
            for u, (dx, dt) in deltas.items():
                rows = np.where(ecs == u)[0]
                if len(rows):
                    DX[rows] = dx
                    DT[rows] = dt
        A, active = model.coeffs(Xa_cur[cpos], T_cur[cpos], W)
        Cc = model.label_coeffs(active, DX, DT, W)
        norms = _estimated_norms(Xa_cur[cpos], DX, A, Cc)
        tot = float(norms.sum())
        if tot > 0 and np.isfinite(tot):
            p = (1.0 - cfg.uniform_mix) * norms / tot + cfg.uniform_mix / Nd
        else:
            p = np.full(Nd, 1.0 / Nd)
        p = p / p.sum()
        draws = rng.choice(Nd, size=cfg.batch_size, replace=True, p=p)
        keep, seen, new_local = [], set(), []
        for j in draws:
            if j not in seen:
                if len(seen) >= budget - used:
                    break
                seen.add(j)
                new_local.append(j)
            keep.append(j)
        draws = np.array(keep, dtype=int)
        new_local = np.array(new_local, dtype=int)
        new_pos = cpos[new_local]

        # Cleaner (ground truth through the oracle)
        clean_rows = oracle([int(i) for i in new_pos])
        Xc = feat.transform_X(clean_rows)
        yc = feat.transform_y(clean_rows)
        n_dirty_new = 0
        for k, i in enumerate(new_pos):
            ch = feat.changed_blocks(Xc[k], Xd[i], yc[k], yd[i])
            if not ch:
                tags[i] = CLEAN_TAG
            else:
                tags[i] = ch[0] if len(ch) == 1 else 'multi'
                n_dirty_new += 1
        Xa_cur[new_pos] = _augment(Xc)
        T_cur[new_pos] = model.targets(yc)
        valid_cur[new_pos] = feat.valid_y(yc)
        cleaned[new_pos] = True
        p_at_clean[new_pos] = p[new_local]
        order.extend(int(i) for i in new_pos)
        cleaned_step.extend([t] * len(new_pos))
        used += len(new_pos)

        # Updater
        draw_pos = cpos[draws]
        draw_coef = (1.0 / (len(draws) * Nd * p[draws])) * valid_cur[draw_pos]
        step = {'Nd': Nd, 'Nc': Nc, 'draw_pos': draw_pos, 'draw_coef': draw_coef,
                'clean_cleaned': clean_cleaned, 'clean_detected': clean_detected}
        steps.append(step)
        g = _step_gradient(model, W, step, Xa_cur, T_cur, Xa_d, T_d)
        W = W - (cfg.primary_step / L) / t * g

        # Detector and estimator updates from all cleaned records
        cl = np.array(order, dtype=int)
        raw_tags = np.array([tags[i] for i in cl], dtype=object)
        n_clean_tag = int(np.sum(raw_tags == CLEAN_TAG))
        n_dirty_tag = len(cl) - n_clean_tag
        if (n_clean_tag >= cfg.detector_min_per_class
                and n_dirty_tag >= cfg.detector_min_per_class):
            vals, cnts = np.unique(raw_tags, return_counts=True)
            rare = set(v for v, c in zip(vals, cnts)
                       if v != CLEAN_TAG and c < cfg.detector_rare_class)
            det_tags = np.array([('other_error' if v in rare else v) for v in raw_tags],
                                dtype=object)
            detector = AdaptiveDetector(cfg.detector_margin, cfg.seed).fit(Zd[cl], det_tags)
            deltas = {}
            for u in set(det_tags) - {CLEAN_TAG}:
                rows = cl[det_tags == u]
                w = 1.0 / p_at_clean[rows]
                w = w / w.sum()
                dx = (w[:, None] * (Xa_cur[rows] - Xa_d[rows])).sum(axis=0)
                dt = (w[:, None] * (T_cur[rows] - T_d[rows])).sum(axis=0)
                deltas[u] = (dx, dt)
        history.append({'t': t, 'Nd': Nd, 'Nc': Nc, 'n_detected_clean': len(clean_detected),
                        'draws': int(len(draws)), 'new_rows': int(len(new_pos)),
                        'new_dirty_rows': int(n_dirty_new), 'cleaned_total': int(used),
                        'detector_active_next': detector is not None,
                        'gamma_primary': (cfg.primary_step / L) / t,
                        'grad_norm': float(np.linalg.norm(g))})
        say('[ActiveClean] t=%d |R_dirty|=%d |R_clean|=%d new=%d (dirty %d) cleaned=%d/%d'
            % (t, Nd, Nc, len(new_pos), n_dirty_new, used, budget))

    # Replay of the recorded iterations for every step-size constant; the clean validation
    # split picks the constant and the stopping iterate (convergence check)
    trajectories = {}
    for c in cfg.step_grid:
        Wc = W0.copy()
        traj = [Wc.copy()]
        for k, step in enumerate(steps, 1):
            Wc = Wc - (c / L) / k * _step_gradient(model, Wc, step, Xa_cur, T_cur, Xa_d, T_d)
            traj.append(Wc.copy())
        trajectories[c] = traj
    if steps:
        drift = float(np.max(np.abs(trajectories[cfg.primary_step][-1] - W)))
        if drift > 1e-6 * (1.0 + float(np.max(np.abs(W)))):
            raise RuntimeError('replay of the primary trajectory diverged from the online run')
    val_curves = {}
    best = None
    for c in cfg.step_grid:
        curve = [_val_score(model, Wt, Xva, yva, cfg.task) for Wt in trajectories[c]]
        val_curves[str(c)] = curve
        for k, s in enumerate(curve):
            if not np.isfinite(s):
                continue
            key = (round(s, 10), k, -abs(math.log(c / cfg.primary_step)))
            if best is None or key > best[0]:
                best = (key, c, k)
    if best is None:
        best = ((0, 0, 0), cfg.primary_step, 0)
    sel_c, sel_t = best[1], best[2]
    final = model.copy_with(trajectories[sel_c][sel_t])
    for h in history:
        h['val_primary'] = val_curves[str(cfg.primary_step)][h['t']]

    info = {
        'dirty_model': fit_info,
        'n_features': feat.n_features,
        'n_outputs': K,
        'smoothness_L': L,
        'iterations': len(steps),
        'rows_cleaned': int(used),
        'rows_cleaned_dirty': int(sum(1 for i in order if tags[i] != CLEAN_TAG)),
        'selected_step_constant': sel_c,
        'selected_iterate': sel_t,
        'val_score_selected': val_curves[str(sel_c)][sel_t],
        'val_score_dirty_model': val_curves[str(sel_c)][0],
        'val_score_primary_last': val_curves[str(cfg.primary_step)][-1],
        'val_curves': val_curves,
        'detector_active_at_end': detector is not None,
        'detector_classes': (detector.classes_ if detector is not None else []),
        'tag_counts': dict((str(k), int(v)) for k, v in
                           pd.Series([tags[i] for i in order]).value_counts().items())
        if order else {},
        'secs_core': time.time() - t_start,
    }
    return ActiveCleanResult(feat, final, order, cleaned_step, history, info)
