"""Comet: step-by-step, feature-wise data cleaning recommendations.

Reimplementation of S. Mohammed, F. Naumann, H. Harmouch, "Step-by-Step Data
Cleaning Recommendations to Improve ML Prediction Accuracy", EDBT 2025
(arXiv 2503.11366).  The paper text is the specification.  The authors' code
(github.com/sedirmohammed/COMET, folders single_error_scenario and
multi_error_scenario) is used to fix the details that the paper leaves open.

Modules (paper Section 3)
-------------------------
Polluter (3.1)     For a candidate (feature, error type) pair it injects the error
                   type into one more pollution step (1% of the training rows) and
                   then into a second one, cumulatively, for ten random row
                   combinations.  Error types (3.4): missing values, Gaussian noise
                   (std drawn from U[1, 5] times the std of the sampled values),
                   scaling by a factor drawn from {10, 100, 1000}, categorical
                   shift (the sampled categories are permuted so that every entry
                   changes when possible).
Estimator (3.2)    Fits the measured scores at pollution levels 0, 1 and 2 steps
                   with the authors' Bayesian degree-2 polynomial regression and
                   extrapolates it to one cleaning step (level -1 step).  The
                   authors sample the posterior with NUTS.  Their model has a fixed
                   observation noise and Gaussian priors, so the posterior is
                   Gaussian; this module computes it in closed form, which is the
                   distribution the sampler targets.
Recommender (3.3)  (A) keeps the pairs whose predicted score after one cleaning
                   step exceeds the current score, (B) ranks them by
                   (P_next - U) / C with U the width of the 95% credible interval,
                   (C) cleans the best pair by one step, (D) reverts the step if the
                   score decreases and keeps the cleaned cells in a buffer (applying
                   a buffered step later costs nothing), (E) falls back to the
                   attempt with the highest score after cleaning when no attempt
                   improved, or, when no pair was predicted to improve, to the pair
                   with the highest score measured after its last cleaning.  After
                   every cleaning the discrepancy between measured and predicted
                   score is stored and its running mean is added to later
                   predictions of the same pair.
Cleaner            Simulated with ground truth, as in the authors' SimulationCleaning:
                   it repairs one step of dirty cells of the recommended feature and
                   error type, first among the rows the Polluter sampled, then
                   the other dirty rows from the end of the row order.  The
                   authors' code skips the subsampling to one step when few dirty
                   rows remain outside the samples, which can clean more than one
                   step at once; this module always cleans at most one step.

Details taken from the authors' code: the ten pollution seeds, the 2% row sample
split into two halves for the two pollution levels, the ceil(1%) cleaning step,
the prior and noise of the Bayesian model, the 2.5/97.5 percentiles, the score
without a positive-gain filter in the code (the paper's filter (A) is applied
here), the acceptance test (score after cleaning >= current score), budget spent
on reverted attempts, the error-correction term (mean of measured minus adjusted
prediction), and the constant/one-shot/linear cost functions of Section 4.2.

This module contains no IO, no paths, no argument parsing and no test-set
evaluation.  The caller supplies encoded matrices, the model factory and the
scoring function.
"""

import math
import random
import time

import numpy as np

MISSING_VALUES = 'MissingValues'
GAUSSIAN_NOISE = 'GaussianNoise'
SCALING = 'Scaling'
CATEGORICAL_SHIFT = 'CategoricalShift'

# Integer codes used in error-type matrices (0 marks a clean cell).
ERROR_CODES = {MISSING_VALUES: 1, GAUSSIAN_NOISE: 2, SCALING: 3, CATEGORICAL_SHIFT: 4}
ERROR_NAMES = {v: k for k, v in ERROR_CODES.items()}
NUMERICAL_ERROR_TYPES = (MISSING_VALUES, GAUSSIAN_NOISE, SCALING)
CATEGORICAL_ERROR_TYPES = (MISSING_VALUES, CATEGORICAL_SHIFT)

# Seeds of the ten pollution combinations (authors' artifical_pollution_new.py).
POLLUTER_SEEDS = (87263, 53219, 78604, 2023, 38472, 11, 9834, 4567, 909090, 56789)
SCALING_FACTORS = (10, 100, 1000)
Z_975 = 1.959963984540054


# ---------------------------------------------------------------------------
# Error typing of real dirty cells (used to build the Cleaner's error map)
# ---------------------------------------------------------------------------
def classify_errors(dirty_mask, dirty_raw, clean_raw, is_categorical, scale_tolerance=0.05):
    """Assign one of Comet's four error types to every dirty cell.

    dirty_mask      bool (n, d), True where the dirty cell differs from ground truth
    dirty_raw       float (n, d), dirty cells as the downstream encoder sees them
                    (category codes for categorical columns, numbers otherwise,
                    NaN for missing or unparseable values), before imputation
    clean_raw       float (n, d), the ground-truth cells encoded the same way
    is_categorical  bool (d,)

    Rules: a dirty cell that the encoder turns into NaN is a missing value; any
    other dirty categorical cell is a categorical shift; a numerical cell whose
    ratio to the clean value is 10^k with |k| in {1, 2, 3} (tolerance
    scale_tolerance on log10) is a scaling error; the remaining numerical cells
    are Gaussian noise.  Returns an int8 matrix of ERROR_CODES (0 = clean).
    """
    dirty_mask = np.asarray(dirty_mask, dtype=bool)
    codes = np.zeros(dirty_mask.shape, dtype=np.int8)
    for j in range(dirty_mask.shape[1]):
        rows = np.flatnonzero(dirty_mask[:, j])
        for r in rows:
            dv = dirty_raw[r, j]
            if np.isnan(dv):
                codes[r, j] = ERROR_CODES[MISSING_VALUES]
            elif is_categorical[j]:
                codes[r, j] = ERROR_CODES[CATEGORICAL_SHIFT]
            else:
                cv = clean_raw[r, j]
                code = ERROR_CODES[GAUSSIAN_NOISE]
                if not np.isnan(cv) and cv != 0.0 and dv != 0.0:
                    lr = math.log10(abs(dv / cv))
                    k = int(round(lr))
                    if 1 <= abs(k) <= 3 and abs(lr - k) <= scale_tolerance:
                        code = ERROR_CODES[SCALING]
                codes[r, j] = code
    return codes


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------
class BayesianPolynomialEstimator:
    """Closed-form posterior of the authors' PyMC model (regression_model.predict_test).

    mu(x) = alpha + beta_0 + beta_1 x + beta_2 x^2,
    alpha ~ N(y_first, 0.1^2), beta_k ~ N(1, 10^2),
    y_i ~ N(mu(x_i), sigma_obs^2) with sigma_obs = sqrt((1e-10)^4 + 0.01).
    predict returns the posterior mean of mu(x_star) and the central interval
    with probability `ci` (the authors take the 2.5 and 97.5 percentiles of the
    sampled mu(x_star)).
    """

    def __init__(self, degree=2, alpha_sd=0.1, beta_mean=1.0, beta_sd=10.0,
                 sigma_obs=math.sqrt(1e-40 + 0.01), z=Z_975):
        self.degree = degree
        self.alpha_sd = alpha_sd
        self.beta_mean = beta_mean
        self.beta_sd = beta_sd
        self.sigma_obs = sigma_obs
        self.z = z

    def _design(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        cols = [np.ones_like(x)] + [x ** k for k in range(self.degree + 1)]
        return np.column_stack(cols)

    def predict(self, x, y, x_star):
        y = np.asarray(y, dtype=float).reshape(-1)
        phi = self._design(x)
        p = self.degree + 1
        prior_mean = np.concatenate([[y[0]], np.full(p, self.beta_mean)])
        prior_prec = np.concatenate([[1.0 / self.alpha_sd ** 2], np.full(p, 1.0 / self.beta_sd ** 2)])
        inv_var = 1.0 / self.sigma_obs ** 2
        precision = np.diag(prior_prec) + inv_var * phi.T.dot(phi)
        rhs = prior_prec * prior_mean + inv_var * phi.T.dot(y)
        cov = np.linalg.inv(precision)
        mean = cov.dot(rhs)
        f = self._design([x_star])[0]
        m = float(f.dot(mean))
        s = float(math.sqrt(max(f.dot(cov).dot(f), 0.0)))
        return m, m - self.z * s, m + self.z * s


# ---------------------------------------------------------------------------
# Polluter
# ---------------------------------------------------------------------------
class Polluter:
    """Row samples and error injection of the Polluter (paper 3.1, authors' execute_pollution).

    For every seed the authors draw int(n * step * n_steps) distinct rows and split
    them into n_steps consecutive parts, one per pollution level; the levels are
    cumulative.  The same samples are used for every feature and iteration because
    they depend only on the number of rows and the seed.
    """

    def __init__(self, n_rows, step_fraction=0.01, n_steps=2, seeds=POLLUTER_SEEDS):
        self.n_rows = n_rows
        self.levels = tuple(step_fraction * (i + 1) for i in range(n_steps))
        size = int(n_rows * step_fraction * n_steps)
        self.samples = []
        for seed in seeds:
            rows = np.random.RandomState(seed).choice(n_rows, size=size, replace=False)
            bounds = [int(size * i / n_steps) for i in range(n_steps + 1)]
            # pandas DataFrame.sample(n, random_state=seed) draws RandomState(seed).choice(n_rows, n,
            # replace=False); the authors keep the draw order, which fixes which row gets which noise draw
            self.samples.append((seed, [rows[bounds[i]:bounds[i + 1]] for i in range(n_steps)]))
        cand = set()
        for _, parts in self.samples:
            for part in parts:
                cand.update(part.tolist())
        self.candidate_rows = frozenset(cand)

    @staticmethod
    def inject(X, col, error_code, rows, rng):
        """Inject one error type into X[rows, col] in place (raw, unscaled values)."""
        if len(rows) == 0:
            return
        vals = X[rows, col].astype(float)
        if error_code == ERROR_CODES[MISSING_VALUES]:
            X[rows, col] = np.nan
        elif error_code == ERROR_CODES[GAUSSIAN_NOISE]:
            finite = vals[~np.isnan(vals)]
            std = float(np.std(finite)) if len(finite) else 0.0
            scale = rng.uniform(1, 5)
            noise = rng.normal(0, scale * std, size=len(rows))
            X[rows, col] = vals + noise
        elif error_code == ERROR_CODES[SCALING]:
            factor = rng.choice(SCALING_FACTORS)
            X[rows, col] = vals * factor
        elif error_code == ERROR_CODES[CATEGORICAL_SHIFT]:
            # authors' ModifiedCategoricalShift: shuffle the sampled values and, where
            # a value would stay unchanged, take the first shuffled value that differs
            token = np.where(np.isnan(vals), np.inf, vals)
            shuffled = token.copy()
            rng.shuffle(shuffled)
            out = shuffled.copy()
            for i in range(len(token)):
                if shuffled[i] == token[i]:
                    avail = shuffled[shuffled != token[i]]
                    out[i] = avail[0] if len(avail) else token[i]
            out[np.isinf(out)] = np.nan
            X[rows, col] = out
        else:
            raise ValueError('unknown error code %r' % (error_code,))


# ---------------------------------------------------------------------------
# Model fitting helpers (module level so that joblib can pickle them)
# ---------------------------------------------------------------------------
def prepare_matrix(X_raw, center, scale):
    """Mean-impute NaN per column (0.0 for all-NaN columns), then standardize.

    Mirrors the benchmark encoder: raw NaN cells get the column mean of the
    matrix being encoded, then the fixed scaler fitted on the dirty training split
    is applied.
    """
    X = np.array(X_raw, dtype=float, copy=True)
    nan = np.isnan(X)
    if nan.any():
        counts = X.shape[0] - nan.sum(axis=0)
        X[nan] = 0.0
        sums = X.sum(axis=0)
        means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
        np.copyto(X, np.broadcast_to(means, X.shape), where=nan)
    X -= center
    X /= scale
    return X


def score_matrix(X_raw, ctx):
    model = ctx['model_factory']()
    X = prepare_matrix(X_raw, ctx['center'], ctx['scale'])
    model.fit(X[ctx['fit_rows']], ctx['y_fit'])
    pred = model.predict(ctx['X_val'])
    return float(ctx['score_fn'](ctx['y_val'], pred)), pred


def pollution_task(X_cur, col, error_code, parts, seed, ctx):
    """Scores after 1, 2, ... cumulative pollution steps for one seed."""
    rng = np.random.RandomState(seed)
    X = np.array(X_cur, dtype=float, copy=True)
    scores = []
    for rows in parts:
        Polluter.inject(X, col, error_code, rows, rng)
        scores.append(score_matrix(X, ctx)[0])
    return scores


# ---------------------------------------------------------------------------
# Cost models (paper 4.2, authors' CostFunction)
# ---------------------------------------------------------------------------
def step_cost(cost_model, error_code, n_steps_before):
    """Cost of the next cleaning step of a pair that was cleaned n_steps_before times."""
    if cost_model == 'constant':
        return 1
    if cost_model == 'multi_error':
        if error_code == ERROR_CODES[MISSING_VALUES]:
            return 2 if n_steps_before == 0 else 0      # one-shot
        if error_code == ERROR_CODES[GAUSSIAN_NOISE]:
            return n_steps_before + 1                   # linear
        return 1                                        # constant (shift, scaling)
    raise ValueError('unknown cost model %r' % (cost_model,))


# ---------------------------------------------------------------------------
# Comet
# ---------------------------------------------------------------------------
class CometResult:
    def __init__(self):
        self.cleaned_cells = []          # (row, col) cells set to ground truth, in order
        self.consumed_cells = []         # every cell whose ground truth was used (incl. reverted)
        self.trace = []                  # one dict per iteration
        self.steps = []                  # applied steps
        self.stop_reason = None
        self.cost_used = 0
        self.iterations = 0
        self.attempts = 0
        self.initial_score = None
        self.final_score = None
        self.final_report = None
        self.secs = 0.0


class Comet:
    """Comet recommendation loop with a simulated ground-truth Cleaner.

    model_factory   callable returning an unfitted model with fit(X, y) and predict(X)
    score_fn        callable (y_true, y_pred) -> float, the prediction metric Comet
                    maximizes (the paper uses macro F1; any higher-is-better score)
    report_fn       optional callable (y_true, y_pred) -> dict logged for the current
                    state of every iteration
    step_fraction   cleaning and pollution step as a fraction of the training rows
    cost_budget     budget in cost units (paper: 50); None for no cost limit
    gt_budget       budget in ground-truth cells; None for no cell limit.  A step
                    is truncated to the remaining cells.
    cost_model      'constant' (1 unit per step, the paper's single-error setting)
                    or 'multi_error' (one-shot / linear / constant, paper 4.2)
    select_positives  paper step (A); False reproduces the authors' code, which
                    ranks all pairs
    n_jobs          joblib workers for the pollution fits (1 = in process)
    target_dtype    int for class labels (the paper's setting), float for real-valued
                    targets when the utility is a regression score such as R^2
                    (extension beyond the paper, which covers classification only)
    """

    def __init__(self, model_factory, score_fn, report_fn=None, step_fraction=0.01,
                 pollution_steps=2, seeds=POLLUTER_SEEDS, cost_budget=50, gt_budget=None,
                 cost_model='constant', select_positives=True, uncertainty_weight=1.0,
                 cleaner_seed=42, n_jobs=1, log=None, target_dtype=int):
        if cost_budget is None and gt_budget is None:
            raise ValueError('need a cost budget or a ground-truth budget')
        self.model_factory = model_factory
        self.score_fn = score_fn
        self.report_fn = report_fn
        self.step_fraction = step_fraction
        self.pollution_steps = pollution_steps
        self.seeds = tuple(seeds)
        self.cost_budget = cost_budget
        self.gt_budget = gt_budget
        self.cost_model = cost_model
        self.select_positives = select_positives
        self.uncertainty_weight = uncertainty_weight
        self.cleaner_seed = cleaner_seed
        self.n_jobs = n_jobs
        self.log = log or (lambda msg: None)
        self.target_dtype = target_dtype
        self.estimator = BayesianPolynomialEstimator()

    # -- budget helpers ---------------------------------------------------
    def _budget_left(self, cost_used, gt_used):
        left = []
        if self.cost_budget is not None:
            left.append(self.cost_budget - cost_used)
        if self.gt_budget is not None:
            left.append(self.gt_budget - gt_used)
        return min(left)

    # -- main loop ----------------------------------------------------------
    def run(self, X_dirty, X_clean, y_train, X_val, y_val, error_codes, is_categorical,
            center, scale, feature_names=None):
        """Run Comet on the training split.

        X_dirty, X_clean  raw (unscaled, NaN = missing) encoded training matrices
        y_train           encoded training labels (NaN rows are not used for fitting)
        X_val, y_val      prepared (imputed, standardized) validation matrix and labels
        error_codes       int matrix from classify_errors (0 = clean)
        center, scale     the fixed standardization fitted on the dirty training split
        """
        t_start = time.time()
        X_dirty = np.asarray(X_dirty, dtype=float)
        X_clean = np.asarray(X_clean, dtype=float)
        n, d = X_dirty.shape
        names = list(feature_names) if feature_names is not None else ['f%d' % j for j in range(d)]
        y_train = np.asarray(y_train, dtype=float)
        fit_rows = np.flatnonzero(~np.isnan(y_train))
        ctx = {'model_factory': self.model_factory, 'score_fn': self.score_fn,
               'center': np.asarray(center, dtype=float), 'scale': np.asarray(scale, dtype=float),
               'fit_rows': fit_rows, 'y_fit': y_train[fit_rows].astype(self.target_dtype),
               'X_val': np.asarray(X_val, dtype=float), 'y_val': np.asarray(y_val).astype(self.target_dtype)}

        # candidate (feature, error type) pairs with dirty cells
        remaining = {}
        order = []
        for j in range(d):
            allowed = CATEGORICAL_ERROR_TYPES if is_categorical[j] else NUMERICAL_ERROR_TYPES
            allowed_codes = [ERROR_CODES[t] for t in allowed]
            present = set(np.unique(error_codes[:, j]).tolist()) - {0}
            if present - set(allowed_codes):
                raise ValueError('column %s has error types not allowed for its type' % names[j])
            for code in allowed_codes:
                rows = np.flatnonzero(error_codes[:, j] == code)
                if len(rows):
                    remaining[(j, code)] = set(rows.tolist())
                    order.append((j, code))
        rank_of = {p: i for i, p in enumerate(order)}

        polluter = Polluter(n, self.step_fraction, self.pollution_steps, self.seeds)
        levels = polluter.levels
        step_cells = int(math.ceil(round(n * self.step_fraction, 9)))

        X = X_dirty.copy()
        res = CometResult()
        buffer = {}
        err_hist = {}
        last_after = {}
        n_cleaned_steps = {p: 0 for p in order}
        cost_used = 0
        gt_used = 0

        def pair_label(p):
            return '%s/%s' % (names[p[0]], ERROR_NAMES[p[1]])

        def select_rows(p, k):
            # authors' SimulationCleaning: dirty rows among the Polluter's samples first,
            # random.seed(42) subsample when there are more than k of them, otherwise top up
            # with the remaining dirty rows taken from the end of the index-ordered list (pop())
            dirty = sorted(remaining[p])
            in_cand = [r for r in dirty if r in polluter.candidate_rows]
            if len(in_cand) >= k:
                return sorted(random.Random(self.cleaner_seed).sample(in_cand, k))
            others = [r for r in dirty if r not in polluter.candidate_rows]
            need = min(k - len(in_cand), len(others))
            extra = others[len(others) - need:] if need > 0 else []
            return sorted(in_cand + extra)

        iteration = 0
        while True:
            t_it = time.time()
            active = [p for p in order if remaining.get(p)]
            if not active:
                res.stop_reason = 'no dirty cells left in candidate features'
                break
            if self._budget_left(cost_used, gt_used) <= 0:
                res.stop_reason = 'budget exhausted'
                break
            iteration += 1
            f1_cur, pred_cur = score_matrix(X, ctx)
            if res.initial_score is None:
                res.initial_score = f1_cur
            report_cur = self.report_fn(ctx['y_val'], pred_cur) if self.report_fn else None

            # Polluter + Estimator for every active pair
            tasks = [(p, seed, parts) for p in active for seed, parts in polluter.samples]
            if self.n_jobs == 1:
                outs = [pollution_task(X, p[0], p[1], parts, seed, ctx) for p, seed, parts in tasks]
            else:
                from joblib import Parallel, delayed
                outs = Parallel(n_jobs=self.n_jobs)(
                    delayed(pollution_task)(X, p[0], p[1], parts, seed, ctx) for p, seed, parts in tasks)
            per_pair = {p: [] for p in active}
            for (p, seed, parts), out in zip(tasks, outs):
                per_pair[p].append(out)
            est = {}
            for p in active:
                xs, ys = [0.0] * len(self.seeds), [f1_cur] * len(self.seeds)
                for out in per_pair[p]:
                    for lvl, s in zip(levels, out):
                        xs.append(lvl)
                        ys.append(s)
                m, lo, hi = self.estimator.predict(xs, ys, -self.step_fraction)
                adj = float(np.mean(err_hist[p])) if err_hist.get(p) else 0.0
                p_adj = m + adj
                c_next = 0 if p in buffer else step_cost(self.cost_model, p[1], n_cleaned_steps[p])
                score = (p_adj - self.uncertainty_weight * (hi - lo)) / (c_next if c_next > 0 else 1)
                est[p] = {'pair': pair_label(p), 'feature': names[p[0]], 'error_type': ERROR_NAMES[p[1]],
                          'n_dirty_left': len(remaining[p]),
                          'mean_f1_per_level': [float(np.mean([o[i] for o in per_pair[p]])) for i in range(len(levels))],
                          'p_next': m, 'lower': lo, 'upper': hi, 'adjustment': adj, 'p_adj': p_adj,
                          'score': score, 'positive': bool(p_adj > f1_cur), 'buffered': p in buffer}

            attempts = []

            def attempt(p):
                """Clean one step of pair p (or reuse its buffered step) and score it."""
                nonlocal cost_used, gt_used
                if self._budget_left(cost_used, gt_used) <= 0:
                    return None
                if p in buffer:
                    rows, new = buffer[p], False
                else:
                    cost = step_cost(self.cost_model, p[1], n_cleaned_steps[p])
                    if self.cost_budget is not None and cost_used + cost > self.cost_budget:
                        return None
                    k = min(step_cells, len(remaining[p]))
                    if self.gt_budget is not None:
                        k = min(k, self.gt_budget - gt_used)
                    if k <= 0:
                        return None
                    rows, new = select_rows(p, k), True
                    cost_used += cost
                    gt_used += len(rows)
                    n_cleaned_steps[p] += 1
                    res.consumed_cells.extend((r, p[0]) for r in rows)
                X_try = X.copy()
                X_try[rows, p[0]] = X_clean[rows, p[0]]
                f1_after = score_matrix(X_try, ctx)[0]
                err_hist.setdefault(p, []).append(f1_after - est[p]['p_adj'])
                last_after[p] = f1_after
                rec = {'pair': p, 'rows': rows, 'new': new, 'f1_after': f1_after}
                attempts.append(rec)
                res.attempts += 1
                return rec

            candidates = [p for p in active if est[p]['positive']] if self.select_positives else list(active)
            candidates.sort(key=lambda p: (-est[p]['score'], rank_of[p]))
            accepted, mode = None, None
            for p in candidates:
                rec = attempt(p)
                if rec is None:
                    continue
                if rec['f1_after'] >= f1_cur:
                    accepted, mode = rec, 'improved'
                    break
                buffer[p] = rec['rows']
            if accepted is None and attempts:
                accepted = max(attempts, key=lambda r: r['f1_after'])
                mode = 'fallback_best_attempt'
            if accepted is None:
                fb = sorted(active, key=lambda p: (0 if p in last_after else 1, -last_after.get(p, 0.0),
                                                   -est[p]['score'], rank_of[p]))
                for p in fb:
                    rec = attempt(p)
                    if rec is not None:
                        accepted, mode = rec, 'fallback_history'
                        break
            if accepted is None:
                res.stop_reason = 'budget exhausted'
                res.trace.append(self._trace_row(iteration, f1_cur, report_cur, est, active, attempts,
                                                 None, None, cost_used, gt_used, res, t_it, pair_label))
                break

            p = accepted['pair']
            rows = accepted['rows']
            X[rows, p[0]] = X_clean[rows, p[0]]
            remaining[p] -= set(rows)
            buffer.pop(p, None)
            res.cleaned_cells.extend((r, p[0]) for r in rows)
            res.steps.append({'iteration': iteration, 'pair': pair_label(p), 'feature': names[p[0]],
                              'error_type': ERROR_NAMES[p[1]], 'cells': len(rows), 'mode': mode,
                              'f1_before': f1_cur, 'f1_after': accepted['f1_after']})
            res.trace.append(self._trace_row(iteration, f1_cur, report_cur, est, active, attempts,
                                             accepted, mode, cost_used, gt_used, res, t_it, pair_label))
            self.log('iter %d  f1 %.4f -> %.4f  %s %s cells=%d  cost=%s gt_used=%d applied=%d  active=%d  %.1fs'
                     % (iteration, f1_cur, accepted['f1_after'], mode, pair_label(p), len(rows), cost_used,
                        gt_used, len(res.cleaned_cells), len(active), time.time() - t_it))

        final, pred = score_matrix(X, ctx)
        res.final_score = final
        res.final_report = self.report_fn(ctx['y_val'], pred) if self.report_fn else None
        if res.initial_score is None:
            res.initial_score = final
        res.cost_used = cost_used
        res.iterations = iteration
        res.secs = time.time() - t_start
        return res

    @staticmethod
    def _trace_row(iteration, f1_cur, report_cur, est, active, attempts, accepted, mode,
                   cost_used, gt_used, res, t_it, pair_label):
        return {
            'iteration': iteration,
            'score_current': f1_cur,
            'report_current': report_cur,
            'n_active_pairs': len(active),
            'estimates': [est[p] for p in active],
            'attempts': [{'pair': pair_label(a['pair']), 'cells': len(a['rows']), 'new': a['new'],
                          'f1_after': a['f1_after']} for a in attempts],
            'applied': pair_label(accepted['pair']) if accepted else None,
            'mode': mode,
            'cost_used': cost_used,
            'gt_cells_consumed': gt_used,
            'gt_cells_applied': len(res.cleaned_cells),
            'secs': time.time() - t_it,
        }
