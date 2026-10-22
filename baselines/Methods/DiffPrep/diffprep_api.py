"""DiffPrep adapter for the DemandClean benchmark.

DiffPrep: Differentiable Data Preprocessing Pipeline Search for Learning over Tabular Data,
Peng Li, Zhiyi Chen, Xu Chu, Kexin Rong, SIGMOD 2023.
Upstream code: https://github.com/chu-data-lab/diffprep, commit 6499edf ("init"), not bundled
(no license published). Clone it unchanged into ./diffprep_official (LICENSE_NOTE.md).

What runs: DiffPrep-Fix (the `--method` default of upstream main.py) with the logistic-regression
end model (`--model log`, the default), the default search space `prep_space.space`
(missing-value imputation, normalization, outlier cleaning, discretization, in this fixed order),
and the hyper-parameters of upstream main.py (2000 epochs, batch 512, SGD momentum 0.9,
lr grid {0.1, 0.01, 0.001} selected by validation loss, patience 3 checked every 100 epochs,
temperature 0.1, pipeline update sample size 512, no Gumbel sampling, default init).

This module holds the method only. It parses no arguments, reads and writes no files and computes
no test metric. The caller passes the training and validation tables; the test table reaches the
method only through `DiffPrepResult.features` after the search has finished.

Differences from upstream `experiment/diffprep_experiment.py` (they are also returned in
`DEVIATIONS` so the runner can record them):
  1. The caller supplies the split. Upstream draws its own random 60/20/20 split.
  2. `remove_large_cat` and `remove_all_na` look at the training rows only. Upstream runs
     `remove_large_cat` on all rows before splitting.
  3. The test table is not passed to `DiffPrepFixPipeline.init_parameters`. The validation table
     stands in for it, so the decision whether the imputation step exists uses training and
     validation rows only. The test cache of the first transformer is filled after the search.
  4. The training loop is upstream `DiffPrepSGD.fit` without the per-epoch test pass. Upstream
     evaluates the test set every epoch and only records it, so model selection is unchanged.
     The loop also keeps a deep copy of the fitted transformation functions at the best epoch,
     because upstream `state_dict` holds only the probabilities and the end-model weights.
  5. Labels are encoded with a LabelEncoder fitted on training and validation labels. Upstream
     fits it on all labels, test included.
  6. Regression (not covered by the paper): the end model is a linear layer with one output,
     the loss is MSELoss on the target standardized with training mean and std, and the
     validation "accuracy" that drives early stopping is R^2. Everything else is unchanged.

Optional stabilization (off by default, every switch is reported in `stabilization_deviations`).
It exists because the regression extension diverges on datasets whose dirty training rows hold
extreme values (nasa, mercedes, soilmoisture):
  grad_clip       upstream main.py declares `--grad_clip` but the trainer never reads it. When set,
                  the end-model gradient is clipped to this L2 norm before every SGD step (the real
                  steps and the virtual step of the bi-level update) and the gradient of the pipeline
                  probabilities is clipped to the same norm before every Adam step.
  sanitize_alpha  non-finite entries of the pipeline-probability gradient (they come from the finite
                  difference Hessian product once the end model has blown up) are set to 0 before
                  the Adam step, so one bad step cannot turn the probability logits into NaN.
  input_clip      the pipeline output is clamped to [-input_clip, input_clip] before it enters the
                  end model, also in the exported features. Upstream Normalizer already clamps its
                  output to [-1e10, 1e10]; this tightens that bound.
  float64         all tensors in float64 instead of float32.
"""

import os
import sys
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.autograd import Variable
from torch.distributions.utils import logits_to_probs

_UPSTREAM = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diffprep_official')
if _UPSTREAM not in sys.path:
    sys.path.insert(0, _UPSTREAM)

from model import LogisticRegression  # noqa: E402  (upstream model.py)
from pipeline.diffprep_fix_pipeline import DiffPrepFixPipeline  # noqa: E402
from prep_space import space as DEFAULT_SPACE  # noqa: E402
from trainer.diffprep_trainer import DiffPrepSGD  # noqa: E402

UPSTREAM_COMMIT = '6499edf'

DEVIATIONS = [
    'split supplied by the benchmark (60/20/20, seed 42, train rows dirty, val/test rows clean) '
    'instead of the upstream random split',
    'remove_large_cat and remove_all_na computed on training rows only (upstream: all rows)',
    'test rows not passed to init_parameters; imputation step presence decided on train+val; '
    'test cache filled after the search',
    'no per-epoch test evaluation (upstream only records it, selection unchanged); best epoch '
    'snapshot also stores the fitted transformation functions',
    'label encoder fitted on train+val labels (upstream: all labels)',
]
REGRESSION_DEVIATION = ('regression extension (not in the paper): linear head with 1 output, '
                        'MSELoss on standardized target, R^2 replaces accuracy in early stopping')

# Hyper-parameters of upstream main.py (params + auto_prep_params) for --method diffprep_fix --model log.
DEFAULT_PARAMS = {
    'num_epochs': 2000,
    'batch_size': 512,
    'device': 'cpu',
    'model_lr': [0.1, 0.01, 0.001],
    'weight_decay': 0,
    'model': 'log',
    'train_seed': 1,
    'split_seed': 1,
    'method': 'diffprep_fix',
    'save_model': True,
    'logging': False,
    'no_crash': False,
    'patience': 3,
    'momentum': 0.9,
    'prep_lr': None,
    'temperature': 0.1,
    'grad_clip': None,
    'pipeline_update_sample_size': 512,
    'init_method': 'default',
    'diff_method': 'num_diff',
    'sample': False,
    # stabilization switches (not in upstream main.py except grad_clip, which upstream never uses)
    'sanitize_alpha': False,
    'input_clip': None,
    'float64': False,
}


def stabilization_deviations(p):
    out = []
    if p.get('grad_clip'):
        out.append('gradient clipping wired up (upstream declares grad_clip but never uses it): end-model gradient '
                   'L2 norm clipped to {} before every SGD step incl. the virtual step, pipeline-probability gradient '
                   'clipped to the same norm before every Adam step'.format(p['grad_clip']))
    if p.get('sanitize_alpha'):
        out.append('non-finite entries of the pipeline-probability gradient set to 0 before the Adam step')
    if p.get('input_clip'):
        out.append('pipeline output clamped to [-{0}, {0}] before the end model and in the exported features '
                   '(upstream Normalizer clamps to [-1e10, 1e10])'.format(p['input_clip']))
    if p.get('float64'):
        out.append('tensors in float64 instead of float32')
    return out


# ----------------------------------------------------------------------------------------------
# Verbatim copies of upstream experiment/experiment_utils.py helpers (commit 6499edf).
# ----------------------------------------------------------------------------------------------
def remove_large_cat_columns(X):
    """Upstream remove_large_cat: drop categorical columns with over 1000 categories."""
    large_cat_columns = []
    cat_columns = X.select_dtypes(exclude='number').columns
    for c in cat_columns:
        n_cat = len(set(X[c].dropna().values))
        if n_cat > 1000:
            large_cat_columns.append(c)
    return [c for c in X.columns if c not in large_cat_columns]


def all_na_columns(X_train):
    """Upstream remove_all_na: columns that are entirely missing in the training rows."""
    is_all_na = X_train.isnull().values.all(axis=0)
    return list(X_train.columns[is_all_na])


def set_random_seed(params):
    random_state = params['train_seed']
    torch.manual_seed(random_state)
    np.random.seed(random_state)


def get_param_candidates(param_grid):
    from itertools import product
    fixed_params, tuned_params, candidate_params = {}, {}, []
    for name, parameter in param_grid.items():
        if type(parameter) == list:
            tuned_params[name] = parameter
        else:
            fixed_params[name] = parameter
    for tuned_params_cand in product(*tuned_params.values()):
        param_cand = deepcopy(fixed_params)
        for n, p in zip(tuned_params.keys(), tuned_params_cand):
            param_cand[n] = p
        candidate_params.append(param_cand)
    return candidate_params


# ----------------------------------------------------------------------------------------------
# End models and trainers
# ----------------------------------------------------------------------------------------------
class LinearRegressionHead(nn.Module):
    """Regression counterpart of upstream LogisticRegression: one linear output, squeezed."""

    def __init__(self, input_dim):
        super(LinearRegressionHead, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class DiffPrepTrainer(DiffPrepSGD):
    """Upstream DiffPrepSGD with fit() reproduced minus the test pass, plus a full best-epoch snapshot.

    forward_propagate, update_model and update_prep_pipeline are upstream code with the optional
    stabilization switches inserted (input_clip, grad_clip, sanitize_alpha). With the switches off
    they compute exactly what upstream computes.
    """

    def __init__(self, *args, log=None, **kwargs):
        super(DiffPrepTrainer, self).__init__(*args, **kwargs)
        self.log = log
        self.n_alpha_sanitized = 0
        self.n_alpha_sanitize_steps = 0

    def forward_propagate(self, X, y, X_type, require_transform_grad=False,
                          require_model_grad=False, max_only=False):
        with torch.set_grad_enabled(require_transform_grad):
            X_trans = self.prep_pipeline.transform(X, X_type=X_type, max_only=max_only, resample=False,
                                                   require_grad=require_transform_grad)
            c = self.params.get('input_clip')
            if c:
                X_trans = torch.clamp(X_trans, -c, c)
        if X_type == "train":
            self.model.train()
        else:
            self.model.eval()
        with torch.set_grad_enabled(require_model_grad or require_transform_grad):
            X_trans = X_trans.to(self.device)
            output = self.model(X_trans)
        y = y.to(self.device)
        loss = self.loss_fn(output, y)
        return output, loss

    def _clip_model_grad(self):
        c = self.params.get('grad_clip')
        if c:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), c)

    def update_model(self, X_train, y_train):
        self.model_optimizer.zero_grad()
        output_train, loss_train = self.forward_propagate(X_train, y_train, X_type='train',
                                                          require_model_grad=True)
        loss_train.backward()
        self._clip_model_grad()
        self.model_optimizer.step()
        _, preds = torch.max(output_train, 1)
        correct = torch.sum(preds.cpu() == y_train)
        return loss_train.item(), correct.item()

    def update_prep_pipeline(self, X_train, y_train, X_val, y_val):
        self.prep_pipeline_optimizer.zero_grad()
        dval_dalpha, dval_dw = self.compute_dval(X_train, y_train, X_val, y_val)
        hessian_product = self.compute_hessian_product(X_train, y_train, dval_dw)
        bad_step = False
        for i, alpha in enumerate(self.prep_pipeline.parameters()):
            dval = dval_dalpha[i]
            dtrain = hessian_product[i]
            dalpha = dval - self.model_optimizer.param_groups[0]['lr'] * dtrain
            if self.params.get('sanitize_alpha'):
                bad = ~torch.isfinite(dalpha)
                n_bad = int(bad.sum())
                if n_bad:
                    self.n_alpha_sanitized += n_bad
                    bad_step = True
                    dalpha = torch.where(bad, torch.zeros_like(dalpha), dalpha)
            if alpha.grad is None:
                alpha.grad = Variable(dalpha.data.clone())
            else:
                alpha.grad.data.copy_(dalpha.data.clone())
        self.n_alpha_sanitize_steps += int(bad_step)
        c = self.params.get('grad_clip')
        if c:
            torch.nn.utils.clip_grad_norm_(self.prep_pipeline.parameters(), c)
        self.prep_pipeline_optimizer.step()

    def snapshot(self):
        return {
            'prep_state': deepcopy(self.prep_pipeline.state_dict()),
            'model_state': deepcopy(self.model.state_dict()),
            'tf_options': [deepcopy(t.tf_options) for t in self.prep_pipeline.pipeline[1:]],
        }

    def fit(self, X_train, y_train, X_val, y_val, X_test=None, y_test=None, verbose=False):
        best_val_loss = float('inf')
        best_result, best_state = None, None
        last_best_val_acc = float('-inf')
        patience = self.params['patience']
        e = 0
        tic = time.time()
        while e < self.params['num_epochs']:
            self.global_step = e
            tr_loss, tr_acc = self.train(X_train, y_train, X_val, y_val)
            val_loss, val_acc = self.evaluate(X_val, y_val, X_type='val', max_only=False)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_result = {'best_epoch': e, 'best_val_loss': val_loss,
                               'best_tr_acc': tr_acc, 'best_val_acc': val_acc}
                best_state = self.snapshot()

            if e % 100 == 0:
                if best_result is None:
                    raise FloatingPointError('validation loss is not finite at epoch {}'.format(e))
                if best_result['best_val_acc'] - last_best_val_acc < 0.001:
                    patience = patience - 1
                else:
                    last_best_val_acc = best_result['best_val_acc']
                    patience = self.params['patience']
                if self.log is not None:
                    self.log('  epoch {:4d} tr_loss {:.4f} val_loss {:.4f} val_acc {:.4f} | best epoch {} '
                             'val_loss {:.4f} val_acc {:.4f} | patience {} | {:.0f}s'.format(
                                 e, tr_loss, val_loss, val_acc, best_result['best_epoch'],
                                 best_result['best_val_loss'], best_result['best_val_acc'],
                                 patience, time.time() - tic))
            if patience <= 0:
                break
            e += 1
        best_result['epochs_run'] = min(e + 1, self.params['num_epochs'])
        best_result['alpha_grad_entries_sanitized'] = self.n_alpha_sanitized
        best_result['alpha_steps_sanitized'] = self.n_alpha_sanitize_steps
        return best_result, best_state


class DiffPrepTrainerRegression(DiffPrepTrainer):
    """Regression variant: accuracy is replaced by R^2, everything else is DiffPrepTrainer."""

    def evaluate(self, X, y, X_type, max_only=True):
        output, loss = self.forward_propagate(X, y, X_type=X_type, max_only=max_only)
        out = output.detach().cpu()
        sse = torch.sum((out - y) ** 2).item()
        sst = torch.sum((y - y.mean()) ** 2).item()
        r2 = 1.0 - sse / sst if sst > 0 else 0.0
        return loss.item(), r2

    def update_model(self, X_train, y_train):
        self.model_optimizer.zero_grad()
        output_train, loss_train = self.forward_propagate(X_train, y_train, X_type='train',
                                                          require_model_grad=True)
        loss_train.backward()
        self._clip_model_grad()
        self.model_optimizer.step()
        return loss_train.item(), 0


# ----------------------------------------------------------------------------------------------
# Result object
# ----------------------------------------------------------------------------------------------
class DiffPrepResult(object):
    """Best pipeline and end model of the search; applies them to new rows after the search."""

    def __init__(self, task, pipeline, model, params, columns, dropped_columns, label_encoder,
                 y_mean, y_std, best, runs, deviations, secs):
        self.task = task
        self.pipeline = pipeline
        self.model = model
        self.params = params
        self.columns = columns
        self.dropped_columns = dropped_columns
        self.label_encoder = label_encoder
        self.y_mean = y_mean
        self.y_std = y_std
        self.best = best
        self.runs = runs
        self.deviations = deviations
        self.search_secs = secs
        self._n_new = 0

    @property
    def classes(self):
        return None if self.label_encoder is None else [str(c) for c in self.label_encoder.classes_]

    def encode_labels(self, y):
        """Class codes in the search's label space; labels unseen in train+val map to -1."""
        if self.label_encoder is None:
            return np.asarray(pd.to_numeric(pd.Series(y), errors='coerce'), dtype=float)
        lut = {c: i for i, c in enumerate(self.label_encoder.classes_)}
        return np.array([lut.get(str(v), -1) for v in y], dtype=np.int64)

    @property
    def n_features_out(self):
        return int(self.pipeline.out_features)

    def _x_type(self, X, x_type):
        if x_type in ('train', 'val'):
            cache = self.pipeline.pipeline[0].cache[x_type]
            cached = [v for v in (cache['X_num_trans'], cache['X_cat_trans']) if torch.is_tensor(v)]
            if cached and cached[0].shape[0] != len(X):
                raise ValueError('{} rows given, the search cached {} {} rows'.format(
                    len(X), cached[0].shape[0], x_type))
            return x_type
        key = 'new{}'.format(self._n_new)
        self._n_new += 1
        self.pipeline.pipeline[0].pre_cache(X[self.columns].reset_index(drop=True), key)
        return key

    def features(self, X, x_type, chunk=8192):
        """Return (soft, hard) feature matrices of X.

        soft: the probability-weighted pipeline output, the input DiffPrep's own model is trained,
              validated and tested on in upstream code (max_only=False).
        hard: every feature passes the transformation with the highest probability (max_only=True).
        x_type 'train' and 'val' reuse the rows given to the search; anything else is new rows.
        """
        X = X[self.columns].reset_index(drop=True)
        key = self._x_type(X, x_type)
        soft, hard = [], []
        with torch.no_grad():
            for a in range(0, len(X), chunk):
                part = X.iloc[a:a + chunk]
                soft.append(self.pipeline.transform(part, X_type=key, max_only=False,
                                                    resample=False, require_grad=False).numpy())
                hard.append(self.pipeline.transform(part, X_type=key, max_only=True,
                                                    resample=False, require_grad=False).numpy())
        soft, hard = np.concatenate(soft, 0), np.concatenate(hard, 0)
        c = self.params.get('input_clip')
        if c:
            soft, hard = np.clip(soft, -c, c), np.clip(hard, -c, c)
        return soft.astype(np.float32), hard.astype(np.float32)

    def predict(self, X_soft):
        """DiffPrep's end model on soft features: (predictions, class probabilities or None)."""
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.tensor(X_soft, dtype=torch.get_default_dtype()))
        if self.task == 'regression':
            return (out.numpy().astype(np.float64) * self.y_std + self.y_mean), None
        proba = torch.softmax(out, dim=1).numpy()
        return proba.argmax(1).astype(np.int64), proba.astype(np.float32)

    def describe_pipeline(self):
        """Per output feature: chosen transformation and its probability for each step."""
        first = self.pipeline.pipeline[0]
        names = list(first.feature_names)
        steps = {}
        if first.num_tf_prob_logits is not None:
            p = logits_to_probs(first.num_tf_prob_logits.detach()).numpy()
            for i in range(p.shape[0]):
                steps.setdefault(names[i], {})['missing_value_imputation'] = [
                    first.num_tf_methods[int(p[i].argmax())], round(float(p[i].max()), 4)]
        if first.cat_tf_prob_logits is not None:
            p = logits_to_probs(first.cat_tf_prob_logits.detach()).numpy()
            for i in range(p.shape[0]):
                steps.setdefault(names[i + first.out_num_features], {})['missing_value_imputation'] = [
                    first.cat_tf_methods[int(p[i].argmax())], round(float(p[i].max()), 4)]
        summary = {}
        for t in self.pipeline.pipeline[1:]:
            p = logits_to_probs(t.tf_prob_logits.detach()).numpy()
            counts = {}
            for i in range(p.shape[0]):
                m = t.tf_methods[int(p[i].argmax())]
                steps.setdefault(names[i], {})[t.name] = [m, round(float(p[i].max()), 4)]
                counts[m] = counts.get(m, 0) + 1
            summary[t.name] = counts
        imp = {}
        for name in names:
            m = steps.get(name, {}).get('missing_value_imputation')
            if m is not None:
                imp[m[0]] = imp.get(m[0], 0) + 1
        summary['missing_value_imputation'] = imp
        return {'feature_names': names, 'per_feature': steps, 'summary': summary}


# ----------------------------------------------------------------------------------------------
# Search
# ----------------------------------------------------------------------------------------------
class DiffPrepDiverged(RuntimeError):
    """No learning rate of the grid finished: non-finite validation loss at epoch 0, or a crash."""

    def __init__(self, runs):
        super(DiffPrepDiverged, self).__init__('every learning rate diverged: {}'.format(runs))
        self.runs = runs


def _encode_targets(task, y_train, y_val):
    if task == 'regression':
        ytr = pd.to_numeric(pd.Series(y_train), errors='coerce').values.astype(float)
        yva = pd.to_numeric(pd.Series(y_val), errors='coerce').values.astype(float)
        mean, std = float(np.mean(ytr)), float(np.std(ytr))
        std = std if std > 0 else 1.0
        return (torch.Tensor((ytr - mean) / std), torch.Tensor((yva - mean) / std), None, mean, std)
    enc = LabelEncoder().fit(np.concatenate([np.asarray(y_train, dtype=str), np.asarray(y_val, dtype=str)]))
    ytr = torch.tensor(enc.transform(np.asarray(y_train, dtype=str))).long()
    yva = torch.tensor(enc.transform(np.asarray(y_val, dtype=str))).long()
    return ytr, yva, enc, None, None


def _one_run(task, X_train, y_train, X_val, y_val, prep_space, params, n_classes, log):
    set_random_seed(params)
    prep_pipeline = DiffPrepFixPipeline(prep_space, temperature=params['temperature'],
                                        use_sample=params['sample'], diff_method=params['diff_method'],
                                        init_method=params['init_method'])
    # the validation table stands in for the test table (deviation 3)
    prep_pipeline.init_parameters(X_train, X_val, X_val)
    input_dim = prep_pipeline.out_features

    set_random_seed(params)
    if task == 'regression':
        model, loss_fn, trainer_cls = LinearRegressionHead(input_dim), nn.MSELoss(), DiffPrepTrainerRegression
    else:
        model, loss_fn, trainer_cls = LogisticRegression(input_dim, n_classes), nn.CrossEntropyLoss(), DiffPrepTrainer
    model = model.to(params['device'])

    model_optimizer = torch.optim.SGD(model.parameters(), lr=params['model_lr'],
                                      weight_decay=params['weight_decay'], momentum=params['momentum'])
    prep_lr = params['model_lr'] if params['prep_lr'] is None else params['prep_lr']
    prep_pipeline_optimizer = torch.optim.Adam(prep_pipeline.parameters(), lr=prep_lr,
                                               betas=(0.5, 0.999), weight_decay=params['weight_decay'])
    trainer = trainer_cls(prep_pipeline, model, loss_fn, model_optimizer, prep_pipeline_optimizer,
                          None, None, params, writer=None, log=log)
    result, state = trainer.fit(X_train, y_train, X_val, y_val)
    return result, state, prep_pipeline, model, trainer


def run_diffprep(X_train, y_train, X_val, y_val, task='classification', params=None, log=None):
    """Search a DiffPrep-Fix pipeline on the training rows, select with the validation rows.

    X_train, X_val: DataFrames with identical columns. Numeric columns have a numeric dtype and
        np.nan for missing cells, categorical columns have object dtype and np.nan for missing cells.
    y_train, y_val: label arrays (class labels, or numbers for regression) without missing values.
    task: 'classification' or 'regression'.
    params: overrides of DEFAULT_PARAMS.
    Returns a DiffPrepResult.
    """
    if task not in ('classification', 'regression'):
        raise ValueError('DiffPrep supports classification and regression only, got {}'.format(task))
    tic = time.time()
    log = log or (lambda *_: None)
    p = deepcopy(DEFAULT_PARAMS)
    p.update(params or {})
    torch.set_default_dtype(torch.float64 if p['float64'] else torch.float32)

    # upstream build_data column filters, computed on the training rows (deviation 2)
    keep = remove_large_cat_columns(X_train)
    large_cat = [c for c in X_train.columns if c not in keep]
    all_na = all_na_columns(X_train[keep])
    columns = [c for c in keep if c not in all_na]
    Xtr = X_train[columns].reset_index(drop=True)
    Xva = X_val[columns].reset_index(drop=True)

    ytr, yva, enc, y_mean, y_std = _encode_targets(task, y_train, y_val)
    n_classes = None if enc is None else len(enc.classes_)

    runs, best = [], None
    for cand in get_param_candidates(p):
        log('lr {}'.format(cand['model_lr']))
        t0 = time.time()
        try:
            result, state, pipe, model, trainer = _one_run(task, Xtr, ytr, Xva, yva, deepcopy(DEFAULT_SPACE),
                                                           cand, n_classes, log)
        except FloatingPointError as err:
            runs.append({'model_lr': cand['model_lr'], 'error': str(err), 'secs': round(time.time() - t0, 1)})
            log('  failed: {}'.format(err))
            continue
        except RuntimeError as err:
            # upstream would stop the whole grid here; the grid point is recorded as crashed instead
            runs.append({'model_lr': cand['model_lr'], 'error': 'crashed: {}'.format(str(err)[:300]),
                         'secs': round(time.time() - t0, 1)})
            log('  crashed: {}'.format(err))
            continue
        result = dict(result, model_lr=cand['model_lr'], secs=round(time.time() - t0, 1))
        runs.append(result)
        log('  done: best epoch {} val_loss {:.4f} val_acc {:.4f} after {} epochs, {:.0f}s'.format(
            result['best_epoch'], result['best_val_loss'], result['best_val_acc'], result['epochs_run'],
            result['secs']))
        if best is None or result['best_val_loss'] < best[0]['best_val_loss']:
            best = (result, state, pipe, model, trainer, cand)
    if best is None:
        raise DiffPrepDiverged(runs)

    result, state, pipe, model, trainer, cand = best
    pipe.load_state_dict(state['prep_state'])
    model.load_state_dict(state['model_state'])
    for t, opts in zip(pipe.pipeline[1:], state['tf_options']):
        t.tf_options = opts
    for t in pipe.pipeline:
        t.sample(temperature=cand['temperature'], use_sample=cand['sample'])
    # the restored pipeline must reproduce the selected validation loss
    check_loss, check_acc = trainer.evaluate(Xva, yva, X_type='val', max_only=False)
    result = dict(result, restore_check_val_loss=check_loss, restore_check_val_acc=check_acc)

    deviations = (list(DEVIATIONS) + ([REGRESSION_DEVIATION] if task == 'regression' else [])
                  + stabilization_deviations(p))
    dropped = {'large_categorical': large_cat, 'all_missing_in_train': all_na}
    return DiffPrepResult(task, pipe, model, cand, columns, dropped, enc, y_mean, y_std, result, runs,
                          deviations, time.time() - tic)
