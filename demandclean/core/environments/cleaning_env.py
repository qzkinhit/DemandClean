"""
Cleaning environment
====================

Cleaning environment used during training and inference.

Training: use injected errors, with repair_value set to the pre-injection
  value (treated as ground truth).
Inference: use real errors, with repair_value set to the supplied ground
  truth.

Regression/classification compatibility:
  - Score normalization: classification uses accuracy in [0,1];
    regression uses 1/(1+MSE) in [0,1].
  - Label replacement: classification uses majority vote; regression uses
    KNN mean.
  - Minimum retention: deletions are capped at 80% to avoid wiping out
    the dataset.

Reward design:
  - Eval step (every N steps, N adaptive): train the downstream model;
    perf_diff - repair_lambda * (action==1).
  - Intermediate step: heuristic reward shaping (auto-scaled to the
    perf_diff magnitude).
  - Final: score_improvement * 5 + keep_rate * 0.2 - repair_cost.
  - Adaptive N: total eval cost ~= constant; config.reward_eval_interval=0
    enables auto mode.
"""

import os
from typing import Dict, List, Set, Tuple, Optional, Any
import math
import random
import numpy as np

from ...config import DemandCleanConfig, TaskType
from ...models import ModelAdapter
from ..state import StateExtractor
from .value_estimation import ValueEstimator
from ..state.state_extractor import LEGACY_STATE_SIZE


class CleaningEnv:
    """
    Cleaning environment.

    State space (8 dims):
    - error_type: error type (0=missing, 1=semantic, 2=syntactic, 3=label_noise)
    - feature_importance: feature importance
    - distance_to_boundary: distance to decision boundary / mean
    - row_position: row position
    - col_index: column index (-1 for a label error)
    - col_error_rate: error rate of the current column
    - sample_retention: sample retention rate
    - var_retention: variance retention rate

    Special handling for label errors:
    - When repair_value=None, action=1 (repair) is auto-degraded to
      action=2 (delete).
    - Reward shaping prefers delete for label errors, since estimated
      label values are unreliable.
    """

    def __init__(self,
                 X_dirty: np.ndarray,
                 y: np.ndarray,
                 error_list: List[Dict[str, Any]],
                 model_adapter: ModelAdapter,
                 state_extractor: StateExtractor,
                 config: DemandCleanConfig,
                 X_base: Optional[np.ndarray] = None,
                 y_base: Optional[np.ndarray] = None,
                 value_estimator: Optional['ValueEstimator'] = None,
                 shaping_weight: float = 1.0):
        """
        Initialize the cleaning environment.

        Args:
            X_dirty: dirty feature matrix
            y: label vector
            error_list: list of errors
                [{'idx': int, 'col': int, 'type': int, 'repair_value': float|None}, ...]
                type: 0=missing, 1=semantic, 2=syntactic, 3=label_noise
                col: -1 indicates a label error
                repair_value: pre-injection value during training (or None if
                    no ground truth is available); the true value during inference
            model_adapter: model adapter
            state_extractor: state extractor
            config: config object
            X_base: validation feature matrix (passed at training time;
                None at inference)
            y_base: validation label vector (passed at training time;
                None at inference)
        """
        self.X_dirty_original = X_dirty.copy()
        self.y_original = y.copy()
        self.error_list = error_list
        self.model_adapter = model_adapter
        self.state_extractor = state_extractor
        self.config = config

        self.repair_lambda = config.repair_lambda
        self.min_truth_budget = config.min_truth_budget
        self.max_truth_budget = config.max_truth_budget

        # Repair budget: a share of the detected errors, capped by max_truth_budget
        n_errors = len(error_list)
        self.min_repair_ratio = config.min_repair_ratio
        self.max_repair_ratio = config.max_repair_ratio
        self.max_repair_count = int(n_errors * config.max_repair_ratio) if config.max_repair_ratio < 1.0 else n_errors
        # Absolute cap on GT-repairs (0 for variant ngt)
        if config.max_truth_budget is not None:
            self.max_repair_count = min(self.max_repair_count, config.max_truth_budget)
        # Annotation cost of a GT-repair by error type (missing, semantic, syntactic, label).
        # DC_COST_BY_TYPE="1,3,1,2" prices semantic errors three times a missing value. The budget
        # max_repair_count is then read in cost units. Unset means unit cost for every cell.
        _costs = os.environ.get('DC_COST_BY_TYPE')
        self._cost_by_type = [float(c) for c in _costs.split(',')] if _costs else [1.0, 1.0, 1.0, 1.0]
        self.cost_used = 0.0

        # Validation set (passed at training time; None at inference)
        self.X_base = X_base
        self.y_base = y_base
        self._eval_sample_indices = self._build_eval_sample_indices()

        # Reward evaluation cache
        self._cached_score = None
        self._steps_since_eval = 0

        # Adaptive eval interval (auto-computed when config=0, manual when >0)
        self._reward_eval_interval = self._compute_eval_interval(
            len(error_list), len(X_dirty), config.reward_eval_interval
        )

        # Heuristic reward auto-scaling state
        self._perf_diff_ema = 0.0              # EMA of |perf_diff|
        self._perf_diff_ema_initialized = False
        self._shaping_scale = 1.0              # scaling factor (=1.0 before first eval, no scaling)
        self._shaping_weight = shaping_weight  # training-progress decay factor in [0,1], supplied by trainer

        # State variables
        self.X_current: Optional[np.ndarray] = None
        self.y_current: Optional[np.ndarray] = None
        self.current_error_idx = 0
        self.deleted_rows: Set[int] = set()
        self.action_counts = {
            'no_action': 0,
            'repair_value': 0,
            'delete': 0,
            'replace_nearby': 0
        }

        # Incremental caches (avoid recomputing _fill_nan and keep_mask)
        self._keep_mask = np.ones(len(X_dirty), dtype=bool)
        self._X_filled_cache: Optional[np.ndarray] = None

        # Repair log
        self.repair_log: List[Dict] = []

        # Full decision log (details for all 4 action types)
        self.decision_log: List[Dict] = []

        # Precomputed statistics
        self._precompute_stats()

        # Value estimator (FD + KNN + DOMAIN), can be shared across episodes
        if value_estimator is not None:
            self.value_estimator = value_estimator
        else:
            self.value_estimator = ValueEstimator(config)

        # Feature-importance refresh interval
        if config.importance_refresh_interval is not None:
            self.importance_refresh_interval = config.importance_refresh_interval
        else:
            self.importance_refresh_interval = max(20, len(error_list) // 10)

        # Initialize state extractor (also populates the X_filled cache)
        X_filled = self._init_state_extractor()

        # Baseline performance (reuse the model fit in _init_state_extractor
        # to avoid a double fit)
        self.baseline_score = self._compute_baseline_score(X_filled)

        # Final-score cache at episode end (reused by trainer to avoid a
        # double evaluate)
        self.last_episode_score: Optional[float] = None

    def _build_eval_sample_indices(self) -> Optional[np.ndarray]:
        """Build validation-set sampling indices."""
        if self.X_base is None:
            return None
        n = len(self.X_base)
        ratio = getattr(self.config, 'eval_sample_ratio', 1.0)
        if ratio >= 1.0:
            return None  # no sampling, use the full set
        k = max(10, int(n * ratio))
        return np.random.choice(n, k, replace=False)

    @staticmethod
    def _compute_eval_interval(n_errors: int, n_samples: int,
                                config_value: int) -> int:
        """Compute the adaptive evaluation interval.

        config_value > 0: user-specified, returned directly.
        config_value == 0: auto-computed from the data size.

        Key formula: total eval cost ~ (n_errors / N) * n_samples ~= constant budget.
        """
        if config_value > 0:
            return config_value
        if n_errors == 0:
            return 10

        BUDGET = 500_000       # reference budget (cost on the scale of beers)
        MIN_EVALS = 20         # minimum number of evaluations per episode
        MAX_INTERVAL = 200     # upper bound on the interval

        cost_based_n = max(1, int(n_errors * n_samples / BUDGET))
        max_n_for_min_evals = max(1, n_errors // MIN_EVALS)
        return max(5, min(cost_based_n, max_n_for_min_evals, MAX_INTERVAL))

    def _precompute_stats(self) -> None:
        """Precompute statistics."""
        self.col_means = np.nanmean(self.X_dirty_original, axis=0)
        self.col_vars = np.nanvar(self.X_dirty_original, axis=0)
        self.col_stds = np.sqrt(self.col_vars)

        # Valid values per column
        self.all_values = {}
        for col in range(self.X_dirty_original.shape[1]):
            valid = self.X_dirty_original[:, col][~np.isnan(self.X_dirty_original[:, col])]
            self.all_values[col] = valid

        # Per-column error rate (label errors with col=-1 are not counted as feature columns)
        n_cols = self.X_dirty_original.shape[1]
        col_error_counts = np.zeros(n_cols)
        label_error_count = 0
        for error in self.error_list:
            col = error['col']
            if col == -1:
                label_error_count += 1
            elif col < n_cols:
                col_error_counts[col] += 1

        total_errors = len(self.error_list)
        if total_errors > 0:
            self.col_error_rate = col_error_counts / total_errors
        else:
            self.col_error_rate = np.zeros(n_cols)
        self.label_error_rate = label_error_count / max(total_errors, 1)

    def _init_state_extractor(self) -> np.ndarray:
        """Initialize the state extractor and return X_filled for later reuse.

        Fits the model on the clean subset (with detected error rows
        removed), so that distance_to_boundary has a consistent
        distribution across training and inference (both compute the
        decision boundary on relatively clean data).
        """
        # Fill NaNs so the model can be trained
        X_filled = self._fill_nan(self.X_dirty_original.copy())
        self._X_filled_cache = X_filled.copy()

        # Fit the model on the clean subset (error rows removed) so the
        # state distribution is consistent across training and inference.
        error_rows = set(e['idx'] for e in self.error_list)
        clean_mask = np.array([i not in error_rows for i in range(len(X_filled))])
        if self.config.state_size == LEGACY_STATE_SIZE:
            # Legacy 8-dim checkpoints (earlier state builder) were trained
            # with the init model fit on all rows; their feature_importance input
            # is reproduced only with the same fit.
            clean_mask = np.zeros(len(X_filled), dtype=bool)
        try:
            if clean_mask.sum() >= 20:
                self.model_adapter.fit(X_filled[clean_mask], self.y_original[clean_mask])
            else:
                self.model_adapter.fit(X_filled, self.y_original)
        except Exception:
            try:
                self.model_adapter.fit(X_filled, self.y_original)
            except Exception:
                pass

        # Compute feature importance
        try:
            feature_importance = self.model_adapter.get_feature_importance()
        except Exception:
            feature_importance = np.ones(self.X_dirty_original.shape[1]) / self.X_dirty_original.shape[1]

        # Configure the state extractor
        self.state_extractor.set_model_adapter(self.model_adapter)
        self.state_extractor.set_feature_importance(feature_importance)
        self.state_extractor.set_col_error_rate(self.col_error_rate)
        self.state_extractor.set_col_stats(self.col_means, self.col_stds, self.col_vars)
        # Record sample count so compute_retention() can compute sample_retention correctly
        self.state_extractor._n_samples = len(X_filled)

        return X_filled

    def _fill_nan(self, X: np.ndarray) -> np.ndarray:
        """Fill NaN values."""
        X_filled = X.copy()
        for col in range(X_filled.shape[1]):
            col_mean = np.nanmean(X_filled[:, col])
            nan_mask = np.isnan(X_filled[:, col])
            if nan_mask.any():
                X_filled[nan_mask, col] = col_mean if not np.isnan(col_mean) else 0
        return X_filled

    def _normalize_raw_score(self, raw_score: float) -> float:
        """Normalize a raw model score to [0, 1].

        - Classification: accuracy is already in [0, 1].
        - Regression: -MSE -> 1/(1+log(1+MSE)) in (0, 1]. The log
          compresses large MSE values so the normalized perf_diff
          signal does not become too weak.
        - Clustering: silhouette in [-1, 1] -> (s+1)/2 in [0, 1].
        """
        if self.config.task_type == TaskType.REGRESSION:
            mse = abs(raw_score)
            if self.config.regression_log_normalize:
                return 1.0 / (1.0 + math.log(1.0 + mse))
            else:
                return 1.0 / (1.0 + mse)
        elif self.config.task_type == TaskType.CLUSTERING:
            return (raw_score + 1.0) / 2.0
        else:
            return raw_score

    def _compute_baseline_score(self, X_filled: np.ndarray) -> float:
        """Compute the baseline score reusing the model already fit in _init_state_extractor."""
        try:
            if self.X_base is not None and self.y_base is not None:
                if self._eval_sample_indices is not None:
                    X_eval = self.X_base[self._eval_sample_indices]
                    y_eval = self.y_base[self._eval_sample_indices]
                else:
                    X_eval = self.X_base
                    y_eval = self.y_base
                raw_score = self.model_adapter.evaluate(X_eval, y_eval)
            else:
                raw_score = self.model_adapter.evaluate(X_filled, self.y_original)
            return self._normalize_raw_score(raw_score)
        except Exception:
            return 0.0

    def _update_filled_cache(self, idx: int, col: int, value: float) -> None:
        """Incrementally update a single element of _X_filled_cache."""
        if self._X_filled_cache is not None:
            if np.isnan(value):
                self._X_filled_cache[idx, col] = self.col_means[col]
            else:
                self._X_filled_cache[idx, col] = value

    def _evaluate_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate model performance on the current data.

        With X_base: fit(X_train) -> evaluate(X_base_sample, y_base_sample).
        Without X_base: keep the original logic (backward compatibility;
        no reward is used during inference).

        Uses the incremental caches _X_filled_cache and _keep_mask to
        avoid recomputation.
        """
        # Use the cache (always available in the training loop); fall
        # back to a full recomputation otherwise.
        if self._X_filled_cache is not None:
            X_filled = self._X_filled_cache
        else:
            X_filled = self._fill_nan(X.copy())

        keep_mask = self._keep_mask
        if keep_mask.sum() < 10:
            return 0.0

        X_train = X_filled[keep_mask]
        y_train = y[keep_mask]

        try:
            self.model_adapter.fit(X_train, y_train)

            # Evaluate on the validation set when one is available
            if self.X_base is not None and self.y_base is not None:
                if self._eval_sample_indices is not None:
                    X_eval = self.X_base[self._eval_sample_indices]
                    y_eval = self.y_base[self._eval_sample_indices]
                else:
                    X_eval = self.X_base
                    y_eval = self.y_base
                raw_score = self.model_adapter.evaluate(X_eval, y_eval)
            else:
                # Backward compatibility: evaluate on the full data when no validation set is provided
                raw_score = self.model_adapter.evaluate(X_filled, y)

            score = self._normalize_raw_score(raw_score)

            # Also refresh feature_importance (no extra fit cost, reusing the already-fit model).
            # Legacy 8-dim checkpoints were trained with the importance fixed at
            # environment init (earlier state builder), so it stays fixed for them.
            if self.config.state_size != LEGACY_STATE_SIZE:
                try:
                    new_imp = self.model_adapter.get_feature_importance()
                    if new_imp is not None:
                        self.state_extractor.feature_importance = new_imp
                except Exception:
                    pass

            return score
        except Exception:
            return 0.0

    def reset(self) -> np.ndarray:
        """Reset the environment."""
        self.X_current = self.X_dirty_original.copy()
        self.y_current = self.y_original.copy()
        self.current_error_idx = 0
        self.deleted_rows = set()
        self.action_counts = {k: 0 for k in self.action_counts}
        self.repair_log = []
        self.cost_used = 0.0
        self.decision_log = []
        self._cached_score = None
        self._steps_since_eval = 0
        # Reset incremental caches
        self._X_filled_cache = self._fill_nan(self.X_current.copy())
        self._keep_mask = np.ones(len(self.X_dirty_original), dtype=bool)
        self.last_episode_score = None
        random.shuffle(self.error_list)
        return self._get_state()

    def _cell_cost(self, error: Dict[str, Any]) -> float:
        """Annotation cost of GT-repairing this candidate cell."""
        if error.get('col', 0) == -1:
            return self._cost_by_type[3]
        t = int(error.get('type', 0))
        return self._cost_by_type[t] if 0 <= t < 3 else 1.0

    def _get_state(self) -> np.ndarray:
        """Get the current state (8 error-level features + 2 global features).

        With config.state_size == LEGACY_STATE_SIZE (a checkpoint trained with
        the earlier 8-dim state builder) the two global features are dropped.
        """
        if self.config.state_size == LEGACY_STATE_SIZE:
            return self._full_state()[:LEGACY_STATE_SIZE]
        return self._full_state()

    def _full_state(self) -> np.ndarray:
        """8 error-level features + 2 global features (also used by reward shaping)."""
        if self.current_error_idx >= len(self.error_list):
            return np.zeros(max(self.config.state_size, LEGACY_STATE_SIZE + 2), dtype=np.float32)

        error = self.error_list[self.current_error_idx]
        base_state = self.state_extractor.extract(
            self.X_current,
            self.y_current,
            error,
            self.deleted_rows
        )

        # Append global features so the DQN knows how much budget is left
        # and how many errors still need handling.
        # [8] remaining_budget_ratio: remaining ground-truth budget ratio in [0,1]
        repair_used = self.cost_used
        remaining_budget = max(0, self.max_repair_count - repair_used)
        remaining_budget_ratio = remaining_budget / max(self.max_repair_count, 1)

        # [9] remaining_errors_ratio: fraction of errors still to be handled, in [0,1]
        total_errors = max(len(self.error_list), 1)
        remaining_errors = max(0, total_errors - self.current_error_idx)
        remaining_errors_ratio = remaining_errors / total_errors

        return np.concatenate([
            base_state,
            np.array([remaining_budget_ratio, remaining_errors_ratio], dtype=np.float32)
        ])

    def _get_nearby_value(self, idx: int, col: int) -> float:
        """Get the nearby value (multi-dim KNN with rule priority)."""
        return self.value_estimator.estimate_feature_value(
            self.X_current, idx, col,
            self.deleted_rows, self.col_means
        )

    def _get_majority_label(self, idx: int, k: int = 5) -> float:
        """
        Get a nearest-neighbor label estimate.

        Classification: majority vote.
        Regression: KNN weighted mean.

        Args:
            idx: target row index
            k: number of neighbors

        Returns:
            Estimated label value.
        """
        X_filled = self._X_filled_cache if self._X_filled_cache is not None else self._fill_nan(self.X_current.copy())
        target = X_filled[idx]

        # Compute distances
        distances = np.linalg.norm(X_filled - target, axis=1)
        distances[idx] = np.inf  # exclude self

        # Exclude deleted rows
        for d_idx in self.deleted_rows:
            distances[d_idx] = np.inf

        # Take the top-k nearest neighbors
        k = min(k, (distances < np.inf).sum())
        if k == 0:
            return self.y_current[idx]

        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = self.y_current[nearest_indices]

        valid_labels = nearest_labels[~np.isnan(nearest_labels)]
        if len(valid_labels) == 0:
            return self.y_current[idx]

        # Regression: KNN weighted mean (inverse-distance weights)
        if self.config.task_type == TaskType.REGRESSION:
            nearest_dists = distances[nearest_indices]
            valid_mask = ~np.isnan(nearest_labels)
            valid_dists = nearest_dists[valid_mask]
            # Inverse-distance weights (avoid division by zero)
            weights = 1.0 / (valid_dists + 1e-8)
            weights /= weights.sum()
            return float(np.average(valid_labels, weights=weights))

        # Classification: majority vote
        unique, counts = np.unique(valid_labels, return_counts=True)
        return unique[np.argmax(counts)]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take one step.

        Args:
            action: action index
                0: no_action
                1: repair_value
                2: delete
                3: replace_nearby

        Returns:
            (next_state, reward, done, info)
        """
        if self.current_error_idx >= len(self.error_list):
            return np.zeros(self.config.state_size, dtype=np.float32), 0, True, {}

        error = self.error_list[self.current_error_idx]
        idx, col, error_type = error['idx'], error['col'], error['type']
        current_state = self._full_state()  # saved for shaping reward (reads the global features)
        repair_value = error['repair_value']

        # Skip rows that have already been deleted
        if idx in self.deleted_rows:
            self.current_error_idx += 1
            return self._get_state(), 0, self.current_error_idx >= len(self.error_list), {}

        # Record the original action (before any degradation)
        original_action = action

        # Ground-truth budget check (all actions share max_repair_count).
        # Over-budget requests are degraded to replace_nearby (KNN
        # replacement works well and avoids losing data).
        cell_cost = self._cell_cost(error)
        if action == 1 and self.cost_used + cell_cost > self.max_repair_count:
            action = 3  # replace_nearby

        # Check whether this is a label-noise error (col == -1)
        is_label_error = (col == -1)

        # Label error + repair_value=None: auto-degrade to delete.
        # (In self-supervised mode the ground-truth label is unreliable, so force delete.)
        if action == 1 and is_label_error and repair_value is None:
            action = 2  # degrade to delete

        # Record the dirty value
        if is_label_error:
            dirty_value = self.y_current[idx]
        else:
            dirty_value = self.X_current[idx, col]
        dirty_value_safe = dirty_value if not (isinstance(dirty_value, float) and np.isnan(dirty_value)) else None

        # Execute the action
        result_value = None
        if action == 0:
            self.action_counts['no_action'] += 1
        elif action == 1:
            if is_label_error:
                self.repair_log.append({
                    'idx': idx,
                    'col': -1,
                    'dirty_value': dirty_value_safe,
                    'clean_value': repair_value,
                    'error_type': error_type
                })
                self.y_current[idx] = repair_value
            else:
                self.repair_log.append({
                    'idx': idx,
                    'col': col,
                    'dirty_value': dirty_value_safe,
                    'clean_value': repair_value,
                    'error_type': error_type
                })
                self.X_current[idx, col] = repair_value
                self._update_filled_cache(idx, col, repair_value)
            self.action_counts['repair_value'] += 1
            self.cost_used += cell_cost
            result_value = repair_value
        elif action == 2:
            # Safety policy: keep at least 20% of the data; once the
            # deletion cap is reached, force the action to no_action.
            n_total = len(self.X_current)
            max_deletions = int(n_total * 0.8)
            if len(self.deleted_rows) >= max_deletions:
                # Deletion cap reached: fall back to no_action
                action = 0
                self.action_counts['no_action'] += 1
            else:
                self.deleted_rows.add(idx)
                self._keep_mask[idx] = False
                self.action_counts['delete'] += 1
        elif action == 3:
            if is_label_error:
                # Label noise: nearby replacement uses majority vote
                nearby_label = self._get_majority_label(idx)
                self.y_current[idx] = nearby_label
                result_value = nearby_label
            else:
                nearby_val = self._get_nearby_value(idx, col)
                self.X_current[idx, col] = nearby_val
                self._update_filled_cache(idx, col, nearby_val)
                result_value = nearby_val
            self.action_counts['replace_nearby'] += 1

        # Record the full decision log
        self.decision_log.append({
            'error_idx': self.current_error_idx,
            'row_idx': idx,
            'col': col,
            'error_type': error_type,
            'action': action,
            'original_action': original_action,
            'dirty_value': dirty_value_safe,
            'result_value': result_value,
        })

        self.current_error_idx += 1
        done = self.current_error_idx >= len(self.error_list)

        # Compute the reward
        self._steps_since_eval += 1

        if done:
            # End of episode: force an evaluation
            reward = self._calculate_final_reward()
            info = {'stage1_reward': reward, 'stage2_reward': reward}
        else:
            # Shaping bonus (always computed, decayed by _shaping_weight)
            shaping_bonus = 0.0
            if self._shaping_weight > 0.01:
                shaping_bonus = self._get_shaping_reward(
                    action, error_type, current_state)

            if self._steps_since_eval >= self._reward_eval_interval or self._cached_score is None:
                # Eval step: train the downstream model for the real
                # performance signal, plus the shaping bonus.
                current_score = self._evaluate_score(self.X_current, self.y_current)
                self._cached_score = current_score
                self._steps_since_eval = 0

                perf_diff = current_score - self.baseline_score
                self._update_perf_diff_ema(perf_diff)

                repair_cost = self.repair_lambda * cell_cost if action == 1 else 0.0
                eval_reward = perf_diff - repair_cost

                # Combine: eval reward (main signal) + shaping bonus (directional guidance)
                reward = eval_reward + shaping_bonus
                stage1_reward = reward
                stage2_reward = reward if action in (1, 3) else None
            else:
                # Intermediate step: shaping only (no downstream training)
                reward = shaping_bonus
                stage1_reward = reward
                stage2_reward = reward if action in (1, 3) else None

            info = {'stage1_reward': stage1_reward, 'stage2_reward': stage2_reward}

        return self._get_state(), reward, done, info

    def _update_perf_diff_ema(self, perf_diff: float) -> None:
        """Update the perf_diff EMA and recompute the shaping scale."""
        abs_diff = abs(perf_diff)
        if not self._perf_diff_ema_initialized:
            self._perf_diff_ema = abs_diff
            self._perf_diff_ema_initialized = True
        else:
            self._perf_diff_ema = (
                0.3 * abs_diff + 0.7 * self._perf_diff_ema
            )
        SHAPING_REFERENCE = 0.03  # typical absolute value of the heuristic reward
        if self._perf_diff_ema > 1e-6:
            self._shaping_scale = max(0.1, min(
                self._perf_diff_ema / SHAPING_REFERENCE, 10.0
            ))

    def _get_shaping_reward(self, action: int, error_type: int,
                            state: np.ndarray) -> float:
        """State-aware reward shaping (priority guidance).

        Uses continuous state signals to teach the DQN a priority ranking
        without hard-coded error-type rules:
        - High priority (near the boundary, important feature, high error
          rate) -> repair > replace.
        - Low priority -> replace > repair (don't waste the budget).
        - Budget exhausted -> repair is naturally worse than replace.

        State dimensions:
          [0] error_type, [1] feature_importance, [2] distance_to_boundary,
          [3] row_position, [4] col_index, [5] col_error_rate,
          [6] sample_retention, [7] var_retention,
          [8] remaining_budget_ratio, [9] remaining_errors_ratio
        """
        s = self._shaping_scale
        urgency = 1.0 - state[2]       # boundary urgency in [0,1]
        importance = state[1]           # feature importance in [0,1]
        error_rate = state[5]           # column error rate in [0,1]

        # Combined priority: near boundary + important feature + high
        # error rate -> more worth repairing with ground truth.
        priority = (urgency + importance + error_rate) / 3.0

        # Budget awareness: use state[8] (remaining_budget_ratio) directly.
        # Repair is attractive when the budget is plentiful and becomes
        # less attractive as it runs out.
        remaining_budget_ratio = float(state[8])
        remaining_errors_ratio = float(state[9])
        # Relative slack: remaining_budget > remaining_errors -> repair
        # can be used aggressively.
        budget_slack = remaining_budget_ratio - remaining_errors_ratio  # [-1, 1]
        # Sigmoid: close to 1 when there is slack, close to 0 when tight.
        budget_ok = 1.0 / (1.0 + math.exp(-6.0 * budget_slack))

        if action == 0:    # no_action: ignoring a known error -> penalize
            reward = -0.03
        elif action == 1:  # repair: much better than replace when priority is high; worse when low
            # high priority: (0.02+0.04)=0.06, low priority: (0.02+0)=0.02
            # vs replace at 0.03 -> repair only beats replace when priority>0.25
            reward = (0.02 + 0.04 * priority) * budget_ok
        elif action == 2:  # delete: losing data -> penalize (larger penalty for regression to avoid over-deletion)
            reward = self.config.delete_shaping_reward
        elif action == 3:  # replace: free and effective -> stable positive reward
            reward = 0.03
        else:
            reward = 0.0

        return reward * s * self._shaping_weight

    def _calculate_final_reward(self) -> float:
        """Compute the final reward (simplified: performance gain - repair cost + small retention bonus)."""
        final_score = self._evaluate_score(self.X_current, self.y_current)
        self.last_episode_score = final_score  # cached for the trainer to reuse
        score_improvement = final_score - self.baseline_score

        keep_rate = 1 - len(self.deleted_rows) / len(self.X_current)
        repair_cost = self.cost_used * self.repair_lambda

        # Core: performance gain (main signal) + retention bonus - repair cost
        reward = score_improvement * 5 + keep_rate * self.config.keep_rate_weight - repair_cost
        return reward

    def get_cleaned_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the cleaned data.

        Returns:
            (X_clean, y_clean, keep_mask)
        """
        keep_mask = np.array([i not in self.deleted_rows for i in range(len(self.X_current))])
        X_result = self.X_current[keep_mask].copy()
        y_result = self.y_current[keep_mask]

        # Fill remaining NaNs (use ValueEstimator for per-cell estimation)
        if np.isnan(X_result).any():
            col_means = np.nanmean(X_result, axis=0)
            # Map keep_mask back to original row indices
            original_indices = np.where(keep_mask)[0]
            for col in range(X_result.shape[1]):
                nan_mask = np.isnan(X_result[:, col])
                if nan_mask.any():
                    for i in np.where(nan_mask)[0]:
                        X_result[i, col] = self.value_estimator.estimate_feature_value(
                            X_result, i, col, set(), col_means,
                            dirty_df_row_indices=original_indices,
                        )

        return X_result, y_result, keep_mask

    def get_repair_log(self) -> List[Dict]:
        """Return the repair log."""
        return self.repair_log

    def get_action_counts(self) -> Dict[str, int]:
        """Return the action counts."""
        return self.action_counts.copy()

    def print_repair_log(self, max_rows: int = 20) -> None:
        """Print the repair log."""
        error_type_names = {0: 'missing', 1: 'semantic', 2: 'syntactic', 3: 'label_noise'}

        print(f"\n{'='*70}")
        print(f"Ground-truth usage log ({len(self.repair_log)} entries)")
        print(f"{'='*70}")

        if len(self.repair_log) == 0:
            print("  (no ground truth used)")
            return

        print(f"{'Index':<8} {'Col':<6} {'Dirty':<15} {'Clean':<15} {'ErrorType':<10}")
        print("-" * 70)

        display_log = self.repair_log[:max_rows] if max_rows else self.repair_log

        for record in display_log:
            dirty_str = f"{record['dirty_value']:.4f}" if record['dirty_value'] is not None else "NaN"
            clean_str = f"{record['clean_value']:.4f}"
            error_type_str = error_type_names.get(record['error_type'], 'unknown')
            print(f"{record['idx']:<8} {record['col']:<6} {dirty_str:<15} {clean_str:<15} {error_type_str:<10}")

        if max_rows and len(self.repair_log) > max_rows:
            print(f"... {len(self.repair_log) - max_rows} more entries omitted ...")

        # Stats
        print("-" * 70)
        by_type = {}
        for r in self.repair_log:
            t = error_type_names.get(r['error_type'], 'unknown')
            by_type[t] = by_type.get(t, 0) + 1

        print(f"Stats: ", end="")
        print(", ".join([f"{k}={v}" for k, v in by_type.items()]))
        print(f"{'='*70}\n")

    def get_decision_log(self) -> List[Dict]:
        """Return the full decision log (details for all 4 action types)."""
        return self.decision_log

    def print_decision_summary(self, max_rows: int = 30) -> None:
        """
        Print a categorized decision-log summary.

        Grouped by action type: repair -> replace -> delete -> no_action.
        """
        ACTION_NAMES = {0: 'no_action', 1: 'repair_value', 2: 'delete', 3: 'replace_nearby'}
        ERROR_TYPE_NAMES = {0: 'missing', 1: 'semantic', 2: 'syntactic', 3: 'label_noise'}

        total = len(self.decision_log)
        if total == 0:
            print("  (no decisions recorded)")
            return

        # Action distribution
        print(f"\n  Action distribution ({total} errors):")
        for act_id, act_name in ACTION_NAMES.items():
            count = self.action_counts.get(act_name, 0)
            pct = count / total * 100 if total > 0 else 0
            bar = '#' * int(pct / 2.5)
            print(f"    {act_name:<16} {count:>5} ({pct:5.1f}%) {bar}")

        # Degradation stats
        degraded = [d for d in self.decision_log if d['action'] != d['original_action']]
        if degraded:
            print(f"\n  Action degradations: {len(degraded)}")
            for d in degraded[:5]:
                print(f"    row {d['row_idx']}: {ACTION_NAMES[d['original_action']]} -> {ACTION_NAMES[d['action']]}")
            if len(degraded) > 5:
                print(f"    ... {len(degraded)} degradations in total")

        # Grouped details
        repairs = [d for d in self.decision_log if d['action'] == 1]
        replaces = [d for d in self.decision_log if d['action'] == 3]
        deletes = [d for d in self.decision_log if d['action'] == 2]

        def _fmt_val(v):
            if v is None:
                return 'NaN'
            return f'{v:.4f}'

        # Repair details
        if repairs:
            n_show = min(max_rows, len(repairs))
            print(f"\n  Repair details (repair_value): {len(repairs)} entries")
            print(f"    {'Row':<8} {'Col':<6} {'Dirty':<12} -> {'Repaired':<12} {'ErrorType':<10}")
            print(f"    {'-'*55}")
            for d in repairs[:n_show]:
                print(f"    {d['row_idx']:<8} {d['col']:<6} "
                      f"{_fmt_val(d['dirty_value']):<12} -> "
                      f"{_fmt_val(d['result_value']):<12} "
                      f"{ERROR_TYPE_NAMES.get(d['error_type'], '?'):<10}")
            if len(repairs) > n_show:
                print(f"    ... {len(repairs) - n_show} more entries omitted")

        # Replace details
        if replaces:
            n_show = min(max_rows, len(replaces))
            print(f"\n  Replace details (replace_nearby): {len(replaces)} entries")
            print(f"    {'Row':<8} {'Col':<6} {'Dirty':<12} -> {'Replaced':<12} {'ErrorType':<10}")
            print(f"    {'-'*55}")
            for d in replaces[:n_show]:
                print(f"    {d['row_idx']:<8} {d['col']:<6} "
                      f"{_fmt_val(d['dirty_value']):<12} -> "
                      f"{_fmt_val(d['result_value']):<12} "
                      f"{ERROR_TYPE_NAMES.get(d['error_type'], '?'):<10}")
            if len(replaces) > n_show:
                print(f"    ... {len(replaces) - n_show} more entries omitted")

        # Delete details
        if deletes:
            n_show = min(max_rows, len(deletes))
            print(f"\n  Delete details (delete): {len(deletes)} entries")
            print(f"    {'Row':<8} {'Col':<6} {'Dirty':<12} {'ErrorType':<10}")
            print(f"    {'-'*40}")
            for d in deletes[:n_show]:
                print(f"    {d['row_idx']:<8} {d['col']:<6} "
                      f"{_fmt_val(d['dirty_value']):<12} "
                      f"{ERROR_TYPE_NAMES.get(d['error_type'], '?'):<10}")
            if len(deletes) > n_show:
                print(f"    ... {len(deletes) - n_show} more entries omitted")

    def _refresh_feature_importance(self) -> None:
        """Periodically refresh feature importance.

        After every importance_refresh_interval errors, retrain the model
        on the current cleaned data and update feature_importance.
        """
        X_filled = self._X_filled_cache if self._X_filled_cache is not None else self._fill_nan(self.X_current.copy())
        keep_mask = self._keep_mask
        if keep_mask.sum() < 10:
            return
        try:
            self.model_adapter.fit(X_filled[keep_mask], self.y_current[keep_mask])
            new_importance = self.model_adapter.get_feature_importance()
            if new_importance is not None:
                self.state_extractor.feature_importance = new_importance
        except Exception:
            pass
