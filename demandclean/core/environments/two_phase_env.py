"""
Two-phase inference environment
===============================

Environment for the inference case where ground-truth values are not known up front.

Phase 1: predict every action. repair_value only adds an item to the plan;
         other actions execute immediately.
Phase 2: after the user supplies ground truths, execute the repairs queued in the plan.

Safeguards:
  - Minimum retention: cap deletions at 80% to prevent deleting everything.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np

from ...config import DemandCleanConfig, TaskType
from ...models import ModelAdapter
from ..state import StateExtractor
from .value_estimation import ValueEstimator
from ..state.state_extractor import LEGACY_STATE_SIZE


class TwoPhaseCleaningEnv:
    """
    Two-phase inference environment.

    State computation (identical to the training environment):
    - 10-dim state = 8-dim per-error features + 2-dim global context
    - Per-error features are delegated to state_extractor.extract()
    - Global context: [8] remaining_budget_ratio, [9] remaining_errors_ratio
    - When action == 1, the ValueEstimator estimate is written into X_current
      so that state computation matches training
    - When action == 3, the ValueEstimator is used (FD -> KNN -> DOMAIN)
    - feature_importance is refreshed periodically (matches training)
    """

    def __init__(self,
                 X_dirty: np.ndarray,
                 y: np.ndarray,
                 error_list: List[Dict[str, Any]],
                 model_adapter: ModelAdapter,
                 state_extractor: StateExtractor,
                 config: DemandCleanConfig):
        """
        Initialize the two-phase inference environment.

        Args:
            X_dirty: dirty data (X_clean is not required)
            y: labels
            error_list: detected errors; repair_value may be an estimate or None
            model_adapter: model adapter
            state_extractor: state extractor
            config: configuration object
        """
        self.X_dirty_original = X_dirty.copy()
        self.y_original = y.copy()
        self.error_list = error_list
        self.model_adapter = model_adapter
        self.state_extractor = state_extractor
        self.config = config
        self.repair_lambda = config.repair_lambda

        # Runtime state
        self.X_current: Optional[np.ndarray] = None
        self.y_current: Optional[np.ndarray] = None
        self.current_error_idx = 0
        self.deleted_rows: Set[int] = set()

        # Action counts
        self.action_counts = {
            'no_action': 0,
            'repair_value': 0,
            'delete': 0,
            'replace_nearby': 0
        }

        # Two-phase core: repair plan
        self.repair_plan: List[Dict] = []
        self.planned_repairs: Set[Tuple[int, int]] = set()

        # Maximum repair budget (used for the global-context state).
        # Matches CleaningEnv: based on max_repair_ratio, with legacy
        # max_truth_budget kept for backward compatibility.
        n_errors = len(error_list) if error_list else 1
        ratio_budget = int(n_errors * config.max_repair_ratio) if config.max_repair_ratio < 1.0 else n_errors
        if config.max_truth_budget is not None:
            ratio_budget = min(config.max_truth_budget, ratio_budget)
        self.max_repair_count = ratio_budget

        # Full decision log (records details for all four action types)
        self.decision_log: List[Dict] = []

        # Pre-compute stats
        self._precompute_stats()

        # Value estimator (FD + KNN + DOMAIN)
        self.value_estimator = ValueEstimator(config)

        # Feature-importance refresh interval
        if config.importance_refresh_interval is not None:
            self.importance_refresh_interval = config.importance_refresh_interval
        else:
            self.importance_refresh_interval = max(20, len(error_list) // 10)

        self._init_state_extractor()

    def _precompute_stats(self) -> None:
        """Pre-compute column statistics."""
        n_cols = self.X_dirty_original.shape[1]

        self.col_means = np.nanmean(self.X_dirty_original, axis=0)
        self.col_stds = np.nanstd(self.X_dirty_original, axis=0)
        self.col_vars = np.nanvar(self.X_dirty_original, axis=0)

        # Handle NaNs
        for col in range(n_cols):
            if np.isnan(self.col_means[col]):
                self.col_means[col] = 0
            if np.isnan(self.col_stds[col]) or self.col_stds[col] == 0:
                self.col_stds[col] = 1
            if np.isnan(self.col_vars[col]) or self.col_vars[col] == 0:
                self.col_vars[col] = 1

        # Count errors per column (label errors with col=-1 are excluded)
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
            self.col_error_rates = col_error_counts / total_errors
            self.label_error_rate = label_error_count / total_errors
        else:
            self.col_error_rates = np.zeros(n_cols)
            self.label_error_rate = 0.0

        # Track remaining errors per column
        self.col_remaining_errors = col_error_counts.copy()
        self.total_remaining_errors = total_errors

    def _init_state_extractor(self) -> None:
        """Initialize the state extractor."""
        X_filled = self._fill_nan(self.X_dirty_original.copy())

        try:
            self.model_adapter.fit(X_filled, self.y_original)
        except Exception:
            pass

        try:
            feature_importance = self.model_adapter.get_feature_importance()
        except Exception:
            feature_importance = np.ones(self.X_dirty_original.shape[1]) / self.X_dirty_original.shape[1]

        self.state_extractor.set_model_adapter(self.model_adapter)
        self.state_extractor.set_feature_importance(feature_importance)
        self.state_extractor.set_col_error_rate(self.col_error_rates)
        self.state_extractor.set_col_stats(self.col_means, self.col_stds, self.col_vars)
        # Set sample count so compute_retention() can compute sample_retention correctly
        self.state_extractor._n_samples = len(X_filled)

    def _fill_nan(self, X: np.ndarray) -> np.ndarray:
        """Fill NaN values with column means."""
        X_filled = X.copy()
        for col in range(X_filled.shape[1]):
            col_mean = np.nanmean(X_filled[:, col])
            nan_mask = np.isnan(X_filled[:, col])
            if nan_mask.any():
                X_filled[nan_mask, col] = col_mean if not np.isnan(col_mean) else 0
        return X_filled

    def reset(self) -> np.ndarray:
        """Reset the environment."""
        self.X_current = self.X_dirty_original.copy()
        self.y_current = self.y_original.copy()
        self.current_error_idx = 0
        self.deleted_rows = set()
        self.action_counts = {k: 0 for k in self.action_counts}
        self.repair_plan = []
        self.planned_repairs = set()
        self.decision_log = []

        # Reset error tracking
        self._precompute_stats()

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Return the current state (10-dim = 8-dim per-error features + 2-dim global context)."""
        if self.current_error_idx >= len(self.error_list):
            return np.zeros(self.config.state_size, dtype=np.float32)

        error = self.error_list[self.current_error_idx]
        # 8-dim per-error features
        base_state = self.state_extractor.extract(
            self.X_current,
            self.y_current,
            error,
            self.deleted_rows
        )
        if self.config.state_size == LEGACY_STATE_SIZE:
            # Legacy 8-dim checkpoint (earlier state builder): per-error features only.
            return base_state

        # [8] remaining_budget_ratio: fraction of truth-repair budget still available [0, 1]
        repair_used = self.action_counts['repair_value']
        remaining_budget = max(0, self.max_repair_count - repair_used)
        remaining_budget_ratio = remaining_budget / max(self.max_repair_count, 1)

        # [9] remaining_errors_ratio: pending errors as a fraction of the total [0, 1]
        total_errors = max(len(self.error_list), 1)
        remaining_errors = max(0, total_errors - self.current_error_idx)
        remaining_errors_ratio = remaining_errors / total_errors

        # Concatenate into the 10-dim state
        return np.concatenate([
            base_state,
            np.array([remaining_budget_ratio, remaining_errors_ratio], dtype=np.float32)
        ])

    def _get_majority_label(self, idx: int, k: int = 5) -> float:
        """
        Estimate the label from the nearest neighbors.

        Classification: majority vote.
        Regression: KNN weighted mean.

        Matches CleaningEnv._get_majority_label().

        Args:
            idx: target row index
            k: number of neighbors

        Returns:
            Estimated label value.
        """
        X_filled = self._fill_nan(self.X_current.copy())
        target = X_filled[idx]

        # Compute distances
        distances = np.linalg.norm(X_filled - target, axis=1)
        distances[idx] = np.inf  # Exclude the target itself

        # Exclude already-deleted rows
        for d_idx in self.deleted_rows:
            distances[d_idx] = np.inf

        # Pick the k nearest neighbors
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
            weights = 1.0 / (valid_dists + 1e-8)
            weights /= weights.sum()
            return float(np.average(valid_labels, weights=weights))

        # Classification: majority vote
        unique, counts = np.unique(valid_labels, return_counts=True)
        return unique[np.argmax(counts)]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute an action.

        For repair_value (action = 1):
        - Do not actually repair; add an item to repair_plan instead
        - Update state using the estimated value

        Args:
            action: action index

        Returns:
            (next_state, reward, done, info)
        """
        if self.current_error_idx >= len(self.error_list):
            return self._get_state(), 0, True, {}

        error = self.error_list[self.current_error_idx]
        idx, col = error['idx'], error['col']
        error_type = error['type']
        is_label_error = (col == -1)

        # Keep the original action before any downgrade
        original_action = action

        # Record the dirty value
        if is_label_error:
            dirty_value = self.y_current[idx]
        else:
            dirty_value = self.X_current[idx, col]
        dirty_value_safe = dirty_value if not (isinstance(dirty_value, float) and np.isnan(dirty_value)) else None

        result_value = None

        if action == 0:  # no_action
            self.action_counts['no_action'] += 1

        elif action == 1:  # repair_value -> queue in the plan, write the estimated value into X_current
            self.action_counts['repair_value'] += 1

            if is_label_error:
                # Label error: KNN majority-vote estimate
                estimated_value = self._get_majority_label(idx)
            else:
                # Feature error: FD -> multi-dim KNN -> DOMAIN clipping
                estimated_value = self.value_estimator.estimate_feature_value(
                    self.X_current, idx, col,
                    self.deleted_rows, self.col_means
                )

            # Add to the repair plan
            self.repair_plan.append({
                'idx': idx,
                'col': col,
                'error_type': error_type,
                'estimated_value': estimated_value,
                'current_dirty_value': dirty_value_safe
            })

            # Mark as planned
            self.planned_repairs.add((idx, col))

            # Update current data with the estimate
            if is_label_error:
                self.y_current[idx] = estimated_value
            else:
                self.X_current[idx, col] = estimated_value

            result_value = estimated_value

            # Update error counts
            if not is_label_error and 0 <= col < len(self.col_remaining_errors):
                self.col_remaining_errors[col] = max(0, self.col_remaining_errors[col] - 1)
            self.total_remaining_errors = max(0, self.total_remaining_errors - 1)

        elif action == 2:  # delete - executed immediately
            # Safeguard: keep at least 20% of the data; downgrade to no_action once the cap is hit
            n_total = len(self.X_current)
            max_deletions = int(n_total * 0.8)
            if len(self.deleted_rows) >= max_deletions:
                # Deletion cap reached, downgrade to no_action
                action = 0
                self.action_counts['no_action'] += 1
            else:
                self.action_counts['delete'] += 1
                self.deleted_rows.add(idx)

                # Update error counts (only when deletion actually happens)
                if not is_label_error and 0 <= col < len(self.col_remaining_errors):
                    self.col_remaining_errors[col] = max(0, self.col_remaining_errors[col] - 1)
                self.total_remaining_errors = max(0, self.total_remaining_errors - 1)

        elif action == 3:  # replace_nearby - executed immediately
            self.action_counts['replace_nearby'] += 1

            if is_label_error:
                # Label error: replace via majority vote / KNN
                nearby_val = self._get_majority_label(idx)
                self.y_current[idx] = nearby_val
                result_value = nearby_val
            else:
                # Feature error: FD -> multi-dim KNN -> DOMAIN clipping
                nearby_val = self.value_estimator.estimate_feature_value(
                    self.X_current, idx, col,
                    self.deleted_rows, self.col_means
                )
                self.X_current[idx, col] = nearby_val
                result_value = nearby_val

            # Update error counts
            if not is_label_error and 0 <= col < len(self.col_remaining_errors):
                self.col_remaining_errors[col] = max(0, self.col_remaining_errors[col] - 1)
            self.total_remaining_errors = max(0, self.total_remaining_errors - 1)

        # Log the full decision
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

        # Periodically refresh feature_importance
        # (kept fixed for legacy 8-dim checkpoints, as in their training code)
        if (self.current_error_idx % self.importance_refresh_interval == 0
                and not done and self.config.state_size != LEGACY_STATE_SIZE):
            self._refresh_feature_importance()

        return self._get_state(), 0, done, {}

    def get_repair_plan(self) -> List[Dict]:
        """Return the repair plan (Phase 1 output)."""
        return self.repair_plan

    def get_current_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the current data state (unfiltered, includes estimates).

        Returns:
            (X_current, y_current, keep_mask) - unfiltered data and keep mask.
        """
        keep_mask = np.array([i not in self.deleted_rows for i in range(len(self.X_current))])
        return self.X_current.copy(), self.y_current.copy(), keep_mask

    def get_cleaned_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the cleaned data (filtered, NaNs imputed).

        Returns:
            (X_clean, y_clean, keep_mask)
        """
        keep_mask = np.array([i not in self.deleted_rows for i in range(len(self.X_current))])
        X_result = self.X_current[keep_mask].copy()
        y_result = self.y_current[keep_mask]

        # Fill remaining NaNs (missing values kept under no_action) with per-cell
        # ValueEstimator estimates.
        if np.isnan(X_result).any():
            col_means = np.nanmean(X_result, axis=0)
            # Map back to original row indices via keep_mask
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

    def get_action_counts(self) -> Dict[str, int]:
        """Return action counts."""
        return self.action_counts.copy()

    def execute_repair_plan(self,
                            X_dirty: np.ndarray,
                            true_values: Dict[Tuple[int, int], float],
                            y_dirty: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Execute the repair plan (Phase 2).

        Args:
            X_dirty: original dirty data
            true_values: ground-truth dict {(idx, col): value};
                         col = -1 indicates a label repair
            y_dirty: original dirty labels (required for label repairs)

        Returns:
            (X_result, y_result) - repaired features and labels.
        """
        X_result = X_dirty.copy()
        y_result = y_dirty.copy() if y_dirty is not None else None

        for plan_item in self.repair_plan:
            idx, col = plan_item['idx'], plan_item['col']
            repair_val = true_values.get((idx, col), plan_item['estimated_value'])

            if col == -1:
                # Label repair
                if y_result is not None:
                    y_result[idx] = repair_val
            else:
                # Feature repair
                X_result[idx, col] = repair_val

        # Drop deleted rows
        keep_mask = np.array([i not in self.deleted_rows for i in range(len(X_result))])
        X_result = X_result[keep_mask]
        if y_result is not None:
            y_result = y_result[keep_mask]

        # Fill remaining NaNs (per-cell ValueEstimator estimate)
        if np.isnan(X_result).any():
            col_means = np.nanmean(X_result, axis=0)
            # Map back to original row indices via keep_mask
            original_indices = np.where(keep_mask)[0]
            for col in range(X_result.shape[1]):
                nan_mask = np.isnan(X_result[:, col])
                if nan_mask.any():
                    for i in np.where(nan_mask)[0]:
                        X_result[i, col] = self.value_estimator.estimate_feature_value(
                            X_result, i, col, set(), col_means,
                            dirty_df_row_indices=original_indices,
                        )

        return X_result, y_result

    def print_repair_plan(self, max_rows: int = 20) -> None:
        """Print the repair plan."""
        error_type_names = {0: 'missing', 1: 'semantic', 2: 'syntactic', 3: 'label_noise'}

        print(f"\n{'='*70}")
        print(f"Repair plan ({len(self.repair_plan)} items; ground truth needed)")
        print(f"{'='*70}")

        if len(self.repair_plan) == 0:
            print("  (no repairs needed)")
            return

        print(f"{'idx':<8} {'col':<6} {'dirty':<15} {'estimated':<15} {'error_type':<10}")
        print("-" * 70)

        display_plan = self.repair_plan[:max_rows] if max_rows else self.repair_plan

        for record in display_plan:
            dirty_str = f"{record['current_dirty_value']:.4f}" if record['current_dirty_value'] is not None else "NaN"
            est_str = f"{record['estimated_value']:.4f}"
            error_type_str = error_type_names.get(record['error_type'], 'unknown')
            print(f"{record['idx']:<8} {record['col']:<6} {dirty_str:<15} {est_str:<15} {error_type_str:<10}")

        if max_rows and len(self.repair_plan) > max_rows:
            print(f"... {len(self.repair_plan) - max_rows} more omitted ...")

        print(f"{'='*70}\n")

    def get_plan_positions(self) -> List[Tuple[int, int]]:
        """
        Return the positions that need ground-truth repairs.

        Returns:
            [(idx, col), ...] - positions the user must supply truths for.
        """
        return [(p['idx'], p['col']) for p in self.repair_plan]

    def get_decision_log(self) -> List[Dict]:
        """Return the full decision log (details for all four action types)."""
        return self.decision_log

    def save_plan_csv(self, filepath: str) -> None:
        """
        Export the repair plan to a CSV file.

        Args:
            filepath: output CSV path
        """
        import csv
        error_type_names = {0: 'missing', 1: 'semantic', 2: 'syntactic', 3: 'label_noise'}

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['row_idx', 'col', 'error_type', 'dirty_value', 'estimated_value'])
            for item in self.repair_plan:
                writer.writerow([
                    item['idx'],
                    item['col'],
                    error_type_names.get(item['error_type'], 'unknown'),
                    item['current_dirty_value'],
                    item['estimated_value']
                ])
        print(f"  Repair plan saved: {filepath} ({len(self.repair_plan)} items)")

    def print_decision_summary(self, max_rows: int = 30) -> None:
        """
        Print a grouped summary of the decision log.

        Groups by action type in the order: repair(plan) -> replace -> delete -> no_action.
        """
        ACTION_NAMES = {0: 'no_action', 1: 'repair_value(plan)', 2: 'delete', 3: 'replace_nearby'}
        ERROR_TYPE_NAMES = {0: 'missing', 1: 'semantic', 2: 'syntactic', 3: 'label_noise'}

        total = len(self.decision_log)
        if total == 0:
            print("  (no decisions recorded)")
            return

        # Action distribution
        print(f"\n  Action distribution ({total} errors):")
        ac_names = {0: 'no_action', 1: 'repair_value', 2: 'delete', 3: 'replace_nearby'}
        for act_id in [0, 1, 2, 3]:
            act_name = ac_names[act_id]
            display_name = ACTION_NAMES[act_id]
            count = self.action_counts.get(act_name, 0)
            pct = count / total * 100 if total > 0 else 0
            bar = '#' * int(pct / 2.5)
            print(f"    {display_name:<20} {count:>5} ({pct:5.1f}%) {bar}")

        # Downgrade stats
        degraded = [d for d in self.decision_log if d['action'] != d['original_action']]
        if degraded:
            print(f"\n  Action downgrades: {len(degraded)}")

        # Replace / delete detail
        replaces = [d for d in self.decision_log if d['action'] == 3]
        deletes = [d for d in self.decision_log if d['action'] == 2]

        def _fmt_val(v):
            if v is None:
                return 'NaN'
            return f'{v:.4f}'

        if replaces:
            n_show = min(max_rows, len(replaces))
            print(f"\n  Replace details (replace_nearby): {len(replaces)} total")
            print(f"    {'row':<8} {'col':<6} {'dirty':<12}   {'replacement':<12} {'error_type':<10}")
            print(f"    {'-'*55}")
            for d in replaces[:n_show]:
                print(f"    {d['row_idx']:<8} {d['col']:<6} "
                      f"{_fmt_val(d['dirty_value']):<12} -> "
                      f"{_fmt_val(d['result_value']):<12} "
                      f"{ERROR_TYPE_NAMES.get(d['error_type'], '?'):<10}")
            if len(replaces) > n_show:
                print(f"    ... {len(replaces) - n_show} more omitted")

        if deletes:
            n_show = min(max_rows, len(deletes))
            print(f"\n  Delete details (delete): {len(deletes)} total")
            print(f"    {'row':<8} {'col':<6} {'dirty':<12} {'error_type':<10}")
            print(f"    {'-'*40}")
            for d in deletes[:n_show]:
                print(f"    {d['row_idx']:<8} {d['col']:<6} "
                      f"{_fmt_val(d['dirty_value']):<12} "
                      f"{ERROR_TYPE_NAMES.get(d['error_type'], '?'):<10}")
            if len(deletes) > n_show:
                print(f"    ... {len(deletes) - n_show} more omitted")

    def _refresh_feature_importance(self) -> None:
        """Periodically refresh feature importance.

        After processing every importance_refresh_interval errors, retrain the
        model on the currently cleaned data and update feature_importance.
        Matches CleaningEnv._refresh_feature_importance.
        """
        X_filled = self._fill_nan(self.X_current.copy())
        keep_mask = np.array([
            i not in self.deleted_rows for i in range(len(X_filled))
        ])
        if keep_mask.sum() < 10:
            return
        try:
            self.model_adapter.fit(X_filled[keep_mask], self.y_current[keep_mask])
            new_importance = self.model_adapter.get_feature_importance()
            if new_importance is not None:
                self.state_extractor.feature_importance = new_importance
        except Exception:
            pass
