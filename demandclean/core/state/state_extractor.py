"""
State-feature extractor base class
==================================

Build the DQN state vector from data and error info.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ...models.base_adapter import ModelAdapter
    from ...config.config import DemandCleanConfig

# Width of the per-error feature vector built by the extractors. Agents trained
# with the earlier state builder (before the two global features
# remaining_budget_ratio / remaining_errors_ratio) take exactly this vector as
# input. The environments drop the global features when config.state_size
# equals this value; DemandClean.load() sets it from the checkpoint.
LEGACY_STATE_SIZE = 8


class StateExtractor(ABC):
    """
    Abstract base class for state-feature extractors.

    Produces an 8-dim state vector from data and error info:
        1. error_type: error type (normalized to [0, 1])
        2. feature_importance: feature importance
        3. distance_to_boundary: distance to the decision boundary
        4. row_position: row position
        5. col_index: column index
        6. col_error_rate: error rate of the current column
        7. sample_retention: sample retention ratio
        8. var_retention: variance retention ratio
    """

    def __init__(self, model_adapter: 'ModelAdapter', config: 'DemandCleanConfig'):
        """
        Initialize the state extractor.

        Args:
            model_adapter: model adapter
            config: configuration object
        """
        self.model_adapter = model_adapter
        self.config = config

        # Column statistics
        self.col_stats: Dict[int, Dict[str, float]] = {}

        # Feature importance
        self.feature_importance: Optional[np.ndarray] = None

        # Column error rates
        self.col_error_rate: Optional[np.ndarray] = None

        # Original data stats
        self._n_samples: int = 0
        self._n_features: int = 0

    def initialize(self,
                   X: np.ndarray,
                   y: np.ndarray,
                   error_list: List[Dict[str, Any]]) -> None:
        """
        Initialize the state extractor.

        Args:
            X: data matrix
            y: labels
            error_list: list of errors
        """
        self._n_samples = len(X)
        self._n_features = X.shape[1] if X.ndim > 1 else 1

        self._compute_col_stats(X)
        self._compute_col_error_rate(error_list)
        self._train_reference_model(X, y)

    def _compute_col_stats(self, X: np.ndarray) -> None:
        """Compute per-column statistics."""
        n_cols = X.shape[1] if X.ndim > 1 else 1

        for col in range(n_cols):
            col_data = X[:, col] if X.ndim > 1 else X
            valid = col_data[~np.isnan(col_data)]

            if len(valid) > 0:
                self.col_stats[col] = {
                    'mean': float(np.mean(valid)),
                    'std': float(np.std(valid)) + 1e-6,
                    'var': float(np.var(valid)),
                    'min': float(np.min(valid)),
                    'max': float(np.max(valid)),
                    'median': float(np.median(valid)),
                }
            else:
                self.col_stats[col] = {
                    'mean': 0.0,
                    'std': 1.0,
                    'var': 1.0,
                    'min': 0.0,
                    'max': 1.0,
                    'median': 0.0,
                }

    def _compute_col_error_rate(self, error_list: List[Dict[str, Any]]) -> None:
        """Compute per-column error rate."""
        col_error_counts = np.zeros(self._n_features)

        for error in error_list:
            col = error.get('col', 0)
            if 0 <= col < self._n_features:
                col_error_counts[col] += 1
            # col == -1 (label errors) are not counted toward feature columns

        total = len(error_list) if error_list else 1
        self.col_error_rate = col_error_counts / total

    def _train_reference_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the reference model used for feature importance and boundary distance."""
        X_filled = self.fill_nan(X)

        try:
            self.model_adapter.fit(X_filled, y)
            self.feature_importance = self.model_adapter.get_feature_importance()
        except Exception as e:
            # Fall back to a uniform distribution if training fails
            print(f"Reference model training failed: {e}")
            self.feature_importance = np.ones(self._n_features) / self._n_features

    def fill_nan(self, X: np.ndarray) -> np.ndarray:
        """Fill NaNs with column means."""
        X_filled = X.copy()
        n_cols = X_filled.shape[1] if X_filled.ndim > 1 else 1

        for col in range(n_cols):
            col_data = X_filled[:, col] if X_filled.ndim > 1 else X_filled
            nan_mask = np.isnan(col_data)
            if nan_mask.any():
                mean_val = self.col_stats.get(col, {}).get('mean', 0)
                if X_filled.ndim > 1:
                    X_filled[nan_mask, col] = mean_val
                else:
                    X_filled[nan_mask] = mean_val

        return X_filled

    @abstractmethod
    def extract(self,
                X_current: np.ndarray,
                y: np.ndarray,
                error: Dict[str, Any],
                deleted_rows: Set[int]) -> np.ndarray:
        """
        Build the state feature vector.

        Args:
            X_current: current data matrix
            y: labels
            error: current error info
            deleted_rows: set of already-deleted row indices

        Returns:
            8-dim state vector.
        """
        pass

    @abstractmethod
    def get_distance_to_boundary(self,
                                  X_current: np.ndarray,
                                  idx: int,
                                  col: int) -> float:
        """
        Return the normalized distance to the decision boundary.

        Args:
            X_current: current data matrix
            idx: row index
            col: column index

        Returns:
            Normalized distance in [0, 1].
        """
        pass

    def compute_retention(self,
                          X_current: np.ndarray,
                          col: int,
                          deleted_rows: Set[int]) -> tuple:
        """
        Compute the sample retention and variance retention ratios.

        Args:
            X_current: current data matrix
            col: column index
            deleted_rows: set of already-deleted row indices

        Returns:
            (sample_retention, var_retention)
        """
        keep_mask = np.array([i not in deleted_rows for i in range(len(X_current))])
        n_kept = keep_mask.sum()

        if n_kept < 2:
            return 0.0, 0.0

        # Avoid division by zero
        if self._n_samples == 0:
            sample_retention = 1.0
        else:
            sample_retention = n_kept / self._n_samples

        # Variance retention
        X_kept = X_current[keep_mask]
        col_data = X_kept[:, col] if X_kept.ndim > 1 else X_kept
        valid_kept = ~np.isnan(col_data)

        if valid_kept.sum() < 2:
            return sample_retention, 1.0

        result_var = np.var(col_data[valid_kept])
        original_var = self.col_stats.get(col, {}).get('var', 1.0)

        if original_var > 1e-6:
            var_retention = np.clip(result_var / original_var, 0, 1.5)
        else:
            var_retention = 1.0

        return sample_retention, var_retention

    def get_nearby_value(self, X: np.ndarray, idx: int, col: int) -> float:
        """
        Return a nearby value.

        Args:
            X: data matrix
            idx: row index
            col: column index

        Returns:
            Nearby value.
        """
        current_val = X[idx, col] if X.ndim > 1 else X[idx]
        col_stats = self.col_stats.get(col, {})

        if np.isnan(current_val):
            return col_stats.get('mean', 0.0)

        # Gather all non-NaN values in the column
        col_vals = X[:, col] if X.ndim > 1 else X
        valid_vals = col_vals[~np.isnan(col_vals)]

        if len(valid_vals) == 0:
            return current_val

        # Find the closest distinct value
        distances = np.abs(valid_vals - current_val)
        mask = distances > 0.01
        if mask.sum() > 0:
            min_idx = np.argmin(distances[mask])
            return float(valid_vals[mask][min_idx])

        return current_val

    # Setter methods for use by environments
    def set_model_adapter(self, model_adapter: 'ModelAdapter') -> None:
        """Set the model adapter."""
        self.model_adapter = model_adapter

    def set_feature_importance(self, importance: np.ndarray) -> None:
        """Set the feature importance."""
        self.feature_importance = importance

    def set_col_error_rate(self, error_rate: np.ndarray) -> None:
        """Set the column error rates."""
        self.col_error_rate = error_rate

    def set_col_stats(self,
                      col_means: np.ndarray,
                      col_stds: np.ndarray,
                      col_vars: np.ndarray) -> None:
        """Set column statistics."""
        n_cols = len(col_means)
        self._n_features = n_cols

        for col in range(n_cols):
            self.col_stats[col] = {
                'mean': float(col_means[col]) if not np.isnan(col_means[col]) else 0.0,
                'std': float(col_stds[col]) if not np.isnan(col_stds[col]) else 1.0,
                'var': float(col_vars[col]) if not np.isnan(col_vars[col]) else 1.0,
            }
