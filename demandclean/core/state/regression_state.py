"""
State extractor for regression tasks
====================================
"""

from typing import Dict, Any, Set
import numpy as np

from .state_extractor import StateExtractor


class RegressionStateExtractor(StateExtractor):
    """
    State-feature extractor for regression tasks.

    Uses the deviation of the prediction from the mean as distance_to_boundary.
    """

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
            error: current error info {'idx', 'col', 'type', ...}
            deleted_rows: set of already-deleted row indices

        Returns:
            8-dim state vector.
        """
        idx = error.get('idx', 0)
        col = error.get('col', 0)
        error_type = error.get('type', 0)

        # If the row has been deleted, return an 8-dim zero vector (global
        # features are appended by _get_state).
        if idx in deleted_rows:
            return np.zeros(8, dtype=np.float32)

        # Label-error flag (col == -1, type == 3)
        is_label_error = (col == -1)

        # 1. error_type (normalized to [0, 1])
        # type: 0=missing, 1=semantic, 2=syntactic, 3=label_noise
        error_type_norm = min(error_type / 3.0, 1.0)

        # 2. feature_importance
        if is_label_error:
            # Label error: set to 1.0 (the label is the most important signal)
            feat_imp = 1.0
        elif self.feature_importance is not None and 0 <= col < len(self.feature_importance):
            feat_imp = self.feature_importance[col]
        else:
            feat_imp = 0.5

        # 3. distance_to_boundary (for regression, this is an influence measure)
        if is_label_error:
            # Label error: use the distance of this sample to the mean,
            # proxied via the first feature column.
            distance_norm = self.get_distance_to_boundary(X_current, idx, 0)
        else:
            distance_norm = self.get_distance_to_boundary(X_current, idx, col)

        # 4. row_position
        n_rows = len(X_current)
        row_pos = idx / (n_rows - 1) if n_rows > 1 else 0

        # 5. col_index
        n_cols = X_current.shape[1] if X_current.ndim > 1 else 1
        if is_label_error:
            # Mark the label column as 1.0 (beyond the feature-column range)
            col_norm = 1.0
        else:
            col_norm = col / (n_cols - 1) if n_cols > 1 else 0

        # 6. col_error_rate
        if is_label_error:
            col_err_rate = 0.5  # use a mid-range error rate for label errors
        elif self.col_error_rate is not None and 0 <= col < len(self.col_error_rate):
            col_err_rate = self.col_error_rate[col]
        else:
            col_err_rate = 0.0

        # 7-8. sample_retention and var_retention
        if is_label_error:
            # Label errors do not affect feature-column variance; use overall retention
            keep_mask = np.array([i not in deleted_rows for i in range(len(X_current))])
            sample_retention = keep_mask.sum() / max(len(X_current), 1)
            var_retention = 1.0
        else:
            sample_retention, var_retention = self.compute_retention(
                X_current, col, deleted_rows
            )

        return np.array([
            error_type_norm,
            feat_imp,
            distance_norm,
            row_pos,
            col_norm,
            col_err_rate,
            sample_retention,
            var_retention
        ], dtype=np.float32)

    def get_distance_to_boundary(self,
                                  X_current: np.ndarray,
                                  idx: int,
                                  col: int) -> float:
        """
        Return the normalized distance to the "boundary".

        For regression the "boundary" is redefined as the influence of this
        data point on the model.

        Prediction close to the mean  -> low influence -> small distance
        Prediction far from the mean  -> high influence -> large distance
        """
        if not self.model_adapter.is_fitted:
            return 0.5

        try:
            X_point = X_current[idx:idx+1].copy()
            X_point = self.fill_nan(X_point)
            distance = self.model_adapter.get_distance_to_boundary(X_point)[0]
            return float(np.clip(distance, 0, 1))
        except Exception:
            return 0.5
