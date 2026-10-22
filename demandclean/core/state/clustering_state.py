"""
State extractor for clustering tasks
====================================

State-feature extractor tailored to clustering tasks (e.g. KMeans).

Key differences from classification / regression:
- distance_to_boundary uses cluster-margin (nearest vs. second-nearest center distance gap)
- feature_importance is derived from how much each dimension separates cluster centers
- Clustering has no concept of labels, so col == -1 errors use the mean of
  feature importances.
"""

from typing import Dict, Any, Set
import numpy as np

from .state_extractor import StateExtractor


class ClusteringStateExtractor(StateExtractor):
    """
    State-feature extractor for clustering tasks.

    Key semantics:
    1. distance_to_boundary is the distance from the sample to the cluster boundary (margin).
       - Large margin -> deep inside a cluster -> low risk of changing assignment on edit
       - Small margin -> near a cluster boundary -> editing may flip the cluster assignment
    2. A label error (col == -1) in clustering means the true cluster label disagrees
       with the clustering result.
       - Such errors are not repaired directly at the label level; instead we clean
         features to improve clustering quality.
    """

    def extract(self,
                X_current: np.ndarray,
                y: np.ndarray,
                error: Dict[str, Any],
                deleted_rows: Set[int]) -> np.ndarray:
        """
        Build the 8-dim state feature vector.

        Args:
            X_current: current data matrix
            y: cluster labels (either true labels or clustering predictions)
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
            # A clustering "label error" means the sample is in the wrong cluster.
            # Use the maximum feature importance since we don't know which feature
            # caused the misassignment.
            if self.feature_importance is not None and len(self.feature_importance) > 0:
                feat_imp = float(np.max(self.feature_importance))
            else:
                feat_imp = 0.5
        elif self.feature_importance is not None and 0 <= col < len(self.feature_importance):
            feat_imp = self.feature_importance[col]
        else:
            feat_imp = 0.5

        # 3. distance_to_boundary (clustering: cluster margin)
        if is_label_error:
            # Label error: average boundary distance across feature dimensions
            distance_norm = self._get_avg_boundary_distance(X_current, idx)
        else:
            distance_norm = self.get_distance_to_boundary(X_current, idx, col)

        # 4. row_position
        n_rows = len(X_current)
        row_pos = idx / (n_rows - 1) if n_rows > 1 else 0

        # 5. col_index
        n_cols = X_current.shape[1] if X_current.ndim > 1 else 1
        if is_label_error:
            col_norm = 1.0
        else:
            col_norm = col / (n_cols - 1) if n_cols > 1 else 0

        # 6. col_error_rate
        if is_label_error:
            col_err_rate = 0.5
        elif self.col_error_rate is not None and 0 <= col < len(self.col_error_rate):
            col_err_rate = self.col_error_rate[col]
        else:
            col_err_rate = 0.0

        # 7-8. sample_retention and var_retention
        if is_label_error:
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
        Return the normalized distance to the cluster boundary.

        For clustering, "boundary" is the inter-cluster decision surface:
        - Uses KMeansAdapter.get_distance_to_boundary(), which returns
          (second_nearest_center_dist - nearest_center_dist) / max_margin.
        - Large value -> deep inside a cluster; editing the feature is unlikely
          to change cluster assignment.
        - Small value -> near a cluster boundary; editing the feature may flip
          the cluster assignment.
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

    def _get_avg_boundary_distance(self,
                                    X_current: np.ndarray,
                                    idx: int) -> float:
        """
        Return the average distance of a sample to the cluster boundary.

        Used for the label-error case, since label errors do not correspond to
        any particular feature column.
        """
        return self.get_distance_to_boundary(X_current, idx, 0)
