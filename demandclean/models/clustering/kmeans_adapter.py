"""
KMeans Clustering Adapter
=========================

Uses silhouette_score as the evaluation metric; supports clustering tasks in
the DQN environment.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Optional

from ..base_adapter import ModelAdapter


class KMeansAdapter(ModelAdapter):
    """
    KMeans clustering adapter.

    evaluate() returns silhouette_score in [-1, 1]; larger is better.
    The DQN environment normalizes it to [0, 1].

    For large datasets (>10000 samples), silhouette_score is computed on a
    subsample to avoid the O(n^2) full-distance-matrix bottleneck.
    """

    # Silhouette sampling threshold: use sampling once the row count exceeds this
    _SILHOUETTE_SAMPLE_THRESHOLD = 10000
    _SILHOUETTE_SAMPLE_SIZE = 5000

    def __init__(self,
                 n_clusters: int = None,
                 random_state: int = 42,
                 n_init: int = 10,
                 **kwargs):
        super().__init__()
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self._kwargs = kwargs
        self.model = None
        self._labels: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'KMeansAdapter':
        """Train the KMeans model.

        Args:
            X: Feature matrix
            y: Ground-truth labels (used only to infer n_clusters; not used in training)
        """
        # Infer the cluster count from y automatically
        if self.n_clusters is None:
            if y is not None:
                self.n_clusters = len(np.unique(y))
            else:
                self.n_clusters = 5  # Default

        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
            **self._kwargs
        )
        self._labels = self.model.fit_predict(X)
        self._is_fitted = True

        # Feature importance: deviation of cluster centers from the global mean
        if X.shape[1] > 0:
            global_mean = X.mean(axis=0)
            center_deviation = np.abs(self.model.cluster_centers_ - global_mean).mean(axis=0)
            self._feature_importance = self._normalize_importance(center_deviation)
        else:
            self._feature_importance = np.array([1.0])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels."""
        if not self._is_fitted:
            raise RuntimeError("Model is not trained; please call fit() first.")
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray = None) -> float:
        """Evaluate clustering quality.

        Returns silhouette_score in [-1, 1]. For large datasets (>10000 samples)
        a subsample is used automatically to avoid the O(n^2) bottleneck.
        """
        if not self._is_fitted:
            return 0.0

        try:
            labels = self.model.predict(X)
            n_unique = len(np.unique(labels))
            if n_unique < 2 or n_unique >= len(X):
                return 0.0

            n_samples = len(X)
            if n_samples > self._SILHOUETTE_SAMPLE_THRESHOLD:
                return silhouette_score(
                    X, labels,
                    sample_size=self._SILHOUETTE_SAMPLE_SIZE,
                    random_state=self.random_state,
                )
            return silhouette_score(X, labels)
        except Exception:
            return 0.0

    def get_distance_to_boundary(self, X: np.ndarray) -> np.ndarray:
        """Return the distance to the clustering boundary.

        Uses the gap between each sample's distance to its nearest and
        second-nearest cluster centers:
        - Large gap -> deep inside a cluster -> far from boundary
        - Small gap -> close to the boundary between two clusters
        """
        if not self._is_fitted:
            return np.ones(len(X)) * 0.5

        try:
            distances = self.model.transform(X)  # (n_samples, n_clusters)
            sorted_dists = np.sort(distances, axis=1)

            if sorted_dists.shape[1] < 2:
                return np.ones(len(X)) * 0.5

            # Second-nearest minus nearest (a larger gap means more certainty)
            margin = sorted_dists[:, 1] - sorted_dists[:, 0]

            # Normalize to [0, 1]
            max_margin = margin.max()
            if max_margin < 1e-10:
                return np.ones(len(X)) * 0.5
            return np.clip(margin / max_margin, 0, 1)
        except Exception:
            return np.ones(len(X)) * 0.5

    def get_feature_importance(self) -> np.ndarray:
        """Return feature importance."""
        if self._feature_importance is None:
            raise RuntimeError("Model is not trained; please call fit() first.")
        return self._feature_importance

    def clone(self) -> 'KMeansAdapter':
        """Create an untrained clone."""
        return KMeansAdapter(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
            **self._kwargs
        )
