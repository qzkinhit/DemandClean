"""
SVM Classifier Adapter
======================
"""

import numpy as np
from sklearn.svm import SVC
from typing import Optional

from ..base_adapter import ModelAdapter


class SVMAdapter(ModelAdapter):
    """
    SVM classifier adapter.

    Supports multiple kernels; defaults to a linear kernel.
    """

    def __init__(self, kernel: str = 'linear', C: float = 1.0,
                 max_iter: int = -1, **kwargs):
        """
        Initialize the SVM adapter.

        Args:
            kernel: Kernel type ('linear', 'rbf', 'poly')
            C: Regularization parameter
            max_iter: Maximum iterations for the SMO solver.
                      -1 means adaptive: min(50000, max(10000, n_samples * 20))
                      A positive value sets a fixed upper bound.
            **kwargs: Additional arguments forwarded to SVC
        """
        super().__init__()
        self.kernel = kernel
        self.C = C
        self._max_iter_config = max_iter  # Keep the configured value; computed at fit time
        self.max_iter = max_iter
        self.model = SVC(kernel=kernel, C=C, max_iter=10000, **kwargs)  # Initial value, updated at fit time
        self._y_classes: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVMAdapter':
        """Train the SVM model."""
        # Adaptive max_iter: scale with data size
        if self._max_iter_config == -1:
            # Adaptive formula: 10000 for small data, scales with size, capped at 50000
            adaptive_iter = min(50000, max(10000, len(X) * 20))
            self.model.max_iter = adaptive_iter
            self.max_iter = adaptive_iter
        else:
            self.model.max_iter = self._max_iter_config
            self.max_iter = self._max_iter_config

        self.model.fit(X, y)
        self._y_classes = np.unique(y)
        self._is_fitted = True

        # Compute feature importance
        if self.kernel == 'linear' and hasattr(self.model, 'coef_'):
            self._feature_importance = self._normalize_importance(self.model.coef_[0])
        else:
            # For non-linear kernels, use a uniform importance
            self._feature_importance = np.ones(X.shape[1]) / X.shape[1]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self._is_fitted:
            raise RuntimeError("Model is not trained; please call fit() first.")
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def get_distance_to_boundary(self, X: np.ndarray) -> np.ndarray:
        """
        Return the distance to the decision boundary.

        Uses the absolute value of decision_function, normalized to [0, 1] via a
        sigmoid. A smaller distance means closer to the boundary (more important).
        """
        if not self._is_fitted:
            return np.ones(len(X)) * 0.5

        try:
            distances = np.abs(self.model.decision_function(X))
            # Sigmoid normalization: 1 / (1 + exp(-d + 1))
            # Large distance -> large value (far from boundary)
            # Small distance -> small value (close to boundary)
            normalized = 1.0 / (1.0 + np.exp(-distances + 1))
            return normalized
        except Exception:
            return np.ones(len(X)) * 0.5

    def get_feature_importance(self) -> np.ndarray:
        """Return feature importance."""
        if self._feature_importance is None:
            raise RuntimeError("Model is not trained; please call fit() first.")
        return self._feature_importance

    def clone(self) -> 'SVMAdapter':
        """Create an untrained clone."""
        return SVMAdapter(kernel=self.kernel, C=self.C, max_iter=self.max_iter)
