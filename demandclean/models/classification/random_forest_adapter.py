"""
RandomForest Classifier Adapter
===============================
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Optional

from ..base_adapter import ModelAdapter


class RandomForestAdapter(ModelAdapter):
    """
    Random forest classifier adapter.

    Provides feature importance and predicted probabilities.
    """

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize the random forest adapter.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            random_state: Random seed
            **kwargs: Additional arguments forwarded to RandomForestClassifier
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )
        self._y_classes: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestAdapter':
        """Train the random forest model."""
        self.model.fit(X, y)
        self._y_classes = np.unique(y)
        self._is_fitted = True

        # Feature importance
        self._feature_importance = self._normalize_importance(
            self.model.feature_importances_
        )

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

        Uses the maximum predicted probability as a confidence score.
        High confidence -> far from the boundary -> large distance.
        Low confidence (close to 0.5) -> close to the boundary -> small distance.
        """
        if not self._is_fitted:
            return np.ones(len(X)) * 0.5

        try:
            proba = self.model.predict_proba(X)
            # Use the max probability as the confidence
            max_proba = np.max(proba, axis=1)
            # Confidence 0.5 -> 0, confidence 1.0 -> 1
            # (max_proba - 0.5) * 2 maps [0.5, 1] to [0, 1]
            distances = (max_proba - 0.5) * 2
            return np.clip(distances, 0, 1)
        except Exception:
            return np.ones(len(X)) * 0.5

    def get_feature_importance(self) -> np.ndarray:
        """Return feature importance."""
        if self._feature_importance is None:
            raise RuntimeError("Model is not trained; please call fit() first.")
        return self._feature_importance

    def clone(self) -> 'RandomForestAdapter':
        """Create an untrained clone."""
        return RandomForestAdapter(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
