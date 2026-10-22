"""
RandomForest Regressor Adapter
==============================
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import Optional

from ..base_adapter import ModelAdapter


class RandomForestRegressorAdapter(ModelAdapter):
    """
    Random forest regressor adapter.

    Provides feature importance and regression predictions.
    """

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize the random forest regression adapter.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            random_state: Random seed
            **kwargs: Additional arguments forwarded to RandomForestRegressor
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestRegressorAdapter':
        """Train the random forest regression model."""
        self.model.fit(X, y)
        self._y_mean = np.mean(y)
        self._y_std = np.std(y) + 1e-6
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
        """
        Compute the negative MSE.

        Returns negative MSE; closer to 0 is better.
        """
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        return -mse

    def get_distance_to_boundary(self, X: np.ndarray) -> np.ndarray:
        """
        Return the distance to the "boundary".

        For regression, uses how far the prediction deviates from the mean as
        the "distance".
        """
        if not self._is_fitted:
            return np.ones(len(X)) * 0.5

        try:
            predictions = self.predict(X)
            influence = np.abs(predictions - self._y_mean) / (self._y_std * 2)
            return np.clip(influence, 0, 1)
        except Exception:
            return np.ones(len(X)) * 0.5

    def get_feature_importance(self) -> np.ndarray:
        """Return feature importance."""
        if self._feature_importance is None:
            raise RuntimeError("Model is not trained; please call fit() first.")
        return self._feature_importance

    def clone(self) -> 'RandomForestRegressorAdapter':
        """Create an untrained clone."""
        return RandomForestRegressorAdapter(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
