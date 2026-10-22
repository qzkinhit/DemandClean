"""
XGBoost Regressor Adapter
=========================
"""

import numpy as np
from typing import Optional
import warnings

from ..base_adapter import ModelAdapter


class XGBoostRegressorAdapter(ModelAdapter):
    """
    XGBoost regressor adapter.

    Supports gradient-boosted regression.
    """

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize the XGBoost regression adapter.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            random_state: Random seed
            **kwargs: Additional arguments forwarded to XGBRegressor
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

    def _create_model(self):
        """Create the XGBoost regression model."""
        try:
            from xgboost import XGBRegressor
            self.model = XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                **self.kwargs
            )
        except ImportError:
            warnings.warn("XGBoost is not installed; falling back to GradientBoostingRegressor.")
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'XGBoostRegressorAdapter':
        """Train the XGBoost regression model."""
        if self.model is None:
            self._create_model()

        self.model.fit(X, y)
        self._y_mean = np.mean(y)
        self._y_std = np.std(y) + 1e-6
        self._is_fitted = True

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self._feature_importance = self._normalize_importance(
                self.model.feature_importances_
            )
        else:
            self._feature_importance = np.ones(X.shape[1]) / X.shape[1]

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

        Uses how far the prediction deviates from the mean.
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

    def clone(self) -> 'XGBoostRegressorAdapter':
        """Create an untrained clone."""
        return XGBoostRegressorAdapter(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            **self.kwargs
        )
