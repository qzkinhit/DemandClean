"""
Linear Regression Adapter
=========================
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Optional

from ..base_adapter import ModelAdapter


class LinearAdapter(ModelAdapter):
    """
    Linear regression adapter.
    """

    def __init__(self, **kwargs):
        """
        Initialize the linear regression adapter.

        Args:
            **kwargs: Arguments forwarded to LinearRegression
        """
        super().__init__()
        self.kwargs = kwargs
        self.model = LinearRegression(**kwargs)
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearAdapter':
        """Train the linear regression model."""
        self.model.fit(X, y)
        self._y_mean = np.mean(y)
        self._y_std = np.std(y) + 1e-6
        self._is_fitted = True

        # Feature importance based on absolute coefficients
        self._feature_importance = self._normalize_importance(self.model.coef_)

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

        For regression, the "boundary" is redefined as how strongly the
        prediction for this point influences the model:

        Prediction close to the mean -> small influence -> small distance
        Prediction far from the mean -> large influence -> large distance
        """
        if not self._is_fitted:
            return np.ones(len(X)) * 0.5

        try:
            predictions = self.predict(X)
            # Normalized influence
            influence = np.abs(predictions - self._y_mean) / (self._y_std * 2)
            return np.clip(influence, 0, 1)
        except Exception:
            return np.ones(len(X)) * 0.5

    def get_feature_importance(self) -> np.ndarray:
        """Return feature importance."""
        if self._feature_importance is None:
            raise RuntimeError("Model is not trained; please call fit() first.")
        return self._feature_importance

    def clone(self) -> 'LinearAdapter':
        """Create an untrained clone."""
        return LinearAdapter(**self.kwargs)
