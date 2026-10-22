"""
XGBoost Classifier Adapter
==========================
"""

import numpy as np
from typing import Optional
import warnings

from ..base_adapter import ModelAdapter


class XGBoostClassifierAdapter(ModelAdapter):
    """
    XGBoost classifier adapter.

    Supports gradient-boosted classification.
    """

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize the XGBoost adapter.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            random_state: Random seed
            **kwargs: Additional arguments forwarded to XGBClassifier
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = None
        self._y_classes: Optional[np.ndarray] = None

    def _create_model(self):
        """Create the XGBoost model."""
        try:
            from xgboost import XGBClassifier
            self.model = XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                **self.kwargs
            )
        except ImportError:
            warnings.warn("XGBoost is not installed; falling back to RandomForest.")
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'XGBoostClassifierAdapter':
        """Train the XGBoost model."""
        if self.model is None:
            self._create_model()

        self.model.fit(X, y)
        self._y_classes = np.unique(y)
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
        """Compute accuracy."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def get_distance_to_boundary(self, X: np.ndarray) -> np.ndarray:
        """
        Return the distance to the decision boundary.

        Uses predicted probabilities.
        """
        if not self._is_fitted:
            return np.ones(len(X)) * 0.5

        try:
            proba = self.model.predict_proba(X)
            max_proba = np.max(proba, axis=1)
            distances = (max_proba - 0.5) * 2
            return np.clip(distances, 0, 1)
        except Exception:
            return np.ones(len(X)) * 0.5

    def get_feature_importance(self) -> np.ndarray:
        """Return feature importance."""
        if self._feature_importance is None:
            raise RuntimeError("Model is not trained; please call fit() first.")
        return self._feature_importance

    def clone(self) -> 'XGBoostClassifierAdapter':
        """Create an untrained clone."""
        return XGBoostClassifierAdapter(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            **self.kwargs
        )
