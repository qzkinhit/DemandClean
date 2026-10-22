"""
Base Model Adapter
==================

Defines the unified interface for model adapters, enabling support for multiple
machine learning models.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class ModelAdapter(ABC):
    """
    Abstract base class for model adapters.

    Unifies the interface for classification and regression models so that the
    DQN environment can interact with different backends.

    Core methods:
        - fit: Train the model
        - predict: Make predictions
        - evaluate: Evaluate model performance
        - get_distance_to_boundary: Return the distance to the decision boundary (critical!)
        - get_feature_importance: Return feature importance
    """

    def __init__(self):
        self.model = None
        self._feature_importance: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ModelAdapter':
        """
        Train the model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Label vector (n_samples,)

        Returns:
            self
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predictions (n_samples,)
        """
        pass

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate model performance.

        Args:
            X: Feature matrix
            y: Ground-truth labels

        Returns:
            Performance score (accuracy for classification, negative MSE for regression)
        """
        pass

    @abstractmethod
    def get_distance_to_boundary(self, X: np.ndarray) -> np.ndarray:
        """
        Return the distance to the decision boundary.

        This is a key component of the DQN state features.

        Classification: uses decision_function or predicted probabilities.
        Regression: uses the deviation of the prediction from the global mean.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Distance array (n_samples,), normalized to [0, 1]
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> np.ndarray:
        """
        Return feature importance.

        Returns:
            Feature-importance array whose elements sum to 1 after normalization
        """
        pass

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been trained."""
        return self._is_fitted

    def _normalize_importance(self, importance: np.ndarray) -> np.ndarray:
        """Normalize feature importance."""
        total = np.sum(np.abs(importance))
        if total < 1e-10:
            return np.ones_like(importance) / len(importance)
        return np.abs(importance) / total

    def clone(self) -> 'ModelAdapter':
        """
        Create an untrained clone.

        Returns:
            A new ModelAdapter instance
        """
        return type(self)()
