"""
Evaluation Metrics
==================

Metric computation for classification and regression tasks.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np


class Metrics:
    """Evaluation metric utilities."""

    @staticmethod
    def classification_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute classification accuracy.

        Args:
            y_true: Ground-truth labels
            y_pred: Predicted labels

        Returns:
            Accuracy in [0, 1]
        """
        return np.mean(y_true == y_pred)

    @staticmethod
    def regression_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute regression score (negative MSE).

        Args:
            y_true: Ground-truth values
            y_pred: Predicted values

        Returns:
            Negative MSE (closer to 0 is better)
        """
        mse = np.mean((y_true - y_pred) ** 2)
        return -mse

    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the R2 coefficient of determination.

        Args:
            y_true: Ground-truth values
            y_pred: Predicted values

        Returns:
            R2 value
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot < 1e-10:
            return 1.0
        return 1 - (ss_res / ss_tot)

    @staticmethod
    def authenticity(X_result: np.ndarray,
                     X_clean: np.ndarray,
                     keep_mask: Optional[np.ndarray] = None,
                     col: int = 1,
                     tolerance: float = 0.01) -> float:
        """
        Compute data authenticity.

        Authenticity = number of correct values / current number of rows.

        Args:
            X_result: Cleaned data
            X_clean: Clean reference data
            keep_mask: Mask indicating kept rows
            col: Column index to compare
            tolerance: Tolerance threshold

        Returns:
            Authenticity in [0, 1]
        """
        if len(X_result) == 0:
            return 0.0

        if keep_mask is not None:
            X_clean_kept = X_clean[keep_mask]
        else:
            X_clean_kept = X_clean

        n = min(len(X_result), len(X_clean_kept))
        correct = 0
        for i in range(n):
            if abs(X_result[i, col] - X_clean_kept[i, col]) < tolerance:
                correct += 1

        return correct / len(X_result)

    @staticmethod
    def diversity(X_result: np.ndarray,
                  X_clean: np.ndarray,
                  X_dirty: np.ndarray,
                  keep_mask: Optional[np.ndarray] = None,
                  col: int = 1) -> Tuple[float, float, float]:
        """
        Compute data diversity.

        Diversity = sample retention rate x variance retention rate.

        Args:
            X_result: Cleaned data
            X_clean: Clean reference data
            X_dirty: Dirty data
            keep_mask: Mask indicating kept rows
            col: Column index used to compute variance

        Returns:
            (diversity, sample_retention, var_retention)
        """
        n_total = len(X_clean)
        n_result = len(X_result)

        if n_result == 0:
            return 0.0, 0.0, 0.0

        # Sample retention rate
        sample_retention = n_result / n_total

        # Variance retention rate
        if n_result < 2:
            return 0.0, sample_retention, 0.0

        if keep_mask is not None:
            X_clean_kept = X_clean[keep_mask]
        else:
            X_clean_kept = X_clean[:n_result]

        result_var = np.var(X_result[:, col])
        clean_var = np.var(X_clean_kept[:, col])

        if clean_var > 1e-6:
            var_retention = np.clip(result_var / clean_var, 0, 1.5)
        else:
            var_retention = 1.0

        diversity = sample_retention * var_retention

        return diversity, sample_retention, var_retention

    @staticmethod
    def compute_all(X_result: np.ndarray,
                    y_result: np.ndarray,
                    X_clean: np.ndarray,
                    y_clean: np.ndarray,
                    X_dirty: np.ndarray,
                    keep_mask: Optional[np.ndarray] = None,
                    col: int = 1) -> Dict[str, Any]:
        """
        Compute all metrics.

        Args:
            X_result: Cleaned features
            y_result: Cleaned labels
            X_clean: Clean features
            y_clean: Clean labels
            X_dirty: Dirty features
            keep_mask: Mask indicating kept rows
            col: Column index used for metrics

        Returns:
            Dictionary containing all metrics
        """
        # Authenticity
        auth = Metrics.authenticity(X_result, X_clean, keep_mask, col)

        # Diversity
        div, sample_ret, var_ret = Metrics.diversity(
            X_result, X_clean, X_dirty, keep_mask, col
        )

        return {
            'authenticity': auth,
            'diversity': div,
            'sample_retention': sample_ret,
            'var_retention': var_ret,
            'n_samples': len(X_result),
            'n_original': len(X_clean),
        }

    @staticmethod
    def action_distribution(action_counts: Dict[str, int]) -> Dict[str, float]:
        """
        Compute the action-distribution percentages.

        Args:
            action_counts: Action count dictionary

        Returns:
            Action percentage dictionary
        """
        total = sum(action_counts.values())
        if total == 0:
            return {k: 0.0 for k in action_counts}

        return {k: v / total for k, v in action_counts.items()}
