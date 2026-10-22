"""Comet (Mohammed, Naumann, Harmouch, EDBT 2025): step-by-step feature-wise cleaning recommendations."""

from .comet import (  # noqa: F401
    CATEGORICAL_ERROR_TYPES,
    ERROR_CODES,
    ERROR_NAMES,
    NUMERICAL_ERROR_TYPES,
    POLLUTER_SEEDS,
    BayesianPolynomialEstimator,
    Comet,
    CometResult,
    Polluter,
    classify_errors,
    prepare_matrix,
)
