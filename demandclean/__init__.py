"""
DemandClean - Demand-Driven Data Cleaning System
=================================================

A self-supervised data cleaning framework based on deep reinforcement learning.

Key features:
    - Self-supervised learning without access to clean data during training
    - Supports both classification and regression tasks
    - Supports multiple machine learning models (SVM, RandomForest, XGBoost, etc.)
    - Provides single-phase and two-phase inference modes

Usage example:
    >>> from demandclean import DemandClean
    >>>
    >>> # Create an instance
    >>> dc = DemandClean(
    ...     task_type='classification',
    ...     model_type='random_forest',
    ...     max_truth_budget=50
    ... )
    >>>
    >>> # Train (no clean data required)
    >>> dc.fit(X_dirty, y, semantic_errors=[(10, 1), (25, 1)])
    >>>
    >>> # Single-phase inference
    >>> X_clean, y_clean, stats = dc.clean(X_dirty, y, X_clean_ref)
    >>>
    >>> # Or two-phase inference
    >>> plan = dc.plan(X_dirty, y)
    >>> X_clean = dc.execute(X_dirty, plan, true_values)
"""

__version__ = '1.0.0'
__author__ = 'DemandClean Team'

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == 'DemandClean':
        from .api.demand_clean import DemandClean
        return DemandClean
    elif name == 'DemandCleanConfig':
        from .config.config import DemandCleanConfig
        return DemandCleanConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'DemandClean',
    'DemandCleanConfig',
]
