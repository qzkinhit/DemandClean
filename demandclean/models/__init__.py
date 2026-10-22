"""Model adapter module"""

from .base_adapter import ModelAdapter
from .classification import SVMAdapter, RandomForestAdapter, XGBoostClassifierAdapter
from .regression import LinearAdapter, RidgeAdapter, XGBoostRegressorAdapter, RandomForestRegressorAdapter
from .clustering import KMeansAdapter

__all__ = [
    'ModelAdapter',
    'SVMAdapter',
    'RandomForestAdapter',
    'XGBoostClassifierAdapter',
    'LinearAdapter',
    'RidgeAdapter',
    'XGBoostRegressorAdapter',
    'RandomForestRegressorAdapter',
    'KMeansAdapter',
    'get_adapter',
    'create_model_adapter',
]


def get_adapter(model_type: str, task_type: str = 'classification', **kwargs):
    """
    Get an adapter by model type.

    Args:
        model_type: Model type ('svm', 'random_forest', 'xgboost', 'linear', 'ridge', 'xgboost_reg', 'kmeans')
        task_type: Task type ('classification', 'regression', 'clustering')
        **kwargs: Arguments forwarded to the adapter

    Returns:
        ModelAdapter instance
    """
    adapters = {
        # Classification
        ('svm', 'classification'): SVMAdapter,
        ('random_forest', 'classification'): RandomForestAdapter,
        ('xgboost', 'classification'): XGBoostClassifierAdapter,
        # Regression
        ('linear', 'regression'): LinearAdapter,
        ('ridge', 'regression'): RidgeAdapter,
        ('xgboost_reg', 'regression'): XGBoostRegressorAdapter,
        ('random_forest', 'regression'): RandomForestRegressorAdapter,
        # Clustering
        ('kmeans', 'clustering'): KMeansAdapter,
    }

    key = (model_type, task_type)
    if key not in adapters:
        raise ValueError(f"Unsupported model type: {model_type} for {task_type}")

    return adapters[key](**kwargs)


def create_model_adapter(model_type, task_type, **kwargs):
    """
    Create a model adapter from enum values.

    Args:
        model_type: ModelType enum or string
        task_type: TaskType enum or string
        **kwargs: Arguments forwarded to the adapter

    Returns:
        ModelAdapter instance
    """
    from ..config import ModelType, TaskType

    # Handle enum inputs
    if hasattr(model_type, 'value'):
        model_type_str = model_type.value
    else:
        model_type_str = str(model_type)

    if hasattr(task_type, 'value'):
        task_type_str = task_type.value
    else:
        task_type_str = str(task_type)

    # Model type mapping
    model_map = {
        'svm': 'svm',
        'random_forest': 'random_forest',
        'xgboost': 'xgboost',
        'linear': 'linear',
        'ridge': 'ridge',
        'xgboost_reg': 'xgboost_reg',
        'kmeans': 'kmeans',
    }

    model_key = model_map.get(model_type_str, model_type_str)

    return get_adapter(model_key, task_type_str, **kwargs)
