"""
Model Adapter Tests
===================
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# import pytest  # Optional
import numpy as np
from demandclean.models import create_model_adapter
from demandclean.config import ModelType, TaskType


def test_create_svm_adapter():
    """Test SVM adapter creation."""
    adapter = create_model_adapter(ModelType.SVM, TaskType.CLASSIFICATION)
    assert adapter is not None
    print("✓ SVM adapter creation test passed")


def test_create_random_forest_adapter():
    """Test RandomForest adapter creation."""
    adapter = create_model_adapter(ModelType.RANDOM_FOREST, TaskType.CLASSIFICATION)
    assert adapter is not None
    print("✓ RandomForest adapter creation test passed")


def test_create_linear_adapter():
    """Test Linear adapter creation."""
    adapter = create_model_adapter(ModelType.LINEAR, TaskType.REGRESSION)
    assert adapter is not None
    print("✓ Linear adapter creation test passed")


def test_create_ridge_adapter():
    """Test Ridge adapter creation."""
    adapter = create_model_adapter(ModelType.RIDGE, TaskType.REGRESSION)
    assert adapter is not None
    print("✓ Ridge adapter creation test passed")


def test_create_adapter_with_string():
    """Test adapter creation using strings."""
    adapter = create_model_adapter('svm', 'classification')
    assert adapter is not None
    print("✓ String-based adapter creation test passed")


def test_svm_fit_predict():
    """Test SVM train and predict."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    adapter = create_model_adapter(ModelType.SVM, TaskType.CLASSIFICATION)
    adapter.fit(X, y)

    pred = adapter.predict(X)
    assert len(pred) == 100
    assert all(p in [0, 1] for p in pred)
    print("✓ SVM train/predict test passed")


def test_svm_evaluate():
    """Test SVM evaluation."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    adapter = create_model_adapter(ModelType.SVM, TaskType.CLASSIFICATION)
    adapter.fit(X, y)

    score = adapter.evaluate(X, y)
    assert 0 <= score <= 1  # Accuracy in [0, 1]
    print(f"✓ SVM evaluation test passed: accuracy={score:.4f}")


def test_svm_distance_to_boundary():
    """Test SVM distance-to-boundary."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    adapter = create_model_adapter(ModelType.SVM, TaskType.CLASSIFICATION)
    adapter.fit(X, y)

    distance = adapter.get_distance_to_boundary(X[:5])
    assert len(distance) == 5
    assert all(0 <= d <= 1 for d in distance)
    print(f"✓ SVM distance-to-boundary test passed: {distance[:3]}")


def test_svm_feature_importance():
    """Test SVM feature importance."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    adapter = create_model_adapter(ModelType.SVM, TaskType.CLASSIFICATION)
    adapter.fit(X, y)

    importance = adapter.get_feature_importance()
    assert len(importance) == 5
    assert np.isclose(importance.sum(), 1.0)  # Normalized
    print(f"✓ SVM feature importance test passed: {importance}")


def test_random_forest_fit_predict():
    """Test RandomForest train and predict."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    adapter = create_model_adapter(ModelType.RANDOM_FOREST, TaskType.CLASSIFICATION)
    adapter.fit(X, y)

    pred = adapter.predict(X)
    assert len(pred) == 100
    print("✓ RandomForest train/predict test passed")


def test_random_forest_distance_to_boundary():
    """Test RandomForest distance-to-boundary."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    adapter = create_model_adapter(ModelType.RANDOM_FOREST, TaskType.CLASSIFICATION)
    adapter.fit(X, y)

    distance = adapter.get_distance_to_boundary(X[:5])
    assert len(distance) == 5
    assert all(0 <= d <= 1 for d in distance)
    print(f"✓ RandomForest distance-to-boundary test passed: {distance[:3]}")


def test_linear_fit_predict():
    """Test Linear regression train and predict."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randn(100)

    adapter = create_model_adapter(ModelType.LINEAR, TaskType.REGRESSION)
    adapter.fit(X, y)

    pred = adapter.predict(X)
    assert len(pred) == 100
    print("✓ Linear regression train/predict test passed")


def test_linear_evaluate():
    """Test Linear regression evaluation."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randn(100)

    adapter = create_model_adapter(ModelType.LINEAR, TaskType.REGRESSION)
    adapter.fit(X, y)

    score = adapter.evaluate(X, y)
    assert score <= 0  # Negative MSE
    print(f"✓ Linear regression evaluation test passed: -MSE={score:.4f}")


def test_ridge_fit_predict():
    """Test Ridge regression train and predict."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randn(100)

    adapter = create_model_adapter(ModelType.RIDGE, TaskType.REGRESSION)
    adapter.fit(X, y)

    pred = adapter.predict(X)
    assert len(pred) == 100
    print("✓ Ridge regression train/predict test passed")


def test_ridge_distance_to_boundary():
    """Test Ridge regression influence."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randn(100)

    adapter = create_model_adapter(ModelType.RIDGE, TaskType.REGRESSION)
    adapter.fit(X, y)

    distance = adapter.get_distance_to_boundary(X[:5])
    assert len(distance) == 5
    assert all(0 <= d <= 1 for d in distance)
    print(f"✓ Ridge regression influence test passed: {distance[:3]}")


def test_model_clone():
    """Test model clone."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    adapter = create_model_adapter(ModelType.SVM, TaskType.CLASSIFICATION)
    adapter.fit(X, y)

    cloned = adapter.clone()
    assert not cloned._is_fitted  # Clone should be untrained
    print("✓ Model clone test passed")


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("Model Adapter Tests")
    print("=" * 50 + "\n")

    test_create_svm_adapter()
    test_create_random_forest_adapter()
    test_create_linear_adapter()
    test_create_ridge_adapter()
    test_create_adapter_with_string()
    test_svm_fit_predict()
    test_svm_evaluate()
    test_svm_distance_to_boundary()
    test_svm_feature_importance()
    test_random_forest_fit_predict()
    test_random_forest_distance_to_boundary()
    test_linear_fit_predict()
    test_linear_evaluate()
    test_ridge_fit_predict()
    test_ridge_distance_to_boundary()
    test_model_clone()

    print("\nAll tests passed!")
