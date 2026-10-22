"""
DemandClean API Tests
=====================
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# import pytest  # Optional
import numpy as np
import tempfile
import os
from demandclean import DemandClean
from demandclean.utils.model_io import ModelIO


def test_demandclean_init_classification():
    """Test classification task initialization."""
    dc = DemandClean(
        task_type='classification',
        model_type='svm',
        agent_type='single_stage',
        n_episodes=5
    )

    assert dc.config.task_type.value == 'classification'
    assert dc.config.model_type.value == 'svm'
    assert dc.config.agent_type.value == 'single'
    assert not dc.is_fitted
    print("✓ Classification task initialization test passed")


def test_demandclean_init_regression():
    """Test regression task initialization."""
    dc = DemandClean(
        task_type='regression',
        model_type='ridge',
        agent_type='two_stage',
        n_episodes=5
    )

    assert dc.config.task_type.value == 'regression'
    assert dc.config.model_type.value == 'ridge'
    assert dc.config.agent_type.value == 'two_stage'
    print("✓ Regression task initialization test passed")


def test_demandclean_fit():
    """Test training functionality."""
    np.random.seed(42)
    X = np.random.randn(50, 4)
    y = np.random.randint(0, 2, 50)

    dc = DemandClean(
        task_type='classification',
        model_type='random_forest',
        agent_type='single_stage',
        n_episodes=3
    )

    dc.fit(X, y, verbose=False)
    assert dc.is_fitted
    assert dc.agent is not None
    print("✓ Training functionality test passed")


def test_demandclean_fit_with_semantic_errors():
    """Test training with semantic errors."""
    np.random.seed(42)
    X = np.random.randn(50, 4)
    y = np.random.randint(0, 2, 50)
    semantic_errors = [(5, 0), (10, 1), (15, 2)]

    dc = DemandClean(
        task_type='classification',
        model_type='svm',
        agent_type='single_stage',
        n_episodes=3
    )

    dc.fit(X, y, semantic_errors=semantic_errors, verbose=False)
    assert dc.is_fitted
    print("✓ Training with semantic errors test passed")


def test_demandclean_detect_errors():
    """Test error detection."""
    np.random.seed(42)
    X_dirty = np.random.randn(50, 4)
    X_dirty[5, 0] = np.nan
    X_dirty[10, 1] = np.nan
    y = np.random.randint(0, 2, 50)

    dc = DemandClean(
        task_type='classification',
        model_type='svm',
        n_episodes=3
    )
    dc.fit(X_dirty, y, verbose=False)

    detected = dc.detect_errors(X_dirty, verbose=False)
    assert 'missing' in detected
    assert 'semantic' in detected
    assert 'syntactic' in detected
    assert len(detected['missing']) >= 2  # at least the 2 we injected
    print(f"✓ Error detection test passed: {len(detected['missing'])} missing values")


def test_demandclean_get_config():
    """Test configuration retrieval."""
    dc = DemandClean(
        task_type='classification',
        model_type='svm',
        n_episodes=100,
        repair_lambda=0.05
    )

    config = dc.get_config()
    assert config.n_episodes == 100
    assert config.repair_lambda == 0.05
    print("✓ Get config test passed")


def test_demandclean_get_training_history():
    """Test training history retrieval."""
    np.random.seed(42)
    X = np.random.randn(30, 3)
    y = np.random.randint(0, 2, 30)

    dc = DemandClean(
        task_type='classification',
        model_type='svm',
        n_episodes=3
    )
    dc.fit(X, y, verbose=False)

    history = dc.get_training_history()
    assert 'episode' in history
    assert 'score' in history
    assert 'reward' in history
    assert len(history['episode']) == 3
    print("✓ Get training history test passed")


def test_demandclean_save_load():
    """Test model save and load."""
    np.random.seed(42)
    X = np.random.randn(30, 3)
    y = np.random.randint(0, 2, 30)

    # Train
    dc = DemandClean(
        task_type='classification',
        model_type='svm',
        n_episodes=3
    )
    dc.fit(X, y, verbose=False)

    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'model.pt')
        dc.save(model_path)
        assert ModelIO.agent_model_exists(model_path)

        # Load
        dc2 = DemandClean(
            task_type='classification',
            model_type='svm'
        )
        dc2.load(model_path)
        assert dc2.is_fitted

    print("✓ Model save/load test passed")


def test_demandclean_not_fitted_error():
    """Test error when calling without training."""
    dc = DemandClean()

    try:
        dc.clean(np.random.randn(10, 3), np.random.randint(0, 2, 10),
                 np.random.randn(10, 3), verbose=False)
        assert False, "Should raise an exception"
    except ValueError as e:
        assert "not trained" in str(e).lower()
        print("✓ Not-trained error test passed")


def test_demandclean_plan_not_fitted_error():
    """Test error when calling plan without training."""
    dc = DemandClean()

    try:
        dc.plan(np.random.randn(10, 3), np.random.randint(0, 2, 10), verbose=False)
        assert False, "Should raise an exception"
    except ValueError as e:
        assert "not trained" in str(e).lower()
        print("✓ plan not-trained error test passed")


def test_demandclean_execute_no_plan_error():
    """Test error when calling execute without plan."""
    np.random.seed(42)
    X = np.random.randn(30, 3)
    y = np.random.randint(0, 2, 30)

    dc = DemandClean(n_episodes=3)
    dc.fit(X, y, verbose=False)

    try:
        dc.execute(X, {}, verbose=False)
        assert False, "Should raise an exception"
    except ValueError as e:
        assert "plan" in str(e).lower()
        print("✓ execute without plan error test passed")


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("DemandClean API Tests")
    print("=" * 50 + "\n")

    test_demandclean_init_classification()
    test_demandclean_init_regression()
    test_demandclean_fit()
    test_demandclean_fit_with_semantic_errors()
    test_demandclean_detect_errors()
    test_demandclean_get_config()
    test_demandclean_get_training_history()
    test_demandclean_save_load()
    test_demandclean_not_fitted_error()
    test_demandclean_plan_not_fitted_error()
    test_demandclean_execute_no_plan_error()

    print("\nAll tests passed!")
