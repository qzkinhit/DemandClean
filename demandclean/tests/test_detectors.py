"""
Detector Module Tests
=====================
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# import pytest  # Optional
import numpy as np
from demandclean.detectors import ErrorInjector, RahaBasedDetector


def test_error_injector_init():
    """Test error injector initialization."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    injector = ErrorInjector(X, y)
    assert injector.X_base.shape == (100, 5)
    assert injector.y_base.shape == (100,)
    print("✓ Error injector initialization test passed")


def test_error_injector_missing():
    """Test missing value injection."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    injector = ErrorInjector(X, y)
    X_dirty, y_dirty, injected = injector.inject_errors(missing_rate=0.1)

    assert np.isnan(X_dirty).sum() > 0
    assert len(injected['missing']) > 0
    print(f"✓ Missing value injection test passed: {len(injected['missing'])} missing values")


def test_error_injector_semantic():
    """Test semantic error injection."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    injector = ErrorInjector(X, y)
    X_dirty, y_dirty, injected = injector.inject_errors(semantic_rate=0.1)

    assert len(injected['semantic']) > 0
    # Semantic errors replace original values
    diff_count = np.sum(X != X_dirty)
    assert diff_count >= len(injected['semantic'])
    print(f"✓ Semantic error injection test passed: {len(injected['semantic'])} semantic errors")


def test_error_injector_syntactic():
    """Test syntactic error injection."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    injector = ErrorInjector(X, y)
    X_dirty, y_dirty, injected = injector.inject_errors(syntactic_rate=0.15)

    assert len(injected['syntactic']) > 0
    print(f"✓ Syntactic error injection test passed: {len(injected['syntactic'])} syntactic errors")


def test_error_injector_combined():
    """Test combined error injection."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    injector = ErrorInjector(X, y)
    X_dirty, y_dirty, injected = injector.inject_errors(
        missing_rate=0.05,
        semantic_rate=0.1,
        syntactic_rate=0.15
    )

    total_errors = len(injected['missing']) + len(injected['semantic']) + len(injected['syntactic'])
    assert total_errors > 0
    print(f"✓ Combined error injection test passed: {total_errors} errors total")


def test_error_injector_build_error_list():
    """Test error list construction."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    injector = ErrorInjector(X, y)
    X_dirty, y_dirty, injected = injector.inject_errors(
        missing_rate=0.05,
        semantic_rate=0.1,
        syntactic_rate=0.15
    )

    error_list = injector.build_error_list(injected)

    assert len(error_list) > 0
    assert all('idx' in e and 'col' in e and 'type' in e for e in error_list)
    print(f"✓ Error list construction test passed: {len(error_list)} errors")


def test_raha_detector_init():
    """Test RAHA detector initialization (RahaBasedDetector is an alias of AutoDetector)."""
    detector = RahaBasedDetector()

    assert detector.is_fitted == False
    assert isinstance(detector.col_stats, dict)
    print("✓ RAHA detector initialization test passed")


def test_raha_detector_col_stats():
    """Test column statistics computation."""
    np.random.seed(42)
    X = np.random.randn(100, 3)

    detector = RahaBasedDetector()
    detector._compute_col_stats(X)

    assert len(detector.col_stats) == 3
    assert 'mean' in detector.col_stats[0]
    assert 'std' in detector.col_stats[0]
    print("✓ Column statistics test passed")


def test_raha_detector_detect_missing():
    """Test missing value detection."""
    np.random.seed(42)
    X = np.random.randn(100, 3)
    # Inject some NaN
    X[5, 0] = np.nan
    X[10, 1] = np.nan
    X[20, 2] = np.nan

    detector = RahaBasedDetector()
    detector._compute_col_stats(X)
    detected = detector.detect(X, verbose=False)

    assert len(detected['missing']) == 3
    print("✓ Missing value detection test passed")


def test_raha_detector_detect_with_semantic():
    """Test detection with semantic error positions."""
    np.random.seed(42)
    X = np.random.randn(100, 3)

    detector = RahaBasedDetector()
    detector._compute_col_stats(X)

    semantic_positions = [(5, 0), (10, 1), (15, 2)]
    detected = detector.detect(X, semantic_positions=semantic_positions, verbose=False)

    assert len(detected['semantic']) == 3
    print("✓ Semantic error detection test passed")


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("Detector Module Tests")
    print("=" * 50 + "\n")

    test_error_injector_init()
    test_error_injector_missing()
    test_error_injector_semantic()
    test_error_injector_syntactic()
    test_error_injector_combined()
    test_error_injector_build_error_list()
    test_raha_detector_init()
    test_raha_detector_col_stats()
    test_raha_detector_detect_missing()
    test_raha_detector_detect_with_semantic()

    print("\nAll tests passed!")
