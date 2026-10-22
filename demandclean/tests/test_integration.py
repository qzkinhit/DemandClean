"""
Integration Tests
=================

Exercise the full pipeline with real data.
"""

import sys
import os
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_TEST_DIR, '..', '..'))
sys.path.insert(0, _PROJECT_ROOT)

# import pytest  # Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from demandclean import DemandClean


# Dataset path
DATASETS_PATH = os.path.join(_PROJECT_ROOT, 'experiment', 'ablation_beers', 'datasets')


def load_beers_data():
    """Load the beers dataset (classification task)."""
    clean_path = os.path.join(DATASETS_PATH, 'beers/clean.csv')
    dirty_path = os.path.join(DATASETS_PATH, 'beers/dirty.csv')

    if not os.path.exists(clean_path):
        return None, None, None

    clean_df = pd.read_csv(clean_path)
    dirty_df = pd.read_csv(dirty_path)

    # Select numeric columns
    feature_cols = ['abv', 'ibu']
    target_col = 'style'

    # Handle percentage strings (e.g. '0.09%' -> 0.09)
    def convert_to_float(x):
        if isinstance(x, str):
            if x.endswith('%'):
                try:
                    return float(x[:-1])
                except ValueError:
                    return np.nan
            try:
                return float(x)
            except ValueError:
                return np.nan
        return x

    for col in feature_cols:
        clean_df[col] = clean_df[col].apply(convert_to_float)
        dirty_df[col] = dirty_df[col].apply(convert_to_float)

    # Extract features and labels
    X_clean = clean_df[feature_cols].values.astype(np.float64)
    X_dirty = dirty_df[feature_cols].values.astype(np.float64)

    # Encode labels (convert to numeric)
    le = LabelEncoder()
    y = le.fit_transform(clean_df[target_col].values)

    return X_clean, X_dirty, y


def load_synthetic_data():
    """Generate synthetic data for testing."""
    np.random.seed(42)

    # Generate classification data
    n_samples = 200
    n_features = 4

    # Generate two classes
    X_clean = np.vstack([
        np.random.randn(n_samples // 2, n_features) + np.array([1, 1, 0, 0]),
        np.random.randn(n_samples // 2, n_features) + np.array([-1, -1, 0, 0])
    ])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    # Inject errors
    X_dirty = X_clean.copy()
    # Missing values
    for _ in range(10):
        i, j = np.random.randint(0, n_samples), np.random.randint(0, n_features)
        X_dirty[i, j] = np.nan
    # Syntactic errors (add noise)
    for _ in range(20):
        i, j = np.random.randint(0, n_samples), np.random.randint(0, n_features)
        X_dirty[i, j] += np.random.randn() * 3

    return X_clean, X_dirty, y


def test_integration_synthetic_single_stage():
    """Synthetic data, single-stage test."""
    print("\n" + "=" * 50)
    print("Integration: synthetic data - single stage")
    print("=" * 50)

    X_clean, X_dirty, y = load_synthetic_data()
    print(f"Data: {X_dirty.shape}")

    # Use only the first 100 rows for a quick test
    n = min(100, len(X_clean))
    X_clean, X_dirty, y = X_clean[:n], X_dirty[:n], y[:n]

    dc = DemandClean(
        task_type='classification',
        model_type='svm',
        agent_type='single_stage',
        n_episodes=5
    )

    # Train
    dc.fit(X_dirty, y, verbose=False)
    assert dc.is_fitted
    print("✓ Training completed")

    # Single-stage cleaning
    X_result, y_result, stats = dc.clean(X_dirty, y, X_clean, verbose=False)
    print(f"✓ Cleaning completed: {len(X_result)} rows")
    print(f"  actions: {stats['action_counts']}")
    print(f"  truth cost: {stats['truth_cost']}")

    return True


def test_integration_synthetic_two_stage():
    """Synthetic data, two-stage test."""
    print("\n" + "=" * 50)
    print("Integration: synthetic data - two stage")
    print("=" * 50)

    X_clean, X_dirty, y = load_synthetic_data()

    # Use only the first 100 rows for a quick test
    n = min(100, len(X_clean))
    X_clean, X_dirty, y = X_clean[:n], X_dirty[:n], y[:n]

    dc = DemandClean(
        task_type='classification',
        model_type='svm',
        agent_type='two_stage',
        n_episodes=5
    )

    # Train
    dc.fit(X_dirty, y, verbose=False)
    print("✓ Training completed")

    # Phase 1: plan
    plan = dc.plan(X_dirty, y, verbose=False)
    print(f"✓ Plan generated: {len(plan)} truth-value requests")

    # Get the positions that need ground truth
    positions = dc.get_plan_positions()

    # Extract true values from X_clean
    true_values = {}
    for idx, col in positions:
        true_values[(idx, col)] = X_clean[idx, col]

    # Phase 2: execute
    X_result, y_result, keep_mask = dc.execute(X_dirty, true_values, verbose=False)
    print(f"✓ Execution completed: {len(X_result)} rows")

    return True


def test_integration_beers_data():
    """Real data (beers) test."""
    print("\n" + "=" * 50)
    print("Integration: beers data")
    print("=" * 50)

    X_clean, X_dirty, y = load_beers_data()
    if X_clean is None:
        print("! Could not load beers data; skipping")
        return True

    print(f"Data: clean={X_clean.shape}, dirty={X_dirty.shape}")

    # Use only the first 100 rows for a quick test
    n = min(100, len(X_clean))
    X_clean, X_dirty, y = X_clean[:n], X_dirty[:n], y[:n]

    dc = DemandClean(
        task_type='classification',
        model_type='random_forest',
        agent_type='single_stage',
        n_episodes=5
    )

    # Train
    dc.fit(X_dirty, y, verbose=False)
    print("✓ Training completed")

    # Detect errors
    detected = dc.detect_errors(X_dirty, X_clean, verbose=False)
    print(f"✓ Detected: missing={len(detected['missing'])}, "
          f"semantic={len(detected['semantic'])}, syntactic={len(detected['syntactic'])}")

    # Clean
    X_result, y_result, stats = dc.clean(X_dirty, y, X_clean, verbose=False)
    print(f"✓ Cleaning completed: {len(X_result)} rows")
    print(f"  actions: {stats['action_counts']}")

    return True


def test_integration_regression():
    """Regression task test."""
    print("\n" + "=" * 50)
    print("Integration: regression task")
    print("=" * 50)

    np.random.seed(42)
    n_samples = 100
    n_features = 4

    # Generate regression data
    X_clean = np.random.randn(n_samples, n_features)
    y = X_clean[:, 0] * 2 + X_clean[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1

    # Inject errors
    X_dirty = X_clean.copy()
    for _ in range(10):
        i, j = np.random.randint(0, n_samples), np.random.randint(0, n_features)
        X_dirty[i, j] = np.nan

    dc = DemandClean(
        task_type='regression',
        model_type='ridge',
        agent_type='single_stage',
        n_episodes=5
    )

    # Train
    dc.fit(X_dirty, y, verbose=False)
    print("✓ Training completed")

    # Clean
    X_result, y_result, stats = dc.clean(X_dirty, y, X_clean, verbose=False)
    print(f"✓ Cleaning completed: {len(X_result)} rows")
    print(f"  actions: {stats['action_counts']}")

    return True


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("DemandClean Integration Tests")
    print("=" * 60)

    tests = [
        ("Synthetic - single stage", test_integration_synthetic_single_stage),
        ("Synthetic - two stage", test_integration_synthetic_two_stage),
        ("Beers data", test_integration_beers_data),
        ("Regression task", test_integration_regression),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ {name} test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Integration result: {passed} passed, {failed} failed")
    print("=" * 60)
