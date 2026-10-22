"""
DemandClean Functional Tests
============================

Exercise the full system on real data.
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Add project path
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_TEST_DIR, '..', '..'))
sys.path.insert(0, _PROJECT_ROOT)

# Dataset path
DATASETS_PATH = os.path.join(_PROJECT_ROOT, 'experiment', 'ablation_beers', 'datasets')


def load_beers_data():
    """Load the beers dataset (classification task)."""
    clean_path = os.path.join(DATASETS_PATH, 'beers/clean.csv')
    dirty_path = os.path.join(DATASETS_PATH, 'beers/dirty.csv')

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


def load_bike_data():
    """Load the bike dataset (regression task)."""
    clean_path = os.path.join(DATASETS_PATH, 'bike/clean.csv')
    dirty_path = os.path.join(DATASETS_PATH, 'bike/dirty.csv')

    clean_df = pd.read_csv(clean_path)
    dirty_df = pd.read_csv(dirty_path)

    # Select numeric columns (adjust to actual column names)
    numeric_cols = clean_df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        return None, None, None

    # Use the last column as the target
    feature_cols = numeric_cols[:-1]
    target_col = numeric_cols[-1]

    X_clean = clean_df[feature_cols].values.astype(np.float64)
    X_dirty = dirty_df[feature_cols].values.astype(np.float64)
    y = clean_df[target_col].values.astype(np.float64)

    return X_clean, X_dirty, y


def test_error_injector():
    """Test the error injector."""
    print("\n" + "=" * 60)
    print("Test: ErrorInjector")
    print("=" * 60)

    from demandclean.detectors import ErrorInjector

    # Generate test data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    injector = ErrorInjector(X, y)
    X_dirty, y_dirty, injected = injector.inject_errors(
        missing_rate=0.05,
        semantic_rate=0.1,
        syntactic_rate=0.15
    )

    print(f"✓ Original data: {X.shape}")
    print(f"✓ Injected missing values: {len(injected['missing'])}")
    print(f"✓ Injected semantic errors: {len(injected['semantic'])}")
    print(f"✓ Injected syntactic errors: {len(injected['syntactic'])}")
    print(f"✓ NaN count after injection: {np.isnan(X_dirty).sum()}")

    # Convert to an error list
    error_list = injector.build_error_list(injected)
    print(f"✓ Error list length: {len(error_list)}")

    return True


def test_model_adapters_with_data():
    """Test model adapters with real data."""
    print("\n" + "=" * 60)
    print("Test: ModelAdapters with Real Data")
    print("=" * 60)

    from demandclean.models import create_model_adapter
    from demandclean.config import ModelType, TaskType

    # Generate test data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y_class = np.random.randint(0, 2, 100)
    y_reg = np.random.randn(100)

    # Test classification models
    for model_type in [ModelType.SVM, ModelType.RANDOM_FOREST]:
        adapter = create_model_adapter(model_type, TaskType.CLASSIFICATION)
        adapter.fit(X, y_class)
        acc = adapter.evaluate(X, y_class)
        distance = adapter.get_distance_to_boundary(X[:5])
        print(f"✓ {model_type.value}: accuracy={acc:.4f}, boundary distance={distance[:3]}")

    # Test regression models
    for model_type in [ModelType.LINEAR, ModelType.RIDGE]:
        adapter = create_model_adapter(model_type, TaskType.REGRESSION)
        adapter.fit(X, y_reg)
        score = adapter.evaluate(X, y_reg)
        distance = adapter.get_distance_to_boundary(X[:5])
        print(f"✓ {model_type.value}: score={score:.4f}, influence={distance[:3]}")

    return True


def test_cleaning_env():
    """Test the cleaning environment."""
    print("\n" + "=" * 60)
    print("Test: CleaningEnv")
    print("=" * 60)

    from demandclean.core.environments import CleaningEnv
    from demandclean.core.state import ClassificationStateExtractor
    from demandclean.models import create_model_adapter
    from demandclean.config import DemandCleanConfig, ModelType, TaskType
    from demandclean.detectors import ErrorInjector

    # Generate test data
    np.random.seed(42)
    X = np.random.randn(50, 4)
    y = np.random.randint(0, 2, 50)

    # Inject errors
    injector = ErrorInjector(X, y)
    X_dirty, y_dirty, injected = injector.inject_errors(0.05, 0.1, 0.15)
    error_list = injector.build_error_list(injected)

    # Create components
    config = DemandCleanConfig(
        task_type=TaskType.CLASSIFICATION,
        model_type=ModelType.SVM
    )
    model_adapter = create_model_adapter(config.model_type, config.task_type)
    state_extractor = ClassificationStateExtractor(model_adapter, config)

    # Create environment
    env = CleaningEnv(X_dirty, y, error_list, model_adapter, state_extractor, config)

    # Reset environment
    state = env.reset()
    print(f"✓ Initial state dim: {state.shape}")

    # Simulate a few steps
    total_reward = 0
    steps = 0
    while steps < 10:
        action = np.random.randint(0, 4)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break

    print(f"✓ Steps executed: {steps}")
    print(f"✓ Cumulative reward: {total_reward:.4f}")
    print(f"✓ Action counts: {env.get_action_counts()}")

    return True


def test_dqn_agent():
    """Test the DQN Agent."""
    print("\n" + "=" * 60)
    print("Test: DQN Agent")
    print("=" * 60)

    from demandclean.core.agents import SingleStageDQNAgent, TwoStageDQNAgent

    # Test single-stage Agent
    agent_single = SingleStageDQNAgent(state_size=8, action_size=4)
    state = np.random.randn(8).astype(np.float32)

    action = agent_single.act(state, training=True)
    print(f"✓ SingleStageDQNAgent: action={action}")

    # Test two-stage Agent
    agent_two = TwoStageDQNAgent(state_size=8)
    final_action, stage1_action, stage2_action = agent_two.act(state, training=True)
    print(f"✓ TwoStageDQNAgent: final={final_action}, stage1={stage1_action}, stage2={stage2_action}")

    # Test experience replay
    next_state = np.random.randn(8).astype(np.float32)
    agent_single.remember(state, action, 0.5, next_state, False)
    print(f"✓ Experience stored successfully")

    return True


def test_demandclean_basic():
    """Test basic DemandClean functionality."""
    print("\n" + "=" * 60)
    print("Test: DemandClean basic")
    print("=" * 60)

    from demandclean import DemandClean

    # Create a classification instance
    dc_class = DemandClean(
        task_type='classification',
        model_type='svm',
        agent_type='single_stage',
        n_episodes=5
    )
    print(f"✓ DemandClean (classification) created")

    # Create a regression instance
    dc_reg = DemandClean(
        task_type='regression',
        model_type='ridge',
        agent_type='two_stage',
        n_episodes=5,
        max_truth_budget=10
    )
    print(f"✓ DemandClean (regression) created")

    # Check config
    config = dc_class.get_config()
    print(f"✓ Config retrieved: task={config.task_type.value}")

    return True


def test_demandclean_training_mini():
    """Test DemandClean mini training."""
    print("\n" + "=" * 60)
    print("Test: DemandClean mini training (5 episodes)")
    print("=" * 60)

    from demandclean import DemandClean

    # Generate small test data
    np.random.seed(42)
    X = np.random.randn(50, 4)
    y = np.random.randint(0, 2, 50)

    # Create and train
    dc = DemandClean(
        task_type='classification',
        model_type='random_forest',
        agent_type='single_stage',
        n_episodes=5
    )

    print("Training started...")
    dc.fit(X, y, verbose=False)
    print(f"✓ Training completed")

    # Check fitted flag
    print(f"✓ Fitted: {dc.is_fitted}")

    return True


def test_full_pipeline_with_beers():
    """End-to-end pipeline test using beers data."""
    print("\n" + "=" * 60)
    print("Test: Full pipeline (Beers data)")
    print("=" * 60)

    try:
        X_clean, X_dirty, y = load_beers_data()
        if X_clean is None:
            print("! Could not load beers data; skipping")
            return True

        print(f"✓ Data loaded: clean={X_clean.shape}, dirty={X_dirty.shape}")

        # Use only the first 100 rows for a quick test
        n_samples = min(100, len(X_clean))
        X_clean = X_clean[:n_samples]
        X_dirty = X_dirty[:n_samples]
        y = y[:n_samples]

        from demandclean import DemandClean

        # Create and train
        dc = DemandClean(
            task_type='classification',
            model_type='svm',
            agent_type='single_stage',
            n_episodes=10
        )

        print("Training started...")
        dc.fit(X_dirty, y, verbose=False)
        print(f"✓ Training completed")

        # Detect errors
        detected = dc.detect_errors(X_dirty, X_clean, verbose=False)
        print(f"✓ Errors detected: missing={len(detected.get('missing', []))}, "
              f"semantic={len(detected.get('semantic', []))}, "
              f"syntactic={len(detected.get('syntactic', []))}")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("DemandClean Functional Tests")
    print("=" * 60 + "\n")

    tests = [
        ("ErrorInjector", test_error_injector),
        ("ModelAdapters", test_model_adapters_with_data),
        ("CleaningEnv", test_cleaning_env),
        ("DQN Agent", test_dqn_agent),
        ("DemandClean basic", test_demandclean_basic),
        ("DemandClean training", test_demandclean_training_mini),
        ("Full pipeline (Beers)", test_full_pipeline_with_beers),
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
            print(f"✗ {name} test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Test result: {passed} passed, {failed} failed")
    print("=" * 60)
