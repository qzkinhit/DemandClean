"""
Environment Tests
=================
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# import pytest  # Optional
import numpy as np
from demandclean.core.environments import CleaningEnv, TwoPhaseCleaningEnv
from demandclean.core.state import ClassificationStateExtractor
from demandclean.models import create_model_adapter
from demandclean.config import DemandCleanConfig, ModelType, TaskType
from demandclean.detectors import ErrorInjector


def create_test_env():
    """Helper that builds a test environment."""
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

    return X_dirty, y, error_list, model_adapter, state_extractor, config


def test_cleaning_env_init():
    """Test cleaning environment initialization."""
    X_dirty, y, error_list, model_adapter, state_extractor, config = create_test_env()

    env = CleaningEnv(X_dirty, y, error_list, model_adapter, state_extractor, config)
    assert env is not None
    print("✓ Cleaning environment initialization test passed")


def test_cleaning_env_reset():
    """Test cleaning environment reset."""
    X_dirty, y, error_list, model_adapter, state_extractor, config = create_test_env()

    env = CleaningEnv(X_dirty, y, error_list, model_adapter, state_extractor, config)
    state = env.reset()

    assert state.shape == (10,)
    assert not np.isnan(state).any()
    print(f"✓ Cleaning environment reset test passed: state shape={state.shape}")


def test_cleaning_env_step():
    """Test cleaning environment step."""
    X_dirty, y, error_list, model_adapter, state_extractor, config = create_test_env()

    env = CleaningEnv(X_dirty, y, error_list, model_adapter, state_extractor, config)
    state = env.reset()

    # Test each action
    for action in [0, 1, 2, 3]:
        env.reset()
        next_state, reward, done, info = env.step(action)
        assert next_state.shape == (10,)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        print(f"  action {action}: reward={reward:.4f}, done={done}")

    print("✓ Cleaning environment step test passed")


def test_cleaning_env_episode():
    """Test cleaning environment full episode."""
    X_dirty, y, error_list, model_adapter, state_extractor, config = create_test_env()

    env = CleaningEnv(X_dirty, y, error_list, model_adapter, state_extractor, config)
    state = env.reset()

    total_reward = 0
    steps = 0
    max_steps = 100

    while steps < max_steps:
        action = np.random.randint(0, 4)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1
        state = next_state
        if done:
            break

    assert steps > 0
    print(f"✓ Cleaning environment full episode test passed: {steps} steps, reward={total_reward:.4f}")


def test_cleaning_env_action_counts():
    """Test cleaning environment action statistics."""
    X_dirty, y, error_list, model_adapter, state_extractor, config = create_test_env()

    env = CleaningEnv(X_dirty, y, error_list, model_adapter, state_extractor, config)
    env.reset()

    # Execute some actions
    for _ in range(10):
        action = np.random.randint(0, 4)
        _, _, done, _ = env.step(action)
        if done:
            break

    counts = env.get_action_counts()
    assert 'no_action' in counts
    assert 'repair_value' in counts
    assert 'delete' in counts
    assert 'replace_nearby' in counts
    print(f"✓ Cleaning environment action statistics test passed: {counts}")


def test_cleaning_env_get_cleaned_data():
    """Test retrieval of cleaned data."""
    X_dirty, y, error_list, model_adapter, state_extractor, config = create_test_env()

    env = CleaningEnv(X_dirty, y, error_list, model_adapter, state_extractor, config)
    env.reset()

    # Execute some actions
    for _ in range(10):
        action = np.random.randint(0, 4)
        _, _, done, _ = env.step(action)
        if done:
            break

    X_clean, y_clean, keep_mask = env.get_cleaned_data()
    assert len(X_clean) == len(y_clean)
    assert len(keep_mask) == len(y)
    print(f"✓ Get cleaned data test passed: {len(X_clean)} rows")


def test_two_phase_env_init():
    """Test two-phase environment initialization."""
    X_dirty, y, error_list, model_adapter, state_extractor, config = create_test_env()

    env = TwoPhaseCleaningEnv(X_dirty, y, error_list, model_adapter, state_extractor, config)
    assert env is not None
    print("✓ Two-phase environment initialization test passed")


def test_two_phase_env_plan():
    """Test two-phase environment plan generation."""
    X_dirty, y, error_list, model_adapter, state_extractor, config = create_test_env()

    env = TwoPhaseCleaningEnv(X_dirty, y, error_list, model_adapter, state_extractor, config)
    state = env.reset()

    # Run until done to generate plan
    while True:
        action = np.random.randint(0, 4)
        next_state, _, done, _ = env.step(action)
        if done:
            break

    plan = env.get_repair_plan()
    assert isinstance(plan, list)
    print(f"✓ Two-phase environment plan generation test passed: {len(plan)} plan items")


def test_two_phase_env_execute():
    """Test two-phase environment plan execution."""
    X_dirty, y, error_list, model_adapter, state_extractor, config = create_test_env()

    env = TwoPhaseCleaningEnv(X_dirty, y, error_list, model_adapter, state_extractor, config)
    state = env.reset()

    # Run until done
    while True:
        action = np.random.randint(0, 4)
        _, _, done, _ = env.step(action)
        if done:
            break

    plan = env.get_repair_plan()
    positions = env.get_plan_positions()

    # Create ground-truth values
    true_values = {}
    for idx, col in positions:
        true_values[(idx, col)] = np.random.randn()

    # Execute plan
    X_result, y_result = env.execute_repair_plan(X_dirty, true_values)
    assert X_result is not None
    print(f"✓ Two-phase environment plan execution test passed")


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("Environment Tests")
    print("=" * 50 + "\n")

    test_cleaning_env_init()
    test_cleaning_env_reset()
    test_cleaning_env_step()
    test_cleaning_env_episode()
    test_cleaning_env_action_counts()
    test_cleaning_env_get_cleaned_data()
    test_two_phase_env_init()
    test_two_phase_env_plan()
    test_two_phase_env_execute()

    print("\nAll tests passed!")
