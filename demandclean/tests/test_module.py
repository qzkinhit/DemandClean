"""
DemandClean Module Test Script
==============================

Verify that modules import correctly and that basic functionality works.
"""

import sys
import os

# Add project path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def test_imports():
    """Test that all modules import correctly."""
    print("=" * 60)
    print("Module import tests")
    print("=" * 60)

    tests_passed = 0
    tests_failed = 0

    # 1. Main package
    try:
        from demandclean import DemandClean, DemandCleanConfig
        print("✓ Main package imported: DemandClean, DemandCleanConfig")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Main package import failed: {e}")
        tests_failed += 1

    # 2. Config module
    try:
        from demandclean.config import DemandCleanConfig, TaskType, ModelType, AgentType
        print("✓ Config module imported")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Config module import failed: {e}")
        tests_failed += 1

    # 3. Model adapters
    try:
        from demandclean.models import (
            ModelAdapter, create_model_adapter,
            SVMAdapter, RandomForestAdapter,
            LinearAdapter, RidgeAdapter
        )
        print("✓ Model adapter module imported")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Model adapter module import failed: {e}")
        tests_failed += 1

    # 4. State extractors
    try:
        from demandclean.core.state import (
            StateExtractor,
            ClassificationStateExtractor,
            RegressionStateExtractor
        )
        print("✓ State extractor module imported")
        tests_passed += 1
    except Exception as e:
        print(f"✗ State extractor module import failed: {e}")
        tests_failed += 1

    # 5. DQN Agent
    try:
        from demandclean.core.agents import (
            BaseAgent,
            SingleStageDQNAgent,
            TwoStageDQNAgent
        )
        print("✓ DQN Agent module imported")
        tests_passed += 1
    except Exception as e:
        print(f"✗ DQN Agent module import failed: {e}")
        tests_failed += 1

    # 6. Environment module
    try:
        from demandclean.core.environments import (
            CleaningEnv,
            TwoPhaseCleaningEnv
        )
        print("✓ Environment module imported")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Environment module import failed: {e}")
        tests_failed += 1

    # 7. Detector module
    try:
        from demandclean.detectors import (
            ErrorInjector,
            RahaBasedDetector
        )
        print("✓ Detector module imported")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Detector module import failed: {e}")
        tests_failed += 1

    # 8. Training module
    try:
        from demandclean.training import Trainer
        print("✓ Training module imported")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Training module import failed: {e}")
        tests_failed += 1

    # 9. Inference module
    try:
        from demandclean.inference import (
            SinglePhaseInference,
            TwoPhaseInference
        )
        print("✓ Inference module imported")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Inference module import failed: {e}")
        tests_failed += 1

    # 10. Utils module
    try:
        from demandclean.utils import (
            DemandCleanLogger,
            ModelIO,
            Metrics
        )
        print("✓ Utils module imported")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Utils module import failed: {e}")
        tests_failed += 1

    print("\n" + "=" * 60)
    print(f"Test result: {tests_passed} passed, {tests_failed} failed")
    print("=" * 60)

    return tests_failed == 0


def test_config_creation():
    """Test configuration object creation."""
    print("\n" + "=" * 60)
    print("Config object creation test")
    print("=" * 60)

    try:
        from demandclean.config import DemandCleanConfig, TaskType, ModelType, AgentType

        # Default config
        config = DemandCleanConfig()
        print(f"✓ Default config created")
        print(f"  task type: {config.task_type.value}")
        print(f"  model type: {config.model_type.value}")
        print(f"  agent type: {config.agent_type.value}")

        # Custom config
        config = DemandCleanConfig(
            task_type=TaskType.REGRESSION,
            model_type=ModelType.RIDGE,
            agent_type=AgentType.SINGLE_STAGE,
            n_episodes=100,
            max_truth_budget=50
        )
        print(f"✓ Custom config created")
        print(f"  task type: {config.task_type.value}")
        print(f"  model type: {config.model_type.value}")
        print(f"  agent type: {config.agent_type.value}")
        print(f"  episodes: {config.n_episodes}")
        print(f"  max truth budget: {config.max_truth_budget}")

        return True
    except Exception as e:
        print(f"✗ Config creation failed: {e}")
        return False


def test_model_adapter_creation():
    """Test model adapter creation."""
    print("\n" + "=" * 60)
    print("Model adapter creation test")
    print("=" * 60)

    try:
        from demandclean.models import create_model_adapter
        from demandclean.config import ModelType, TaskType

        # Classification models
        for model_type in [ModelType.SVM, ModelType.RANDOM_FOREST]:
            adapter = create_model_adapter(model_type, TaskType.CLASSIFICATION)
            print(f"✓ {model_type.value} classification adapter created")

        # Regression models
        for model_type in [ModelType.LINEAR, ModelType.RIDGE]:
            adapter = create_model_adapter(model_type, TaskType.REGRESSION)
            print(f"✓ {model_type.value} regression adapter created")

        return True
    except Exception as e:
        print(f"✗ Model adapter creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_demandclean_api():
    """Test the DemandClean API."""
    print("\n" + "=" * 60)
    print("DemandClean API creation test")
    print("=" * 60)

    try:
        from demandclean import DemandClean

        # Classification task
        dc = DemandClean(
            task_type='classification',
            model_type='svm',
            agent_type='two_stage',
            n_episodes=10
        )
        print(f"✓ DemandClean (classification, SVM, two-stage) created")

        # Regression task
        dc = DemandClean(
            task_type='regression',
            model_type='ridge',
            agent_type='single_stage',
            n_episodes=10,
            max_truth_budget=20
        )
        print(f"✓ DemandClean (regression, Ridge, single-stage) created")

        return True
    except Exception as e:
        print(f"✗ DemandClean API creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("DemandClean Module Tests")
    print("=" * 60 + "\n")

    all_passed = True

    all_passed &= test_imports()
    all_passed &= test_config_creation()
    all_passed &= test_model_adapter_creation()
    all_passed &= test_demandclean_api()

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed; see the errors above")
    print("=" * 60)
