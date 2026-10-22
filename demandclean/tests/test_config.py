"""
Config Module Tests
===================
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from demandclean.config import DemandCleanConfig, TaskType, ModelType, AgentType


def test_default_config():
    """Test default configuration."""
    config = DemandCleanConfig()

    assert config.task_type == TaskType.CLASSIFICATION
    assert config.model_type == ModelType.SVM
    assert config.agent_type == AgentType.SINGLE_STAGE
    assert config.n_episodes == 300
    assert config.state_size == 10
    print("✓ Default config test passed")


def test_custom_config():
    """Test custom configuration."""
    config = DemandCleanConfig(
        task_type=TaskType.REGRESSION,
        model_type=ModelType.RIDGE,
        agent_type=AgentType.TWO_STAGE,
        n_episodes=100,
        repair_lambda=0.05
    )

    assert config.task_type == TaskType.REGRESSION
    assert config.model_type == ModelType.RIDGE
    assert config.agent_type == AgentType.TWO_STAGE
    assert config.n_episodes == 100
    assert config.repair_lambda == 0.05
    print("✓ Custom config test passed")


def test_string_to_enum():
    """Test string-to-enum conversion."""
    config = DemandCleanConfig(
        task_type='classification',
        model_type='random_forest',
        agent_type='two_stage'
    )

    assert config.task_type == TaskType.CLASSIFICATION
    assert config.model_type == ModelType.RANDOM_FOREST
    assert config.agent_type == AgentType.TWO_STAGE
    print("✓ String-to-enum test passed")


def test_config_properties():
    """Test configuration properties."""
    config_class = DemandCleanConfig(task_type=TaskType.CLASSIFICATION)
    config_reg = DemandCleanConfig(task_type=TaskType.REGRESSION)

    assert config_class.is_classification == True
    assert config_class.is_regression == False
    assert config_reg.is_classification == False
    assert config_reg.is_regression == True

    # Test epsilon property
    assert config_class.epsilon == config_class.epsilon_start
    print("✓ Config properties test passed")


def test_config_to_dict():
    """Test configuration serialization."""
    config = DemandCleanConfig(n_episodes=50, repair_lambda=0.1)
    d = config.to_dict()

    assert d['n_episodes'] == 50
    assert d['repair_lambda'] == 0.1
    assert d['task_type'] == 'classification'
    print("✓ Config serialization test passed")


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("Config Module Tests")
    print("=" * 50 + "\n")

    test_default_config()
    test_custom_config()
    test_string_to_enum()
    test_config_properties()
    test_config_to_dict()

    print("\nAll tests passed!")
