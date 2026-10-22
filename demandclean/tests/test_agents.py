"""
DQN Agent Tests
===============
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# import pytest  # Optional
import numpy as np
from demandclean.core.agents import SingleStageDQNAgent, TwoStageDQNAgent


def test_single_stage_agent_init():
    """Test single-stage Agent initialization."""
    agent = SingleStageDQNAgent(state_size=8, action_size=4)

    assert agent.state_size == 8
    assert agent.action_size == 4
    assert agent.epsilon == 1.0
    print("✓ Single-stage Agent initialization test passed")


def test_single_stage_agent_act_training():
    """Test single-stage Agent action selection during training."""
    agent = SingleStageDQNAgent(state_size=8, action_size=4)
    state = np.random.randn(8).astype(np.float32)

    action = agent.act(state, training=True)
    assert action in [0, 1, 2, 3]
    print(f"✓ Single-stage Agent training action test passed: action={action}")


def test_single_stage_agent_act_inference():
    """Test single-stage Agent action selection during inference."""
    agent = SingleStageDQNAgent(state_size=8, action_size=4)
    agent.epsilon = 0  # Disable exploration
    state = np.random.randn(8).astype(np.float32)

    action = agent.act(state, training=False)
    assert action in [0, 1, 2, 3]
    print(f"✓ Single-stage Agent inference action test passed: action={action}")


def test_single_stage_agent_remember():
    """Test single-stage Agent experience replay storage."""
    agent = SingleStageDQNAgent(state_size=8, action_size=4, memory_size=100)
    state = np.random.randn(8).astype(np.float32)
    next_state = np.random.randn(8).astype(np.float32)

    agent.remember(state, 1, 0.5, next_state, False)
    assert len(agent.memory) == 1
    print("✓ Single-stage Agent experience storage test passed")


def test_single_stage_agent_replay():
    """Test single-stage Agent experience replay."""
    agent = SingleStageDQNAgent(state_size=8, action_size=4, memory_size=100)

    # Add enough experience
    for _ in range(64):
        state = np.random.randn(8).astype(np.float32)
        next_state = np.random.randn(8).astype(np.float32)
        agent.remember(state, np.random.randint(4), np.random.randn(), next_state, False)

    # Execute replay
    agent.replay(batch_size=32)
    print("✓ Single-stage Agent experience replay test passed")


def test_single_stage_agent_update_target():
    """Test single-stage Agent target network update."""
    agent = SingleStageDQNAgent(state_size=8, action_size=4)
    agent.update_target_model()
    print("✓ Single-stage Agent target network update test passed")


def test_single_stage_agent_epsilon_decay():
    """Test single-stage Agent exploration rate decay."""
    agent = SingleStageDQNAgent(state_size=8, action_size=4, epsilon_decay=0.99)
    initial_eps = agent.epsilon

    # Add experience and replay
    for _ in range(64):
        state = np.random.randn(8).astype(np.float32)
        next_state = np.random.randn(8).astype(np.float32)
        agent.remember(state, np.random.randint(4), np.random.randn(), next_state, False)

    agent.replay(batch_size=32)
    assert agent.epsilon < initial_eps
    print(f"✓ Single-stage Agent epsilon decay test passed: {initial_eps:.3f} -> {agent.epsilon:.3f}")


def test_two_stage_agent_init():
    """Test two-stage Agent initialization."""
    agent = TwoStageDQNAgent(state_size=8)

    assert agent.state_size == 8
    assert agent.stage1_action_size == 3  # no_action, process, delete
    assert agent.stage2_action_size == 2  # repair_value, replace_nearby
    print("✓ Two-stage Agent initialization test passed")


def test_two_stage_agent_act_training():
    """Test two-stage Agent action selection during training."""
    agent = TwoStageDQNAgent(state_size=8)
    state = np.random.randn(8).astype(np.float32)

    final_action, stage1_action, stage2_action = agent.act(state, training=True)
    assert final_action in [0, 1, 2, 3]
    assert stage1_action in [0, 1, 2]
    # stage2_action may be None (when stage1 is not "process")
    print(f"✓ Two-stage Agent training action test passed: final={final_action}, s1={stage1_action}, s2={stage2_action}")


def test_two_stage_agent_act_inference():
    """Test two-stage Agent action selection during inference."""
    agent = TwoStageDQNAgent(state_size=8)
    agent.epsilon = 0
    state = np.random.randn(8).astype(np.float32)

    final_action, stage1_action, stage2_action = agent.act(state, training=False)
    assert final_action in [0, 1, 2, 3]
    print(f"✓ Two-stage Agent inference action test passed: final={final_action}")


def test_two_stage_agent_remember():
    """Test two-stage Agent experience replay storage."""
    agent = TwoStageDQNAgent(state_size=8, memory_size=100)
    state = np.random.randn(8).astype(np.float32)
    next_state = np.random.randn(8).astype(np.float32)

    agent.remember_stage1(state, 1, 0.5, next_state, False)
    agent.remember_stage2(state, 0, 0.3, next_state, False)

    assert len(agent.stage1_memory) == 1
    assert len(agent.stage2_memory) == 1
    print("✓ Two-stage Agent experience storage test passed")


def test_two_stage_agent_replay():
    """Test two-stage Agent experience replay."""
    agent = TwoStageDQNAgent(state_size=8, memory_size=100)

    # Add enough experience
    for _ in range(64):
        state = np.random.randn(8).astype(np.float32)
        next_state = np.random.randn(8).astype(np.float32)
        agent.remember_stage1(state, np.random.randint(3), np.random.randn(), next_state, False)
        agent.remember_stage2(state, np.random.randint(2), np.random.randn(), next_state, False)

    # Execute replay
    agent.replay(batch_size=32)
    print("✓ Two-stage Agent experience replay test passed")


def test_agent_get_set_weights():
    """Test getting and setting weights."""
    agent = SingleStageDQNAgent(state_size=8, action_size=4)

    weights = agent.get_weights()
    assert weights is not None

    agent.set_weights(weights)
    print("✓ Agent get/set weights test passed")


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("DQN Agent Tests")
    print("=" * 50 + "\n")

    test_single_stage_agent_init()
    test_single_stage_agent_act_training()
    test_single_stage_agent_act_inference()
    test_single_stage_agent_remember()
    test_single_stage_agent_replay()
    test_single_stage_agent_update_target()
    test_single_stage_agent_epsilon_decay()
    test_two_stage_agent_init()
    test_two_stage_agent_act_training()
    test_two_stage_agent_act_inference()
    test_two_stage_agent_remember()
    test_two_stage_agent_replay()
    test_agent_get_set_weights()

    print("\nAll tests passed!")
