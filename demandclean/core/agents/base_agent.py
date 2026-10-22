"""
DQN Agent base class
====================

Defines the interface shared by all DQN agents.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any
import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class for DQN agents.

    Specifies the interface every agent must implement.
    """

    def __init__(self, state_size: int = 8):
        """
        Initialize the agent.

        Args:
            state_size: dimensionality of the state vector
        """
        self.state_size = state_size
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.learning_rate = 0.0005

        # Metadata for resuming training
        self.total_episodes: int = 0
        self.best_score: float = -float('inf')
        self.best_episode: int = 0

    @abstractmethod
    def act(self, state: np.ndarray, training: bool = True) -> Any:
        """
        Select an action given the current state.

        Args:
            state: state vector
            training: whether the agent is in training mode (enables exploration)

        Returns:
            The selected action.
        """
        pass

    @abstractmethod
    def remember(self, state: np.ndarray, action: int,
                 reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Store a transition in the replay buffer.

        Args:
            state: current state
            action: action taken
            reward: reward received
            next_state: next state
            done: whether the episode ended
        """
        pass

    @abstractmethod
    def replay(self, batch_size: int = 64) -> None:
        """
        Train from replayed experience.

        Args:
            batch_size: mini-batch size
        """
        pass

    @abstractmethod
    def update_target_model(self) -> None:
        """Update the target network."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the model.

        Args:
            path: destination path
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the model.

        Args:
            path: path to the saved model
        """
        pass

    def decay_epsilon(self) -> None:
        """Decay the exploration rate."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Return Q-values for the given state without selecting an action.

        Args:
            state: state vector of shape (state_size,)

        Returns:
            Q-value array. Shape depends on the agent type:
            - single-stage: (action_size,)
            - two-stage: (stage1_action_size + stage2_action_size,), concatenated.
        """
        raise NotImplementedError

    def get_weights(self) -> Any:
        """Return a deep copy of the model weights (PyTorch state_dict)."""
        pass

    def set_weights(self, weights: Any) -> None:
        """Load model weights from a PyTorch state_dict."""
        pass
