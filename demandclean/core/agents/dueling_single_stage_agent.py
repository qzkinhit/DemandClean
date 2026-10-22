"""
Dueling Double DQN single-stage agent (PyTorch)
================================================

Single-stage agent combining a Dueling network, Double DQN, and soft target updates.
"""

from typing import Any, Dict
from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base_agent import BaseAgent
from .dueling_network import DuelingNetwork


class DuelingSingleStageAgent(BaseAgent):
    """
    Dueling Double DQN single-stage agent.

    Outputs one of four actions directly:
        0: no_action
        1: repair_value
        2: delete
        3: replace_nearby

    Differences from plain DQN:
        - Uses a Dueling network that separates V(s) and A(s, a)
        - Soft target updates (tau)
        - Larger hidden layers (128)
    """

    def __init__(self,
                 state_size: int = 8,
                 action_size: int = 4,
                 memory_size: int = 5000,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.995,
                 learning_rate: float = 0.0005,
                 tau: float = 0.1,
                 hidden_size: int = 128):
        super().__init__(state_size)
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.tau = tau

        self.device = torch.device('cpu')

        # Build networks
        self.model = DuelingNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_model = DuelingNetwork(state_size, action_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Initialize target network weights
        self.target_model.load_state_dict(self.model.state_dict())

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor(arr).to(self.device)

    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            q = self.model(self._to_tensor(state).unsqueeze(0))
            return int(q.argmax(dim=1).item())

    def remember(self, state: np.ndarray, action: int,
                 reward: float, next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size: int = 64) -> None:
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states = self._to_tensor(np.array([x[0] for x in batch]))
        actions = torch.LongTensor([x[1] for x in batch]).to(self.device)
        rewards = self._to_tensor(np.array([x[2] for x in batch]))
        next_states = self._to_tensor(np.array([x[3] for x in batch]))
        dones = self._to_tensor(np.array([x[4] for x in batch], dtype=np.float32))

        # Double DQN: online net picks the action, target net evaluates it
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.model(next_states).argmax(dim=1)
            next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.decay_epsilon()

    def update_target_model(self) -> None:
        """Soft-update the target network: target = tau * model + (1 - tau) * target."""
        for t_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            t_param.data.copy_(self.tau * param.data + (1 - self.tau) * t_param.data)

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Return Q-values for the state, shape (action_size,)."""
        with torch.no_grad():
            q = self.model(self._to_tensor(state).unsqueeze(0))
            return q.cpu().numpy().flatten()

    def save(self, path: str) -> None:
        """Save the model (.pt format) together with resume-training metadata."""
        path = path.replace('.h5', '.pt')
        torch.save({
            'model_state': self.model.state_dict(),
            'target_state': self.target_model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_episodes': self.total_episodes,
            'best_score': self.best_score,
            'best_episode': self.best_episode,
        }, path)

    def load(self, path: str) -> None:
        """Load the model (backward compatible with older checkpoints)."""
        path = path.replace('.h5', '.pt')
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state'])
        self.target_model.load_state_dict(ckpt.get('target_state', ckpt['model_state']))
        if 'optimizer_state' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.epsilon = ckpt.get('epsilon', self.epsilon)
        self.total_episodes = ckpt.get('total_episodes', 0)
        self.best_score = ckpt.get('best_score', -float('inf'))
        self.best_episode = ckpt.get('best_episode', 0)

    def get_weights(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.model.state_dict().items()}

    def set_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(weights)
        self.target_model.load_state_dict(self.model.state_dict())
