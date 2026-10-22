"""
Dueling Double DQN two-stage agent (PyTorch)
=============================================

Two-stage agent combining a Dueling network, Double DQN, and soft target updates.
"""

from typing import Tuple, Optional, Dict
from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base_agent import BaseAgent
from .dueling_network import DuelingNetwork


class DuelingTwoStageAgent(BaseAgent):
    """
    Dueling Double DQN two-stage agent.

    Stage 1: triage decision (3 actions, all free)
        0: no_action  - skip the error
        1: delete     - drop the row
        2: VE-fill    - fill with a value-estimator prediction, then move to Stage 2

    Stage 2: quality decision (2 actions, only when stage1 == VE-fill)
        0: keep_VE       - keep the VE estimate (free)
        1: truth_repair  - spend budget to replace with the ground truth

    Final action mapping:
        0: no_action                       (stage1 = 0)
        1: repair_value   (stage1 = 2, stage2 = 1)
        2: delete                          (stage1 = 1)
        3: replace_nearby (stage1 = 2, stage2 = 0)
    """

    def __init__(self,
                 state_size: int = 8,
                 memory_size: int = 5000,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.995,
                 learning_rate: float = 0.0005,
                 tau: float = 0.1,
                 hidden_size: int = 128):
        super().__init__(state_size)
        # Set by DemandClean.load() for 8-dim legacy checkpoints (see act()).
        self.legacy_action_map = False
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.tau = tau

        self.device = torch.device('cpu')

        # Stage 1: policy selection (3 actions)
        self.stage1_action_size = 3
        self.stage1_memory = deque(maxlen=memory_size)
        self.stage1_model = DuelingNetwork(state_size, self.stage1_action_size, hidden_size).to(self.device)
        self.stage1_target = DuelingNetwork(state_size, self.stage1_action_size, hidden_size).to(self.device)
        self.stage1_optimizer = optim.Adam(self.stage1_model.parameters(), lr=learning_rate)

        # Stage 2: repair strategy (2 actions)
        self.stage2_action_size = 2
        self.stage2_memory = deque(maxlen=memory_size)
        self.stage2_model = DuelingNetwork(state_size, self.stage2_action_size, hidden_size).to(self.device)
        self.stage2_target = DuelingNetwork(state_size, self.stage2_action_size, hidden_size).to(self.device)
        self.stage2_optimizer = optim.Adam(self.stage2_model.parameters(), lr=learning_rate)

        # Sync target networks
        self.stage1_target.load_state_dict(self.stage1_model.state_dict())
        self.stage2_target.load_state_dict(self.stage2_model.state_dict())

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor(arr).to(self.device)

    def act_stage1(self, state: np.ndarray, training: bool = True) -> int:
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.stage1_action_size)
        with torch.no_grad():
            q = self.stage1_model(self._to_tensor(state).unsqueeze(0))
            return int(q.argmax(dim=1).item())

    def act_stage2(self, state: np.ndarray, training: bool = True) -> int:
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.stage2_action_size)
        with torch.no_grad():
            q = self.stage2_model(self._to_tensor(state).unsqueeze(0))
            return int(q.argmax(dim=1).item())

    def act(self, state: np.ndarray, training: bool = True) -> Tuple[int, int, Optional[int]]:
        """
        Two-stage action selection.

        Mapping:
          s1 = 0 -> no_action(0)
          s1 = 1 -> delete(2)
          s1 = 2 -> VE-fill -> Stage 2 decision:
            s2 = 0 -> keep_VE -> replace_nearby(3)
            s2 = 1 -> truth_repair -> repair_value(1)

        Returns:
            (final_action, stage1_action, stage2_action)
        """
        s1 = self.act_stage1(state, training)

        if getattr(self, 'legacy_action_map', False):
            # Checkpoints with the 8-dim state (the har policy of the reported run)
            # were trained with the earlier head layout:
            #   s1 = 0 -> no_action(0), s1 = 2 -> delete(2),
            #   s1 = 1 -> Stage 2: s2 = 0 -> repair_value(1), s2 = 1 -> replace_nearby(3)
            if s1 == 0:
                return 0, s1, None
            elif s1 == 2:
                return 2, s1, None
            s2 = self.act_stage2(state, training)
            return (1 if s2 == 0 else 3), s1, s2

        if s1 == 0:
            return 0, s1, None        # no_action
        elif s1 == 1:
            return 2, s1, None        # delete
        else:  # s1 == 2: VE-fill -> Stage 2 decision
            s2 = self.act_stage2(state, training)
            return (3 if s2 == 0 else 1), s1, s2
            # s2 = 0: keep_VE -> replace_nearby(3)
            # s2 = 1: truth_repair -> repair_value(1)

    def remember_stage1(self, state: np.ndarray, action: int,
                        reward: float, next_state: np.ndarray, done: bool) -> None:
        self.stage1_memory.append((state, action, reward, next_state, done))

    def remember_stage2(self, state: np.ndarray, action: int,
                        reward: float, next_state: np.ndarray, done: bool) -> None:
        self.stage2_memory.append((state, action, reward, next_state, done))

    def remember(self, state: np.ndarray, action: int,
                 reward: float, next_state: np.ndarray, done: bool) -> None:
        self.remember_stage1(state, action, reward, next_state, done)

    def _replay_stage(self, memory: deque, model: nn.Module,
                      target: nn.Module, optimizer: optim.Optimizer,
                      batch_size: int) -> None:
        if len(memory) < batch_size:
            return

        batch = random.sample(memory, batch_size)
        states = self._to_tensor(np.array([x[0] for x in batch]))
        actions = torch.LongTensor([x[1] for x in batch]).to(self.device)
        rewards = self._to_tensor(np.array([x[2] for x in batch]))
        next_states = self._to_tensor(np.array([x[3] for x in batch]))
        dones = self._to_tensor(np.array([x[4] for x in batch], dtype=np.float32))

        # Double DQN
        current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = model(next_states).argmax(dim=1)
            next_q = target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q, target_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def replay(self, batch_size: int = 64) -> None:
        self._replay_stage(self.stage1_memory, self.stage1_model,
                           self.stage1_target, self.stage1_optimizer, batch_size)
        self._replay_stage(self.stage2_memory, self.stage2_model,
                           self.stage2_target, self.stage2_optimizer, batch_size)
        self.decay_epsilon()

    def _soft_update(self, model: nn.Module, target: nn.Module) -> None:
        """Soft update: target = tau * model + (1 - tau) * target."""
        for t_param, param in zip(target.parameters(), model.parameters()):
            t_param.data.copy_(self.tau * param.data + (1 - self.tau) * t_param.data)

    def update_target_model(self) -> None:
        self._soft_update(self.stage1_model, self.stage1_target)
        self._soft_update(self.stage2_model, self.stage2_target)

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Return concatenated Q-values for both stages, shape (stage1_action_size + stage2_action_size,)."""
        with torch.no_grad():
            st = self._to_tensor(state).unsqueeze(0)
            q1 = self.stage1_model(st).cpu().numpy().flatten()
            q2 = self.stage2_model(st).cpu().numpy().flatten()
            return np.concatenate([q1, q2])

    def save(self, path: str) -> None:
        """Save both stage models (.pt format) together with resume-training metadata."""
        base = path.replace('.h5', '').replace('.pt', '')
        stage1_path = base + '_stage1.pt'
        stage2_path = base + '_stage2.pt'

        torch.save({
            'model_state': self.stage1_model.state_dict(),
            'target_state': self.stage1_target.state_dict(),
            'optimizer_state': self.stage1_optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_episodes': self.total_episodes,
            'best_score': self.best_score,
            'best_episode': self.best_episode,
        }, stage1_path)

        torch.save({
            'model_state': self.stage2_model.state_dict(),
            'target_state': self.stage2_target.state_dict(),
            'optimizer_state': self.stage2_optimizer.state_dict(),
        }, stage2_path)

    def load(self, path: str) -> None:
        """Load both stage models (backward compatible with older checkpoints)."""
        base = path.replace('.h5', '').replace('.pt', '')
        stage1_path = base + '_stage1.pt'
        stage2_path = base + '_stage2.pt'

        ckpt1 = torch.load(stage1_path, map_location=self.device, weights_only=False)
        self.stage1_model.load_state_dict(ckpt1['model_state'])
        self.stage1_target.load_state_dict(ckpt1.get('target_state', ckpt1['model_state']))
        if 'optimizer_state' in ckpt1:
            self.stage1_optimizer.load_state_dict(ckpt1['optimizer_state'])
        self.epsilon = ckpt1.get('epsilon', self.epsilon)
        self.total_episodes = ckpt1.get('total_episodes', 0)
        self.best_score = ckpt1.get('best_score', -float('inf'))
        self.best_episode = ckpt1.get('best_episode', 0)

        ckpt2 = torch.load(stage2_path, map_location=self.device, weights_only=False)
        self.stage2_model.load_state_dict(ckpt2['model_state'])
        self.stage2_target.load_state_dict(ckpt2.get('target_state', ckpt2['model_state']))
        if 'optimizer_state' in ckpt2:
            self.stage2_optimizer.load_state_dict(ckpt2['optimizer_state'])

    def get_weights(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        s1 = {k: v.clone() for k, v in self.stage1_model.state_dict().items()}
        s2 = {k: v.clone() for k, v in self.stage2_model.state_dict().items()}
        return (s1, s2)

    def set_weights(self, weights: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]) -> None:
        s1_weights, s2_weights = weights
        self.stage1_model.load_state_dict(s1_weights)
        self.stage2_model.load_state_dict(s2_weights)
        self.stage1_target.load_state_dict(self.stage1_model.state_dict())
        self.stage2_target.load_state_dict(self.stage2_model.state_dict())
