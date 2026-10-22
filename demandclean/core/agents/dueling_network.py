"""
Dueling DQN network
===================

Shared feature trunk with separate Value and Advantage heads.
Q(s, a) = V(s) + A(s, a) - mean(A).
"""

import torch
import torch.nn as nn


class DuelingNetwork(nn.Module):
    """
    Dueling DQN network.

    Architecture:
        shared trunk:   state_size -> hidden -> hidden (ReLU)
        value head:     hidden -> 1
        advantage head: hidden -> action_size
        output:         Q = V + A - mean(A)
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()

        # Shared feature trunk
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Value head: estimates the state value V(s)
        self.value_head = nn.Linear(hidden_size, 1)

        # Advantage head: estimates the action advantage A(s, a)
        self.advantage_head = nn.Linear(hidden_size, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.shared(x)
        value = self.value_head(features)                   # (batch, 1)
        advantage = self.advantage_head(features)           # (batch, action_size)
        # Q(s, a) = V(s) + A(s, a) - mean(A)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q
