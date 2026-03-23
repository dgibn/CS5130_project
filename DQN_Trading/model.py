"""
Dueling Deep Q-Network.

Architecture
------------
shared feature layers  →  value  head  V(s)        (scalar)
                       →  advantage head A(s, a)    (one per action)

Q(s, a) = V(s) + A(s, a) - mean_a[A(s, a)]
"""

import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden: int = 128,
        dueling: bool = True,
    ):
        super().__init__()
        self.dueling = dueling

        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        if dueling:
            self.value_head = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, 1),
            )
            self.advantage_head = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, n_actions),
            )
        else:
            self.q_head = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, n_actions),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)

        if self.dueling:
            value     = self.value_head(feat)         # (B, 1)
            advantage = self.advantage_head(feat)      # (B, n_actions)
            q = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            q = self.q_head(feat)

        return q