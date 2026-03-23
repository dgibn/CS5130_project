"""
DQN Agent — ties together the network, target network,
replay buffer, and epsilon-greedy policy.

Features
--------
- Double DQN (policy net selects, target net evaluates)
- Dueling architecture (via model.py)
- Soft target-network updates (Polyak averaging)
- Huber (Smooth-L1) loss
- Gradient clipping
"""

import random
import os

import torch
import torch.nn as nn
import torch.optim as optim

from model import DQN
from replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay: float = 0.995,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        dueling: bool = True,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.policy_net = DQN(state_dim, n_actions, hidden, dueling).to(self.device)
        self.target_net = DQN(state_dim, n_actions, hidden, dueling).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

        self.memory = ReplayBuffer(buffer_size)

    def select_action(self, state, eval_mode: bool = False) -> int:
        """ε-greedy in training; greedy in eval."""
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.policy_net(s)
            return q.argmax(dim=1).item()


    def learn(self) -> float | None:
        """Sample a batch, compute Double-DQN targets, do one SGD step."""
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size, self.device
        )

        q_values = (
            self.policy_net(states)
            .gather(1, actions.unsqueeze(1))
            .squeeze(1)
        )

        with torch.no_grad():
            best_actions = self.policy_net(next_states).argmax(dim=1)
            next_q = (
                self.target_net(next_states)
                .gather(1, best_actions.unsqueeze(1))
                .squeeze(1)
            )
            target = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()


    def update_target(self):
        for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tp.data.copy_(self.tau * pp.data + (1 - self.tau) * tp.data)

    def decay_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)


    def save(self, path: str = "checkpoints/dqn_agent.pt"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )
        print(f"Agent saved → {path}")

    def load(self, path: str = "checkpoints/dqn_agent.pt"):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        print(f"Agent loaded ← {path}")