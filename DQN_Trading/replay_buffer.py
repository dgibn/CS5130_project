"""
Experience Replay Buffer for DQN training.

Stores (state, action, reward, next_state, done) transitions and
supports uniform random sampling of mini-batches.
"""

import random
from collections import deque, namedtuple

import numpy as np
import torch

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int, device: torch.device | str = "cpu"):
        batch = random.sample(self.buffer, batch_size)
        states      = torch.FloatTensor(np.array([t.state      for t in batch])).to(device)
        actions     = torch.LongTensor( [t.action     for t in batch]).to(device)
        rewards     = torch.FloatTensor([t.reward     for t in batch]).to(device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch])).to(device)
        dones       = torch.FloatTensor([float(t.done) for t in batch]).to(device)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)