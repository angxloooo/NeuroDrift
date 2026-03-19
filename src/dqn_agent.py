"""DQN Agent module - Deep Q-Network with replay buffer and epsilon-greedy exploration."""

import random
from collections import deque
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    """Stores (state, action, reward, next_state, done) transitions for off-policy learning."""

    def __init__(self, capacity: int = 10000):
        self.buffer: deque[tuple[Any, int, float, Any, bool]] = deque(maxlen=capacity)

    def add(self, transition: tuple[Any, int, float, Any, bool]) -> None:
        """Add a transition (state, action, reward, next_state, done)."""
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random batch. Returns (states, actions, rewards, next_states, dones)."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = torch.tensor([t[0] for t in batch], dtype=torch.float32)
        actions = torch.tensor([t[1] for t in batch], dtype=torch.long)
        rewards = torch.tensor([t[2] for t in batch], dtype=torch.float32)
        next_states = torch.tensor([t[3] for t in batch], dtype=torch.float32)
        dones = torch.tensor([t[4] for t in batch], dtype=torch.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


class DQN(nn.Module):
    """Simple MLP for Q-value approximation."""

    def __init__(self, state_dim: int = 6, action_dim: int = 3, hidden_dims: tuple[int, ...] = (64, 64)):
        super().__init__()
        layers = []
        prev = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(prev, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))


class DQNAgent:
    """DQN agent with epsilon-greedy exploration and target network."""

    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 3,
        buffer_size: int = 10000,
        batch_size: int = 64,
        gamma: float = 0.99,
        lr: float = 1e-3,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        target_update_freq: int = 100,
    ):
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_freq = target_update_freq
        self._train_steps = 0

        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> int:
        """Epsilon-greedy action selection. Returns action index 0, 1, or 2."""
        if not deterministic and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            q = self.policy_net(state)
            return int(q.argmax(dim=-1).item())

    def train_step(self) -> float | None:
        """Perform one DQN training step. Returns loss if trained, else None."""
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]
            targets = rewards + self.gamma * next_q * (1 - dones)

        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._train_steps += 1
        if self._train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()
