import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from agents.base_agent import BaseAgent


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent(BaseAgent):
    def __init__(
        self,
        state_dim: int = 2,
        n_actions: int = 8,
        hidden_size: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        buffer_capacity: int = 50_000,
        batch_size: int = 64,
        warmup: int = 1_000,
        target_update_freq: int = 100,
        device: str = None,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.warmup = warmup
        self.target_update_freq = target_update_freq
        self.training_mode = True
        self._steps = 0

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.online_net = QNetwork(state_dim, n_actions, hidden_size).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def take_action(self, state) -> int:
        if self.training_mode and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def store_transition(self, state, action: int, reward: float, next_state, done: bool):
        self.replay_buffer.push(state, action, reward, next_state, float(done))

    def train_step(self) -> float | None:
        if len(self.replay_buffer) < self.warmup:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.tensor(states, device=self.device)
        actions_t = torch.tensor(actions, device=self.device)
        rewards_t = torch.tensor(rewards, device=self.device)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t = torch.tensor(dones, device=self.device)

        # Q(s, a) for the actions that were actually taken
        q_pred = self.online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # TD target: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            q_next = self.target_net(next_states_t).max(dim=1).values
            q_target = rewards_t + self.gamma * q_next * (1.0 - dones_t)

        loss = nn.functional.smooth_l1_loss(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        self._steps += 1
        if self._steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps": self._steps,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self._steps = ckpt["steps"]
