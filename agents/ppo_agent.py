"""PPO Agent — Proximal Policy Optimization with Actor-Critic.

Uses a clipped surrogate objective and Generalized Advantage Estimation (GAE)
to stably update a neural policy on continuous observation vectors.

State input: a float32 numpy array produced by ContinuousEnv (GPS, raycasting,
or both). The dimension is passed in as state_dim, matching DQN's interface.

Update strategy: fixed-step rollout — the training script collects
`rollout_steps` transitions (spanning multiple episodes) then calls learn().
If the rollout ends mid-episode, the last state value is bootstrapped from
the critic rather than treated as a terminal.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from agents import BaseAgent


class _ActorCritic(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_size: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor):
        return self.actor(x), self.critic(x).squeeze(-1)

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor):
        logits, values = self.forward(states)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), values, dist.entropy()


class PPOAgent(BaseAgent):
    def __init__(
        self,
        state_dim: int = 2,
        n_actions: int = 8,
        hidden_size: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        k_epochs: int = 4,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.05,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        minibatch_size: int = 64,
        rng_seed: int | None = None,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.k_epochs = k_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.minibatch_size = minibatch_size

        if rng_seed is not None:
            torch.manual_seed(rng_seed)
            np.random.seed(rng_seed)

        self.policy = _ActorCritic(state_dim=state_dim, n_actions=n_actions,
                                   hidden_size=hidden_size)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self._reset_buffer()

    # ------------------------------------------------------------------
    # State encoding
    # ------------------------------------------------------------------

    def _encode(self, state: np.ndarray) -> torch.Tensor:
        return torch.tensor(state, dtype=torch.float32)

    def state_value(self, state: np.ndarray) -> float:
        with torch.no_grad():
            _, v = self.policy(self._encode(state))
        return v.item()

    # ------------------------------------------------------------------
    # Rollout buffer helpers
    # ------------------------------------------------------------------

    def _reset_buffer(self):
        self._buf_states: list = []
        self._buf_actions: list[int] = []
        self._buf_log_probs: list[float] = []
        self._buf_values: list[float] = []
        self._buf_rewards: list[float] = []
        self._buf_dones: list[float] = []

    def store_reward(self, reward: float, done: bool):
        self._buf_rewards.append(float(reward))
        self._buf_dones.append(float(done))

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Sample from the policy (training) or act greedily (evaluation)."""
        x = self._encode(state)
        with torch.no_grad():
            logits, value = self.policy(x)
            if training:
                dist = Categorical(logits=logits)
                action_t = dist.sample()
                self._buf_states.append(state)
                self._buf_actions.append(action_t.item())
                self._buf_log_probs.append(dist.log_prob(action_t).item())
                self._buf_values.append(value.item())
                return action_t.item()
            else:
                return int(torch.argmax(logits).item())

    def take_action(self, state: np.ndarray) -> int:
        """Greedy action for evaluation."""
        return self.select_action(state, training=False)

    def update(self, state, reward, action):
        raise RuntimeError(
            "PPOAgent.update() is not used. Drive the agent through "
            "select_action / store_reward / learn in train_ppo.py."
        )

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def learn(self, last_value: float = 0.0) -> dict:
        """Run PPO update on the buffered rollout and flush the buffer.

        Args:
            last_value: Critic estimate for the state after the last stored
                        transition. Pass 0.0 on terminal, V(s_last) on timeout.

        Returns:
            Dict with scalar losses for logging.
        """
        if not self._buf_rewards:
            return {}

        # --- GAE advantage computation ---
        advantages = []
        gae = 0.0
        next_val = last_value
        for r, d, v in zip(
            reversed(self._buf_rewards),
            reversed(self._buf_dones),
            reversed(self._buf_values),
        ):
            delta = r + self.gamma * next_val * (1.0 - d) - v
            gae = delta + self.gamma * self.gae_lambda * (1.0 - d) * gae
            advantages.insert(0, gae)
            next_val = v

        adv_t = torch.tensor(advantages, dtype=torch.float32)
        ret_t = adv_t + torch.tensor(self._buf_values, dtype=torch.float32)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        states_t = torch.stack([self._encode(s) for s in self._buf_states])
        actions_t = torch.tensor(self._buf_actions, dtype=torch.long)
        old_lp_t = torch.tensor(self._buf_log_probs, dtype=torch.float32)

        n = adv_t.shape[0]
        actor_loss_val = critic_loss_val = entropy_val = 0.0

        # K epochs of SGD over shuffled minibatches (canonical PPO).
        for _ in range(self.k_epochs):
            perm = torch.randperm(n)
            for start in range(0, n, self.minibatch_size):
                mb = perm[start:start + self.minibatch_size]
                log_probs, values, entropy = self.policy.evaluate(
                    states_t[mb], actions_t[mb])

                ratios = torch.exp(log_probs - old_lp_t[mb])
                surr1 = ratios * adv_t[mb]
                surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t[mb]
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.functional.mse_loss(values, ret_t[mb])
                entropy_loss = -entropy.mean()

                loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                actor_loss_val = actor_loss.item()
                critic_loss_val = critic_loss.item()
                entropy_val = entropy.mean().item()

        self._reset_buffer()
        return {
            "actor_loss": actor_loss_val,
            "critic_loss": critic_loss_val,
            "entropy": entropy_val,
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])