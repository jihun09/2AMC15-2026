"""SARSA(lambda) — on-policy TD control with eligibility traces (replacing traces).

Each step computes the TD-error and updates every (state, action) seen
in the current episode, weighted by its eligibility:

    delta = r + gamma * Q(s', a') - Q(s, a)        (non-terminal)
    delta = r - Q(s, a)                            (terminal)
    e(s, a) <- 1                                    (replacing-traces variant)
    for every (s, a) with nonzero e:
        Q(s, a) += alpha * delta * e(s, a)
        e(s, a) *= gamma * lambda                   (decay; e drops to 0 between episodes)

This is *replacing* traces (sets e(s,a)=1 on visit rather than accumulating).
Replacing tends to be more stable than accumulating on tasks with revisits.

Lambda=0 reduces to vanilla SARSA. Lambda=1 approximates Monte Carlo over
a single episode. Practical sweet spot is typically 0.5-0.9.
"""
from collections import defaultdict
import random

import numpy as np

from agents import BaseAgent


class SARSALambdaAgent(BaseAgent):
    def __init__(self, n_actions=4, alpha=0.1, gamma=0.95, epsilon=0.1,
                 epsilon_end=None, epsilon_decay_episodes=1, lambda_=0.9,
                 trace_threshold=1e-4, rng_seed=None):
        super().__init__()
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        self.trace_threshold = trace_threshold
        self.epsilon_start = epsilon
        self.epsilon_end = epsilon_end if epsilon_end is not None else epsilon
        self.epsilon_decay_episodes = max(1, epsilon_decay_episodes)
        self._episode_count = 0
        self.Q = defaultdict(lambda: np.zeros(self.n_actions))
        self.E = defaultdict(lambda: np.zeros(self.n_actions))  # eligibility traces
        self._rng = random.Random(rng_seed)

    def start_episode(self):
        self._episode_count += 1
        # Reset eligibility traces at the start of each new episode.
        self.E = defaultdict(lambda: np.zeros(self.n_actions))

    @property
    def epsilon(self):
        if self.epsilon_decay_episodes <= 1:
            return self.epsilon_start
        frac = min(1.0, self._episode_count / self.epsilon_decay_episodes)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * frac

    def select_action(self, state, training=True):
        if training and self._rng.random() < self.epsilon:
            return self._rng.randint(0, self.n_actions - 1)
        return int(np.argmax(self.Q[state]))

    def learn(self, state, action, reward, next_state, next_action, done):
        if done:
            delta = reward - self.Q[state][action]
        else:
            delta = (reward + self.gamma * self.Q[next_state][next_action]
                     - self.Q[state][action])

        # Replacing trace: set this (s, a) to 1, leave others.
        self.E[state][action] = 1.0

        # Update every (s, a) with non-trivial eligibility, then decay.
        decay = self.gamma * self.lambda_
        to_drop = []
        for s, e_vec in self.E.items():
            self.Q[s] += self.alpha * delta * e_vec
            e_vec *= decay
            if np.all(e_vec < self.trace_threshold):
                to_drop.append(s)
        for s in to_drop:
            del self.E[s]

    def take_action(self, state):
        return self.select_action(state, training=False)

    def update(self, state, reward, action):
        raise RuntimeError(
            "SARSALambdaAgent.update called via the thin BaseAgent interface. "
            "Use sarsa.py training loop which calls .learn(s, a, r, s', a', done).")
