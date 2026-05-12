"""Q-learning agent (off-policy TD control).

Same interface as SARSAAgent but uses the off-policy update:
    Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]

That is, the bootstrap target is the *greedy* future value, not the value
of the action the behavior policy actually takes. The bootstrap is dropped
on terminal transitions, same as SARSA.
"""
from collections import defaultdict
import random

import numpy as np

from agents import BaseAgent


class QLearningAgent(BaseAgent):
    def __init__(self, n_actions=4, alpha=0.1, gamma=0.95, epsilon=0.1,
                 epsilon_end=None, epsilon_decay_episodes=1, rng_seed=None):
        super().__init__()
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon
        self.epsilon_end = epsilon_end if epsilon_end is not None else epsilon
        self.epsilon_decay_episodes = max(1, epsilon_decay_episodes)
        self._episode_count = 0
        self.Q = defaultdict(lambda: np.zeros(self.n_actions))
        self._rng = random.Random(rng_seed)

    def start_episode(self):
        self._episode_count += 1

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
        # next_action is ignored — off-policy uses max over next state
        if done:
            target = reward
        else:
            target = reward + self.gamma * float(np.max(self.Q[next_state]))
        self.Q[state][action] += self.alpha * (target - self.Q[state][action])

    def take_action(self, state):
        return self.select_action(state, training=False)

    def update(self, state, reward, action):
        raise RuntimeError(
            "QLearningAgent.update called via the thin BaseAgent interface. "
            "Use sarsa.py training loop which calls .learn(s, a, r, s', a', done).")
