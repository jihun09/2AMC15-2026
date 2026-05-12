"""SARSA Agent.

Tabular on-policy TD control. Uses an extended training interface
(`select_action` + `learn`) called from `train_sarsa.py`. The thin
abstract methods inherited from `BaseAgent` are kept compatible so the
class still satisfies the ABC, but they are not the real entry points.
"""
from collections import defaultdict
import random

import numpy as np

from agents import BaseAgent


class SARSAAgent(BaseAgent):
    """Tabular SARSA.

    Update rule:
        Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]

    On terminal transitions the bootstrap is dropped: target = R.
    """

    def __init__(
        self,
        n_actions: int = 4,
        alpha: float = 0.3,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        epsilon_end: float | None = None,
        epsilon_decay_episodes: int = 1,
        rng_seed: int | None = None,
    ):
        """
        epsilon                — exploration rate. If epsilon_end is None,
                                 stays fixed at this value (no decay).
        epsilon_end            — final exploration rate. If set, epsilon
                                 linearly decays from `epsilon` to this
                                 value over `epsilon_decay_episodes` calls
                                 to `start_episode()`.
        epsilon_decay_episodes — number of episodes over which the decay
                                 happens. Decay schedule is linear and
                                 clamps at the end value thereafter.
        """
        super().__init__()
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon
        self.epsilon_end = epsilon_end if epsilon_end is not None else epsilon
        self.epsilon_decay_episodes = max(1, epsilon_decay_episodes)
        self._episode_count = 0
        self.Q: dict = defaultdict(lambda: np.zeros(self.n_actions))
        self._rng = random.Random(rng_seed)

    # ---- episode bookkeeping (called by training loop) ----

    def start_episode(self) -> None:
        """Advance the internal episode counter. Drives epsilon decay.

        Safe to call once per training episode. If no decay is configured
        (epsilon_end == epsilon), this is a harmless no-op effect on the
        policy, but the counter still advances.
        """
        self._episode_count += 1

    @property
    def epsilon(self) -> float:
        """Current effective exploration rate (after any decay)."""
        if self.epsilon_decay_episodes <= 1:
            return self.epsilon_start
        frac = min(1.0, self._episode_count / self.epsilon_decay_episodes)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * frac

    # ---- real training-time interface (called by train_sarsa.py) ----

    def select_action(self, state, training: bool = True) -> int:
        """Epsilon-greedy during training, greedy during evaluation.

        During training, uses the current (possibly decayed) epsilon.
        """
        if training and self._rng.random() < self.epsilon:
            return self._rng.randint(0, self.n_actions - 1)
        return int(np.argmax(self.Q[state]))

    def learn(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        next_action: int,
        done: bool,
    ) -> None:
        """Apply one SARSA update from a single transition."""
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.Q[next_state][next_action]
        td_error = target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

    # ---- BaseAgent abstract interface (kept compatible only) ----

    def take_action(self, state) -> int:
        """Greedy action. Used by Environment.evaluate_agent during eval.

        Note: this is intentionally greedy (no exploration) so eval
        reflects the learned policy, not the exploring behavior policy.
        For training, `train_sarsa.py` calls `select_action` directly.
        """
        return self.select_action(state, training=False)

    def update(self, state, reward: float, action: int) -> None:
        """No-op. The thin BaseAgent.update signature lacks the
        information SARSA needs (previous state, next action, done).

        Raises if accidentally invoked from `train.py` so the foot-gun
        is loud rather than silent.
        """
        raise RuntimeError(
            "SARSAAgent.update was called via the thin BaseAgent interface. "
            "Use train_sarsa.py, which calls SARSAAgent.learn(state, action, "
            "reward, next_state, next_action, done) instead."
        )
