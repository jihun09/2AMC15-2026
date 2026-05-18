"""SARSA Agent — tabular on-policy TD control.

Update rule:
    Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]

On terminal transitions the bootstrap is dropped: target = R.
"""
import random

from agents import BaseAgent


class SARSAAgent(BaseAgent):
    """Tabular SARSA with multiplicative epsilon decay.

    State of the table:
        q_table: dict[(state, action) -> float]
        Both `state` and `action` are hashable. Default value for any
        unseen (state, action) key is 0.0.
    """

    def __init__(
        self,
        n_actions: int = 4,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        epsilon_decay: float = 1.0,
        epsilon_min: float = 0.0,
        rng_seed: int | None = None,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table: dict[tuple, float] = {}
        self._rng = random.Random(rng_seed)

    def _q(self, state, action: int) -> float:
        return self.q_table.get((state, action), 0.0)

    def _greedy(self, state) -> int:
        return max(range(self.n_actions), key=lambda a: self._q(state, a))

    def select_action(self, state, training: bool = True) -> int:
        """Epsilon-greedy during training, greedy during evaluation."""
        if training and self._rng.random() < self.epsilon:
            return self._rng.randint(0, self.n_actions - 1)
        return self._greedy(state)

    def take_action(self, state) -> int:
        """Greedy action (no exploration). Used by env.evaluate_agent."""
        return self._greedy(state)

    def learn(self, state, action: int, reward: float,
              next_state, next_action: int, done: bool) -> None:
        """One SARSA update from a single transition."""
        if done:
            target = reward
        else:
            target = reward + self.gamma * self._q(next_state, next_action)
        td_error = target - self._q(state, action)
        self.q_table[(state, action)] = self._q(state, action) + self.alpha * td_error

    def update(self, state, reward: float, action: int) -> None:
        """No-op — SARSA updates via learn()."""
        pass

    def decay_epsilon(self) -> None:
        """Multiplicative epsilon decay; clamps at epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_greedy_action(self) -> dict:
        """Snapshot of the greedy policy for convergence detection.

        Returns dict[state -> argmax_a q_table[(state, a)]]. Only states
        that have been visited (have any q_table entry) are included.
        """
        states = {s for (s, _a) in self.q_table.keys()}
        return {s: self._greedy(s) for s in states}
