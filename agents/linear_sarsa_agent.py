"""Linear-function-approximation SARSA — Phase 2 scaffold.

Replaces the tabular Q-table with Q(s, a) = w_a . phi(s), where:
- phi(s) is a fixed feature vector built from agent position + grid context
- w_a is a learned weight vector per action

This is the bridge from tabular SARSA toward DQN. The same `train_agent`
training loop drives it; only the agent's internal representation changes.

Crucially, phi(s) generalizes across grids — train on A1, evaluate on
small_grid, and the features remain meaningful. Tabular SARSA's transfer
score was 0% (Section 4.5); this is the first thing that can do better.
"""
from __future__ import annotations
import random

import numpy as np

from agents import BaseAgent


class LinearSARSAAgent(BaseAgent):
    """SARSA with linear function approximation over hand-crafted features.

    Feature vector phi(s) (10-d) for state s = (row, col):
      0  dx_norm     = (target_row - row) / grid_height
      1  dy_norm     = (target_col - col) / grid_width
      2  |dx_norm|
      3  |dy_norm|
      4  mdist_norm  = (|dx| + |dy|) / (grid_h + grid_w)
      5  wall_N      = 1 if cell above is wall/obstacle/OOB else 0
      6  wall_S
      7  wall_E
      8  wall_W
      9  bias        = 1.0

    Q(s, a) = self.W[a] @ phi(s).  Update: W[a] += alpha * delta * phi(s).

    Normalizing dx, dy by the grid dimensions keeps feature magnitudes in
    the same range across grids of different sizes — without this the
    same weight matrix would behave very differently on an 8x8 vs a 32x22
    grid, defeating the point of function approximation for transfer.
    """

    N_FEATURES = 10

    def __init__(
        self,
        n_actions: int = 4,
        alpha: float = 0.05,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        epsilon_end: float | None = None,
        epsilon_decay_episodes: int = 1,
        q_init: float | None = None,  # accepted for interface parity; ignored
        rng_seed: int | None = None,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon
        self.epsilon_end = epsilon_end if epsilon_end is not None else epsilon
        self.epsilon_decay_episodes = max(1, epsilon_decay_episodes)
        self._episode_count = 0
        # One weight row per action; columns are features. Zero init is the
        # standard choice for linear TD and avoids the optimistic-init
        # numerical issues we saw with the tabular SARSA(lambda) agent.
        self.W = np.zeros((n_actions, self.N_FEATURES), dtype=float)
        self._rng = random.Random(rng_seed)
        # Grid context — set per-episode via set_context().
        self._grid: np.ndarray | None = None
        self._target: tuple[int, int] | None = None
        self._h: int = 0
        self._w: int = 0


    # EPISODE BOOKKEEPING

    def start_episode(self) -> None:
        self._episode_count += 1

    @property
    def epsilon(self) -> float:
        if self.epsilon_decay_episodes <= 1:
            return self.epsilon_start
        frac = min(1.0, self._episode_count / self.epsilon_decay_episodes)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * frac


    # GRID CONTEXT: CALLED BY TRAINING LOOP RIGHT AFTER env.reset()

    def set_context(self, grid: np.ndarray) -> None:
        """Cache the grid and target position for this episode.

        Must be called once per episode after env.reset(); featurize()
        relies on the cached values to avoid scanning the grid every step.
        """
        self._grid = grid
        self._h, self._w = grid.shape
        target_rows, target_cols = np.where(grid == 3)
        if len(target_rows) > 0:
            self._target = (int(target_rows[0]), int(target_cols[0]))
        else:
            self._target = None  # no target on this grid (terminal-reached)


    # FEATURE EXTRACTOR

    def featurize(self, state: tuple[int, int]) -> np.ndarray:
        """Build the 10-d feature vector for the current (row, col)."""
        if self._target is None or self._grid is None:
            # Degenerate: no context. Return bias-only so Q is still defined.
            phi = np.zeros(self.N_FEATURES, dtype=float)
            phi[-1] = 1.0
            return phi
        r, c = state
        tr, tc = self._target
        dx = (tr - r) / max(1, self._h)
        dy = (tc - c) / max(1, self._w)
        mdist = (abs(tr - r) + abs(tc - c)) / max(1, self._h + self._w)
        return np.array([
            dx, dy, abs(dx), abs(dy), mdist,
            self._wall(r - 1, c),  # N
            self._wall(r + 1, c),  # S
            self._wall(r, c + 1),  # E
            self._wall(r, c - 1),  # W
            1.0,                   # bias
        ], dtype=float)

    def _wall(self, rr: int, cc: int) -> float:
        if rr < 0 or rr >= self._h or cc < 0 or cc >= self._w:
            return 1.0  # off-grid counts as wall
        v = self._grid[rr, cc]
        return 1.0 if (v == 1 or v == 2) else 0.0

    def q_values(self, state) -> np.ndarray:
        """Per-action Q-vector for `state`. Shape: (n_actions,)."""
        return self.W @ self.featurize(state)

    # policy + learning

    def select_action(self, state, training: bool = True) -> int:
        if training and self._rng.random() < self.epsilon:
            return self._rng.randint(0, self.n_actions - 1)
        return int(np.argmax(self.q_values(state)))

    def learn(self, state, action: int, reward: float, next_state,
              next_action: int, done: bool) -> None:
        """Apply one SARSA update to the weight matrix."""
        phi = self.featurize(state)
        q_sa = float(self.W[action] @ phi)
        if done:
            target = reward
        else:
            phi_next = self.featurize(next_state)
            q_next = float(self.W[next_action] @ phi_next)
            target = reward + self.gamma * q_next
        delta = target - q_sa
        self.W[action] += self.alpha * delta * phi

    def take_action(self, state) -> int:
        """Greedy. Used by Environment.evaluate_agent during eval."""
        return self.select_action(state, training=False)

    def update(self, state, reward: float, action: int) -> None:
        raise RuntimeError(
            "LinearSARSAAgent.update called via the thin BaseAgent interface. "
            "Use sarsa.py's train_agent which calls .learn(s, a, r, s', a', "
            "done) and .set_context(grid) per episode.")
