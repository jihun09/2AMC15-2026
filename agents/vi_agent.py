"""Value Iteration agent (private; do not push to shared branches).

Solves the gridworld MDP exactly given the env's known dynamics:
  - Target cells (value 3) are terminal.
  - Walls/obstacles (1, 2) are blocking — attempting to enter leaves the
    agent in place but still incurs the wall reward.
  - Stochasticity: with prob (1 - sigma) the requested action is taken;
    with prob sigma a uniformly random action from {0,1,2,3} is taken
    instead.

Wall-reward semantics note
--------------------------
The environment computes the reward from the *intended* new_pos (before
blocking), so bumping into a wall yields -5 even though the agent stays
put.  _bellman_q therefore passes the *intended* next cell (s_intended)
to reward_fn, regardless of whether the move is blocked.  The actual
resulting state (s_next, which equals s when blocked) is used only for
the value bootstrap term.
"""
from __future__ import annotations
import numpy as np

from agents.base_agent import BaseAgent
from utils import compute_bfs_distances, reward_fn, shaped_reward


# Action mapping matches world/helpers.py:ACTIONS_TO_DIRECTIONS exactly.
# 0=col+1, 1=col-1, 2=row-1, 3=row+1.
DELTAS = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0)}


class ValueIterationAgent(BaseAgent):
    """Tabular Value Iteration planner.

    Call .plan() once before .take_action(). The planner records the
    iteration count at convergence so run_matrix.py can report Metric 2.
    """

    def __init__(self, grid_cells: np.ndarray, sigma: float, gamma: float,
                 theta: float, shaping_weight: float = 0.0):
        super().__init__()
        # Normalize start cells (4 → 0) to match the env's reset() behavior.
        # The env converts cell-4 to 0 before any step, so reward_fn never
        # sees 4 during live execution. VI loads the raw grid, so we mirror
        # that conversion here.
        self.grid = grid_cells.copy()
        self.grid[self.grid == 4] = 0
        self.sigma = sigma
        self.gamma = gamma
        self.theta = theta
        self.shaping_weight = shaping_weight
        self.V: np.ndarray | None = None
        self.policy: dict[tuple[int, int], int] = {}
        self.iterations_to_converge: int | None = None
        self._bfs = compute_bfs_distances(grid_cells) if shaping_weight else None
        # Cache target coordinate(s).
        targets = list(zip(*np.where(grid_cells == 3)))
        if not targets:
            raise ValueError("Grid has no target (cell value 3).")
        self._targets = set(map(tuple, targets))

    # ---- BaseAgent compatibility ----

    def take_action(self, state):
        if not self.policy:
            raise RuntimeError("Call .plan() before .take_action().")
        return int(self.policy[state])

    def update(self, *_, **__):
        # VI is offline planning — no-op.
        pass

    # ---- Planning ----

    def plan(self):
        walkable = [
            (r, c) for r in range(self.grid.shape[0])
                   for c in range(self.grid.shape[1])
                   if self.grid[r, c] in (0, 3, 4)
        ]
        V = np.zeros(self.grid.shape, dtype=float)
        policy: dict[tuple[int, int], int] = {}

        k = 0
        while True:
            V_new = V.copy()
            for s in walkable:
                if s in self._targets:
                    policy[s] = 0  # terminal — policy value is moot
                    continue
                qs = [self._bellman_q(s, a, V) for a in range(4)]
                V_new[s] = max(qs)
                policy[s] = int(np.argmax(qs))
            delta = float(np.max(np.abs(V_new - V)))
            V = V_new
            k += 1
            if delta < self.theta:
                break
            if k > 10_000:
                raise RuntimeError(f"VI failed to converge after {k} iterations")

        self.V = V
        self.policy = policy
        self.iterations_to_converge = k

    def _bellman_q(self, s, a, V):
        """Compute Q(s, a) using the env's actual dynamics.

        The outer expectation is over the actual action realized by
        stochasticity:
          P(actual=ap | requested=a) = (1 - sigma) * I[ap==a] + sigma/4.

        Reward is computed from the *intended* target cell (before
        blocking), matching the env's behavior where reward_fn receives
        new_pos even when _move_agent keeps the agent in place.
        """
        q = 0.0
        for ap in range(4):
            p = (1.0 - self.sigma) * (1.0 if ap == a else 0.0) + self.sigma / 4.0
            s_intended, s_next = self._step(s, ap)
            r = reward_fn(self.grid, s_intended)
            if self.shaping_weight:
                r = shaped_reward(r, s, s_next, s_next in self._targets,
                                  self._bfs, self.shaping_weight)
            done = s_next in self._targets
            v_next = 0.0 if done else V[s_next]
            q += p * (r + self.gamma * v_next)
        return q

    def _step(self, s, a):
        """Deterministic transition with correct wall-reward semantics.

        Returns (s_intended, s_next):
          s_intended: the cell the agent tried to enter (may be a wall).
          s_next:     the agent's resulting position (stays put if blocked).

        This split mirrors the environment exactly:
          reward = reward_fn(grid, new_pos)   -- uses s_intended
          _move_agent(new_pos)                -- yields s_next
        """
        dr, dc = DELTAS[a]
        nr, nc = s[0] + dr, s[1] + dc
        if not (0 <= nr < self.grid.shape[0] and 0 <= nc < self.grid.shape[1]):
            # Out of bounds: agent stays put; treat boundary as wall (-5).
            # Use s itself as s_intended so reward_fn reads cell 0/3/4.
            # Boundary cells are always 1 (wall) in practice, so the agent
            # can never stand on one; if it somehow could, s_intended = s
            # gives the empty-cell reward which is conservative but safe.
            return s, s
        s_intended = (nr, nc)
        if self.grid[nr, nc] in (0, 3, 4):
            s_next = (nr, nc)
        else:
            # Blocked: agent stays at s, but reward fires on the wall cell.
            s_next = s
        return s_intended, s_next
