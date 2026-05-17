"""Value Iteration Agent.

Implements Value Iteration (Dynamic Programming). The optimal value function
and policy are computed entirely offline from the grid before any environment
interaction begins.
"""
import numpy as np
from agents import BaseAgent
from world.helpers import ACTIONS_TO_DIRECTIONS
from utils import compute_bfs_distances, shaped_reward


class ValueIterationAgent(BaseAgent):
    """Computes the optimal policy via Value Iteration, then follows it."""

    def __init__(
        self,
        grid: np.ndarray,
        sigma: float = 0.0,
        gamma: float = 0.9,
        theta: float = 1e-6,
        max_iterations: int = 20000,
        shaping_weight: float = 0.0,
    ):
        """
        Args:
            grid:           2D numpy array of the environment grid.
            sigma:          Stochasticity — probability that a random action is
                            taken instead of the intended one (matches env).
            gamma:          Discount factor.
            theta:          Convergence threshold for value updates.
            max_iterations: Hard cap on Bellman sweeps.
            shaping_weight: Scaling factor for BFS potential-based reward shaping.
                            Set to 0.0 (default) to disable shaping entirely.
        """
        super().__init__()
        self.policy: dict[tuple[int, int], int] = {}
        self.V: dict[tuple[int, int], float] = {}
        self._dist = compute_bfs_distances(grid) if shaping_weight > 0.0 else None
        self._shaping_weight = shaping_weight
        self._train(grid, sigma, gamma, theta, max_iterations)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_reward(grid: np.ndarray, new_pos: tuple[int, int]) -> float:
        """Mirrors the default environment reward function."""
        cell = grid[new_pos]
        if cell == 0:
            return -1.0
        elif cell in (1, 2):
            return -5.0
        elif cell == 3:
            return 10.0
        return -5.0

    def _transitions(
        self,
        grid: np.ndarray,
        state: tuple[int, int],
        action: int,
        sigma: float,
    ) -> list[tuple[float, tuple[int, int], float, bool]]:
        """Returns all (prob, next_state, reward, is_terminal) outcomes.

        The environment picks the intended action with probability (1-sigma)
        and a uniformly random action with probability sigma.
        When shaping_weight > 0, potential-based BFS shaping is applied
        inside the Bellman update.
        """
        n_actions = 4
        outcomes: list[tuple[float, tuple[int, int], float, bool]] = []

        for actual in range(n_actions):
            prob = (1 - sigma + sigma / n_actions) if actual == action else (sigma / n_actions)
            dr, dc = ACTIONS_TO_DIRECTIONS[actual]
            new_pos = (state[0] + dr, state[1] + dc)
            cell = grid[new_pos]
            base_reward = self._get_reward(grid, new_pos)

            if cell == 0:
                true_next, done = new_pos, False
            elif cell in (1, 2):
                true_next, done = state, False
            elif cell == 3:
                true_next, done = new_pos, True
            else:
                true_next, done = state, False

            if self._shaping_weight > 0.0 and self._dist is not None:
                reward = shaped_reward(
                    base_reward, state, true_next, done,
                    self._dist, self._shaping_weight,
                )
            else:
                reward = base_reward

            outcomes.append((prob, true_next, reward, done))

        return outcomes

    def _q_value(
        self,
        grid: np.ndarray,
        state: tuple[int, int],
        action: int,
        sigma: float,
        gamma: float,
        V: dict,
    ) -> float:
        return sum(
            prob * (reward + (0.0 if done else gamma * V.get(ns, 0.0)))
            for prob, ns, reward, done in self._transitions(grid, state, action, sigma)
        )

    # ------------------------------------------------------------------
    # Offline planning
    # ------------------------------------------------------------------

    def _train(
        self,
        grid: np.ndarray,
        sigma: float,
        gamma: float,
        theta: float,
        max_iterations: int,
    ) -> None:
        # Non-terminal states: all empty cells (value 0)
        states = [
            (r, c)
            for r in range(grid.shape[0])
            for c in range(grid.shape[1])
            if grid[r, c] == 0
        ]

        V: dict[tuple[int, int], float] = {s: 0.0 for s in states}

        for iteration in range(max_iterations):
            delta = 0.0
            for s in states:
                v_old = V[s]
                V[s] = max(
                    self._q_value(grid, s, a, sigma, gamma, V)
                    for a in range(4)
                )
                delta = max(delta, abs(v_old - V[s]))
            if delta < theta:
                print(f"Value Iteration converged after {iteration + 1} sweeps "
                      f"(delta={delta:.2e}).")
                break
        else:
            print(f"Value Iteration reached max_iterations={max_iterations} "
                  f"without full convergence (delta={delta:.2e}).")

        # Greedy policy extraction
        for s in states:
            q_values = [
                self._q_value(grid, s, a, sigma, gamma, V)
                for a in range(4)
            ]
            self.policy[s] = int(np.argmax(q_values))

        self.V = V

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def take_action(self, state: tuple[int, int]) -> int:
        """Return the greedy action for the given state."""
        return self.policy.get(state, 0)

    def update(self, state: tuple[int, int], reward: float, action: int) -> None:
        """No-op — Value Iteration is fully offline."""
        pass
