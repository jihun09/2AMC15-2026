import numpy as np
from world.environment import Environment

MODES = ("gps", "raycasting", "both")


class ContinuousEnv:
    """Wrapper around Environment that converts the discrete (row, col)
    state into a continuous observation vector.

    Three modes are available:
      - 'gps':        normalized (row, col) coordinates — state_dim = 2
      - 'raycasting': 8-directional distance sensor — state_dim = 8
      - 'both':       GPS + raycasting concatenated — state_dim = 10

    Note: the raycasting sensor returns only obstacle distances and does NOT
    identify the target (no target-vs-obstacle flag), to comply with the
    assignment rule against features that reveal the optimal action.

    Args:
        env:       The base Environment to wrap.
        mode:      One of 'gps', 'raycasting', or 'both'.
        max_range: Max ray length in cells. None = full raycasting.
    """

    def __init__(self, env: Environment, mode: str = "gps", max_range: int = None):
        assert mode in MODES, f"mode must be one of {MODES}"
        self.env = env
        self.n_actions = 8
        self.mode = mode
        self.max_range = max_range
        self.state_dim = {"gps": 2, "raycasting": 8, "both": 10}[mode]

    def _get_gps(self, agent_pos) -> np.ndarray:
        """Normalized GPS coordinates (row/n_rows, col/n_cols). Shape (2,)."""
        row, col = agent_pos
        n_rows, n_cols = self.env.grid.shape
        return np.array([row / n_rows, col / n_cols], dtype=np.float32)

    def _get_raycasting(self, agent_pos, grid) -> np.ndarray:
        """8-directional raycasting: normalized distance to the nearest blocking
        cell per direction.

        The sensor reports only distances and does NOT identify what it hit:
        walls, obstacles and the target are indistinguishable. This deliberately
        avoids a target-vs-obstacle flag, which would reveal the optimal action
        and is disallowed by the assignment. max_range limits the ray length if
        set. Shape (8,).
        """
        directions = [
            (0, 1),   # Down
            (0, -1),  # Up
            (-1, 0),  # Left
            (1, 0),   # Right
            (1, 1),   # Down-Right
            (-1, 1),  # Down-Left
            (1, -1),  # Up-Right
            (-1, -1)  # Up-Left
        ]
        readings = []
        max_dist = max(grid.shape)

        for dx, dy in directions:
            dist = 0
            x, y = agent_pos

            while True:
                x += dx
                y += dy
                dist += 1
                if grid[x, y] != 0:  # hit wall, obstacle or target (unlabeled)
                    break
                if self.max_range is not None and dist >= self.max_range:
                    break  # reached sensor range limit

            norm_dist = dist / max_dist
            readings.append(norm_dist)

        return np.array(readings, dtype=np.float32)

    def _get_obs(self, agent_pos) -> np.ndarray:
        """Build the observation vector according to the current mode."""
        if self.mode == "gps":
            return self._get_gps(agent_pos)
        elif self.mode == "raycasting":
            return self._get_raycasting(agent_pos, self.env.grid)
        else:  # both
            gps = self._get_gps(agent_pos)
            rays = self._get_raycasting(agent_pos, self.env.grid)
            return np.concatenate([gps, rays])  # shape (18,)

    def reset(self, **kwargs) -> np.ndarray:
        pos = self.env.reset(**kwargs)
        return self._get_obs(pos)

    def step(self, action: int):
        pos, reward, done, info = self.env.step(action)
        state = self._get_obs(pos)
        return state, reward, done, info