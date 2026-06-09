import numpy as np
from world.environment import Environment


class ContinuousEnv:
    """Wrapper around Environment that converts the discrete (row, col)
    state into a continuous sensor vector using raycasting.
    
    Each ray returns a normalized distance and a binary flag indicating
    whether the cell hit was the target (1.0) or a wall/obstacle (0.0).
    The state dimension is therefore 2 * 8 = 16.

    Args:
        env: The base Environment to wrap.
        max_range: Maximum number of cells a ray can travel. If None,
            rays travel until they hit a non-empty cell (full raycasting).
            If set to an integer, rays stop after that many cells even if
            nothing was hit — simulates a sensor with limited range.
    """

    def __init__(self, env: Environment, max_range: int = None):
        self.env = env
        self.n_actions = 8
        self.max_range = max_range  # None = full raycasting, int = limited range

    def _get_sensor_vector(self, agent_pos, grid) -> np.ndarray:
        """Cast rays in 8 directions, return normalized distances and target flags.
        
        For each direction, the sensor returns:
          - Normalized distance to the nearest non-empty cell (float in (0, 1])
          - Binary flag: 1.0 if the cell hit is the target, 0.0 if wall/obstacle

        The sensor does NOT distinguish between walls and obstacles — both
        return flag 0.0. Only the target returns flag 1.0. This avoids giving
        the agent a direct distance-to-target signal, which would trivialise
        the task and violate the assignment constraints.

        If max_range is set, the ray stops after that many cells even if
        nothing was hit, and the flag is 0.0 (no target seen in range).
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
            hit_obstacle = False

            while True:
                x += dx
                y += dy
                dist += 1
                if grid[x, y] != 0:  # hit wall, obstacle or target
                    hit_obstacle = True
                    break
                if self.max_range is not None and dist >= self.max_range:
                    break  # reached sensor range limit without hitting anything

            # Normalized distance
            norm_dist = dist / max_dist
            # Binary target flag: 1.0 only if the ray actually hit the target
            is_target = 1.0 if (hit_obstacle and grid[x, y] == 3) else 0.0

            readings.append(norm_dist)
            readings.append(is_target)

        return np.array(readings, dtype=np.float32)  # shape (16,)

    def reset(self, **kwargs) -> np.ndarray:
        pos = self.env.reset(**kwargs)
        return self._get_sensor_vector(pos, self.env.grid)

    def step(self, action: int):
        pos, reward, done, info = self.env.step(action)
        state = self._get_sensor_vector(pos, self.env.grid)
        return state, reward, done, info