import numpy as np
from world.environment import Environment


class ContinuousEnv:
    """Wrapper around Environment that converts the discrete (row, col)
    state into a continuous sensor vector using raycasting."""

    def __init__(self, env: Environment):
        self.env = env
        self.n_actions = 8

    def _get_sensor_vector(self, agent_pos, grid) -> np.ndarray:
        """Cast rays in 8 directions, return normalized distances.
        
        The sensor does not distinguish between walls, obstacles, and
        targets — it only returns how far away the nearest non-empty
        cell is in each direction. This avoids giving the agent
        information that would trivialise the task.
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
        for dx, dy in directions:
            dist = 0
            x, y = agent_pos
            while True:
                x += dx
                y += dy
                dist += 1
                if grid[x, y] != 0:  # hit wall, obstacle or target
                    break
            readings.append(dist)

        max_dist = max(grid.shape)
        return np.array(readings, dtype=np.float32) / max_dist

    def reset(self, **kwargs) -> np.ndarray:
        pos = self.env.reset(**kwargs)
        return self._get_sensor_vector(pos, self.env.grid)

    def step(self, action: int):
        pos, reward, done, info = self.env.step(action)
        state = self._get_sensor_vector(pos, self.env.grid)
        return state, reward, done, info