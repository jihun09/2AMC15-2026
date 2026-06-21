"""
Shared utilities for all RL agents.

Every training script imports from here to ensure a consistent
reward function and BFS shaping across all algorithms.

Usage
-----
All algorithms:
    from utils import reward_fn, compute_bfs_distances, shaped_reward

MC (apply shaping inside the step loop, before appending reward):
    dist = compute_bfs_distances(grid_cells)
    ...
    prev_state = state
    new_state, reward, terminated, info = env.step(action)
    reward = shaped_reward(reward, prev_state, new_state, terminated, dist, shaping_weight)

SARSA (same as MC — apply shaping at every step):

VI (apply shaping directly in the Bellman update):
    dist = compute_bfs_distances(grid_cells)
    ...
    r = reward_fn(grid_cells, s_next)
    r = shaped_reward(r, s, s_next, s_next == target, dist, shaping_weight)
"""

from collections import deque
import numpy as np


def reward_fn(grid: np.ndarray, agent_pos: tuple[int, int]) -> float:
    """Base reward function used by all algorithms.

    Args:
        grid:      2D numpy array of the grid (cell type values).
        agent_pos: (row, col) position the agent just moved to.

    Returns:
        -1   for a free cell (step penalty — encourages faster navigation)
        -5   for a wall or obstacle (invalid/costly move)
        +100 for reaching the target (strong positive signal)
    """
    match grid[agent_pos]:
        case 0:     return -1
        case 1 | 2: return -5
        case 3:     return 100
        case _:
            raise ValueError(f"Unexpected grid value {grid[agent_pos]} "
                             f"at position {agent_pos}")


def compute_bfs_distances(grid: np.ndarray, diagonal: bool = True) -> np.ndarray:
    """Shortest-path distance from every cell to the target via BFS.

    Args:
        grid:     2D numpy array of the grid.
        diagonal: If True (default), use 8-connected moves (4 cardinal + 4
                  diagonal), matching the 8-action environment, and forbid
                  diagonal corner-cutting exactly as the environment does
                  (a diagonal is blocked when both shared adjacent cardinal
                  cells are walls/obstacles). If False, use 4-connected moves.

    Returns:
        dist: float array of same shape as grid.
              dist[r, c] = shortest number of steps from (r, c) to target.
              Walls, obstacles, and unreachable cells are set to np.inf.
    """
    dist = np.full(grid.shape, np.inf)
    targets = list(zip(*np.where(grid == 3)))
    if not targets:
        return dist
    queue = deque()
    for t in targets:
        dist[t] = 0
        queue.append(t)

    cardinal = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    diagonals = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    directions = cardinal + diagonals if diagonal else cardinal

    n_rows, n_cols = grid.shape
    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n_rows and 0 <= nc < n_cols):
                continue
            if dist[nr, nc] != np.inf or grid[nr, nc] not in (0, 3, 4):
                continue
            # Corner-cutting prevention (matches the environment): a diagonal
            # move is blocked if both shared adjacent cardinal cells are
            # walls/obstacles (values 1 or 2).
            if dr != 0 and dc != 0:
                if grid[nr, c] in (1, 2) and grid[r, nc] in (1, 2):
                    continue
            dist[nr, nc] = dist[r, c] + 1
            queue.append((nr, nc))
    return dist


def shaped_reward(base_reward: float,
                  prev_state: tuple[int, int],
                  next_state: tuple[int, int],
                  terminated: bool,
                  dist: np.ndarray,
                  shaping_weight: float = 3.0) -> float:
    """Applies potential-based reward shaping on top of the base reward.

    Moving closer to the target gives a positive bonus.
    Moving away gives a negative penalty.
    No shaping is applied on the terminal step to avoid distorting
    the final reward signal.

    Args:
        base_reward:    Reward returned by the environment.
        prev_state:     (row, col) before the step.
        next_state:     (row, col) after the step.
        terminated:     Whether the episode ended on this step.
        dist:           BFS distance array from compute_bfs_distances().
        shaping_weight: Scaling factor for the shaping bonus (default 3.0).

    Returns:
        Shaped reward as a float.
    """
    if terminated:
        return base_reward
    shaping = shaping_weight * (dist[prev_state] - dist[next_state])
    # If next_state is unreachable (inf distance), do not apply shaping
    if not np.isfinite(dist[next_state]):
        return base_reward
    return base_reward + shaping
