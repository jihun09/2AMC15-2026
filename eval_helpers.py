"""Shared evaluation helpers used by run_matrix.py and the sweep driver.

`bfs_optimal_steps` computes the BFS shortest path length from a start
cell to the nearest target cell. `policy_optimality_ratio` runs greedy
evaluation episodes and computes optimal_steps / mean(actual_steps).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from utils import compute_bfs_distances


def bfs_optimal_steps(grid: np.ndarray, start: tuple[int, int]) -> int:
    """Shortest path length from `start` to the nearest target cell.

    Raises ValueError if the target is unreachable from `start`.
    """
    dist = compute_bfs_distances(grid)
    d = dist[start]
    if not np.isfinite(d):
        raise ValueError(f"target unreachable from start={start}")
    return int(d)


def policy_optimality_ratio(
    grid_path: Path,
    start: tuple[int, int],
    agent,
    sigma_eval: float,
    n_eval_episodes: int,
    max_steps: int,
    seed: int,
) -> tuple[float, float]:
    """Run greedy evaluation episodes; return (POR, mean_actual_steps).

    POR = bfs_optimal_steps / mean(actual_steps).  Bounded [0, 1] when
    sigma_eval == 0 and the agent reaches the target.
    """
    from world import Environment   # late import to avoid pulling pygame on test discovery
    from world.grid import Grid

    grid_cells = Grid.load_grid(grid_path).cells
    optimal = bfs_optimal_steps(grid_cells, start)

    steps_list: list[int] = []
    for i in range(n_eval_episodes):
        env = Environment(
            grid_fp=grid_path, no_gui=True, sigma=sigma_eval,
            target_fps=-1, agent_start_pos=start,
            random_seed=seed + 10_000 + i,
        )
        state = env.reset()
        steps_taken = max_steps
        for s in range(max_steps):
            action = agent.take_action(state)
            state, _, term, _ = env.step(action)
            if term:
                steps_taken = s + 1
                break
        steps_list.append(steps_taken)

    mean_steps = float(np.mean(steps_list))
    por = optimal / mean_steps
    return por, mean_steps
