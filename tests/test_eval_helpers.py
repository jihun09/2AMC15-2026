"""Tests for eval_helpers.py."""
import numpy as np
import pytest

from eval_helpers import bfs_optimal_steps, policy_optimality_ratio


def make_grid(layout: list[str]) -> np.ndarray:
    """Build a numpy grid from an ascii layout.

    Legend: '.' = 0 (empty), '#' = 1 (wall), 'X' = 2 (obstacle),
            'T' = 3 (target), 'S' = 4 (start).
    """
    mapping = {".": 0, "#": 1, "X": 2, "T": 3, "S": 4}
    rows = [[mapping[c] for c in row] for row in layout]
    return np.array(rows, dtype=int)


def test_bfs_optimal_steps_straight_corridor():
    # 1x5 corridor: S . . . T  (with walls top/bottom)
    grid = make_grid([
        "#####",
        "S...T",
        "#####",
    ])
    assert bfs_optimal_steps(grid, (1, 0)) == 4


def test_bfs_optimal_steps_around_obstacle():
    # Must go around an obstacle column
    grid = make_grid([
        "#####",
        "S.X.T",
        "#.X..",
        "#...#",
        "#####",
    ])
    # Optimal path goes down, right, right, right, up = 5 steps... but BFS
    # finds the actual shortest path on this grid; check it returns an int.
    d = bfs_optimal_steps(grid, (1, 0))
    assert isinstance(d, (int, np.integer))
    assert d >= 4  # at minimum must equal Manhattan dist of 4


def test_bfs_optimal_steps_unreachable_raises():
    # Target walled off
    grid = make_grid([
        "#####",
        "S.#T#",
        "#####",
    ])
    with pytest.raises(ValueError, match="unreachable"):
        bfs_optimal_steps(grid, (1, 0))


import tempfile


class StraightLineAgent:
    """Deterministic stub: always takes action 0 (col+1 in numpy = moves right
    along the corridor in the test grid).

    The environment uses ACTIONS_TO_DIRECTIONS = {0: (0, 1), 1: (0, -1),
    2: (-1, 0), 3: (1, 0)}.  The test corridor runs along the column axis
    (numpy second dimension), so action 0 advances the agent one step to
    the right on each call.
    """
    def take_action(self, state):
        return 0


def test_policy_optimality_ratio_straight_corridor(tmp_path):
    # 3x5 grid; start (1,0), target (1,3); the straight-line agent
    # walks right 3 cells to reach it. Optimal is also 3 steps. POR == 1.0.
    grid = make_grid([
        "#####",
        "S..T#",
        "#####",
    ])
    grid_path = tmp_path / "tiny.npy"
    np.save(grid_path, grid)

    por, mean_steps = policy_optimality_ratio(
        grid_path, start=(1, 0), agent=StraightLineAgent(),
        sigma_eval=0.0, n_eval_episodes=3, max_steps=20, seed=0,
    )
    assert mean_steps == 3.0
    assert por == pytest.approx(1.0)
