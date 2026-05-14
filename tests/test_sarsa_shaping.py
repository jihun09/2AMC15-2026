"""Confirm shaping_weight argument changes the rewards observed during training."""
import numpy as np

from sarsa import make_env
from utils import compute_bfs_distances, shaped_reward


def test_shaping_pulls_agent_toward_target(tmp_path):
    # 3x5 grid; target at (1,3). Action 0 increments col, so moving from
    # (1,0) with action 0 reduces BFS distance by 1 each step.
    grid = np.array([
        [1, 1, 1, 1, 1],
        [0, 0, 0, 3, 1],
        [1, 1, 1, 1, 1],
    ], dtype=int)
    p = tmp_path / "g.npy"
    np.save(p, grid)

    env = make_env(p, sigma=0.0, seed=0, start_pos=(1, 0))
    env.reset()
    bfs = compute_bfs_distances(env.grid)

    state = env.agent_pos
    next_state, r_base, term, _ = env.step(0)  # action 0 -> col+1
    r_shaped = shaped_reward(r_base, state, next_state, term, bfs, shaping_weight=3.0)
    # Distance decreased by 1 -> shaping adds +3 on top of the base -1 step.
    assert r_shaped == r_base + 3.0
