"""Smoke tests for sarsa.py's reward wiring."""
import numpy as np

from sarsa import make_env


def test_make_env_uses_utils_reward_fn(tmp_path):
    # 3x5 grid where (1,3) is the target.
    grid = np.array([
        [1, 1, 1, 1, 1],
        [0, 0, 0, 3, 1],   # target at col 3
        [1, 1, 1, 1, 1],
    ], dtype=int)
    p = tmp_path / "g.npy"
    np.save(p, grid)

    env = make_env(p, sigma=0.0, seed=0, start_pos=(1, 0))
    env.reset()
    # Action 0 moves col+1; three steps go (1,0) -> (1,1) -> (1,2) -> (1,3).
    _, r1, _, _ = env.step(0)
    _, r2, _, _ = env.step(0)
    _, r3, term, _ = env.step(0)
    assert r1 == -1 and r2 == -1
    # utils.reward_fn returns +100 for a target; the env default returns +10.
    assert r3 == 100, f"expected +100 from utils.reward_fn, got {r3}"
    assert term is True
