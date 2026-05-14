"""Patience-based stopping criterion in sarsa.train_agent.

The greedy policy hash (a tuple of argmax over visited states) is the
basis. If the hash is unchanged for `patience` consecutive episodes,
training stops early and `convergence_episode` records the episode index
on which stability was reached.
"""
import numpy as np

from sarsa import train_agent


def test_patience_triggers_on_easy_grid(tmp_path):
    # 3x5 corridor: very easy. Convergence should fire well before 1000.
    grid = np.array([
        [1, 1, 1, 1, 1],
        [0, 0, 0, 3, 1],
        [1, 1, 1, 1, 1],
    ], dtype=int)
    p = tmp_path / "g.npy"
    np.save(p, grid)

    agent, returns, steps, succ, conv_ep = train_agent(
        p, start=(1, 0), alpha=0.5, gamma=0.95, epsilon=0.1,
        sigma=0.0, episodes=1000, max_steps=50, seed=0,
        patience=20, track_per_episode=True,
    )
    assert conv_ep is not None, "convergence_episode should be set"
    assert conv_ep < 1000, f"expected early stop, got {conv_ep}"


def test_patience_disabled_returns_none(tmp_path):
    # Same corridor, but patience=0 disables early stop.
    grid = np.array([
        [1, 1, 1, 1, 1],
        [0, 0, 0, 3, 1],
        [1, 1, 1, 1, 1],
    ], dtype=int)
    p = tmp_path / "g.npy"
    np.save(p, grid)

    agent, returns, steps, succ, conv_ep = train_agent(
        p, start=(1, 0), alpha=0.5, gamma=0.95, epsilon=0.1,
        sigma=0.0, episodes=200, max_steps=50, seed=0,
        patience=0, track_per_episode=True,
    )
    assert conv_ep is None
