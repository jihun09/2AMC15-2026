"""Value Iteration agent on a tiny corridor."""
import numpy as np
import pytest

from agents.vi_agent import ValueIterationAgent


def test_vi_converges_on_corridor():
    # 3x5 grid; target at column 3.
    grid = np.array([
        [1, 1, 1, 1, 1],
        [0, 0, 0, 3, 1],
        [1, 1, 1, 1, 1],
    ], dtype=int)

    agent = ValueIterationAgent(
        grid_cells=grid, sigma=0.0, gamma=0.95, theta=1e-6,
        shaping_weight=0.0,
    )
    agent.plan()

    # From (1, 0), greedy policy must choose action 0 (col+1).
    assert agent.policy[(1, 0)] == 0
    assert agent.iterations_to_converge is not None
    assert agent.iterations_to_converge > 0
    # Should converge in fewer iterations than the corridor length squared.
    assert agent.iterations_to_converge < 50


def test_vi_take_action_returns_policy_value():
    grid = np.array([
        [1, 1, 1, 1, 1],
        [0, 0, 0, 3, 1],
        [1, 1, 1, 1, 1],
    ], dtype=int)
    agent = ValueIterationAgent(grid, sigma=0.0, gamma=0.95, theta=1e-6,
                                shaping_weight=0.0)
    agent.plan()
    # All three states on the corridor should choose action 0 (col+1).
    assert agent.take_action((1, 0)) == 0
    assert agent.take_action((1, 1)) == 0
    assert agent.take_action((1, 2)) == 0
