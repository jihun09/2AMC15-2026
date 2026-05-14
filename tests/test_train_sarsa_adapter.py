"""train_sarsa adapter has the uniform signature run_matrix.py expects."""
import numpy as np

from sarsa import train_sarsa


def test_train_sarsa_returns_agent_and_convergence(tmp_path):
    grid = np.array([
        [1, 1, 1, 1, 1],
        [0, 0, 0, 3, 1],
        [1, 1, 1, 1, 1],
    ], dtype=int)
    p = tmp_path / "g.npy"
    np.save(p, grid)

    setup = {
        "alpha": 0.5, "gamma": 0.95,
        "epsilon": 0.1, "epsilon_end": None, "epsilon_decay_episodes": 1,
        "sigma_train": 0.0, "episodes": 300, "max_steps": 50,
        "shaping_weight": 0.0, "q_init": 0.0,
    }
    agent, conv = train_sarsa(p, (1, 0), setup, seed=0, patience=20)
    assert agent is not None
    assert hasattr(agent, "take_action")
    # convergence_episode can be None (didn't converge) but on this trivial
    # grid it should fire.
    assert conv is not None and conv > 0
