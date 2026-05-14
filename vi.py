"""Driver for the Value Iteration agent (private).

Exposes a `train_vi(...)` adapter that run_matrix.py calls with the same
shape as `train_sarsa` and `train_mc`. VI is offline planning, so there
are no episodes — `setup["theta"]` controls the convergence threshold
and the convergence_metric returned is the iteration count.
"""
from __future__ import annotations
from pathlib import Path

from agents.vi_agent import ValueIterationAgent
from world.grid import Grid


def train_vi(grid_path: Path, start_pos, setup: dict, seed: int,
             patience: int):
    """Uniform adapter for run_matrix.py.

    `setup` keys: gamma, theta, sigma_plan (defaults to 0.1),
                  shaping_weight (defaults to 0.0).
    `patience` is ignored (VI's stopping criterion is theta).
    """
    grid_cells = Grid.load_grid(grid_path).cells
    agent = ValueIterationAgent(
        grid_cells=grid_cells,
        sigma=setup.get("sigma_plan", 0.1),
        gamma=setup["gamma"],
        theta=setup["theta"],
        shaping_weight=setup.get("shaping_weight", 0.0),
    )
    agent.plan()
    return agent, agent.iterations_to_converge


if __name__ == "__main__":
    # Manual smoke test: run VI on A1_grid and print the policy.
    import sys
    grid_path = Path(sys.argv[1] if len(sys.argv) > 1 else "grid_configs/A1_grid.npy")
    setup = {"gamma": 0.95, "theta": 1e-4, "sigma_plan": 0.1, "shaping_weight": 0.0}
    agent, k = train_vi(grid_path, (1, 12), setup, seed=0, patience=0)
    print(f"VI converged in {k} iterations on {grid_path.name}")
