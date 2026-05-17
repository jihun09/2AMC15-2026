"""Experiment 2.1 oracle reference: Value Iteration on A1_grid, sigma=0.

Mirrors the 2.1 evaluation protocol used for SARSA and MC:
- Grid: A1_grid.npy
- Start: (1, 12)
- sigma = 0 (deterministic)
- 10 greedy rollouts per gamma, step cap 500
- BFS-optimal distance d_BFS = 28

VI is deterministic given the model, so there are no seeds and no std.
"""
from pathlib import Path

import numpy as np

from agents.value_iteration_agent import ValueIterationAgent
from world.grid import Grid
from world.helpers import ACTIONS_TO_DIRECTIONS

GRID_FP = Path(__file__).parent / "grid_configs" / "A1_grid.npy"
START = (1, 12)
D_BFS = 28
N_ROLLOUTS = 10
STEP_CAP = 500
GAMMAS = (0.6, 0.9)


def greedy_rollout(grid: np.ndarray, agent: ValueIterationAgent,
                   start: tuple[int, int], step_cap: int):
    grid = grid.copy()
    pos = start
    for step in range(1, step_cap + 1):
        action = agent.policy.get(pos, 0)
        dr, dc = ACTIONS_TO_DIRECTIONS[action]
        new_pos = (pos[0] + dr, pos[1] + dc)
        cell = grid[new_pos]
        if cell == 0:
            pos = new_pos
        elif cell == 3:
            return step, True
        # walls/obstacles: bounce, agent stays put
    return step_cap, False


def main():
    grid = Grid.load_grid(GRID_FP).cells
    # Match Environment._initialize_agent_pos: start-marker cells (value 4)
    # are treated as empty (0). VI bypasses the env and would otherwise treat
    # them as unknown / wall-like.
    grid = np.where(grid == 4, 0, grid)
    n_reachable = int(np.sum(grid == 0))

    print(f"Grid: {GRID_FP}, shape={grid.shape}, reachable empty cells={n_reachable}")
    print(f"Start: {START}, d_BFS={D_BFS}, sigma=0, step_cap={STEP_CAP}\n")

    rows = []
    for gamma in GAMMAS:
        print(f"--- gamma = {gamma} ---")
        agent = ValueIterationAgent(grid=grid, sigma=0.0, gamma=gamma)
        steps_list = []
        successes = 0
        for _ in range(N_ROLLOUTS):
            steps, reached = greedy_rollout(grid, agent, START, STEP_CAP)
            steps_list.append(steps)
            if reached:
                successes += 1
        mean_steps = float(np.mean(steps_list))
        success_rate = successes / N_ROLLOUTS
        por = min(1.0, D_BFS / mean_steps) if mean_steps > 0 else 0.0
        states_visited = len(agent.policy)
        rows.append({
            "gamma": gamma,
            "por": por,
            "success": success_rate,
            "mean_steps": mean_steps,
            "states_visited": states_visited,
            "sweeps": agent.iterations_to_converge,
            "final_delta": agent.final_delta,
        })
        print(f"  POR={por:.3f}  success={success_rate*100:.0f}%  "
              f"mean_steps={mean_steps:.1f}  states={states_visited}  "
              f"sweeps={agent.iterations_to_converge}  "
              f"final_delta={agent.final_delta:.2e}\n")

    print("\n=== Summary (Experiment 2.1 VI oracle table) ===")
    print(f"{'gamma':>6} {'POR':>6} {'Success':>8} {'MeanSteps':>10} "
          f"{'States':>7} {'Sweeps':>7} {'FinalDelta':>11}")
    for r in rows:
        print(f"{r['gamma']:>6} {r['por']:>6.3f} {r['success']*100:>7.0f}% "
              f"{r['mean_steps']:>10.1f} {r['states_visited']:>7d} "
              f"{r['sweeps']:>7d} {r['final_delta']:>11.2e}")


if __name__ == "__main__":
    main()
