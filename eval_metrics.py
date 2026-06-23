from pathlib import Path
import csv
import random

import numpy as np

from world.continuous_env import ContinuousEnv
from utils import compute_bfs_distances


def evaluate_policy(agent, base_env, state_mode, max_range, start_pos,
                    n_episodes: int = 100, max_steps: int = 500,
                    bfs_dist=None) -> dict:
    
    start_pos = tuple(int(x) for x in start_pos)
    # Reload a pristine grid before BFS: base_env.grid may have been mutated
    # mid-rollout (e.g. the target consumed on reach), which would make BFS=inf.
    base_env.reset(agent_start_pos=start_pos)
    if bfs_dist is None:
        bfs_dist = compute_bfs_distances(base_env.grid, diagonal=True)
    optimal_steps = float(bfs_dist[start_pos])
    if not np.isfinite(optimal_steps):
        raise ValueError(f"start {start_pos} cannot reach the target (BFS=inf)")

    rng_state = random.getstate()
    if hasattr(agent, "training_mode"):
        agent.training_mode = False

    eval_env = ContinuousEnv(base_env, mode=state_mode, max_range=max_range)
    successes = 0
    success_ratios = []
    success_steps = []

    for _ in range(n_episodes):
        state = eval_env.reset(agent_start_pos=start_pos)
        for step in range(max_steps):
            if hasattr(agent, "select_action"):          # PPO
                action = agent.select_action(state, training=False)
            else:                                        # DQN
                action = agent.take_action(state)
            state, _, done, _ = eval_env.step(action)
            if done:
                steps = step + 1
                successes += 1
                success_steps.append(steps)
                success_ratios.append(min(1.0, optimal_steps / steps))
                break

    if hasattr(agent, "training_mode"):
        agent.training_mode = True
    random.setstate(rng_state)

    return {
        "success_rate": successes / n_episodes,
        "mean_path_eff": float(np.mean(success_ratios)) if success_ratios else 0.0,
        "std_path_eff": float(np.std(success_ratios)) if success_ratios else 0.0,
        "mean_steps_success": float(np.mean(success_steps)) if success_steps else float("nan"),
        "optimal_steps": optimal_steps,
        "n_success": successes,
        "n_episodes": n_episodes,
    }


SUMMARY_FIELDS = [
    "algo", "grid", "state_mode", "seed", "sigma", "lr", "gamma", "hidden_size",
    "epsilon_min", "epsilon_decay", "episodes", "max_steps", "eval_episodes",
    "success_rate", "mean_path_eff", "std_path_eff", "mean_steps_success",
    "optimal_steps", "n_success",
]


def append_summary(csv_path, row: dict):
    """Append one run's metrics to a shared summary CSV (writes header once)."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})
