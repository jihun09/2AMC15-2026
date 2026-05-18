"""Phase 2 sanity + transfer test for LinearSARSAAgent.

Runs two experiments:
  A) Same-grid: train and evaluate on each grid independently.
  B) Transfer:  train on one grid, evaluate greedily on others.

Writes a single tidy CSV to results/. Standalone because eval-configs
doesn't currently expose a "train on grid X, eval on grids {Y, Z, ...}"
shape — easier to do this once than generalize the CLI.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import io
import random
import time
from pathlib import Path

import numpy as np

from world import Environment
from agents.linear_sarsa_agent import LinearSARSAAgent


GRIDS = ["example_grid", "small_grid", "A1_grid", "large_grid"]


def train_then_eval(train_grid: str, eval_grid: str, seed: int, *,
                    episodes: int, max_steps: int,
                    alpha: float, gamma: float,
                    eps_start: float, eps_end: float):
    """Train on `train_grid`, then run one greedy episode on `eval_grid`."""
    tp = Path("grid_configs") / f"{train_grid}.npy"
    ep_ = Path("grid_configs") / f"{eval_grid}.npy"

    # Suppress the env's stdout prints from random start-pos placement.
    with contextlib.redirect_stdout(io.StringIO()):
        env0 = Environment(grid_fp=tp, no_gui=True, sigma=0.0, target_fps=-1,
                           agent_start_pos=None, random_seed=seed)
        train_start = env0.reset()

    random.seed(seed); np.random.seed(seed)
    env = Environment(grid_fp=tp, no_gui=True, sigma=0.1, target_fps=-1,
                      agent_start_pos=train_start, random_seed=seed)
    env.reset(); env.agent_start_pos = train_start

    agent = LinearSARSAAgent(
        alpha=alpha, gamma=gamma,
        epsilon=eps_start, epsilon_end=eps_end,
        epsilon_decay_episodes=int(episodes * 0.7),
        rng_seed=seed,
    )

    for ep in range(episodes):
        if ep > 0:
            env.reset()
        agent.start_episode()
        agent.set_context(env.grid)
        s = env.agent_pos
        a = agent.select_action(s, training=True)
        for _ in range(max_steps):
            ns, r, term, info = env.step(a)
            na = agent.select_action(ns, training=True)
            agent.learn(s, info["actual_action"], r, ns, na, term)
            s, a = ns, na
            if term:
                break

    # Greedy eval on eval_grid (sigma=0)
    with contextlib.redirect_stdout(io.StringIO()):
        env_e0 = Environment(grid_fp=ep_, no_gui=True, sigma=0.0, target_fps=-1,
                             agent_start_pos=None, random_seed=seed + 10_000)
        eval_start = env_e0.reset()

    random.seed(seed + 10_000); np.random.seed(seed + 10_000)
    env_e = Environment(grid_fp=ep_, no_gui=True, sigma=0.0, target_fps=-1,
                        agent_start_pos=eval_start, random_seed=seed + 10_000)
    s = env_e.reset()
    agent.set_context(env_e.grid)
    for _ in range(max_steps):
        a = agent.take_action(s)
        s, _, term, _ = env_e.step(a)
        if term:
            break
    ws = env_e.world_stats
    return (ws["total_steps"], ws["total_failed_moves"],
            ws["cumulative_reward"], int(ws["total_targets_reached"] > 0))


def summarize(rows):
    arr = np.array(rows, dtype=float)
    return (arr[:, 0].mean(), arr[:, 0].std(),
            arr[:, 2].mean(), arr[:, 3].mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--alpha", type=float, default=0.02)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eps_start", type=float, default=0.30)
    p.add_argument("--eps_end", type=float, default=0.05)
    p.add_argument("--transfer_train_grid", type=str, default="example_grid",
                   help="grid used as the source for the cross-grid transfer test")
    p.add_argument("--out_dir", type=Path, default=Path("results"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seeds))
    stamp = time.strftime("%Y-%m-%d__%H-%M-%S")
    out_csv = args.out_dir / f"phase2_linear_eval_{stamp}.csv"

    rows = []
    t0 = time.time()

    print("--- A: Same-grid (train = eval) ---")
    print(f"{'grid':<14} | {'steps':>10} | {'std':>6} | {'reward':>8} | {'success':>7}")
    for g in GRIDS:
        trials = [train_then_eval(g, g, s, episodes=args.episodes,
                                  max_steps=args.max_steps,
                                  alpha=args.alpha, gamma=args.gamma,
                                  eps_start=args.eps_start, eps_end=args.eps_end)
                  for s in seeds]
        m_steps, sd_steps, m_reward, m_success = summarize(trials)
        print(f"{g:<14} | {m_steps:>10.1f} | {sd_steps:>6.1f} | "
              f"{m_reward:>+8.1f} | {m_success:>6.0%}")
        for i, t in enumerate(trials):
            rows.append({
                "phase": "same_grid", "train_grid": g, "eval_grid": g,
                "seed": seeds[i], "steps": int(t[0]), "failed_moves": int(t[1]),
                "reward": float(t[2]), "success": int(t[3]),
            })

    print()
    print(f"--- B: Transfer (train={args.transfer_train_grid}, eval = each grid) ---")
    print(f"{'eval_grid':<14} | {'steps':>10} | {'std':>6} | {'reward':>8} | {'success':>7}")
    for g in GRIDS:
        trials = [train_then_eval(args.transfer_train_grid, g, s,
                                  episodes=args.episodes,
                                  max_steps=args.max_steps,
                                  alpha=args.alpha, gamma=args.gamma,
                                  eps_start=args.eps_start, eps_end=args.eps_end)
                  for s in seeds]
        m_steps, sd_steps, m_reward, m_success = summarize(trials)
        print(f"{g:<14} | {m_steps:>10.1f} | {sd_steps:>6.1f} | "
              f"{m_reward:>+8.1f} | {m_success:>6.0%}")
        for i, t in enumerate(trials):
            rows.append({
                "phase": "transfer",
                "train_grid": args.transfer_train_grid, "eval_grid": g,
                "seed": seeds[i], "steps": int(t[0]), "failed_moves": int(t[1]),
                "reward": float(t[2]), "success": int(t[3]),
            })

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nelapsed: {time.time() - t0:.1f}s  ->  {out_csv}")


if __name__ == "__main__":
    main()
