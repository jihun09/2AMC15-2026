"""Episode-based SARSA training script.

Bypasses the thin BaseAgent.update API. Calls SARSAAgent.learn(...)
directly with full (S, A, R, S', A', done) information. Handles
episode resets and terminal updates correctly so the +10 target reward
is actually written into the Q-table.
"""
from argparse import ArgumentParser
from pathlib import Path
import csv
import random
import time

import numpy as np
from tqdm import trange

from world import Environment
from agents.sarsa_agent import SARSAAgent


def parse_args():
    p = ArgumentParser(description="SARSA trainer (DIC 2AMC15).")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Path(s) to grid file(s).")
    p.add_argument("--no_gui", action="store_true",
                   help="Disable rendering (faster). Recommended for training.")
    p.add_argument("--episodes", type=int, default=2000,
                   help="Number of training episodes per grid.")
    p.add_argument("--max_steps", type=int, default=500,
                   help="Max steps per episode (truncation cap).")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="Environment stochasticity in [0, 1].")
    p.add_argument("--alpha", type=float, default=0.3,
                   help="Learning rate.")
    p.add_argument("--gamma", type=float, default=0.95,
                   help="Discount factor.")
    p.add_argument("--epsilon", type=float, default=0.1,
                   help="Epsilon for epsilon-greedy.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed for env + agent.")
    p.add_argument("--start_pos", type=str, default=None,
                   help="Agent start position 'row,col'.")
    p.add_argument("--log_dir", type=Path, default=Path("results"),
                   help="Directory to save per-episode CSV logs.")
    p.add_argument("--fps", type=int, default=30,
                   help="GUI render rate (only if --no_gui not set).")
    return p.parse_args()


def parse_start_pos(s: str | None):
    if s is None:
        return None
    a, b = s.split(",")
    return (int(a), int(b))


def train_one_grid(grid_path: Path, args) -> dict:
    """Train SARSA on a single grid. Returns summary dict."""
    start_pos = parse_start_pos(args.start_pos)

    env = Environment(
        grid_fp=grid_path,
        no_gui=args.no_gui,
        sigma=args.sigma,
        target_fps=args.fps,
        agent_start_pos=start_pos,
        random_seed=args.random_seed,
    )

    # Seed Python's global RNG too — env uses random.seed internally.
    # Agent uses its own Random() instance for isolation.
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    agent = SARSAAgent(
        n_actions=4,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        rng_seed=args.random_seed,
    )

    initial_pos = env.reset()
    # Lock the start position so every episode begins at the same cell.
    # Without this, env._initialize_agent_pos picks a fresh random start
    # on each reset (when no explicit start_pos was provided), which
    # turns the task into a different problem every episode.
    env.agent_start_pos = initial_pos

    episode_log = []  # list of dicts: episode, return, steps, success
    t0 = time.time()

    for ep in trange(args.episodes, desc=f"{grid_path.name}"):
        if ep > 0:
            env.reset()
        state = env.agent_pos
        action = agent.select_action(state, training=True)

        ep_return = 0.0
        ep_steps = 0
        success = False

        for _ in range(args.max_steps):
            next_state, reward, terminated, info = env.step(action)
            actual_action = info["actual_action"]

            # Pick A' for SARSA before the update.
            next_action = agent.select_action(next_state, training=True)

            agent.learn(
                state=state,
                action=actual_action,
                reward=reward,
                next_state=next_state,
                next_action=next_action,
                done=terminated,
            )

            ep_return += reward
            ep_steps += 1
            state = next_state
            action = next_action

            if terminated:
                success = True
                break

        episode_log.append({
            "episode": ep,
            "return": ep_return,
            "steps": ep_steps,
            "success": int(success),
        })

    elapsed = time.time() - t0

    # Save per-episode CSV.
    args.log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d__%H-%M-%S")
    csv_path = args.log_dir / f"sarsa_{grid_path.stem}_{stamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "return", "steps", "success"])
        writer.writeheader()
        writer.writerows(episode_log)

    # Final evaluation on the same grid (greedy policy).
    Environment.evaluate_agent(
        grid_fp=grid_path,
        agent=agent,
        max_steps=args.max_steps,
        sigma=args.sigma,
        agent_start_pos=initial_pos,
        random_seed=args.random_seed,
    )

    # Summary stats: mean return over last 10% of episodes.
    tail = episode_log[max(1, int(0.9 * len(episode_log))):]
    mean_tail_return = float(np.mean([e["return"] for e in tail])) if tail else 0.0
    success_rate_tail = float(np.mean([e["success"] for e in tail])) if tail else 0.0

    summary = {
        "grid": str(grid_path),
        "episodes": args.episodes,
        "elapsed_sec": round(elapsed, 2),
        "mean_return_last10pct": round(mean_tail_return, 2),
        "success_rate_last10pct": round(success_rate_tail, 3),
        "csv_log": str(csv_path),
    }
    return summary


def main():
    args = parse_args()
    print(
        f"SARSA training | episodes={args.episodes} alpha={args.alpha} "
        f"gamma={args.gamma} epsilon={args.epsilon} sigma={args.sigma} "
        f"seed={args.random_seed}"
    )
    for grid in args.GRID:
        summary = train_one_grid(grid, args)
        print("\n--- summary ---")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print()


if __name__ == "__main__":
    main()
