"""Cross-grid transfer test.

Train SARSA on grid A, then evaluate the learned Q-table (greedy policy)
on grid B without any further learning. Tests whether the Q-values
generalize across grid layouts.

Reports both train-grid and test-grid eval performance.
"""
from argparse import ArgumentParser
from pathlib import Path
import csv
import random
import sys
import time

import numpy as np
from tqdm import trange

from world import Environment
from agents.sarsa_agent import SARSAAgent

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def parse_args():
    p = ArgumentParser(description="SARSA cross-grid transfer test.")
    p.add_argument("--train_grid", type=Path, required=True)
    p.add_argument("--test_grid", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--eval_episodes", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--epsilon", type=float, default=0.5)
    p.add_argument("--epsilon_end", type=float, default=0.05)
    p.add_argument("--epsilon_decay_episodes", type=int, default=1500)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--out_dir", type=Path, default=Path("results"))
    return p.parse_args()


def train_on_grid(grid_path, args, seed):
    random.seed(seed)
    np.random.seed(seed)
    env = Environment(
        grid_fp=grid_path, no_gui=True, sigma=args.sigma, target_fps=-1,
        agent_start_pos=None, random_seed=seed,
    )
    env.reset()
    initial_pos = env.agent_pos
    env.agent_start_pos = initial_pos

    agent = SARSAAgent(
        n_actions=4, alpha=args.alpha, gamma=args.gamma,
        epsilon=args.epsilon, epsilon_end=args.epsilon_end,
        epsilon_decay_episodes=args.epsilon_decay_episodes, rng_seed=seed,
    )
    for ep in range(args.episodes):
        if ep > 0:
            env.reset()
        agent.start_episode()
        state = env.agent_pos
        action = agent.select_action(state, training=True)
        for _ in range(args.max_steps):
            next_state, reward, terminated, info = env.step(action)
            actual_action = info["actual_action"]
            next_action = agent.select_action(next_state, training=True)
            agent.learn(state=state, action=actual_action, reward=reward,
                        next_state=next_state, next_action=next_action,
                        done=terminated)
            state = next_state
            action = next_action
            if terminated:
                break
    return agent


def eval_on_grid(grid_path, agent, args, seed):
    """Run frozen-policy evaluation episodes. Returns (returns, successes, steps)."""
    random.seed(seed + 10000)  # different seed for eval
    np.random.seed(seed + 10000)
    env = Environment(
        grid_fp=grid_path, no_gui=True, sigma=args.sigma, target_fps=-1,
        agent_start_pos=None, random_seed=seed + 10000,
    )
    env.reset()
    initial_pos = env.agent_pos
    env.agent_start_pos = initial_pos

    returns = np.zeros(args.eval_episodes, dtype=float)
    successes = np.zeros(args.eval_episodes, dtype=int)
    steps = np.zeros(args.eval_episodes, dtype=int)

    for ep in range(args.eval_episodes):
        if ep > 0:
            env.reset()
        state = env.agent_pos
        ep_return = 0.0
        ep_steps = 0
        success = False
        for _ in range(args.max_steps):
            action = agent.select_action(state, training=False)  # greedy
            next_state, reward, terminated, info = env.step(action)
            ep_return += reward
            ep_steps += 1
            state = next_state
            if terminated:
                success = True
                break
        returns[ep] = ep_return
        successes[ep] = int(success)
        steps[ep] = ep_steps
    return returns, successes, steps


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d__%H-%M-%S")

    print(f"Transfer | train={args.train_grid.name} test={args.test_grid.name} "
          f"seeds={args.seeds} train_eps={args.episodes} eval_eps={args.eval_episodes}")

    # Per-seed results
    train_eval = []  # eval on train_grid (sanity check)
    test_eval = []   # eval on test_grid (transfer)

    t0 = time.time()
    for seed in trange(args.seeds, desc="seed"):
        agent = train_on_grid(args.train_grid, args, seed)
        # Eval on training grid (sanity check)
        r1, s1, st1 = eval_on_grid(args.train_grid, agent, args, seed)
        train_eval.append((r1, s1, st1))
        # Eval on test grid (transfer)
        r2, s2, st2 = eval_on_grid(args.test_grid, agent, args, seed)
        test_eval.append((r2, s2, st2))
    elapsed = time.time() - t0

    def summarize(results, label):
        ret = np.array([r for r, _, _ in results])
        suc = np.array([s for _, s, _ in results])
        st = np.array([t for _, _, t in results])
        return {
            "label": label,
            "mean_return": ret.mean(),
            "std_return": ret.mean(axis=1).std(),
            "success_rate": suc.mean(),
            "mean_steps": st.mean(),
        }

    a = summarize(train_eval, "eval on train grid")
    b = summarize(test_eval, "eval on test grid")

    csv_path = args.out_dir / f"transfer_{args.train_grid.stem}_to_{args.test_grid.stem}_{stamp}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "mean_return", "std_return",
                    "success_rate", "mean_steps"])
        for x in (a, b):
            w.writerow([x["label"], x["mean_return"], x["std_return"],
                        x["success_rate"], x["mean_steps"]])

    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"\n{'scenario':<22} | {'mean ret':>10} | {'std':>7} | {'success':>9} | {'steps':>7}")
    print("-" * 70)
    for x in (a, b):
        print(f"{x['label']:<22} | {x['mean_return']:>+10.2f} | "
              f"{x['std_return']:>7.2f} | {x['success_rate']:>8.2%} | "
              f"{x['mean_steps']:>7.1f}")
    print(f"\nWrote: {csv_path}")


if __name__ == "__main__":
    main()
