"""Gamma sweep — does the discount factor matter on grid worlds?

For each gamma in the sweep, trains SARSA across N seeds for M episodes.
"""
from argparse import ArgumentParser
from pathlib import Path
import csv
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from world import Environment
from agents.sarsa_agent import SARSAAgent

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def parse_args():
    p = ArgumentParser(description="SARSA gamma-sweep study.")
    p.add_argument("GRID", type=Path)
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--gammas", type=str, default="0.5,0.8,0.9,0.95,0.99")
    p.add_argument("--start_pos", type=str, default=None)
    p.add_argument("--out_dir", type=Path, default=Path("results"))
    return p.parse_args()


def parse_start_pos(s):
    if s is None:
        return None
    a, b = s.split(",")
    return (int(a), int(b))


def train_one_seed(grid_path, args, gamma, seed):
    start_pos = parse_start_pos(args.start_pos)
    random.seed(seed)
    np.random.seed(seed)
    env = Environment(
        grid_fp=grid_path, no_gui=True, sigma=args.sigma, target_fps=-1,
        agent_start_pos=start_pos, random_seed=seed,
    )
    env.reset()
    initial_pos = env.agent_pos
    env.agent_start_pos = initial_pos

    agent = SARSAAgent(
        n_actions=4, alpha=args.alpha, gamma=gamma,
        epsilon=args.epsilon, rng_seed=seed,
    )
    returns = np.zeros(args.episodes, dtype=float)
    successes = np.zeros(args.episodes, dtype=int)

    for ep in range(args.episodes):
        if ep > 0:
            env.reset()
        state = env.agent_pos
        action = agent.select_action(state, training=True)
        ep_return = 0.0
        success = False
        for _ in range(args.max_steps):
            next_state, reward, terminated, info = env.step(action)
            actual_action = info["actual_action"]
            next_action = agent.select_action(next_state, training=True)
            agent.learn(state=state, action=actual_action, reward=reward,
                        next_state=next_state, next_action=next_action,
                        done=terminated)
            ep_return += reward
            state = next_state
            action = next_action
            if terminated:
                success = True
                break
        returns[ep] = ep_return
        successes[ep] = int(success)
    return returns, successes


def smooth(x, k=10):
    if k <= 1:
        return x
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="valid")


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d__%H-%M-%S")
    gammas = [float(g) for g in args.gammas.split(",")]

    print(f"Gamma sweep | grid={args.GRID.name} gammas={gammas} "
          f"seeds={args.seeds} episodes={args.episodes}")

    all_returns = {}
    all_succ = {}
    t0 = time.time()
    for gamma in gammas:
        ret = np.zeros((args.seeds, args.episodes))
        suc = np.zeros((args.seeds, args.episodes), dtype=int)
        for seed in trange(args.seeds, desc=f"gamma={gamma}"):
            r, s = train_one_seed(args.GRID, args, gamma, seed)
            ret[seed] = r
            suc[seed] = s
        all_returns[gamma] = ret
        all_succ[gamma] = suc
    elapsed = time.time() - t0

    smooth_k = max(1, args.episodes // 50)
    eps = np.arange(args.episodes)
    eps_s = eps[smooth_k - 1:] if smooth_k > 1 else eps

    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("viridis")
    for i, g in enumerate(gammas):
        c = cmap(i / max(1, len(gammas) - 1))
        m = smooth(all_returns[g].mean(axis=0), smooth_k)
        s = smooth(all_returns[g].std(axis=0), smooth_k)
        ax.plot(eps_s, m, label=f"gamma={g}", color=c, linewidth=2)
        ax.fill_between(eps_s, m - s, m + s, color=c, alpha=0.13)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(f"SARSA learning curves vs gamma - {args.GRID.name} | "
                 f"sigma={args.sigma}, alpha={args.alpha}, eps={args.epsilon}, "
                 f"{args.seeds} seeds")
    ax.legend(loc="lower right", title="Discount")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = args.out_dir / f"sweep_gamma_curves_{stamp}.png"
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)

    last_n = max(1, int(0.1 * args.episodes))
    csv_path = args.out_dir / f"sweep_gamma_{stamp}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gamma", "mean_return_last10pct", "std_return_last10pct",
                    "success_rate_last10pct"])
        for g in gammas:
            m = all_returns[g][:, -last_n:].mean()
            s = all_returns[g][:, -last_n:].mean(axis=1).std()
            sr = all_succ[g][:, -last_n:].mean()
            w.writerow([g, m, s, sr])

    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"\n{'gamma':>6} | {'mean ret':>10} | {'std':>7} | {'success':>9}")
    print("-" * 42)
    for g in gammas:
        m = all_returns[g][:, -last_n:].mean()
        s = all_returns[g][:, -last_n:].mean(axis=1).std()
        sr = all_succ[g][:, -last_n:].mean()
        print(f"{g:>6.2f} | {m:>+10.2f} | {s:>7.2f} | {sr:>8.2%}")
    print(f"\nWrote:\n  {plot_path}\n  {csv_path}")


if __name__ == "__main__":
    main()
