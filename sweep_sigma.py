"""Sigma sweep — how robust is SARSA to environment stochasticity?

For each sigma in the sweep, trains SARSA across N seeds for M episodes.
Produces:
  - Learning-curve plot, one line per sigma (mean ± std band).
  - Bar chart of final-episodes mean return + success rate per sigma.
  - Summary CSV.
"""
from argparse import ArgumentParser
from pathlib import Path
import csv
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from world import Environment
from agents.sarsa_agent import SARSAAgent


def parse_args():
    p = ArgumentParser(description="SARSA sigma-sweep robustness study.")
    p.add_argument("GRID", type=Path)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--sigmas", type=str, default="0.0,0.1,0.3,0.5",
                   help="Comma-separated list of sigma values to sweep.")
    p.add_argument("--start_pos", type=str, default=None)
    p.add_argument("--out_dir", type=Path, default=Path("results"))
    return p.parse_args()


def parse_start_pos(s):
    if s is None:
        return None
    a, b = s.split(",")
    return (int(a), int(b))


def train_sarsa_one_seed(grid_path, args, sigma, seed):
    """Train SARSA for one seed at one sigma. Returns (returns, successes)."""
    start_pos = parse_start_pos(args.start_pos)
    random.seed(seed)
    np.random.seed(seed)

    env = Environment(
        grid_fp=grid_path,
        no_gui=True,
        sigma=sigma,
        target_fps=-1,
        agent_start_pos=start_pos,
        random_seed=seed,
    )
    env.reset()
    initial_pos = env.agent_pos
    env.agent_start_pos = initial_pos

    agent = SARSAAgent(
        n_actions=4,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        rng_seed=seed,
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
            agent.learn(
                state=state,
                action=actual_action,
                reward=reward,
                next_state=next_state,
                next_action=next_action,
                done=terminated,
            )
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
    sigmas = [float(s) for s in args.sigmas.split(",")]

    print(
        f"Sigma sweep | grid={args.GRID.name} sigmas={sigmas} "
        f"seeds={args.seeds} episodes={args.episodes}"
    )
    # Reconfigure stdout to UTF-8 so Greek letters in tqdm/print work on Windows.
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # Shape: {sigma: ndarray(seeds, episodes)}
    all_returns = {}
    all_successes = {}
    t0 = time.time()
    for sigma in sigmas:
        ret = np.zeros((args.seeds, args.episodes))
        suc = np.zeros((args.seeds, args.episodes), dtype=int)
        for seed in trange(args.seeds, desc=f"sigma={sigma}"):
            r, s = train_sarsa_one_seed(args.GRID, args, sigma, seed)
            ret[seed] = r
            suc[seed] = s
        all_returns[sigma] = ret
        all_successes[sigma] = suc
    elapsed = time.time() - t0

    # --- Plot 1: learning curves overlaid ---
    smooth_k = max(1, args.episodes // 50)
    eps = np.arange(args.episodes)
    eps_smooth = eps[smooth_k - 1:] if smooth_k > 1 else eps

    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("viridis")
    for i, sigma in enumerate(sigmas):
        color = cmap(i / max(1, len(sigmas) - 1))
        mean = all_returns[sigma].mean(axis=0)
        std = all_returns[sigma].std(axis=0)
        m_s = smooth(mean, smooth_k)
        s_s = smooth(std, smooth_k)
        ax.plot(eps_smooth, m_s, label=f"σ={sigma}", color=color, linewidth=2)
        ax.fill_between(eps_smooth, m_s - s_s, m_s + s_s, color=color, alpha=0.15)

    ax.axhline(0, color="gray", linestyle=":", linewidth=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(
        f"SARSA learning curves vs σ — {args.GRID.name}  |  "
        f"α={args.alpha}, γ={args.gamma}, ε={args.epsilon}, "
        f"{args.seeds} seeds, smoothed (k={smooth_k})"
    )
    ax.legend(loc="lower right", title="Stochasticity")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    curve_path = args.out_dir / f"sweep_sigma_curves_{stamp}.png"
    plt.savefig(curve_path, dpi=120)
    plt.close(fig)

    # --- Plot 2: bar chart of final perf ---
    last_n = max(1, int(0.1 * args.episodes))
    final_means = [all_returns[s][:, -last_n:].mean() for s in sigmas]
    final_stds = [all_returns[s][:, -last_n:].mean(axis=1).std() for s in sigmas]
    final_succ = [all_successes[s][:, -last_n:].mean() for s in sigmas]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    xs = np.arange(len(sigmas))
    ax1.bar(xs, final_means, yerr=final_stds, capsize=5,
            color=[cmap(i / max(1, len(sigmas) - 1)) for i in range(len(sigmas))])
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"σ={s}" for s in sigmas])
    ax1.set_ylabel(f"Mean return (last {last_n} eps)")
    ax1.set_title("Asymptotic return vs σ")
    ax1.axhline(0, color="gray", linestyle=":", linewidth=0.7)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(xs, [s * 100 for s in final_succ],
            color=[cmap(i / max(1, len(sigmas) - 1)) for i in range(len(sigmas))])
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"σ={s}" for s in sigmas])
    ax2.set_ylabel(f"Success rate % (last {last_n} eps)")
    ax2.set_title("Asymptotic success rate vs σ")
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    bar_path = args.out_dir / f"sweep_sigma_bars_{stamp}.png"
    plt.savefig(bar_path, dpi=120)
    plt.close(fig)

    # --- Summary CSV ---
    csv_path = args.out_dir / f"sweep_sigma_{stamp}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma", "mean_return_last10pct",
                    "std_return_last10pct", "success_rate_last10pct"])
        for sigma, m, s, sr in zip(sigmas, final_means, final_stds, final_succ):
            w.writerow([sigma, m, s, sr])

    # --- Console summary ---
    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"\n{'sigma':>6} | {'mean ret':>10} | {'std':>7} | {'success':>9}")
    print("-" * 42)
    for sigma, m, s, sr in zip(sigmas, final_means, final_stds, final_succ):
        print(f"{sigma:>6.2f} | {m:>+10.2f} | {s:>7.2f} | {sr:>8.2%}")
    print(f"\nWrote:")
    print(f"  {curve_path}")
    print(f"  {bar_path}")
    print(f"  {csv_path}")


if __name__ == "__main__":
    main()
