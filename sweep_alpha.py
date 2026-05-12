"""Alpha sweep — what learning rate is best for SARSA on this grid?

For each alpha in the sweep, trains SARSA across N seeds for M episodes
at a fixed sigma. Produces:
  - Learning-curve plot, one line per alpha (mean +/- std).
  - Bar chart of final-episodes mean return + convergence speed per alpha.
  - Summary CSV.
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
    p = ArgumentParser(description="SARSA alpha-sweep hyperparameter study.")
    p.add_argument("GRID", type=Path)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--alphas", type=str, default="0.05,0.1,0.3,0.5,0.7",
                   help="Comma-separated list of alpha values to sweep.")
    p.add_argument("--start_pos", type=str, default=None)
    p.add_argument("--out_dir", type=Path, default=Path("results"))
    return p.parse_args()


def parse_start_pos(s):
    if s is None:
        return None
    a, b = s.split(",")
    return (int(a), int(b))


def train_sarsa_one_seed(grid_path, args, alpha, seed):
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
        n_actions=4, alpha=alpha, gamma=args.gamma,
        epsilon=args.epsilon, rng_seed=seed,
    )

    returns = np.zeros(args.episodes, dtype=float)
    steps = np.zeros(args.episodes, dtype=int)
    successes = np.zeros(args.episodes, dtype=int)

    for ep in range(args.episodes):
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
            next_action = agent.select_action(next_state, training=True)
            agent.learn(
                state=state, action=actual_action, reward=reward,
                next_state=next_state, next_action=next_action, done=terminated,
            )
            ep_return += reward
            ep_steps += 1
            state = next_state
            action = next_action
            if terminated:
                success = True
                break
        returns[ep] = ep_return
        steps[ep] = ep_steps
        successes[ep] = int(success)
    return returns, steps, successes


def smooth(x, k=10):
    if k <= 1:
        return x
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="valid")


def episodes_to_threshold(returns_per_ep, threshold):
    """Return earliest episode where smoothed return >= threshold, else -1."""
    s = smooth(returns_per_ep, max(1, len(returns_per_ep) // 50))
    above = np.where(s >= threshold)[0]
    return int(above[0]) if len(above) > 0 else -1


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d__%H-%M-%S")
    alphas = [float(a) for a in args.alphas.split(",")]

    print(
        f"Alpha sweep | grid={args.GRID.name} alphas={alphas} "
        f"sigma={args.sigma} seeds={args.seeds} episodes={args.episodes}"
    )

    all_returns = {}
    all_steps = {}
    all_successes = {}
    t0 = time.time()
    for alpha in alphas:
        ret = np.zeros((args.seeds, args.episodes))
        stp = np.zeros((args.seeds, args.episodes), dtype=int)
        suc = np.zeros((args.seeds, args.episodes), dtype=int)
        for seed in trange(args.seeds, desc=f"alpha={alpha}"):
            r, st, s = train_sarsa_one_seed(args.GRID, args, alpha, seed)
            ret[seed] = r
            stp[seed] = st
            suc[seed] = s
        all_returns[alpha] = ret
        all_steps[alpha] = stp
        all_successes[alpha] = suc
    elapsed = time.time() - t0

    # --- Plot 1: learning curves overlaid ---
    smooth_k = max(1, args.episodes // 50)
    eps = np.arange(args.episodes)
    eps_smooth = eps[smooth_k - 1:] if smooth_k > 1 else eps

    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("plasma")
    for i, alpha in enumerate(alphas):
        color = cmap(i / max(1, len(alphas) - 1))
        mean = all_returns[alpha].mean(axis=0)
        std = all_returns[alpha].std(axis=0)
        m_s = smooth(mean, smooth_k)
        s_s = smooth(std, smooth_k)
        ax.plot(eps_smooth, m_s, label=f"alpha={alpha}", color=color, linewidth=2)
        ax.fill_between(eps_smooth, m_s - s_s, m_s + s_s, color=color, alpha=0.13)

    ax.axhline(0, color="gray", linestyle=":", linewidth=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(
        f"SARSA learning curves vs alpha - {args.GRID.name}  |  "
        f"sigma={args.sigma}, gamma={args.gamma}, epsilon={args.epsilon}, "
        f"{args.seeds} seeds, smoothed (k={smooth_k})"
    )
    ax.legend(loc="lower right", title="Learning rate")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    curve_path = args.out_dir / f"sweep_alpha_curves_{stamp}.png"
    plt.savefig(curve_path, dpi=120)
    plt.close(fig)

    # --- Plot 2: bar chart of asymptotic return + convergence speed ---
    last_n = max(1, int(0.1 * args.episodes))
    final_means = [all_returns[a][:, -last_n:].mean() for a in alphas]
    final_stds = [all_returns[a][:, -last_n:].mean(axis=1).std() for a in alphas]
    # Convergence: episodes-to-threshold (return >= 0).
    conv_means = []
    conv_stds = []
    for a in alphas:
        per_seed = [episodes_to_threshold(all_returns[a][s], 0.0)
                    for s in range(args.seeds)]
        valid = [v for v in per_seed if v >= 0]
        if valid:
            conv_means.append(float(np.mean(valid)))
            conv_stds.append(float(np.std(valid)))
        else:
            conv_means.append(float("nan"))
            conv_stds.append(0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    xs = np.arange(len(alphas))
    colors = [cmap(i / max(1, len(alphas) - 1)) for i in range(len(alphas))]
    ax1.bar(xs, final_means, yerr=final_stds, capsize=5, color=colors)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"alpha={a}" for a in alphas])
    ax1.set_ylabel(f"Mean return (last {last_n} eps)")
    ax1.set_title("Asymptotic return vs alpha")
    ax1.axhline(0, color="gray", linestyle=":", linewidth=0.7)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(xs, conv_means, yerr=conv_stds, capsize=5, color=colors)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"alpha={a}" for a in alphas])
    ax2.set_ylabel("Episodes to return >= 0 (lower = faster)")
    ax2.set_title("Convergence speed vs alpha")
    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    bar_path = args.out_dir / f"sweep_alpha_bars_{stamp}.png"
    plt.savefig(bar_path, dpi=120)
    plt.close(fig)

    # --- Summary CSV ---
    csv_path = args.out_dir / f"sweep_alpha_{stamp}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "mean_return_last10pct", "std_return_last10pct",
                    "episodes_to_zero_return_mean", "episodes_to_zero_return_std",
                    "success_rate_last10pct"])
        for alpha, m, s, cm, cs in zip(alphas, final_means, final_stds,
                                       conv_means, conv_stds):
            sr = all_successes[alpha][:, -last_n:].mean()
            w.writerow([alpha, m, s, cm, cs, sr])

    # --- Console summary ---
    print(f"\nElapsed: {elapsed:.1f}s")
    header = f"{'alpha':>6} | {'mean ret':>10} | {'std':>7} | {'eps2>=0':>9} | {'success':>9}"
    print("\n" + header)
    print("-" * len(header))
    for alpha, m, s, cm in zip(alphas, final_means, final_stds, conv_means):
        sr = all_successes[alpha][:, -last_n:].mean()
        cm_str = f"{cm:>9.0f}" if cm == cm else "      n/a"  # nan check
        print(f"{alpha:>6.2f} | {m:>+10.2f} | {s:>7.2f} | {cm_str} | {sr:>8.2%}")
    print(f"\nWrote:")
    print(f"  {curve_path}")
    print(f"  {bar_path}")
    print(f"  {csv_path}")


if __name__ == "__main__":
    main()
