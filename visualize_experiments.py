"""Visualize the DQN/PPO hyperparameter sweep produced by run_experiments_dqn_ppo.sh.

Reads the per-run CSVs/PNGs that train_dqn.py and train_ppo.py write to
results/, and produces a small set of comparison figures in
results/figures/. Runs that are missing (not yet executed) are skipped
with a warning rather than failing the whole script.

Usage:
    python3 visualize_experiments.py
"""
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results")
FIG_DIR = RESULTS_DIR / "figures"

GRID = "A1_grid"
STATE_MODE = "gps"
SIGMA = 0.0
GAMMA = 0.99
HIDDEN = 256
SEEDS = [0, 42]

SWEEPS = {
    "dqn": [0.0001, 0.0005, 0.001],
    "ppo": [0.0001, 0.0003, 0.0005],
}
COLORS = {"dqn": "steelblue", "ppo": "darkorange"}


def run_name(algo: str, lr: float, seed: int) -> str:
    return (f"{algo}_{GRID}_{STATE_MODE}_seed{seed}_sigma{SIGMA}"
            f"_lr{lr}_g{GAMMA}_h{HIDDEN}_rangefull")


def load_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(v) for v in row] for row in reader]
    return header, np.array(rows)


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def load_runs(algo: str) -> dict:
    """Loads {lr: {"train": [arrays per seed], "eval": [arrays per seed]}}."""
    runs = {}
    for lr in SWEEPS[algo]:
        train_arrs, eval_arrs = [], []
        for seed in SEEDS:
            name = run_name(algo, lr, seed)
            train_path = RESULTS_DIR / f"{name}_training.csv"
            eval_path = RESULTS_DIR / f"{name}_eval.csv"
            if not train_path.exists() or not eval_path.exists():
                print(f"[skip] missing run: {name}")
                continue
            _, train_data = load_csv(train_path)
            _, eval_data = load_csv(eval_path)
            train_arrs.append(train_data)
            eval_arrs.append(eval_data)
        if train_arrs:
            runs[lr] = {"train": train_arrs, "eval": eval_arrs}
    return runs


def stack_min_len(arrs: list[np.ndarray]) -> np.ndarray:
    n = min(len(a) for a in arrs)
    return np.stack([a[:n] for a in arrs])


def plot_learning_curves(data: dict, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for ax, algo in zip(axes, ("dqn", "ppo")):
        for lr, runs in data[algo].items():
            rewards = stack_min_len([r[:, 1] for r in runs["train"]])
            mean_r = rewards.mean(axis=0)
            std_r = rewards.std(axis=0)
            window = max(1, len(mean_r) // 25)
            x = np.arange(window - 1, len(mean_r)) if window > 1 else np.arange(len(mean_r))
            mean_s = smooth(mean_r, window)
            std_s = smooth(std_r, window)
            ax.plot(x, mean_s, label=f"lr={lr}")
            ax.fill_between(x, mean_s - std_s, mean_s + std_s, alpha=0.15)
        ax.set_title(f"{algo.upper()} — Reward per Episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_eval_success_rate(data: dict, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, algo in zip(axes, ("dqn", "ppo")):
        for lr, runs in data[algo].items():
            evals = stack_min_len(runs["eval"])
            episodes = evals[0, :, 0]
            success = evals[:, :, 1]
            mean_s = success.mean(axis=0)
            std_s = success.std(axis=0)
            ax.plot(episodes, mean_s, marker="o", label=f"lr={lr}")
            ax.fill_between(episodes, mean_s - std_s, mean_s + std_s, alpha=0.15)
        ax.set_title(f"{algo.upper()} — Greedy Eval Success Rate")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Success Rate")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_final_performance_bars(data: dict, save_path: Path) -> None:
    labels, success_means, reward_means, colors = [], [], [], []
    for algo in ("dqn", "ppo"):
        for lr, runs in data[algo].items():
            evals = stack_min_len(runs["eval"])
            final = evals[:, -1, :]
            labels.append(f"{algo.upper()}\nlr={lr}")
            success_means.append(final[:, 1].mean())
            reward_means.append(final[:, 2].mean())
            colors.append(COLORS[algo])

    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.bar(x, success_means, color=colors)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("Final Success Rate")
    ax1.set_title("Final Eval Success Rate by Config")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.bar(x, reward_means, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Final Mean Reward")
    ax2.set_title("Final Eval Mean Reward by Config")
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def best_config(data: dict, algo: str) -> tuple[float, int]:
    """Returns (lr, seed) of the single run with the highest final
    success rate (tie-broken by mean reward)."""
    best = None
    for lr, runs in data[algo].items():
        for seed, eval_arr in zip(SEEDS, runs["eval"]):
            success_rate, mean_reward = eval_arr[-1, 1], eval_arr[-1, 2]
            key = (success_rate, mean_reward)
            if best is None or key > best[0]:
                best = (key, lr, seed)
    _, lr, seed = best
    return lr, seed


def plot_best_run_paths(data: dict, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, algo in zip(axes, ("dqn", "ppo")):
        lr, seed = best_config(data, algo)
        name = run_name(algo, lr, seed)
        img_path = RESULTS_DIR / f"{name}_path.png"
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Best {algo.upper()} run\nlr={lr}, seed={seed}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_head_to_head_best(data: dict, save_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for algo in ("dqn", "ppo"):
        lr, seed = best_config(data, algo)
        runs = data[algo][lr]
        seed_idx = SEEDS.index(seed)
        train = runs["train"][seed_idx]
        eval_arr = runs["eval"][seed_idx]

        rewards = train[:, 1]
        window = max(1, len(rewards) // 25)
        smoothed = smooth(rewards, window)
        x = np.arange(window - 1, len(rewards)) if window > 1 else np.arange(len(rewards))
        ax1.plot(x, smoothed, color=COLORS[algo], label=f"{algo.upper()} (lr={lr}, seed={seed})")

        ax2.plot(eval_arr[:, 0], eval_arr[:, 1], marker="o", color=COLORS[algo],
                 label=f"{algo.upper()} (lr={lr}, seed={seed})")

    ax1.set_title("Best Run — Reward per Episode")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Cumulative Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Best Run — Greedy Eval Success Rate")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Success Rate")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    data = {algo: load_runs(algo) for algo in ("dqn", "ppo")}
    for algo in ("dqn", "ppo"):
        if not data[algo]:
            print(f"No completed {algo.upper()} runs found in {RESULTS_DIR}/. "
                  f"Run run_experiments_dqn_ppo.sh first.")
            return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_learning_curves(data, FIG_DIR / "learning_curves.png")
    plot_eval_success_rate(data, FIG_DIR / "eval_success_rate.png")
    plot_final_performance_bars(data, FIG_DIR / "final_performance_bars.png")
    plot_best_run_paths(data, FIG_DIR / "best_run_paths.png")
    plot_head_to_head_best(data, FIG_DIR / "head_to_head_best.png")


if __name__ == "__main__":
    main()
