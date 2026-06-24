"""Aggregated figures for the report (mean +/- std over seeds).

Fig 1 (convergence): 2x2 grid; columns DQN | PPO, rows episode reward |success rate. 
Each curve is the mean over seeds with a +/-1 std shaded band,
read from the per-seed results/*_rewards.npy and *_successes.npy arrays.

Usage:
    python plot_results.py            # writes results/fig1_convergence.png and saves also the single panels as separate assets
"""
import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("results")

# Config per method shown in the convergence figure: (algo, sigma, lr, hidden).
# hidden filters out stray runs of a different network size (None = any).
DQN_CFG = ("dqn", "0.1", "0.0005", "256")
PPO_CFG = ("ppo", "0.1", "0.0005", "256")
DQN_SMOOTH = 20
PPO_SMOOTH = 50
DQN_COLOR = "#1f77b4"
PPO_COLOR = "#d62728"

# How each state mode is labelled in titles.
MODE_LABEL = {"gps": "GPS", "raycasting": "raycasting", "both": "GPS+raycasting"}


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    """Trailing moving average, length-preserving (front padded with the
    progressive mean so early episodes aren't dropped)."""
    x = np.asarray(x, dtype=float)
    if w <= 1 or len(x) < w:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    body = (c[w:] - c[:-w]) / w
    head = np.array([x[: i + 1].mean() for i in range(w - 1)])
    return np.concatenate([head, body])


def load_runs(algo, state_mode, sigma, lr, hidden=None):
    """Stack per-seed reward and success arrays for one config. Returns
    (rewards[n_seeds, T], successes[n_seeds, T], seeds[list]).

    state_mode ('gps' | 'raycasting' | 'both') is matched literally in the
    filename, so only runs of that representation are loaded."""
    pat = f"{algo}_*_{state_mode}_seed*_sigma{sigma}_lr{lr}_*_rewards.npy"
    rewards, succ, seeds = [], [], []
    for rf in sorted(glob.glob(str(RESULTS / pat))):
        base = os.path.basename(rf)
        if hidden and f"_h{hidden}_" not in base:
            continue
        sf = rf.replace("_rewards.npy", "_successes.npy")
        if not os.path.exists(sf):
            continue
        seed = int(re.search(r"_seed(\d+)_", base).group(1))
        if seed in seeds:           # guard against duplicate runs of same seed
            continue
        seeds.append(seed)
        rewards.append(np.load(rf))
        succ.append(np.load(sf))
    if not rewards:
        raise FileNotFoundError(f"no runs for {algo} sigma{sigma} lr{lr} h{hidden}")
    L = min(len(r) for r in rewards)        # align to shortest seed
    R = np.stack([r[:L] for r in rewards])
    S = np.stack([s[:L] for s in succ])
    return R, S, sorted(seeds)


def agg_curve(ax, data, w, color, label, ylabel):
    """Smooth each seed, then plot mean +/- std across seeds vs episode."""
    sm = np.stack([smooth(d, w) for d in data])
    m, sd = sm.mean(0), sm.std(0)
    ep = np.arange(1, len(m) + 1)
    ax.plot(ep, m, color=color, lw=1.6, label=label)
    ax.fill_between(ep, m - sd, m + sd, color=color, alpha=0.18)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def fig1_convergence(dqn, ppo, state_mode):
    """Compact 2x2 figure for the report: columns DQN | PPO, rows reward |
    success rate."""
    (dqn_r, dqn_s, dqn_seeds) = dqn
    (ppo_r, ppo_s, ppo_seeds) = ppo

    fig, ax = plt.subplots(2, 2, figsize=(10, 6))
    sig = DQN_CFG[1]
    fig.suptitle(f"Convergence on A1_grid ({MODE_LABEL[state_mode]}, "
                 f"$\\sigma$={sig}), mean $\\pm$ std over {len(dqn_seeds)} seeds",
                 fontsize=12)

    agg_curve(ax[0, 0], dqn_r, DQN_SMOOTH, DQN_COLOR, "DQN", "Episode reward")
    agg_curve(ax[0, 1], ppo_r, PPO_SMOOTH, PPO_COLOR, "PPO", "Episode reward")
    agg_curve(ax[1, 0], dqn_s, DQN_SMOOTH, DQN_COLOR, "DQN", "Success rate")
    agg_curve(ax[1, 1], ppo_s, PPO_SMOOTH, PPO_COLOR, "PPO", "Success rate")
    ax[0, 0].set_title("DQN (baseline)")
    ax[0, 1].set_title("PPO (main method)")
    for a in ax[1, :]:
        a.set_ylim(-0.02, 1.02)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    suffix = "" if state_mode == "gps" else f"_{state_mode}"
    out = RESULTS / f"fig1_convergence{suffix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def single_panels(dqn, ppo, state_mode):
    """Standalone version of each panel — kept as separate assets (slides,
    space-tight layouts). Not all of these go in the report."""
    (dqn_r, dqn_s, _) = dqn
    (ppo_r, ppo_s, _) = ppo
    suffix = "" if state_mode == "gps" else f"_{state_mode}"
    panels = [
        ("dqn_reward",  dqn_r, DQN_SMOOTH, DQN_COLOR, "DQN",  "Episode reward"),
        ("dqn_success", dqn_s, DQN_SMOOTH, DQN_COLOR, "DQN",  "Success rate"),
        ("ppo_reward",  ppo_r, PPO_SMOOTH, PPO_COLOR, "PPO",  "Episode reward"),
        ("ppo_success", ppo_s, PPO_SMOOTH, PPO_COLOR, "PPO",  "Success rate"),
    ]
    for name, data, w, color, label, ylabel in panels:
        fig, a = plt.subplots(figsize=(5, 3.2))
        agg_curve(a, data, w, color, label, ylabel)
        a.set_title(f"{label} (A1_grid, {MODE_LABEL[state_mode]}, "
                    f"$\\sigma$={DQN_CFG[1]})")
        if "success" in name:
            a.set_ylim(-0.02, 1.02)
        fig.tight_layout()
        out = RESULTS / f"fig1_{name}{suffix}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Aggregated convergence figures.")
    ap.add_argument("--state_mode", choices=["gps", "raycasting", "both"],
                    default="gps", help="Which state representation to plot.")
    args = ap.parse_args()

    sm = args.state_mode
    dqn = load_runs(DQN_CFG[0], sm, DQN_CFG[1], DQN_CFG[2], DQN_CFG[3])
    ppo = load_runs(PPO_CFG[0], sm, PPO_CFG[1], PPO_CFG[2], PPO_CFG[3])
    print(f"[{sm}] DQN: {len(dqn[2])} seeds {dqn[2]} | final success {dqn[1][:, -50:].mean():.3f}")
    print(f"[{sm}] PPO: {len(ppo[2])} seeds {ppo[2]} | final success {ppo[1][:, -50:].mean():.3f}")
    fig1_convergence(dqn, ppo, sm)
    single_panels(dqn, ppo, sm)
