"""Multi-seed performance experiment for SARSA vs RandomAgent.

For each seed:
  - Train SARSA for N episodes; log per-episode return.
  - Run RandomAgent in the same episode shape; log per-episode return.

Then aggregate (mean ± std across seeds) and produce a comparison plot.
"""
from argparse import ArgumentParser
from pathlib import Path
import csv
import random
import time

import matplotlib
matplotlib.use("Agg")  # no display required
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from world import Environment
from agents.sarsa_agent import SARSAAgent
from agents.random_agent import RandomAgent


def parse_args():
    p = ArgumentParser(description="SARSA vs RandomAgent multi-seed experiment.")
    p.add_argument("GRID", type=Path,
                   help="Path to the grid file.")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--epsilon_end", type=float, default=None,
                   help="Final epsilon for linear decay. If unset, no decay.")
    p.add_argument("--epsilon_decay_episodes", type=int, default=1,
                   help="Episodes over which to linearly decay epsilon.")
    p.add_argument("--seeds", type=int, default=5,
                   help="Number of random seeds (run for each agent).")
    p.add_argument("--start_pos", type=str, default=None,
                   help="Agent start as 'row,col'. If unset, env picks one.")
    p.add_argument("--random_start", action="store_true",
                   help="If set, do NOT lock start position; env re-randomizes "
                        "each episode reset. Tests policy generalization.")
    p.add_argument("--out_dir", type=Path, default=Path("results"))
    return p.parse_args()


def parse_start_pos(s):
    if s is None:
        return None
    a, b = s.split(",")
    return (int(a), int(b))


def run_one_episode(env, agent, max_steps, training, is_sarsa):
    """Run a single episode. Returns (return, steps, success)."""
    state = env.agent_pos
    if is_sarsa:
        action = agent.select_action(state, training=training)
    else:
        action = agent.take_action(state)

    ep_return = 0.0
    ep_steps = 0
    success = False

    for _ in range(max_steps):
        next_state, reward, terminated, info = env.step(action)
        actual_action = info["actual_action"]

        if is_sarsa:
            next_action = agent.select_action(next_state, training=training)
            if training:
                agent.learn(
                    state=state,
                    action=actual_action,
                    reward=reward,
                    next_state=next_state,
                    next_action=next_action,
                    done=terminated,
                )
        else:
            next_action = agent.take_action(next_state)

        ep_return += reward
        ep_steps += 1
        state = next_state
        action = next_action
        if terminated:
            success = True
            break

    return ep_return, ep_steps, success


def run_agent_across_seeds(grid_path, agent_factory, args, label, is_sarsa):
    """Train (or run) an agent for `args.seeds` seeds × `args.episodes` episodes.

    Returns:
      returns  — ndarray (seeds, episodes) of per-episode returns
      steps    — ndarray (seeds, episodes) of per-episode step counts
      success  — ndarray (seeds, episodes) of 0/1 success flags
    """
    start_pos = parse_start_pos(args.start_pos)
    returns = np.zeros((args.seeds, args.episodes), dtype=float)
    steps = np.zeros((args.seeds, args.episodes), dtype=int)
    success = np.zeros((args.seeds, args.episodes), dtype=int)

    for seed_idx in range(args.seeds):
        seed = seed_idx  # 0..seeds-1
        random.seed(seed)
        np.random.seed(seed)

        env = Environment(
            grid_fp=grid_path,
            no_gui=True,
            sigma=args.sigma,
            target_fps=-1,
            agent_start_pos=start_pos,
            random_seed=seed,
        )
        initial_pos = env.reset()
        if not getattr(args, "random_start", False):
            env.agent_start_pos = initial_pos  # lock start

        agent = agent_factory(seed)

        for ep in trange(args.episodes, desc=f"{label} seed={seed}", leave=False):
            if ep > 0:
                env.reset()
            # Drive epsilon-decay in SARSA agents that support it.
            if is_sarsa and hasattr(agent, "start_episode"):
                agent.start_episode()
            r, s, ok = run_one_episode(
                env, agent, args.max_steps,
                training=True, is_sarsa=is_sarsa,
            )
            returns[seed_idx, ep] = r
            steps[seed_idx, ep] = s
            success[seed_idx, ep] = int(ok)

    return returns, steps, success


def smooth(x, k=20):
    """Trailing moving average for plotting smoother curves."""
    if k <= 1:
        return x
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="valid")


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d__%H-%M-%S")

    print(
        f"Experiment | grid={args.GRID.name} episodes={args.episodes} "
        f"max_steps={args.max_steps} sigma={args.sigma} "
        f"alpha={args.alpha} gamma={args.gamma} epsilon={args.epsilon} "
        f"seeds={args.seeds}"
    )

    sarsa_factory = lambda seed: SARSAAgent(
        n_actions=4,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_end=args.epsilon_end,
        epsilon_decay_episodes=args.epsilon_decay_episodes,
        rng_seed=seed,
    )
    random_factory = lambda seed: RandomAgent()

    t0 = time.time()
    sarsa_returns, sarsa_steps, sarsa_success = run_agent_across_seeds(
        args.GRID, sarsa_factory, args, "SARSA", is_sarsa=True,
    )
    random_returns, random_steps, random_success = run_agent_across_seeds(
        args.GRID, random_factory, args, "Random", is_sarsa=False,
    )
    elapsed = time.time() - t0

    # ---- Aggregate ----
    sarsa_mean = sarsa_returns.mean(axis=0)
    sarsa_std = sarsa_returns.std(axis=0)
    random_mean = random_returns.mean(axis=0)
    random_std = random_returns.std(axis=0)
    sarsa_steps_mean = sarsa_steps.mean(axis=0)
    sarsa_steps_std = sarsa_steps.std(axis=0)
    random_steps_mean = random_steps.mean(axis=0)
    random_steps_std = random_steps.std(axis=0)

    # ---- Save raw CSV ----
    csv_path = args.out_dir / f"experiment_sarsa_vs_random_{stamp}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode",
                    "sarsa_return_mean", "sarsa_return_std",
                    "random_return_mean", "random_return_std",
                    "sarsa_steps_mean", "sarsa_steps_std",
                    "random_steps_mean", "random_steps_std",
                    "sarsa_success_rate", "random_success_rate"])
        for ep in range(args.episodes):
            w.writerow([ep,
                        sarsa_mean[ep], sarsa_std[ep],
                        random_mean[ep], random_std[ep],
                        sarsa_steps_mean[ep], sarsa_steps_std[ep],
                        random_steps_mean[ep], random_steps_std[ep],
                        sarsa_success[:, ep].mean(),
                        random_success[:, ep].mean()])

    # ---- Plot (two-panel: returns + episode length) ----
    smooth_k = max(1, args.episodes // 50)  # ~50 plot points
    eps_axis = np.arange(args.episodes)
    eps_axis_smooth = eps_axis[smooth_k - 1:] if smooth_k > 1 else eps_axis

    decay_str = (
        f", eps_end={args.epsilon_end}, decay_eps={args.epsilon_decay_episodes}"
        if args.epsilon_end is not None and args.epsilon_end != args.epsilon
        else ""
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    # Panel 1 — returns
    s_ret_m = smooth(sarsa_mean, smooth_k)
    s_ret_s = smooth(sarsa_std, smooth_k)
    r_ret_m = smooth(random_mean, smooth_k)
    r_ret_s = smooth(random_std, smooth_k)
    ax1.plot(eps_axis_smooth, s_ret_m, label="SARSA", color="tab:blue", linewidth=2)
    ax1.fill_between(eps_axis_smooth, s_ret_m - s_ret_s, s_ret_m + s_ret_s,
                     color="tab:blue", alpha=0.18)
    ax1.plot(eps_axis_smooth, r_ret_m, label="Random", color="tab:orange", linewidth=2)
    ax1.fill_between(eps_axis_smooth, r_ret_m - r_ret_s, r_ret_m + r_ret_s,
                     color="tab:orange", alpha=0.18)
    ax1.axhline(0, color="gray", linestyle=":", linewidth=0.7)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Return")
    ax1.set_title("Return per episode (mean +/- 1 std)")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Panel 2 — episode length (steps)
    s_st_m = smooth(sarsa_steps_mean, smooth_k)
    s_st_s = smooth(sarsa_steps_std, smooth_k)
    r_st_m = smooth(random_steps_mean, smooth_k)
    r_st_s = smooth(random_steps_std, smooth_k)
    ax2.plot(eps_axis_smooth, s_st_m, label="SARSA", color="tab:blue", linewidth=2)
    ax2.fill_between(eps_axis_smooth, s_st_m - s_st_s, s_st_m + s_st_s,
                     color="tab:blue", alpha=0.18)
    ax2.plot(eps_axis_smooth, r_st_m, label="Random", color="tab:orange", linewidth=2)
    ax2.fill_between(eps_axis_smooth, r_st_m - r_st_s, r_st_m + r_st_s,
                     color="tab:orange", alpha=0.18)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps to terminate or hit cap")
    ax2.set_title("Episode length (mean +/- 1 std)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"SARSA vs Random — {args.GRID.name}  |  "
        f"sigma={args.sigma}, alpha={args.alpha}, gamma={args.gamma}, "
        f"eps={args.epsilon}{decay_str}  |  "
        f"{args.seeds} seeds, smoothed (k={smooth_k})",
        fontsize=11,
    )
    plt.tight_layout()

    plot_path = args.out_dir / f"experiment_sarsa_vs_random_{stamp}.png"
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)

    # ---- Summary ----
    last10 = max(1, int(0.1 * args.episodes))
    sarsa_tail = sarsa_returns[:, -last10:].mean()
    random_tail = random_returns[:, -last10:].mean()
    sarsa_succ_tail = sarsa_success[:, -last10:].mean()
    random_succ_tail = random_success[:, -last10:].mean()

    sarsa_steps_tail = sarsa_steps[:, -last10:].mean()
    random_steps_tail = random_steps[:, -last10:].mean()

    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"\n--- Last {last10} episodes (per-seed mean across {args.seeds} seeds) ---")
    print(f"  SARSA  | mean return = {sarsa_tail:+7.2f}   "
          f"mean steps = {sarsa_steps_tail:6.1f}   success rate = {sarsa_succ_tail:.2%}")
    print(f"  Random | mean return = {random_tail:+7.2f}   "
          f"mean steps = {random_steps_tail:6.1f}   success rate = {random_succ_tail:.2%}")
    print(f"  Improvement over Random: {sarsa_tail - random_tail:+.2f} per episode")
    print(f"\nWrote:")
    print(f"  {csv_path}")
    print(f"  {plot_path}")


if __name__ == "__main__":
    main()
