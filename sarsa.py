"""Tabular TD training and experiments for 2AMC15 DIC.

Single entry point with subcommands. Dispatches to SARSA, Q-learning, or
SARSA(lambda) via the `algo` field. Replaces the previous 10 scripts:
  train_sarsa, experiment_sarsa, sweep_{alpha,gamma,sigma,epsilon},
  transfer_test, push_best, multi_grid_eval, rescue_super_hard.

Subcommands:
  train         Train SARSA on a grid; per-episode CSV.
  compare       Multi-seed SARSA vs Random with mean+/-std plots.
  sweep         Numeric hyperparameter sweep (alpha/gamma/sigma).
  sweep-eps     Epsilon-schedule sweep (fixed vs decay).
  transfer      Train on grid A, evaluate greedily on grid B.
  eval-configs  Train each of a list of configs; greedy eval at sigma=0 and 0.1.

Run `python sarsa.py <cmd> --help` for per-subcommand options.
Use run_experiments.sh to reproduce all results in REPORT_SARSA.md.
"""
from argparse import ArgumentParser
from pathlib import Path
import csv
import json
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
from agents.qlearning_agent import QLearningAgent
from agents.sarsa_lambda_agent import SARSALambdaAgent
from agents.linear_sarsa_agent import LinearSARSAAgent
from agents.random_agent import RandomAgent

AGENT_CLASSES = {
    "sarsa": SARSAAgent,
    "qlearning": QLearningAgent,
    "sarsa-lambda": SARSALambdaAgent,
    "linear-sarsa": LinearSARSAAgent,
}


# ====================================================================
# Core helpers
# ====================================================================

def parse_start_pos(s):
    if s is None:
        return None
    a, b = s.split(",")
    return (int(a), int(b))


def parse_values(s, cast=float):
    return [cast(v) for v in s.split(",")]


def lock_start(grid_path, seed=0):
    """Return the env's default start position for a given seed."""
    env = Environment(grid_fp=grid_path, no_gui=True, sigma=0.0,
                      target_fps=-1, agent_start_pos=None, random_seed=seed)
    return env.reset()


def make_env(grid_path, sigma, seed, start_pos):
    return Environment(
        grid_fp=grid_path, no_gui=True, sigma=sigma, target_fps=-1,
        agent_start_pos=start_pos, random_seed=seed,
    )


def make_agent(*, algo="sarsa", alpha, gamma, epsilon, epsilon_end=None,
               epsilon_decay_episodes=1, lambda_=None, q_init=None, seed=0):
    cls = AGENT_CLASSES[algo]
    kwargs = dict(n_actions=4, alpha=alpha, gamma=gamma,
                  epsilon=epsilon, epsilon_end=epsilon_end,
                  epsilon_decay_episodes=epsilon_decay_episodes,
                  q_init=q_init, rng_seed=seed)
    if algo == "sarsa-lambda":
        kwargs["lambda_"] = lambda_ if lambda_ is not None else 0.9
    return cls(**kwargs)


def train_agent(grid_path, start, *, algo="sarsa", alpha, gamma, epsilon,
                epsilon_end=None, epsilon_decay_episodes=1, lambda_=None,
                sigma, episodes, max_steps, q_init=None,
                random_start=False, seed=0, track_per_episode=True,
                desc=None):
    """Train one of {sarsa, qlearning, sarsa-lambda}. Returns
    (agent, returns, steps, successes) arrays of length `episodes` when
    track_per_episode=True, otherwise (agent, None, None, None).
    """
    random.seed(seed); np.random.seed(seed)
    env_start = None if random_start else start
    env = make_env(grid_path, sigma, seed, env_start)
    env.reset()
    if not random_start:
        env.agent_start_pos = start

    agent = make_agent(algo=algo, alpha=alpha, gamma=gamma, epsilon=epsilon,
                       epsilon_end=epsilon_end,
                       epsilon_decay_episodes=epsilon_decay_episodes,
                       lambda_=lambda_,
                       q_init=q_init, seed=seed)

    returns = np.zeros(episodes) if track_per_episode else None
    steps = np.zeros(episodes, dtype=int) if track_per_episode else None
    successes = np.zeros(episodes, dtype=int) if track_per_episode else None

    iterator = trange(episodes, desc=desc, leave=False) if desc else range(episodes)
    for ep in iterator:
        if ep > 0:
            env.reset()
        agent.start_episode()
        # Function-approximation agents need the grid context (target
        # position, obstacle map) refreshed each episode. Tabular agents
        # don't implement set_context — duck-typed.
        if hasattr(agent, "set_context"):
            agent.set_context(env.grid)
        state = env.agent_pos
        action = agent.select_action(state, training=True)
        ep_return = 0.0
        ep_steps = 0
        success = False
        for _ in range(max_steps):
            ns, r, term, info = env.step(action)
            na = agent.select_action(ns, training=True)
            agent.learn(state=state, action=info["actual_action"], reward=r,
                        next_state=ns, next_action=na, done=term)
            ep_return += r
            ep_steps += 1
            state, action = ns, na
            if term:
                success = True
                break
        if track_per_episode:
            returns[ep] = ep_return
            steps[ep] = ep_steps
            successes[ep] = int(success)
    return agent, returns, steps, successes


def run_random(grid_path, start, *, sigma, episodes, max_steps, seed=0, desc=None):
    """Run RandomAgent for `episodes` episodes (no learning). Returns same shape as train_agent."""
    random.seed(seed); np.random.seed(seed)
    env = make_env(grid_path, sigma, seed, start)
    env.reset()
    agent = RandomAgent()
    returns = np.zeros(episodes)
    steps = np.zeros(episodes, dtype=int)
    successes = np.zeros(episodes, dtype=int)
    iterator = trange(episodes, desc=desc, leave=False) if desc else range(episodes)
    for ep in iterator:
        if ep > 0:
            env.reset()
        state = env.agent_pos
        ep_return = 0.0
        ep_steps = 0
        success = False
        for _ in range(max_steps):
            action = agent.take_action(state)
            ns, r, term, _ = env.step(action)
            ep_return += r
            ep_steps += 1
            state = ns
            if term:
                success = True
                break
        returns[ep] = ep_return
        steps[ep] = ep_steps
        successes[ep] = int(success)
    return None, returns, steps, successes


def eval_greedy(grid_path, agent, start, *, sigma, max_steps, seed=0):
    """Run one greedy episode. Returns (steps, failed, cum_reward, reached)."""
    random.seed(seed + 10_000); np.random.seed(seed + 10_000)
    env = make_env(grid_path, sigma, seed + 10_000, start)
    state = env.reset()
    # Function-approximation agents need the grid context for the
    # *evaluation* grid (which may differ from the training grid).
    if hasattr(agent, "set_context"):
        agent.set_context(env.grid)
    for _ in range(max_steps):
        action = agent.take_action(state)
        state, _, term, _ = env.step(action)
        if term:
            break
    ws = env.world_stats
    return (ws["total_steps"], ws["total_failed_moves"],
            ws["cumulative_reward"], int(ws["total_targets_reached"] > 0))


def smooth(x, k=10):
    if k <= 1:
        return x
    return np.convolve(x, np.ones(k) / k, mode="valid")


def out_path(out_dir, prefix, stamp, ext):
    return out_dir / f"{prefix}_{stamp}.{ext}"


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_meta(csv_path, meta: dict):
    """Write a sidecar JSON next to a CSV recording the hyperparameters
    used to produce it. Path is the CSV path with `.csv` replaced by
    `.meta.json`. Values that aren't JSON-serializable are stringified.
    """
    def jsonable(v):
        if isinstance(v, Path):
            return str(v)
        try:
            json.dumps(v)
            return v
        except TypeError:
            return str(v)
    out = {k: jsonable(v) for k, v in meta.items()}
    meta_path = Path(str(csv_path).removesuffix(".csv") + ".meta.json")
    meta_path.write_text(json.dumps(out, indent=2))


# ====================================================================
# Subcommand: train
# ====================================================================

def cmd_train(args):
    stamp = time.strftime("%Y-%m-%d__%H-%M-%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for grid_path in args.GRID:
        start = parse_start_pos(args.start_pos) or lock_start(grid_path, args.seed)
        t0 = time.time()
        agent, returns, steps, succ = train_agent(
            grid_path, start,
            alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon,
            epsilon_end=args.epsilon_end,
            epsilon_decay_episodes=args.epsilon_decay_episodes,
            sigma=args.sigma, episodes=args.episodes, max_steps=args.max_steps,
            q_init=args.q_init, seed=args.seed,
            desc=f"train {grid_path.stem}",
        )
        elapsed = time.time() - t0
        rows = [{"episode": i, "return": float(returns[i]),
                 "steps": int(steps[i]), "success": int(succ[i])}
                for i in range(args.episodes)]
        path = out_path(args.out_dir, f"train_{grid_path.stem}", stamp, "csv")
        write_csv(path, ["episode", "return", "steps", "success"], rows)
        write_meta(path, {
            "cmd": "train", "grid": grid_path, "start_pos": start,
            "algo": "sarsa", "alpha": args.alpha, "gamma": args.gamma,
            "epsilon": args.epsilon, "epsilon_end": args.epsilon_end,
            "epsilon_decay_episodes": args.epsilon_decay_episodes,
            "sigma": args.sigma, "episodes": args.episodes,
            "max_steps": args.max_steps, "q_init": args.q_init,
            "seed": args.seed, "elapsed_sec": round(elapsed, 1),
        })
        tail = max(1, int(0.1 * args.episodes))
        print(f"[{grid_path.stem}] {elapsed:.1f}s  "
              f"asymptotic return={returns[-tail:].mean():+.2f}  "
              f"success_rate_last10pct={succ[-tail:].mean():.2%}  "
              f"-> {path}")


# ====================================================================
# Subcommand: compare (SARSA vs Random)
# ====================================================================

def cmd_compare(args):
    stamp = time.strftime("%Y-%m-%d__%H-%M-%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    grid_path = args.GRID
    start = parse_start_pos(args.start_pos) or lock_start(grid_path, 0)

    def collect(agent_kind):
        ret = np.zeros((args.seeds, args.episodes))
        stp = np.zeros((args.seeds, args.episodes), dtype=int)
        suc = np.zeros((args.seeds, args.episodes), dtype=int)
        for seed in range(args.seeds):
            kw = dict(sigma=args.sigma, episodes=args.episodes,
                      max_steps=args.max_steps, seed=seed,
                      desc=f"{agent_kind} seed={seed}")
            if agent_kind == "SARSA":
                _, r, s, c = train_agent(
                    grid_path, start, alpha=args.alpha, gamma=args.gamma,
                    epsilon=args.epsilon, epsilon_end=args.epsilon_end,
                    epsilon_decay_episodes=args.epsilon_decay_episodes,
                    random_start=args.random_start, **kw)
            else:
                _, r, s, c = run_random(grid_path, start, **kw)
            ret[seed], stp[seed], suc[seed] = r, s, c
        return ret, stp, suc

    t0 = time.time()
    s_ret, s_stp, s_suc = collect("SARSA")
    r_ret, r_stp, r_suc = collect("Random")
    elapsed = time.time() - t0

    # CSV
    csv_path = out_path(args.out_dir, "compare_sarsa_vs_random", stamp, "csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "sarsa_return_mean", "sarsa_return_std",
                    "random_return_mean", "random_return_std",
                    "sarsa_steps_mean", "sarsa_steps_std",
                    "random_steps_mean", "random_steps_std",
                    "sarsa_success_rate", "random_success_rate"])
        for ep in range(args.episodes):
            w.writerow([ep, s_ret[:, ep].mean(), s_ret[:, ep].std(),
                        r_ret[:, ep].mean(), r_ret[:, ep].std(),
                        s_stp[:, ep].mean(), s_stp[:, ep].std(),
                        r_stp[:, ep].mean(), r_stp[:, ep].std(),
                        s_suc[:, ep].mean(), r_suc[:, ep].mean()])

    # Two-panel plot
    k = max(1, args.episodes // 50)
    x = np.arange(args.episodes)
    xs = x[k - 1:] if k > 1 else x
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.5))
    for ret, label, color in [(s_ret, "SARSA", "tab:blue"),
                              (r_ret, "Random", "tab:orange")]:
        m, sd = smooth(ret.mean(axis=0), k), smooth(ret.std(axis=0), k)
        a1.plot(xs, m, label=label, color=color, linewidth=2)
        a1.fill_between(xs, m - sd, m + sd, color=color, alpha=0.18)
    a1.axhline(0, color="gray", linestyle=":", linewidth=0.7)
    a1.set(xlabel="Episode", ylabel="Return",
           title="Return per episode (mean +/- 1 std)")
    a1.legend(loc="lower right"); a1.grid(True, alpha=0.3)

    for stp, label, color in [(s_stp, "SARSA", "tab:blue"),
                              (r_stp, "Random", "tab:orange")]:
        m, sd = smooth(stp.mean(axis=0), k), smooth(stp.std(axis=0), k)
        a2.plot(xs, m, label=label, color=color, linewidth=2)
        a2.fill_between(xs, m - sd, m + sd, color=color, alpha=0.18)
    a2.set(xlabel="Episode", ylabel="Steps",
           title="Episode length (mean +/- 1 std)")
    a2.legend(loc="upper right"); a2.grid(True, alpha=0.3)
    fig.suptitle(f"SARSA vs Random - {grid_path.name} | sigma={args.sigma}, "
                 f"alpha={args.alpha}, gamma={args.gamma}, eps={args.epsilon} | "
                 f"{args.seeds} seeds, smoothed (k={k})", fontsize=11)
    plt.tight_layout()
    plot_path = out_path(args.out_dir, "compare_sarsa_vs_random", stamp, "png")
    plt.savefig(plot_path, dpi=120); plt.close(fig)

    write_meta(csv_path, {
        "cmd": "compare", "grid": grid_path, "start_pos": start,
        "alpha": args.alpha, "gamma": args.gamma, "epsilon": args.epsilon,
        "epsilon_end": args.epsilon_end,
        "epsilon_decay_episodes": args.epsilon_decay_episodes,
        "sigma": args.sigma, "episodes": args.episodes,
        "max_steps": args.max_steps, "seeds": args.seeds,
        "random_start": args.random_start,
        "plot": plot_path, "elapsed_sec": round(elapsed, 1),
    })

    tail = max(1, int(0.1 * args.episodes))
    print(f"Elapsed {elapsed:.1f}s")
    print(f"  SARSA  asymptotic return = {s_ret[:, -tail:].mean():+.2f}  "
          f"success = {s_suc[:, -tail:].mean():.2%}")
    print(f"  Random asymptotic return = {r_ret[:, -tail:].mean():+.2f}  "
          f"success = {r_suc[:, -tail:].mean():.2%}")
    print(f"  -> {csv_path}\n  -> {plot_path}")


# ====================================================================
# Subcommand: sweep (alpha/gamma/sigma)
# ====================================================================

def cmd_sweep(args):
    stamp = time.strftime("%Y-%m-%d__%H-%M-%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    grid_path = args.GRID
    start = parse_start_pos(args.start_pos) or lock_start(grid_path, 0)
    values = parse_values(args.values)
    param = args.param  # alpha | gamma | sigma
    print(f"Sweeping {param} over {values} on {grid_path.name}, "
          f"{args.seeds} seeds, {args.episodes} eps")

    all_returns = {}
    all_success = {}
    t0 = time.time()
    for v in values:
        ret = np.zeros((args.seeds, args.episodes))
        suc = np.zeros((args.seeds, args.episodes), dtype=int)
        for seed in trange(args.seeds, desc=f"{param}={v}"):
            kwargs = dict(alpha=args.alpha, gamma=args.gamma,
                          epsilon=args.epsilon, sigma=args.sigma)
            kwargs[param] = v
            _, r, _, c = train_agent(
                grid_path, start,
                **kwargs, episodes=args.episodes, max_steps=args.max_steps,
                seed=seed,
            )
            ret[seed], suc[seed] = r, c
        all_returns[v] = ret
        all_success[v] = suc
    elapsed = time.time() - t0

    # Curves plot
    k = max(1, args.episodes // 50)
    x = np.arange(args.episodes)
    xs = x[k - 1:] if k > 1 else x
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("plasma")
    for i, v in enumerate(values):
        color = cmap(i / max(1, len(values) - 1))
        m, sd = smooth(all_returns[v].mean(axis=0), k), smooth(all_returns[v].std(axis=0), k)
        ax.plot(xs, m, label=f"{param}={v}", color=color, linewidth=2)
        ax.fill_between(xs, m - sd, m + sd, color=color, alpha=0.13)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.7)
    ax.set(xlabel="Episode", ylabel="Return",
           title=f"SARSA learning curves vs {param} - {grid_path.name} | "
                 f"{args.seeds} seeds, smoothed (k={k})")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    curve_path = out_path(args.out_dir, f"sweep_{param}_curves", stamp, "png")
    plt.savefig(curve_path, dpi=120); plt.close(fig)

    # CSV summary
    last_n = max(1, int(0.1 * args.episodes))
    csv_path = out_path(args.out_dir, f"sweep_{param}", stamp, "csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([param, "mean_return_last10pct", "std_return_last10pct",
                    "success_rate_last10pct"])
        for v in values:
            tail = all_returns[v][:, -last_n:].mean()
            tail_std = all_returns[v][:, -last_n:].mean(axis=1).std()
            tail_suc = all_success[v][:, -last_n:].mean()
            w.writerow([v, tail, tail_std, tail_suc])

    write_meta(csv_path, {
        "cmd": "sweep", "param": param, "values": values,
        "grid": grid_path, "start_pos": start,
        "alpha": args.alpha, "gamma": args.gamma, "epsilon": args.epsilon,
        "sigma": args.sigma, "episodes": args.episodes,
        "max_steps": args.max_steps, "seeds": args.seeds,
        "curves_plot": curve_path, "elapsed_sec": round(elapsed, 1),
    })

    print(f"\nElapsed {elapsed:.1f}s")
    print(f"{param:>8} | {'mean ret':>10} | {'std':>7} | {'success':>9}")
    print("-" * 44)
    for v in values:
        tail = all_returns[v][:, -last_n:].mean()
        tail_std = all_returns[v][:, -last_n:].mean(axis=1).std()
        tail_suc = all_success[v][:, -last_n:].mean()
        print(f"{v:>8.3f} | {tail:>+10.2f} | {tail_std:>7.2f} | {tail_suc:>8.2%}")
    print(f"  -> {curve_path}\n  -> {csv_path}")


# ====================================================================
# Subcommand: sweep-eps (epsilon-schedule sweep)
# ====================================================================

SCHEDULES_DEFAULT = [
    # (label, eps0, eps_end, decay_fraction_of_episodes)
    ("fixed_0.05",       0.05, None, 1.0),
    ("fixed_0.10",       0.10, None, 1.0),
    ("fixed_0.30",       0.30, None, 1.0),
    ("decay_0.30->0.05", 0.30, 0.05, 0.7),
]


def cmd_sweep_eps(args):
    stamp = time.strftime("%Y-%m-%d__%H-%M-%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    grid_path = args.GRID
    start = parse_start_pos(args.start_pos) or lock_start(grid_path, 0)

    all_returns = {}
    all_success = {}
    t0 = time.time()
    for label, eps0, eps_end, frac in SCHEDULES_DEFAULT:
        decay_eps = max(1, int(args.episodes * frac)) if eps_end is not None else 1
        ret = np.zeros((args.seeds, args.episodes))
        suc = np.zeros((args.seeds, args.episodes), dtype=int)
        for seed in trange(args.seeds, desc=label):
            _, r, _, c = train_agent(
                grid_path, start,
                alpha=args.alpha, gamma=args.gamma,
                epsilon=eps0, epsilon_end=eps_end, epsilon_decay_episodes=decay_eps,
                sigma=args.sigma, episodes=args.episodes, max_steps=args.max_steps,
                seed=seed)
            ret[seed], suc[seed] = r, c
        all_returns[label] = ret
        all_success[label] = suc
    elapsed = time.time() - t0

    k = max(1, args.episodes // 50)
    x = np.arange(args.episodes)
    xs = x[k - 1:] if k > 1 else x
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("viridis")
    for i, (label, *_) in enumerate(SCHEDULES_DEFAULT):
        color = cmap(i / max(1, len(SCHEDULES_DEFAULT) - 1))
        m, sd = smooth(all_returns[label].mean(axis=0), k), smooth(all_returns[label].std(axis=0), k)
        ax.plot(xs, m, label=label, color=color, linewidth=2)
        ax.fill_between(xs, m - sd, m + sd, color=color, alpha=0.13)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.7)
    ax.set(xlabel="Episode", ylabel="Return",
           title=f"Learning curves vs epsilon schedule - {grid_path.name}")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    curve_path = out_path(args.out_dir, "sweep_epsilon_curves", stamp, "png")
    plt.savefig(curve_path, dpi=120); plt.close(fig)

    last_n = max(1, int(0.1 * args.episodes))
    csv_path = out_path(args.out_dir, "sweep_epsilon", stamp, "csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["schedule", "mean_return_last10pct", "std_return_last10pct",
                    "success_rate_last10pct"])
        for label, *_ in SCHEDULES_DEFAULT:
            tail = all_returns[label][:, -last_n:].mean()
            tail_std = all_returns[label][:, -last_n:].mean(axis=1).std()
            tail_suc = all_success[label][:, -last_n:].mean()
            w.writerow([label, tail, tail_std, tail_suc])

    write_meta(csv_path, {
        "cmd": "sweep-eps",
        "schedules": [{"label": lab, "eps0": e0, "eps_end": ee,
                       "decay_frac": fr}
                      for lab, e0, ee, fr in SCHEDULES_DEFAULT],
        "grid": grid_path, "start_pos": start,
        "alpha": args.alpha, "gamma": args.gamma, "sigma": args.sigma,
        "episodes": args.episodes, "max_steps": args.max_steps,
        "seeds": args.seeds, "curves_plot": curve_path,
        "elapsed_sec": round(elapsed, 1),
    })

    print(f"\nElapsed {elapsed:.1f}s")
    print(f"{'schedule':<20} | {'mean ret':>10} | {'std':>7} | {'success':>9}")
    print("-" * 56)
    for label, *_ in SCHEDULES_DEFAULT:
        tail = all_returns[label][:, -last_n:].mean()
        tail_std = all_returns[label][:, -last_n:].mean(axis=1).std()
        tail_suc = all_success[label][:, -last_n:].mean()
        print(f"{label:<20} | {tail:>+10.2f} | {tail_std:>7.2f} | {tail_suc:>8.2%}")
    print(f"  -> {curve_path}\n  -> {csv_path}")


# ====================================================================
# Subcommand: transfer
# ====================================================================

def cmd_transfer(args):
    stamp = time.strftime("%Y-%m-%d__%H-%M-%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_start = lock_start(args.train_grid, 0)

    train_evals, test_evals = [], []
    t0 = time.time()
    for seed in trange(args.seeds, desc="seed"):
        agent, *_ = train_agent(
            args.train_grid, train_start,
            alpha=args.alpha, gamma=args.gamma,
            epsilon=args.epsilon, epsilon_end=args.epsilon_end,
            epsilon_decay_episodes=args.epsilon_decay_episodes,
            sigma=args.sigma, episodes=args.episodes, max_steps=args.max_steps,
            seed=seed, track_per_episode=False,
        )
        # Eval on train grid
        per_seed_train = [eval_greedy(args.train_grid, agent, train_start,
                                      sigma=args.sigma, max_steps=args.max_steps,
                                      seed=seed * 31 + i)
                          for i in range(args.eval_episodes)]
        # Eval on test grid (use its own locked start)
        test_start = lock_start(args.test_grid, seed + 10_000)
        per_seed_test = [eval_greedy(args.test_grid, agent, test_start,
                                     sigma=args.sigma, max_steps=args.max_steps,
                                     seed=seed * 37 + i)
                         for i in range(args.eval_episodes)]
        train_evals.append(per_seed_train)
        test_evals.append(per_seed_test)
    elapsed = time.time() - t0

    def summarize(evals, label):
        arr = np.array(evals, dtype=float)  # (seeds, eval_eps, 4)
        return {
            "scenario": label,
            "mean_return": float(arr[..., 2].mean()),
            "std_return": float(arr[..., 2].mean(axis=1).std()),
            "success_rate": float(arr[..., 3].mean()),
            "mean_steps": float(arr[..., 0].mean()),
        }

    rows = [summarize(train_evals, f"eval on {args.train_grid.stem} (train grid)"),
            summarize(test_evals, f"eval on {args.test_grid.stem} (test grid)")]
    csv_path = out_path(
        args.out_dir,
        f"transfer_{args.train_grid.stem}_to_{args.test_grid.stem}",
        stamp, "csv")
    write_csv(csv_path, list(rows[0].keys()), rows)
    write_meta(csv_path, {
        "cmd": "transfer", "train_grid": args.train_grid,
        "test_grid": args.test_grid, "train_start_pos": train_start,
        "alpha": args.alpha, "gamma": args.gamma, "epsilon": args.epsilon,
        "epsilon_end": args.epsilon_end,
        "epsilon_decay_episodes": args.epsilon_decay_episodes,
        "sigma": args.sigma, "episodes": args.episodes,
        "eval_episodes": args.eval_episodes, "max_steps": args.max_steps,
        "seeds": args.seeds, "elapsed_sec": round(elapsed, 1),
    })

    print(f"\nElapsed {elapsed:.1f}s")
    print(f"{'scenario':<46} | {'return':>10} | {'success':>9} | {'steps':>7}")
    print("-" * 85)
    for r in rows:
        print(f"{r['scenario']:<46} | {r['mean_return']:>+10.2f} | "
              f"{r['success_rate']:>8.2%} | {r['mean_steps']:>7.1f}")
    print(f"  -> {csv_path}")


# ====================================================================
# Subcommand: eval-configs (multi-config greedy eval, optionally multi-grid)
# ====================================================================

def cmd_eval_configs(args):
    """Train each config × seed and run dual-sigma greedy eval.

    Configs are read from --configs (JSON list). Each config dict supports:
      label, alpha, gamma, epsilon, epsilon_end, epsilon_decay_episodes,
      sigma_train, episodes, q_init, random_start, max_steps_train, max_steps_eval

    Grids can be specified positionally; defaults to all grids in grid_configs/.
    """
    stamp = time.strftime("%Y-%m-%d__%H-%M-%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configs = json.loads(Path(args.configs).read_text())
    if args.GRID:
        grids = list(args.GRID)
    else:
        grids = sorted((Path(__file__).parent / "grid_configs").glob("*.npy"))
    seeds = list(range(args.seeds))

    rows = []
    t0 = time.time()
    for grid_path in grids:
        start = lock_start(grid_path, 0)
        print(f"\n--- {grid_path.stem} (start={start}) ---")
        for cfg in configs:
            mst = cfg.get("max_steps_train", 500)
            mse = cfg.get("max_steps_eval", 500)
            per_det, per_n10 = [], []
            t_start = time.time()
            for seed in seeds:
                agent, *_ = train_agent(
                    grid_path, start,
                    algo=cfg.get("algo", "sarsa"),
                    alpha=cfg["alpha"], gamma=cfg["gamma"],
                    epsilon=cfg["epsilon"],
                    epsilon_end=cfg.get("epsilon_end"),
                    epsilon_decay_episodes=cfg.get("epsilon_decay_episodes", 1),
                    lambda_=cfg.get("lambda_"),
                    sigma=cfg.get("sigma_train", 0.1),
                    episodes=cfg["episodes"], max_steps=mst,
                    q_init=cfg.get("q_init"),
                    random_start=cfg.get("random_start", False),
                    seed=seed, track_per_episode=False,
                )
                per_det.append(eval_greedy(grid_path, agent, start,
                                           sigma=0.0, max_steps=mse, seed=seed))
                per_n10.append(eval_greedy(grid_path, agent, start,
                                           sigma=0.1, max_steps=mse, seed=seed))
            dt = time.time() - t_start
            d, n = np.array(per_det, dtype=float), np.array(per_n10, dtype=float)
            row = {
                "grid": grid_path.stem, "label": cfg["label"],
                "det_steps_mean": d[:, 0].mean(), "det_steps_std": d[:, 0].std(),
                "det_failed_mean": d[:, 1].mean(), "det_failed_std": d[:, 1].std(),
                "det_reward_mean": d[:, 2].mean(), "det_reward_std": d[:, 2].std(),
                "det_success": d[:, 3].mean(),
                "n10_steps_mean": n[:, 0].mean(), "n10_steps_std": n[:, 0].std(),
                "n10_reward_mean": n[:, 2].mean(), "n10_reward_std": n[:, 2].std(),
                "n10_success": n[:, 3].mean(),
                "elapsed_sec": round(dt, 1),
            }
            rows.append(row)
            print(f"  {cfg['label']:30s} | det: steps={row['det_steps_mean']:6.1f}"
                  f"+-{row['det_steps_std']:5.1f} r={row['det_reward_mean']:+8.1f} "
                  f"succ={row['det_success']:4.0%} | s=.1: steps={row['n10_steps_mean']:6.1f}"
                  f"+-{row['n10_steps_std']:5.1f} succ={row['n10_success']:4.0%} [{dt:.1f}s]")

    elapsed = time.time() - t0
    csv_path = out_path(args.out_dir, f"eval_configs_{args.tag}", stamp, "csv")
    write_csv(csv_path, list(rows[0].keys()), rows)
    write_meta(csv_path, {
        "cmd": "eval-configs", "tag": args.tag,
        "configs_file": args.configs, "configs": configs,
        "grids": [str(g) for g in grids], "seeds": args.seeds,
        "elapsed_sec": round(elapsed, 1),
    })
    print(f"\nElapsed {elapsed:.1f}s\n  -> {csv_path}")


# ====================================================================
# Entry point
# ====================================================================

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    p = ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    pt = sub.add_parser("train", help="Train SARSA on a grid.")
    pt.add_argument("GRID", type=Path, nargs="+")
    pt.add_argument("--episodes", type=int, default=1000)
    pt.add_argument("--max_steps", type=int, default=500)
    pt.add_argument("--alpha", type=float, default=0.1)
    pt.add_argument("--gamma", type=float, default=0.95)
    pt.add_argument("--epsilon", type=float, default=0.1)
    pt.add_argument("--epsilon_end", type=float, default=None)
    pt.add_argument("--epsilon_decay_episodes", type=int, default=1)
    pt.add_argument("--sigma", type=float, default=0.1)
    pt.add_argument("--q_init", type=float, default=None)
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--start_pos", type=str, default=None)
    pt.add_argument("--out_dir", type=Path, default=Path("results"))
    pt.set_defaults(func=cmd_train)

    # compare
    pc = sub.add_parser("compare", help="Multi-seed SARSA vs Random.")
    pc.add_argument("GRID", type=Path)
    pc.add_argument("--episodes", type=int, default=1000)
    pc.add_argument("--max_steps", type=int, default=500)
    pc.add_argument("--alpha", type=float, default=0.1)
    pc.add_argument("--gamma", type=float, default=0.95)
    pc.add_argument("--epsilon", type=float, default=0.1)
    pc.add_argument("--epsilon_end", type=float, default=None)
    pc.add_argument("--epsilon_decay_episodes", type=int, default=1)
    pc.add_argument("--sigma", type=float, default=0.1)
    pc.add_argument("--seeds", type=int, default=5)
    pc.add_argument("--random_start", action="store_true")
    pc.add_argument("--start_pos", type=str, default=None)
    pc.add_argument("--out_dir", type=Path, default=Path("results"))
    pc.set_defaults(func=cmd_compare)

    # sweep
    ps = sub.add_parser("sweep", help="Sweep one numeric hyperparameter (alpha/gamma/sigma).")
    ps.add_argument("GRID", type=Path)
    ps.add_argument("--param", choices=["alpha", "gamma", "sigma"], required=True)
    ps.add_argument("--values", type=str, required=True,
                    help="Comma-separated list, e.g. '0.01,0.05,0.1,0.3,0.5'")
    ps.add_argument("--episodes", type=int, default=1000)
    ps.add_argument("--max_steps", type=int, default=500)
    ps.add_argument("--alpha", type=float, default=0.1)
    ps.add_argument("--gamma", type=float, default=0.95)
    ps.add_argument("--epsilon", type=float, default=0.1)
    ps.add_argument("--sigma", type=float, default=0.1)
    ps.add_argument("--seeds", type=int, default=5)
    ps.add_argument("--start_pos", type=str, default=None)
    ps.add_argument("--out_dir", type=Path, default=Path("results"))
    ps.set_defaults(func=cmd_sweep)

    # sweep-eps
    pe = sub.add_parser("sweep-eps", help="Epsilon-schedule sweep (fixed vs decay).")
    pe.add_argument("GRID", type=Path)
    pe.add_argument("--episodes", type=int, default=1000)
    pe.add_argument("--max_steps", type=int, default=500)
    pe.add_argument("--alpha", type=float, default=0.1)
    pe.add_argument("--gamma", type=float, default=0.95)
    pe.add_argument("--sigma", type=float, default=0.1)
    pe.add_argument("--seeds", type=int, default=5)
    pe.add_argument("--start_pos", type=str, default=None)
    pe.add_argument("--out_dir", type=Path, default=Path("results"))
    pe.set_defaults(func=cmd_sweep_eps)

    # transfer
    px = sub.add_parser("transfer", help="Train on grid A, eval greedily on grid B.")
    px.add_argument("--train_grid", type=Path, required=True)
    px.add_argument("--test_grid", type=Path, required=True)
    px.add_argument("--episodes", type=int, default=2000)
    px.add_argument("--eval_episodes", type=int, default=50)
    px.add_argument("--max_steps", type=int, default=500)
    px.add_argument("--alpha", type=float, default=0.1)
    px.add_argument("--gamma", type=float, default=0.95)
    px.add_argument("--epsilon", type=float, default=0.3)
    px.add_argument("--epsilon_end", type=float, default=0.05)
    px.add_argument("--epsilon_decay_episodes", type=int, default=1500)
    px.add_argument("--sigma", type=float, default=0.1)
    px.add_argument("--seeds", type=int, default=5)
    px.add_argument("--out_dir", type=Path, default=Path("results"))
    px.set_defaults(func=cmd_transfer)

    # eval-configs
    pec = sub.add_parser("eval-configs", help="Train each config x seed and dual-sigma greedy eval.")
    pec.add_argument("GRID", type=Path, nargs="*",
                     help="Grids to evaluate on; if empty, uses all grid_configs/*.npy")
    pec.add_argument("--configs", type=Path, required=True,
                     help="Path to JSON file with list of config dicts.")
    pec.add_argument("--seeds", type=int, default=5)
    pec.add_argument("--tag", type=str, default="run",
                     help="Tag for output filename: eval_configs_<tag>_<stamp>.csv")
    pec.add_argument("--out_dir", type=Path, default=Path("results"))
    pec.set_defaults(func=cmd_eval_configs)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
