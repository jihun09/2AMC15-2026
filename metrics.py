import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from agents import BaseAgent
from world import Environment

# Metric 1 — Cumulative reward (tracked in training loop, plotted here)

def plot_learning_curve(
    episode_rewards: list[float],
    episode_successes: list[int],
    title: str,
    save_path: Path,
    convergence_ep: int = -1,
    smoothing_window: int = 50,
) -> None:
    """Plot cumulative reward and success rate over training episodes.

    Args:
        episode_rewards:   List of cumulative reward per episode.
        episode_successes: List of 1/0 (success/failure) per episode.
        title:             Plot title (include algorithm name and grid).
        save_path:         Where to save the PNG file.
        convergence_ep:    Episode at which convergence was detected (-1 if never).
        smoothing_window:  Rolling average window size.
    """
    if not episode_rewards:
        print("No episode data to plot.")
        return

    n = len(episode_rewards)
    window = min(smoothing_window, max(1, n // 10))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    # --- Cumulative reward ---
    ax1.plot(episode_rewards, alpha=0.25, color="steelblue", label="Per episode")
    if window > 1:
        smoothed = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
        ax1.plot(range(window - 1, n), smoothed, color="steelblue",
                 linewidth=2, label=f"Rolling avg (w={window})")
    if convergence_ep > 0:
        ax1.axvline(convergence_ep, color="green", linestyle="--",
                    label=f"Converged @ ep {convergence_ep}")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Cumulative Reward")
    ax1.set_title(f"{title} — Cumulative Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Success rate ---
    if window > 1:
        success_rate = np.convolve(
            episode_successes, np.ones(window) / window, mode="valid"
        )
        ax2.plot(range(window - 1, n), success_rate, color="purple",
                 linewidth=2, label=f"Success rate (w={window})")
    else:
        ax2.plot(episode_successes, color="purple", linewidth=2)
    if convergence_ep > 0:
        ax2.axvline(convergence_ep, color="green", linestyle="--",
                    label=f"Converged @ ep {convergence_ep}")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Success Rate")
    ax2.set_title(f"{title} — Success Rate")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Learning curve saved: {save_path}")


# Metric 2 — Policy optimality ratio

def compute_optimality_ratio(
    env: Environment,
    agent: BaseAgent,
    start_pos: tuple[int, int],
    bfs_dist: np.ndarray,
    n_eval_episodes: int = 100,
    max_steps: int = 500,
) -> dict:
    """Evaluate the trained agent and compute the policy optimality ratio.

    optimality_ratio = optimal_steps / actual_steps (capped at 1.0).
    Mean and std are reported over successful episodes only.

    Returns dict with keys:
        optimal_steps, mean_actual_steps, std_actual_steps,
        success_rate, mean_optimality_ratio, std_optimality_ratio
    """
    optimal_steps = int(bfs_dist[start_pos])
    actual_steps_list = []
    success_flags = []
    cumulative_rewards = [] # stores cumulative rewards accross all tested iterations of the same environment

    for _ in range(n_eval_episodes):
        state = env.reset(agent_start_pos=start_pos)
        steps = 0
        reached = False
        for _ in range(max_steps):
            # Greedy action: use select_action if available (SARSA),
            # otherwise use take_action (VI always greedy; MC epsilon-greedy)
            if hasattr(agent, "select_action"):
                action = agent.select_action(state, training=False)
            else:
                action = agent.take_action(state)
            state, _, terminated, _ = env.step(action)
            steps += 1
            if terminated:
                reached = True
                break
        actual_steps_list.append(steps)
        success_flags.append(int(reached))
        cumulative_rewards.append(env.world_stats["cumulative_reward"])

    success_rate = np.mean(success_flags)
    successful_steps = [s for s, ok in zip(actual_steps_list, success_flags) if ok]
    mean_cumulative_rewards = float(np.mean(cumulative_rewards))
    std_cumulative_rewards = float(np.std(cumulative_rewards))


    if successful_steps:
        mean_steps = float(np.mean(successful_steps))
        std_steps = float(np.std(successful_steps))
        mean_ratio = min(1.0, optimal_steps / mean_steps)
        # Propagate std through ratio = optimal / steps
        std_ratio = (optimal_steps / (mean_steps ** 2)) * std_steps if mean_steps > 0 else 0.0
    else:
        mean_steps = float(np.mean(actual_steps_list))
        std_steps = float(np.std(actual_steps_list))
        mean_ratio = 0.0
        std_ratio = 0.0

    return {
        "optimal_steps":         optimal_steps,
        "mean_actual_steps":     mean_steps,
        "std_actual_steps":      std_steps,
        "success_rate":          float(success_rate),
        "mean_optimality_ratio": mean_ratio,
        "std_optimality_ratio":  std_ratio,
        "mean_cumulative_rewards" : mean_cumulative_rewards,
        "std_cumulative_rewards" : std_cumulative_rewards
    }


# Metric 3 — Unique states visited

def extract_visited_states(agent) -> tuple[set, int]:
    """Extract visited states and total reachable states from a trained agent.

    Returns:
        Tuple of (visited_states set, total_reachable_states int).
    """
    # SARSA: q_table keys are (state, action) tuples
    if hasattr(agent, "q_table"):
        visited = set(s for (s, a) in agent.q_table.keys())
        return visited, len(visited)  # SARSA only adds visited states to q_table

    # MC: state_action_indexer maps ALL reachable states -> index
    if hasattr(agent, "state_action_indexer"):
        all_states = set(agent.state_action_indexer.keys())
        # returns visited via q-values updated (non-zero returns)
        visited = set(
            state for state, idx in agent.state_action_indexer.items()
            if any(len(agent.returns.get((idx, a), [])) > 0
                   for a in range(len(agent.state_action_space[idx])))
        )
        return visited, len(all_states)

    # VI: V maps ALL states -> value (visits all by design)
    if hasattr(agent, "V"):
        all_states = set(agent.V.keys())
        return all_states, len(all_states)

    return set(), 0


# Summary printer

def print_metrics_summary(
    algorithm: str,
    grid_name: str,
    convergence_ep: int,
    optimality: dict,
    visited_states: set,
    total_states: int,
) -> None:
    """Print a formatted summary of all three metrics.

    Args:
        algorithm:       Algorithm name (e.g. "MC on-policy", "SARSA", "VI").
        grid_name:       Grid file stem (e.g. "A1_grid").
        convergence_ep:  Episode/iteration at convergence (-1 if not converged).
        optimality:      Dict returned by compute_optimality_ratio().
        visited_states:  Set of visited states from extract_visited_states().
        total_states:    Total reachable states from extract_visited_states().
    """
    conv_str = (f"episode {convergence_ep}"
                if convergence_ep > 0 else "not converged")
    n_visited = len(visited_states)
    coverage = 100 * n_visited / total_states if total_states > 0 else 0.0

    print(f"\n{'='*50}")
    print(f"  {algorithm} | {grid_name}")
    print(f"{'='*50}")
    print(f"  Convergence speed:        {conv_str}")
    print(f"  BFS optimal path:         {optimality['optimal_steps']} steps")
    print(f"  Mean actual steps:        {optimality['mean_actual_steps']:.1f} "
          f"± {optimality['std_actual_steps']:.1f}")
    print(f"  Eval success rate:        {optimality['success_rate']:.2%}")
    print(f"  Mean optimality ratio:    {optimality['mean_optimality_ratio']:.4f} "
          f"± {optimality['std_optimality_ratio']:.4f}")
    print(f"  Mean Cumulative rewards:    {optimality['mean_cumulative_rewards']:.4f} "
          f"± {optimality['std_cumulative_rewards']:.4f}")
    print(f"  State coverage:           {n_visited}/{total_states} "
          f"({coverage:.1f}%)")
    print(f"{'='*50}\n")
