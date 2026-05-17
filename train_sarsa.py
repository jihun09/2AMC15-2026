"""
SARSA Training Script.

Trains a SARSA agent across multiple episodes and evaluates it.
Reports two key metrics:
  1. Policy optimality ratio: BFS_shortest_path / actual_steps (0-1, 1=optimal)
  2. Convergence speed: episode at which stopping criterion triggers

Usage examples:
    python train_sarsa.py grid_configs/A1_grid.npy --no_gui
    python train_sarsa.py grid_configs/A1_grid.npy grid_configs/large_grid.npy --no_gui --episodes 1000
"""

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt

from world import Environment
from world.grid import Grid
from agents.sarsa_agent import SarsaAgent
from agents.random_agent import RandomAgent
from utils import reward_fn, compute_bfs_distances, shaped_reward


# =============================================================================
# Convergence detection
# =============================================================================

def detect_convergence(episode_successes: list[int], window: int = 50,
                       threshold: float = 0.95) -> int:
    """Detect the episode at which the agent has converged.

    Convergence is defined as the first episode where the rolling success
    rate (over `window` episodes) reaches `threshold`.

    Returns:
        Episode number (0-indexed) at which convergence is reached,
        or -1 if never reached.
    """
    if len(episode_successes) < window:
        return -1
    success_arr = np.array(episode_successes, dtype=float)
    rolling = np.convolve(success_arr, np.ones(window) / window, mode='valid')
    above = np.where(rolling >= threshold)[0]
    if len(above) > 0:
        return int(above[0]) + window  # episode at which window ends
    return -1


# =============================================================================
# Greedy evaluation for policy optimality ratio
# =============================================================================

def evaluate_greedy(env: Environment, agent: SarsaAgent, start_pos: tuple[int, int],
                    bfs_dist: np.ndarray, max_steps: int = 500,
                    n_eval_episodes: int = 10) -> dict:
    """Evaluate the trained agent greedily and compute policy optimality ratio.

    Args:
        env: Environment instance.
        agent: Trained SARSA agent.
        start_pos: Starting position.
        bfs_dist: BFS distance array.
        max_steps: Max steps per evaluation episode.
        n_eval_episodes: Number of evaluation episodes to average over.

    Returns:
        Dict with evaluation metrics.
    """
    optimal_steps = int(bfs_dist[start_pos])
    actual_steps_list = []
    successes = 0

    for _ in range(n_eval_episodes):
        state = env.reset(agent_start_pos=start_pos)
        steps = 0
        for _ in range(max_steps):
            action = agent.select_action(state, training=False)
            state, _, terminated, _ = env.step(action)
            steps += 1
            if terminated:
                successes += 1
                break
        actual_steps_list.append(steps)

    avg_steps = np.mean(actual_steps_list)
    success_rate = successes / n_eval_episodes

    # Policy optimality ratio: optimal / actual (capped at 1.0)
    # Only meaningful when agent reaches the goal
    if success_rate > 0:
        # Average only over successful episodes
        successful_steps = [s for s, succ in
                           zip(actual_steps_list,
                               [1 if s < max_steps else 0 for s in actual_steps_list])
                           if succ]
        if successful_steps:
            avg_successful_steps = np.mean(successful_steps)
            optimality_ratio = min(1.0, optimal_steps / avg_successful_steps)
        else:
            optimality_ratio = 0.0
    else:
        optimality_ratio = 0.0

    return {
        "optimal_steps": optimal_steps,
        "avg_eval_steps": avg_steps,
        "success_rate": success_rate,
        "optimality_ratio": optimality_ratio,
    }


# =============================================================================
# Training
# =============================================================================

def parse_args():
    p = ArgumentParser(description="SARSA Agent Trainer and Evaluator.")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file(s) to use.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster.")
    p.add_argument("--sigma", type=float, default=0.02,
                   help="Stochasticity of the environment (default: 0.02).")
    p.add_argument("--shaping_weight", type=float, default=3.0,
                   help="BFS reward shaping weight (default: 3.0).")
    p.add_argument("--episodes", type=int, default=1000,
                   help="Number of training episodes (default: 1000).")
    p.add_argument("--max_steps", type=int, default=1000,
                   help="Max steps per episode (default: 1000).")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed for the environment.")
    p.add_argument("--start_pos", type=str, default="1,1",
                   help="Agent start position as row,col (default: 1,1).")

    # SARSA hyperparameters
    p.add_argument("--alpha", type=float, default=0.1,
                   help="Learning rate (default: 0.1).")
    p.add_argument("--gamma", type=float, default=0.9,
                   help="Discount factor (default: 0.9).")
    p.add_argument("--epsilon", type=float, default=0.1,
                   help="Exploration rate for ε-greedy (default: 0.1).")
    p.add_argument("--epsilon_decay", type=float, default=0.995,
                   help="Multiplicative epsilon decay per episode (default: 0.995).")
    p.add_argument("--epsilon_min", type=float, default=0.01,
                   help="Minimum/final epsilon value (default: 0.01).")
    p.add_argument("--epsilon_decay_mode", type=str, default="fixed",
                   choices=["multiplicative", "linear", "fixed"],
                   help="Epsilon decay strategy (default: fixed).")
    p.add_argument("--epsilon_decay_episodes", type=int, default=None,
                   help="Episodes over which linear decay happens "
                        "(default: same as --episodes).")

    # Evaluation
    p.add_argument("--eval_steps", type=int, default=500,
                   help="Max steps during greedy evaluation (default: 500).")
    p.add_argument("--eval_episodes", type=int, default=10,
                   help="Number of greedy evaluation episodes (default: 10).")
    p.add_argument("--convergence_window", type=int, default=50,
                   help="Window size for convergence detection (default: 50).")
    p.add_argument("--convergence_threshold", type=float, default=0.95,
                   help="Success rate threshold for convergence (default: 0.95).")
    p.add_argument("--name", type=str, default=None,
                   help="Experiment name for output files.")
    return p.parse_args()


def train_sarsa(env: Environment, agent: SarsaAgent, episodes: int,
                max_steps: int, start_pos: tuple[int, int] | None,
                bfs_dist: np.ndarray | None = None,
                shaping_weight: float = 3.0) -> tuple:
    """Train the SARSA agent over multiple episodes.

    Returns:
        Tuple of (episode_rewards, episode_steps, episode_successes).
    """
    episode_rewards = []
    episode_steps = []
    episode_successes = []

    for ep in trange(episodes, desc="Training SARSA"):
        state = env.reset(agent_start_pos=start_pos) if start_pos else env.reset()
        agent.reset_episode()

        action = agent.select_action(state, training=True)

        cumulative_reward = 0.0
        steps = 0
        success = False

        for step in range(max_steps):
            next_state, reward, terminated, info = env.step(action)
            actual_action = info["actual_action"]

            # Apply BFS reward shaping
            if bfs_dist is not None:
                reward = shaped_reward(reward, state, next_state,
                                       terminated, bfs_dist, shaping_weight)

            next_action = agent.select_action(next_state, training=True)

            agent.learn(state=state,
                        action=actual_action,
                        reward=reward,
                        next_state=next_state,
                        next_action=next_action,
                        done=terminated)

            cumulative_reward += reward
            steps += 1
            state = next_state
            action = next_action

            if terminated:
                success = True
                break

        agent.decay_epsilon()
        episode_rewards.append(cumulative_reward)
        episode_steps.append(steps)
        episode_successes.append(int(success))

    return episode_rewards, episode_steps, episode_successes


def plot_learning_curve(episode_rewards, episode_steps, episode_successes,
                        convergence_ep, title, save_path):
    """Plot learning curve with convergence marker."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    window = min(50, len(episode_rewards) // 5) if len(episode_rewards) > 10 else 1

    # Reward
    ax1.plot(episode_rewards, alpha=0.3, color='blue', label='Per episode')
    if window > 1:
        smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), smoothed,
                 color='red', linewidth=2, label=f'Rolling avg (w={window})')
    if convergence_ep > 0:
        ax1.axvline(convergence_ep, color='green', linestyle='--',
                    label=f'Converged @ ep {convergence_ep}')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Cumulative Reward')
    ax1.set_title(f'{title} - Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Steps
    ax2.plot(episode_steps, alpha=0.3, color='green', label='Per episode')
    if window > 1:
        smoothed_steps = np.convolve(episode_steps, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(episode_steps)), smoothed_steps,
                 color='red', linewidth=2, label=f'Rolling avg (w={window})')
    if convergence_ep > 0:
        ax2.axvline(convergence_ep, color='green', linestyle='--')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title(f'{title} - Steps per Episode')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Success rate
    if window > 1:
        success_rate = np.convolve(episode_successes,
                                   np.ones(window)/window, mode='valid')
        ax3.plot(range(window-1, len(episode_successes)), success_rate,
                 color='purple', linewidth=2, label=f'Success rate (w={window})')
    else:
        ax3.plot(episode_successes, color='purple', linewidth=2)
    if convergence_ep > 0:
        ax3.axvline(convergence_ep, color='green', linestyle='--',
                    label=f'Converged @ ep {convergence_ep}')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Success Rate')
    ax3.set_title(f'{title} - Success Rate')
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    args = parse_args()

    start_pos = None
    if args.start_pos is not None:
        parts = args.start_pos.split(',')
        start_pos = (int(parts[0]), int(parts[1]))

    epsilon_decay_episodes = args.epsilon_decay_episodes or args.episodes

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    for grid_path in args.GRID:
        grid_name = grid_path.stem

        print(f"\n{'='*60}")
        print(f"Grid: {grid_path} | Start: {start_pos}")
        print(f"α={args.alpha}, γ={args.gamma}, ε={args.epsilon} "
              f"({args.epsilon_decay_mode}), σ={args.sigma}")
        print(f"{'='*60}")

        # Experiment name
        if args.name:
            exp_name = f"{args.name}_{grid_name}"
        else:
            exp_name = (f"sarsa_{grid_name}_a{args.alpha}_g{args.gamma}"
                        f"_e{args.epsilon}_s{args.sigma}")

        # Environment with shared reward function
        env = Environment(grid_path, no_gui=True, sigma=args.sigma,
                          target_fps=-1, agent_start_pos=start_pos,
                          reward_fn=reward_fn,
                          random_seed=args.random_seed)

        # BFS distances (for shaping + optimality ratio)
        grid_cells = Grid.load_grid(grid_path).cells
        bfs_dist = compute_bfs_distances(grid_cells)
        optimal_path_length = int(bfs_dist[start_pos])
        print(f"BFS optimal path from {start_pos}: {optimal_path_length} steps")

        # Agent
        sarsa_agent = SarsaAgent(
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
            epsilon_min=args.epsilon_min,
            epsilon_decay_mode=args.epsilon_decay_mode,
            epsilon_decay_episodes=epsilon_decay_episodes,
            rng_seed=args.random_seed,
        )

        # Train
        episode_rewards, episode_steps, episode_successes = train_sarsa(
            env, sarsa_agent, args.episodes, args.max_steps, start_pos,
            bfs_dist=bfs_dist, shaping_weight=args.shaping_weight
        )

        # --- Metric 1: Convergence speed ---
        convergence_ep = detect_convergence(
            episode_successes,
            window=args.convergence_window,
            threshold=args.convergence_threshold
        )

        # --- Metric 2: Policy optimality ratio (greedy eval) ---
        eval_results = evaluate_greedy(
            env, sarsa_agent, start_pos, bfs_dist,
            max_steps=args.eval_steps,
            n_eval_episodes=args.eval_episodes
        )

        # Print results
        print(f"\n{'─'*40}")
        print(f"RESULTS: {grid_name}")
        print(f"{'─'*40}")
        print(f"  Convergence speed:      episode {convergence_ep} "
              f"({'not converged' if convergence_ep < 0 else 'converged'})")
        print(f"  BFS optimal path:       {eval_results['optimal_steps']} steps")
        print(f"  Avg eval steps:         {eval_results['avg_eval_steps']:.1f}")
        print(f"  Eval success rate:      {eval_results['success_rate']:.2%}")
        print(f"  Policy optimality ratio: {eval_results['optimality_ratio']:.4f}")
        print(f"  Q-table size:           {len(sarsa_agent.q_table)} entries")
        print(f"  Final epsilon:          {sarsa_agent.epsilon:.4f}")
        print(f"{'─'*40}\n")

        # Plot
        plot_title = (f"SARSA {grid_name} "
                      f"(α={args.alpha} γ={args.gamma} ε={args.epsilon} σ={args.sigma})")
        plot_path = results_dir / f"{exp_name}_learning_curve.png"
        plot_learning_curve(episode_rewards, episode_steps, episode_successes,
                            convergence_ep, plot_title, plot_path)
        print(f"Plot saved: {plot_path}")


if __name__ == '__main__':
    main()
