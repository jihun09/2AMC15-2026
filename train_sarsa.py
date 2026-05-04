"""
SARSA Training Script.

Trains a SARSA agent across multiple episodes and evaluates it.
Supports experimenting with different hyperparameters and grids.

Usage examples:
    # Basic training on A1_grid with defaults
    python train_sarsa.py grid_configs/A1_grid.npy --start_pos 1,12 --no_gui

    # Experiment with different hyperparameters
    python train_sarsa.py grid_configs/A1_grid.npy --start_pos 1,12 --no_gui \
        --alpha 0.1 --gamma 0.9 --epsilon 0.1 --sigma 0.02 --episodes 500

    # Compare two grids
    python train_sarsa.py grid_configs/A1_grid.npy grid_configs/example_grid.npy \
        --start_pos 1,12 --no_gui --episodes 1000
"""

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt

from world import Environment
from agents.sarsa_agent import SarsaAgent
from agents.random_agent import RandomAgent


def parse_args():
    p = ArgumentParser(description="SARSA Agent Trainer and Evaluator.")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file(s) to use.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster.")
    p.add_argument("--sigma", type=float, default=0.02,
                   help="Stochasticity of the environment (default: 0.02).")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second for GUI rendering.")
    p.add_argument("--episodes", type=int, default=500,
                   help="Number of training episodes (default: 500).")
    p.add_argument("--max_steps", type=int, default=1000,
                   help="Max steps per episode (default: 1000).")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed for the environment.")
    p.add_argument("--start_pos", type=str, default=None,
                   help="Agent start position as row,col (e.g. 1,12).")

    # SARSA hyperparameters
    p.add_argument("--alpha", type=float, default=0.1,
                   help="Learning rate (default: 0.1).")
    p.add_argument("--gamma", type=float, default=0.9,
                   help="Discount factor (default: 0.9).")
    p.add_argument("--epsilon", type=float, default=0.1,
                   help="Exploration rate for ε-greedy (default: 0.1).")
    p.add_argument("--epsilon_decay", type=float, default=0.995,
                   help="Epsilon decay per episode (default: 0.995).")
    p.add_argument("--epsilon_min", type=float, default=0.01,
                   help="Minimum epsilon value (default: 0.01).")

    # Evaluation
    p.add_argument("--eval_steps", type=int, default=200,
                   help="Max steps during evaluation (default: 200).")
    p.add_argument("--eval_random", action="store_true",
                   help="Also evaluate a random agent for comparison.")
    p.add_argument("--name", type=str, default=None,
                   help="Experiment name for output files. If not set, "
                        "auto-generates from hyperparameters.")
    return p.parse_args()


def train_sarsa(env: Environment, agent: SarsaAgent, episodes: int,
                max_steps: int, start_pos: tuple[int, int] | None) -> list[float]:
    """Train the SARSA agent over multiple episodes.

    Args:
        env: The environment to train in.
        agent: The SARSA agent.
        episodes: Number of episodes to train.
        max_steps: Maximum steps per episode.
        start_pos: Fixed starting position (for fair comparison).

    Returns:
        List of cumulative rewards per episode (for plotting learning curves).
    """
    episode_rewards = []
    episode_steps = []

    for ep in trange(episodes, desc="Training SARSA"):
        # Reset environment and agent episode state
        state = env.reset(agent_start_pos=start_pos) if start_pos else env.reset()
        agent.reset_episode()

        # Choose first action
        action = agent.take_action(state)

        cumulative_reward = 0.0
        steps = 0

        for step in range(max_steps):
            # Take action in environment
            next_state, reward, terminated, info = env.step(action)
            actual_action = info["actual_action"]

            # Choose next action from next_state (SARSA is on-policy)
            next_action = agent.take_action(next_state)

            # SARSA update: Q(s,a) += alpha * [r + gamma*Q(s',a') - Q(s,a)]
            current_q = agent.q_table[(state, actual_action)]
            next_q = agent.q_table[(next_state, next_action)]
            td_target = reward + agent.gamma * next_q
            td_error = td_target - current_q
            agent.q_table[(state, actual_action)] = current_q + agent.alpha * td_error

            cumulative_reward += reward
            steps += 1

            # Move to next state and action
            state = next_state
            action = next_action

            if terminated:
                break

        # Decay exploration rate after each episode
        agent.decay_epsilon()
        episode_rewards.append(cumulative_reward)
        episode_steps.append(steps)

    return episode_rewards, episode_steps


def plot_learning_curve(episode_rewards: list[float], episode_steps: list[float],
                        title: str, save_path: Path):
    """Plot and save the learning curve.

    Args:
        episode_rewards: Cumulative reward per episode.
        episode_steps: Steps taken per episode.
        title: Plot title.
        save_path: Where to save the figure.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Reward curve
    ax1.plot(episode_rewards, alpha=0.3, color='blue', label='Per episode')
    # Smoothed (rolling average)
    window = min(50, len(episode_rewards) // 5) if len(episode_rewards) > 10 else 1
    if window > 1:
        smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), smoothed,
                 color='red', linewidth=2, label=f'Rolling avg (window={window})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Cumulative Reward')
    ax1.set_title(f'{title} - Reward per Episode')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Steps curve
    ax2.plot(episode_steps, alpha=0.3, color='green', label='Per episode')
    if window > 1:
        smoothed_steps = np.convolve(episode_steps, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(episode_steps)), smoothed_steps,
                 color='red', linewidth=2, label=f'Rolling avg (window={window})')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps to Termination')
    ax2.set_title(f'{title} - Steps per Episode')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Learning curve saved to: {save_path}")


def main():
    args = parse_args()

    start_pos = None
    if args.start_pos is not None:
        parts = args.start_pos.split(',')
        start_pos = (int(parts[0]), int(parts[1]))

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    for grid_path in args.GRID:
        print(f"\n{'='*60}")
        print(f"Grid: {grid_path}")
        print(f"Hyperparameters: α={args.alpha}, γ={args.gamma}, "
              f"ε={args.epsilon}, ε_decay={args.epsilon_decay}, σ={args.sigma}")
        print(f"{'='*60}")

        # Build experiment name
        grid_name = grid_path.stem
        if args.name:
            exp_name = args.name
        else:
            exp_name = (f"sarsa_{grid_name}_a{args.alpha}_g{args.gamma}"
                        f"_e{args.epsilon}_s{args.sigma}")

        # Initialize environment
        env = Environment(grid_path, no_gui=True, sigma=args.sigma,
                          target_fps=-1, agent_start_pos=start_pos,
                          random_seed=args.random_seed)

        # Initialize SARSA agent
        sarsa_agent = SarsaAgent(
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
            epsilon_min=args.epsilon_min
        )

        # Train
        episode_rewards, episode_steps = train_sarsa(
            env, sarsa_agent, args.episodes, args.max_steps, start_pos
        )

        # Plot learning curve
        plot_title = (f"SARSA on {grid_name} "
                      f"(α={args.alpha}, γ={args.gamma}, ε={args.epsilon}, σ={args.sigma})")
        plot_path = results_dir / f"{exp_name}_learning_curve.png"
        plot_learning_curve(episode_rewards, episode_steps, plot_title, plot_path)

        # Evaluate trained SARSA agent
        print(f"\nEvaluating trained SARSA agent on {grid_name}...")
        Environment.evaluate_agent(grid_path, sarsa_agent, args.eval_steps,
                                   sigma=args.sigma,
                                   agent_start_pos=start_pos,
                                   random_seed=args.random_seed)

        # Optionally evaluate random agent for comparison
        if args.eval_random:
            print(f"\nEvaluating Random agent on {grid_name} (for comparison)...")
            random_agent = RandomAgent()
            Environment.evaluate_agent(grid_path, random_agent, args.eval_steps,
                                       sigma=args.sigma,
                                       agent_start_pos=start_pos,
                                       random_seed=args.random_seed)

        print(f"\nQ-table size: {len(sarsa_agent.q_table)} entries")
        print(f"Final epsilon: {sarsa_agent.epsilon:.4f}")


if __name__ == '__main__':
    main()
