"""Train PPO agent on the continuous-state delivery robot task.

Mirrors train_dqn.py: uses ContinuousEnv for state observations, supports
gps / raycasting / both state modes, and logs the same metrics.

Usage:
    python3 train_ppo.py --grid grid_configs/A1_grid.npy --no_gui --episodes 2000
    python3 train_ppo.py --grid grid_configs/A1_grid.npy --no_gui --state_mode raycasting
    python3 train_ppo.py --grid grid_configs/A1_grid.npy --no_gui --state_mode both --shaping_weight 3.0
"""

import argparse
import csv
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from world.environment import Environment
from world.continuous_env import ContinuousEnv
from world.grid import Grid
from agents.ppo_agent import PPOAgent
from utils import compute_bfs_distances, shaped_reward
from metrics import plot_learning_curve


def custom_reward(grid, agent_pos):
    match grid[agent_pos]:
        case 0:
            return -0.1
        case 1 | 2:
            return -1.0
        case 3:
            return 10.0
        case _:
            raise ValueError(f"Unexpected grid value {grid[agent_pos]} at {agent_pos}")


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO on the continuous delivery robot task.")
    p.add_argument("--grid", type=Path, default=Path("grid_configs/A1_grid.npy"),
                   help="Path to the grid file.")
    p.add_argument("--episodes", type=int, default=2000,
                   help="Number of training episodes.")
    p.add_argument("--max_steps", type=int, default=500,
                   help="Maximum environment steps per episode.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disable pygame rendering.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility.")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="Environment stochasticity.")
    p.add_argument("--max_range", type=int, default=None,
                   help="Maximum raycasting range in cells. None = full raycasting.")
    p.add_argument("--state_mode", choices=["gps", "raycasting", "both"], default="gps",
                   help="State representation: gps (2), raycasting (16), or both (18).")
    p.add_argument("--shaping_weight", type=float, default=0.0,
                   help="BFS potential-based reward shaping weight (0 = off).")
    # PPO hyperparameters
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Adam learning rate.")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="Discount factor.")
    p.add_argument("--hidden_size", type=int, default=128,
                   help="Hidden units per layer in actor/critic MLPs.")
    p.add_argument("--clip_eps", type=float, default=0.2,
                   help="PPO clipping epsilon.")
    p.add_argument("--k_epochs", type=int, default=4,
                   help="Gradient update epochs per rollout.")
    p.add_argument("--gae_lambda", type=float, default=0.95,
                   help="GAE lambda for advantage estimation.")
    p.add_argument("--entropy_coef", type=float, default=0.05,
                   help="Entropy bonus coefficient.")
    p.add_argument("--value_coef", type=float, default=0.5,
                   help="Critic loss coefficient.")
    p.add_argument("--rollout_steps", type=int, default=2048,
                   help="Steps to collect per PPO update (spans multiple episodes).")
    # Logging
    p.add_argument("--print_freq", type=int, default=10,
                   help="Print training stats every N episodes.")
    p.add_argument("--eval_freq", type=int, default=50,
                   help="Run greedy evaluation every N episodes.")
    p.add_argument("--eval_episodes", type=int, default=10,
                   help="Number of episodes per greedy evaluation.")
    return p.parse_args()


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_greedy(
    agent: PPOAgent,
    env: Environment,
    state_mode: str,
    max_range,
    n_episodes: int,
    max_steps: int,
) -> dict:
    eval_env = ContinuousEnv(env, mode=state_mode, max_range=max_range)
    successes = 0
    rewards = []
    steps_list = []

    for _ in range(n_episodes):
        state = eval_env.reset()
        ep_reward = 0.0
        for step in range(max_steps):
            action = agent.select_action(state, training=False)
            state, reward, done, _ = eval_env.step(action)
            ep_reward += reward
            if done:
                successes += 1
                steps_list.append(step + 1)
                break
        else:
            steps_list.append(max_steps)
        rewards.append(ep_reward)

    return {
        "success_rate": successes / n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "mean_steps": float(np.mean(steps_list)),
    }


def main():
    args = parse_args()
    set_seeds(args.seed)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    base_env = Environment(
        args.grid,
        no_gui=args.no_gui,
        sigma=args.sigma,
        reward_fn=custom_reward,
        random_seed=args.seed,
    )
    cont_env = ContinuousEnv(base_env, mode=args.state_mode, max_range=args.max_range)

    dist = compute_bfs_distances(Grid.load_grid(args.grid).cells) if args.shaping_weight else None

    agent = PPOAgent(
        state_dim=cont_env.state_dim,
        n_actions=8,
        hidden_size=args.hidden_size,
        lr=args.lr,
        gamma=args.gamma,
        clip_eps=args.clip_eps,
        k_epochs=args.k_epochs,
        gae_lambda=args.gae_lambda,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        rng_seed=args.seed,
    )

    run_name = (
        f"ppo_{args.state_mode}_seed{args.seed}_sigma{args.sigma}"
        f"_lr{args.lr}_g{args.gamma}_h{args.hidden_size}"
        f"_range{'full' if args.max_range is None else args.max_range}"
    )
    train_csv = results_dir / f"{run_name}_training.csv"
    eval_csv  = results_dir / f"{run_name}_eval.csv"

    with open(train_csv, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "reward", "steps", "success"])
    with open(eval_csv, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "success_rate", "mean_reward", "mean_steps"])

    print(f"PPO Training | grid={args.grid} | episodes={args.episodes} | seed={args.seed}")
    print(f"state_mode={args.state_mode} (dim={cont_env.state_dim}) | sigma={args.sigma} | max_range={'full' if args.max_range is None else args.max_range}")
    print(f"hidden={args.hidden_size} | lr={args.lr} | gamma={args.gamma} | clip={args.clip_eps}")
    print(f"rollout_steps={args.rollout_steps} | k_epochs={args.k_epochs} | entropy={args.entropy_coef}")
    print("-" * 70)

    episode_rewards: list[float] = []
    episode_successes: list[int] = []
    window = 50

    # Rollout-based loop: collect rollout_steps transitions spanning many
    # episodes before each PPO gradient update, matching standard PPO practice.
    ep = 0
    state = cont_env.reset()
    ep_return = 0.0
    ep_step = 0
    terminated = False

    while ep < args.episodes:
        steps_collected = 0
        while steps_collected < args.rollout_steps and ep < args.episodes:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, _ = cont_env.step(action)

            if args.shaping_weight and dist is not None:
                prev_pos = base_env.agent_pos
                reward = shaped_reward(reward, prev_pos, base_env.agent_pos,
                                       terminated, dist, args.shaping_weight)

            agent.store_reward(reward, terminated)
            ep_return += reward
            ep_step += 1
            steps_collected += 1
            state = next_state

            if terminated or ep_step >= args.max_steps:
                episode_rewards.append(ep_return)
                episode_successes.append(int(terminated))

                with open(train_csv, "a", newline="") as f:
                    csv.writer(f).writerow([ep + 1, ep_return, ep_step, int(terminated)])

                if (ep + 1) % args.print_freq == 0:
                    avg_r = np.mean(episode_rewards[-window:])
                    sr = np.mean(episode_successes[-window:]) * 100
                    print(
                        f"Ep {ep + 1:5d}/{args.episodes} | "
                        f"reward={ep_return:8.2f} | avg{window}={avg_r:8.2f} | "
                        f"sr={sr:5.1f}%"
                    )

                if (ep + 1) % args.eval_freq == 0:
                    stats = evaluate_greedy(agent, base_env, args.state_mode,
                                            args.max_range, args.eval_episodes,
                                            args.max_steps)
                    with open(eval_csv, "a", newline="") as f:
                        csv.writer(f).writerow([ep + 1, stats["success_rate"],
                                                stats["mean_reward"], stats["mean_steps"]])
                    print(
                        f"  [eval] success={stats['success_rate']*100:.1f}% | "
                        f"mean_r={stats['mean_reward']:.2f} | "
                        f"mean_steps={stats['mean_steps']:.1f}"
                    )

                ep += 1
                if ep < args.episodes:
                    state = cont_env.reset()
                    ep_return = 0.0
                    ep_step = 0
                    terminated = False

        last_value = 0.0 if terminated else agent.state_value(state)
        agent.learn(last_value=last_value)

    plot_learning_curve(
        episode_rewards, episode_successes,
        title=(f"PPO | {args.grid.stem} | {args.state_mode} | "
               f"lr={args.lr} gamma={args.gamma} sigma={args.sigma}"),
        save_path=results_dir / (
            f"ppo_{args.grid.stem}_{args.state_mode}_learning_curve_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        ),
    )

    print(f"\nDone. Results saved to {results_dir}/")


if __name__ == "__main__":
    main()