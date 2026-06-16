"""
Train DQN agent on the continuous-state delivery robot task.

Usage:
    python train_dqn.py --grid grid_configs/A1_grid.npy --episodes 1000 --no_gui --seed 42 --sigma 0.1
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch

from world.environment import Environment
from world.continuous_env import ContinuousEnv
from agents.dqn import DQNAgent


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
    p = argparse.ArgumentParser(description="Train DQN on the continuous delivery robot task.")
    p.add_argument("--grid", type=Path, default=Path("grid_configs/A1_grid.npy"),
                   help="Path to the grid file.")
    p.add_argument("--episodes", type=int, default=1000,
                   help="Number of training episodes.")
    p.add_argument("--max_steps", type=int, default=500,
                   help="Maximum environment steps per episode.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disable pygame rendering.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility.")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="Environment stochasticity (probability of random action).")
    p.add_argument("--max_range", type=int, default=None,
                   help="Maximum raycasting range in cells. None = full raycasting.")
    p.add_argument("--state_mode", choices=["gps", "raycasting", "both"], default="gps",
                   help="State representation: gps (2), raycasting (16), or both (18).")
    # DQN hyperparameters
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Adam learning rate.")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="Discount factor.")
    p.add_argument("--hidden_size", type=int, default=128,
                   help="Hidden layer width (two layers of this size).")
    p.add_argument("--epsilon", type=float, default=1.0,
                   help="Initial epsilon for epsilon-greedy exploration.")
    p.add_argument("--epsilon_decay", type=float, default=0.995,
                   help="Multiplicative epsilon decay applied after each episode.")
    p.add_argument("--epsilon_min", type=float, default=0.05,
                   help="Minimum epsilon value.")
    p.add_argument("--buffer_capacity", type=int, default=50_000,
                   help="Replay buffer capacity (number of transitions).")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Mini-batch size for gradient updates.")
    p.add_argument("--warmup", type=int, default=1_000,
                   help="Minimum transitions in buffer before training starts.")
    p.add_argument("--target_update_freq", type=int, default=100,
                   help="Steps between hard target network updates.")
    # Logging / checkpointing
    p.add_argument("--print_freq", type=int, default=10,
                   help="Print training stats every N episodes.")
    p.add_argument("--eval_freq", type=int, default=50,
                   help="Run greedy evaluation every N episodes.")
    p.add_argument("--eval_episodes", type=int, default=10,
                   help="Number of episodes per greedy evaluation.")
    p.add_argument("--save_freq", type=int, default=200,
                   help="Save a checkpoint every N episodes.")
    return p.parse_args()


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_greedy(
    agent: DQNAgent,
    env: Environment,
    state_mode: str,
    max_range,
    n_episodes: int,
    max_steps: int,
) -> dict:
    agent.training_mode = False
    successes = 0
    rewards = []
    steps_list = []

    eval_env = ContinuousEnv(env, mode=state_mode, max_range=max_range)
    for _ in range(n_episodes):
        state = eval_env.reset()
        ep_reward = 0.0
        for step in range(max_steps):
            action = agent.take_action(state)
            state, reward, done, _ = eval_env.step(action)
            ep_reward += reward
            if done:
                successes += 1
                steps_list.append(step + 1)
                break
        else:
            steps_list.append(max_steps)
        rewards.append(ep_reward)

    agent.training_mode = True
    return {
        "success_rate": successes / n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "mean_steps": float(np.mean(steps_list)),
    }


def main():
    args = parse_args()
    set_seeds(args.seed)

    results_dir = Path("results")
    checkpoints_dir = Path("checkpoints")
    results_dir.mkdir(exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)

    base_env = Environment(
        args.grid,
        no_gui=args.no_gui,
        sigma=args.sigma,
        reward_fn=custom_reward,
        random_seed=args.seed,
    )
    cont_env = ContinuousEnv(base_env, mode=args.state_mode, max_range=args.max_range)

    agent = DQNAgent(
        state_dim=cont_env.state_dim,
        n_actions=8,
        hidden_size=args.hidden_size,
        lr=args.lr,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        warmup=args.warmup,
        target_update_freq=args.target_update_freq,
    )

    run_name = (
        f"dqn_{args.state_mode}_seed{args.seed}_sigma{args.sigma}"
        f"_lr{args.lr}_g{args.gamma}_h{args.hidden_size}"
        f"_range{'full' if args.max_range is None else args.max_range}"
    )
    train_csv = results_dir / f"{run_name}_training.csv"
    eval_csv = results_dir / f"{run_name}_eval.csv"

    with open(train_csv, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "reward", "steps", "success", "epsilon"])
    with open(eval_csv, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "success_rate", "mean_reward", "mean_steps"])

    print(f"DQN Training | grid={args.grid} | episodes={args.episodes} | seed={args.seed}")
    print(f"state_mode={args.state_mode} (dim={cont_env.state_dim}) | sigma={args.sigma} | max_range={'full' if args.max_range is None else args.max_range}")
    print(f"device={agent.device} | hidden={args.hidden_size} | lr={args.lr} | gamma={args.gamma}")
    print(f"eps: {args.epsilon} → {args.epsilon_min} (decay={args.epsilon_decay})")
    print(f"buffer={args.buffer_capacity} | batch={args.batch_size} | warmup={args.warmup} | target_C={args.target_update_freq}")
    print("-" * 70)

    episode_rewards = []
    episode_successes = []
    window = 50

    for ep in range(1, args.episodes + 1):
        state = cont_env.reset()
        ep_reward = 0.0
        success = False

        for step in range(args.max_steps):
            action = agent.take_action(state)
            next_state, reward, done, _ = cont_env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            ep_reward += reward
            if done:
                success = True
                break

        agent.decay_epsilon()
        episode_rewards.append(ep_reward)
        episode_successes.append(int(success))

        with open(train_csv, "a", newline="") as f:
            csv.writer(f).writerow([ep, ep_reward, step + 1, int(success), agent.epsilon])

        if ep % args.print_freq == 0:
            avg_r = np.mean(episode_rewards[-window:])
            sr = np.mean(episode_successes[-window:]) * 100
            print(
                f"Ep {ep:5d}/{args.episodes} | "
                f"reward={ep_reward:8.2f} | avg{window}={avg_r:8.2f} | "
                f"sr={sr:5.1f}% | eps={agent.epsilon:.4f} | "
                f"buf={len(agent.replay_buffer)}"
            )

        if ep % args.eval_freq == 0:
            stats = evaluate_greedy(agent, base_env, args.state_mode, args.max_range,
                                    args.eval_episodes, args.max_steps)
            with open(eval_csv, "a", newline="") as f:
                csv.writer(f).writerow([ep, stats["success_rate"],
                                        stats["mean_reward"], stats["mean_steps"]])
            print(
                f"  [eval] success={stats['success_rate']*100:.1f}% | "
                f"mean_r={stats['mean_reward']:.2f} | mean_steps={stats['mean_steps']:.1f}"
            )

        if ep % args.save_freq == 0:
            agent.save(str(checkpoints_dir / f"{run_name}_ep{ep}.pt"))

    agent.save(str(checkpoints_dir / f"{run_name}_final.pt"))

    np.save(results_dir / f"{run_name}_rewards.npy", np.array(episode_rewards))
    np.save(results_dir / f"{run_name}_successes.npy", np.array(episode_successes))

    print(f"\nDone. Results: {results_dir}/{run_name}_*.{{csv,npy}}")
    print(f"Final model:  {checkpoints_dir}/{run_name}_final.pt")


if __name__ == "__main__":
    main()
