"""Train a PPO agent on a gridworld.

Follows the same structure as sarsa.py / train_mc_on_policy.py so all
training scripts have a consistent interface and produce the same metrics.

Usage:
    python3 train_ppo.py grid_configs/A1_grid.npy --no_gui --episodes 2000
    python3 train_ppo.py grid_configs/A1_grid.npy --no_gui --lr 1e-3 --shaping_weight 3.0
"""

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from agents.ppo_agent import PPOAgent
from utils import reward_fn, compute_bfs_distances, shaped_reward
from world import Environment
from world.grid import Grid
from metrics import (compute_optimality_ratio, plot_learning_curve,
                     print_metrics_summary)


def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer (PPO).")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file(s) to train on.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables GUI rendering to train faster.")
    p.add_argument("--sigma", type=float, default=0.0,
                   help="Environment stochasticity.")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--iter", type=int, default=500,
                   help="Max steps per episode.")
    p.add_argument("--random_seed", type=int, default=27)
    p.add_argument("--start_pos", type=str, default=None,
                   help="Agent start position as row,col (e.g. 2,3).")
    p.add_argument("--episodes", type=int, default=2000,
                   help="Total number of training episodes.")
    p.add_argument("--rollout_steps", type=int, default=2048,
                   help="Steps to collect per PPO update (spans multiple episodes).")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="Discount factor.")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Adam learning rate.")
    p.add_argument("--clip_eps", type=float, default=0.2,
                   help="PPO clipping epsilon.")
    p.add_argument("--k_epochs", type=int, default=4,
                   help="Gradient update epochs per rollout.")
    p.add_argument("--gae_lambda", type=float, default=0.95,
                   help="GAE lambda for advantage estimation.")
    p.add_argument("--entropy_coef", type=float, default=0.05,
                   help="Entropy bonus coefficient (higher = more exploration).")
    p.add_argument("--value_coef", type=float, default=0.5,
                   help="Critic loss coefficient.")
    p.add_argument("--hidden_size", type=int, default=64,
                   help="Hidden units per layer in actor/critic MLPs.")
    p.add_argument("--shaping_weight", type=float, default=0.0,
                   help="BFS potential-based reward shaping weight (0 = off).")
    return p.parse_args()


def main(
    grid_paths: list[Path],
    no_gui: bool,
    iters: int,
    fps: int,
    sigma: float,
    random_seed: int,
    start_pos: tuple[int, int] | None,
    episodes: int,
    rollout_steps: int,
    gamma: float,
    lr: float,
    clip_eps: float,
    k_epochs: int,
    gae_lambda: float,
    entropy_coef: float,
    value_coef: float,
    hidden_size: int,
    shaping_weight: float,
) -> list[dict]:
    """Train PPO on each grid in grid_paths.

    Returns a list of per-grid result dicts containing:
        agent, episode_rewards, episode_successes, initial_pos, grid_path.
    """
    results = []

    for grid in grid_paths:
        env = Environment(
            grid, no_gui, sigma=sigma, target_fps=fps,
            agent_start_pos=start_pos,
            random_seed=random_seed,
            reward_fn=reward_fn,
        )
        grid_cells = Grid.load_grid(grid).cells
        dist = compute_bfs_distances(grid_cells)

        agent = PPOAgent(
            n_actions=4,
            grid_shape=grid_cells.shape,
            hidden_size=hidden_size,
            lr=lr,
            gamma=gamma,
            clip_eps=clip_eps,
            k_epochs=k_epochs,
            gae_lambda=gae_lambda,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            rng_seed=random_seed,
        )

        episode_rewards: list[float] = []
        episode_successes: list[int] = []
        window_successes = 0
        window_steps: list[int] = []

        # --- rollout-based training loop ---
        # PPO collects `rollout_steps` transitions (spanning many episodes)
        # before each gradient update. This gives stable batch sizes.

        ep = 0
        initial_pos = env.reset()
        state = initial_pos
        ep_return = 0.0
        ep_step = 0
        terminated = False

        pbar = tqdm(total=episodes, desc=f"PPO {grid.stem}")

        while ep < episodes:
            # --- collect one rollout ---
            steps_collected = 0
            while steps_collected < rollout_steps and ep < episodes:
                action = agent.select_action(state, training=True)
                next_state, reward, terminated, _ = env.step(action)

                if shaping_weight:
                    reward = shaped_reward(
                        reward, state, next_state, terminated, dist, shaping_weight
                    )

                agent.store_reward(reward, terminated)
                ep_return += reward  # track un-normalised return for logging
                ep_step += 1
                steps_collected += 1
                state = next_state

                if terminated or ep_step >= iters:
                    episode_rewards.append(ep_return)
                    episode_successes.append(int(terminated))
                    if terminated:
                        window_successes += 1
                        window_steps.append(ep_step)

                    ep += 1
                    pbar.update(1)

                    ep_num = ep
                    if ep_num % 500 == 0 and ep_num > 0:
                        success_rate = 100 * window_successes / min(ep_num, 500)
                        avg_steps = int(np.mean(window_steps)) if window_steps else 0
                        pbar.write(
                            f"Episode {ep_num:5d}/{episodes} | "
                            f"Success rate: {success_rate:.0f}% | "
                            f"Avg steps (successes): {avg_steps}"
                        )
                        window_successes = 0
                        window_steps = []

                    if ep < episodes:
                        initial_pos = env.reset()
                        state = initial_pos
                        ep_return = 0.0
                        ep_step = 0
                        terminated = False

            # --- PPO update on the collected rollout ---
            last_value = 0.0 if terminated else agent.state_value(state)
            agent.learn(last_value=last_value)

        pbar.close()

        optimality = compute_optimality_ratio(
            env, agent, initial_pos, dist,
            n_eval_episodes=10, max_steps=iters,
        )
        visited_states = agent.visited_states
        print_metrics_summary(
            "PPO", grid.stem,
            convergence_ep=-1,
            optimality=optimality,
            visited_states=visited_states,
            total_states=len(visited_states),
        )

        results_dir = Path("results")
        plot_learning_curve(
            episode_rewards, episode_successes,
            title=(f"PPO | {grid.stem} | lr={lr} gamma={gamma} "
                   f"clip={clip_eps} sigma={sigma}"),
            save_path=results_dir / (
                f"ppo_{grid.stem}_learning_curve_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            ),
        )

        Environment.evaluate_agent(
            grid, agent, iters, sigma,
            agent_start_pos=initial_pos,
            random_seed=random_seed,
        )

        results.append({
            "agent": agent,
            "episode_rewards": episode_rewards,
            "episode_successes": episode_successes,
            "initial_pos": initial_pos,
            "grid_path": grid,
        })

    return results


if __name__ == "__main__":
    args = parse_args()
    start_pos = None
    if args.start_pos is not None:
        parts = args.start_pos.split(",")
        start_pos = (int(parts[0]), int(parts[1]))
    main(
        args.GRID, args.no_gui, args.iter, args.fps, args.sigma,
        args.random_seed, start_pos, args.episodes, args.rollout_steps,
        args.gamma, args.lr, args.clip_eps, args.k_epochs, args.gae_lambda,
        args.entropy_coef, args.value_coef, args.hidden_size,
        args.shaping_weight,
    )
