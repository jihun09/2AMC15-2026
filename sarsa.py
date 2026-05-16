"""Train a SARSA agent on a gridworld.

Mirrors train_mc_on_policy.py so the two algorithms have the same shape,
the same metrics output, and the same CLI/import interface.
"""
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import trange

from agents.sarsa_agent import SARSAAgent
from utils import reward_fn, compute_bfs_distances, shaped_reward
from world import Environment
from world.grid import Grid
from metrics import (compute_optimality_ratio, extract_visited_states,
                     plot_learning_curve, print_metrics_summary)


def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer (SARSA).")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use. There can be more than one.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster.")
    p.add_argument("--sigma", type=float, default=0.0,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second to render at. Only used if no_gui is not set.")
    p.add_argument("--iter", type=int, default=500,
                   help="Max steps per episode.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    p.add_argument("--start_pos", type=str, default=None,
                   help="Agent start position as col,row (e.g. 2,3).")
    p.add_argument("--episodes", type=int, default=2000,
                   help="Number of training episodes.")
    p.add_argument("--gamma", type=float, default=0.9,
                   help="Discount factor.")
    p.add_argument("--alpha", type=float, default=0.1,
                   help="Learning rate.")
    p.add_argument("--epsilon", type=float, default=0.1,
                   help="Initial exploration rate.")
    p.add_argument("--epsilon_decay", type=float, default=1.0,
                   help="Multiplicative decay applied to epsilon after each episode.")
    p.add_argument("--epsilon_min", type=float, default=0.0,
                   help="Minimum value epsilon can decay to.")
    p.add_argument("--patience", type=int, default=100,
                   help="Stop training if greedy policy is stable for this many consecutive episodes.")
    p.add_argument("--shaping_weight", type=float, default=0.0,
                   help="Scaling factor for BFS potential-based reward shaping (0 to disable).")
    return p.parse_args()


def main(grid_paths: list[Path], no_gui: bool, iters: int, fps: int,
         sigma: float, random_seed: int, start_pos: tuple[int, int] | None,
         episodes: int, gamma: float, alpha: float, epsilon: float,
         epsilon_decay: float, epsilon_min: float, patience: int,
         shaping_weight: float):
    """Train SARSA on each grid in `grid_paths`.

    Returns a list of per-grid result dicts, each containing:
        agent, episode_rewards, episode_successes, convergence_ep,
        initial_pos, grid_path.
    """
    results = []

    for grid in grid_paths:
        env = Environment(grid, no_gui, sigma=sigma, target_fps=fps,
                          agent_start_pos=start_pos,
                          random_seed=random_seed,
                          reward_fn=reward_fn)
        grid_cells = Grid.load_grid(grid).cells
        dist = compute_bfs_distances(grid_cells)

        agent = SARSAAgent(n_actions=4, alpha=alpha, gamma=gamma,
                           epsilon=epsilon, epsilon_decay=epsilon_decay,
                           epsilon_min=epsilon_min, rng_seed=random_seed)

        episode_rewards = []
        episode_successes = []
        window_successes = 0
        window_steps = []
        prev_greedy_policy = None
        stable_count = 0
        last_ep = 0

        for ep in trange(episodes):
            initial_pos = env.reset()
            state = initial_pos
            action = agent.select_action(state, training=True)
            ep_return = 0.0
            success = False

            for step in range(iters):
                next_state, reward, terminated, info = env.step(action)
                actual_action = info["actual_action"]
                if shaping_weight:
                    reward = shaped_reward(reward, state, next_state,
                                           terminated, dist, shaping_weight)
                next_action = agent.select_action(next_state, training=True)
                agent.learn(state=state, action=actual_action, reward=reward,
                            next_state=next_state, next_action=next_action,
                            done=terminated)
                ep_return += reward
                state, action = next_state, next_action
                if terminated:
                    success = True
                    window_successes += 1
                    window_steps.append(step)
                    break

            episode_rewards.append(ep_return)
            episode_successes.append(int(success))
            agent.decay_epsilon()
            last_ep = ep

            current_greedy_policy = agent.get_greedy_action()
            if current_greedy_policy == prev_greedy_policy:
                stable_count += 1
            else:
                stable_count = 0
            prev_greedy_policy = current_greedy_policy

            if patience > 0 and stable_count >= patience:
                print(f"Policy stable for {patience} consecutive episodes. "
                      f"Stopping at episode {ep + 1}.")
                break

            episode_num = ep + 1
            if episode_num % 500 == 0:
                success_rate = 100 * window_successes / 500
                avg_steps = int(np.mean(window_steps)) if window_steps else 0
                print(f"Episode {episode_num:5d}/{episodes} | "
                      f"Epsilon: {agent.epsilon:.3f} | "
                      f"Success rate: {success_rate:.0f}% | "
                      f"Avg steps (successes): {avg_steps}")
                window_successes = 0
                window_steps = []

        convergence_ep = last_ep + 1 if (patience > 0 and stable_count >= patience) else -1

        optimality = compute_optimality_ratio(
            env, agent, initial_pos, dist,
            n_eval_episodes=10, max_steps=iters,
        )
        visited_states, total_states = extract_visited_states(agent)
        print_metrics_summary("SARSA", grid.stem, convergence_ep,
                              optimality, visited_states, total_states)

        results_dir = Path("results")
        plot_learning_curve(
            episode_rewards, episode_successes,
            title=f"SARSA | {grid.stem} | alpha={alpha} gamma={gamma} "
                  f"eps={epsilon} sigma={sigma}",
            save_path=results_dir / f"sarsa_{grid.stem}_learning_curve_"
                                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            convergence_ep=convergence_ep,
        )

        Environment.evaluate_agent(grid, agent, iters, sigma,
                                   agent_start_pos=initial_pos,
                                   random_seed=random_seed)

        results.append({
            "agent": agent,
            "episode_rewards": episode_rewards,
            "episode_successes": episode_successes,
            "convergence_ep": convergence_ep,
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
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma,
         args.random_seed, start_pos, args.episodes, args.gamma, args.alpha,
         args.epsilon, args.epsilon_decay, args.epsilon_min, args.patience,
         args.shaping_weight)
