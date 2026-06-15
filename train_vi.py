"""
Train your RL Agent in this file.
"""

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange

from world import Environment
from world.grid import Grid
from agents.value_iteration_agent import ValueIterationAgent

from utils import compute_bfs_distances
from metrics import (compute_optimality_ratio, extract_visited_states,
                     print_metrics_summary)

def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer — Value Iteration.")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use. There can be more than one.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster.")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second to render at. Only used if no_gui is not set.")
    p.add_argument("--iter", type=int, default=1000,
                   help="Number of steps to run per episode after planning.") # Irrelevant for VI but kept for consistency with other agents and evaluation.
    p.add_argument("--random_seed", type=int, default=27,
                   help="Random seed value for the environment.")
    p.add_argument("--start_pos", type=str, default=None,
                   help="Agent start position as col,row (e.g. 1,12).")
    p.add_argument("--gamma", type=float, default=0.9,
                   help="Discount factor for Value Iteration.")
    p.add_argument("--shaping_weight", type=float, default=0.0,
                   help="Scaling factor for BFS potential-based reward shaping. "
                        "Set to 0 to disable.")
    return p.parse_args()


def main(
    grid_paths: list[Path],
    no_gui: bool,
    iters: int,
    fps: int,
    sigma: float,
    random_seed: int,
    start_pos: tuple[int, int] | None,
    gamma: float,
    shaping_weight: float,
):
    """Main loop of the program."""

    for grid in grid_paths:

        # Set up the environment
        env = Environment(grid, no_gui, sigma=sigma, target_fps=fps,
                          agent_start_pos=start_pos,
                          random_seed=random_seed)

        # Initialize agent — VI plans offline from the grid before any interaction
        grid_array = Grid.load_grid(grid).cells
        agent = ValueIterationAgent(
            grid_array,
            sigma=sigma,
            gamma=gamma,
            shaping_weight=shaping_weight,
        )

        # Always reset the environment to initial state
        initial_pos = env.reset()
        state = initial_pos

        for _ in trange(iters):
            action = agent.take_action(state)
            state, reward, terminated, info = env.step(action)
            if terminated:
                break
            agent.update(state, reward, info["actual_action"])

        # Evaluate the agent
        dist = compute_bfs_distances(grid_array)

        optimality = compute_optimality_ratio(
            env, agent, initial_pos, dist,
            n_eval_episodes=10, max_steps=iters,
        )
        visited_states, total_states = extract_visited_states(agent)

        print_metrics_summary(
            "Value Iteration", grid.stem,
            convergence_ep=-1,
            optimality=optimality,
            visited_states=visited_states,
            total_states=total_states,
        )

        Environment.evaluate_agent(grid, agent, iters, sigma,
                                agent_start_pos=initial_pos,
                                random_seed=random_seed)


if __name__ == '__main__':
    args = parse_args()
    start_pos = None
    if args.start_pos is not None:
        parts = args.start_pos.split(',')
        start_pos = (int(parts[0]), int(parts[1]))
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma,
         args.random_seed, start_pos, args.gamma, args.shaping_weight)
