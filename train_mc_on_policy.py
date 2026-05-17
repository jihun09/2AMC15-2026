"""
Train your RL Agent in this file. 
"""

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
import numpy as np


from world import Environment
from agents.mc_on_policy_agent import McOnPolicyAgent
from agents.null_agent import NullAgent
from world.grid import Grid
from utils import reward_fn, compute_bfs_distances, shaped_reward
from metrics import (compute_optimality_ratio, extract_visited_states,
                     plot_learning_curve, print_metrics_summary)

from datetime import datetime # To save every plot with a unique name based on the current timestamp

def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use. There can be more than "
                        "one.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second to render at. Only used if "
                        "no_gui is not set.")
    p.add_argument("--iter", type=int, default=1000,
                   help="Number of iterations to go through.")
    p.add_argument("--random_seed", type=int, default=None,
                   help="Random seed value for the environment.")
    p.add_argument("--start_pos", type=str, default=None,
                   help="Agent start position as col,row (e.g. 2,3). "
                        "If not set, the GUI lets you click to place it. "
                        "In no_gui mode, defaults to random placement.")
    p.add_argument("--episodes", type=int, default=1000,
                   help="Number of episodes to be analysed by the agent.")
    p.add_argument("--delta", type=float, default=0.9,
                   help="Discount factor of reward.")
    p.add_argument("--epsilon", type=float, default=0.1,
                   help="Determines how often the agent performs greedy actions.")
    p.add_argument("--epsilon_decay", type=float, default=0.999,
                   help="Multiplicative decay factor applied to epsilon after each episode.")
    p.add_argument("--epsilon_min", type=float, default=0.0,
                   help="Minimum value epsilon can decay to.")
    p.add_argument("--patience", type=int, default=100,
                   help="Stop training if greedy policy is stable for this many consecutive episodes.")
    p.add_argument("--shaping_weight", type=float, default=3.0,
                   help="Scaling factor for BFS potential-based reward shaping (0 to disable).")
    
    
    return p.parse_args()


def main(grid_paths: list[Path], no_gui: bool, iters: int, fps: int,
         sigma: float, random_seed: int, start_pos: tuple[int, int] | None, 
         episodes: int, delta: float, epsilon: float, epsilon_decay: float,
         epsilon_min: float, patience: int, shaping_weight: float):
    """Main loop of the program."""


    for grid in grid_paths:
        
        # Set up the environment
        env = Environment(grid, no_gui, sigma=sigma, target_fps=fps,
                          agent_start_pos=start_pos,
                          random_seed=random_seed,
                          reward_fn=reward_fn)
        
        # 0. Loading the grid
        grid_cells = Grid.load_grid(grid).cells

        # Compute BFS distances for reward shaping
        dist = compute_bfs_distances(grid_cells)
        
        # 1. Initialisation of policy, state action pairs and returns
        agent = McOnPolicyAgent(epsilon, epsilon_decay, epsilon_min)
        agent.create_state_action_space(grid_cells)
        indx_position = agent.state_action_indexer

        # 2. Initialisation of variables for tracking success rate and average steps every 100 episodes
        window_successes = 0
        window_steps = []
        episode_rewards = []
        episode_successes = []

        # 3. Initialisation of variables for stopping criterion
        prev_greedy_policy = None
        stable_count = 0

        for _  in trange(episodes):
            # Always reset the environment to initial state
            initial_pos = env.reset()
            state = initial_pos

            # generate a lookup_table of visits in an episode
            look_up = agent.look_up_first_visited()

            # generate an episode 
            states = []
            actions_taken = []
            rewards = []

            # 2. Episode generation
            for step in range(iters):

                state_idx = indx_position[state]

                action = agent.take_action(state)
                new_state, reward, terminated, info = env.step(action)
                actual_action = info['actual_action']
                # reward = shaped_reward(reward, state, new_state, terminated, dist, shaping_weight) Uncoment it to add reward shaping

                key = (state_idx, actual_action)

                # due to stochasticity, we only train the model on the legal behaviour
                # we only store legal events
                if key in look_up: 
                    # update the lookup; updates when the value was not yet updated
                    if look_up[key] == -1:
                        look_up[key] = len(rewards)

                    states.append(state)
                    actions_taken.append(actual_action)
                    rewards.append(reward)

                # new state after a step
                state = new_state

                if terminated:
                    window_successes += 1
                    window_steps.append(step)
                    episode_successes.append(1)
                    break

            else:
                episode_successes.append(0)

            episode_rewards.append(sum(rewards))
                
                # 3. evaluating episode and updating the agent
                # number of steps

            g = 0
            steps_taken = len(rewards)
            for i in range(steps_taken-1, -1, -1): # we compute the return backwards as it is easier for its evaluation
                g = delta * g + rewards[i]
                state_i = states[i]
                action_i = actions_taken[i]
                state_i_idx = indx_position[state_i]
                key = (state_i_idx, action_i)


                # update only if we encounter a first visit pair
                # state_i: tuple[int, int]
                # g: float
                # action_i: int
                if look_up[key] == i: 
                    agent.update(state_i, g, action_i)

            agent.decay_epsilon()

            # Stopping criterion: greedy policy unchanged for `patience` consecutive episodes
            current_greedy_policy = agent.get_greedy_action()
            if current_greedy_policy == prev_greedy_policy:
                stable_count += 1
            else:
                stable_count = 0
            prev_greedy_policy = current_greedy_policy
 
            if stable_count >= patience:
                print(f"Policy stable for {patience} consecutive episodes. Stopping at episode {_ + 1}.")
                break

            # get the summary every 100 episodes 
            episode_num = _ + 1
            if episode_num % 500 == 0:
                success_rate = 100 * window_successes / 500
                avg_steps = int(np.mean(window_steps)) if window_steps else 0 # prevents negative value when there are no successes
                print(f"Episode {episode_num:5d}/{episodes} | "
                      f"Epsilon: {agent.epsilon:.3f} | "
                      f"Success rate: {success_rate:.0f}% | "
                      f"Avg steps (successes): {avg_steps}")
                window_successes = 0
                window_steps = []


        # --- Metrics ---
        convergence_ep = _ + 1 if stable_count >= patience else -1

        optimality = compute_optimality_ratio(
            env, agent, initial_pos, dist,
            n_eval_episodes=10, max_steps=iters
        )

        visited_states, total_states = extract_visited_states(agent)

        print_metrics_summary("MC on-policy", grid.stem, convergence_ep,
                              optimality, visited_states, total_states)

        results_dir = Path("results")
        plot_learning_curve(
            episode_rewards, episode_successes,
            title=f"MC on-policy | {grid.stem} | eps={epsilon} delta={delta} sigma={sigma}",
            save_path=results_dir / f"mc_{grid.stem}_learning_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            convergence_ep=convergence_ep,
        )

        # Evaluate the agent
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
         args.random_seed, start_pos, args.episodes, args.delta, args.epsilon,
         args.epsilon_decay, args.epsilon_min, args.patience, args.shaping_weight)
