"""
Train your RL Agent in this file.

Usage:
    python train.py VI   grid_configs/A1_grid.npy --no_gui --gamma 0.9
    python train.py MC   grid_configs/A1_grid.npy --no_gui --episodes 1000
    python train.py SARSA grid_configs/A1_grid.npy --no_gui --alpha 0.1
"""

from argparse import ArgumentParser
from pathlib import Path


def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument("agent", choices=["VI", "MC", "SARSA"],
                   help="Which agent to train.")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use. There can be more than one.")

    # ── shared ────────────────────────────────────────────────────────────────
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster.")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second to render at. Only used if no_gui is not set.")
    p.add_argument("--iter", type=int, default=1000,
                   help="Max steps per episode.")
    p.add_argument("--random_seed", type=int, default=27,
                   help="Random seed value for the environment.")
    p.add_argument("--start_pos", type=str, default=None,
                   help="Agent start position as col,row (e.g. 2,3).")
    p.add_argument("--gamma", type=float, default=0.9,
                   help="Discount factor. (VI, SARSA; also maps to --delta for MC)")
    p.add_argument("--shaping_weight", type=float, default=0.0,
                   help="Scaling factor for BFS potential-based reward shaping (0 to disable).")

    # ── MC and SARSA ───────────────────────────────────────────────────────────
    p.add_argument("--episodes", type=int, default=1000,
                   help="Number of training episodes. (MC, SARSA)")
    p.add_argument("--epsilon", type=float, default=0.1,
                   help="Initial exploration rate. (MC, SARSA)")
    p.add_argument("--epsilon_decay", type=float, default=1.0,
                   help="Multiplicative decay applied to epsilon after each episode. (MC, SARSA)")
    p.add_argument("--epsilon_min", type=float, default=0.0,
                   help="Minimum value epsilon can decay to. (MC, SARSA)")
    p.add_argument("--patience", type=int, default=100,
                   help="Stop if greedy policy is stable for this many consecutive episodes. (MC, SARSA)")

    # ── SARSA only ─────────────────────────────────────────────────────────────
    p.add_argument("--alpha", type=float, default=0.1,
                   help="Learning rate. (SARSA only)")

    return p.parse_args()


def main():
    args = parse_args()

    start_pos = None
    if args.start_pos is not None:
        parts = args.start_pos.split(",")
        start_pos = (int(parts[0]), int(parts[1]))

    if args.agent == "VI":
        from train_vi import main as vi_main
        vi_main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma,
                args.random_seed, start_pos, args.gamma, args.shaping_weight)

    elif args.agent == "MC":
        from train_mc_on_policy import main as mc_main
        mc_main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma,
                args.random_seed, start_pos, args.episodes, args.gamma,
                args.epsilon, args.epsilon_decay, args.epsilon_min,
                args.patience, args.shaping_weight)

    elif args.agent == "SARSA":
        from sarsa import main as sarsa_main
        sarsa_main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma,
                   args.random_seed, start_pos, args.episodes, args.gamma,
                   args.alpha, args.epsilon, args.epsilon_decay,
                   args.epsilon_min, args.patience, args.shaping_weight)


if __name__ == "__main__":
    main()
