"""Path Visualizer.

This script is used to visualize the path of the agents in the environment.

"""
import sys
from pathlib import Path as _Path

if __package__ in (None, ""):
    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from colorcet import bmw, glasbey_hv
from PIL import Image
from PIL import ImageDraw
import numpy as np

from world import GUI

def draw_base_image(cells: np.ndarray,
                    scalar: int,
                    image_size: tuple[int, int],) -> Image.Image:
    """Draws the base image containing the grid and objects on the grid.

    Args:
        cells: The cell array underlying the grid representation of the
            environment.
        scalar: How much to scale the original grid by in the output image.
        image_size: Output image size.

    Returns:
        An RGBA image with the grid on it.
    """
    grid_size = cells.shape
    base_image = Image.new(mode="RGBA", size=image_size,
                           color=(255, 255, 255, 255))
    draw = ImageDraw.ImageDraw(base_image)
    for row in range(grid_size[1]):
        y = (row * scalar) + 1
        for col in range(grid_size[0]):
            x = (col * scalar) + 1
            val = cells[col, row]
            color = GUI.CELL_COLORS[val]

            draw.rectangle((x, y, x + scalar, y + scalar),
                           color,
                           outline=(255, 255, 255))

    return base_image


def draw_starting_square(starting_square: tuple[int, int],
                         grid_scalar: int,
                         image_size: tuple[int, int]) -> Image.Image:
    """Draws the starting square as a yellow square."""
    square_image = Image.new(mode="RGBA", size=image_size,
                             color=(255, 255, 255, 0))
    draw = ImageDraw.ImageDraw(square_image)
    x = (starting_square[0] * grid_scalar) + 1
    y = (starting_square[1] * grid_scalar) + 1

    draw.rectangle((x, y, x + grid_scalar, y + grid_scalar),
                   (242, 211, 82),
                   outline=(255, 255, 255))

    return square_image


def draw_freq_image(agent_path: list[tuple[int, int]],
                    grid_shape: tuple[int, int],
                    grid_scalar: int,
                    freq_scalar: int,
                    image_size: tuple[int, int]) -> Image.Image:
    """Draws the cell visit frequency image.

    Args:
        agent_path: The path that each agent took through the environment.
        grid_shape: The actual shape of the grid.
        grid_scalar: The size of each grid cell to draw. For example, a value of
            30 would result in an image where each grid cell is 30x30 px.
        freq_scalar: The size of each frequency color square to draw. For
            example, in a 30x30 grid square, if we set the freq_scalar to 20,
            then the freq color square we draw is 20x20 centered in the middle
            of the grid square.
        image_size: The size of the final image to draw.

    Returns:
        An image that is transparent except where the frequency squares are.
    """
    # Create the frequency grid array to figure out how often a cell if
    # traversed
    freq_grid = np.zeros(grid_shape, dtype=float)
    for pos in agent_path:
        freq_grid[pos] += 1.

    # Normalize by the max value to 0-255
    freq_grid /= np.max(freq_grid)
    freq_grid *= 255.
    freq_grid = freq_grid.astype(int)

    cell_offset = (grid_scalar - freq_scalar) // 2

    freq_image = Image.new(mode="RGBA", size=image_size,
                           color=(255, 255, 255, 0))
    draw = ImageDraw.ImageDraw(freq_image)

    for row in range(grid_shape[1]):
        y = (row * grid_scalar) + 1 + cell_offset
        for col in range(grid_shape[0]):
            x = (col * grid_scalar) + 1 + cell_offset
            val = freq_grid[col, row]
            if val == 0:
                # Don't draw anything if the cells has never been traversed.
                continue
            try:
                # minus because we want to start from white.
                color = bmw[-val]
            except IndexError:
                # There is no chance the value is < 1 here, but just in case.
                color = bmw[0]

            draw.rectangle((x, y,
                            x + freq_scalar, y + freq_scalar),
                           color)
    return freq_image


def draw_path(agent_path: list[tuple[int, int]],
              grid_scalar: int,
              line_width: int,
              line_color: tuple[int, int, int],
              image_size: tuple[int, int]) -> Image.Image:
    """Draws the path of each agent on the grid.

    Args:
        agent_path: The path that the agent took through the environment.
        grid_scalar: The size of each grid cell to draw. For example, a value of
            30 would result in an image where each grid cell is 30x30 px.
        line_width: The width of the path line to draw.
        line_color: The color of the path line to draw.
        image_size: The size of the final image to draw.

    Returns:
         An image that is transparent except where the line paths are.
    """
    path_image = Image.new(mode="RGBA", size=image_size,
                           color=(255, 255, 255, 0))
    draw = ImageDraw.ImageDraw(path_image)

    for i in range(len(agent_path) - 1):
        start = agent_path[i]
        end = agent_path[i + 1]
        offset = grid_scalar // 2
        start_x = (start[0] * grid_scalar) + offset
        start_y = (start[1] * grid_scalar) + offset
        end_x = (end[0] * grid_scalar) + offset
        end_y = (end[1] * grid_scalar) + offset
        draw.line((start_x, start_y, end_x, end_y),
                  fill=line_color, width=line_width)

    return path_image


def float_rgb_to_int(rgb: tuple[float, float, float]) -> tuple[int, int, int]:
    """Converts an RGB tuple of floats to an RGB tuple of ints.

    Args:
        rgb: The RGB tuple of floats.

    Returns:
        The RGB tuple of ints.
    """
    return tuple(int(c * 255) for c in rgb)


def visualize_path(grid_cells: np.ndarray,
                   agent_path: list[tuple[int, int]],
                   show_frequency: bool = False) \
        -> Image.Image:
    """Visualizes the path of (multiple) agents through the environment.

    Args:
        grid_cells: The grid cells that are underlying the Grid object.
        agent_paths: A list of tuples containing the x and y coordinates of
            the agent's path.

    Returns:
        An image showing the grid and the frequency of the agent
        traversing each position on the grid.
    """
    grid_size = grid_cells.shape
    scalar = 30
    freq_scalar = 20
    image_size = tuple((g * scalar) + 2 for g in grid_size)

    base_image = draw_base_image(grid_cells, scalar, image_size)
    starting_square_img = draw_starting_square(agent_path[0], scalar, image_size)
    img = Image.alpha_composite(base_image, starting_square_img)

    if show_frequency:
        overlay = draw_freq_image(agent_path, grid_size, scalar, freq_scalar, image_size)
    else:
        overlay = draw_path(agent_path, scalar, 2,
                            float_rgb_to_int(glasbey_hv[0]), image_size)

    return Image.alpha_composite(img, overlay)


def _custom_reward(grid, agent_pos):
    match grid[agent_pos]:
        case 0:
            return -0.1
        case 1 | 2:
            return -1.0
        case 3:
            return 10.0
        case _:
            raise ValueError(f"Unexpected grid value {grid[agent_pos]} at {agent_pos}")


def _build_agent(algo: str, state_dim: int, hidden_size: int):
    if algo == "dqn":
        from agents.dqn import DQNAgent
        return DQNAgent(state_dim=state_dim, n_actions=8, hidden_size=hidden_size)
    from agents.ppo_agent import PPOAgent
    return PPOAgent(state_dim=state_dim, n_actions=8, hidden_size=hidden_size)


def _main():
    import argparse
    from pathlib import Path

    from world.environment import Environment
    from world.continuous_env import ContinuousEnv

    p = argparse.ArgumentParser(
        description="Load a trained checkpoint and render its path as a "
                     "single-episode trajectory and a multi-episode "
                     "visit-frequency heatmap.")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a saved .pt checkpoint (from agent.save()).")
    p.add_argument("--algo", choices=["dqn", "ppo"], required=True)
    p.add_argument("--grid", type=Path, default=Path("grid_configs/A1_grid.npy"))
    p.add_argument("--state_mode", choices=["gps", "raycasting", "both"], default="gps")
    p.add_argument("--max_range", type=int, default=None)
    p.add_argument("--hidden_size", type=int, default=256,
                   help="Must match the hidden_size used to train the checkpoint.")
    p.add_argument("--start_pos", type=str, default=None,
                   help="Fixed start 'row,col' (e.g. 1,12). Default: grid start cell or random.")
    p.add_argument("--sigma", type=float, default=0.0)
    p.add_argument("--episodes", type=int, default=20,
                   help="Greedy episodes to run; the frequency heatmap aggregates all of them.")
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=Path, default=Path("results/figures"))
    p.add_argument("--tag", type=str, default=None,
                   help="Output filename prefix. Defaults to the checkpoint's stem.")
    args = p.parse_args()

    start_pos = tuple(int(x) for x in args.start_pos.split(",")) if args.start_pos else None

    base_env = Environment(args.grid, no_gui=True, sigma=args.sigma,
                           agent_start_pos=start_pos, reward_fn=_custom_reward,
                           random_seed=args.seed)
    cont_env = ContinuousEnv(base_env, mode=args.state_mode, max_range=args.max_range)

    agent = _build_agent(args.algo, cont_env.state_dim, args.hidden_size)
    agent.load(args.checkpoint)
    if hasattr(agent, "training_mode"):
        agent.training_mode = False

    first_path = None
    all_positions: list[tuple[int, int]] = []
    successes = 0

    for ep in range(args.episodes):
        state = cont_env.reset()
        path = [base_env.agent_pos]
        for _ in range(args.max_steps):
            action = agent.take_action(state)
            state, _, done, _ = cont_env.step(action)
            path.append(base_env.agent_pos)
            if done:
                successes += 1
                break
        all_positions.extend(path)
        if ep == 0:
            first_path = path

    print(f"Greedy eval: {successes}/{args.episodes} episodes reached the target.")

    tag = args.tag or Path(args.checkpoint).stem
    args.output_dir.mkdir(parents=True, exist_ok=True)

    path_out = args.output_dir / f"{tag}_path.png"
    freq_out = args.output_dir / f"{tag}_freq.png"
    visualize_path(base_env.grid, first_path).save(path_out)
    visualize_path(base_env.grid, all_positions, show_frequency=True).save(freq_out)

    print(f"Saved: {path_out}")
    print(f"Saved: {freq_out}")


if __name__ == "__main__":
    _main()