
Welcome to Data Intelligence Challenge-2AMC15!
This is the repository containing the challenge environment code.

## Quickstart

1. Create a virtual environment for this course with Python >= 3.10. Using conda, you can do: `conda create -n dic2025 python=3.11`. Use `conda activate dic2025` to activate it and `conda deactivate` to deactivate it.
2. Clone this repository: `git clone https://github.com/RL-In-Practice/2AMC15-2026.git`.
3. Install the required packages: `pip install -r requirements.txt`.
4. Run training with your chosen agent:

```bash
python train.py VI    grid_configs/A1_grid.npy --no_gui
python train.py MC    grid_configs/A1_grid.npy --no_gui
python train.py SARSA grid_configs/A1_grid.npy --no_gui
```

## `train.py` usage

`train.py` is the unified training entry point. The first positional argument selects the agent (`VI`, `MC`, or `SARSA`); all remaining arguments are passed through to the corresponding trainer.

```bash
usage: train.py [-h] [--no_gui] [--sigma SIGMA] [--fps FPS] [--iter ITER]
                [--random_seed RANDOM_SEED] [--start_pos START_POS]
                [--gamma GAMMA] [--shaping_weight SHAPING_WEIGHT]
                [--episodes EPISODES] [--epsilon EPSILON]
                [--epsilon_decay EPSILON_DECAY] [--epsilon_min EPSILON_MIN]
                [--patience PATIENCE] [--alpha ALPHA]
                {VI,MC,SARSA} GRID [GRID ...]
```

| Argument | Type | Default | Agents | Description |
|---|---|---|---|---|
| `agent` | `{VI,MC,SARSA}` | — | all | Which agent to train |
| `GRID` | path | — | all | Path(s) to `.npy` grid file(s) |
| `--no_gui` | flag | off | all | Disable rendering for faster training |
| `--sigma` | float | 0.1 | all | Environment stochasticity (0 = deterministic) |
| `--fps` | int | 30 | all | Render frame rate (ignored with `--no_gui`) |
| `--iter` | int | 1000 | all | Max steps per episode |
| `--random_seed` | int | 27 | all | Random seed |
| `--start_pos` | str | None | all | Start position as `col,row` (e.g. `1,12`) |
| `--gamma` | float | 0.9 | all | Discount factor |
| `--shaping_weight` | float | 0.0 | all | BFS reward shaping scale (0 = disabled) |
| `--episodes` | int | 1000 | MC, SARSA | Number of training episodes |
| `--epsilon` | float | 0.1 | MC, SARSA | Initial exploration rate |
| `--epsilon_decay` | float | 1.0 | MC, SARSA | Multiplicative epsilon decay per episode |
| `--epsilon_min` | float | 0.0 | MC, SARSA | Minimum epsilon value |
| `--patience` | int | 100 | MC, SARSA | Episodes of stable policy before early stopping |
| `--alpha` | float | 0.1 | SARSA | Learning rate |

### Examples

```bash
# Value Iteration with reward shaping
python train.py VI grid_configs/A1_grid.npy --no_gui --gamma 0.95 --shaping_weight 1.0

# Monte Carlo with epsilon decay
python train.py MC grid_configs/A1_grid.npy --no_gui --episodes 2000 --epsilon 0.2 --epsilon_decay 0.995

# SARSA with custom hyperparameters
python train.py SARSA grid_configs/A1_grid.npy --no_gui --alpha 0.1 --gamma 0.95 --epsilon 0.05 --episodes 1000
```

## Code guide

The code is made up of 2 modules:

1. `agents`
2. `world`

### The `agents` module

The `agents` module contains the `BaseAgent` class and the implemented RL agents:

| File | Agent |
|---|---|
| `value_iteration_agent.py` | Value Iteration (offline, model-based) |
| `mc_on_policy_agent.py` | Monte Carlo on-policy (first-visit) |
| `sarsa_agent.py` | SARSA (on-policy TD) |
| `random_agent.py` | Random baseline |
| `null_agent.py` | Null baseline (always action 0) |

The `BaseAgent` is an abstract class and all RL agents for DIC must inherit from/implement it.
If you know/understand class inheritance, skip the following section:

#### `BaseAgent` as an abstract class
Here you can find an explanation about abstract classes [Geeks for Geeks](https://www.geeksforgeeks.org/abstract-classes-in-python/).

Think of this like how all models in PyTorch start like 

```python
class NewModel(nn.Module):
    def __init__(self):
        super().__init__()
    ...
```

In this case, `NewModel` inherits from `nn.Module`, which gives it the ability to do back propagation, store parameters, etc. without you having to manually code that every time.
It also ensures that every class that inherits from `nn.Module` contains _at least_ the `forward()` method, which allows a forward pass to actually happen.

In the case of your RL agent, inheriting from `BaseAgent` guarantees that your agent implements `update()` and `take_action()`.
This ensures that no matter what RL agent you make and however you code it, the environment and training code can always interact with it in the same way.
Check out the benchmark agents to see examples.

### The `world` module

The world module contains:
1. `grid_creator.py`
2. `environment.py`
3. `grid.py`
4. `gui.py`

#### Grid creator
Run this file to create new grids.

```bash
$ python grid_creator.py
```

This will start up a web server where you create new grids, of different sizes with various elements arrangements.
To view the grid creator itself, go to `127.0.0.1:5000`.
All levels will be saved to the `grid_configs/` directory.


#### The Environment

The `Environment` is very important because it contains everything we hold dear, including ourselves [^1].
It is also the name of the class which our RL agent will act within. Most of the action happens in there.

The main interaction with `Environment` is through the methods:

- `Environment()` to initialize the environment
- `reset()` to reset the environment
- `step()` to actually take a time step with the environment
- `Environment().evaluate_agent()` to evaluate the agent after training.

[^1]: In case you missed it, this sentence is a joke. Please do not write all your code in the `Environment` class.

#### The Grid

The `Grid` class is the the actual representation of the world on which the agent moves. It is a 2D Numpy array.

#### The GUI

The Graphical User Interface provides a way for you to actually see what the RL agent is doing.
While performant and written using PyGame, it is still about 1300x slower than not running a GUI.
Because of this, we recommend using it only while testing/debugging and not while training.
