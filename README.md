# 2AMC15 Assignment 2: Deep RL Delivery Robot

Solving the Assignment 1 delivery task with **Deep RL** in a continuous-state environment
(the action space stays discrete). Two methods, both implemented **from scratch** in
PyTorch, no RL libraries:

- **DQN**: baseline (Mnih et al., 2015)
- **PPO**: main method (Schulman et al., 2017)

All results are reported as **mean ± std over multiple seeds**.

## Quickstart

1. Create a virtual environment with Python >= 3.10 (e.g. `conda create -n a2 python=3.11 && conda activate a2`).
2. Install dependencies: `pip install -r requirements.txt`.
3. Check it runs:

```bash
python train_dqn.py --grid grid_configs/A1_grid.npy --no_gui
python train_ppo.py --grid grid_configs/A1_grid.npy --no_gui
```

## Demo (under 2 hours)

A minimal end-to-end demo, trains DQN and PPO for one seed on the main grid, prints the
evaluation metrics, and writes a learning curve + path figure to `results/`:

```bash
python train_dqn.py --grid grid_configs/A1_grid.npy --no_gui --seed 0 --sigma 0.1 --episodes 500  --hidden_size 256 --lr 5e-4
python train_ppo.py --grid grid_configs/A1_grid.npy --no_gui --seed 0 --sigma 0.1 --episodes 3000 --hidden_size 256 --lr 5e-4
```

Both run comfortably within the 2-hour cap on a CPU. The full multi-seed results reported in
the paper take longer (reproduce them with `bash run_experiments.sh`).

## Environment & state representation

`world/environment.py` is the base grid world: a 2D NumPy array (0 = empty, 1 = wall,
2 = obstacle, 3 = target, 4 = start). For this assignment it uses **8 actions** (4 cardinal
+ 4 diagonal) with corner-cutting prevention, and a `sigma` parameter for stochasticity,
with probability `sigma` the chosen action is replaced by a random one (`sigma = 0` is
deterministic). The reward is supplied via `reward_fn` (−0.1 per step, −0.5 for hitting a
wall/obstacle, +10 for reaching the target). Its interface is `Environment(...)`,
`reset(**kwargs) -> (row, col)`, and `step(action) -> (agent_pos, reward, done, info)`.

`world/continuous_env.py` (`ContinuousEnv`) wraps it and turns the discrete `(row, col)`
position into a continuous observation vector, selected with `--state_mode`:

- **`gps`** (dim 2): normalised `(row, col)` coordinates
- **`raycasting`** (dim 8): distance to the nearest blocking cell in 8 directions
- **`both`** (dim 10): gps + raycasting concatenated

Per the assignment's rule against shortcut features, the state never includes
distance-to-target or a target-vs-obstacle ID: the raycasting sensor reports **only
distances** and cannot tell walls, obstacles and the target apart (no target flag). Always
use `cont_env.state_dim` as the network input size; `max_range` optionally caps the ray length.

## Metrics

Computed in `eval_metrics.py` at the end of each run (100 greedy evaluation episodes) and
appended as one row to `results/a2_summary_metrics.csv`:

- **Success rate**: fraction of eval episodes that reach the target within `max_steps`
  (`1.0` = always reaches it). The primary measure of task performance.
- **Path efficiency**: BFS-optimal steps ÷ actual steps, averaged over *successful*
  episodes (`1.0` = shortest possible path; lower = the agent takes detours).
- **Steps to target**: mean steps to reach the target over successful episodes
  (lower = faster); `optimal_steps` is the BFS shortest path, reported for reference.

During training, `*_training.csv` logs per episode the **cumulative reward**, **steps**, and
a **success** flag: these feed the convergence curves in `plot_results.py`.
`aggregate_metrics.py` groups the summary CSV by configuration and prints mean ± std over seeds.

## Parameters you can vary

Pass these on the command line to `train_dqn.py` and `train_ppo.py`. For options shown in
parentheses, write **one** of them (e.g. `--state_mode both`); the rest take a number.

**Shared by both scripts**

| Flag | Values / type | Default | Meaning |
|---|---|---|---|
| `--grid` | path to a `.npy` | `grid_configs/A1_grid.npy` | Map (others in `grid_configs/`) |
| `--state_mode` | (`gps`, `raycasting`, `both`) | `gps` | State representation: gps=2, raycasting=8, both=10 |
| `--sigma` | float | DQN `0`, PPO `0.1` | Env stochasticity (`0` = deterministic) |
| `--seed` | int | `42` | Random seed (sets numpy + torch) |
| `--episodes` | int | DQN `1000`, PPO `2000` | Training episodes |
| `--lr` | float | DQN `1e-3`, PPO `3e-4` | Adam learning rate |
| `--gamma` | float | `0.99` | Discount factor |
| `--hidden_size` | int | `128` | Hidden layer width (two layers) |
| `--max_range` | int | full | Raycasting range in cells |
| `--start_pos` | `row,col` | `1,12` / grid cell | Fixed start position |
| `--max_steps` | int | `500` | Max steps per episode |
| `--no_gui` | flag | off | Disable rendering (use for training) |
| `--eval_episodes` | int | `10` | Episodes per periodic evaluation |
| `--summary_csv` | path | `results/a2_summary_metrics.csv` | Where the metrics row is appended |

**DQN only:** `--epsilon` `--epsilon_decay` `--epsilon_min` (ε-greedy schedule),
`--buffer_capacity` `--batch_size` `--warmup` `--target_update_freq` (replay + target net).

**PPO only:** `--clip_eps` (clip ε, 0.2), `--gae_lambda` (GAE λ, 0.95),
`--entropy_coef` (entropy bonus, set `0` to ablate), `--value_coef`, `--k_epochs`,
`--rollout_steps`, `--minibatch_size`.

Example (PPO, both-mode, σ=0.1, seed 0):

```bash
python train_ppo.py --grid grid_configs/A1_grid.npy --no_gui --state_mode both --sigma 0.1 --seed 0 --episodes 3000 --hidden_size 256 --lr 5e-4
```

## Experiments: to run train_dqn and train_ppo

Single run (one seed), each appends one row to `results/a2_summary_metrics.csv`:

```bash
python train_dqn.py --grid grid_configs/A1_grid.npy --no_gui --state_mode gps --sigma 0.1 --seed 0 --episodes 500  --hidden_size 256 --lr 5e-4
python train_ppo.py --grid grid_configs/A1_grid.npy --no_gui --state_mode gps --sigma 0.1 --seed 0 --episodes 3000 --hidden_size 256 --lr 5e-4
```

Multiple seeds (PowerShell):

```powershell
foreach ($s in 0,1,2,3,4) {
  python train_dqn.py --grid grid_configs/A1_grid.npy --no_gui --state_mode gps --sigma 0.1 --seed $s --episodes 500  --hidden_size 256 --lr 5e-4
  python train_ppo.py --grid grid_configs/A1_grid.npy --no_gui --state_mode gps --sigma 0.1 --seed $s --episodes 3000 --hidden_size 256 --lr 5e-4
}
```

Multiple seeds (bash):

```bash
for s in 0 1 2 3 4; do
  python train_dqn.py --grid grid_configs/A1_grid.npy --no_gui --state_mode gps --sigma 0.1 --seed $s --episodes 500  --hidden_size 256 --lr 5e-4
  python train_ppo.py --grid grid_configs/A1_grid.npy --no_gui --state_mode gps --sigma 0.1 --seed $s --episodes 3000 --hidden_size 256 --lr 5e-4
done
```

To vary the state representation or stochasticity, change `--state_mode` (`gps` / `raycasting`
/ `both`) and `--sigma`. Reproduce the full matrix at once with `bash run_experiments.sh`
(writes the summary CSV and prints the aggregated table). Re-print the table any time with
`python aggregate_metrics.py`.

## Plots: to visualize the plots

```bash
python plot_results.py                       # GPS runs
python plot_results.py --state_mode both      # GPS+raycasting runs
python plot_results.py --state_mode raycasting
```

- **DQN vs PPO:** the compact figure `fig1_convergence*.png` shows both side by side
  (left column = DQN, right = PPO; top row = reward, bottom = success rate). To get each
  one **separately**, use the single-panel files written alongside it:
  `fig1_dqn_reward.png`, `fig1_dqn_success.png`, `fig1_ppo_reward.png`, `fig1_ppo_success.png`.
- **gps vs raycasting vs both:** selected with `--state_mode`. GPS writes the base filenames;
  the other modes append a suffix so nothing is overwritten, e.g. `fig1_convergence_both.png`,
  `fig1_ppo_success_both.png`.
- The plotted configuration (σ, lr, hidden size) is set at the top of `plot_results.py`
  (`DQN_CFG` / `PPO_CFG`). **The runs you want to plot must match it** (default: σ=0.1,
  lr=5e-4, hidden=256), otherwise no files match and you get a `FileNotFoundError`.

## Code reference

**Agents** (`agents/`, all inherit the abstract `BaseAgent` and implement its
`take_action` / `select_action` and training hooks):

| File | Agent |
|---|---|
| `dqn.py` | DQN - Deep Q-Network (baseline) |
| `ppo_agent.py` | PPO - Proximal Policy Optimization (main method) |
| `random_agent.py`, `null_agent.py` | trivial baselines |

**World** (`world/`): `environment.py` (base grid), `continuous_env.py` (continuous-state
wrapper), `grid.py` (2D-array grid), `gui.py` (pygame renderer about 1300× slower than headless,
so use `--no_gui` while training), `grid_creator.py` (`python grid_creator.py` serves a grid
editor at `127.0.0.1:5000`, saving to `grid_configs/`).

## File map

| File | Purpose |
|---|---|
| `train_dqn.py`, `train_ppo.py` | Train one config; append its metrics row |
| `agents/dqn.py`, `agents/ppo_agent.py` | The two algorithms (from scratch) |
| `world/continuous_env.py` | Continuous-state wrapper (gps / raycasting / both) |
| `eval_metrics.py` | Policy evaluation (success rate, path efficiency, steps) |
| `aggregate_metrics.py` | Group the summary CSV by config; print mean ± std over seeds |
| `plot_results.py` | Convergence figures (mean ± std bands) |
| `merge_parts.py` | Fold parallel per-run CSVs into the main CSV (used by the runner) |
| `run_experiments.sh` | Reproduce everything |
