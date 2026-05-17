# SARSA Agent Implementation

## Overview

This implements a tabular SARSA (State-Action-Reward-State-Action) agent for the delivery robot grid world. SARSA is an on-policy TD(0) method that learns Q-values by following and updating the same ε-greedy policy.

## Files

| File | Description |
|------|-------------|
| `agents/sarsa_agent.py` | SARSA agent with Q-table, ε-greedy policy, and update logic |
| `train_sarsa.py` | Training script with multi-episode loop, learning curves, and evaluation |
| `run_experiments.sh` | Shell script that runs all required experiment combinations |

## How It Works

**SARSA update rule:**
```
Q(s, a) ← Q(s, a) + α * [r + γ * Q(s', a') - Q(s, a)]
```

**Training loop (per episode):**
1. Reset environment, get initial state `s`
2. Choose action `a` using ε-greedy policy
3. Take step → get next state `s'`, reward `r`, actual action executed
4. Choose next action `a'` from `s'` (on-policy)
5. Update Q(s, a) using the SARSA rule
6. Set `s = s'`, `a = a'`, repeat until terminal or max steps
7. Decay epsilon after each episode

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
pip install matplotlib

# Train and evaluate on A1_grid (required start position)
python train_sarsa.py grid_configs/A1_grid.npy \
    --start_pos 1,12 --no_gui \
    --alpha 0.1 --gamma 0.9 --epsilon 0.1 --sigma 0.02 \
    --episodes 500 --max_steps 1000 --eval_steps 500 --eval_random

# Run all experiments
bash run_experiments.sh
```

## Hyperparameters

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Learning rate | `--alpha` | 0.1 | How fast the agent updates Q-values |
| Discount factor | `--gamma` | 0.9 | How much future rewards are valued |
| Exploration rate | `--epsilon` | 0.1 | Probability of random action (ε-greedy) |
| Epsilon decay | `--epsilon_decay` | 0.995 | Multiplicative decay per episode |
| Epsilon minimum | `--epsilon_min` | 0.01 | Floor for epsilon |
| Stochasticity | `--sigma` | 0.02 | Probability environment ignores your action |
| Episodes | `--episodes` | 500 | Number of training episodes |
| Max steps | `--max_steps` | 1000 | Max steps per episode |
| Eval steps | `--eval_steps` | 200 | Max steps during evaluation |

## Experiments

The assignment requires testing across:
- **2 grids:** `A1_grid.npy` (start: 1,12) and `example_grid.npy` (start: 1,1)
- **2 discount factors:** γ = 0.6 and 0.9
- **2 stochasticity values:** σ = 0.02 and 0.5
- **2 learning rates:** α = 0.01 and 0.1
- **2 epsilon values:** ε = 0.1 and 0.3

Each experiment trains and evaluates on the same grid (Q-tables are grid-specific).

## Output

Results are saved in `results/`:
- `<name>_learning_curve.png` — reward and steps per episode over training
- `<timestamp>.png` — path visualization (blue lines showing agent's route)
- `<timestamp>.txt` — evaluation stats (steps, failed moves, targets reached)



## Key Design Decisions

- **On-policy (SARSA vs Q-learning):** SARSA updates using the action it actually takes next, making it conservative in stochastic environments. With high σ, it learns to avoid paths near walls since it accounts for accidental missteps.
- **Actual action used for updates:** The environment may execute a different action than intended (due to σ). We update Q-values based on `info["actual_action"]` — the action that was really taken.
- **Epsilon decay:** Starts with exploration, gradually shifts to exploitation as the Q-table converges.

## Potential Improvements

### Reward Function

| Change | Rationale |
|--------|-----------|
| Increase target reward (+10 → +100) | Stronger signal that propagates further back on large grids, helping with slow convergence |
| Harsher wall penalty (-5 → -10) | Stronger discouragement near obstacles, especially useful under high stochasticity |
| Distance-based shaping (e.g., small bonus for moving closer to target) | Speeds up learning on large grids by providing intermediate guidance, but risks biasing the policy |

### Algorithm Extensions

- **SARSA(λ) with eligibility traces:** Propagates reward information multiple steps back in a single update, significantly speeding up learning on long paths (like A1_grid's ~27-step optimal path).
- **Optimistic Q-value initialization:** Initialize Q-values to a positive number instead of 0. This naturally encourages exploration (every action looks worse than expected, so the agent tries new ones) without relying solely on ε-greedy.
- **Adaptive/decaying learning rate:** Decrease α over time (e.g., α = 1/N(s,a)) to guarantee convergence. Currently α is fixed, which means Q-values can keep oscillating slightly.
- **Double SARSA:** Maintain two Q-tables to reduce overestimation bias, similar to Double Q-learning but on-policy.

### Convergence Across Methods

With the same reward function, grid, and stochasticity, SARSA should converge to a policy close to the optimal one found by value iteration or Monte Carlo. Under high σ, SARSA's policy may be slightly more conservative (longer path but safer) because it's on-policy — it accounts for its own exploration noise in the Q-values. For a real delivery robot, this conservative behavior near obstacles is arguably a feature rather than a limitation.
