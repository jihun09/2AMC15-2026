#!/bin/bash
# =============================================================================
# SARSA Experiment Runner
# =============================================================================
# Runs all required experiment combinations:
#   - 2 grids: A1_grid, example_grid
#   - 2 discount factors (gamma): 0.6, 0.9
#   - 2 stochasticity values (sigma): 0.02, 0.5
#   - 2 learning rates (alpha): 0.01, 0.1
#   - 2 epsilon values: 0.1, 0.3
#
# Fixed: start_pos=1,12 on A1_grid (as required by assignment)
#        start_pos=1,1 on example_grid (pick a valid empty cell)
#
# Each experiment produces:
#   - results/<name>_learning_curve.png  (training progress)
#   - results/<timestamp>.png            (path visualization)
#   - results/<timestamp>.txt            (evaluation stats)
# =============================================================================

# --- BASELINE: Random agent on both grids (for comparison) ---

echo "========== BASELINE: Random agent =========="
python train_sarsa.py grid_configs/A1_grid.npy \
    --start_pos 1,12 --no_gui --episodes 1 --max_steps 1000 \
    --eval_steps 1000 --eval_random \
    --name baseline_random_A1grid

python train_sarsa.py grid_configs/example_grid.npy \
    --start_pos 1,1 --no_gui --episodes 1 --max_steps 1000 \
    --eval_steps 1000 --eval_random \
    --name baseline_random_example

# =============================================================================
# EXPERIMENT SET 1: Varying learning rate (alpha)
# Fixed: gamma=0.9, epsilon=0.1, sigma=0.02
# =============================================================================

echo "========== EXP 1: Varying alpha =========="

# alpha=0.1
python train_sarsa.py grid_configs/A1_grid.npy \
    --start_pos 1,12 --no_gui \
    --alpha 0.1 --gamma 0.9 --epsilon 0.1 --sigma 0.02 \
    --episodes 500 --max_steps 1000 --eval_steps 500 \
    --name exp_A1_alpha0.1_gamma0.9_eps0.1_sigma0.02

python train_sarsa.py grid_configs/example_grid.npy \
    --start_pos 1,1 --no_gui \
    --alpha 0.1 --gamma 0.9 --epsilon 0.1 --sigma 0.02 \
    --episodes 500 --max_steps 1000 --eval_steps 500 \
    --name exp_example_alpha0.1_gamma0.9_eps0.1_sigma0.02

# alpha=0.01
python train_sarsa.py grid_configs/A1_grid.npy \
    --start_pos 1,12 --no_gui \
    --alpha 0.01 --gamma 0.9 --epsilon 0.1 --sigma 0.02 \
    --episodes 500 --max_steps 1000 --eval_steps 500 \
    --name exp_A1_alpha0.01_gamma0.9_eps0.1_sigma0.02

python train_sarsa.py grid_configs/example_grid.npy \
    --start_pos 1,1 --no_gui \
    --alpha 0.01 --gamma 0.9 --epsilon 0.1 --sigma 0.02 \
    --episodes 500 --max_steps 1000 --eval_steps 500 \
    --name exp_example_alpha0.01_gamma0.9_eps0.1_sigma0.02

# =============================================================================
# EXPERIMENT SET 2: Varying discount factor (gamma)
# Fixed: alpha=0.1, epsilon=0.1, sigma=0.02
# =============================================================================

echo "========== EXP 2: Varying gamma =========="

# gamma=0.9 (already done above in exp set 1)

# gamma=0.6
python train_sarsa.py grid_configs/A1_grid.npy \
    --start_pos 1,12 --no_gui \
    --alpha 0.1 --gamma 0.6 --epsilon 0.1 --sigma 0.02 \
    --episodes 500 --max_steps 1000 --eval_steps 500 \
    --name exp_A1_alpha0.1_gamma0.6_eps0.1_sigma0.02

python train_sarsa.py grid_configs/example_grid.npy \
    --start_pos 1,1 --no_gui \
    --alpha 0.1 --gamma 0.6 --epsilon 0.1 --sigma 0.02 \
    --episodes 500 --max_steps 1000 --eval_steps 500 \
    --name exp_example_alpha0.1_gamma0.6_eps0.1_sigma0.02

# =============================================================================
# EXPERIMENT SET 3: Varying stochasticity (sigma)
# Fixed: alpha=0.1, gamma=0.9, epsilon=0.1
# =============================================================================

echo "========== EXP 3: Varying sigma =========="

# sigma=0.02 (already done above in exp set 1)

# sigma=0.5
python train_sarsa.py grid_configs/A1_grid.npy \
    --start_pos 1,12 --no_gui \
    --alpha 0.1 --gamma 0.9 --epsilon 0.1 --sigma 0.5 \
    --episodes 500 --max_steps 1000 --eval_steps 500 \
    --name exp_A1_alpha0.1_gamma0.9_eps0.1_sigma0.5

python train_sarsa.py grid_configs/example_grid.npy \
    --start_pos 1,1 --no_gui \
    --alpha 0.1 --gamma 0.9 --epsilon 0.1 --sigma 0.5 \
    --episodes 500 --max_steps 1000 --eval_steps 500 \
    --name exp_example_alpha0.1_gamma0.9_eps0.1_sigma0.5

# =============================================================================
# EXPERIMENT SET 4: Varying epsilon
# Fixed: alpha=0.1, gamma=0.9, sigma=0.02
# =============================================================================

echo "========== EXP 4: Varying epsilon =========="

# epsilon=0.1 (already done above in exp set 1)

# epsilon=0.3
python train_sarsa.py grid_configs/A1_grid.npy \
    --start_pos 1,12 --no_gui \
    --alpha 0.1 --gamma 0.9 --epsilon 0.3 --sigma 0.02 \
    --episodes 500 --max_steps 1000 --eval_steps 500 \
    --name exp_A1_alpha0.1_gamma0.9_eps0.3_sigma0.02

python train_sarsa.py grid_configs/example_grid.npy \
    --start_pos 1,1 --no_gui \
    --alpha 0.1 --gamma 0.9 --epsilon 0.3 --sigma 0.02 \
    --episodes 500 --max_steps 1000 --eval_steps 500 \
    --name exp_example_alpha0.1_gamma0.9_eps0.3_sigma0.02

# =============================================================================
# EXPERIMENT SET 5: High stochasticity + high gamma (stress test)
# Shows SARSA's conservative behavior under uncertainty
# =============================================================================

echo "========== EXP 5: Stress test (high sigma + high gamma) =========="

python train_sarsa.py grid_configs/A1_grid.npy \
    --start_pos 1,12 --no_gui \
    --alpha 0.1 --gamma 0.9 --epsilon 0.1 --sigma 0.5 \
    --episodes 1000 --max_steps 2000 --eval_steps 500 \
    --name exp_A1_stress_sigma0.5_episodes1000

python train_sarsa.py grid_configs/example_grid.npy \
    --start_pos 1,1 --no_gui \
    --alpha 0.1 --gamma 0.9 --epsilon 0.1 --sigma 0.5 \
    --episodes 1000 --max_steps 2000 --eval_steps 500 \
    --name exp_example_stress_sigma0.5_episodes1000

echo "========== ALL EXPERIMENTS COMPLETE =========="
echo "Results saved in results/ directory"
