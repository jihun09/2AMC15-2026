#!/bin/bash
# =============================================================================
# SARSA: 6 Experimental Setups (for cross-algorithm comparison)
# =============================================================================
# These match configs/experiments.json — same setups will be run for VI and MC
# by teammates. Results go to results/raw_results.txt
#
# Factors tested:
#   - Stochasticity: σ=0.02 vs σ=0.5
#   - Discount factor: γ=0.95 vs γ=0.6
#   - Reward shaping: BFS shaped vs unshaped
#
# Tuned SARSA hyperparameters (from configs/sarsa_best.json):
#   α=0.1, ε=0.05 (fixed), episodes=1000, max_steps=1000
# =============================================================================

set -e

GRID="grid_configs/A1_grid.npy"
START_POS="1,12"
ALPHA=0.1
EPSILON=0.05
EPISODES=1000
MAX_STEPS=1000
EVAL_STEPS=500
EVAL_EPISODES=20
SEED=0

RESULTS_FILE="results/raw_results.txt"
mkdir -p results

echo "================================================================" > "$RESULTS_FILE"
echo "SARSA — 6 Experimental Setups — $(date)" >> "$RESULTS_FILE"
echo "Grid: $GRID | Start: $START_POS" >> "$RESULTS_FILE"
echo "Tuned: α=$ALPHA, ε=$EPSILON (fixed)" >> "$RESULTS_FILE"
echo "================================================================" >> "$RESULTS_FILE"

run_exp() {
    local label="$1"
    shift
    echo "" >> "$RESULTS_FILE"
    echo "=== $label ===" >> "$RESULTS_FILE"
    echo "python train_sarsa.py $*" >> "$RESULTS_FILE"
    echo "---" >> "$RESULTS_FILE"
    python train_sarsa.py "$@" 2>&1 | tee -a "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
}

# =============================================================================
# EXP 1: Stochasticity — low noise (σ=0.02)
# =============================================================================
echo ">>> Experiment 1: σ=0.02 (baseline)"

run_exp "EXP1: sigma=0.02 (baseline)" "$GRID" \
    --start_pos "$START_POS" --no_gui \
    --alpha $ALPHA --gamma 0.95 --epsilon $EPSILON --sigma 0.02 \
    --epsilon_decay_mode fixed --no_shaping \
    --episodes $EPISODES --max_steps $MAX_STEPS \
    --eval_steps $EVAL_STEPS --eval_episodes $EVAL_EPISODES \
    --random_seed $SEED \
    --name "exp1_sigma0.02"

# =============================================================================
# EXP 2: Stochasticity — high noise (σ=0.5)
# =============================================================================
echo ">>> Experiment 2: σ=0.5"

run_exp "EXP2: sigma=0.5 (high noise)" "$GRID" \
    --start_pos "$START_POS" --no_gui \
    --alpha $ALPHA --gamma 0.95 --epsilon $EPSILON --sigma 0.5 \
    --epsilon_decay_mode fixed --no_shaping \
    --episodes $EPISODES --max_steps $MAX_STEPS \
    --eval_steps $EVAL_STEPS --eval_episodes $EVAL_EPISODES \
    --random_seed $SEED \
    --name "exp2_sigma0.5"

# =============================================================================
# EXP 3: Discount factor — high (γ=0.95)
# =============================================================================
echo ">>> Experiment 3: γ=0.95"

run_exp "EXP3: gamma=0.95 (high discount)" "$GRID" \
    --start_pos "$START_POS" --no_gui \
    --alpha $ALPHA --gamma 0.95 --epsilon $EPSILON --sigma 0.02 \
    --epsilon_decay_mode fixed --no_shaping \
    --episodes $EPISODES --max_steps $MAX_STEPS \
    --eval_steps $EVAL_STEPS --eval_episodes $EVAL_EPISODES \
    --random_seed $SEED \
    --name "exp3_gamma0.95"

# =============================================================================
# EXP 4: Discount factor — low (γ=0.6)
# =============================================================================
echo ">>> Experiment 4: γ=0.6"

run_exp "EXP4: gamma=0.6 (low discount)" "$GRID" \
    --start_pos "$START_POS" --no_gui \
    --alpha $ALPHA --gamma 0.6 --epsilon $EPSILON --sigma 0.02 \
    --epsilon_decay_mode fixed --no_shaping \
    --episodes $EPISODES --max_steps $MAX_STEPS \
    --eval_steps $EVAL_STEPS --eval_episodes $EVAL_EPISODES \
    --random_seed $SEED \
    --name "exp4_gamma0.6"

# =============================================================================
# EXP 5: Reward — without BFS shaping (baseline)
# =============================================================================
echo ">>> Experiment 5: BFS shaping OFF (baseline)"

run_exp "EXP5: BFS shaping OFF (baseline)" "$GRID" \
    --start_pos "$START_POS" --no_gui \
    --alpha $ALPHA --gamma 0.95 --epsilon $EPSILON --sigma 0.02 \
    --epsilon_decay_mode fixed --no_shaping \
    --episodes $EPISODES --max_steps $MAX_STEPS \
    --eval_steps $EVAL_STEPS --eval_episodes $EVAL_EPISODES \
    --random_seed $SEED \
    --name "exp5_unshaped"

# =============================================================================
# EXP 6: Reward — with BFS shaping
# =============================================================================
echo ">>> Experiment 6: BFS shaping ON"

run_exp "EXP6: BFS shaping ON" "$GRID" \
    --start_pos "$START_POS" --no_gui \
    --alpha $ALPHA --gamma 0.95 --epsilon $EPSILON --sigma 0.02 \
    --epsilon_decay_mode fixed \
    --episodes $EPISODES --max_steps $MAX_STEPS \
    --eval_steps $EVAL_STEPS --eval_episodes $EVAL_EPISODES \
    --random_seed $SEED \
    --name "exp6_shaped"

# =============================================================================
echo ""
echo "========== ALL 6 EXPERIMENTS COMPLETE =========="
echo "Results: $RESULTS_FILE"
echo "Plots:   results/exp*_learning_curve.png"
