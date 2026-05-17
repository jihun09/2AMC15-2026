#!/bin/bash
# =============================================================================
# Experiment 2.4 — Stochasticity Resilience in Large Environment
# Grid: large_grid.npy | Sigma: 0.1, 0.25 | Default reward (no shaping)
#
# VI:   sigma x gamma=0.9
# MC:   sigma x max_ep_len (500, 2000) x epsilon (0.05, 0.2) | delta=0.9
# SARSA: sigma x alpha (0.05, 0.2) x epsilon (0.05, 0.2)     | gamma=0.9
# =============================================================================

set -e

GRID="grid_configs/large_grid.npy"
START_POS="1,12"
SEED=27
RESULTS_FILE="results/exp24_results.txt"
mkdir -p results

echo "================================================================" > "$RESULTS_FILE"
echo "Experiment 2.4 — Large Grid Stochasticity Resilience — $(date)" >> "$RESULTS_FILE"
echo "Grid: $GRID | Sigmas: 0.1, 0.25" >> "$RESULTS_FILE"
echo "================================================================" >> "$RESULTS_FILE"

run_vi() {
    local label="$1"; shift
    echo "" >> "$RESULTS_FILE"
    echo "=== [VI] $label ===" >> "$RESULTS_FILE"
    python3 train_VI.py "$@" 2>&1 | tee -a "$RESULTS_FILE"
}

run_mc() {
    local label="$1"; shift
    echo "" >> "$RESULTS_FILE"
    echo "=== [MC] $label ===" >> "$RESULTS_FILE"
    python3 train_mc_on_policy.py "$@" 2>&1 | tee -a "$RESULTS_FILE"
}

run_sarsa() {
    local label="$1"; shift
    echo "" >> "$RESULTS_FILE"
    echo "=== [SARSA] $label ===" >> "$RESULTS_FILE"
    python3 train_sarsa.py "$@" 2>&1 | tee -a "$RESULTS_FILE"
}

# =============================================================================
# VALUE ITERATION
# =============================================================================
echo ">>> VI runs"

for SIGMA in 0.1 0.25; do
    run_vi "sigma=$SIGMA gamma=0.9" \
        "$GRID" --no_gui --start_pos "$START_POS" --iter 2000 \
        --sigma "$SIGMA" --gamma 0.9 --random_seed "$SEED"
done

# =============================================================================
# MONTE CARLO
# =============================================================================
echo ">>> MC runs"

for SIGMA in 0.1 0.25; do
    for ITER in 500 2000; do
        for EPS in 0.05 0.2; do
            run_mc "sigma=$SIGMA iter=$ITER eps=$EPS" \
                "$GRID" --no_gui --start_pos "$START_POS" \
                --sigma "$SIGMA" --delta 0.9 \
                --iter "$ITER" --episodes 1000 \
                --epsilon "$EPS" --epsilon_decay 0.999 --epsilon_min 0.0 \
                --shaping_weight 0.0 \
                --random_seed "$SEED"
        done
    done
done

# =============================================================================
# SARSA
# =============================================================================
echo ">>> SARSA runs"

for SIGMA in 0.1 0.25; do
    for ALPHA in 0.05 0.2; do
        for EPS in 0.05 0.2; do
            run_sarsa "sigma=$SIGMA alpha=$ALPHA eps=$EPS" \
                "$GRID" --no_gui --start_pos "$START_POS" \
                --sigma "$SIGMA" --gamma 0.9 \
                --alpha "$ALPHA" --epsilon "$EPS" \
                --epsilon_decay_mode fixed --shaping_weight 0.0 \
                --episodes 1000 --max_steps 2000 \
                --eval_steps 1000 --eval_episodes 10 \
                --random_seed "$SEED"
        done
    done
done

echo ""
echo "========== EXPERIMENT 2.4 COMPLETE =========="
echo "Results: $RESULTS_FILE"
