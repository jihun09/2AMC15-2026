#!/bin/bash

"""
# Experiment 2.4 — Stochasticity Resilience in Large Environment
# Grid: large_grid.npy | Sigma: 0.1, 0.25 | Default reward (no shaping)
#
# VI:   sigma (0.1, 0.4) x gamma (0.6, 0.9)
# MC:   sigma (0.1, 0.4) x delta (0.6, 0.9) x max_ep (250, 750) x epsilon (0.1, 0.3) 
# SARSA: sigma (0.1, 0.4) x gamma (0.6, 0.9) x alpha (0.1, 0.3) x epsilon (0.1, 0.3) x max_ep (250, 750)    
"""

set -e

GRID="grid_configs/large_grid.npy"
START_POS="1,1"
SEED=27
RESULTS_FILE="results/exp24_results.txt"
mkdir -p results

echo "----------------------------------------------------------------" > "$RESULTS_FILE"
echo "Experiment 2.4 — Large Grid Stochasticity Resilience — $(date)" >> "$RESULTS_FILE"
echo "Grid: $GRID | Sigmas: 0.1, 0.4" >> "$RESULTS_FILE"
echo "----------------------------------------------------------------" >> "$RESULTS_FILE"

run_vi() {
    local label="$1"; shift
    echo "" >> "$RESULTS_FILE"
    echo "--- [VI] $label ---" >> "$RESULTS_FILE"
    python3 train_VI.py "$@" 2>&1 | tee -a "$RESULTS_FILE"
}

run_mc() {
    local label="$1"; shift
    echo "" >> "$RESULTS_FILE"
    echo "--- [MC] $label ---" >> "$RESULTS_FILE"
    python3 train_mc_on_policy.py "$@" 2>&1 | tee -a "$RESULTS_FILE"
}

run_sarsa() {
    local label="$1"; shift
    echo "" >> "$RESULTS_FILE"
    echo "--- [SARSA] $label ---" >> "$RESULTS_FILE"
    python3 sarsa.py "$@" 2>&1 | tee -a "$RESULTS_FILE"
}

# VALUE ITERATION

echo ">>> VI runs"

for SIGMA in 0.1 0.4; do
    for GAMMA in 0.6 0.9; do
        run_vi "sigma=$SIGMA gamma=$GAMMA" \
            "$GRID" --no_gui --start_pos "$START_POS" --iter 2000 \
            --sigma "$SIGMA" --gamma "$GAMMA" --random_seed "$SEED" \
            --shaping_weight 0
    done
done


# MONTE CARLO

echo ">>> MC runs"

for SIGMA in 0.1 0.4; do
    for DELTA in 0.6 0.9; do
        for EPS in 0.1 0.3; do
            for EPISODES in 250 750; do
                run_mc "sigma=$SIGMA delta=$DELTA eps=$EPS episodes=$EPISODES" \
                    "$GRID" --no_gui --start_pos "$START_POS" \
                    --sigma "$SIGMA" --delta "$DELTA" \
                    --iter 2000 --episodes "$EPISODES" \
                    --epsilon "$EPS" --epsilon_decay 1 --epsilon_min 0.0 \
                    --shaping_weight 0 \
                    --random_seed "$SEED"
            done
        done
    done
done


# SARSA

echo ">>> SARSA runs"

for SIGMA in 0.1 0.4; do
    for GAMMA in 0.6 0.9; do
        for ALPHA in 0.1 0.3; do
            for EPS in 0.1 0.3; do
                for EPISODES in 250 750; do
                    run_sarsa "sigma=$SIGMA gamma=$GAMMA alpha=$ALPHA eps=$EPS episodes=$EPISODES" \
                        "$GRID" --no_gui --start_pos "$START_POS" \
                        --sigma "$SIGMA" --gamma "$GAMMA" \
                        --alpha "$ALPHA" --epsilon "$EPS" --epsilon_decay 1.0 \
                        --episodes "$EPISODES" --iter 2000 \
                        --shaping_weight 0 \
                        --random_seed "$SEED"
                done
            done
        done
    done
done

echo ""
echo "--- EXPERIMENT 2.4 COMPLETE ---"
echo "Results: $RESULTS_FILE"
