#!/bin/bash

: '
# DQN / PPO Hyperparameter Sweep — A1_grid
# Grid: A1_grid.npy | state_mode: gps | sigma: 0 | start_pos: 1,12
# episodes=500, max_steps=500, seeds: 0, 42
#
# DQN: lr in {0.0001, 0.0005, 0.001}, gamma=0.99, hidden_size=256,
#      epsilon=1, epsilon_decay=0.995, epsilon_min=0.1,
#      buffer_capacity=50000, batch_size=64, warmup=1000, target_update_freq=100
#
# PPO: lr in {0.0001, 0.0003, 0.0005}, gamma=0.99, hidden_size=256,
#      clip_eps=0.2, k_epochs=4, gae_lambda=0.95, entropy_coef=0.05,
#      value_coef=0.5, rollout_steps=2048, minibatch_size=64
'

set -e

GRID="grid_configs/A1_grid.npy"
EPISODES=500
MAX_STEPS=500
STATE_MODE=gps
SIGMA=0
START_POS="1,12"
SEEDS=(0 42)
GAMMA=0.99
HIDDEN_SIZE=256

RESULTS_FILE="results/dqn_ppo_sweep_results.txt"
mkdir -p results

echo "----------------------------------------------------------------" > "$RESULTS_FILE"
echo "DQN / PPO Sweep — A1_grid — $(date)" >> "$RESULTS_FILE"
echo "episodes=$EPISODES max_steps=$MAX_STEPS state_mode=$STATE_MODE sigma=$SIGMA start_pos=$START_POS" >> "$RESULTS_FILE"
echo "----------------------------------------------------------------" >> "$RESULTS_FILE"

run_dqn() {
    local label="$1"; shift
    echo "" >> "$RESULTS_FILE"
    echo "--- [DQN] $label ---" >> "$RESULTS_FILE"
    python3 train_dqn.py "$@" 2>&1 | tee -a "$RESULTS_FILE"
}

run_ppo() {
    local label="$1"; shift
    echo "" >> "$RESULTS_FILE"
    echo "--- [PPO] $label ---" >> "$RESULTS_FILE"
    python3 train_ppo.py "$@" 2>&1 | tee -a "$RESULTS_FILE"
}

# DQN

echo ">>> DQN runs"

for LR in 0.0001 0.0005 0.001; do
    for SEED in "${SEEDS[@]}"; do
        run_dqn "lr=$LR seed=$SEED" \
            --grid "$GRID" --no_gui \
            --episodes "$EPISODES" --max_steps "$MAX_STEPS" \
            --state_mode "$STATE_MODE" --sigma "$SIGMA" \
            --start_pos "$START_POS" --seed "$SEED" \
            --lr "$LR" --gamma "$GAMMA" --hidden_size "$HIDDEN_SIZE" \
            --epsilon 1 --epsilon_decay 0.995 --epsilon_min 0.1 \
            --buffer_capacity 50000 --batch_size 64 \
            --warmup 1000 --target_update_freq 100
    done
done

# PPO

echo ">>> PPO runs"

for LR in 0.0001 0.0003 0.0005; do
    for SEED in "${SEEDS[@]}"; do
        run_ppo "lr=$LR seed=$SEED" \
            --grid "$GRID" --no_gui \
            --episodes "$EPISODES" --max_steps "$MAX_STEPS" \
            --state_mode "$STATE_MODE" --sigma "$SIGMA" \
            --start_pos "$START_POS" --seed "$SEED" \
            --lr "$LR" --gamma "$GAMMA" --hidden_size "$HIDDEN_SIZE" \
            --clip_eps 0.2 --k_epochs 4 --gae_lambda 0.95 \
            --entropy_coef 0.05 --value_coef 0.5 \
            --rollout_steps 2048 --minibatch_size 64
    done
done

echo ""
echo "--- DQN/PPO SWEEP COMPLETE ---"
echo "Results: $RESULTS_FILE"
