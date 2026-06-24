#!/bin/bash

: '
# DQN / PPO Hyperparameter Sweep — A1_grid
# Grid: A1_grid.npy | state_mode: gps | sigma: 0, 0.1 | start_pos: 1,12
# episodes=500, max_steps=500, seeds: 0, 42
#
# DQN: lr in {0.0001, 0.0005, 0.001}, gamma=0.99, hidden_size=256,
#      epsilon=1, epsilon_decay=0.995, epsilon_min=0.1,
#      buffer_capacity=50000, batch_size=64, warmup=1000, target_update_freq=100
#
# PPO: lr in {0.0001, 0.0003, 0.0005, 0.001}, gamma=0.99, hidden_size=256,
#      clip_eps=0.2, k_epochs=4, gae_lambda=0.95, entropy_coef=0.05,
#      value_coef=0.5, rollout_steps=2048, minibatch_size=64
#
# PPO — confirmed best config (seed=123, sigma=0.1, lr=0.0005):
# the 500-episode sweep above is not enough to converge under sigma=0.1 —
# greedy-eval success rate stays near 0% until ~episode 1800, then locks
# onto a 100% success / near-optimal policy by episode 3000. Entropy 0.01
# and 0.1 were both tried and performed worse than the 0.05 default, so the
# fix was more episodes, not different entropy. Run with eval_episodes=20
# (vs default 10) for a less noisy success-rate readout.
'

set -e

GRID="grid_configs/A1_grid.npy"
EPISODES=(3000)
MAX_STEPS=(500)
STATE_MODE=gps
SIGMAS=(0 0.1)
START_POS="1,12"
SEEDS=(0 1 2 3 4 5 6 7 8 42)
GAMMA=0.99
HIDDEN_SIZE=256
LR_DQN=(0.0005)
LR_PPO=(0.0005)

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


# PPO

# PPO

echo ">>> PPO runs"


for SEED in "${SEEDS[@]}"; do
    for LR in "${LR_PPO[@]}"; do
        for SIGMA in "${SIGMAS[@]}"; do
            for episodes in "${EPISODES[@]}"; do
                for max_steps in "${MAX_STEPS[@]}"; do
                     run_ppo "lr=$LR seed=$SEED sigma=$SIGMA episodes=$episodes max_steps=$max_steps" \
                        --grid "$GRID" --no_gui \
                        --episodes "$episodes" --max_steps "$max_steps" \
                        --state_mode "$STATE_MODE" --sigma "$SIGMA" \
                        --start_pos "$START_POS" --seed "$SEED" \
                        --lr "$LR" --gamma "$GAMMA" --hidden_size "$HIDDEN_SIZE" \
                        --clip_eps 0.2 --k_epochs 4 --gae_lambda 0.95 \
                        --entropy_coef 0.05 --value_coef 0.5 \
                        --rollout_steps 2048 --minibatch_size 64
                done
            done
        done
    done
done




echo ""
echo "--- DQN/PPO SWEEP COMPLETE ---"
echo "Results: $RESULTS_FILE"

echo ""
echo ">>> Summarizing all runs"
python3 summarize_experiments.py | tee -a "$RESULTS_FILE"
