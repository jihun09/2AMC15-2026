#!/usr/bin/env bash
# Run the A2 metric experiments: DQN & PPO x seed{0,42} x sigma{0,0.1} on A1_grid (gps).
# Each run appends success_rate + path_efficiency to results/a2_summary_metrics.csv.
cd "$(dirname "$0")"
PY=venv/Scripts/python.exe
COMMON="--grid grid_configs/A1_grid.npy --no_gui --episodes 500 --max_steps 500 \
--state_mode gps --lr 5e-4 --gamma 0.99 --hidden_size 256 --start_pos 1,12"

rm -f results/a2_summary_metrics.csv

for sigma in 0.0 0.1; do
  for seed in 0 42; do
    echo "=========== DQN | seed=$seed | sigma=$sigma ==========="
    $PY train_dqn.py $COMMON --seed "$seed" --sigma "$sigma" \
        --epsilon_min 0.1 --epsilon_decay 0.995 \
      || echo "!!! DQN seed=$seed sigma=$sigma FAILED"
  done
done

for sigma in 0.0 0.1; do
  for seed in 0 42; do
    echo "=========== PPO | seed=$seed | sigma=$sigma ==========="
    $PY train_ppo.py $COMMON --seed "$seed" --sigma "$sigma" \
      || echo "!!! PPO seed=$seed sigma=$sigma FAILED"
  done
done

echo "=========== ALL RUNS DONE ==========="
$PY aggregate_metrics.py
