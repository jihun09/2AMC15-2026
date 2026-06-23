#!/usr/bin/env bash
# Reproduce the A2 metrics (success rate + path efficiency) on A1_grid, gps.
#   DQN: 5 seeds {0,42,1,2,3}     PPO: 2 seeds {0,42} (uniform 0% negative result)
# Each run appends one row to results/a2_summary_metrics.csv; aggregated at the end.
#
# Usage:  bash run_a2_metrics.sh          (~1 h on a GPU; longer on CPU)
cd "$(dirname "$0")"

# Portable interpreter: prefer the project venv, else system python.
if   [ -x venv/Scripts/python.exe ]; then PY=venv/Scripts/python.exe   # Windows
elif [ -x venv/bin/python        ]; then PY=venv/bin/python            # macOS/Linux
else PY=python; fi

COMMON="--grid grid_configs/A1_grid.npy --no_gui --episodes 500 --max_steps 500 \
--state_mode gps --lr 5e-4 --gamma 0.99 --hidden_size 256 --start_pos 1,12"

rm -f results/a2_summary_metrics.csv   # start clean so rows don't accumulate

for sigma in 0.0 0.1; do
  for seed in 0 42 1 2 3; do
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
