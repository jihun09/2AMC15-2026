#!/usr/bin/env bash
# Reproduce all Assignment-2 metrics into results/a2_summary_metrics.csv.
# Runs every config in parallel (one process per core), then aggregates.
#   DQN  (500 eps): lr in {1e-4, 5e-4, 1e-3} at sigma 0  +  lr 5e-4 at sigma 0.1
#   PPO (3000 eps): sigma in {0, 0.1}        +  500-eps baseline (2 seeds) for contrast
# All at: A1_grid, gps, gamma 0.99, hidden 256, start (1,12), 10 seeds.
# Usage: bash run_experiments.sh    (~30 min on a 16-core CPU)
set -u
cd "$(dirname "$0")"

if   [ -x venv/Scripts/python.exe ]; then PY=venv/Scripts/python.exe
elif [ -x venv/bin/python        ]; then PY=venv/bin/python
else PY=python; fi

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
JOBS=10
SEEDS="0 42 1 2 3 4 5 6 7 8"
COMMON="--grid grid_configs/A1_grid.npy --no_gui --max_steps 500 --state_mode gps \
--gamma 0.99 --hidden_size 256 --start_pos 1,12"

rm -rf results/parts; mkdir -p results/parts
rm -f results/a2_summary_metrics.csv

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done; }

dqn() {  # lr seed sigma
  local tag="dqn_lr$1_s$2_sig$3"
  $PY train_dqn.py $COMMON --episodes 500 --epsilon_min 0.1 --epsilon_decay 0.995 \
     --lr "$1" --seed "$2" --sigma "$3" --summary_csv "results/parts/$tag.csv" \
     > "results/parts/$tag.log" 2>&1
}
ppo() {  # episodes seed sigma
  local tag="ppo_e$1_s$2_sig$3"
  $PY train_ppo.py $COMMON --episodes "$1" --lr 5e-4 \
     --seed "$2" --sigma "$3" --summary_csv "results/parts/$tag.csv" \
     > "results/parts/$tag.log" 2>&1
}

for s in $SEEDS; do
  throttle; dqn 1e-4 "$s" 0.0 &
  throttle; dqn 5e-4 "$s" 0.0 &
  throttle; dqn 1e-3 "$s" 0.0 &
  throttle; dqn 5e-4 "$s" 0.1 &
  throttle; ppo 3000 "$s" 0.0 &
  throttle; ppo 3000 "$s" 0.1 &
done
for s in 0 42; do
  throttle; ppo 500 "$s" 0.0 &
  throttle; ppo 500 "$s" 0.1 &
done
wait

$PY merge_parts.py
$PY aggregate_metrics.py
