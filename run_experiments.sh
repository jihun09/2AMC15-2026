#!/usr/bin/env bash
# Reproduce every experiment cited in REPORT_SARSA.md.
#
# Usage: bash run_experiments.sh
# Each step takes a few seconds on a laptop CPU; total ~5 min.

set -euo pipefail

PY="${PY:-./venv/Scripts/python.exe}"     # adjust for non-Windows venvs
GRID=grid_configs
RESULTS=results

echo "==> Section 3: SARSA vs Random on A1_grid (multi-seed)"
$PY sarsa.py compare $GRID/A1_grid.npy \
    --episodes 1000 --seeds 5 --sigma 0.1 --alpha 0.1 --gamma 0.95 --epsilon 0.1

echo "==> Section 4.1: alpha sweep on A1_grid"
$PY sarsa.py sweep $GRID/A1_grid.npy \
    --param alpha --values 0.01,0.05,0.1,0.3,0.5 \
    --episodes 1000 --seeds 5 --sigma 0.1 --gamma 0.95 --epsilon 0.1

echo "==> Section 4.2: gamma sweep on A1_grid"
$PY sarsa.py sweep $GRID/A1_grid.npy \
    --param gamma --values 0.5,0.8,0.9,0.95,0.99 \
    --episodes 1000 --seeds 5 --sigma 0.1 --alpha 0.1 --epsilon 0.1

echo "==> Section 4.3: sigma sweep on A1_grid"
$PY sarsa.py sweep $GRID/A1_grid.npy \
    --param sigma --values 0.0,0.1,0.3,0.5 \
    --episodes 1000 --seeds 5 --alpha 0.1 --gamma 0.95 --epsilon 0.1

echo "==> Section 4.4: epsilon-schedule sweep on A1_grid"
$PY sarsa.py sweep-eps $GRID/A1_grid.npy \
    --episodes 1000 --seeds 5 --sigma 0.1 --alpha 0.1 --gamma 0.95

echo "==> Section 4.5: cross-grid transfer A1 -> super_hard"
$PY sarsa.py transfer \
    --train_grid $GRID/A1_grid.npy --test_grid $GRID/super_hard.npy \
    --episodes 2000 --eval_episodes 50 --seeds 5 \
    --alpha 0.1 --gamma 0.95 --epsilon 0.3 --epsilon_end 0.05 --epsilon_decay_episodes 1500 --sigma 0.1

echo "==> Section 7: A1 head-to-head shootout"
$PY sarsa.py eval-configs $GRID/A1_grid.npy \
    --configs configs/push_best.json --seeds 5 --tag push_best

echo "==> Section 8: cross-grid validation (all grids)"
$PY sarsa.py eval-configs \
    --configs configs/multi_grid.json --seeds 5 --tag multi_grid

echo "==> Section 8.2: super_hard rescue"
$PY sarsa.py eval-configs $GRID/super_hard.npy \
    --configs configs/rescue_super_hard.json --seeds 5 --tag rescue_super_hard

echo "==> Section 9.1: SARSA vs Q-learning vs SARSA(lambda) on 4 normal grids"
$PY sarsa.py eval-configs \
    $GRID/example_grid.npy $GRID/small_grid.npy $GRID/A1_grid.npy $GRID/large_grid.npy \
    --configs configs/algo_compare.json --seeds 5 --tag algo_compare

echo "==> Section 9.2: sample efficiency on A1 (varying episode budgets)"
$PY sarsa.py eval-configs $GRID/A1_grid.npy \
    --configs configs/algo_sample_efficiency.json --seeds 5 --tag sample_eff

echo "==> Section 9.3: algorithm comparison on super_hard"
$PY sarsa.py eval-configs $GRID/super_hard.npy \
    --configs configs/algo_super_hard.json --seeds 5 --tag algo_super_hard

echo "==> Section 10: Phase 2 scaffold — linear function approximation"
$PY phase2_linear_eval.py --seeds 5 --episodes 3000

echo
echo "Done. See REPORT_SARSA.md sections for analysis."
echo "Outputs in $RESULTS/"
