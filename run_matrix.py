"""Multi-algorithm evaluation matrix.

Runs every (algo, setup, grid, seed) cell:
  1. Train the agent using the algo's adapter.
  2. Greedy-evaluate the trained agent at sigma=0.0 and sigma=0.1.
  3. Compute the Policy Optimality Ratio for each.
Writes one CSV row per cell with the schema documented in
REPORT_ALGO_MATRIX.md section 7.

Usage:
  python run_matrix.py --algos sarsa,mc,vi \\
                       --setups configs/final_setups.json \\
                       --grids large_grid,A1_grid \\
                       --seeds 5
"""
from __future__ import annotations
from argparse import ArgumentParser
from pathlib import Path
import csv
import json
import time

from eval_helpers import policy_optimality_ratio, bfs_optimal_steps
from world.grid import Grid


GRID_STARTS = {
    "large_grid": (1, 1),
    "A1_grid":    (1, 12),
}


def _resolve_grid(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.suffix == ".npy" and p.exists():
        return p
    return Path("grid_configs") / f"{name_or_path}.npy"


def _import_trainers(algos):
    """Lazy-import only the requested algo adapters. Keeps VI optional."""
    trainers = {}
    if "sarsa" in algos:
        from sarsa import train_sarsa
        trainers["sarsa"] = train_sarsa
    if "mc" in algos:
        from train_mc_on_policy import train_mc
        trainers["mc"] = train_mc
    if "vi" in algos:
        from vi import train_vi
        trainers["vi"] = train_vi
    return trainers


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--algos", required=True,
                    help="Comma list of {sarsa, mc, vi}")
    ap.add_argument("--setups", type=Path, required=True,
                    help="Path to JSON file with per-algo setup lists")
    ap.add_argument("--grids", default="large_grid,A1_grid",
                    help="Comma list of grid names (matched to GRID_STARTS)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--eval_episodes", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--out_dir", type=Path, default=Path("results/matrix"))
    args = ap.parse_args()

    algos = [a.strip() for a in args.algos.split(",")]
    grids = [g.strip() for g in args.grids.split(",")]
    setups = json.loads(args.setups.read_text())
    trainers = _import_trainers(algos)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d__%H-%M-%S")
    csv_path = args.out_dir / f"matrix_{stamp}.csv"

    fieldnames = [
        "algo", "setup", "grid", "seed",
        "convergence_metric",
        "actual_steps_det", "actual_steps_n10",
        "optimal_steps", "POR_det", "POR_n10",
        "elapsed_sec",
    ]
    rows: list[dict] = []
    t0_all = time.time()

    for algo in algos:
        for setup in setups[algo]:
            for grid_name in grids:
                grid_path = _resolve_grid(grid_name)
                start = GRID_STARTS[grid_name]
                grid_cells = Grid.load_grid(grid_path).cells
                optimal = bfs_optimal_steps(grid_cells, start)

                for seed in range(args.seeds):
                    t0 = time.time()
                    agent, conv = trainers[algo](
                        grid_path, start, setup, seed, args.patience,
                    )
                    por_det, steps_det = policy_optimality_ratio(
                        grid_path, start, agent,
                        sigma_eval=0.0,
                        n_eval_episodes=args.eval_episodes,
                        max_steps=args.max_steps, seed=seed,
                    )
                    por_n10, steps_n10 = policy_optimality_ratio(
                        grid_path, start, agent,
                        sigma_eval=0.1,
                        n_eval_episodes=args.eval_episodes,
                        max_steps=args.max_steps, seed=seed,
                    )
                    row = {
                        "algo": algo, "setup": setup["label"],
                        "grid": grid_name, "seed": seed,
                        "convergence_metric": conv,
                        "actual_steps_det": round(steps_det, 2),
                        "actual_steps_n10": round(steps_n10, 2),
                        "optimal_steps": optimal,
                        "POR_det": round(por_det, 4),
                        "POR_n10": round(por_n10, 4),
                        "elapsed_sec": round(time.time() - t0, 2),
                    }
                    rows.append(row)
                    print(f"  [{algo} {setup['label']:>20s} {grid_name:>10s} "
                          f"seed={seed}] conv={conv} POR_det={por_det:.3f} "
                          f"POR_n10={por_n10:.3f} ({row['elapsed_sec']}s)")

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    meta_path = Path(str(csv_path).removesuffix(".csv") + ".meta.json")
    meta_path.write_text(json.dumps({
        "algos": algos, "grids": grids, "seeds": args.seeds,
        "patience": args.patience, "eval_episodes": args.eval_episodes,
        "max_steps": args.max_steps,
        "setups_file": str(args.setups),
        "elapsed_total_sec": round(time.time() - t0_all, 1),
    }, indent=2))
    print(f"\nElapsed total: {(time.time() - t0_all):.1f}s")
    print(f"  -> {csv_path}")


if __name__ == "__main__":
    main()
