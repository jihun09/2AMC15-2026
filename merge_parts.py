"""Fold per-run CSVs (results/parts/*.csv) into results/a2_summary_metrics.csv.

Each parallel run writes its own one-row CSV (--summary_csv) to avoid races.
This rewrites the main file with one row per config (last run wins), so it is
idempotent and self-healing.
"""
import csv
import glob
from pathlib import Path

from eval_metrics import SUMMARY_FIELDS

MAIN = Path("results/a2_summary_metrics.csv")
CONFIG = ["algo", "grid", "state_mode", "seed", "sigma", "lr", "episodes"]


def main():
    rows = list(csv.DictReader(open(MAIN))) if MAIN.exists() else []
    for p in sorted(glob.glob("results/parts/*.csv")):
        rows.extend(csv.DictReader(open(p)))

    by_key = {tuple(r.get(k, "") for k in CONFIG): r for r in rows}
    with open(MAIN, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for r in by_key.values():
            w.writerow({k: r.get(k, "") for k in SUMMARY_FIELDS})
    print(f"Wrote {len(by_key)} unique rows to {MAIN}")


if __name__ == "__main__":
    main()
