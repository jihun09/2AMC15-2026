"""Aggregate results/a2_summary_metrics.csv across seeds.
Run:  venv/Scripts/python.exe aggregate_metrics.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

PATH = Path("results/a2_summary_metrics.csv")


def main():
    if not PATH.exists():
        print(f"No summary file at {PATH} yet.")
        return
    rows = list(csv.DictReader(open(PATH)))
    groups = defaultdict(list)
    for r in rows:
        groups[(r["algo"], r["state_mode"], r["sigma"])].append(r)

    print(f"\nAggregated over seeds  (source: {PATH}, {len(rows)} runs)\n")
    hdr = f"{'algo':4} {'state':4} {'sigma':5} {'n':2} | {'success_rate':22} | {'path_efficiency':22} | per-seed success"
    print(hdr)
    print("-" * len(hdr))
    for key in sorted(groups):
        g = sorted(groups[key], key=lambda x: int(x["seed"]))
        sr = np.array([float(x["success_rate"]) for x in g])
        # success rate: over ALL seeds; path efficiency: conditional, over the
        # seeds that actually solved it (success_rate > 0) — a failed seed has
        # no path, so averaging its 0 in would conflate the two metrics.
        pe = np.array([float(x["mean_path_eff"]) for x in g if float(x["success_rate"]) > 0])
        n = len(g)
        sr_s = f"{sr.mean():.3f} +/- {sr.std(ddof=1 if n > 1 else 0):.3f}"
        if pe.size:
            pe_s = f"{pe.mean():.3f} +/- {pe.std(ddof=1 if pe.size > 1 else 0):.3f} (n={pe.size})"
        else:
            pe_s = "n/a (no successes)"
        per_seed = ", ".join(f"s{x['seed']}={float(x['success_rate']):.2f}" for x in g)
        print(f"{key[0]:4} {key[1]:4} {key[2]:5} {n:2} | {sr_s:22} | {pe_s:26} | {per_seed}")
    print("\nNote: success over ALL seeds; path_eff over successful seeds only.")
    print("ddof=1 for n>1. 2 seeds -> std indicative only.")


if __name__ == "__main__":
    main()
