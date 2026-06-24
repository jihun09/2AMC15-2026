"""Aggregate results/a2_summary_metrics.csv over seeds, one row per config."""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

PATH = Path("results/a2_summary_metrics.csv")
OUT_PATH = Path("results/aggregated_by_sigma.csv")

FILTER_ALGO = "dqn"
FILTER_STATE_MODE = "gps"
EPISODES = 1500
MAX_STEPS = 500

FIELDNAMES = [
    "sigma", "n_seeds",
    "sum_n_success", "total_episodes",
    "mean_success_rate", "std_success_rate",
    "mean_path_eff", "std_path_eff", "n_path_eff",
    "mean_steps_success", "n_steps_success",
]


def main():
    if not PATH.exists():
        print(f"No summary file at {PATH} yet.")
        return
    all_rows = list(csv.DictReader(open(PATH)))
    rows = [r for r in all_rows
            if r["algo"] == FILTER_ALGO and r["state_mode"] == FILTER_STATE_MODE and int(r["episodes"]) == EPISODES and int(r["max_steps"]) == MAX_STEPS]
    by_sigma = defaultdict(list)
    for r in rows:
        print(r)
        by_sigma[r["sigma"]].append(r)

    if not rows:
        print(f"No rows match algo={FILTER_ALGO} state_mode={FILTER_STATE_MODE} episodes={EPISODES} max_steps={MAX_STEPS} in {PATH}.")
        return

    print(f"\nFilter: algo={FILTER_ALGO} state_mode={FILTER_STATE_MODE}  "
          f"({len(rows)}/{len(all_rows)} runs)")
    print(f"Summed over seeds (mean where a sum isn't meaningful), "
          f"grouped by sigma  ({PATH})\n")
    hdr = (f"{'sigma':5} {'n_seeds':7} | {'sum n_success/total':19} | "
           f"{'success_rate':22} | {'path_efficiency':26} | mean_steps_success")
    print(hdr)
    print("-" * len(hdr))

    out_rows = []
    for sigma in sorted(by_sigma, key=float):
        g = by_sigma[sigma]
        sr = np.array([float(x["success_rate"]) for x in g])
        n_success = np.array([int(x["n_success"]) for x in g])
        eval_episodes = np.array([int(x["eval_episodes"]) for x in g])
        n_seeds = len(g)

        sum_n_success = int(n_success.sum())
        total_episodes = int(eval_episodes.sum())
        # success_rate = successes / n_episodes, applied to the pooled totals
        # across every seed (not an average of per-seed ratios).
        mean_sr = sum_n_success / total_episodes
        std_sr = sr.std(ddof=1 if n_seeds > 1 else 0)

        # mean_path_eff / std_path_eff / mean_steps_success are each computed
        # per-seed as mean()/std() over that seed's per-episode success_ratios
        # (resp. success_steps). To get the true pooled statistic over every
        # successful episode across all seeds — not just an unweighted mean of
        # per-seed means — combine the per-seed (count, mean, std) triples
        # weighted by each seed's n_success.
        mask = n_success > 0
        if mask.any():
            w = n_success[mask].astype(float)
            pe_means = np.array([float(x["mean_path_eff"]) for x in g])[mask]
            pe_stds = np.array([float(x["std_path_eff"]) for x in g])[mask]
            step_means = np.array([float(x["mean_steps_success"]) for x in g])[mask]

            mean_pe = float(np.sum(w * pe_means) / np.sum(w))
            pooled_var_pe = np.sum(w * (pe_stds ** 2 + (pe_means - mean_pe) ** 2)) / np.sum(w)
            std_pe = float(np.sqrt(pooled_var_pe))
            mean_steps = float(np.sum(w * step_means) / np.sum(w))
        else:
            mean_pe = std_pe = mean_steps = None
        pe_size = mask.sum()
        steps_size = mask.sum()

        sr_s = f"{mean_sr:.3f} +/- {std_sr:.3f}"
        pe_s = (f"{mean_pe:.3f} +/- {std_pe:.3f} (n={pe_size})"
                if pe_size else "n/a (no successes)")
        steps_s = f"{mean_steps:.1f}" if steps_size else "n/a"
        print(f"{sigma:5} {n_seeds:7} | {f'{sum_n_success}/{total_episodes}':19} | "
              f"{sr_s:22} | {pe_s:26} | {steps_s}")

        out_rows.append({
            "sigma": sigma, "n_seeds": n_seeds,
            "sum_n_success": sum_n_success, "total_episodes": total_episodes,
            "mean_success_rate": mean_sr, "std_success_rate": std_sr,
            "mean_path_eff": mean_pe, "std_path_eff": std_pe, "n_path_eff": pe_size,
            "mean_steps_success": mean_steps, "n_steps_success": steps_size,
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
