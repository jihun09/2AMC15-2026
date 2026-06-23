"""Summarize all DQN/PPO experiment results into one table.

Scans results/*_eval.csv and the matching *_training.csv produced by
train_dqn.py / train_ppo.py, parses the hyperparameters encoded in each
filename, and writes one row per run (final metrics + params) to
results/experiments_summary.csv. Also prints the table to stdout.

Usage:
    python3 summarize_experiments.py
"""
import csv
import re
from pathlib import Path

RESULTS_DIR = Path("results")
SUMMARY_CSV = RESULTS_DIR / "experiments_summary.csv"

RUN_RE = re.compile(
    r"^(?P<algo>dqn|ppo)_(?P<grid>.+)_(?P<state_mode>gps|raycasting|both)"
    r"_seed(?P<seed>\d+)_sigma(?P<sigma>[0-9.]+)_lr(?P<lr>[0-9.eE+-]+)"
    r"_g(?P<gamma>[0-9.]+)_h(?P<hidden>\d+)"
    r"(?:_episodes(?P<episodes_cfg>\d+)_max_steps(?P<max_steps_cfg>\d+))?"
    r"_range(?P<range>\w+)_eval\.csv$"
)


def load_csv_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def summarize_run(eval_path: Path) -> dict | None:
    m = RUN_RE.match(eval_path.name)
    if not m:
        print(f"[skip] filename doesn't match expected pattern: {eval_path.name}")
        return None
    info = m.groupdict()

    eval_rows = load_csv_rows(eval_path)
    if not eval_rows:
        return None
    final_eval = eval_rows[-1]

    train_path = eval_path.with_name(eval_path.name.replace("_eval.csv", "_training.csv"))
    episodes_run, max_steps_seen = info["episodes_cfg"], info["max_steps_cfg"]
    if train_path.exists():
        train_rows = load_csv_rows(train_path)
        if train_rows:
            if not episodes_run:
                episodes_run = train_rows[-1]["episode"]
            if not max_steps_seen:
                max_steps_seen = max(int(r["steps"]) for r in train_rows)

    return {
        "algo": info["algo"].upper(),
        "grid": info["grid"],
        "state_mode": info["state_mode"],
        "seed": info["seed"],
        "sigma": info["sigma"],
        "lr": info["lr"],
        "gamma": info["gamma"],
        "hidden_size": info["hidden"],
        "max_range": info["range"],
        "episodes": episodes_run or "",
        "max_steps": max_steps_seen or "",
        "final_success_rate": final_eval["success_rate"],
        "final_mean_reward": final_eval["mean_reward"],
        "final_mean_steps": final_eval["mean_steps"],
        "run_name": eval_path.name.replace("_eval.csv", ""),
    }


def print_table(rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys())
    col_widths = {k: max(len(k), max(len(str(r[k])) for r in rows)) for k in fieldnames}
    header = " | ".join(k.ljust(col_widths[k]) for k in fieldnames)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(str(r[k]).ljust(col_widths[k]) for k in fieldnames))


def main():
    rows = []
    for eval_path in sorted(RESULTS_DIR.glob("*_eval.csv")):
        row = summarize_run(eval_path)
        if row:
            rows.append(row)

    if not rows:
        print(f"No experiment results found in {RESULTS_DIR}/.")
        return

    fieldnames = list(rows[0].keys())
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print_table(rows)
    print(f"\nSaved: {SUMMARY_CSV} ({len(rows)} runs)")


if __name__ == "__main__":
    main()