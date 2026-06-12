"""Single-parameter sensitivity analysis of ``remote_share``.

Regime: the thesis-relevant ~1-year horizon (multi-sprint, learning + turnover
ON), all other parameters at their model defaults. We sweep ``remote_share``
alone and record per-run (NOT pre-averaged) macroscopic observables so the full
run-to-run distribution is preserved for the distribution / mean-representativeness
analysis.

Two outputs:
  * results/sens_remote_main.csv  -- fine grid (21 levels) x REPS_MAIN reps
  * results/sens_remote_conv.csv  -- 3 levels x REPS_CONV reps (convergence study)

Both are raw per-run rows. Aggregation and plotting live in plot_sensitivity.py.

Usage:
    python sensitivity_remote.py            # full study (parallel)
    python sensitivity_remote.py --quick    # tiny grid for a smoke test
"""

import argparse
import csv
import os
from functools import partial
from multiprocessing import Pool

from model import TeamModel

OUT_DIR = "results"

# Fixed context for the whole study: the multi-sprint thesis regime at defaults.
# remote_share is the ONLY parameter that varies.
BASE = dict(n_devs=10, n_sprints=24)

# Fine grid over the full feasible range of a probability.
LEVELS_MAIN = [round(i / 20, 2) for i in range(21)]   # 0.00, 0.05, ..., 1.00
REPS_MAIN = 80

# Convergence sub-study: many reps at low / mid / high remote_share.
LEVELS_CONV = [0.0, 0.5, 1.0]
REPS_CONV = 400

SEED_BASE = 7_000


def _seed(level, rep):
    """Deterministic, distinct, non-negative seed for a (level, rep) cell."""
    return SEED_BASE + rep + 100_000 * int(round(level * 100))


def run_one(task):
    """Worker: run one full-horizon simulation, return macroscopic observables.

    ``task`` is a (remote_share, rep) tuple so the call is trivially picklable
    for multiprocessing.
    """
    remote_share, rep = task
    seed = _seed(remote_share, rep)
    m = TeamModel(remote_share=remote_share, seed=seed, **BASE)
    while m.running:
        m.step()
    waits = m.wait_times
    avg_wait = sum(waits) / len(waits) if waits else 0.0
    mean_skill = sum(d.solo_skill for d in m.devs) / len(m.devs) if m.devs else 0.0
    return {
        "remote_share": remote_share,
        "rep": rep,
        "seed": seed,
        "total_velocity": sum(m.sprint_velocities),
        "final_velocity": m.velocity,
        "avg_wait": avg_wait,
        "blockers_resolved": len(waits),
        "completed_tasks": len(m.completed_tasks),
        "mean_skill": mean_skill,
        "attrition_count": m.attrition_count,
    }


def _tasks(levels, reps):
    return [(lv, r) for lv in levels for r in range(reps)]


def _run_parallel(tasks, workers, label):
    print(f"[{label}] {len(tasks)} runs on {workers} workers")
    rows = []
    with Pool(workers) as pool:
        for i, row in enumerate(pool.imap_unordered(run_one, tasks, chunksize=8), 1):
            rows.append(row)
            if i % 200 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}")
    # stable order for reproducible CSVs
    rows.sort(key=lambda r: (r["remote_share"], r["rep"]))
    return rows


def _write(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--quick", action="store_true",
                        help="tiny grid for a smoke test")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    levels_main, reps_main = LEVELS_MAIN, REPS_MAIN
    levels_conv, reps_conv = LEVELS_CONV, REPS_CONV
    if args.quick:
        levels_main, reps_main = [0.0, 0.5, 1.0], 5
        levels_conv, reps_conv = [0.0, 1.0], 10

    main_rows = _run_parallel(_tasks(levels_main, reps_main), args.workers, "main")
    _write(main_rows, f"{OUT_DIR}/sens_remote_main.csv")

    conv_rows = _run_parallel(_tasks(levels_conv, reps_conv), args.workers, "conv")
    _write(conv_rows, f"{OUT_DIR}/sens_remote_conv.csv")

    print("done")


if __name__ == "__main__":
    main()
