"""Batch parameter sweep over remote_share.

Runs N replications at each level of remote_share (0.0 .. 1.0) and writes
per-run results to a CSV. Use this for the actual sensitivity analysis.

Usage:
    python experiment.py --reps 30 --levels 11 --out sweep_results.csv
"""

import argparse
import csv

from model import TeamModel


def run_one(remote_share, mean_solo_skill, solo_skill_spread, seed, **kwargs):
    """Run a (possibly multi-sprint) simulation and return summary metrics.

    ``velocity`` is the final sprint's value; ``total_velocity`` sums all
    sprints. ``mean_skill`` and ``attrition_count`` capture the long-run
    capability and retention outcomes introduced by the learning/turnover
    mechanics.
    """
    m = TeamModel(
        remote_share=remote_share,
        mean_solo_skill=mean_solo_skill,
        solo_skill_spread=solo_skill_spread,
        seed=seed,
        **kwargs,
    )
    while m.running:
        m.step()
    avg_wait = sum(m.wait_times) / len(m.wait_times) if m.wait_times else 0.0
    mean_skill = sum(d.solo_skill for d in m.devs) / len(m.devs) if m.devs else 0.0
    return {
        "remote_share": remote_share,
        "mean_solo_skill": mean_solo_skill,
        "solo_skill_spread": solo_skill_spread,
        "seed": seed,
        "velocity": m.velocity,
        "total_velocity": sum(m.sprint_velocities),
        "avg_wait": avg_wait,
        "blockers_resolved": len(m.wait_times),
        "completed_tasks": len(m.completed_tasks),
        "mean_skill": mean_skill,
        "attrition_count": m.attrition_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=20,
                        help="repetitions per remote_share level")
    parser.add_argument("--levels", type=int, default=11,
                        help="number of remote_share levels (uniform 0..1)")
    parser.add_argument("--out", default="sweep_results.csv")
    parser.add_argument("--mean_solo_skill", type=float, default=0.5)
    parser.add_argument("--solo_skill_spread", type=float, default=0.2)
    args = parser.parse_args()

    levels = [i / (args.levels - 1) for i in range(args.levels)]
    total = len(levels) * args.reps
    rows = []
    done = 0

    print(f"Running {total} simulations: {args.levels} levels x {args.reps} reps")
    for rs in levels:
        for r in range(args.reps):
            seed = 100 * r + int(round(rs * 1000))
            rows.append(run_one(
                rs, args.mean_solo_skill, args.solo_skill_spread, seed,
            ))
            done += 1
            if done % 10 == 0 or done == total:
                print(f"  {done}/{total}")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
