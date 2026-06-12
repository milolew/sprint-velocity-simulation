"""Batch experiments for n_devs=10.

Runs three sweeps:
    1. Baseline: remote_share sweep with default params.
    2. remote_share x mean_solo_skill interaction.
    3. remote_share x block_prob interaction.

For each cell: 20 replications with deterministic seeds.
Writes CSVs and renders PNG plots with 95% confidence bands.
"""

import csv
import os
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np

from model import TeamModel

OUT_DIR = "results"
N_DEVS = 10
REPS = 20
SEED_OFFSET = 1000
# The original three sweeps run single-sprint for comparability with the
# pre-extension report; flip this to run the multi-sprint learning/turnover
# sweeps as well (much slower).
RUN_EXTENDED_SWEEPS = False


def run_one(seed, **kwargs):
    m = TeamModel(seed=seed, **kwargs)
    while m.running:
        m.step()
    avg_wait = sum(m.wait_times) / len(m.wait_times) if m.wait_times else 0.0
    mean_skill = sum(d.solo_skill for d in m.devs) / len(m.devs) if m.devs else 0.0
    return {
        "velocity": m.velocity,
        "total_velocity": sum(m.sprint_velocities),
        "avg_wait": avg_wait,
        "blockers_resolved": len(m.wait_times),
        "completed_tasks": len(m.completed_tasks),
        "mean_skill": mean_skill,
        "attrition_count": m.attrition_count,
    }


def sweep_one_factor(levels, reps, base_params, factor_name):
    """Run reps replications at each level of one factor."""
    rows = []
    total = len(levels) * reps
    done = 0
    print(f"  {total} runs ({len(levels)} levels x {reps} reps)")
    for lv in levels:
        params = dict(base_params)
        params[factor_name] = lv
        for r in range(reps):
            seed = SEED_OFFSET + r * 100 + int(round(lv * 1000))
            res = run_one(seed=seed, **params)
            row = {factor_name: lv, "rep": r, "seed": seed, **res}
            rows.append(row)
            done += 1
            if done % 50 == 0 or done == total:
                print(f"    {done}/{total}")
    return rows


def _cell_seed(x, g, rep):
    """Deterministic, distinct, NON-NEGATIVE seed for a (x, group, rep) cell.

    Mesa 3.x feeds the seed to ``np.random.default_rng``, which rejects negative
    integers, so a factor that can be < 0 (e.g. remote_attrition_coef in
    [-1, 1]) must not be allowed to drive the seed negative. Each factor is
    encoded into a non-overlapping band; the group term is shifted by
    GROUP_SHIFT so it stays non-negative for any group value >= -GROUP_SHIFT.
    """
    GROUP_SHIFT = 100_000  # tolerates group values down to -100 (in milli-units)
    return (
        SEED_OFFSET
        + rep
        + 1_000 * int(round(x * 1000))
        + 10_000_000 * (int(round(g * 1000)) + GROUP_SHIFT)
    )


def sweep_two_factor(x_levels, group_levels, reps, base_params, x_name, group_name):
    """Cross-sweep: every (x, group) cell gets reps replications."""
    rows = []
    total = len(x_levels) * len(group_levels) * reps
    done = 0
    print(f"  {total} runs ({len(x_levels)}x{len(group_levels)} cells x {reps} reps)")
    for g in group_levels:
        for x in x_levels:
            params = dict(base_params)
            params[x_name] = x
            params[group_name] = g
            for r in range(reps):
                seed = _cell_seed(x, g, r)
                res = run_one(seed=seed, **params)
                row = {x_name: x, group_name: g, "rep": r, "seed": seed, **res}
                rows.append(row)
                done += 1
                if done % 60 == 0 or done == total:
                    print(f"    {done}/{total}")
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}")


def summarize(rows, group_keys, metric):
    """Group rows by group_keys, return list of dicts with mean/sd/ci."""
    groups = {}
    for r in rows:
        k = tuple(r[gk] for gk in group_keys)
        groups.setdefault(k, []).append(r[metric])
    out = []
    for k, vals in sorted(groups.items()):
        m = mean(vals)
        sd = stdev(vals) if len(vals) > 1 else 0.0
        # 95% CI of the mean assuming normal approx (t ~ 2.0 for n=20)
        ci = 2.093 * sd / (len(vals) ** 0.5)
        rec = {gk: gv for gk, gv in zip(group_keys, k)}
        rec.update({"n": len(vals), f"{metric}_mean": m, f"{metric}_sd": sd, f"{metric}_ci": ci})
        out.append(rec)
    return out


def plot_baseline(rows, path):
    summ = summarize(rows, ["remote_share"], "velocity")
    summ_w = summarize(rows, ["remote_share"], "avg_wait")
    xs = [s["remote_share"] for s in summ]
    ys = [s["velocity_mean"] for s in summ]
    cis = [s["velocity_ci"] for s in summ]
    yw = [s["avg_wait_mean"] for s in summ_w]
    cw = [s["avg_wait_ci"] for s in summ_w]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].errorbar(xs, ys, yerr=cis, fmt="-o", color="tab:blue", capsize=3)
    axes[0].set_title(f"Velocity vs remote_share (n_devs={N_DEVS})")
    axes[0].set_xlabel("remote_share")
    axes[0].set_ylabel("velocity (story points)")
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(xs, yw, yerr=cw, fmt="-o", color="tab:red", capsize=3)
    axes[1].set_title(f"Average wait per blocker vs remote_share (n_devs={N_DEVS})")
    axes[1].set_xlabel("remote_share")
    axes[1].set_ylabel("avg_wait (ticks)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    print(f"  wrote {path}")


def plot_interaction(rows, group_name, group_label, path):
    groups = sorted(set(r[group_name] for r in rows))
    summ = summarize(rows, [group_name, "remote_share"], "velocity")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(groups)))
    for color, g in zip(colors, groups):
        sub = [s for s in summ if s[group_name] == g]
        xs = [s["remote_share"] for s in sub]
        ys = [s["velocity_mean"] for s in sub]
        cis = [s["velocity_ci"] for s in sub]
        ax.errorbar(xs, ys, yerr=cis, fmt="-o", color=color, capsize=3,
                    label=f"{group_label}={g}")
    ax.set_title(f"Velocity vs remote_share, by {group_label} (n_devs={N_DEVS})")
    ax.set_xlabel("remote_share")
    ax.set_ylabel("velocity (story points)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    print(f"  wrote {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # Single-sprint base keeps Exp 1-3 comparable to the pre-extension report.
    base = dict(n_devs=N_DEVS, n_sprints=1)

    # Exp 1: baseline remote_share sweep, 11 levels
    print("[Exp 1] baseline remote_share sweep")
    levels1 = [round(i / 10, 1) for i in range(11)]
    rows1 = sweep_one_factor(levels1, REPS, base, "remote_share")
    write_csv(rows1, f"{OUT_DIR}/exp1_baseline.csv")
    plot_baseline(rows1, f"{OUT_DIR}/exp1_baseline.png")

    # Exp 2: remote_share x mean_solo_skill, 6 x 3 cells
    print("[Exp 2] remote_share x mean_solo_skill")
    levels_rs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    levels_skill = [0.3, 0.5, 0.7]
    rows2 = sweep_two_factor(levels_rs, levels_skill, REPS, base,
                             "remote_share", "mean_solo_skill")
    write_csv(rows2, f"{OUT_DIR}/exp2_skill_interaction.csv")
    plot_interaction(rows2, "mean_solo_skill", "mean_solo_skill",
                     f"{OUT_DIR}/exp2_skill_interaction.png")

    # Exp 3: remote_share x block_prob, 6 x 3 cells
    print("[Exp 3] remote_share x block_prob")
    levels_bp = [0.01, 0.02, 0.05]
    rows3 = sweep_two_factor(levels_rs, levels_bp, REPS, base,
                             "remote_share", "block_prob")
    write_csv(rows3, f"{OUT_DIR}/exp3_blockprob_interaction.csv")
    plot_interaction(rows3, "block_prob", "block_prob",
                     f"{OUT_DIR}/exp3_blockprob_interaction.png")

    if RUN_EXTENDED_SWEEPS:
        _run_extended_sweeps(levels_rs)

    print("done")


def _extended_sweep_specs():
    """The multi-sprint extended sweeps as data, so a subset can be re-run.

    Ext 1 sweeps the horizon itself, so its base omits ``n_sprints``; the rest
    run a fixed ~1-year horizon.
    """
    base_multi = dict(n_devs=N_DEVS, n_sprints=24)
    return [
        dict(name="ext1", label="remote_share x n_sprints (capability compounding)",
             group="n_sprints", levels=[6, 12, 24],
             base=dict(n_devs=N_DEVS), out="ext1_remote_x_nsprints.csv"),
        dict(name="ext2", label="remote_share x annual_attrition (churn cost)",
             group="annual_attrition", levels=[0.08, 0.12, 0.20],
             base=base_multi, out="ext2_remote_x_attrition.csv"),
        dict(name="ext3", label="remote_share x remote_attrition_coef (retention vs isolation)",
             group="remote_attrition_coef", levels=[-1.0, 0.0, 1.0],
             base=base_multi, out="ext3_remote_x_attrcoef.csv"),
        dict(name="ext4", label="remote_share x sync_help_weight (mentorship premium)",
             group="sync_help_weight", levels=[1.0, 1.5, 2.0],
             base=base_multi, out="ext4_remote_x_syncweight.csv"),
    ]


def _run_extended_sweeps(levels_rs, names=None):
    """Run the multi-sprint extended sweeps, optionally a subset by name.

    These write CSVs only (no plots); the headline metrics to analyse are
    `total_velocity`, `mean_skill`, and `attrition_count`. Each cell runs a
    full ~1-year horizon, so this is far slower than Exp 1-3.
    """
    for spec in _extended_sweep_specs():
        if names and spec["name"] not in names:
            continue
        print(f"[{spec['name']}] {spec['label']}")
        rows = sweep_two_factor(levels_rs, spec["levels"], REPS,
                                spec["base"], "remote_share", spec["group"])
        write_csv(rows, f"{OUT_DIR}/{spec['out']}")


if __name__ == "__main__":
    main()
