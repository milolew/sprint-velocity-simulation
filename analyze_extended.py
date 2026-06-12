"""Analyse the extended multi-sprint sweeps (``results/ext{1..4}_*.csv``).

Headline metric is CUMULATIVE velocity (``total_velocity``) over the full
horizon: per the calibration finding (calibrate_learning.py / ADR-0001), the
remote->learning coupling lives in cumulative output, NOT the asymptotic
``mean_skill`` (which converges once both teams plateau). For each sweep this
prints a summary table, the office (remote_share=0) vs full-remote
(remote_share=1) gap, and writes a comparison PNG.

Run after run_extended.py: ``python analyze_extended.py``
"""

from __future__ import annotations

import csv
import os
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "results"
T_95 = 2.093  # Student-t for n=20 reps, two-sided 95%

# (csv file, grouping factor, human label)
SWEEPS = [
    ("ext1_remote_x_nsprints.csv", "n_sprints", "horizon (sprints)"),
    ("ext2_remote_x_attrition.csv", "annual_attrition", "annual attrition"),
    ("ext3_remote_x_attrcoef.csv", "remote_attrition_coef", "remote attrition coef"),
    ("ext4_remote_x_syncweight.csv", "sync_help_weight", "sync-help weight"),
]


def read_csv(path: str) -> list[dict]:
    """Read a sweep CSV, coercing numeric columns to float."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (ValueError, TypeError):
                pass
    return rows


def summarize(rows: list[dict], group_keys: list[str], metric: str) -> list[dict]:
    """Mean / sd / 95% CI of ``metric`` grouped by ``group_keys``."""
    groups: dict[tuple, list[float]] = {}
    for r in rows:
        key = tuple(r[gk] for gk in group_keys)
        groups.setdefault(key, []).append(r[metric])
    out = []
    for key, vals in sorted(groups.items()):
        m = mean(vals)
        sd = stdev(vals) if len(vals) > 1 else 0.0
        ci = T_95 * sd / (len(vals) ** 0.5)
        rec = dict(zip(group_keys, key))
        rec.update({"n": len(vals), "mean": m, "sd": sd, "ci": ci})
        out.append(rec)
    return out


def office_vs_remote_gap(rows: list[dict], group_name: str, metric: str) -> list[dict]:
    """Per group level, the metric at remote_share=0 vs =1 and the % gap."""
    summ = summarize(rows, [group_name, "remote_share"], metric)
    gaps = []
    for g in sorted({s[group_name] for s in summ}):
        office = next((s["mean"] for s in summ
                       if s[group_name] == g and s["remote_share"] == 0.0), None)
        remote = next((s["mean"] for s in summ
                       if s[group_name] == g and s["remote_share"] == 1.0), None)
        if office is None or remote is None or remote == 0:
            continue
        gaps.append({group_name: g, "office": office, "remote": remote,
                     "gap_pct": (office - remote) / remote * 100.0})
    return gaps


def print_table(rows: list[dict], group_name: str, label: str) -> None:
    """Print total_velocity / mean_skill / attrition by (group, remote_share)."""
    print(f"\n{'=' * 78}\n{label.upper()}  ({group_name})\n{'=' * 78}")
    tv = summarize(rows, [group_name, "remote_share"], "total_velocity")
    ms = {(s[group_name], s["remote_share"]): s["mean"]
          for s in summarize(rows, [group_name, "remote_share"], "mean_skill")}
    at = {(s[group_name], s["remote_share"]): s["mean"]
          for s in summarize(rows, [group_name, "remote_share"], "attrition_count")}

    print(f"  {group_name:>18} | remote |   total_vel ± ci   | mean_skill | attrition")
    print(f"  {'-' * 18}-+--------+--------------------+------------+----------")
    for s in tv:
        g, rs = s[group_name], s["remote_share"]
        print(f"  {g:>18} |  {rs:>4.1f} | {s['mean']:>8.0f} ± {s['ci']:>5.0f} "
              f"|   {ms[(g, rs)]:>6.4f}   |  {at[(g, rs)]:>5.2f}")

    print(f"\n  Office (remote=0) vs full-remote (remote=1) — CUMULATIVE VELOCITY:")
    for gap in office_vs_remote_gap(rows, group_name, "total_velocity"):
        print(f"    {group_name}={gap[group_name]:<8}: office {gap['office']:>8.0f}  "
              f"remote {gap['remote']:>8.0f}  ->  office "
              f"{'+' if gap['gap_pct'] >= 0 else ''}{gap['gap_pct']:.2f}%")


def plot_sweep(rows: list[dict], group_name: str, label: str, path: str) -> None:
    """total_velocity vs remote_share, one line per group level, 95% CI bands."""
    groups = sorted({r[group_name] for r in rows})
    summ = summarize(rows, [group_name, "remote_share"], "total_velocity")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(groups)))
    for color, g in zip(colors, groups):
        sub = [s for s in summ if s[group_name] == g]
        xs = [s["remote_share"] for s in sub]
        ys = [s["mean"] for s in sub]
        cis = [s["ci"] for s in sub]
        ax.errorbar(xs, ys, yerr=cis, fmt="-o", color=color, capsize=3,
                    label=f"{label}={g}")
    ax.set_title(f"Cumulative velocity vs remote_share, by {label}")
    ax.set_xlabel("remote_share")
    ax.set_ylabel("total velocity (story points, full horizon)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    print(f"  wrote {path}")


def main() -> None:
    for fname, group_name, label in SWEEPS:
        path = os.path.join(OUT_DIR, fname)
        if not os.path.exists(path):
            print(f"  SKIP (missing): {path}")
            continue
        rows = read_csv(path)
        print_table(rows, group_name, label)
        plot_sweep(rows, group_name, label,
                   os.path.join(OUT_DIR, fname.replace(".csv", "_totalvel.png")))


if __name__ == "__main__":
    main()
