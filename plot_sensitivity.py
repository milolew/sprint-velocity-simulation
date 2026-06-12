"""Figures for the remote_share sensitivity analysis.

Reads the raw per-run CSVs from sensitivity_remote.py and renders five figures
that each answer a specific assignment question:

  fig1  relationship       -- total_velocity & avg_wait vs remote_share (mean+-CI)
  fig2  dispersion         -- per-level distribution + sd(total_velocity) growth
  fig3  mean representative-- attrition (Poisson, mean lies) vs total_velocity (Gaussian)
  fig4  convergence        -- CI half-width vs #reps, throughput vs rare-event count
  fig5  secondary channels -- mean_skill & attrition_count vs remote_share (flat)

Usage:
    python plot_sensitivity.py
"""

import csv
import math
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "results"
MAIN = f"{OUT_DIR}/sens_remote_main.csv"
CONV = f"{OUT_DIR}/sens_remote_conv.csv"

NUM = {"remote_share", "seed", "total_velocity", "final_velocity", "avg_wait",
       "blockers_resolved", "completed_tasks", "mean_skill", "attrition_count"}


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in r:
            if k in NUM:
                r[k] = float(r[k])
    return rows


def group(rows, key="remote_share"):
    by = defaultdict(list)
    for r in rows:
        by[r[key]].append(r)
    return by


def col(rows, name):
    return [r[name] for r in rows]


def mean_ci(vals, z=1.96):
    m = np.mean(vals)
    sd = np.std(vals, ddof=1)
    hw = z * sd / math.sqrt(len(vals))
    return m, hw, sd


# --------------------------------------------------------------------------- #
def fig1_relationship(by):
    lv = sorted(by)
    tv = [mean_ci(col(by[l], "total_velocity")) for l in lv]
    aw = [mean_ci(col(by[l], "avg_wait")) for l in lv]

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].errorbar(lv, [t[0] for t in tv], yerr=[t[1] for t in tv],
                   fmt="-o", color="tab:blue", capsize=3, ms=4)
    ax[0].set_title("Cumulative throughput vs remote_share\n(mean $\\pm$ 95% CI; CI band is smaller than markers)")
    ax[0].set_xlabel("remote_share")
    ax[0].set_ylabel("total_velocity (story points / year)")
    ax[0].grid(True, alpha=0.3)
    # annotate the total drop
    ax[0].annotate(f"{100*(tv[-1][0]-tv[0][0])/tv[0][0]:+.1f}% over the full range",
                   xy=(0.5, 0.92), xycoords="axes fraction", ha="center",
                   fontsize=10, color="tab:blue")

    ax[1].errorbar(lv, [a[0] for a in aw], yerr=[a[1] for a in aw],
                   fmt="-o", color="tab:red", capsize=3, ms=4)
    ax[1].set_title("Average wait per blocker vs remote_share\n(the mechanistic driver)")
    ax[1].set_xlabel("remote_share")
    ax[1].set_ylabel("avg_wait (ticks)")
    ax[1].grid(True, alpha=0.3)
    ax[1].annotate(f"{100*(aw[-1][0]-aw[0][0])/aw[0][0]:+.0f}%",
                   xy=(0.5, 0.1), xycoords="axes fraction", ha="center",
                   fontsize=11, color="tab:red")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/sens_fig1_relationship.png", dpi=140)
    plt.close()
    print("wrote sens_fig1_relationship.png")


def fig2_dispersion(by):
    lv = sorted(by)
    sub = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    data = [col(by[l], "total_velocity") for l in sub]

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    parts = ax[0].violinplot(data, positions=sub, widths=0.12,
                             showmeans=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("tab:blue")
        pc.set_alpha(0.45)
    ax[0].set_title("Distribution of total_velocity at selected remote_share levels\n"
                    "(spread widens as remote_share rises)")
    ax[0].set_xlabel("remote_share")
    ax[0].set_ylabel("total_velocity (story points / year)")
    ax[0].grid(True, alpha=0.3)

    sds = [np.std(col(by[l], "total_velocity"), ddof=1) for l in lv]
    ax[1].plot(lv, sds, "-o", color="tab:purple", ms=4)
    ax[1].set_title("Run-to-run std. dev. of total_velocity vs remote_share\n"
                    f"(grows $\\approx\\times${sds[-1]/sds[0]:.1f} — risk is more sensitive than the mean)")
    ax[1].set_xlabel("remote_share")
    ax[1].set_ylabel("std. dev. of total_velocity")
    ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/sens_fig2_dispersion.png", dpi=140)
    plt.close()
    print("wrote sens_fig2_dispersion.png")


def fig3_representative(by):
    """Two observables, two verdicts on whether the mean is representative."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    # (a) attrition_count at rs=0 and rs=1 -- discrete, low-count -> mean lies
    for rs, color in [(0.0, "tab:green"), (1.0, "tab:orange")]:
        counts = Counter(int(x) for x in col(by[rs], "attrition_count"))
        n = sum(counts.values())
        ks = sorted(counts)
        xs = np.array(ks) + (0.15 if rs == 1.0 else -0.15)
        ax[0].bar(xs, [counts[k] / n for k in ks], width=0.3, color=color,
                  alpha=0.7, label=f"remote_share={rs} (mean={np.mean(col(by[rs],'attrition_count')):.2f})")
    ax[0].set_title("attrition_count is discrete & Poisson-like\n"
                    "the mean (~1.1) describes almost no single run")
    ax[0].set_xlabel("departures over the year (count)")
    ax[0].set_ylabel("share of runs")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # (b) total_velocity at rs=1 -- near-Gaussian, tight -> mean is representative
    vals = col(by[1.0], "total_velocity")
    m, hw, sd = mean_ci(vals)
    ax[1].hist(vals, bins=15, color="tab:blue", alpha=0.7, density=True)
    ax[1].axvline(m, color="k", lw=2, label=f"mean={m:.0f}")
    ax[1].axvline(np.median(vals), color="r", ls="--", lw=1.5,
                  label=f"median={np.median(vals):.0f}")
    ax[1].set_title(f"total_velocity at remote_share=1.0 is tight & ~Gaussian\n"
                    f"(CV={100*sd/m:.2f}% — the mean IS representative)")
    ax[1].set_xlabel("total_velocity (story points / year)")
    ax[1].set_ylabel("density")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/sens_fig3_representative.png", dpi=140)
    plt.close()
    print("wrote sens_fig3_representative.png")


def fig4_convergence(conv):
    by = group(conv)
    ns = [5, 10, 20, 40, 80, 160, 400]

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    # (a) throughput: CI half-width (% of mean) vs n, per level
    for rs, color in [(0.0, "tab:blue"), (0.5, "tab:green"), (1.0, "tab:orange")]:
        vals = col(by[rs], "total_velocity")
        m = np.mean(vals)
        hw = [100 * (1.96 * np.std(vals[:n], ddof=1) / math.sqrt(n)) / m for n in ns]
        ax[0].plot(ns, hw, "-o", color=color, ms=4, label=f"remote_share={rs}")
    ax[0].axhline(0.1, color="grey", ls=":", lw=1)
    ax[0].set_xscale("log")
    ax[0].set_title("Convergence of the total_velocity mean\n"
                    "(half-width of 95% CI, % of mean)")
    ax[0].set_xlabel("number of replications (log scale)")
    ax[0].set_ylabel("CI half-width (% of mean)")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3, which="both")

    # (b) throughput vs rare-event count: relative CI at rs=1.0
    vals_tv = col(by[1.0], "total_velocity")
    vals_at = col(by[1.0], "attrition_count")
    def relci(vals, n):
        sub = vals[:n]
        return 100 * (1.96 * np.std(sub, ddof=1) / math.sqrt(n)) / np.mean(sub)
    ax[1].plot(ns, [relci(vals_tv, n) for n in ns], "-o", color="tab:blue",
               ms=4, label="total_velocity (throughput)")
    ax[1].plot(ns, [relci(vals_at, n) for n in ns], "-s", color="tab:orange",
               ms=4, label="attrition_count (rare-event count)")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_title("Required reps depend on the observable (remote_share=1.0)\n"
                    "rare-event counts need orders of magnitude more reps")
    ax[1].set_xlabel("number of replications (log scale)")
    ax[1].set_ylabel("relative 95% CI half-width (%, log)")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/sens_fig4_convergence.png", dpi=140)
    plt.close()
    print("wrote sens_fig4_convergence.png")


def fig5_secondary(by):
    lv = sorted(by)
    sk = [mean_ci(col(by[l], "mean_skill")) for l in lv]
    at = [mean_ci(col(by[l], "attrition_count")) for l in lv]

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].errorbar(lv, [s[0] for s in sk], yerr=[s[1] for s in sk],
                   fmt="-o", color="tab:green", capsize=3, ms=4)
    ax[0].set_ylim(0.80, 0.92)
    ax[0].set_title("Capability channel: end-of-year mean_skill vs remote_share\n"
                    "(flat — remote work does NOT erode capability here)")
    ax[0].set_xlabel("remote_share")
    ax[0].set_ylabel("mean_skill at horizon")
    ax[0].grid(True, alpha=0.3)

    ax[1].errorbar(lv, [a[0] for a in at], yerr=[a[1] for a in at],
                   fmt="-o", color="tab:orange", capsize=3, ms=4)
    ax[1].set_title("Retention channel: attrition_count vs remote_share\n"
                    "(flat by design: remote_attrition_coef=0 — a control)")
    ax[1].set_xlabel("remote_share")
    ax[1].set_ylabel("departures over the year")
    ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/sens_fig5_secondary.png", dpi=140)
    plt.close()
    print("wrote sens_fig5_secondary.png")


def main():
    main_rows = load(MAIN)
    conv_rows = load(CONV)
    by = group(main_rows)
    fig1_relationship(by)
    fig2_dispersion(by)
    fig3_representative(by)
    fig4_convergence(conv_rows)
    fig5_secondary(by)
    print("all figures written")


if __name__ == "__main__":
    main()
