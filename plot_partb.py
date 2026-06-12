"""Story figures for Part B of the report (learning + turnover).

Three purpose-built figures, rather than the raw sweep line-plots:
  1. partb_penalty.png   - the remote penalty is front-loaded, and learning attenuates it
  2. partb_skillramp.png - the mechanism: office teams ramp faster, then both plateau
  3. partb_turnover.png  - turnover scenarios: remote helps or hurts retention

Run after the extended sweeps: ``python plot_partb.py``
"""

from __future__ import annotations

import os
from statistics import mean

import matplotlib.pyplot as plt

from analyze_extended import OUT_DIR, read_csv, summarize
from calibrate_learning import skill_path_and_velocity
from run_experiments import run_one, N_DEVS

OFFICE, REMOTE = "#2a9d8f", "#e76f51"   # consistent palette across the figures
LR = 0.0123                              # calibrated default
REPS = 20


def _annotate_bars(ax, bars, fmt="{:+.1f}%"):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h, fmt.format(h),
                ha="center", va="bottom", fontsize=10, fontweight="bold")


def office_advantage_by_horizon():
    """Office vs full-remote cumulative-velocity advantage (%) per horizon, from ext1."""
    rows = read_csv(os.path.join(OUT_DIR, "ext1_remote_x_nsprints.csv"))
    summ = summarize(rows, ["n_sprints", "remote_share"], "total_velocity")
    out = {}
    for ns in sorted({s["n_sprints"] for s in summ}):
        o = next(s["mean"] for s in summ if s["n_sprints"] == ns and s["remote_share"] == 0.0)
        r = next(s["mean"] for s in summ if s["n_sprints"] == ns and s["remote_share"] == 1.0)
        out[int(ns)] = (o - r) / r * 100
    return out


def office_advantage_learning_on_off():
    """Office advantage (%) over a year with learning off vs on (live sims)."""
    def adv(lr):
        def vel(rs):
            return mean(run_one(seed=2000 + r, n_devs=N_DEVS, n_sprints=24,
                                remote_share=rs, learning_rate=lr)["total_velocity"]
                        for r in range(REPS))
        o, r = vel(0.0), vel(1.0)
        return (o - r) / r * 100
    return adv(0.0), adv(LR)


def turnover_scenarios():
    """Mean yearly departures, office vs full-remote, for retention/isolation, from ext3."""
    rows = read_csv(os.path.join(OUT_DIR, "ext3_remote_x_attrcoef.csv"))
    summ = summarize(rows, ["remote_attrition_coef", "remote_share"], "attrition_count")

    def at(coef, rs):
        return next(s["mean"] for s in summ
                    if s["remote_attrition_coef"] == coef and s["remote_share"] == rs)
    return {
        "retention\n(remote keeps people)": (at(-1.0, 0.0), at(-1.0, 1.0)),
        "isolation\n(remote pushes them out)": (at(1.0, 0.0), at(1.0, 1.0)),
    }


def fig_penalty():
    horizon = office_advantage_by_horizon()
    off, on = office_advantage_learning_on_off()

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.6))

    labels = [f"{ns}\n(~{ns // 2} mo)" for ns in horizon]
    bars = ax0.bar(labels, list(horizon.values()), color=OFFICE, width=0.6)
    _annotate_bars(ax0, bars)
    ax0.set_title("The remote penalty is front-loaded")
    ax0.set_xlabel("horizon (sprints)")
    ax0.set_ylabel("office advantage in cumulative velocity (%)")
    ax0.set_ylim(0, max(horizon.values()) * 1.25)
    ax0.grid(axis="y", alpha=0.3)

    bars = ax1.bar(["learning OFF", "learning ON"], [off, on],
                   color=["#bdbdbd", OFFICE], width=0.6)
    _annotate_bars(ax1, bars)
    ax1.set_title("Learning attenuates the penalty")
    ax1.set_ylabel("office advantage over a year (%)")
    ax1.set_ylim(0, off * 1.25)
    ax1.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "partb_penalty.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def fig_skillramp():
    n = 24
    office, _ = skill_path_and_velocity(LR, n, 0.0)
    remote, _ = skill_path_and_velocity(LR, n, 1.0)
    sprints = list(range(1, n + 1))
    cap = 0.90

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(cap, ls="--", color="gray", lw=1, label="skill ceiling")
    ax.plot(sprints, office, "-o", color=OFFICE, ms=4, label="office (face-to-face help)")
    ax.plot(sprints, remote, "-o", color=REMOTE, ms=4, label="fully remote (async help)")
    ax.fill_between(sprints, remote, office, color=OFFICE, alpha=0.12)

    peak = max(range(n), key=lambda i: office[i] - remote[i])
    ax.annotate(f"widest gap\nsprint {peak + 1}",
                xy=(peak + 1, (office[peak] + remote[peak]) / 2),
                xytext=(peak + 4, remote[peak] - 0.06),
                arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)

    ax.set_title("Why: office teams ramp faster, then both plateau")
    ax.set_xlabel("sprint")
    ax.set_ylabel("mean solo skill (chance of clearing a blocker alone)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "partb_skillramp.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def fig_turnover():
    data = turnover_scenarios()
    scenarios = list(data)
    x = range(len(scenarios))
    w = 0.36
    office_vals = [data[s][0] for s in scenarios]
    remote_vals = [data[s][1] for s in scenarios]

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar([i - w / 2 for i in x], office_vals, w, label="office", color=OFFICE)
    b2 = ax.bar([i + w / 2 for i in x], remote_vals, w, label="fully remote", color=REMOTE)
    _annotate_bars(ax, b1, fmt="{:.1f}")
    _annotate_bars(ax, b2, fmt="{:.1f}")

    ax.set_title("Turnover: remote work helps or hurts, depending on the channel")
    ax.set_ylabel("mean departures over a year")
    ax.set_xticks(list(x))
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "partb_turnover.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig_penalty()
    fig_skillramp()
    fig_turnover()
    print("done")


if __name__ == "__main__":
    main()
