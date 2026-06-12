"""Calibrate ``learning_rate`` to a literature-grounded ramp-up trajectory.

WHY this script exists
----------------------
The per-event ``learning_rate`` cannot be read off the literature: no study
measures "how much skill a developer gains from resolving one blocker". It is
an internal model quantity. What IS empirically grounded is the *ramp-up
trajectory*: a new developer reaches near-full productivity (an early, hard
plateau) in roughly **6 months** (median; 3-12 month band) — Sackman 1968,
Dieste/Tubio 2017, Peitek 2022, Microsoft Research Ramp-Up (Rastogi 2015),
DX time-to-10th-PR ~91 days, GitLab 2024 DevSecOps.

So we *calibrate* ``learning_rate`` to reproduce that documented trajectory.
This is pattern-oriented calibration to a stylized fact (Grimm et al.) — the
defensible move for a thesis that evaluates ABM as a method.

THE MATH
--------
The deterministic update is ``new = old + lr * weight * (cap - old)``, i.e. the
gap to the ceiling shrinks geometrically::

    gap_{n+1} = gap_n * (1 - lr * weight)

After a stream of events with weights w_1..w_n::

    gap_n / gap_0 = prod_j (1 - lr * w_j)  ~=  exp(-lr * sum_j w_j)

To close a fraction ``f`` of the gap we need ``sum_j (lr * w_j) = ln(1/(1-f))``,
so the analytic estimate is ``lr ~= ln(1/(1-f)) / W`` where ``W`` is the total
learning weight a developer accumulates over the target window. ``W`` is the
piece that was missing until now ("pending the events-per-sprint count" in the
research doc); we measure it from the simulation. The bisection below is the
authoritative answer (it captures the event mix shifting as skill rises); the
analytic formula is the cross-check and the explanation for the thesis.

Run: ``python calibrate_learning.py``
"""

from __future__ import annotations

import math
from statistics import mean

import model as model_mod
from model import TeamModel

# --- Calibration target (from the literature; see module docstring) ----------
TARGET_GAP_CLOSURE = 0.90          # close 90% of the gap to skill_cap ...
TARGET_SPRINTS = 12                # ... within 12 sprints (~6 months @ 2wk)
ROBUSTNESS_SPRINTS = (6, 12, 24)   # 3 / 6 / 12-month band for reporting

# --- Representative DoE conditions to calibrate under -------------------------
CALIB_BLOCK_PROB = 0.02            # central DoE level (run_experiments Exp 3)
CALIB_REMOTE_SHARE = 0.5           # neutral: a sync/async help blend
CALIB_N_DEVS = 10                  # DoE base team size
CALIB_SEEDS = tuple(range(7000, 7020))  # average over 20 seeds for stability

# --- Bisection settings -------------------------------------------------------
LR_LOW, LR_HIGH = 1e-4, 0.05       # learning_rate is monotone in mean skill
BISECT_TOL = 1e-4                  # tolerance on the achieved skill target
BISECT_MAX_ITERS = 60


def _junior_start_skill() -> tuple[float, float]:
    """The junior start skill (``junior_mean_skill * skill_cap``) and the cap.

    Read from a reference model so the script never drifts from the defaults.
    """
    ref = TeamModel(seed=0, n_sprints=1)
    return ref.junior_mean_skill * ref.skill_cap, ref.skill_cap


def _build_junior_team(learning_rate: float, n_sprints: int, *, seed: int,
                       remote_share: float = CALIB_REMOTE_SHARE) -> TeamModel:
    """A turnover-free team that all starts at the junior skill level.

    Turnover off ⇒ the founding cohort is tracked end to end (no churn noise);
    a zero spread makes every founder start at exactly the junior level so the
    measured mean skill is the clean ramp of a single representative junior.
    """
    start_skill, _cap = _junior_start_skill()
    return TeamModel(
        seed=seed,
        n_devs=CALIB_N_DEVS,
        n_sprints=n_sprints,
        block_prob=CALIB_BLOCK_PROB,
        remote_share=remote_share,
        mean_solo_skill=start_skill,
        solo_skill_spread=0.0,
        learning_rate=learning_rate,
        annual_attrition=0.0,
    )


def _mean_skill_at_horizon(m: TeamModel) -> float:
    return sum(d.solo_skill for d in m.devs) / len(m.devs)


def skill_path_and_velocity(learning_rate: float, n_sprints: int,
                            remote_share: float) -> tuple[list[float], float]:
    """Per-sprint mean-skill path and total velocity, averaged over seeds.

    Used to locate WHERE the office/remote coupling shows up: the end-skill
    asymptotes converge once both teams plateau, so the signal must be read off
    the ramp (mid-run divergence) and the cumulative output instead.
    """
    paths = [[] for _ in range(n_sprints)]
    velocities = []
    for seed in CALIB_SEEDS:
        m = _build_junior_team(learning_rate, n_sprints, seed=seed,
                               remote_share=remote_share)
        recorded = 0
        while m.running:
            m.step()
            if len(m.sprint_velocities) > recorded:   # a sprint just closed
                paths[recorded].append(_mean_skill_at_horizon(m))
                recorded += 1
        velocities.append(sum(m.sprint_velocities))
    avg_path = [mean(col) for col in paths]
    return avg_path, mean(velocities)


def mean_skill_after(learning_rate: float, n_sprints: int,
                     *, remote_share: float = CALIB_REMOTE_SHARE) -> float:
    """Mean junior-cohort skill after ``n_sprints``, averaged over seeds."""
    finals = []
    for seed in CALIB_SEEDS:
        m = _build_junior_team(learning_rate, n_sprints, seed=seed,
                               remote_share=remote_share)
        while m.running:
            m.step()
        finals.append(_mean_skill_at_horizon(m))
    return mean(finals)


def calibrate_learning_rate(target_skill: float) -> float:
    """Bisect ``learning_rate`` so the cohort hits ``target_skill`` at the
    target horizon. Mean skill is monotone increasing in ``learning_rate``."""
    lo, hi = LR_LOW, LR_HIGH
    for _ in range(BISECT_MAX_ITERS):
        mid = (lo + hi) / 2
        achieved = mean_skill_after(mid, TARGET_SPRINTS)
        if abs(achieved - target_skill) < BISECT_TOL:
            return mid
        if achieved < target_skill:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def measure_total_weight(learning_rate: float, n_sprints: int) -> float:
    """Total learning weight one developer accumulates over the window.

    Instruments the single learning seam (``Dev._apply_learning``) to sum every
    gain weight, then divides by the (constant, turnover-off) head count. This
    feeds the analytic cross-check ``lr ~= ln(1/(1-f)) / W``.
    """
    seen: list[float] = []
    original = model_mod.Dev._apply_learning

    def counting(self, weight):
        seen.append(weight)
        return original(self, weight)

    model_mod.Dev._apply_learning = counting
    try:
        per_seed_totals = []
        for seed in CALIB_SEEDS:
            seen.clear()
            m = _build_junior_team(learning_rate, n_sprints, seed=seed)
            while m.running:
                m.step()
            per_seed_totals.append(sum(seen) / len(m.devs))
    finally:
        model_mod.Dev._apply_learning = original
    return mean(per_seed_totals)


def main() -> None:
    start_skill, cap = _junior_start_skill()
    gap0 = cap - start_skill
    target_skill = cap - (1.0 - TARGET_GAP_CLOSURE) * gap0

    print("=" * 70)
    print("LEARNING-RATE CALIBRATION (pattern-oriented, to a ramp-up target)")
    print("=" * 70)
    print(f"  skill_cap                : {cap:.4f}")
    print(f"  junior start skill       : {start_skill:.4f}  "
          f"(= junior_mean_skill * cap)")
    print(f"  initial gap to cap       : {gap0:.4f}")
    print(f"  target                   : close {TARGET_GAP_CLOSURE:.0%} of the "
          f"gap in {TARGET_SPRINTS} sprints (~6 months)")
    print(f"  => target mean skill     : {target_skill:.4f} at sprint "
          f"{TARGET_SPRINTS}")
    print(f"  conditions               : block_prob={CALIB_BLOCK_PROB}, "
          f"remote_share={CALIB_REMOTE_SHARE}, n_devs={CALIB_N_DEVS}, "
          f"{len(CALIB_SEEDS)} seeds, turnover off")
    print()

    lr_star = calibrate_learning_rate(target_skill)
    achieved = mean_skill_after(lr_star, TARGET_SPRINTS)

    # Analytic cross-check from the measured weight budget.
    total_weight = measure_total_weight(lr_star, TARGET_SPRINTS)
    lr_analytic = math.log(1.0 / (1.0 - TARGET_GAP_CLOSURE)) / total_weight
    weight_per_sprint = total_weight / TARGET_SPRINTS

    print("-" * 70)
    print("RESULT")
    print("-" * 70)
    print(f"  calibrated learning_rate : {lr_star:.5f}  "
          f"(achieved skill {achieved:.4f} vs target {target_skill:.4f})")
    print(f"  learning weight / dev    : {total_weight:.2f} over "
          f"{TARGET_SPRINTS} sprints  (~{weight_per_sprint:.2f} / sprint)")
    print(f"  analytic cross-check     : lr ~= ln(10)/W = "
          f"{lr_analytic:.5f}  (small-step approx)")
    print()

    print("-" * 70)
    print("RAMP under the calibrated rate (mean skill at each milestone)")
    print("-" * 70)
    for ns in ROBUSTNESS_SPRINTS:
        ms = mean_skill_after(lr_star, ns)
        closed = (ms - start_skill) / gap0
        print(f"  after {ns:2d} sprints (~{ns // 2} months): "
              f"mean skill {ms:.4f}  ({closed:.0%} of gap closed)")
    print()

    print("-" * 70)
    print("WHERE THE COUPLING SHOWS UP: office (all sync) vs remote (all async)")
    print("  junior-level start, 24 sprints, calibrated rate")
    print("-" * 70)
    horizon = 24
    office_path, office_vel = skill_path_and_velocity(lr_star, horizon, 0.0)
    remote_path, remote_vel = skill_path_and_velocity(lr_star, horizon, 1.0)

    diffs = [o - r for o, r in zip(office_path, remote_path)]
    peak_sprint = max(range(horizon), key=lambda i: diffs[i])
    print(f"  end-skill   office {office_path[-1]:.4f}  remote "
          f"{remote_path[-1]:.4f}  gap {diffs[-1]:+.4f}  (asymptotes converge)")
    print(f"  peak ramp gap at sprint {peak_sprint + 1:>2} "
          f"(~{(peak_sprint + 1) // 2} months): office "
          f"{office_path[peak_sprint]:.4f}  remote "
          f"{remote_path[peak_sprint]:.4f}  gap {diffs[peak_sprint]:+.4f}")
    print(f"  cumulative velocity      : office {office_vel:.0f}  remote "
          f"{remote_vel:.0f}  "
          f"(office +{(office_vel - remote_vel) / remote_vel:.1%})")
    print()
    print("  => The end-skill asymptotes converge (both plateau within a year),")
    print("     so 'final mean_skill' is the WRONG headline metric. The remote")
    print("     penalty lives in the RAMP (peak mid-run skill gap above) and in")
    print("     CUMULATIVE VELOCITY. Report those, not the asymptote.")
    print()
    print(f"SUGGESTED DEFAULT: learning_rate = {lr_star:.4f}")


if __name__ == "__main__":
    main()
