# Extended multi-sprint sweep findings (learning & turnover)

Date: 2026-05-21
Status: accepted
Author: Miłosz + Claude
Related to: docs/adr/0001_learning_and_turnover.md, docs/research/learning_turnover_parameters.md, calibrate_learning.py

---

## Setup

Four two-factor sweeps over `remote_share ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}` crossed with a second
factor, 20 replications per cell, `n_devs=10`, calibrated `learning_rate=0.0123`, full ~1-year
horizon (`n_sprints=24`) unless the horizon is the swept factor. Headline metric: **cumulative
velocity** (`total_velocity`) over the horizon — per the calibration finding, the remote→learning
coupling lives in cumulative output, not the asymptotic `mean_skill`. Run via `run_extended.py`,
analysed via `analyze_extended.py`; plots in `results/ext{1..4}_*_totalvel.png`.

## Headline result

Co-located teams deliver **+2.5–2.8% more cumulative velocity per year** than fully-remote teams.
The effect is monotonic in `remote_share` with tight CIs (±~20 SP on ~17,000), and robust across all
four sweeps.

## Finding 1 — the remote penalty is FRONT-LOADED, not compounding

ADR-0001 hypothesised that the remote velocity penalty would *compound* over the year as skill
diverged. The data shows the opposite: the **percentage** penalty is largest early and *shrinks* as
the horizon grows.

| Horizon | Office advantage (cumulative velocity) |
|---|---|
| 6 sprints (~3 mo) | **+4.69%** |
| 12 sprints (~6 mo) | +3.31% |
| 24 sprints (~1 yr) | **+2.64%** |

Decomposition by toggling the learning engine (24 sprints, office=remote_share 0 vs remote=1):

| Configuration | Office advantage |
|---|---|
| learning OFF (`learning_rate=0`) | **+6.40%** |
| learning ON (`learning_rate=0.0123`) | +2.44% |
| **net effect of the learning channel** | **−3.96 pp** |

**Mechanism.** The remote penalty is fundamentally a *help-channel* penalty (long asynchronous waits
when a remote dev needs help). Learning lifts *both* teams toward `skill_cap`, so over time both need
help less often — shrinking the very channel where remote loses. Learning therefore **attenuates** the
remote penalty rather than amplifying it, concentrating the cost in the early ramp phase (junior,
blocker-heavy) and letting it decay as the team matures. Without learning the model would *overstate*
the steady-state remote penalty by ~2.6× (6.40% vs 2.44%).

## Finding 2 — the turnover knob reproduces both literature scenarios

`remote_attrition_coef` is the contested headline DSS knob. Its signal lives in `attrition_count`, not
velocity (under immediate junior replacement, turnover is nearly velocity-neutral). Mean attrition
events over 24 sprints, office vs full-remote:

| `remote_attrition_coef` | office → full-remote | reading |
|---|---|---|
| −1 (retention, Bloom *Nature* 2024) | 1.90 → **0.70** | remote *reduces* quitting |
| 0 (agnostic) | 0.95 → 1.30 | ~neutral |
| +1 (isolation, Power of Proximity) | 0.60 → **2.05** | remote *raises* quitting |

The knob cleanly produces the two named scenarios the literature demands — reporting both arms is the
core DSS deliverable.

## Finding 3 — learning-channel levers are weak under default conditions

- **Ext 4** (`sync_help_weight` 1.0 → 2.0): office advantage flat (2.52% → 2.48%).
- **Ext 2** (`annual_attrition` 0.08 → 0.20): office advantage 2.47% → 2.77% (barely above noise).

Both are weak because at `block_prob=0.02` most learning weight comes from **task completion**
(location-independent), not help (location-dependent), and a team starting at `mean_solo_skill=0.5`
plateaus fast. To make the learning coupling material, raise `block_prob` and/or use junior-heavy
starting cohorts.

## Implications

- **DSS recommendation is team-state-contingent**, not a blanket verdict: remote is cheap for mature,
  stable, high-skill teams; costly for junior-heavy / high-churn / onboarding teams (where the cost
  front-loads). This conditional structure is exactly what a decision-support instrument should output.
- **Headline metrics** for the thesis: cumulative velocity, time-to-plateau, and the turnover
  retention/isolation scenarios — not asymptotic `mean_skill`.
- **Method (ABM) value:** the model surfaced a non-obvious *emergent* result (attenuation, not
  compounding) that contradicts the naive prior — a direct argument for ABM over static regression. It
  also defines the behavioural target the Generative-ABM / LLM engine must reproduce to claim validity.

## Limitations

- Single undifferentiated `remote_attrition_coef` — cannot capture the seniority-conditioning the
  literature shows (future ADR-0002).
- Learning-channel signal is muted at low `block_prob` and `mean_solo_skill=0.5` start; the strong-form
  learning coupling needs a blocker-heavy / junior-heavy regime to become material.
