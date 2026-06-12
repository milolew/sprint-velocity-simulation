# ADR-0001: Multi-Sprint Horizon with Event-Driven Learning and Turnover

Date: 2026-05-21
Status: Accepted (implemented 2026-05-21; 125 tests pass, tester + code-reviewer signed off)
Author: Miłosz + Claude
Related to: results/REPORT.md (current single-sprint study), model.py

---

## Context

The model currently simulates **one sprint** (320 ticks = 10 working days) with developers
of *fixed* `solo_skill`. The thesis requires turning it into a Generative-ABM
decision-support model for hybrid-work policy over a ~1-year horizon, where the share of
remote work affects not only short-term velocity but the team's **long-term capability**
through two new mechanisms: developers **learn** (skill grows with experience) and
**turn over** (people leave and are replaced by juniors who must learn).

This reframes the research question from the static *"how does remote share affect this
sprint's velocity?"* to the dynamic *"how does hybrid-work policy affect long-term team
capability through learning and retention?"* — which is far more policy-relevant.

The constraint is **surgical extension, not rewrite**: the per-tick state machine,
two-phase activation, the sync/async help channel, and seeded determinism must survive.

---

## Decisions (fixed by the project owner)

1. **Horizon** — extend single-sprint → multi-sprint / ~1 year (~24 sprints).
2. **Learning engine** — purely **event-driven** (learning-by-doing): skill grows from
   completed tasks and resolved blockers. No time-on-the-job drift.
3. **Turnover** — a dev leaves with a per-period probability; replaced **immediately** by a
   junior new hire with low skill. No vacancy gap.
4. **Remote coupling** — `remote_share` affects **both** learning and turnover.
5. **In-flight leavers** — a dev who is currently helping someone or waiting for help has
   its departure **deferred** to the next sprint boundary at which it is in a settled state.
   (Chosen over full state-unwinding: provably safe, eliminates the entangled-cleanup risk.)
   Implemented as **re-roll**: such a dev is simply skipped at the current boundary and draws
   afresh next boundary (no persistent pending flag) — keeps the per-dev RNG-draw count constant
   and the practical difference is negligible (help sessions rarely coincide with a boundary).

The learning↔remote coupling is **indirect and reuses existing mechanics**: the skill gain
from resolving a blocker depends on *how* it was resolved (solo / sync-help / async-help).
Since daily location is rolled from `remote_share`, more remote ⇒ more async resolutions ⇒
slower skill growth. There is **no new "remote learning" knob**.

---

## Part 1 — Horizon refactor (single-sprint → multi-sprint run)

Today `sprint_length` does triple duty: per-sprint length, total run length, and stop
condition. Separate them:

- New param `n_sprints` (default 24 ≈ one year of 2-week sprints).
- `horizon_ticks = sprint_length * n_sprints`; stop condition becomes `tick >= horizon_ticks`.
- A **sprint boundary** is detected exactly like the existing day boundary:
  `tick > 0 and tick % sprint_length == 0`.

```
def step(self):
    if self.tick >= self.horizon_ticks:
        self.running = False
        return
    if self._is_day_boundary():        # tick % TICKS_PER_DAY == 0
        self._roll_locations()
    if self._is_sprint_boundary():     # tick % sprint_length == 0
        self._close_sprint()           # record + reset velocity, top up backlog
        self._run_turnover_checks()    # Part 3 — fires here, BEFORE pass 1
    # ... existing pass 1 / pass 2 unchanged ...
```

| Quantity | At sprint boundary | Rationale |
|---|---|---|
| `velocity` | record into `sprint_velocities: list[int]`, then reset to 0 | velocity is a per-sprint KPI; cumulative = `sum(sprint_velocities)` |
| `solo_skill`, tenure | **accumulate** (never reset) | learning/tenure are the point |
| `current_task`, `state`, blocker fields | **untouched** | a boundary is a reporting line, devs keep working across it (like day boundaries) |
| `backlog` | top up to `backlog_target` | a 24× longer run needs replenishment or it starves |
| `wait_times`, `completed_tasks` | accumulate (cumulative record) | long-run diagnostics |

`_close_sprint`, `_run_turnover_checks`, `_roll_locations` are separate small methods. The
per-tick state machine and two-phase activation are **unchanged**.

---

## Part 2 — Event-driven learning

`solo_skill` keeps its name and meaning (probability of resolving a blocker alone, used in
`_try_solo`) but now **moves**. Bounded diminishing-returns update toward a ceiling:

```
new_skill = old_skill + learning_rate * gain_weight * (skill_cap - old_skill)
```

The `(skill_cap - old_skill)` factor gives automatic diminishing returns and keeps skill
below `skill_cap` by construction (no per-event clamp needed). The update is **deterministic**
(no RNG) — cleanest for reproducibility; the variance the thesis needs already comes from the
stochastic *event stream* (which blockers occur, which channel resolves them).

Single seam: `Dev._apply_learning(self, weight: float)` reads `learning_rate`/`skill_cap`
from the model. Applied at:

| Event | Where | `gain_weight` |
|---|---|---|
| Task completion | `_step_working`, completion branch | `task_completion_weight` (baseline) |
| Solo blocker resolution | `_unblock("solo")` | `solo_resolution_weight` |
| SYNC help resolution (both in office) | `_unblock("sync")` | `sync_help_weight` (largest) |
| ASYNC help resolution (remote) | `_unblock("async")` | `async_help_weight` |

**Surgical change:** `_unblock` gains a `resolution: str` argument ("solo"/"sync"/"async")
because it currently cannot tell how the block was resolved. The sync/async distinction
already exists in `_try_find_helper`; store it on the requester as `help_channel` when the
session starts and read it in `_tick_help_session`. Invariant: `help_channel` is set iff
`helper` is set, and cleared in `_unblock` alongside `helper`/`help_timer`.

The learning curve form is strongly supported: developer productivity follows
**diminishing returns to an early, hard plateau**, and raw tenure beyond onboarding barely
predicts productivity (Sackman 1968; Dieste/Tubío et al. 2017; Peitek et al. 2022). This
justifies both the bounded form and a `skill_cap < 1.0` reached early.

---

## Part 3 — Turnover

**Fires at the sprint boundary** (not per tick): attrition is an HR-period event; per-sprint
(24 checks/year) is the natural granularity, simplifies the probability conversion, and lets
a leaving dev finish the tick cleanly.

**Annual rate → per-sprint hazard:**
```
p_leave_base = 1 - (1 - annual_attrition) ** (1 / sprints_per_year)
```
Computed once in `__init__`.

**Remote coupling (signed, parametrized):**
```
p_leave = clamp01(p_leave_base * (1 + remote_attrition_coef * (remote_share - remote_baseline)))
```
`remote_attrition_coef` sign/magnitude is the most contested parameter (see Part 8) and is
the headline DSS sensitivity knob. `remote_baseline` (0.5) centers the effect.

**Deferred departure (Decision 5):** if a chosen leaver is currently `helping` or
`blocked/waiting_for_help`, defer its departure to the next boundary where it is settled.
This removes the need to unwind another dev's help session — the hardest correctness case.

**Immediate junior replacement:** a leaver in a settled state is removed and replaced by a
new `Dev` whose skill is drawn from a low junior distribution **scaled to the ceiling**:
center `junior_mean_skill * skill_cap`, half-width `junior_skill_spread * skill_cap` (both
params are fractions of the cap), clamped to [0,1], starting `idle`, tenure 0. So
`junior_mean_skill = 0.40`, `skill_cap = 0.90` ⇒ junior skill ~U(0.27, 0.45).

**Settled-leaver cleanup** (the only cases that can occur after deferral):
- owns a `current_task` (working, or blocked at solo/ask_next_tick) → return task to backlog
  (`state="awaiting"`, `work_remaining = story_points * sp_scaling_k`; partial progress lost —
  a deliberate "knowledge walked out" choice), set `current_task=None`.
- idle → nothing to clean.

**Invariants after each replacement:** `len(devs) == n_devs`; every task is in exactly one of
{a dev's `current_task`, backlog, completed_tasks}; `set(self.devs) == set(self.agents)`
(Mesa AgentSet and our list stay in sync); no surviving dev references a removed dev.

---

## Part 4 — `remote_share` coupling points (complete)

| # | Entry point | Mechanism | Status |
|---|---|---|---|
| 1 | `_roll_locations` | sets home/office per dev with prob `remote_share` | existing |
| 2 | `_try_find_helper` | both-office ⇒ sync (large gain, short wait); else async (small gain, long wait) | existing mechanic, now feeds learning |
| 3 | `_apply_learning` via `gain_weight` | indirect: more remote ⇒ more async ⇒ slower growth | **new, no new knob** |
| 4 | `_run_turnover_checks` via `p_leave` | direct, signed via `remote_attrition_coef` | **new, parametrized** |

---

## Part 5 — Parameters, metrics, UI

**New `TeamModel.__init__` params** (defaults grounded in Part 8; `★` = empirically grounded,
sweep recommended):

```
# Horizon
n_sprints            = 24       # ~1 year, 2-week sprints
sprints_per_year     = 24
backlog_target       = 500      # backlog floor topped up each sprint boundary

# Learning  (per-EVENT rate; calibrate so a junior reaches the plateau in ~6 months ≈ 12 sprints)
learning_rate        = 0.0123   ★ per-event step; CALIBRATED to ~6mo-to-plateau (see post-impl notes)
skill_cap            = 0.90     ★ early hard plateau, < 1.0
task_completion_weight = 0.4    ★ routine-work baseline
solo_resolution_weight = 1.0    ★ reference
sync_help_weight     = 1.5      ★ co-located mentorship premium
async_help_weight    = 1.0      ★ remote help ≈ solo (the ONLY remote-learning difference is the sync premium)

# Turnover
annual_attrition     = 0.12     ★ voluntary; synthesis value, document as limitation
junior_mean_skill    = 0.40     ★ as fraction of skill_cap (~0.36 absolute at cap 0.90)
junior_skill_spread  = 0.10     ★ captures education/talent heterogeneity
remote_attrition_coef = 0.0     ★ agnostic base; sweep [-1, +1] (two named scenarios, Part 8)
remote_baseline      = 0.5      # neutral center (design choice, not empirical)
junior_threshold     = 0.40     # reporting-only: skill below this counts as "junior"
```

Validate at the boundary in `__init__` (fail fast): `0 ≤ skill_cap ≤ 1`, `learning_rate ≥ 0`,
`0 ≤ annual_attrition ≤ 1`, weights `≥ 0`, `n_sprints ≥ 1`, `remote_attrition_coef` finite.

**New DataCollector reporters:** `sprint`, `mean_skill`, `min_skill`, `max_skill`,
`n_juniors` (skill < `junior_threshold`), `attrition_count` (cumulative), `mean_tenure`
(derive from `hired_at_tick`: `m.tick - dev.hired_at_tick`, no per-tick mutation),
`sprint_velocity` (live), plus `sprint_velocities` series on the model.

**New `app.py` sliders:** `n_sprints`, `learning_rate`, `skill_cap`, `sync_help_weight`,
`async_help_weight`, `annual_attrition`, `junior_mean_skill`, and a **signed**
`remote_attrition_coef` slider `(-1.0 … 1.0)` so the contested parameter is visible. Add
plot components for `mean_skill` and for `attrition_count`/`mean_tenure`.

---

## Part 6 — Impact on existing tests & experiments

**Strategy:** gate every new mechanic behind a parameter that defaults to *off in the test
fixture* (`learning_rate=0`, `annual_attrition=0`, `n_sprints=1`), so the existing suite
stays the pure-mechanics regression guard and passes byte-identically. New behavior gets new
modules: `test_learning.py`, `test_turnover.py`, `test_multisprint.py`.

| Test | Why it breaks | Action |
|---|---|---|
| `test_determinism.py` golden (338/134/22) | new mechanics shift the seed=42 trajectory | split: keep a **pure-mechanics golden** (learning/turnover off) matching 338/134/22 byte-for-byte (proves old paths untouched) + add a **new full-model golden** after review |
| `test_state_machine/blockers/help.py` | `_unblock(resolution)` signature + `help_channel` field | set `learning_rate=0` in fixtures; wait arithmetic unaffected (learning adds no ticks) |
| `conftest.py::DEFAULT_OVERRIDES` | single-sprint defaults | add `n_sprints=1, learning_rate=0.0, annual_attrition=0.0` — highest-leverage change to keep suite green |
| `experiment.py` / `test_experiment.py` | `EXPECTED_KEYS` fixed set | extend `run_one` return + `EXPECTED_KEYS` with final `mean_skill`, `attrition_count`, etc. |

**New experiment dimensions:** `remote_share × n_sprints` (does the velocity penalty
*compound* over a year as skill diverges? — the headline result), `remote_share ×
annual_attrition`, `remote_attrition_coef` sign sweep (retention vs isolation scenarios),
`sync_help_weight − async_help_weight` gap sensitivity (the lever carrying the learning
coupling), `junior_mean_skill × annual_attrition` (churn cost).

---

## Part 7 — Determinism & reproducibility

- **Learning gains deterministic** (no RNG) — removes an entire class of ordering concerns.
- **Leave/hire draws use `self.random` only**, at a fixed point in fixed order: in
  `_run_turnover_checks`, iterate a fixed-order copy; **collect all leave decisions first**
  (one `random()` per dev), *then* apply retirements — so the RNG-call count per dev is
  constant regardless of how many leave. New-hire skill drawn once per actual retirement.
  Document this ordering in the `step()` docstring (as the two-phase ordering already is).
- **Mesa add/remove pitfalls:** `super().__init__(model)` auto-adds the hire to `self.agents`
  — also `self.devs.append(hire)`. On retire call `dev.remove()` **and**
  `self.devs.remove(dev)`. The scheduler iterates `self.devs`, so a stale `self.agents`
  silently corrupts agent-based reporters — assert `set(self.devs) == set(self.agents)` in a
  test. Never mutate `self.devs` while iterating it (decide-all-then-apply enforces this).
- **Backlog top-up vs turnover ordering:** top up backlog first, then turnover; fix and
  document this so the RNG stream is stable.

---

## Part 8 — Empirical grounding (literature review, two research passes)

> Limitation to state in the thesis: **no source reports the exact constructs the model uses**
> (population-level voluntary developer turnover; "probability of resolving a blocker solo" by
> tenure). Values below are defensible syntheses/reconstructions from adjacent measures, to be
> used as base cases for **sensitivity analysis**, not precise constants.

**`annual_attrition` = 0.12 (sweep 0.08–0.20).** Tech/developer voluntary turnover converges
on ~10–15%/yr in normal (post-2022) conditions; the cooling 2023–2026 labor market (BLS JOLTS
quits ~2.0%/mo, down from the 2021 ~3.0% peak) centers the mass slightly below the often-cited
13.2%. No clean voluntary-only, developer-specific base rate exists in the literature.
Junior:senior attrition runs **~2–3×** (Visier 23M records: ages 20–25 quit ~3× older cohorts;
ICs ~3× leadership) — supporting evidence for a future tenure-dependent extension (not in scope).
*Sources:* BLS JOLTS & Employee Tenure 2024; LinkedIn (2018, blended); Mercer 2025 US Turnover
Survey (voluntary 13.0%); DevSkiller 2025; Pragmatic Engineer (~10% normal, 6–8% regretted);
Visier resignation-wave analysis; MIS Quarterly 2023 (IT turnover meta-analysis).

**Learning curve.** `junior_mean_skill ≈ 0.40` of `skill_cap` (0.35 = pessimistic edge):
expert↔novice comprehension ~2:1 implies a floor near ⅓–½, and the "experience explains little
variance" findings argue against a very low floor. `skill_cap ≈ 0.90` (early hard plateau).
`learning_rate` per-event, calibrated by simulation so a junior closes ~90% of the gap in
~6 months ≈ 12 sprints (effective per-sprint ~0.15–0.20; 3–12 month full-productivity range as
the robustness band). Shape — diminishing returns to an early, hard plateau — confirmed across
54 years and three methods. *Sources:* Sackman, Erikson & Grant 1968; Dieste/Tubío et al. 2017
(Empirical Software Engineering); Peitek et al. 2022 (ESEC/FSE, EEG+eye-tracking); Microsoft
Research "Ramp-Up Journey" (Rastogi et al. 2015); DX "time-to-10th-PR" ~91 days; GitLab 2024
DevSecOps; Wright's-law experience curves.

**Sync vs async learning (the learning coupling).** Co-located developers receive ~**18–22%**
more code feedback (NBER "Power of Proximity" regression coeff. 18.3%, raw ~22%); ~74% of that
advantage vanishes when teams go remote; the benefit concentrates on **juniors**. Peer-reviewed
call-center evidence shows a smaller ~8–12% remote knowledge/performance gap for lower-skill
work. → `sync_help_weight : async_help_weight ≈ 1.2× general, 1.5–2× for juniors`. Since
help-when-blocked is a mentorship-heavy moment, the model uses **sync 1.5 : async 1.0 (=1.5×)**;
this ratio is the **weakest-evidenced** parameter and should carry the widest sweep.
*Sources:* Emanuel, Harrington & Pallais "Power of Proximity" (NBER WP 31880, 2023; NY Fed
summary 2024); Emanuel & Harrington (AEJ: Applied 2024); Yang, Holtz et al. (Nature Human
Behaviour 2021).

**`remote_attrition_coef` — contested, the headline DSS knob.** Sign is genuinely split and
depends on dose (hybrid vs full remote), seniority, and voluntariness. Base case **0.0**
(agnostic — the two channels roughly offset in an undifferentiated model), with two named
sensitivity scenarios:
- **Retention scenario (coef < 0, e.g. −0.5 … −1.0):** voluntary hybrid *reduces* quitting by
  ~33% (Bloom, Han & Liang, **RCT in Nature 2024**, 1,612 incl. engineers, 2-day hybrid; up to
  ~40% for non-managers) — the highest-quality evidence.
- **Isolation scenario (coef > 0, e.g. +0.5 … +1.0):** losing established in-person mentorship
  *raises* junior quitting (+1.2 pp ≈ "~5× relative" for previously-co-located juniors, Power of
  Proximity; corroborated by remote-onboarding resignation studies). Thinnest leg → widest band.

The model cannot capture the seniority-conditioning the literature shows (single undifferentiated
coefficient) — note as a limitation / future extension. Reporting outcomes under *both* scenarios
is the core DSS contribution. *Sources:* Bloom, Han & Liang (Nature 630, 2024); Emanuel et al.
"Power of Proximity" (2023); Ericsson remote-onboarding resignations (arXiv 2510.05878, 2025);
Gartner / Visier / OWL Labs (survey-level, supporting).

---

## Risks

1. **In-flight leaver corruption** — *mitigated* by Decision 5 (deferral): a helping/waiting dev
   never leaves mid-session, so no foreign help-session unwinding is needed. Add a `test_turnover.py`
   fuzz test (high `annual_attrition`, many seeds, assert invariants every sprint).
2. **Golden-test re-baselining hides bugs** — *mitigated* by the split-golden approach (Part 6):
   the pure-mechanics golden must still equal 338/134/22.
3. **Parameter credibility** — *mitigated* by treating all `★` values as sensitivity-sweep base
   cases with documented ranges, and stating the construct-mismatch limitation explicitly.

## Consequences

- `model.py` grows ~120–150 LOC across small methods; stays well under the file/function ceilings;
  no rewrite warranted (additive around existing seams).
- We give up: partial-progress preservation on turnover, seniority-conditioned attrition, and
  (initially) stochastic learning gains — all noted as future extensions.
- **Revisit when** the thesis needs vacancy gaps, tenure-dependent attrition, or variable team size
  — any of those breaks the "constant size, immediate replacement" invariant and warrants ADR-0002.

---

## Post-implementation notes (2026-05-21)

- **`learning_rate` calibrated, not guessed (resolved 2026-05-21).** The per-event rate cannot come
  from the literature (no study measures skill gain per blocker resolution); what is grounded is the
  *ramp-up trajectory* — ~6 months to plateau (3–12 mo band). So `learning_rate` is calibrated by
  pattern-oriented fitting to that stylized fact (`calibrate_learning.py`): a junior closes ~90% of
  the gap to `skill_cap` in ~12 sprints, at neutral `remote_share=0.5`, central `block_prob=0.02`,
  turnover off. **Result `learning_rate = 0.0123`** (bisection), confirmed by the analytic
  cross-check `lr ≈ ln(10)/W = 0.01234` where `W ≈ 187` is the measured learning-weight budget per
  dev over 12 sprints (~15.6/sprint). This is now the model default (was the `0.015` placeholder).
  Ramp under the fitted rate: 68% of gap closed by 3 mo, 90% by 6 mo, 99% by 12 mo.
- **Key finding — the remote→learning coupling lives in the RAMP, not the asymptote.** Even at the
  calibrated rate, office vs remote *end*-skill converge (~0.8954 vs 0.8939, gap +0.0015) because both
  teams plateau within a year. The signal is in the ramp and in cumulative output: peak mid-run skill
  gap ~+0.028 (around sprint 4, ~2 months), and **cumulative velocity office +2.8% vs remote** over 24
  sprints — *purely through the learning channel* (`remote_attrition_coef=0`, so turnover and the
  direct velocity penalty are excluded). **Headline DSS metric must be cumulative velocity /
  time-to-plateau, NOT final `mean_skill`.** The coupling is also weak at low `block_prob` because most
  learning weight comes from task completion (location-independent), not help (location-dependent) — so
  the `sync_help_weight − async_help_weight` gap and `block_prob` are the levers that amplify it.
- **Determinism contract is locked** by two golden tests: pure-mechanics (gated off ⇒ 320/338/134/22,
  byte-identical to the pre-extension model — unchanged) and full-model (seed=42, all defaults ⇒ 7680
  ticks, Σvelocity 8441, 3008 completed, 683 waits, attrition 0, mean_skill 0.895828). The full-model
  golden was re-baselined when `learning_rate` moved 0.015 → 0.0123 (intentional, reviewed change).
  Turnover verified invariant-clean over 200 seeds under heavy churn.
- **The "compounding" hypothesis (Part 6) was empirically REFUTED — the penalty is front-loaded.** The
  extended sweeps (full findings: `docs/research/extended_sweeps_findings.md`) show the office
  advantage in cumulative velocity is +4.69% at 6 sprints, decaying to +2.64% at 24 sprints — the
  *percentage* penalty shrinks, not compounds. A learning ON/OFF decomposition (24 sprints) shows the
  office advantage is +6.40% with learning OFF vs +2.44% ON: **learning ATTENUATES the remote penalty
  (−3.96 pp)**, because the penalty is a help-channel cost and a skilled team needs help less often. So
  the remote cost concentrates in the ramp phase and the DSS recommendation is team-state-contingent
  (cheap for mature/stable teams, costly for junior-heavy/high-churn ones). The `remote_attrition_coef`
  knob cleanly reproduces both the retention (Bloom) and isolation (Proximity) scenarios in
  `attrition_count`. Learning-channel levers (`sync_help_weight`, `annual_attrition`) are weak at the
  default `block_prob=0.02` — most learning weight is from location-independent task completion.
