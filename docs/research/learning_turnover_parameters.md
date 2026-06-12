# Empirical grounding for learning & turnover mechanisms (hybrid-work ABM)

Date: 2026-05-21
Status: accepted
Author: Miłosz + Claude
Related to: docs/adr/0001_learning_and_turnover.md

---

## Purpose & method

This document grounds the parameters of two new ABM mechanisms — event-driven developer
learning and turnover — in published evidence, for citation in the thesis. Method: a
**two-pass, three-perspective literature review** (pass 1 = broad sourcing; pass 2 =
verification / gap-filling of load-bearing claims), synthesized by the orchestrator across
perspectives to surface convergence and conflict.

**Overarching limitation (state in the thesis):** no source measures the exact constructs the
model uses — population-level *voluntary developer* turnover, or the *"probability of resolving
a blocker solo"* as a function of tenure. Every value below is a defensible synthesis or
reconstruction from adjacent measures, intended as a **sensitivity-analysis base case**, not a
precise constant. Trust grades: **High** = government / large-sample / peer-reviewed
randomized or quasi-experimental; **Med** = large observational / curated practitioner; **Low**
= survey-intent / vendor blog.

---

## A. Developer voluntary turnover → `annual_attrition`

**Finding.** Tech/developer *voluntary* turnover converges on ~10–15%/yr under normal
(post-2022) conditions. The often-cited LinkedIn 13.2% is an 8-year-old (2018), blended
(voluntary+involuntary), industry-level figure — not clean voluntary-only developer data. The
cooling 2023–2026 labor market (BLS JOLTS quits ~2.0%/mo, down from the 2021 ~3.0% peak)
centers the mass slightly below 13%. **No clean voluntary-only, developer-specific population
base rate exists in the literature** — this is a genuine gap, not a search failure.

**Junior:senior multiplier.** Early-tenure/younger staff quit **~2–3×** more: Visier (23M
records, 22k+ companies) finds ages 20–25 resign ~2.4%/mo vs ≤0.8%/mo for older cohorts (~3×),
and individual contributors ~3× leadership; BLS CPS tenure corroborates the direction. Supports
a future *tenure-dependent* attrition extension (out of current scope).

**Verdict:** `annual_attrition` = **0.12** central (0.13 acceptable as deliberately
conservative), sweep **0.08–0.20** (stretch high to 0.22 for a hot-market shock scenario).
Junior:senior ≈ **2× central, 3× high** if/when seniority-conditioned.

**Sources:**
- BLS JOLTS (quits = voluntary separations) — https://www.bls.gov/jlt/ ; news release https://www.bls.gov/news.release/jolts.htm — *High*
- BLS Employee Tenure 2024 — https://www.bls.gov/news.release/pdf/tenure.pdf — *High*
- Mercer 2025 US Turnover Survey (voluntary 13.0%, 2,617 orgs) — https://www.imercer.com/articleinsights/workforce-turnover-trends — *Med-High*
- LinkedIn Talent turnover (2018, blended, tech 13.2%) — https://news.linkedin.com/2018/3/linkedin-data-analysis-reveals-the-latest-talent-turnover-trends — *Med (dated, blended)*
- DevSkiller IT Skills Report 2025 (software ~15%, IT ~10%) — https://skillpanel.com/it-skills-report/ — *Low-Med (vendor)*
- Pragmatic Engineer — Good/Bad Attrition (~10% normal, 6–8% regretted, >20% concerning) — https://newsletter.pragmaticengineer.com/p/attrition — *Med (curated practitioner)*
- Visier resignation-wave analysis (junior:senior ~3×) — https://www.visier.com/blog/four-things-we-learned-about-the-resignation-wave-and-what-to-do-next/ — *Med-High (large sample)*
- MIS Quarterly 2023 — "Are IT Professionals Unique?" turnover meta-analysis — https://aisel.aisnet.org/misq/vol47/iss3/12/ — *High (intention drivers, not base rate)*
- NIRS — Debunking the Job-Hopping Myth (BLS CPS tenure) — https://www.nirsonline.org/research/debunkingjobhopping/ — *High*

---

## B. Learning curve & junior gap → `junior_mean_skill`, `skill_cap`, `learning_rate`

**Shape (high confidence).** Developer productivity follows **diminishing returns to an early,
hard plateau**; raw tenure beyond onboarding barely predicts productivity. Three independent
sources across 54 years and three methods agree: Sackman et al. 1968 (no experience↔performance
link), Dieste/Tubío et al. 2017 (industry experience ≈ no effect; only academic experience and
task knowledge help), Peitek et al. 2022 (EEG+eye-tracking: experience level "limited to no
predictive power"). This validates the bounded update form and a `skill_cap < 1.0` reached early.

**Junior fraction.** Expert↔novice comprehension runs ~2:1, implying a floor near ⅓–½; the
"experience explains little variance" findings argue against a very low floor. → `junior_mean_skill`
≈ **0.40 of `skill_cap`** (0.35 = pessimistic edge, 0.50 = optimistic). The classic ~10–20×
best:worst spread (Sackman) is *talent dispersion within all levels*, NOT junior-vs-senior — do
not use it for the junior gap. Education heterogeneity (Dieste) is captured by the `±spread` draw.

**Ramp & rate.** Full productivity in ~6 months modal, **3–12 month** range (use as robustness
band); baseline competence ~3 months (DX time-to-10th-PR ≈ 91 days). `learning_rate` is
**per-event** and is calibrated by simulation so a junior closes ~90% of the gap to `skill_cap`
in ~6 months ≈ 12 sprints. **Calibration result (2026-05-21, `calibrate_learning.py`):
`learning_rate = 0.0123`**, bisected to the target and confirmed by the analytic cross-check
`lr ≈ ln(10)/W = 0.01234` with measured weight budget `W ≈ 187`/dev over 12 sprints (~15.6/sprint;
this was the previously-missing events-per-sprint count). The fitted rate gives the documented
ramp: 68% of gap closed by 3 mo, 90% by 6 mo, 99% by 12 mo. **Modelling caveat for the thesis:**
because both office and remote teams plateau within a year, the office/remote *end-skill* gap is
tiny (~0.0015) — the remote→learning coupling shows up in **ramp speed and cumulative velocity**
(office +2.8% over 24 sprints via the learning channel alone), not the asymptote; report those.

**Verdict:** `junior_mean_skill` = **0.40·cap**; `skill_cap` = **0.90** (lean low); `learning_rate`
calibrated to ~6mo-to-plateau, biased to the faster half of its band; shape = **early hard plateau**.

**Sources:**
- Sackman, Erikson & Grant 1968 (via McConnell, *Making Software* ch.30) — https://www.oreilly.com/library/view/making-software/9780596808310/ch30s01.html — *High (foundational)*
- Construx (McConnell) — Origins of 10x — https://www.construx.com/blog/the-origins-of-10x-how-valid-is-the-underlying-research/ — *Med*
- Dieste/Tubío et al. 2017, *Empirical Software Engineering* 22(5) — https://link.springer.com/article/10.1007/s10664-016-9471-3 — *High (peer-reviewed meta-study)*
- Peitek et al. 2022, ESEC/FSE (EEG+eye-tracking) — https://arxiv.org/abs/2303.07071 — *High (peer-reviewed)*
- Rastogi et al. (Microsoft Research), Ramp-Up Journey, ESEM 2015 / ISEC 2017 — https://thomas-zimmermann.com/publications/files/rastogi-esem-2015.pdf — *High*
- DX — time-to-10th-PR (~91 days, 6 enterprises) — https://getdx.com/blog/ai-cuts-developer-onboarding-time-in-half/ — *Med (behavioral, vendor)*
- GitLab 2024 Global DevSecOps Survey (70% onboard >1 month) — https://about.gitlab.com/the-source/platform/3-surprising-findings-from-our-2024-global-devsecops-survey/ — *Med*
- Wright's-law / experience curves — https://en.wikipedia.org/wiki/Experience_curve_effect — *Med (foundational concept)*
- Expert/novice chunking (IJHCI) — https://www.tandfonline.com/doi/abs/10.1080/10447319409526085 — *Med (lab, small N)*

---

## C. Remote vs co-located knowledge transfer → `sync_help_weight : async_help_weight`

**Finding.** Co-located developers receive **~18–22% more code feedback** (NBER "Power of
Proximity": regression coefficient 18.3%, raw ~22% — *correction:* the "23%" in some secondary
coverage conflates this with the separate ~21% senior mentoring-output cost). About **74%** of
that advantage vanishes when teams go remote, and the benefit **concentrates on juniors / new
hires / women** ("those with the most to learn"). Peer-reviewed call-center evidence shows a
smaller ~8–12% remote knowledge/performance gap for lower-skill work.

The penalty attaches to the **knowledge-transfer channel specifically**, not throughput —
output parity between remote and office is common (Bloom; Gibbs).

**Verdict:** `sync_help_weight : async_help_weight` ≈ **1.2× general, 1.5–2× for juniors**.
Because help-when-blocked is a mentorship-heavy moment, the model uses **sync 1.5 : async 1.0
(=1.5×)**. This is the **weakest-evidenced** parameter (no clean junior multiplier) → widest sweep.

**Sources:**
- Emanuel, Harrington & Pallais — "The Power of Proximity to Coworkers" (NBER WP 31880, 2023) — https://www.nber.org/papers/w31880 ; NY Fed summary https://libertystreeteconomics.newyorkfed.org/2024/01/the-power-of-proximity-how-working-beside-colleagues-affects-training-and-productivity/ — *High (natural experiment, code-review logs)*
- Emanuel & Harrington — "Working Remotely?" *AEJ: Applied* 16(4), 2024 (~8–12% gap) — https://www.aeaweb.org/articles?id=10.1257/app.20230376 — *High (quasi-experimental, peer-reviewed)*
- Yang, Holtz et al. — remote collaboration, *Nature Human Behaviour* 2021 (61k workers; ~25% less cross-group collaboration) — https://www.nature.com/articles/s41562-021-01196-4 — *High*
- Gibbs, Mengel & Siemroth — WFH & productivity, IT professionals, *JPE Micro* 2023 — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3841457 — *Med-High (observational)*

---

## D. Remote work & attrition → `remote_attrition_coef` (contested — two scenarios)

**Finding.** The sign is **genuinely split** and depends on dose (hybrid vs full remote),
seniority, and voluntariness — these are not contradictory but capture different channels.
This contestation is modeled as the **headline DSS sensitivity knob**, base **0.0**, sweep
**[−1, +1]**, reported under two named scenarios:

- **Retention scenario (coef < 0):** voluntary hybrid *reduces* quitting by **~33%** (2.4pp,
  7.2%→4.8%; up to ~40% non-managers) — Bloom, Han & Liang **RCT in Nature 2024** (1,612 incl.
  computer engineers, 2-day hybrid, null performance effect). Highest-quality evidence; flexibility/
  commute/autonomy channel.
- **Isolation scenario (coef > 0):** losing established in-person mentorship *raises* junior
  quitting by **+1.2pp** (≈ "~5× relative odds" for previously-co-located juniors — *correction:*
  the 5× is the same effect as +1.2pp expressed as relative odds, not an independent magnitude) —
  Power of Proximity; corroborated by remote-onboarding resignation studies. Thinnest leg →
  widest band. Lost-mentorship/weak-ties channel.

The model uses a single undifferentiated coefficient and **cannot capture the
seniority-conditioning** the literature shows — note as a limitation / future extension.

**Sources:**
- Bloom, Han & Liang — "Hybrid working from home improves retention…", *Nature* 630 (2024) — https://www.nature.com/articles/s41586-024-07500-2 — *High (RCT)*
- Emanuel, Harrington & Pallais — Power of Proximity (junior +1.2pp quits) — https://www.nber.org/papers/w31880 — *High*
- "A Wave of Resignations after Remote Onboarding" (Ericsson), arXiv 2510.05878 (2025) — https://arxiv.org/abs/2510.05878 — *Low-Med (observational)*
- Ding & Ma — RTO mandates & tenure distribution (2024) — https://arxiv.org/pdf/2405.04352 — *Med*
- U. Pittsburgh / Fortune — RTO → +14% departures (2024) — https://fortune.com/2024/12/11/return-to-office-mandate-employees-study/ — *Med*
- Gartner / OWL Labs / Gallup — isolation & turnover intent — *Low (survey-intent)*

---

## Summary of recommended values

| Parameter | Base case | Sweep range | Evidence strength |
|---|---|---|---|
| `annual_attrition` | 0.12 | 0.08–0.20 | Med (synthesis; no clean source) |
| `skill_cap` | 0.90 | 0.85–0.95 | High (3 sources, early plateau) |
| `junior_mean_skill` (×cap) | 0.40 | 0.35–0.50 | Med (reconstructed from proxies) |
| `learning_rate` (per event) | **0.0123** (calibrated to ~6mo→plateau) | refit if block_prob/weights change | Med (rate fitted; target High) |
| `sync_help_weight` | 1.5 | — | High (general), Low (junior multiplier) |
| `async_help_weight` | 1.0 | sync:async 1.2×–2× | as above |
| `remote_attrition_coef` | 0.0 | −1 … +1 (2 scenarios) | High both arms, sign contested |

All `★` values are sensitivity-analysis base cases. The thesis should report results across the
sweep ranges and explicitly flag the construct-mismatch limitation and the contested sign of
`remote_attrition_coef`.
