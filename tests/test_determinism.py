"""Pinned-seed regression tests: same seed = same outcome.

These tests catch unintended behavior drift when the model is refactored.
The "golden" values were captured from a known-good run with seed=42 and
default parameters. If a refactor changes them, that's either a bug or an
intentional behavior change that should be reviewed.
"""

from __future__ import annotations

from model import TeamModel


def test_same_seed_produces_identical_results():
    """Two runs with identical inputs must produce identical outputs."""
    def run():
        m = TeamModel(seed=123, sprint_length=200, block_prob=0.05, n_sprints=1)
        while m.running:
            m.step()
        return m.velocity, list(m.wait_times), len(m.completed_tasks)

    v1, w1, c1 = run()
    v2, w2, c2 = run()
    assert v1 == v2
    assert w1 == w2
    assert c1 == c2


def test_different_seeds_produce_different_results():
    """Sanity check: different seeds shouldn't accidentally collide.

    With realistic parameters and 320 ticks, the probability of two seeds
    producing identical results is astronomically small. If this test fails,
    the seed is not actually being used.
    """
    def run(seed):
        m = TeamModel(seed=seed, sprint_length=200, block_prob=0.05, n_sprints=1)
        while m.running:
            m.step()
        return m.velocity, tuple(m.wait_times)

    a = run(1)
    b = run(2)
    assert a != b


def test_golden_values_pure_mechanics():
    """Regression guard for the ORIGINAL single-sprint mechanics.

    Learning and turnover are gated OFF (learning_rate=0, annual_attrition=0)
    and the horizon is a single sprint (n_sprints=1), so this run exercises
    exactly the pre-extension code paths. These values were captured from the
    pre-extension model with seed=42 and must stay byte-for-byte identical —
    they prove the multi-sprint/learning/turnover work did not perturb the old
    paths. The full-model golden lives in test_multisprint.py.
    """
    m = TeamModel(seed=42, n_sprints=1, learning_rate=0.0, annual_attrition=0.0)
    while m.running:
        m.step()

    # Captured from a known-good pre-extension run.
    assert m.tick == 320
    assert m.velocity == 338
    assert len(m.completed_tasks) == 134
    assert len(m.wait_times) == 22


def test_seed_independence_across_replications():
    """Replications with the same seed but different builds are identical.

    Verifies that there is no global mutable state leaking between TeamModel
    instances — each model owns its own RNG.
    """
    m1 = TeamModel(seed=7, sprint_length=64, n_sprints=1)
    m2 = TeamModel(seed=7, sprint_length=64, n_sprints=1)
    # Building a third model in the middle should not affect m2.
    _ = TeamModel(seed=999, sprint_length=64, n_sprints=1)

    while m1.running:
        m1.step()
    while m2.running:
        m2.step()

    assert m1.velocity == m2.velocity
    assert m1.wait_times == m2.wait_times


def test_gating_makes_learning_params_byte_irrelevant():
    """The gating contract: with learning_rate=0 the learning knobs are dead code.

    This is stronger than "skills don't move": it proves that *changing the
    learning weights and the remote-attrition coefficient has zero effect on the
    trajectory* when the gates are off. That is exactly what lets the
    pure-mechanics golden stand in for the pre-extension model — the new code
    paths must not perturb wait_times, velocity, completed tasks, OR skills.
    """
    def run(**learning_knobs):
        m = TeamModel(
            seed=42, sprint_length=200, block_prob=0.05, n_sprints=1,
            learning_rate=0.0, annual_attrition=0.0, **learning_knobs,
        )
        while m.running:
            m.step()
        return (
            m.velocity,
            tuple(m.wait_times),
            len(m.completed_tasks),
            tuple(round(d.solo_skill, 12) for d in m.devs),
        )

    baseline = run()
    # Wildly different learning weights / cap / attrition coef must not matter.
    perturbed = run(
        task_completion_weight=99.0, solo_resolution_weight=99.0,
        sync_help_weight=99.0, async_help_weight=99.0,
        skill_cap=0.3, remote_attrition_coef=0.9,
    )
    assert baseline == perturbed


def test_turnover_decides_all_then_applies_one_draw_per_dev():
    """Part 7 contract: one draw per dev over the ORIGINAL fixed-order roster.

    Turnover must collect every leave decision first (exactly one random() per
    dev, in a fixed order over the un-mutated roster) and only then retire the
    settled leavers. If a retirement could be interleaved with the decisions —
    shrinking the list mid-loop — the set of leavers would not be reconstructable
    from one draw per original dev. We reconstruct it from a cloned RNG and assert
    an exact match, which would break under any interleaving.
    """
    import random as stdlib_random

    m = TeamModel(
        seed=42, n_devs=5, sprint_length=32, n_sprints=2, annual_attrition=0.5,
        block_prob=0.0, learning_rate=0.0, mean_solo_skill=0.9, solo_skill_spread=0.0,
    )
    roster = list(m.devs)                       # fixed-order copy before turnover
    p_leave = m._p_leave()                       # _p_leave makes no RNG draws

    # Predict the leavers from a clone of the model RNG: one draw per dev, in
    # roster order, BEFORE any retirement could perturb the stream.
    clone = stdlib_random.Random()
    clone.setstate(m.random.getstate())
    predicted = {d for d in roster if clone.random() < p_leave}

    m._run_turnover_checks()
    actually_left = {d for d in roster if d not in m.devs}

    # All predicted leavers are settled here (no help sessions), so prediction
    # and reality must coincide exactly.
    assert actually_left == predicted
    assert 0 < len(predicted) < len(roster)      # meaningful: some left, some stayed
