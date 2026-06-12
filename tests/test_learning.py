"""Event-driven learning: solo_skill grows from experience.

Skill updates are deterministic (no RNG): each qualifying event moves skill a
bounded step toward skill_cap, weighted by the event channel. These tests pin
the exact arithmetic and the gating guarantee (learning_rate=0 ⇒ no movement).
"""

from __future__ import annotations

import pytest

from model import Task


def test_learning_rate_zero_is_noop(model_factory):
    """With learning_rate=0 (the suite default), skills never move."""
    m = model_factory(block_prob=0.2, sprint_length=64, seed=1)
    before = [d.solo_skill for d in m.devs]
    while m.running:
        m.step()
    after = [d.solo_skill for d in m.devs]
    assert before == after


def test_task_completion_applies_completion_weight(model_factory):
    """Completing a task credits task_completion_weight of learning."""
    m = model_factory(
        block_prob=0.0, learning_rate=0.1, skill_cap=0.9,
        task_completion_weight=0.5, mean_solo_skill=0.5, solo_skill_spread=0.0,
    )
    dev = m.devs[0]
    skill0 = dev.solo_skill
    dev.current_task = Task(story_points=2, work_remaining=1, state="in_progress")
    dev.state = "working"

    dev._step_working()  # work_remaining -> 0 -> completed -> learning

    expected = skill0 + 0.1 * 0.5 * (0.9 - skill0)
    assert dev.solo_skill == pytest.approx(expected)
    assert dev.state == "idle"


@pytest.mark.parametrize("resolution,weight", [
    ("solo", 1.0),
    ("sync", 1.5),
    ("async", 0.7),
])
def test_resolution_channel_applies_its_weight(model_factory, resolution, weight):
    """Each blocker-resolution channel credits its own learning weight."""
    m = model_factory(
        learning_rate=0.1, skill_cap=0.9,
        solo_resolution_weight=1.0, sync_help_weight=1.5, async_help_weight=0.7,
        mean_solo_skill=0.5, solo_skill_spread=0.0,
    )
    dev = m.devs[0]
    skill0 = dev.solo_skill
    dev.current_task = Task(story_points=1, work_remaining=4, state="blocked")
    dev.block_start_tick = 0

    dev._unblock(resolution)

    expected = skill0 + 0.1 * weight * (0.9 - skill0)
    assert dev.solo_skill == pytest.approx(expected)
    assert dev.help_channel is None  # cleared on unblock


def test_unknown_resolution_channel_raises(model_factory):
    """A bad channel string fails loudly rather than silently mis-learning."""
    dev = model_factory().devs[0]
    with pytest.raises(ValueError):
        dev._weight_for("telepathy")


def test_learning_approaches_cap_monotonically(model_factory):
    """Repeated events drive skill up toward the cap, never past it."""
    m = model_factory(
        learning_rate=0.3, skill_cap=0.8, solo_resolution_weight=1.0,
        mean_solo_skill=0.2, solo_skill_spread=0.0,
    )
    dev = m.devs[0]
    prev = dev.solo_skill
    for _ in range(100):
        dev.current_task = Task(story_points=1, work_remaining=4, state="blocked")
        dev.block_start_tick = 0
        dev._unblock("solo")
        assert dev.solo_skill >= prev          # monotonic non-decreasing
        assert dev.solo_skill <= 0.8 + 1e-12   # bounded by the cap
        prev = dev.solo_skill
    assert dev.solo_skill == pytest.approx(0.8, abs=1e-3)  # converged near cap


def test_learning_never_decreases_skill_above_cap(model_factory):
    """A dev initialised above the cap is not 'unlearned' down to it."""
    m = model_factory(
        learning_rate=0.5, skill_cap=0.6, solo_resolution_weight=1.0,
        mean_solo_skill=0.9, solo_skill_spread=0.0,
    )
    dev = m.devs[0]
    skill0 = dev.solo_skill  # 0.9, above cap 0.6
    dev.current_task = Task(story_points=1, work_remaining=4, state="blocked")
    dev.block_start_tick = 0
    dev._unblock("solo")
    assert dev.solo_skill == pytest.approx(skill0)  # unchanged (gap clamped at 0)


def test_help_channel_set_iff_helper_set(model_factory):
    """Invariant across a real run: help_channel is set exactly when helper is."""
    m = model_factory(
        block_prob=0.3, sprint_length=64, seed=3, remote_share=0.5,
        sync_help_mean=2, async_help_mean=5,
    )
    while m.running:
        m.step()
        for d in m.devs:
            assert (d.help_channel is None) == (d.helper is None)


def _mean_end_skill(model_factory, remote_share, seed):
    """Run a multi-sprint study and return end-of-run mean solo_skill."""
    m = model_factory(
        seed=seed, remote_share=remote_share,
        n_sprints=6, sprint_length=64, n_devs=5, block_prob=0.25,
        learning_rate=0.01, annual_attrition=0.0,
        mean_solo_skill=0.3, solo_skill_spread=0.0, skill_cap=0.95,
        # Wide sync/async gap and a cap the team will not saturate, so the
        # channel mix actually shows up in end-of-run skill (no plateau mask).
        sync_help_weight=2.0, async_help_weight=0.5,
    )
    while m.running:
        m.step()
    return sum(d.solo_skill for d in m.devs) / len(m.devs)


def test_remote_share_slows_mean_skill_growth(model_factory):
    """Headline coupling: more remote work ⇒ slower capability growth.

    All-remote forces every blocker resolution onto the async channel (smaller
    learning weight); all-office forces the sync channel (larger weight). With a
    cap the team does not saturate, end-of-run mean skill must be strictly lower
    under full remote, averaged over seeds. This validates the core thesis
    mechanism — the learning↔remote coupling — that has no dedicated knob and
    rides entirely on the channel mix.
    """
    seeds = range(25)
    office = sum(_mean_end_skill(model_factory, 0.0, s) for s in seeds) / len(seeds)
    remote = sum(_mean_end_skill(model_factory, 1.0, s) for s in seeds) / len(seeds)

    assert office > remote
    # Effect should be substantial, not a rounding artifact, given the 2.0 vs 0.5
    # weight gap; guards against the coupling silently degrading to ~0.
    assert office - remote > 0.05


def test_async_resolution_alone_learns_less_than_sync(model_factory):
    """Direct unit check of the coupling's root cause: sync weight > async weight.

    The run-level coupling reduces to: a single sync resolution must credit more
    skill than a single async resolution, all else equal.
    """
    from model import Task

    def gain(channel):
        m = model_factory(
            learning_rate=0.1, skill_cap=0.9,
            sync_help_weight=1.5, async_help_weight=1.0,
            mean_solo_skill=0.5, solo_skill_spread=0.0,
        )
        dev = m.devs[0]
        before = dev.solo_skill
        dev.current_task = Task(story_points=1, work_remaining=4, state="blocked")
        dev.block_start_tick = 0
        dev._unblock(channel)
        return dev.solo_skill - before

    assert gain("sync") > gain("async") > 0
