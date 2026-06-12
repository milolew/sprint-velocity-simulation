"""Edge cases: tiny backlog, single dev, zero-length sprint, skill clamping."""

from __future__ import annotations


def test_backlog_exhaustion_devs_go_idle_no_crash(model_factory):
    """Tiny backlog: once exhausted, devs sit idle. No crash."""
    m = model_factory(
        block_prob=0.0,
        backlog_size=2,
        sp_scaling_k=1,    # 1 tick per SP, so tasks finish quickly
        sprint_length=64,
    )
    while m.running:
        m.step()
    # Exactly 2 tasks completed; rest of the time spent idle
    assert len(m.completed_tasks) == 2
    # All devs end up idle (nothing to pick up)
    for d in m.devs:
        assert d.state == "idle"
        assert d.current_task is None
    assert m.backlog == []


def test_single_dev_blocker_never_resolved_via_help(model_factory):
    """n_devs=1: there are no candidates for help, so a blocker can only
    be resolved via solo. With solo_skill=0.0, the dev stays blocked."""
    m = model_factory(
        n_devs=1,
        block_prob=1.0,
        mean_solo_skill=0.0,
        solo_skill_spread=0.0,
        sp_scaling_k=20,
        sprint_length=20,
    )
    while m.running:
        m.step()
    dev = m.devs[0]
    # Dev got blocked early and stayed blocked (no helpers ever available)
    assert dev.state == "blocked"
    assert dev.blocker_stage == "ask_next_tick"
    # No blockers resolved
    assert m.wait_times == []


def test_zero_length_sprint_no_work(model_factory):
    """sprint_length=0: no ticks elapse; nothing happens."""
    m = model_factory(sprint_length=0)
    # The model should immediately stop on step()
    m.step()
    assert m.tick == 0
    assert m.velocity == 0
    assert m.completed_tasks == []
    assert m.running is False


def test_skill_clamped_to_unit_interval(model_factory):
    """A negative sampled skill (mean=0.0, spread=0.5) is clamped to [0,1]."""
    m = model_factory(
        mean_solo_skill=0.0,
        solo_skill_spread=0.5,    # samples in [-0.5, 0.5]; some are negative
        n_devs=20,                # enough samples to almost certainly hit < 0
        seed=42,
    )
    for d in m.devs:
        assert 0.0 <= d.solo_skill <= 1.0


def test_skill_clamped_above_one(model_factory):
    """mean_solo_skill=1.0, spread=0.5 → some samples > 1 should clamp to 1."""
    m = model_factory(
        mean_solo_skill=1.0,
        solo_skill_spread=0.5,
        n_devs=20,
        seed=42,
    )
    for d in m.devs:
        assert 0.0 <= d.solo_skill <= 1.0


def test_running_flag_false_when_sprint_complete(model_factory):
    """After ticking past sprint_length, m.running becomes False."""
    m = model_factory(sprint_length=3, block_prob=0.0)
    assert m.running is True
    while m.running:
        m.step()
    assert m.running is False
    assert m.tick == 3


def test_step_after_sprint_end_is_safe_noop(model_factory):
    """Calling step() after sprint completion is a no-op (no crash)."""
    m = model_factory(sprint_length=2, block_prob=0.0)
    while m.running:
        m.step()
    tick_at_end = m.tick
    velocity_at_end = m.velocity

    # Extra steps must not crash and must not advance state
    m.step()
    m.step()
    assert m.tick == tick_at_end
    assert m.velocity == velocity_at_end
    assert m.running is False
