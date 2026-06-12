"""Multi-sprint horizon: per-sprint velocity, backlog top-up, accumulation.

Also holds the FULL-MODEL golden (learning + turnover on, 24 sprints) as a
regression snapshot, complementing the pure-mechanics golden in
test_determinism.py.
"""

from __future__ import annotations

import pytest

from model import TeamModel


def test_sprint_velocity_recorded_once_per_sprint():
    """One velocity entry per sprint; the run length is sprints x sprint_length."""
    m = TeamModel(seed=42, n_sprints=3, sprint_length=32, block_prob=0.0,
                  learning_rate=0.0, annual_attrition=0.0, backlog_size=500)
    while m.running:
        m.step()

    assert m.tick == 3 * 32
    assert len(m.sprint_velocities) == 3
    # The final sprint is recorded but not reset, so the live counter equals it.
    assert m.velocity == m.sprint_velocities[-1]
    # Conservation: total recorded velocity == story points of completed tasks.
    completed_sp = sum(t.story_points for t in m.completed_tasks)
    assert sum(m.sprint_velocities) == completed_sp


def test_team_does_not_starve_over_many_sprints():
    """Backlog top-up keeps every sprint productive (no zero-velocity sprint)."""
    m = TeamModel(seed=1, n_sprints=10, sprint_length=64, block_prob=0.0,
                  learning_rate=0.0, annual_attrition=0.0,
                  backlog_size=50, backlog_target=200, n_devs=5)
    while m.running:
        m.step()
    assert len(m.sprint_velocities) == 10
    assert all(v > 0 for v in m.sprint_velocities)


def test_skill_and_tenure_accumulate_across_sprints():
    """With learning on and no churn, mean skill rises and tenure accrues."""
    m = TeamModel(seed=3, n_sprints=8, sprint_length=64, block_prob=0.05,
                  learning_rate=0.02, annual_attrition=0.0,
                  mean_solo_skill=0.3, solo_skill_spread=0.0, skill_cap=0.9)
    skill_start = sum(d.solo_skill for d in m.devs) / len(m.devs)
    while m.running:
        m.step()
    skill_end = sum(d.solo_skill for d in m.devs) / len(m.devs)
    assert skill_end > skill_start
    assert skill_end <= 0.9 + 1e-9
    # No churn ⇒ founding team intact ⇒ tenure equals the full horizon.
    assert all(d.hired_at_tick == 0 for d in m.devs)


def test_full_model_golden():
    """Regression snapshot of the full model (learning + turnover, 24 sprints).

    Captured with seed=42 and all library defaults. Distinct from the
    pure-mechanics golden (test_determinism.py): this pins the NEW behavior.
    Update only on an intentional, reviewed model change.

    Re-baselined 2026-05-21 when ``learning_rate`` was calibrated 0.015 -> 0.0123
    (see calibrate_learning.py). The slower rate keeps solo skill marginally
    lower, so a few more blockers need help (wait_times 642 -> 683) and a few
    fewer tasks complete (3042 -> 3008) — all coherent with the parameter change.
    The pure-mechanics golden (learning gated off) is unchanged and remains the
    true regression guard for the pre-extension code paths.
    """
    m = TeamModel(seed=42)  # all defaults: 24 sprints, learning + turnover on
    while m.running:
        m.step()

    assert m.tick == 7680
    assert len(m.sprint_velocities) == 24
    assert sum(m.sprint_velocities) == 8441
    assert len(m.completed_tasks) == 3008
    assert len(m.wait_times) == 683
    assert m.attrition_count == 0  # this seed happens to retain everyone
    mean_skill = sum(d.solo_skill for d in m.devs) / len(m.devs)
    assert mean_skill == pytest.approx(0.895828, abs=1e-6)


def test_close_sprint_records_then_resets_velocity():
    """_close_sprint records the live velocity, then zeroes the counter.

    Direct unit check of the reset semantics the per-sprint KPI depends on —
    stronger than inferring it from "every sprint had positive velocity".
    """
    m = TeamModel(seed=9, n_sprints=2, sprint_length=32, block_prob=0.0,
                  learning_rate=0.0, annual_attrition=0.0)
    m.velocity = 99
    m._close_sprint()
    assert m.sprint_velocities[-1] == 99   # the finished sprint was recorded
    assert m.velocity == 0                 # and the counter reset for the next


def test_velocity_counter_resets_between_sprints_in_a_real_run():
    """Across a real run the counter never carries a previous sprint's points.

    Right after each boundary close, the live velocity reflects only the work
    done since the boundary, so it can never exceed the just-recorded sprint's
    total plus one tick's worth of new completions.
    """
    m = TeamModel(seed=11, n_sprints=4, sprint_length=32, block_prob=0.0,
                  learning_rate=0.0, annual_attrition=0.0, backlog_size=500)
    while m.running:
        at_boundary = m.tick > 0 and m.tick % 32 == 0
        recorded_before = len(m.sprint_velocities)
        m.step()
        if at_boundary and len(m.sprint_velocities) > recorded_before:
            # A sprint was just closed this step; the live counter restarted
            # from 0 and only this tick's completions could have landed.
            just_recorded = m.sprint_velocities[-1]
            assert m.velocity <= just_recorded  # did not carry the old total


def test_backlog_top_up_fires_at_sprint_boundary():
    """The backlog is replenished to backlog_target when a sprint closes.

    Start with a backlog far below the target and a high target; after the first
    sprint boundary the floor must have been restored (top-up fired), proving the
    long run will not starve.
    """
    target = 300
    m = TeamModel(seed=2, n_sprints=2, sprint_length=32, block_prob=0.0,
                  learning_rate=0.0, annual_attrition=0.0,
                  backlog_size=20, backlog_target=target, n_devs=5)
    assert len(m.backlog) <= 20            # starts small
    # Run just past the first boundary (tick==32) where _close_sprint tops up.
    while m.tick < 33 and m.running:
        m.step()
    # After the boundary the backlog floor was restored, minus whatever the team
    # pulled in the single tick after top-up (<= n_devs tickets).
    assert len(m.backlog) >= target - m.n_devs


def test_full_model_is_deterministic_under_churn():
    """Two full runs with the same seed are identical, turnover included."""
    def run():
        m = TeamModel(seed=5, n_sprints=6, sprint_length=32, n_devs=6,
                      block_prob=0.1, learning_rate=0.02, annual_attrition=0.5)
        while m.running:
            m.step()
        mean_skill = sum(d.solo_skill for d in m.devs) / len(m.devs)
        return list(m.sprint_velocities), m.attrition_count, round(mean_skill, 9)

    assert run() == run()
