"""Turnover: devs leave at sprint boundaries and are replaced by juniors.

Covers the gating guarantee (rate 0 ⇒ nobody leaves), forced replacement, the
deferral rule for devs mid help session, task return-to-backlog, and a seeded
fuzz test of the state invariants that turnover must never violate.
"""

from __future__ import annotations

import pytest

from model import Task


def test_no_attrition_when_rate_zero(model_factory):
    """annual_attrition=0 ⇒ no retirements across a full multi-sprint run."""
    m = model_factory(annual_attrition=0.0, n_sprints=24, sprint_length=32, n_devs=5)
    while m.running:
        m.step()
    assert m.attrition_count == 0
    assert len(m.devs) == 5


def test_full_attrition_replaces_all_settled_devs(model_factory):
    """With p_leave=1 and no blockers, every dev is replaced at the boundary."""
    m = model_factory(
        annual_attrition=1.0, n_sprints=2, sprint_length=32, n_devs=5,
        block_prob=0.0, mean_solo_skill=0.8, solo_skill_spread=0.0,
        skill_cap=0.9, junior_mean_skill=0.4, junior_skill_spread=0.0,
    )
    while m.tick < 33 and m.running:  # run through the tick==32 boundary
        m.step()

    assert m.attrition_count == 5
    assert len(m.devs) == 5
    for d in m.devs:
        assert d.solo_skill == pytest.approx(0.4 * 0.9)  # junior = mean * cap
        assert d.hired_at_tick == 32


def test_helping_and_waiting_devs_are_deferred(model_factory):
    """Devs mid help session are skipped this boundary; settled ones leave."""
    m = model_factory(annual_attrition=1.0, n_devs=4,
                      mean_solo_skill=0.8, solo_skill_spread=0.0)
    helper, waiter, s1, s2 = m.devs
    helper.state = "helping"
    waiter.state = "blocked"
    waiter.blocker_stage = "waiting_for_help"

    m._run_turnover_checks()  # p_leave == 1, so all four "decide" to leave

    assert helper in m.devs   # deferred — survives
    assert waiter in m.devs
    assert s1 not in m.devs   # settled — replaced
    assert s2 not in m.devs
    assert m.attrition_count == 2
    assert len(m.devs) == 4


def test_leaver_task_returns_to_backlog(model_factory):
    """A working leaver's task goes back to the backlog with progress reset."""
    m = model_factory(annual_attrition=1.0, n_devs=1,
                      mean_solo_skill=0.8, solo_skill_spread=0.0, sp_scaling_k=4)
    dev = m.devs[0]
    task = Task(story_points=3, work_remaining=1, state="in_progress")  # partly done
    dev.current_task = task
    dev.state = "working"
    backlog_before = len(m.backlog)

    m._run_turnover_checks()

    assert dev not in m.devs
    assert task in m.backlog
    assert task.state == "awaiting"
    assert task.work_remaining == 3 * 4          # reset to full
    assert len(m.backlog) == backlog_before + 1


def test_replacement_keeps_devs_and_agents_in_sync(model_factory):
    """Retiring + hiring must update both self.devs and Mesa's AgentSet."""
    m = model_factory(annual_attrition=1.0, n_devs=3,
                      mean_solo_skill=0.8, solo_skill_spread=0.0)
    m._run_turnover_checks()
    assert len(m.devs) == 3
    assert set(m.devs) == set(m.agents)


def test_deferred_dev_leaves_once_settled_at_a_later_boundary(model_factory):
    """A dev deferred while helping survives that boundary, then leaves the next.

    Decision 5 defers (re-rolls) a helping/waiting leaver. This pins the full
    arc: deferred at boundary 1 (still helping), then actually retired at
    boundary 2 once it has returned to a settled state.
    """
    m = model_factory(annual_attrition=1.0, n_devs=3,
                      mean_solo_skill=0.8, solo_skill_spread=0.0)
    deferred = m.devs[0]
    deferred.state = "helping"           # locked → must be deferred
    others = m.devs[1:]

    m._run_turnover_checks()             # boundary 1: p_leave==1 for all
    assert deferred in m.devs            # deferred — survived
    assert all(o not in m.devs for o in others)  # settled colleagues replaced
    assert m.attrition_count == 2        # only the two settled colleagues left

    deferred.state = "idle"              # now settled
    attrition_before_second = m.attrition_count
    m._run_turnover_checks()             # boundary 2
    assert deferred not in m.devs        # the once-deferred dev finally leaves
    # It now counts toward attrition (along with the fresh hires that also rolled
    # to leave at p_leave==1) — the point is it is no longer skipped.
    assert m.attrition_count > attrition_before_second
    assert len(m.devs) == 3              # team size invariant holds throughout


def test_turnover_with_empty_backlog_returns_task_and_keeps_team_size(model_factory):
    """Turnover does not depend on a non-empty backlog.

    A working leaver with an empty backlog still returns its task (the backlog
    grows from 0 to 1) and is replaced, so the team size invariant holds even
    when the team has drained its backlog.
    """
    m = model_factory(annual_attrition=1.0, n_devs=2, sp_scaling_k=4,
                      mean_solo_skill=0.8, solo_skill_spread=0.0, backlog_size=0)
    m.backlog.clear()
    worker = m.devs[0]
    worker.current_task = Task(story_points=2, work_remaining=3, state="in_progress")
    worker.state = "working"

    m._run_turnover_checks()

    assert worker not in m.devs
    assert len(m.devs) == 2                       # replaced 1:1
    assert len(m.backlog) >= 1                     # returned task is now in backlog
    returned = next(t for t in m.backlog if t.story_points == 2)
    assert returned.state == "awaiting"
    assert returned.work_remaining == 2 * 4        # FULL reset, partial progress lost


def test_new_hire_picks_up_work_and_learns(model_factory):
    """A junior hired at a boundary then pulls a ticket and gains skill by working.

    This exercises the replacement as a *live agent*, not just bookkeeping: the
    hire must be schedulable (idle→working via the backlog), credited velocity,
    and credited learning on completion.
    """
    # One churn event at the first boundary only (n_sprints=2 so the hire is not
    # itself retired before we observe it working). High learning_rate makes the
    # gain visible in a few completions.
    m = model_factory(
        annual_attrition=1.0, n_devs=1, n_sprints=2, sprint_length=64,
        block_prob=0.0, learning_rate=0.1, task_completion_weight=0.5,
        mean_solo_skill=0.9, solo_skill_spread=0.0,
        junior_mean_skill=0.4, junior_skill_spread=0.0, skill_cap=0.9,
        backlog_size=50, backlog_target=50,
    )
    # Run through the first sprint boundary (tick==64) so the founder is replaced.
    while m.tick < 65 and m.running:
        m.step()
    hire = m.devs[0]                                 # the specific junior we track
    assert hire.hired_at_tick == 64
    skill_at_hire = hire.solo_skill
    assert skill_at_hire == pytest.approx(0.4 * 0.9)  # junior = mean * cap
    completed_before = len(m.completed_tasks)

    # Let the hire work within its own sprint (run stops at tick 128, before any
    # further boundary could retire it). It should pull from the backlog,
    # complete tasks, and learn from them.
    while m.running:
        m.step()

    assert m.devs[0] is hire                          # same dev, never re-churned
    assert len(m.completed_tasks) > completed_before  # hire produced work
    assert hire.solo_skill > skill_at_hire            # and learned from it
    assert hire.solo_skill <= 0.9 + 1e-9              # still bounded by the cap


def _assert_invariants(m):
    assert len(m.devs) == m.n_devs                 # team size constant
    assert set(m.devs) == set(m.agents)            # both registries agree
    owned = [d.current_task for d in m.devs if d.current_task is not None]
    assert len(owned) == len({id(t) for t in owned})  # no task owned twice
    for d in m.devs:                               # no dangling helper refs
        if d.helper is not None:
            assert d.helper in m.devs


@pytest.mark.parametrize("seed", range(20))
def test_turnover_invariants_hold_every_sprint(model_factory, seed):
    """Fuzz: under heavy churn the state invariants hold after every tick."""
    m = model_factory(
        annual_attrition=0.5, n_sprints=12, sprint_length=32, n_devs=6,
        block_prob=0.1, learning_rate=0.02, remote_share=0.5, seed=seed,
    )
    while m.running:
        m.step()
        _assert_invariants(m)
    assert m.attrition_count >= 0
