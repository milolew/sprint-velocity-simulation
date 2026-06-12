"""Dev state machine transitions.

Forces specific transitions via parameter overrides and direct dev mutation
rather than running thousands of ticks and hoping. Each test verifies one
edge of the transition diagram:

    idle <-> working <-> blocked(solo) -> blocked(ask_next_tick)
                                       -> blocked(waiting_for_help) -> working
    helping (no-op while session active)
"""

from __future__ import annotations


def test_idle_dev_picks_up_task_when_backlog_nonempty(model_factory):
    m = model_factory(block_prob=0.0)
    backlog_size_before = len(m.backlog)
    m.step()

    working = [d for d in m.devs if d.state == "working"]
    # All 5 devs should immediately grab tasks (backlog has plenty)
    assert len(working) == 5
    assert len(m.backlog) == backlog_size_before - 5
    for d in working:
        assert d.current_task is not None
        assert d.current_task.state == "in_progress"


def test_idle_dev_stays_idle_when_backlog_empty(model_factory):
    m = model_factory(block_prob=0.0, backlog_size=0)
    m.step()

    for d in m.devs:
        assert d.state == "idle"
        assert d.current_task is None


def test_working_dev_completes_task_and_credits_velocity(model_factory):
    """When work_remaining hits 0, task is completed and velocity += sp."""
    m = model_factory(block_prob=0.0, sp_scaling_k=1)  # 1 tick per SP
    # Pre-pick the task we want to land on a known dev
    dev = m.devs[0]
    dev.state = "working"
    # Inject a task with known story_points and 1 tick of work left
    from model import Task
    task = Task(story_points=3, work_remaining=1, state="in_progress")
    dev.current_task = task

    velocity_before = m.velocity

    # _step_working will decrement -> 0 -> complete
    dev._step_working()

    assert dev.state == "idle"
    assert dev.current_task is None
    assert task.state == "completed"
    assert task in m.completed_tasks
    assert m.velocity == velocity_before + 3


def test_working_dev_transitions_to_blocked_solo_when_block_rolls(model_factory):
    """With block_prob=1.0, a working dev hits a blocker every step."""
    m = model_factory(block_prob=1.0, sp_scaling_k=10)  # long enough not to complete
    m.step()  # all devs go idle->working
    # Reset velocity bookkeeping isn't relevant; just step once more.
    m.step()  # working dev gets blocked at the FIRST work step (not completed)

    # All devs that were working before should now be blocked-solo
    blocked = [d for d in m.devs if d.state == "blocked"]
    assert len(blocked) == 5
    for d in blocked:
        assert d.blocker_stage == "solo"
        assert d.block_start_tick is not None
        assert d.current_task is not None
        assert d.current_task.state == "blocked"


def test_blocked_solo_resolves_to_working_on_success(model_factory):
    """solo_skill=1.0 means solo attempt always succeeds; dev returns to working."""
    m = model_factory(
        block_prob=1.0,
        mean_solo_skill=1.0,
        solo_skill_spread=0.0,
        sp_scaling_k=10,
    )
    m.step()  # idle -> working
    m.step()  # working -> blocked(solo) for everyone
    block_starts = {d: d.block_start_tick for d in m.devs}

    m.step()  # solo attempt: should succeed for all

    for d in m.devs:
        # Returned to working with bookkeeping cleared
        assert d.state == "working"
        assert d.blocker_stage is None
        assert d.block_start_tick is None
    # One wait_time appended per dev: tick - block_start_tick
    assert len(m.wait_times) == 5
    for d, wait in zip(m.devs, m.wait_times):
        # The exact wait isn't important here, but it must be non-negative
        assert wait >= 0


def test_blocked_solo_failure_advances_to_ask_next_tick_without_finding_helper(
    model_factory,
):
    """Solo failure must NOT also try to recruit a helper in the same tick.

    Regression: there was an earlier version that ran solo and ask in the
    same tick. The fix: solo failure transitions to ask_next_tick and the
    NEXT tick's _try_find_helper does the recruitment.
    """
    m = model_factory(
        block_prob=1.0,
        mean_solo_skill=0.0,
        solo_skill_spread=0.0,
        sp_scaling_k=10,
    )
    m.step()  # idle -> working
    m.step()  # working -> blocked(solo)
    m.step()  # solo fails -> ask_next_tick

    for d in m.devs:
        assert d.state == "blocked"
        assert d.blocker_stage == "ask_next_tick"
        # No helper recruited yet
        assert d.helper is None
        assert d.help_timer is None


def test_helper_locked_in_helping_state_does_not_advance_own_task(model_factory):
    """A dev in the 'helping' state is a no-op: its own task does not progress."""
    m = model_factory(block_prob=0.0)
    helper = m.devs[0]
    # Manually put helper into 'helping' state with a task in progress
    from model import Task
    helper_task = Task(story_points=3, work_remaining=10, state="in_progress")
    helper.current_task = helper_task
    helper.state = "helping"
    helper.prev_state = "working"

    work_before = helper_task.work_remaining
    helper.step()  # noop
    helper.step()  # noop
    helper.step()  # noop

    assert helper.state == "helping"  # still locked
    assert helper_task.work_remaining == work_before  # no progress


def test_waiting_for_help_resolves_and_restores_helper_state(model_factory):
    """When help_timer expires, requester unblocks and helper state is restored."""
    m = model_factory(
        block_prob=0.0,             # avoid spurious re-blocks during the test
        sync_help_mean=1,
        sync_help_jitter=0,
        async_help_mean=1,
        async_help_jitter=0,
        sp_scaling_k=100,           # tasks won't complete during the test
        n_devs=2,
    )
    m.step()  # idle -> working

    helper = m.devs[0]
    requester = m.devs[1]
    helper.location = "office"
    requester.location = "office"

    # Pin requester at ask_next_tick (mid-blocker, ready to recruit)
    requester.state = "blocked"
    requester.blocker_stage = "ask_next_tick"
    requester.block_start_tick = m.tick
    requester.current_task.state = "blocked"

    # Recruit step: requester finds helper, duration=1 → waiting_for_help
    m.step()
    assert requester.blocker_stage == "waiting_for_help"
    assert requester.helper is helper
    assert helper.state == "helping"
    assert helper.prev_state == "working"
    assert requester.help_timer == 1

    # One more tick: timer 1 -> 0 -> resolve
    m.step()
    assert requester.state == "working"
    assert requester.blocker_stage is None
    assert requester.helper is None
    # Helper state restored to what it was before recruitment
    assert helper.state == "working"
    assert helper.prev_state is None
