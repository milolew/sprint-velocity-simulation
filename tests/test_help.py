"""Help-session mechanics: sync vs async, office preference, edge cases."""

from __future__ import annotations


def _force_block_solo_fail(m, requester):
    """Inject a deterministic blocker on `requester` and put it at ask_next_tick.

    Avoids relying on multi-tick simulation to reach the desired state.
    """
    requester.state = "blocked"
    requester.blocker_stage = "ask_next_tick"
    requester.block_start_tick = m.tick


def test_sync_help_duration_used_when_both_in_office(model_factory):
    """When both requester and helper are in office, the SYNC distribution applies.

    Forced sync_help_mean=2, jitter=0 → duration always 2.

    Wait math: with block_start_tick = T (the recruit tick) the recruit step's
    pass 2 sets timer=2 (still tick T). Two more steps decrement the timer
    in pass 1 before _unblock fires; _unblock reads m.tick = T+2, so the
    recorded wait is (T+2) - T = 2.
    """
    m = model_factory(
        block_prob=0.0,
        sync_help_mean=2,
        sync_help_jitter=0,
        async_help_mean=20,
        async_help_jitter=0,
        sp_scaling_k=100,
        n_devs=2,
    )
    m.step()  # both go working

    requester, helper = m.devs[0], m.devs[1]
    requester.location = "office"
    helper.location = "office"

    # Inject blocker on requester at the current tick (block_start_tick=tick)
    _force_block_solo_fail(m, requester)
    requester.current_task.state = "blocked"

    # Tick: requester finds helper. duration=2 → waiting_for_help with timer=2
    m.step()
    assert requester.blocker_stage == "waiting_for_help"
    assert requester.help_timer == 2

    # Two more ticks: timer 2 -> 1 -> 0 (resolved)
    m.step()
    assert requester.blocker_stage == "waiting_for_help"
    assert requester.help_timer == 1

    m.step()
    assert requester.state == "working"
    assert m.wait_times[-1] == 2


def test_async_help_duration_used_when_one_is_remote(model_factory):
    """If either side is remote, ASYNC distribution applies.

    Wait math: with block_start_tick = T (recruit tick), duration=10. The
    timer hits 0 in _tick_help_session when m.tick = T+10, so the recorded
    wait is exactly 10 (the session length).
    """
    m = model_factory(
        block_prob=0.0,
        sync_help_mean=1,
        sync_help_jitter=0,
        async_help_mean=10,
        async_help_jitter=0,
        sp_scaling_k=100,
        n_devs=2,
    )
    m.step()  # both working

    requester, helper = m.devs[0], m.devs[1]
    requester.location = "office"
    helper.location = "home"  # async

    _force_block_solo_fail(m, requester)
    requester.current_task.state = "blocked"

    m.step()  # recruit
    assert requester.blocker_stage == "waiting_for_help"
    assert requester.help_timer == 10

    # Tick down 10 times; resolve on the 10th tick (timer hits 0)
    for _ in range(10):
        m.step()

    assert requester.state == "working"
    assert m.wait_times[-1] == 10


def test_zero_duration_session_resolves_immediately(model_factory):
    """sync_help_mean=0, jitter=0 → duration<=0 → resolves in the recruit tick.

    Regression for fix B: ensures we don't pad a forced 1-tick session when
    the sampled duration is 0.

    Wait math: block_start_tick = T (the recruit tick). _unblock runs in pass 2
    of the same step, when m.tick is still T → wait = T - T = 0. The point of
    this regression: in the buggy version, duration=0 was forced to 1, so the
    session went into waiting_for_help and resolved one tick later, giving
    wait=1.
    """
    m = model_factory(
        block_prob=0.0,
        sync_help_mean=0,
        sync_help_jitter=0,
        async_help_mean=0,
        async_help_jitter=0,
        sp_scaling_k=100,
        n_devs=2,
    )
    m.step()
    requester, helper = m.devs[0], m.devs[1]
    requester.location = "office"
    helper.location = "office"

    _force_block_solo_fail(m, requester)
    requester.current_task.state = "blocked"

    m.step()  # recruit AND resolve in the same tick

    # Never enters waiting_for_help
    assert requester.blocker_stage is None
    assert requester.state == "working"
    # Helper restored
    assert helper.state == "working"
    assert helper.prev_state is None
    # No +1 from a forced 1-tick session — the recorded wait is 0.
    assert m.wait_times[-1] == 0


def test_no_helper_available_keeps_dev_in_ask_next_tick(model_factory):
    """When all other devs are blocked or helping, requester stays put."""
    m = model_factory(
        block_prob=0.0,
        n_devs=2,
    )
    m.step()
    a, b = m.devs[0], m.devs[1]

    # Put both at ask_next_tick, no candidates
    _force_block_solo_fail(m, a)
    _force_block_solo_fail(m, b)
    a.current_task.state = "blocked"
    b.current_task.state = "blocked"

    for _ in range(5):
        m.step()
        assert a.state == "blocked"
        assert b.state == "blocked"
        # Both stay in ask_next_tick (every tick they retry, but no candidates)
        assert a.blocker_stage == "ask_next_tick"
        assert b.blocker_stage == "ask_next_tick"
        assert a.helper is None
        assert b.helper is None
    # Nothing resolved
    assert m.wait_times == []


def test_self_excluded_from_helper_candidates(model_factory):
    """Single-dev model: dev cannot be its own helper, stays blocked forever."""
    m = model_factory(
        block_prob=0.0,
        n_devs=1,
    )
    m.step()
    dev = m.devs[0]
    _force_block_solo_fail(m, dev)
    dev.current_task.state = "blocked"

    for _ in range(20):
        m.step()
        assert dev.state == "blocked"
        assert dev.blocker_stage == "ask_next_tick"
        assert dev.helper is None
    # No help session ever started
    assert m.wait_times == []


def test_blocked_and_helping_devs_excluded_from_candidates(model_factory):
    """Only devs in {working, idle} can be picked as helpers."""
    m = model_factory(
        block_prob=0.0,
        n_devs=4,
    )
    m.step()  # everybody working

    requester = m.devs[0]
    requester.location = "office"
    # Ineligible candidates:
    m.devs[1].state = "blocked"
    m.devs[1].blocker_stage = "solo"
    m.devs[1].block_start_tick = m.tick
    m.devs[2].state = "helping"
    m.devs[2].prev_state = "working"
    # Eligible:
    m.devs[3].state = "working"
    for d in m.devs[1:]:
        d.location = "office"

    # Force ask_next_tick on requester
    _force_block_solo_fail(m, requester)
    requester.current_task.state = "blocked"

    # Run one step. The only eligible candidate is m.devs[3]; it must get picked.
    m.step()

    assert requester.helper is m.devs[3]
    assert m.devs[3].state == "helping"
    # Ineligible candidates were not touched
    assert m.devs[1].state == "blocked"
    assert m.devs[2].state == "helping"


def test_office_preference_when_office_helpers_available(model_factory):
    """An office requester should prefer office helpers over remote ones.

    With one office helper and one remote helper, the recruiter must always
    pick the office one. Run multiple times with different seeds to be sure
    it isn't a randomness coincidence.
    """
    for seed in range(10):
        m = model_factory(seed=seed, block_prob=0.0, n_devs=3)
        m.step()
        requester = m.devs[0]
        office_helper = m.devs[1]
        remote_helper = m.devs[2]

        requester.location = "office"
        office_helper.location = "office"
        remote_helper.location = "home"

        _force_block_solo_fail(m, requester)
        requester.current_task.state = "blocked"

        m.step()
        assert requester.helper is office_helper, (
            f"seed={seed}: office requester picked the remote helper "
            f"despite an office helper being available"
        )


def test_remote_requester_has_no_office_preference(model_factory):
    """Remote requester treats all candidates equally — both can be picked.

    Run many seeds; expect both helpers to be selected at least once.
    """
    picked_office = 0
    picked_remote = 0
    for seed in range(40):
        m = model_factory(seed=seed, block_prob=0.0, n_devs=3)
        m.step()
        requester = m.devs[0]
        office_helper = m.devs[1]
        remote_helper = m.devs[2]

        requester.location = "home"          # remote
        office_helper.location = "office"
        remote_helper.location = "home"

        _force_block_solo_fail(m, requester)
        requester.current_task.state = "blocked"

        m.step()
        if requester.helper is office_helper:
            picked_office += 1
        elif requester.helper is remote_helper:
            picked_remote += 1
        else:
            raise AssertionError(f"seed={seed}: unexpected helper {requester.helper}")
    # With 40 seeds and a 50/50 split expected, both should be > 0.
    assert picked_office > 0
    assert picked_remote > 0


def test_help_timer_counts_through_day_boundary(model_factory):
    """Day rollover (tick 32) does NOT reset the help_timer.

    Sets up a long async session that straddles a day boundary; verifies
    the timer keeps counting down monotonically across the boundary.
    """
    m = model_factory(
        block_prob=0.0,
        sync_help_mean=10,
        sync_help_jitter=0,
        async_help_mean=10,
        async_help_jitter=0,
        sp_scaling_k=10_000,            # tasks never complete during the test
        n_devs=2,
        sprint_length=128,
    )

    # Skip ahead to tick=28 so a duration=10 session crosses tick=32.
    while m.tick < 28:
        m.step()

    requester, helper = m.devs[0], m.devs[1]
    requester.location = "home"
    helper.location = "home"

    # Make sure both still have a current_task (sp_scaling_k=10_000 guarantees this)
    assert helper.current_task is not None
    assert requester.current_task is not None

    _force_block_solo_fail(m, requester)
    requester.current_task.state = "blocked"
    block_tick = m.tick

    # Recruit; duration=10, becomes waiting_for_help with timer=10
    m.step()
    assert requester.blocker_stage == "waiting_for_help"
    assert requester.help_timer == 10
    timer_prev = requester.help_timer

    # Step through the boundary, asserting strictly decreasing timer
    while requester.blocker_stage == "waiting_for_help":
        m.step()
        if requester.blocker_stage == "waiting_for_help":
            assert requester.help_timer < timer_prev, (
                f"Timer reset across day boundary: {timer_prev} -> "
                f"{requester.help_timer} at tick={m.tick}"
            )
            timer_prev = requester.help_timer

    # Resolved exactly when timer hit 0; wait equals the session length (10)
    assert requester.state == "working"
    assert m.wait_times[-1] == 10
    # Sanity: this run did cross the day boundary
    assert m.tick > 32
    assert block_tick < 32
