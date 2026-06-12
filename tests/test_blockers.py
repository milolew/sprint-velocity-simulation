"""Blocker mechanics: when can blockers happen, and how do they progress?"""

from __future__ import annotations


def test_blockers_only_arise_while_working(model_factory):
    """Helpers, idlers, and already-blocked devs cannot trigger NEW blockers.

    block_prob is rolled in _step_working only. So if every dev is already
    blocked or helping, the blocker counter cannot increase further.
    """
    m = model_factory(
        block_prob=1.0,            # would block every working dev every tick
        mean_solo_skill=0.0,        # solo always fails -> ask_next_tick
        solo_skill_spread=0.0,
        sp_scaling_k=20,
        n_devs=2,
        async_help_mean=20,         # very long help session — keeps helper locked
        async_help_jitter=0,
        sync_help_mean=20,
        sync_help_jitter=0,
    )

    m.step()  # idle -> working (both)
    m.step()  # working -> blocked(solo) (both)
    m.step()  # solo fails -> ask_next_tick for both

    # At this point, both devs are blocked. They will try to recruit, but
    # there are no candidates (the only other dev is also blocked).
    # The blocker count is exactly 2 (both currently blocked).
    blocked_count = sum(1 for d in m.devs if d.state == "blocked")
    assert blocked_count == 2

    # Run more ticks; since both are blocked-ask_next_tick with no helpers,
    # neither can resolve. No NEW blockers can be created (no working devs).
    for _ in range(20):
        m.step()
        # Confirm: nobody is working, so no new blockers can fire
        assert all(d.state != "working" for d in m.devs)
        assert all(d.blocker_stage in ("ask_next_tick", "solo", "waiting_for_help")
                   for d in m.devs if d.state == "blocked")

    # No blockers ever resolved (no helpers available)
    assert m.wait_times == []


def test_solo_success_avoids_ask_next_tick(model_factory):
    """Solo success should NOT advance to ask_next_tick.

    Wait time is tick - block_start_tick. With block at tick T, solo success
    at tick T+1, the recorded wait is 1.
    """
    m = model_factory(
        block_prob=1.0,
        mean_solo_skill=1.0,        # solo always succeeds
        solo_skill_spread=0.0,
        sp_scaling_k=20,
        n_devs=1,
    )
    m.step()  # tick=0 step -> idle->working; tick now 1
    # At tick 1, dev is working, will block on this step.
    m.step()  # tick=1 step -> working->blocked(solo) at tick=1; tick now 2
    dev = m.devs[0]
    block_tick = dev.block_start_tick
    assert dev.state == "blocked"
    assert dev.blocker_stage == "solo"

    m.step()  # tick=2 step -> solo success -> working

    assert dev.state == "working"
    assert dev.blocker_stage is None
    # Wait was: tick - block_start_tick computed inside _unblock at tick=2
    expected_wait = 2 - block_tick
    assert m.wait_times == [expected_wait]
    assert m.wait_times[0] == 1  # block at 1, resolve at 2 -> wait 1


def test_solo_failure_delays_help_finding_by_one_tick(model_factory):
    """Block at T, solo fail at T+1, ask at T+2.

    The solo-fail tick must NOT also recruit a helper in the same tick;
    that's what `ask_next_tick` enforces.
    """
    m = model_factory(
        block_prob=1.0,
        mean_solo_skill=0.0,
        solo_skill_spread=0.0,
        sp_scaling_k=50,
        sync_help_mean=1,
        sync_help_jitter=0,
        async_help_mean=1,
        async_help_jitter=0,
        n_devs=2,
    )
    # Force one dev to be a stable helper:
    requester = m.devs[0]
    helper = m.devs[1]
    helper.location = "office"
    requester.location = "office"

    m.step()  # both: idle -> working
    # Make sure the helper never blocks: pin its solo path to be irrelevant
    # by overriding block_prob effectively for it. Easier: run only one
    # tick of working, then take direct control.
    # Reset both: requester will block, helper will stay working forever.
    helper_task = helper.current_task
    helper.state = "working"
    helper_task.work_remaining = 10_000  # never completes during the test

    # Step requester through block manually so helper stays put:
    requester.state = "blocked"
    requester.blocker_stage = "solo"
    requester.block_start_tick = m.tick  # block starts at current tick
    requester.current_task.state = "blocked"
    block_tick = m.tick

    # Now step the model. block_prob=1.0 would re-block helper, so neutralize:
    m.block_prob = 0.0

    # Tick 1: solo attempt fails -> ask_next_tick (no help finding this tick!)
    m.step()
    assert requester.state == "blocked"
    assert requester.blocker_stage == "ask_next_tick"
    assert requester.helper is None
    assert requester.help_timer is None

    # Tick 2: now actually recruit helper (sync, duration=1)
    m.step()
    # Either resolved immediately (duration<=0 path) or in waiting_for_help.
    # With sync_help_mean=1, jitter=0, duration is 1 → waiting_for_help.
    assert requester.blocker_stage == "waiting_for_help"
    assert requester.helper is helper
    assert helper.state == "helping"

    # Block at tick T, solo-fail consumes tick T (recorded inside step at
    # tick=T, then tick advances to T+1). Recruit at T+1 (waiting_for_help,
    # timer=1, then tick advances to T+2). Resolve at T+2 (timer 1->0, unblock
    # uses m.tick which is still T+2, then tick advances to T+3).
    # → wait_times[-1] = T+2 - T = 2.
    m.step()  # timer 1 -> 0, resolve
    assert requester.state == "working"
    assert m.wait_times == [2]
    # m.tick was incremented after the resolve step, so it now reads T+3.
    assert m.tick - block_tick == 3
