"""Two-phase activation regression test.

The critical regression for fix A: a help-seeker MUST act after every other
dev has already taken their tick. So when a working dev is recruited as a
helper, that working dev must NOT also have decremented their task's
work_remaining in the same recruitment tick.

In a single-pass scheduler with an unlucky iteration order [working, blocked],
the working dev would have already done its work step by the time the
blocked dev recruited it — that's the bug.

In the two-pass scheduler, the blocked dev runs in pass 2 only after every
non-seeker has finished pass 1, so any helper picked in pass 2 has already
done its work for the tick — but that "already done" was its OWN choice
(decrement or block), not under the blocked dev's control. The invariant
the test asserts is the OPPOSITE direction: the helper, once recruited,
does NOT step again as a working dev in the same tick.

Concretely: at the START of the recruitment tick, helper.current_task has
work_remaining=W. By the END of the recruitment tick, helper.state should
be "helping" and work_remaining should be at most W-1 (the one decrement in
pass 1, before recruitment). It must NOT be W-2 (would imply double-step).
"""

from __future__ import annotations


def test_helper_does_not_step_twice_in_recruitment_tick(model_factory):
    """Helper performs at most one work decrement in the recruitment tick.

    Setup:
      - 2 devs.
      - Requester pinned at ask_next_tick (will recruit in pass 2).
      - Helper is working with a fresh task; work_remaining captured before tick.
    After the tick:
      - helper.state == "helping"
      - helper.current_task.work_remaining decreased by AT MOST 1.

    Run many seeds: in pass 1, requester (a seeker) is skipped, so helper
    runs alone and decrements once. In pass 2, requester recruits helper.
    If a single-pass implementation regressed, helper might decrement again
    after being recruited (or, depending on order, before AND after).
    """
    for seed in range(20):
        m = model_factory(
            seed=seed,
            block_prob=0.0,
            sync_help_mean=10,         # long enough not to resolve in same tick
            sync_help_jitter=0,
            async_help_mean=10,
            async_help_jitter=0,
            n_devs=2,
            sp_scaling_k=100,           # huge tasks, never complete
        )
        m.step()  # both go idle -> working

        requester, helper = m.devs[0], m.devs[1]
        requester.location = "office"
        helper.location = "office"

        # Pin requester at ask_next_tick (so it's a help-seeker in pass 2)
        requester.state = "blocked"
        requester.blocker_stage = "ask_next_tick"
        requester.block_start_tick = m.tick
        requester.current_task.state = "blocked"

        wr_before = helper.current_task.work_remaining
        helper_state_before = helper.state

        m.step()

        # Helper was working before; in pass 1, it decremented work by 1.
        # Then in pass 2, requester recruited it (state -> "helping").
        # Critically: helper did NOT decrement a SECOND time after being recruited.
        assert helper.state == "helping", f"seed={seed}: helper not recruited"
        assert helper_state_before == "working"
        decrement = wr_before - helper.current_task.work_remaining
        assert decrement == 1, (
            f"seed={seed}: helper.current_task.work_remaining moved by "
            f"{decrement} (expected exactly 1: the pass-1 decrement only). "
            f"This is the single-pass regression."
        )


def test_recruited_helper_does_not_decrement_during_session(model_factory):
    """While in 'helping' state, the helper's task does not progress.

    Once recruited, the helper's pass-1 step is a no-op (state == 'helping').
    Verifies the work_remaining is constant across the session ticks.
    """
    m = model_factory(
        block_prob=0.0,
        sync_help_mean=5,
        sync_help_jitter=0,
        n_devs=2,
        sp_scaling_k=100,
    )
    m.step()  # both working
    requester, helper = m.devs[0], m.devs[1]
    requester.location = "office"
    helper.location = "office"

    requester.state = "blocked"
    requester.blocker_stage = "ask_next_tick"
    requester.block_start_tick = m.tick
    requester.current_task.state = "blocked"

    m.step()  # recruit; helper now in 'helping', timer=5
    assert helper.state == "helping"
    wr_when_helping = helper.current_task.work_remaining

    # Tick through the session; helper's task must not progress while helping
    for _ in range(4):
        m.step()
        if helper.state == "helping":
            assert helper.current_task.work_remaining == wr_when_helping
        else:
            break

    # Final resolution: helper returns to working
    while requester.state == "blocked":
        m.step()
    assert helper.state == "working"
