"""End-to-end smoke tests: does the simulation run without crashing?"""

from __future__ import annotations


def test_full_sprint_runs_to_completion(model_factory):
    # Realistic-ish parameters so we know it isn't trivially passing
    m = model_factory(
        sprint_length=320,
        block_prob=0.02,
        mean_solo_skill=0.5,
        solo_skill_spread=0.2,
    )
    while m.running:
        m.step()

    assert m.tick == 320
    assert m.running is False
    # With block_prob=0.02 and 5 devs over 320 ticks, work definitely gets done.
    assert m.velocity > 0
    assert len(m.completed_tasks) > 0


def test_initial_state_consistent(model_factory):
    m = model_factory()
    assert m.tick == 0
    assert m.running is True
    assert m.velocity == 0
    assert m.wait_times == []
    assert m.completed_tasks == []
    assert len(m.devs) == 5
    # Every dev starts idle and unassigned
    for d in m.devs:
        assert d.state == "idle"
        assert d.current_task is None
        assert d.blocker_stage is None
