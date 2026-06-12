"""Smoke test for experiment.run_one().

Verifies the documented dict shape and that the call runs to completion
without crashing. main() is a CLI wrapper and is intentionally not tested.
"""

from __future__ import annotations

from experiment import run_one


EXPECTED_KEYS = {
    "remote_share",
    "mean_solo_skill",
    "solo_skill_spread",
    "seed",
    "velocity",
    "total_velocity",
    "avg_wait",
    "blockers_resolved",
    "completed_tasks",
    "mean_skill",
    "attrition_count",
}


def test_run_one_returns_expected_dict_shape():
    result = run_one(
        remote_share=0.5,
        mean_solo_skill=0.5,
        solo_skill_spread=0.2,
        seed=42,
        sprint_length=64,    # short, fast
        n_sprints=1,
    )
    assert set(result.keys()) == EXPECTED_KEYS


def test_run_one_echoes_input_parameters():
    """The result dict must echo the inputs that were passed in."""
    result = run_one(
        remote_share=0.7,
        mean_solo_skill=0.4,
        solo_skill_spread=0.1,
        seed=99,
        sprint_length=64,
        n_sprints=1,
    )
    assert result["remote_share"] == 0.7
    assert result["mean_solo_skill"] == 0.4
    assert result["solo_skill_spread"] == 0.1
    assert result["seed"] == 99


def test_run_one_produces_sane_metrics():
    """With a reasonable sprint length, velocity > 0, and counts non-negative."""
    result = run_one(
        remote_share=0.5,
        mean_solo_skill=0.5,
        solo_skill_spread=0.2,
        seed=42,
        sprint_length=64,
        n_sprints=1,
    )
    assert result["velocity"] > 0
    assert result["completed_tasks"] > 0
    assert result["blockers_resolved"] >= 0
    assert result["avg_wait"] >= 0


def test_run_one_is_deterministic():
    """Same inputs to run_one should produce identical outputs."""
    a = run_one(
        remote_share=0.3, mean_solo_skill=0.5, solo_skill_spread=0.1,
        seed=11, sprint_length=64, n_sprints=1,
    )
    b = run_one(
        remote_share=0.3, mean_solo_skill=0.5, solo_skill_spread=0.1,
        seed=11, sprint_length=64, n_sprints=1,
    )
    assert a == b
