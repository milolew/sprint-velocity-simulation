"""Boundary validation: _validate_params fails fast on out-of-range knobs.

The model validates its parameters in __init__ (fail fast at the boundary) so
that a bad sweep configuration crashes immediately rather than producing
silently-wrong long-run results. Each case here pins one rejected input.
"""

from __future__ import annotations

import pytest

from model import TeamModel


@pytest.mark.parametrize("kwargs", [
    pytest.param(dict(skill_cap=1.5), id="skill_cap_above_one"),
    pytest.param(dict(skill_cap=-0.1), id="skill_cap_below_zero"),
    pytest.param(dict(learning_rate=-0.1), id="negative_learning_rate"),
    pytest.param(dict(annual_attrition=1.5), id="attrition_above_one"),
    pytest.param(dict(annual_attrition=-0.1), id="attrition_below_zero"),
    pytest.param(dict(n_sprints=0), id="n_sprints_zero"),
    pytest.param(dict(n_sprints=-3), id="n_sprints_negative"),
    pytest.param(dict(sprints_per_year=0), id="sprints_per_year_zero"),
    pytest.param(dict(sync_help_weight=-1.0), id="negative_sync_weight"),
    pytest.param(dict(async_help_weight=-0.5), id="negative_async_weight"),
    pytest.param(dict(task_completion_weight=-0.5), id="negative_completion_weight"),
    pytest.param(dict(solo_resolution_weight=-0.5), id="negative_solo_weight"),
    pytest.param(dict(remote_attrition_coef=float("inf")), id="infinite_coef"),
    pytest.param(dict(remote_attrition_coef=float("nan")), id="nan_coef"),
    pytest.param(dict(remote_baseline=1.5), id="baseline_above_one"),
    pytest.param(dict(remote_baseline=-0.1), id="baseline_below_zero"),
    pytest.param(dict(junior_mean_skill=-0.1), id="negative_junior_mean"),
    pytest.param(dict(junior_skill_spread=-0.1), id="negative_junior_spread"),
])
def test_validate_params_rejects_bad_input(kwargs):
    """Every out-of-range knob raises ValueError at construction time."""
    with pytest.raises(ValueError):
        TeamModel(seed=1, **kwargs)


def test_validate_params_accepts_valid_boundary_values():
    """The inclusive boundaries (0 and 1 where allowed) must NOT be rejected."""
    # Should construct cleanly: skill_cap at both ends, lr/attrition at edges,
    # n_sprints=1, weights at 0, a finite signed coef, baseline at both ends.
    TeamModel(
        seed=1,
        skill_cap=0.0,
        learning_rate=0.0,
        annual_attrition=0.0,
        n_sprints=1,
        sprints_per_year=1,
        sync_help_weight=0.0,
        async_help_weight=0.0,
        task_completion_weight=0.0,
        solo_resolution_weight=0.0,
        remote_attrition_coef=-1.0,
        remote_baseline=1.0,
        junior_mean_skill=0.0,
        junior_skill_spread=0.0,
    )
    TeamModel(seed=1, skill_cap=1.0, annual_attrition=1.0, remote_baseline=0.0)


def test_validation_message_names_the_offending_param():
    """The error is actionable: it names which knob was out of range.

    A bare ValueError forces the operator to guess; the message must point at
    the bad parameter so a misconfigured sweep is diagnosable.
    """
    with pytest.raises(ValueError, match="n_sprints"):
        TeamModel(seed=1, n_sprints=0)
    with pytest.raises(ValueError, match="skill_cap"):
        TeamModel(seed=1, skill_cap=2.0)
