"""Shared fixtures for the TeamModel test suite.

The simulation is fully deterministic given a seed. Tests force-trigger
specific code paths via parameter overrides (e.g. block_prob=1.0,
mean_solo_skill=1.0 with solo_skill_spread=0.0) rather than running thousands
of ticks and hoping the right state happens to be reached.
"""

from __future__ import annotations

import sys
import os

import pytest

# Make the project root importable so `from model import ...` works
# regardless of the current working directory the test runner uses.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import TeamModel  # noqa: E402


# Defaults chosen to be:
#   - small (fast) sprint
#   - deterministic (seed)
#   - quiet (no blockers, no skill noise) — individual tests override what they need
DEFAULT_OVERRIDES = dict(
    seed=42,
    n_devs=5,
    sprint_length=64,
    backlog_size=200,
    block_prob=0.0,
    mean_solo_skill=0.5,
    solo_skill_spread=0.0,
    sync_help_mean=1,
    sync_help_jitter=0,
    async_help_mean=5,
    async_help_jitter=0,
    remote_share=0.5,
    # New mechanics gated OFF by default so the suite tests pure mechanics.
    n_sprints=1,
    learning_rate=0.0,
    annual_attrition=0.0,
)


@pytest.fixture
def model_factory():
    """Return a callable that builds TeamModel with sensible test defaults.

    Usage:
        m = model_factory(block_prob=1.0, sprint_length=10)

    Any default can be overridden via kwargs.
    """
    def _make(**overrides):
        params = {**DEFAULT_OVERRIDES, **overrides}
        return TeamModel(**params)
    return _make


@pytest.fixture
def model(model_factory):
    """Convenience: a model with default test parameters."""
    return model_factory()
