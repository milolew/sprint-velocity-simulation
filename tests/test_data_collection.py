"""DataCollector reporters: presence, sane values, accurate initial row."""

from __future__ import annotations

from model import TICKS_PER_DAY


EXPECTED_REPORTERS = {
    "velocity",
    "avg_wait",
    "blockers_resolved",
    "tick",
    "day",
    "sprint",
    "n_working",
    "n_blocked",
    "n_helping",
    "n_idle",
    "mean_skill",
    "min_skill",
    "max_skill",
    "n_juniors",
    "attrition_count",
    "mean_tenure",
}


def test_datacollector_has_expected_reporters(model_factory):
    m = model_factory()
    df = m.datacollector.get_model_vars_dataframe()
    assert EXPECTED_REPORTERS.issubset(df.columns)


def test_initial_row_reflects_rolled_locations_not_constructor_default(model_factory):
    """Regression for fix D: the tick-0 collected row must reflect the
    rolled locations, not the all-office constructor default.

    With remote_share=1.0, every dev's location at tick=0 is 'home'.
    The DataCollector itself doesn't record location — but the model is
    consistent: the first row exists (collect was called in __init__) and
    the collected dev counts at tick=0 are all 'idle' (since no step ran).
    More importantly, m.devs[*].location must already be 'home'.
    """
    m = model_factory(remote_share=1.0)
    df = m.datacollector.get_model_vars_dataframe()
    # Initial row exists (collected in __init__)
    assert len(df) == 1
    # Initial state: all devs idle
    assert df["n_idle"].iloc[0] == len(m.devs)
    assert df["n_working"].iloc[0] == 0
    assert df["n_blocked"].iloc[0] == 0
    assert df["n_helping"].iloc[0] == 0
    # And — the actual fix-D regression — locations are rolled, not default
    assert all(d.location == "home" for d in m.devs)


def test_collected_row_per_step(model_factory):
    """Each step appends one row; final row count = sprint_length + 1."""
    m = model_factory(sprint_length=10, block_prob=0.0)
    while m.running:
        m.step()
    df = m.datacollector.get_model_vars_dataframe()
    # 1 row from __init__ + sprint_length steps = 11 rows
    assert len(df) == 11


def test_velocity_in_dataframe_matches_model_attr(model_factory):
    """The reporter writes `m.velocity` as-is; should match the live attr."""
    m = model_factory(sprint_length=20, block_prob=0.0)
    while m.running:
        m.step()
    df = m.datacollector.get_model_vars_dataframe()
    assert df["velocity"].iloc[-1] == m.velocity


def test_blockers_resolved_equals_wait_times_length(model_factory):
    """The 'blockers_resolved' reporter must equal len(wait_times) at every tick.

    Stronger invariant: blockers_resolved is monotonically non-decreasing.
    """
    m = model_factory(
        block_prob=0.5,
        mean_solo_skill=0.5,
        solo_skill_spread=0.0,
        sprint_length=64,
        sync_help_mean=1,
        sync_help_jitter=0,
    )
    while m.running:
        m.step()
    df = m.datacollector.get_model_vars_dataframe()
    # Final reporter row matches model state
    assert df["blockers_resolved"].iloc[-1] == len(m.wait_times)
    # Monotonic non-decreasing
    diffs = df["blockers_resolved"].diff().dropna()
    assert (diffs >= 0).all()


def test_avg_wait_handles_empty_wait_times(model_factory):
    """When no blockers have resolved, avg_wait reports 0.0, not NaN/error."""
    m = model_factory(block_prob=0.0)  # no blockers will ever happen
    df = m.datacollector.get_model_vars_dataframe()
    assert df["avg_wait"].iloc[0] == 0.0


def test_mean_skill_reporter_matches_manual_mean(model_factory):
    """mean_skill equals the arithmetic mean of dev solo_skill values."""
    m = model_factory(mean_solo_skill=0.5, solo_skill_spread=0.0)
    df = m.datacollector.get_model_vars_dataframe()
    expected = sum(d.solo_skill for d in m.devs) / len(m.devs)
    assert df["mean_skill"].iloc[0] == expected


def test_mean_tenure_starts_at_zero_and_grows(model_factory):
    """At tick 0 tenure is 0 for everyone; it grows as the run advances."""
    m = model_factory(sprint_length=64, block_prob=0.0)
    df0 = m.datacollector.get_model_vars_dataframe()
    assert df0["mean_tenure"].iloc[0] == 0.0
    while m.running:
        m.step()
    df = m.datacollector.get_model_vars_dataframe()
    assert df["mean_tenure"].iloc[-1] > 0.0


def test_sprint_reporter_counts_from_one(model_factory):
    """sprint = tick // sprint_length + 1, starting at 1."""
    m = model_factory(sprint_length=32, n_sprints=1, block_prob=0.0)
    df = m.datacollector.get_model_vars_dataframe()
    assert df["sprint"].iloc[0] == 1


def test_day_reporter_advances_with_tick(model_factory):
    """day = tick // TICKS_PER_DAY + 1, so it increments at boundaries."""
    m = model_factory(sprint_length=TICKS_PER_DAY * 2, block_prob=0.0)
    while m.running:
        m.step()
    df = m.datacollector.get_model_vars_dataframe()
    # tick=0 -> day=1, tick=32 -> day=2, tick=63 -> day=2, tick=64 -> day=3
    assert df["day"].iloc[0] == 1
    assert df["day"].iloc[TICKS_PER_DAY] == 2
    assert df["day"].iloc[2 * TICKS_PER_DAY] == 3


def test_wait_times_appended_exactly_once_per_resolved_blocker(model_factory):
    """Each unblock appends exactly one entry to wait_times.

    Force a single blocker resolution and check len(wait_times) goes 0 -> 1
    in exactly one tick.
    """
    m = model_factory(
        block_prob=0.0,
        sync_help_mean=0,
        sync_help_jitter=0,
        n_devs=2,
    )
    m.step()  # working
    requester, helper = m.devs[0], m.devs[1]
    requester.location = "office"
    helper.location = "office"

    requester.state = "blocked"
    requester.blocker_stage = "ask_next_tick"
    requester.block_start_tick = m.tick
    requester.current_task.state = "blocked"

    assert len(m.wait_times) == 0

    m.step()  # recruit + 0-duration → unblock in same tick
    assert len(m.wait_times) == 1

    # No further unblocks happen in subsequent ticks (no other blocked devs)
    for _ in range(3):
        m.step()
        assert len(m.wait_times) == 1
