"""Location rolls: initial state and day-boundary re-rolls.

Regression for fix D: locations are rolled in __init__ (so the tick-0
datacollector row sees the real state, not the constructor default of all
"office") AND re-rolled at every day boundary EXCEPT tick 0.
"""

from __future__ import annotations

from model import TICKS_PER_DAY


def test_all_remote_at_tick_zero_with_remote_share_one(model_factory):
    """remote_share=1.0 → every dev is 'home' at construction time."""
    m = model_factory(remote_share=1.0)
    assert m.tick == 0
    for d in m.devs:
        assert d.location == "home"


def test_all_office_at_tick_zero_with_remote_share_zero(model_factory):
    """remote_share=0.0 → every dev is 'office' at construction time."""
    m = model_factory(remote_share=0.0)
    assert m.tick == 0
    for d in m.devs:
        assert d.location == "office"


def test_locations_rerolled_at_day_boundary(model_factory):
    """After ticking through a full day, locations are re-rolled.

    Concretely: with remote_share=1.0, locations stay 'home' across the
    boundary; with remote_share=0.0, they stay 'office'. Then we test that
    a mid-sprint flip from all-home to all-office takes effect on the next
    boundary.
    """
    m = model_factory(remote_share=1.0)
    # Tick from 0 to 32 (boundary). At tick=32, the boundary reroll fires.
    for _ in range(TICKS_PER_DAY):
        m.step()
    assert m.tick == TICKS_PER_DAY
    # remote_share is still 1.0 so all devs are still home (no actual flip,
    # but the reroll happened — exercise the code path)
    for d in m.devs:
        assert d.location == "home"

    # Flip remote_share to 0.0, step until next boundary; locations must
    # become "office".
    m.remote_share = 0.0
    for _ in range(TICKS_PER_DAY):
        m.step()
    assert m.tick == 2 * TICKS_PER_DAY
    for d in m.devs:
        assert d.location == "office"


def test_no_reroll_on_tick_zero_step(model_factory):
    """The first call to step() must NOT re-roll locations.

    Setup: build with remote_share=1.0 so __init__ rolls all devs home.
    Then flip remote_share to 0.0 and call step(). Because tick=0 at the
    moment step() runs, the guard `if self.tick > 0 and ...` prevents
    re-rolling. Locations must remain 'home'.
    """
    m = model_factory(remote_share=1.0)
    for d in m.devs:
        assert d.location == "home"  # initial roll

    m.remote_share = 0.0  # would force 'office' on next reroll

    m.step()
    # tick=0 at start of step, guard prevents reroll. Locations unchanged.
    for d in m.devs:
        assert d.location == "home"


def test_no_reroll_between_day_boundaries(model_factory):
    """Locations are stable between day boundaries.

    Set remote_share so __init__ pins everyone home. Then flip the share
    mid-day and step a non-boundary tick — locations must NOT change.
    """
    m = model_factory(remote_share=1.0)

    # Step a few ticks to leave tick=0 (so the tick-0 guard is irrelevant)
    m.step()  # tick=1
    m.step()  # tick=2
    assert all(d.location == "home" for d in m.devs)

    m.remote_share = 0.0  # would roll office if a reroll happened
    # Step non-boundary ticks (tick goes 2 -> ... < 32)
    for _ in range(5):
        m.step()

    # Locations must still be 'home' — no reroll at non-boundary ticks
    for d in m.devs:
        assert d.location == "home"


def test_mixed_locations_with_intermediate_remote_share(model_factory):
    """A non-extreme remote_share should produce both 'home' and 'office'
    at tick 0, given a deterministic seed.

    This protects against accidental changes that hard-code locations.
    """
    m = model_factory(remote_share=0.5, n_devs=10, seed=42)
    locs = [d.location for d in m.devs]
    assert "home" in locs
    assert "office" in locs
    # Sanity: only valid values
    for loc in locs:
        assert loc in ("home", "office")
