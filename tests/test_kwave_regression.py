"""Invariant / spec section 7.6 - k-Wave regression.

Frozen one-way transmission for a reference skull-mimic, so a library/config change
that shifts propagation is caught immediately. Module 2 is not yet implemented, so
this is a placeholder that skips when k-Wave is unavailable and xfails (pending
implementation) when it is. Replace the body and freeze the reference value once
``propagate_round_trip`` lands.
"""

from __future__ import annotations

import importlib.util

import pytest

_HAS_KWAVE = importlib.util.find_spec("kwave") is not None

# Frozen reference (to be set once Module 2 produces a stable value). One-way skull
# transmission must land in a physically sane band: single-digit to low-tens-of-
# percent amplitude at these frequencies (spec section 4.3, sanity anchor).
REFERENCE_ONE_WAY_TRANSMISSION = None  # e.g. 0.12
TRANSMISSION_SANE_RANGE = (0.01, 0.40)  # amplitude fraction


@pytest.mark.skipif(
    not _HAS_KWAVE,
    reason="k-wave-python not installed (optional extra: uv sync --extra propagation)",
)
@pytest.mark.xfail(reason="Module 2 propagation not yet implemented", strict=True)
def test_reference_skull_transmission_is_frozen():
    """Pin the reference skull-mimic one-way transmission against config drift."""
    from lucen.propagation import load_skull_map, propagate_round_trip  # noqa: F401

    # Once implemented: build the reference skull-mimic, run an uncorrected round
    # trip, and assert one_way_transmission == REFERENCE_ONE_WAY_TRANSMISSION within
    # tolerance, and that it lies within TRANSMISSION_SANE_RANGE.
    raise NotImplementedError(
        "freeze reference transmission once propagate_round_trip is implemented"
    )
