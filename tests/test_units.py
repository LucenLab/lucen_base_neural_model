"""Invariant 1 / spec section 7.1 - unit consistency.

Any displacement crossing a module boundary is in metres and within physically
sane bounds. Catches the nm/um class of error that silently flips the verdict.
"""

from __future__ import annotations

import math

import pytest

from lucen.base.units import (
    DISPLACEMENT_MAX_M,
    DISPLACEMENT_MIN_M,
    db20,
    is_sane_displacement_m,
    m_to_nm,
    m_to_um,
)
from lucen.source import synchrony_sweep


@pytest.mark.parametrize(
    "value_m, expected",
    [
        (1e-9, True),     # 1 nm - a plausible single-neuron displacement
        (2e-5, True),     # 20 um - a plausible fully-coherent voxel sum
        (1e-15, False),   # sub-picometre - always a units bug
        (1e-1, False),    # 10 cm - always a units bug
        (0.0, False),
        (-1e-9, False),
        (float("nan"), False),
        (float("inf"), False),
    ],
)
def test_is_sane_displacement_bounds(value_m, expected):
    assert is_sane_displacement_m(value_m) is expected


def test_sane_bounds_are_ordered():
    assert DISPLACEMENT_MIN_M < DISPLACEMENT_MAX_M


def test_reporting_converters_are_pure_scaling():
    assert m_to_nm(1e-9) == pytest.approx(1.0)
    assert m_to_um(1e-6) == pytest.approx(1.0)


def test_db20_matches_definition():
    assert db20(10.0) == pytest.approx(20.0)
    assert db20(1.0) == pytest.approx(0.0)
    assert db20(100.0) == pytest.approx(40.0)


def test_db20_rejects_nonpositive_ratio():
    with pytest.raises(ValueError):
        db20(0.0)
    with pytest.raises(ValueError):
        db20(-3.0)


def test_module1_outputs_are_sane_si(d_single, voxel, synchrony_grid):
    """Every displacement Module 1 emits crosses the boundary in sane SI metres."""
    sweep = synchrony_sweep(d_single, voxel, synchrony_grid)
    for s in sweep:
        assert math.isfinite(s.value_m)
        assert is_sane_displacement_m(s.value_m), (
            f"summed displacement {s.value_m} m at synchrony "
            f"{s.synchrony_fraction} is outside sane SI bounds"
        )
