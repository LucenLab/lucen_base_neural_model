"""Invariant / spec section 7.3 - coherence limits.

``summed_displacement`` reproduces ``N*d_1`` at ``s=1`` and ``sqrt(N)*d_1`` at
``s=0`` to tolerance. This is the acceptance boundary of spec section 3.2.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lucen.source import summed_displacement, synchrony_sweep


def test_fully_coherent_limit_is_N_times_d1(d_single, voxel):
    n = voxel.neuron_count
    result = summed_displacement(d_single, voxel, synchrony_fraction=1.0)
    assert result.value_m == pytest.approx(n * d_single.value_m, rel=1e-12)


def test_fully_incoherent_limit_is_sqrtN_times_d1(d_single, voxel):
    n = voxel.neuron_count
    result = summed_displacement(d_single, voxel, synchrony_fraction=0.0)
    assert result.value_m == pytest.approx(
        math.sqrt(n) * d_single.value_m, rel=1e-12
    )


def test_coherent_ceiling_exceeds_incoherent_floor(d_single, voxel):
    """The two limits are genuinely separated (N >> sqrt(N) for N=1e4)."""
    hi = summed_displacement(d_single, voxel, 1.0).value_m
    lo = summed_displacement(d_single, voxel, 0.0).value_m
    assert hi > lo
    assert hi / lo == pytest.approx(math.sqrt(voxel.neuron_count), rel=1e-9)


def test_sweep_is_monotonic_non_decreasing(d_single, voxel, synchrony_grid):
    """Synchrony is the swept variable (Invariant 4); the sum rises with it."""
    sweep = synchrony_sweep(d_single, voxel, synchrony_grid)
    vals = np.array([s.value_m for s in sweep])
    assert np.all(np.diff(vals) >= 0.0)


def test_sweep_endpoints_match_closed_form(d_single, voxel, synchrony_grid):
    sweep = synchrony_sweep(d_single, voxel, synchrony_grid)
    n = voxel.neuron_count
    assert sweep[0].synchrony_fraction == pytest.approx(0.0)
    assert sweep[-1].synchrony_fraction == pytest.approx(1.0)
    assert sweep[0].value_m == pytest.approx(math.sqrt(n) * d_single.value_m)
    assert sweep[-1].value_m == pytest.approx(n * d_single.value_m)


def test_synchrony_fraction_out_of_range_raises(d_single, voxel):
    with pytest.raises(ValueError):
        summed_displacement(d_single, voxel, synchrony_fraction=-0.01)
    with pytest.raises(ValueError):
        summed_displacement(d_single, voxel, synchrony_fraction=1.01)
