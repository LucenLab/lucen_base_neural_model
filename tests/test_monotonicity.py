"""Invariant / spec section 7.5 - monotonicity sanity.

Contrast is non-decreasing in synchrony and non-increasing in skull attenuation. A
violation means a sign error somewhere in the chain. The synchrony direction is
testable now through Module 1's source term; the skull-attenuation direction needs
Module 2 and is marked xfail until propagation lands.
"""

from __future__ import annotations

import numpy as np
import pytest

from lucen.source import passes_stage1_gate, summed_displacement, synchrony_sweep


def test_source_term_non_decreasing_in_synchrony(d_single, voxel, synchrony_grid):
    """More synchrony never reduces the source displacement (monotone up)."""
    sweep = synchrony_sweep(d_single, voxel, synchrony_grid)
    vals = np.array([s.value_m for s in sweep])
    assert np.all(np.diff(vals) >= 0.0)


def test_stage1_gate_monotone_in_floor(d_single, voxel, synchrony_grid):
    """A lower floor is never harder to clear than a higher one (monotone gate)."""
    sweep = synchrony_sweep(d_single, voxel, synchrony_grid)
    peak = max(s.value_m for s in sweep)
    # Floor just below the peak -> passes; floor far above -> fails.
    assert passes_stage1_gate(sweep, unaberrated_floor_m=peak, reach_orders=0.0)
    assert not passes_stage1_gate(
        sweep, unaberrated_floor_m=peak * 1e6, reach_orders=0.0
    )


def test_more_neurons_never_reduce_coherent_sum(d_single, voxel):
    """Sanity: larger N never lowers the fully-coherent sum."""
    from dataclasses import replace

    bigger = replace(voxel, neuron_count=voxel.neuron_count * 2)
    small = summed_displacement(d_single, voxel, 1.0).value_m
    large = summed_displacement(d_single, bigger, 1.0).value_m
    assert large > small


@pytest.mark.xfail(reason="Module 2 propagation not yet implemented", strict=True)
def test_contrast_non_increasing_in_skull_attenuation():
    """Higher skull attenuation must not raise contrast. Enable with Module 2."""
    from lucen.propagation import propagate_round_trip

    propagate_round_trip(  # raises NotImplementedError -> xfail
        source=None,
        geom=None,
        skull=None,
        correction=None,
        array_geometry=None,
        freq_hz=2e6,
    )
