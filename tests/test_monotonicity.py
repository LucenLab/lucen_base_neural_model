"""Monotonicity sanity (no sign errors in the transduction chain).

The tissue displacement is non-decreasing in synchrony, the amplitude gate is
monotone in its floor, and a larger voxel has a lower incoherent pedestal. A
violation means a sign error somewhere in the chain.
"""

from __future__ import annotations

import numpy as np
import pytest

from base_neural_model import (
    displacement_sweep,
    mechanical_displacement,
    passes_stage1_gate,
)


def test_source_term_non_decreasing_in_synchrony(
    d_single, voxel, source_params, synchrony_grid
):
    """More synchrony never reduces the source displacement (monotone up)."""
    sweep = displacement_sweep(d_single, voxel, source_params, synchrony_grid)
    vals = np.array([s.axial_displacement_m for s in sweep])
    assert np.all(np.diff(vals) >= 0.0)


def test_stage1_gate_monotone_in_floor(d_single, voxel, source_params, synchrony_grid):
    """A lower floor is never harder to clear than a higher one (monotone gate)."""
    sweep = displacement_sweep(d_single, voxel, source_params, synchrony_grid)
    peak = max(s.axial_displacement_m for s in sweep)
    # Floor just below the peak -> passes; floor far above -> fails.
    assert passes_stage1_gate(sweep, unaberrated_floor_m=peak, reach_orders=0.0)
    assert not passes_stage1_gate(
        sweep, unaberrated_floor_m=peak * 1e6, reach_orders=0.0
    )


def test_more_neurons_lower_the_incoherent_pedestal(d_single, voxel, source_params):
    """Corrected meaning: the coherent strain is N-independent, but the incoherent
    pedestal falls as 1/sqrt(N), so a bigger voxel has a *cleaner* source (the
    pedestal is noise, not signal - source physics doc, section 2.2/4)."""
    from dataclasses import replace

    bigger = replace(voxel, neuron_count=voxel.neuron_count * 4)
    small = mechanical_displacement(d_single, voxel, source_params, 0.0)
    large = mechanical_displacement(d_single, bigger, source_params, 0.0)
    # Coherent term identical (N-independent); pedestal halves for 4x neurons.
    assert large.incoherent_pedestal_m < small.incoherent_pedestal_m
    assert large.incoherent_pedestal_m == pytest.approx(
        small.incoherent_pedestal_m / 2.0, rel=1e-9
    )
