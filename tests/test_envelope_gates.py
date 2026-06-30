"""Worked-number envelope and the three gates (source physics doc, section 5).

The honest output of Module 1 is not one number but three columns. This test
pins the section-5 table - pessimistic ~0.03 nm, central ~2.5 nm, optimistic
~90 nm for the coherent axial displacement - so a regression in any chain factor
moves a column and is caught. It then exercises the three kill gates against the
~1-10 nm unaberrated floor.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from base_neural_model import (
    displacement_sweep,
    mechanical_displacement,
    passes_content_survival_gate,
    passes_dilatation_gate,
    passes_stage1_gate,
)
from base_neural_model.base.bands import Band
from base_neural_model.base.provenance import Provenance
from base_neural_model.base.types import (
    MechanicsParams,
    NeuronDisplacement,
    VoxelGeometry,
)

# Range gate L = 300 um across all three columns (section 5 table).
_GATE_LEN_M = 300e-6

# (params, synchrony, expected axial displacement in metres) per section-5 column.
_PESSIMISTIC = (
    MechanicsParams(
        membrane_disp_m=1.0e-9,
        cell_radius_m=10e-6,
        cell_volume_fraction=0.10,
        confinement_kappa=1.0 / 3.0,
        dilatation_eta=0.1,
        jitter_sigma_s=3.0e-3,
        content_freq_hz=100.0,
    ),
    0.1,
    0.03e-9,
)
_CENTRAL = (
    MechanicsParams(
        membrane_disp_m=1.5e-9,
        cell_radius_m=8e-6,
        cell_volume_fraction=0.15,
        confinement_kappa=0.5,
        dilatation_eta=0.5,
        jitter_sigma_s=1.0e-3,
        content_freq_hz=100.0,
    ),
    0.4,
    2.53e-9,
)
_OPTIMISTIC = (
    MechanicsParams(
        membrane_disp_m=3.0e-9,
        cell_radius_m=6e-6,
        cell_volume_fraction=0.25,
        confinement_kappa=1.0,
        dilatation_eta=1.0,
        jitter_sigma_s=0.5e-3,
        content_freq_hz=100.0,
    ),
    0.8,
    90.0e-9,
)


def _neuron_for(params: MechanicsParams) -> NeuronDisplacement:
    """A cited-constant neuron pinned to the column's membrane displacement."""
    prov = Provenance(
        source="section-5 envelope column",
        assumptions=("cited Delta r for this column",),
        band=Band.CONTENT_FAST,
    )
    return NeuronDisplacement(
        value_m=params.membrane_disp_m, band=Band.CONTENT_FAST, provenance=prov
    )


def _voxel() -> VoxelGeometry:
    return VoxelGeometry(
        extent_axial_m=_GATE_LEN_M,
        extent_lateral_m=1e-3,
        neuron_count=20_000,
        depth_m=2e-2,
    )


@pytest.mark.parametrize(
    "params, s, expected_m",
    [_PESSIMISTIC, _CENTRAL, _OPTIMISTIC],
    ids=["pessimistic", "central", "optimistic"],
)
def test_section5_envelope_columns(params, s, expected_m):
    """Reproduce the section-5 worked numbers to two significant figures."""
    result = mechanical_displacement(_neuron_for(params), _voxel(), params, s)
    assert result.axial_displacement_m == pytest.approx(expected_m, rel=0.02)


def test_gate1_passes_optimistic_fails_pessimistic():
    """Gate 1 (amplitude) against a 1 nm unaberrated floor."""
    floor_m = 1e-9
    p_params, p_s, _ = _PESSIMISTIC
    o_params, o_s, _ = _OPTIMISTIC

    pess_sweep = [mechanical_displacement(_neuron_for(p_params), _voxel(), p_params, p_s)]
    opt_sweep = [mechanical_displacement(_neuron_for(o_params), _voxel(), o_params, o_s)]

    # Optimistic (~90 nm) clears even with no reach allowance.
    assert passes_stage1_gate(opt_sweep, unaberrated_floor_m=floor_m, reach_orders=0.0)
    # Pessimistic (~0.03 nm) is ~1.5 orders under a 1 nm floor: dies with a tight bar.
    assert not passes_stage1_gate(
        pess_sweep, unaberrated_floor_m=floor_m, reach_orders=1.0
    )


def test_gate3_dilatation_visibility():
    """Gate 3 (dilatation) tracks eta: central/optimistic pass, eta->0 fails."""
    o_params, o_s, _ = _OPTIMISTIC
    opt_sweep = [mechanical_displacement(_neuron_for(o_params), _voxel(), o_params, o_s)]
    assert passes_dilatation_gate(opt_sweep)

    dead = replace(o_params, dilatation_eta=0.0)
    dead_sweep = [mechanical_displacement(_neuron_for(dead), _voxel(), dead, o_s)]
    assert not passes_dilatation_gate(dead_sweep)


def test_gate2_content_survival_tracks_jitter():
    """Gate 2 (content survival): low jitter passes, heavy jitter low-passes out."""
    o_params, _, _ = _OPTIMISTIC
    grid = np.linspace(0.0, 1.0, 21)
    estimate_floor_m = 1e-9

    low_jitter = replace(o_params, jitter_sigma_s=0.2e-3)  # survival ~ 1
    good_sweep = displacement_sweep(_neuron_for(low_jitter), _voxel(), low_jitter, grid)
    assert passes_content_survival_gate(good_sweep, estimate_floor_m=estimate_floor_m)

    # Heavy jitter at a high content corner crushes survival -> content lost.
    heavy = replace(o_params, jitter_sigma_s=10e-3, content_freq_hz=200.0)
    bad_sweep = displacement_sweep(_neuron_for(heavy), _voxel(), heavy, grid)
    assert not passes_content_survival_gate(bad_sweep, estimate_floor_m=estimate_floor_m)
