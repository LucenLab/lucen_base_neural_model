"""Inverse design for the directional channel: minimum beta* and Q*.

These solvers invert the gates for the directional unknowns - the cell anisotropy
beta and the orientation order Q - the directional analogue of min_detectable_eta.
The tests pin: the bisection lands at the pass boundary; the threshold is monotone
in the floor; the well-posedness regimes (isotropic / directional / infeasible /
infeasible_direction) are reported correctly; and raising the directional term across
the beam (g < 0) cannot rescue.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from base_neural_model.base.types import MechanicsParams
from base_neural_model.mechanics.transduction import mechanical_displacement
from base_neural_model.model.gates import passes_content_survival_gate
from base_neural_model.model.min_directional import (
    min_anisotropy,
    min_orientation_coherence,
)


@pytest.fixture
def d_single():
    """Fixed test-scale single-neuron displacement for the directional MACHINERY tests.

    Pinned to the pre-revision 2 nm and decoupled from the honest cited constant
    (~0.4 nm), so the bisection / boundary / monotonicity assertions run in a
    non-degenerate feasible window. The honest Delta r drives the demo verdict, not this
    machinery test.
    """
    from dataclasses import replace

    from base_neural_model import get_single_neuron_displacement

    return replace(get_single_neuron_displacement(), value_m=2e-9)


@pytest.fixture
def weak_aligned(d_single):
    """A weak (low-eta) source whose axes are aligned TO the beam (g > 0), so the
    directional channel can help and the iso signal alone may fail."""
    return replace(
        MechanicsParams.central(),
        membrane_disp_m=d_single.value_m,
        dilatation_eta=0.06,
        orientation_coherence=0.7,   # for the beta solver
        anisotropy=0.6,              # for the Q solver
        mean_axis_projection=1.0,    # aligned to beam -> g > 0
    )


# --- min_anisotropy (beta*) -----------------------------------------------------


def test_beta_star_found_when_directional_helps(d_single, voxel, weak_aligned):
    p = replace(weak_aligned, anisotropy=0.0)  # solver sweeps anisotropy
    res = min_anisotropy(d_single, voxel, p, 1.0, floor_m=1e-9)
    assert res.binding_constraint == "directional"
    assert res.value_star is not None
    assert 0.0 < res.value_star <= 1.0


def test_beta_star_sits_at_the_pass_boundary(d_single, voxel, weak_aligned):
    """At beta*, the surviving displacement equals the estimate floor (Gate-2)."""
    res = min_anisotropy(d_single, voxel, weak_aligned, 1.0, floor_m=1e-9)
    at_star = replace(weak_aligned, anisotropy=res.value_star)
    md = mechanical_displacement(d_single, voxel, at_star, 1.0)
    surviving = md.value_m * md.content_band_survival
    assert surviving == pytest.approx(1e-9, rel=1e-3)
    assert passes_content_survival_gate([md], estimate_floor_m=1e-9)


def test_beta_star_rises_with_floor(d_single, voxel, weak_aligned):
    """A harder floor needs more anisotropy (both above the iso baseline ~0.83 nm)."""
    low = min_anisotropy(d_single, voxel, weak_aligned, 1.0, floor_m=0.95e-9)
    high = min_anisotropy(d_single, voxel, weak_aligned, 1.0, floor_m=1.15e-9)
    assert low.binding_constraint == high.binding_constraint == "directional"
    assert high.value_star > low.value_star


def test_beta_isotropic_when_signal_already_passes(d_single, voxel):
    """A strong source clears the gates with no directional help (beta* = 0)."""
    strong = replace(
        MechanicsParams.central(), membrane_disp_m=d_single.value_m,
        dilatation_eta=1.0, orientation_coherence=0.7, mean_axis_projection=1.0,
    )
    res = min_anisotropy(d_single, voxel, strong, 1.0, floor_m=1e-9)
    assert res.binding_constraint == "isotropic"
    assert res.value_star == 0.0


def test_beta_infeasible_when_floor_too_high(d_single, voxel, weak_aligned):
    res = min_anisotropy(d_single, voxel, weak_aligned, 1.0, floor_m=5e-8)
    assert res.binding_constraint == "infeasible"
    assert res.value_star is None


def test_beta_infeasible_direction_across_beam(d_single, voxel, weak_aligned):
    """Axes across the beam (g < 0): raising beta subtracts signal, cannot rescue."""
    across = replace(weak_aligned, mean_axis_projection=0.0)
    res = min_anisotropy(d_single, voxel, across, 1.0, floor_m=8e-9)
    assert res.binding_constraint == "infeasible_direction"
    assert res.value_star is None
    assert res.feasible is False


# --- min_orientation_coherence (Q*) ---------------------------------------------


def test_q_star_found_when_directional_helps(d_single, voxel, weak_aligned):
    p = replace(weak_aligned, orientation_coherence=0.0)  # solver sweeps Q
    res = min_orientation_coherence(d_single, voxel, p, 1.0, floor_m=1e-9)
    assert res.binding_constraint == "directional"
    assert 0.0 < res.value_star <= 1.0


def test_q_star_sits_at_the_pass_boundary(d_single, voxel, weak_aligned):
    res = min_orientation_coherence(d_single, voxel, weak_aligned, 1.0, floor_m=1e-9)
    at_star = replace(weak_aligned, orientation_coherence=res.value_star)
    md = mechanical_displacement(d_single, voxel, at_star, 1.0)
    surviving = md.value_m * md.content_band_survival
    assert surviving == pytest.approx(1e-9, rel=1e-3)


def test_q_star_rises_with_floor(d_single, voxel, weak_aligned):
    # Both floors within the Q-reachable band (iso ~0.83 nm to Q=1 ~1.16 nm at beta=0.6).
    low = min_orientation_coherence(d_single, voxel, weak_aligned, 1.0, floor_m=0.95e-9)
    high = min_orientation_coherence(d_single, voxel, weak_aligned, 1.0, floor_m=1.10e-9)
    assert low.value_star is not None and high.value_star is not None
    assert high.value_star > low.value_star


def test_q_infeasible_direction_across_beam(d_single, voxel, weak_aligned):
    across = replace(weak_aligned, mean_axis_projection=0.0)
    res = min_orientation_coherence(d_single, voxel, across, 1.0, floor_m=8e-9)
    assert res.binding_constraint == "infeasible_direction"
    assert res.value_star is None


# --- shared contract checks -----------------------------------------------------


def test_threshold_carries_provenance(d_single, voxel, weak_aligned):
    res = min_anisotropy(d_single, voxel, weak_aligned, 1.0, floor_m=1e-9)
    assert res.provenance.source
    assert res.quantity == "anisotropy"
    assert any("directional" in a for a in res.provenance.assumptions)


def test_rejects_bad_synchrony(d_single, voxel, weak_aligned):
    with pytest.raises(ValueError):
        min_anisotropy(d_single, voxel, weak_aligned, 1.5, floor_m=1e-9)
