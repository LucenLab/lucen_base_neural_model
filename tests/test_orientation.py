"""Mechanics - the directional (deviatoric / orientation) channel.

Temporal synchrony (cells firing at the same time) and directional coherence (their
3-D displacement axes pointing the same way) are independent. The directional channel
adds an axial term that survives only with orientation alignment, and it must reduce
exactly to the isotropic volume-change model when it is off. These tests pin the
tensor projection math and that reduction.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from base_neural_model.base.types import MechanicsParams
from base_neural_model.mechanics.orientation import (
    axial_orientation_factor,
    deviatoric_eshelby_response,
    directional_axial_strain,
    legendre_p2,
)
from base_neural_model.mechanics.transduction import mechanical_displacement


@pytest.fixture
def base_params(d_single):
    return replace(MechanicsParams.central(), membrane_disp_m=d_single.value_m)


# --- the orientation projection factor g(Q, mu) = (2/3) Q P2(mu) ----------------


def test_p2_values():
    assert legendre_p2(1.0) == pytest.approx(1.0)
    assert legendre_p2(0.0) == pytest.approx(-0.5)


def test_random_orientation_kills_directional_term():
    """Q = 0 (random axes) gives zero axial factor for ANY director - the sum cancels,
    exactly the isotropic model's assumption."""
    for mu in (-1.0, -0.3, 0.0, 0.5, 1.0):
        assert axial_orientation_factor(0.0, mu) == 0.0


def test_aligned_to_beam_is_maximal():
    """Q = 1, mu = 1 (axes aligned along the beam) gives the max factor +2/3."""
    assert axial_orientation_factor(1.0, 1.0) == pytest.approx(2.0 / 3.0)


def test_aligned_across_beam_is_negative():
    """Q = 1, mu = 0 (axes aligned ACROSS the beam) gives -1/3: sideways expansion is
    a negative axial deviatoric strain."""
    assert axial_orientation_factor(1.0, 0.0) == pytest.approx(-1.0 / 3.0)


def test_orientation_factor_monotone_in_order_parameter():
    vals = [axial_orientation_factor(q, 1.0) for q in (0.0, 0.25, 0.5, 1.0)]
    assert all(b > a for a, b in zip(vals, vals[1:], strict=False))


def test_orientation_factor_rejects_out_of_range():
    with pytest.raises(ValueError):
        axial_orientation_factor(1.5, 1.0)
    with pytest.raises(ValueError):
        axial_orientation_factor(0.5, 2.0)


# --- the deviatoric Eshelby response S_1111 - S_1122 = (8 - 10nu)/[15(1-nu)] -----


def test_deviatoric_response_values():
    assert deviatoric_eshelby_response(0.0) == pytest.approx(8.0 / 15.0)
    # At the incompressible limit it is 0.4, NOT zero (unlike the volume trace).
    assert deviatoric_eshelby_response(0.49) == pytest.approx(
        (8 - 10 * 0.49) / (15 * (1 - 0.49))
    )
    assert deviatoric_eshelby_response(0.49) > 0.39


def test_directional_strain_zero_when_anisotropy_zero():
    g = directional_axial_strain(
        1e-3, anisotropy=0.0, order_parameter=1.0, director_projection=1.0, nu=0.2
    )
    assert g == 0.0


def test_directional_strain_zero_when_random():
    g = directional_axial_strain(
        1e-3, anisotropy=1.0, order_parameter=0.0, director_projection=1.0, nu=0.2
    )
    assert g == 0.0


# --- the chain: directional channel composes with the isotropic one -------------


def test_isotropic_default_reduces_to_volume_model(d_single, base_params, voxel):
    """With the default isotropic params the directional part is zero and the total
    equals the pure volume-change axial term (backward compatibility)."""
    out = mechanical_displacement(d_single, voxel, base_params, 1.0)
    assert out.directional_axial_m == 0.0
    assert out.value_m == pytest.approx(out.isotropic_axial_m)


def test_alignment_adds_signal(d_single, base_params, voxel):
    """Axes aligned to the beam add a positive directional term above the isotropic."""
    iso = mechanical_displacement(d_single, voxel, base_params, 1.0)
    aligned = replace(
        base_params, anisotropy=1.0, orientation_coherence=1.0,
        mean_axis_projection=1.0,
    )
    a = mechanical_displacement(d_single, voxel, aligned, 1.0)
    assert a.directional_axial_m > 0.0
    assert a.value_m > iso.value_m
    assert a.value_m == pytest.approx(a.isotropic_axial_m + a.directional_axial_m)


def test_cross_beam_alignment_reduces_signal(d_single, base_params, voxel):
    """Axes aligned ACROSS the beam give a negative directional term (sideways
    expansion), reducing the net axial magnitude below the isotropic value."""
    iso = mechanical_displacement(d_single, voxel, base_params, 1.0)
    across = replace(
        base_params, anisotropy=1.0, orientation_coherence=1.0,
        mean_axis_projection=0.0,
    )
    a = mechanical_displacement(d_single, voxel, across, 1.0)
    assert a.directional_axial_m < 0.0
    assert a.value_m < iso.value_m


def test_directional_term_scales_with_order_parameter(d_single, base_params, voxel):
    """More orientation coherence (higher Q) -> larger directional term toward beam."""
    dirs = []
    for q in (0.0, 0.3, 0.6, 1.0):
        p = replace(
            base_params, anisotropy=1.0, orientation_coherence=q,
            mean_axis_projection=1.0,
        )
        md = mechanical_displacement(d_single, voxel, p, 1.0)
        dirs.append(md.directional_axial_m)
    assert all(b > a for a, b in zip(dirs, dirs[1:], strict=False))


def test_orientation_provenance_appended_when_active(d_single, base_params, voxel):
    p = replace(base_params, anisotropy=0.5, orientation_coherence=0.5)
    out = mechanical_displacement(d_single, voxel, p, 1.0)
    assert any("deviatoric" in a for a in out.provenance.assumptions)
