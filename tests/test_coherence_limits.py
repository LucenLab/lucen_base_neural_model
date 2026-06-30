"""Invariant / spec section 7.3 - coherence limits (corrected form).

The placeholder asserted ``N*d_1`` at ``s=1`` and ``sqrt(N)*d_1`` at ``s=0``.
That summed membrane displacements as collinear translations - a category error
(source physics doc, section 1). The corrected invariant (section 2.2 / 6) is a
*strain*:

* ``s=1``: coherent volumetric strain ``eps_V = f_cell * (3*Delta r / r)``;
* ``s=0``: the coherent term vanishes; an incoherent ``1/sqrt(N)`` strain
  pedestal remains, and it is noise, not signal.

A band-survival invariant is added: the content-band quantity is computed under
an explicit jitter ``sigma_t``, so the slow envelope can never be substituted
for content.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from base_neural_model import displacement_sweep, mechanical_displacement


def _frac_volume_change(params):
    return 3.0 * params.membrane_disp_m / params.cell_radius_m


def test_fully_coherent_limit_is_fcell_times_volume_change(
    d_single, voxel, source_params
):
    """s=1 coherent strain = f_cell * (3*Delta r / r) (replaces N*d_1)."""
    result = mechanical_displacement(d_single, voxel, source_params, 1.0)
    expected = source_params.cell_volume_fraction * _frac_volume_change(source_params)
    assert result.volumetric_strain == pytest.approx(expected, rel=1e-12)


def test_fully_incoherent_limit_is_sqrtN_pedestal(d_single, voxel, source_params):
    """s=0: coherent term is 0; pedestal = f_cell * (1/sqrt(N)) * (3*Delta r/r)."""
    n = voxel.neuron_count
    result = mechanical_displacement(d_single, voxel, source_params, 0.0)
    assert result.volumetric_strain == pytest.approx(0.0, abs=1e-30)
    expected_incoh_strain = (
        source_params.cell_volume_fraction
        * (1.0 / math.sqrt(n))
        * _frac_volume_change(source_params)
    )
    expected_pedestal_m = (
        source_params.dilatation_eta
        * source_params.confinement_kappa
        * voxel.extent_axial_m
        * expected_incoh_strain
    )
    assert result.incoherent_pedestal_m == pytest.approx(expected_pedestal_m, rel=1e-12)


def test_axial_endpoint_is_eta_kappa_L_strain(d_single, voxel, source_params):
    """Delta z at s=1 = eta * kappa * L * f_cell * (3*Delta r / r) (section 2.3)."""
    result = mechanical_displacement(d_single, voxel, source_params, 1.0)
    expected = (
        source_params.dilatation_eta
        * source_params.confinement_kappa
        * voxel.extent_axial_m
        * source_params.cell_volume_fraction
        * _frac_volume_change(source_params)
    )
    assert result.axial_displacement_m == pytest.approx(expected, rel=1e-12)


def test_coherent_ceiling_exceeds_incoherent_floor(d_single, voxel, source_params):
    """Coherent s=1 axial displacement beats the s=0 incoherent pedestal."""
    hi = mechanical_displacement(d_single, voxel, source_params, 1.0)
    lo = mechanical_displacement(d_single, voxel, source_params, 0.0)
    assert hi.axial_displacement_m > lo.incoherent_pedestal_m


def test_sweep_is_monotonic_non_decreasing(
    d_single, voxel, source_params, synchrony_grid
):
    """Synchrony is the swept variable (Invariant 4); the source rises with it."""
    sweep = displacement_sweep(d_single, voxel, source_params, synchrony_grid)
    vals = np.array([s.axial_displacement_m for s in sweep])
    assert np.all(np.diff(vals) >= 0.0)


def test_sweep_endpoints_match_closed_form(
    d_single, voxel, source_params, synchrony_grid
):
    sweep = displacement_sweep(d_single, voxel, source_params, synchrony_grid)
    assert sweep[0].synchrony_fraction == pytest.approx(0.0)
    assert sweep[-1].synchrony_fraction == pytest.approx(1.0)
    assert sweep[0].axial_displacement_m == pytest.approx(0.0, abs=1e-30)
    expected_top = (
        source_params.dilatation_eta
        * source_params.confinement_kappa
        * voxel.extent_axial_m
        * source_params.cell_volume_fraction
        * _frac_volume_change(source_params)
    )
    assert sweep[-1].axial_displacement_m == pytest.approx(expected_top, rel=1e-12)


def test_content_band_survival_under_explicit_jitter(d_single, voxel, source_params):
    """Band-survival invariant (section 6): content quantity carries s(f_c)."""
    result = mechanical_displacement(d_single, voxel, source_params, 0.5)
    expected = math.exp(
        -2.0
        * math.pi**2
        * source_params.content_freq_hz**2
        * source_params.jitter_sigma_s**2
    )
    assert result.content_band_survival == pytest.approx(expected, rel=1e-12)
    assert 0.0 < result.content_band_survival <= 1.0
    assert result.jitter_sigma_s == source_params.jitter_sigma_s


def test_synchrony_fraction_out_of_range_raises(d_single, voxel, source_params):
    with pytest.raises(ValueError):
        mechanical_displacement(d_single, voxel, source_params, synchrony_fraction=-0.01)
    with pytest.raises(ValueError):
        mechanical_displacement(d_single, voxel, source_params, synchrony_fraction=1.01)
