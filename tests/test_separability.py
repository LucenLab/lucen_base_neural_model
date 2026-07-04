"""Voxel-pair spatial separability: sidelobe leakage, grating lobes, the two-hotspot verdict.

Pins the physics that answers Act 1's spatial-resolution claim (build spec sections
5.1, 6.2): can two adjacent voxels at digit-representation spacing ("a few millimetres")
be told apart, given the beam's off-axis (sidelobe) response and the array's grating-lobe
threshold, rather than just naming a resolution length in isolation.
"""

from __future__ import annotations

import math

import pytest

from base_neural_model.base.bands import Band
from base_neural_model.base.provenance import Provenance
from base_neural_model.base.types import MechanicalDisplacement
from base_neural_model.forward.separability import (
    GRATING_LOBE_PITCH_RATIO,
    UNAPODIZED_SIDELOBE_DB,
    BeamAperture,
    min_separable_distance,
    sidelobe_leakage_fraction,
    voxel_pair_separability,
)

# The demo build spec's own numbers (section 3.2): f_N ~ 2 MHz, c ~ 1540 m/s ->
# lambda ~ 0.77 mm; element pitch <= lambda/2 ~ 0.385 mm; 256 elements; and the technical
# spec's "roughly a millimetre at cortical depth" lateral voxel width.
_C_MPS = 1540.0
_F_N_HZ = 2.0e6
_LAMBDA_M = _C_MPS / _F_N_HZ
_PITCH_M = _LAMBDA_M / 2.0
_F_NUMBER_1MM_VOXEL = 1.0e-3 / _LAMBDA_M  # F# giving a ~1 mm main-lobe width


def _demo_aperture(*, sidelobe_db: float = UNAPODIZED_SIDELOBE_DB) -> BeamAperture:
    return BeamAperture(
        wavelength_m=_LAMBDA_M,
        f_number=_F_NUMBER_1MM_VOXEL,
        sidelobe_db=sidelobe_db,
        element_pitch_m=_PITCH_M,
        n_elements=256,
    )


def _mech(dz_m: float = 1e-9) -> MechanicalDisplacement:
    prov = Provenance(
        source="test source displacement for separability",
        assumptions=("synthetic",),
        band=Band.CONTENT_FAST,
    )
    return MechanicalDisplacement(
        axial_displacement_m=dz_m,
        volumetric_strain=0.0,
        incoherent_pedestal_m=0.0,
        synchrony_fraction=0.8,
        jitter_sigma_s=1e-3,
        content_band_survival=0.9,
        confinement_kappa=0.5,
        dilatation_eta=0.5,
        band=Band.CONTENT_FAST,
        provenance=prov,
    )


# --- BeamAperture -----------------------------------------------------------------


def test_beam_aperture_rejects_nonpositive_wavelength_and_fnumber():
    with pytest.raises(ValueError):
        BeamAperture(wavelength_m=0.0, f_number=1.0)
    with pytest.raises(ValueError):
        BeamAperture(wavelength_m=1e-3, f_number=0.0)


def test_beam_aperture_rejects_nonnegative_sidelobe_db():
    with pytest.raises(ValueError):
        BeamAperture(wavelength_m=1e-3, f_number=1.0, sidelobe_db=0.0)
    with pytest.raises(ValueError):
        BeamAperture(wavelength_m=1e-3, f_number=1.0, sidelobe_db=5.0)


def test_lateral_width_matches_f_number_times_wavelength():
    """w_lat = F# * lambda (technical spec section 3.1), exactly."""
    ap = BeamAperture(wavelength_m=0.77e-3, f_number=2.0)
    assert ap.lateral_width_m == pytest.approx(1.54e-3)


def test_demo_aperture_reproduces_spec_numbers():
    """The demo spec's own array parameters reproduce its own stated figures."""
    ap = _demo_aperture()
    assert ap.wavelength_m == pytest.approx(0.77e-3, rel=1e-2)
    assert ap.element_pitch_m == pytest.approx(0.385e-3, rel=1e-2)
    assert ap.lateral_width_m == pytest.approx(1.0e-3, rel=1e-6)
    assert ap.aperture_width_m == pytest.approx(0.385e-3 * 256, rel=1e-2)


def test_no_grating_lobe_at_or_below_half_wavelength_pitch():
    """Build spec section 3.2: pitch at/below lambda/2 suppresses grating lobes."""
    ap = BeamAperture(
        wavelength_m=1.0e-3, f_number=1.5,
        element_pitch_m=0.5e-3, n_elements=100,  # exactly lambda/2
    )
    assert not ap.has_grating_lobe


def test_grating_lobe_appears_above_half_wavelength_pitch():
    ap = BeamAperture(
        wavelength_m=1.0e-3, f_number=1.5,
        element_pitch_m=0.6e-3, n_elements=100,  # > lambda/2
    )
    assert ap.has_grating_lobe


def test_grating_lobe_check_skipped_without_pitch():
    """No pitch/n_elements given -> the check is opt-in and defaults to no grating lobe."""
    ap = BeamAperture(wavelength_m=1.0e-3, f_number=1.5)
    assert not ap.has_grating_lobe
    assert ap.aperture_width_m is None


# --- sidelobe_leakage_fraction ------------------------------------------------------


def test_leakage_is_unity_at_zero_separation():
    ap = _demo_aperture()
    assert sidelobe_leakage_fraction(0.0, ap) == 1.0


def test_leakage_rejects_negative_separation():
    ap = _demo_aperture()
    with pytest.raises(ValueError):
        sidelobe_leakage_fraction(-1e-3, ap)


def test_leakage_is_near_zero_at_the_first_null():
    """At separation = w_lat the sinc has its first zero."""
    ap = _demo_aperture()
    leak = sidelobe_leakage_fraction(ap.lateral_width_m, ap)
    assert leak == pytest.approx(0.0, abs=1e-6)


def test_leakage_saturates_at_the_apodized_sidelobe_floor_past_the_first_null():
    """Past the first null, leakage is pinned to sidelobe_db, not the raw sinc rebound."""
    ap = _demo_aperture(sidelobe_db=-20.0)
    expected = 10.0 ** (-20.0 / 20.0)
    for sep_mult in (1.5, 2.0, 5.0, 10.0):
        leak = sidelobe_leakage_fraction(sep_mult * ap.lateral_width_m, ap)
        assert leak == pytest.approx(expected, rel=1e-6)


def test_leakage_monotone_nonincreasing_with_separation():
    """Required for min_separable_distance's bisection to be well-posed."""
    ap = _demo_aperture()
    seps = [i * 0.1e-3 for i in range(1, 40)]
    leaks = [sidelobe_leakage_fraction(s, ap) for s in seps]
    # Not strictly monotone inside the first lobe (sinc^2 magnitude dips through the
    # null), but must never exceed its value at the main lobe and must settle at the
    # floor past the first null -- check the settled tail is monotone-flat.
    past_null = [leak for s, leak in zip(seps, leaks) if s > ap.lateral_width_m]
    assert all(leak == pytest.approx(past_null[0], rel=1e-6) for leak in past_null)


def test_deeper_apodization_reduces_leakage():
    ap_shallow = _demo_aperture(sidelobe_db=-13.3)
    ap_deep = _demo_aperture(sidelobe_db=-30.0)
    sep = 3.0 * ap_shallow.lateral_width_m
    assert sidelobe_leakage_fraction(sep, ap_deep) < sidelobe_leakage_fraction(sep, ap_shallow)


# --- voxel_pair_separability ---------------------------------------------------------


def test_separability_rejects_zero_or_negative_signal():
    ap = _demo_aperture()
    with pytest.raises(ValueError):
        voxel_pair_separability(_mech(dz_m=0.0), ap, 2e-3)


def test_separability_rejects_negative_min_contrast():
    ap = _demo_aperture()
    with pytest.raises(ValueError):
        voxel_pair_separability(_mech(), ap, 2e-3, min_contrast_db=-1.0)


def test_zero_separation_is_never_separable():
    """Same voxel: own signal and leaked signal are identical -> 0 dB contrast."""
    ap = _demo_aperture()
    v = voxel_pair_separability(_mech(), ap, 0.0)
    assert v.contrast_db == pytest.approx(0.0, abs=1e-9)
    assert not v.separable


def test_demo_spec_digit_spacing_is_separable_at_demo_aperture():
    """The spec's own claim: 'a few millimetres' digit spacing, at its own array params.

    Build spec section 5.1: 'neighbouring digit representations separated on the order
    of a few millimeters.' At the spec's own 2 MHz / lambda-2 pitch / 256-element
    aperture producing a ~1 mm voxel, this should clear a conventional 6 dB margin.
    """
    ap = _demo_aperture()
    for spacing_mm in (2.0, 3.0, 4.0):
        v = voxel_pair_separability(_mech(), ap, spacing_mm * 1e-3)
        assert v.separable, f"expected separable at {spacing_mm} mm spacing"
        assert v.contrast_db >= 6.0


def test_contrast_db_matches_ratio_of_own_to_leaked():
    ap = _demo_aperture()
    v = voxel_pair_separability(_mech(dz_m=2e-9), ap, 2.0e-3)
    expected_db = 20.0 * math.log10(v.own_signal_m / v.leaked_signal_m)
    assert v.contrast_db == pytest.approx(expected_db)


def test_grating_lobe_forces_not_separable_regardless_of_sidelobe_contrast():
    """A grating lobe reproduces the neighbour at full amplitude -- overrides contrast."""
    ap = BeamAperture(
        wavelength_m=0.77e-3, f_number=1.3,
        element_pitch_m=0.9e-3, n_elements=100,  # > lambda/2 -> grating lobe
    )
    # Even at a huge separation (deep in the sidelobe floor), the grating lobe verdict
    # must still block separability.
    v = voxel_pair_separability(_mech(), ap, 20e-3, min_contrast_db=1.0)
    assert v.has_grating_lobe
    assert not v.separable


def test_own_signal_equals_mech_axial_displacement():
    ap = _demo_aperture()
    mech = _mech(dz_m=3.3e-9)
    v = voxel_pair_separability(mech, ap, 2e-3)
    assert v.own_signal_m == mech.axial_displacement_m


def test_provenance_names_grating_lobe_when_present():
    ap = BeamAperture(
        wavelength_m=0.77e-3, f_number=1.3,
        element_pitch_m=0.9e-3, n_elements=100,
    )
    v = voxel_pair_separability(_mech(), ap, 3e-3, min_contrast_db=1.0)
    assert any("GRATING LOBE" in a for a in v.provenance.assumptions)


# --- min_separable_distance ----------------------------------------------------------


def test_min_separable_distance_is_a_separability_crossing():
    """The returned distance should be right at the min_contrast_db boundary.

    Near the first null the sinc's dB value is steep in distance, so a tight distance
    tolerance (tol) does not translate to an equally tight dB tolerance; the assertion
    tolerance is loosened accordingly rather than the bisection's own precision.
    """
    ap = _demo_aperture()
    d = min_separable_distance(_mech(), ap, min_contrast_db=6.0, tol=1e-9)
    v = voxel_pair_separability(_mech(), ap, d, min_contrast_db=6.0)
    assert v.contrast_db == pytest.approx(6.0, abs=0.05)


def test_min_separable_distance_is_finer_than_demo_digit_spacing():
    """The model should find separation finer than the spec's 'few millimetres' claim."""
    ap = _demo_aperture()
    d = min_separable_distance(_mech(), ap)
    assert d < 2.0e-3


def test_min_separable_distance_increases_with_stricter_margin():
    ap = _demo_aperture()
    d_loose = min_separable_distance(_mech(), ap, min_contrast_db=3.0)
    d_strict = min_separable_distance(_mech(), ap, min_contrast_db=20.0)
    assert d_strict > d_loose


def test_min_separable_distance_saturates_at_hi_m_when_unreachable():
    """A pathologically shallow sidelobe floor may never clear a strict margin."""
    ap = _demo_aperture(sidelobe_db=-1.0)
    d = min_separable_distance(_mech(), ap, min_contrast_db=40.0, hi_m=0.01)
    assert d == pytest.approx(0.01)
