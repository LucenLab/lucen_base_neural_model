"""Acoustic detection layer: integration gain, the phase floor, and the budget.

Pins the two acoustic factors the Gate-A / Stage-1 verdict now rests on -- within-epoch
coherent integration (sqrt(N_ens)) and the Walker-Trahey through-skull displacement
floor -- plus the budget that composes them onto a source displacement and names the
binding denominator (echo SNR vs residual clutter).
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from base_neural_model.base.bands import Band
from base_neural_model.base.provenance import Provenance
from base_neural_model.base.types import MechanicalDisplacement
from base_neural_model.forward.detection import (
    AcquisitionParams,
    detection_budget,
    integration_gain,
    phase_displacement_floor,
    phase_to_displacement_m_per_rad,
)


def _mech(dz_m: float, survival: float = 1.0) -> MechanicalDisplacement:
    """A minimal source deliverable carrying a displacement and a survival factor."""
    prov = Provenance(
        source="test source displacement",
        assumptions=("synthetic Delta z for the detection layer",),
        band=Band.CONTENT_FAST,
    )
    return MechanicalDisplacement(
        axial_displacement_m=dz_m,
        volumetric_strain=0.0,
        incoherent_pedestal_m=0.0,
        synchrony_fraction=1.0,
        jitter_sigma_s=1e-3,
        content_band_survival=survival,
        confinement_kappa=0.5,
        dilatation_eta=0.5,
        band=Band.CONTENT_FAST,
        provenance=prov,
    )


# --- integration gain (Gap 1) ------------------------------------------------------


def test_integration_gain_motor_vs_speech():
    """The demo's load-bearing motor advantage: ~x77 (1.5 s) vs ~x14 (50 ms) at 4 kHz."""
    motor = AcquisitionParams.demo_motor()
    speech = AcquisitionParams.speech_unit()
    assert motor.ensemble_count == 6000
    assert speech.ensemble_count == 200
    assert integration_gain(motor) == pytest.approx(77.46, rel=1e-3)
    assert integration_gain(speech) == pytest.approx(14.14, rel=1e-3)
    # sqrt scaling: 30x the epoch -> sqrt(30) ~ 5.48x the gain.
    assert integration_gain(motor) / integration_gain(speech) == pytest.approx(
        math.sqrt(6000 / 200), rel=1e-6
    )


# --- the phase prefactor and the floor (Gap 2) -------------------------------------


def test_phase_prefactor_matches_spec():
    """lambda/4pi ~ 6.13e-5 m/rad at 2 MHz; a 10 nm shift -> ~1.6e-4 rad (the spec)."""
    acq = AcquisitionParams.demo_motor()
    prefactor = phase_to_displacement_m_per_rad(acq)
    assert prefactor == pytest.approx(6.127e-5, rel=1e-3)
    delta_phi = 10e-9 / prefactor  # invert: phase for a 10 nm displacement
    assert delta_phi == pytest.approx(1.632e-4, rel=1e-3)


def test_phase_floor_falls_with_integration_and_snr():
    """The floor falls as 1/sqrt(N_ens) and 1/sqrt(SNR_echo)."""
    base = AcquisitionParams.demo_motor()
    f0 = phase_displacement_floor(base)

    # 4x the epoch -> 4x N_ens -> 2x the gain -> half the floor.
    longer = replace(base, epoch_s=base.epoch_s * 4.0)
    assert phase_displacement_floor(longer) == pytest.approx(f0 / 2.0, rel=1e-6)

    # 4x the echo SNR -> 2x sqrt(SNR) -> half the floor.
    cleaner = replace(base, echo_snr_linear=base.echo_snr_linear * 4.0)
    assert phase_displacement_floor(cleaner) == pytest.approx(f0 / 2.0, rel=1e-6)


def test_phase_floor_rises_with_skull_loss():
    """More skull dB -> lower through-skull echo SNR -> a higher floor."""
    base = replace(AcquisitionParams.demo_motor(), skull_loss_db_oneway=0.0)
    lossy = replace(base, skull_loss_db_oneway=12.0)  # 24 dB two-way
    ratio = phase_displacement_floor(lossy) / phase_displacement_floor(base)
    assert ratio == pytest.approx(10.0 ** (24.0 / 20.0), rel=1e-6)


# --- the composed budget + the binding denominator ---------------------------------


def test_budget_surviving_dz_folds_survival_and_gain():
    """Surviving displacement = Delta z * content survival * sqrt(N_ens)."""
    acq = AcquisitionParams.demo_motor()
    mech = _mech(3.0e-9, survival=0.8)
    b = detection_budget(mech, acq)
    assert b.surviving_dz_m == pytest.approx(3.0e-9 * 0.8 * acq.integration_gain)
    assert b.ensemble_count == 6000


def test_budget_echo_snr_limited_by_default():
    """With no residual clutter the echo-SNR (Walker-Trahey) floor binds."""
    acq = AcquisitionParams.demo_motor()
    b = detection_budget(_mech(3.0e-9, survival=0.8), acq)
    assert b.limiting_denominator == "echo_snr"
    assert b.floor_m == pytest.approx(phase_displacement_floor(acq))


def test_budget_clutter_limited_when_residual_dominates():
    """A residual clutter floor above the phase floor flips the binding denominator."""
    acq = AcquisitionParams.demo_motor()
    phase_floor = phase_displacement_floor(acq)
    b = detection_budget(
        _mech(3.0e-9, survival=0.8), acq, residual_clutter_m=phase_floor * 5.0
    )
    assert b.limiting_denominator == "clutter"
    assert b.floor_m == pytest.approx(phase_floor * 5.0)


def test_budget_snr_db_and_detectable():
    """SNR is 20 log10 of the displacement ratio; detectable iff it clears the floor."""
    acq = AcquisitionParams.demo_motor()
    floor = phase_displacement_floor(acq)
    # Construct a source whose surviving displacement is exactly 10x the floor.
    dz = 10.0 * floor / acq.integration_gain
    b = detection_budget(_mech(dz, survival=1.0), acq)
    assert b.snr_linear == pytest.approx(10.0, rel=1e-6)
    assert b.snr_db == pytest.approx(20.0, rel=1e-6)
    assert b.detectable


def test_acquisition_rejects_bad_inputs():
    base = AcquisitionParams.demo_motor()
    with pytest.raises(ValueError):
        replace(base, epoch_s=0.0)
    with pytest.raises(ValueError):
        replace(base, skull_loss_db_oneway=-1.0)
    with pytest.raises(ValueError):
        detection_budget(_mech(1e-9), base, residual_clutter_m=-1.0)
