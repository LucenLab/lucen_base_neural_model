"""Acoustic detection layer: gains, the phase floor, the honest-physics terms, budget.

Pins the receive-side coherent gains the Gate-A / Stage-1 verdict rests on -- spatial
beamforming (sqrt(n_eff)) and temporal within-window integration (1/sqrt(N_ens)), BOTH
folded into the Walker-Trahey through-skull floor exactly once -- plus the honest-physics
terms (D1-D9): the coherence/decorrelation-limited effective sample count, the echo
correlation, the residual-aberration phase-noise floor, the aperture-coherence erosion,
the clutter high-pass, reverberation, and the per-element vs post-beamforming switch.

The pure SNR-scaling relations are pinned on ``demo_motor_optimistic()`` (all
honest-physics terms inert), which reproduces the historical -13.03 dB baseline; each
honest term then gets its own test on top of that base, and ``demo_motor()`` (the honest
preset) is asserted to be materially worse.
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
    aberration_displacement_floor,
    detection_budget,
    integration_gain,
    phase_displacement_floor,
    phase_to_displacement_m_per_rad,
    walker_trahey_floor,
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


def _inert(epoch_s: float) -> AcquisitionParams:
    """Inert (all honest terms off) acquisition at a given epoch -- the pre-D1 baseline."""
    return AcquisitionParams(
        center_freq_hz=2e6, frame_rate_hz=4000.0, epoch_s=epoch_s,
        echo_snr_linear=1.0e3, skull_loss_db_oneway=12.0, n_elements=256,
    )


# --- integration gain: raw count and effective (D1) --------------------------------


def test_raw_integration_gain_motor_vs_speech():
    """The naive count: ~x77 (1.5 s) vs ~x14 (50 ms) at 4 kHz, every frame independent."""
    motor = AcquisitionParams.demo_motor_optimistic()  # 1.5 s
    speech = _inert(0.05)
    assert motor.ensemble_count == 6000
    assert speech.ensemble_count == 200
    assert integration_gain(motor) == pytest.approx(77.46, rel=1e-3)
    assert integration_gain(speech) == pytest.approx(14.14, rel=1e-3)


def test_effective_sample_count_capped_by_coherence_and_decorrelation():
    """D1: N_ens is capped below the raw frame count by the coherence window + decorr time."""
    hon = AcquisitionParams.demo_motor()
    # window = min(epoch 1.5, coherence 0.2, 1/highpass 1.0) = 0.2 s.
    assert hon.coherent_window_s == pytest.approx(0.2)
    # N_ens = min(4000*0.2 = 800 frames, 0.2/0.002 = 100 independent looks) = 100.
    assert hon.raw_ensemble_count == 6000
    assert hon.ensemble_count == 100
    assert hon.integration_gain == pytest.approx(10.0)
    # Slower decorrelation -> fewer independent looks.
    slower = replace(hon, frame_decorrelation_time_s=0.004)
    assert slower.ensemble_count == 50


def test_coherence_window_bounded_by_clutter_highpass():
    """D2: a higher clutter high-pass shortens the coherent window (1/cutoff memory)."""
    hon = AcquisitionParams.demo_motor()
    tight = replace(hon, clutter_highpass_hz=20.0)  # 1/20 = 0.05 s < 0.2 coherence
    assert tight.coherent_window_s == pytest.approx(0.05)


# --- the phase prefactor (unchanged) -----------------------------------------------


def test_phase_prefactor_matches_spec():
    """lambda/4pi ~ 6.13e-5 m/rad at 2 MHz; a 10 nm shift -> ~1.6e-4 rad (the spec)."""
    acq = AcquisitionParams.demo_motor()
    prefactor = phase_to_displacement_m_per_rad(acq)
    assert prefactor == pytest.approx(6.127e-5, rel=1e-3)
    delta_phi = 10e-9 / prefactor
    assert delta_phi == pytest.approx(1.632e-4, rel=1e-3)


# --- Walker-Trahey floor scaling (inert base: only the SNR-limited term) -----------


def test_wt_floor_falls_with_integration_and_snr():
    """The WT floor falls as 1/sqrt(N_ens) and 1/sqrt(SNR_echo)."""
    base = AcquisitionParams.demo_motor_optimistic()
    f0 = walker_trahey_floor(base)
    longer = replace(base, epoch_s=base.epoch_s * 4.0)  # 4x N_ens -> half the floor
    assert walker_trahey_floor(longer) == pytest.approx(f0 / 2.0, rel=1e-6)
    cleaner = replace(base, echo_snr_linear=base.echo_snr_linear * 4.0)  # 4x SNR -> half
    assert walker_trahey_floor(cleaner) == pytest.approx(f0 / 2.0, rel=1e-6)


def test_wt_floor_rises_with_skull_loss():
    """More skull dB -> lower through-skull echo SNR -> a higher floor (x20log10 two-way)."""
    base = replace(AcquisitionParams.demo_motor_optimistic(), skull_loss_db_oneway=0.0)
    lossy = replace(base, skull_loss_db_oneway=12.0)  # 24 dB two-way
    ratio = walker_trahey_floor(lossy) / walker_trahey_floor(base)
    assert ratio == pytest.approx(10.0 ** (24.0 / 20.0), rel=1e-6)


def test_wt_floor_falls_with_beamforming_aperture():
    """The floor falls as 1/sqrt(n_elements): the spatial (beamforming) coherent gain."""
    base = replace(AcquisitionParams.demo_motor_optimistic(), n_elements=64)
    bigger = replace(base, n_elements=256)  # 4x elements -> 2x beamforming gain
    assert base.beamforming_gain == pytest.approx(8.0, rel=1e-9)
    assert bigger.beamforming_gain == pytest.approx(16.0, rel=1e-9)
    assert walker_trahey_floor(bigger) == pytest.approx(walker_trahey_floor(base) / 2.0, rel=1e-9)
    assert bigger.beamformed_echo_snr_linear == pytest.approx(256 * bigger.echo_snr_linear)


def test_gains_are_orthogonal_and_each_counted_once():
    """sqrt(n_elements) (spatial) and sqrt(N_ens) (temporal) both live in the floor once."""
    base = AcquisitionParams.demo_motor_optimistic()
    f0 = walker_trahey_floor(base)
    both = replace(base, n_elements=base.n_elements * 4, epoch_s=base.epoch_s * 4)
    assert walker_trahey_floor(both) == pytest.approx(f0 / 4.0, rel=1e-9)


# --- D5 echo correlation rho -------------------------------------------------------


def test_echo_correlation_inflates_floor():
    """D5: a lower per-estimate echo correlation rho raises the floor by 1/rho."""
    base = AcquisitionParams.demo_motor_optimistic()
    f0 = walker_trahey_floor(base)
    rho = replace(base, echo_correlation=0.5)
    assert walker_trahey_floor(rho) == pytest.approx(f0 / 0.5, rel=1e-9)


# --- D4 aperture coherence ---------------------------------------------------------


def test_aperture_coherence_erodes_beamforming():
    """D4: only a coherent fraction of the aperture sums -> a smaller effective count."""
    base = AcquisitionParams.demo_motor_optimistic()
    assert base.effective_n_elements == 256
    quarter = replace(base, aperture_coherence=0.25)  # n_eff = 64
    assert quarter.effective_n_elements == 64
    assert quarter.beamforming_gain == pytest.approx(8.0, rel=1e-9)
    # Floor rises by sqrt(256/64) = 2.
    assert walker_trahey_floor(quarter) == pytest.approx(walker_trahey_floor(base) * 2.0, rel=1e-9)


# --- D3 residual-aberration phase-noise floor --------------------------------------


def test_aberration_floor_adds_in_quadrature_and_ignores_integration():
    """D3: aberration phase noise adds in quadrature and does NOT average with integration."""
    base = AcquisitionParams.demo_motor_optimistic()
    wt = walker_trahey_floor(base)
    aber = replace(base, aberration_phase_rad=1e-4)
    sigma_aber = aberration_displacement_floor(aber)
    assert sigma_aber == pytest.approx((base.wavelength_m / (4.0 * math.pi)) * 1e-4, rel=1e-9)
    assert phase_displacement_floor(aber) == pytest.approx(math.hypot(wt, sigma_aber), rel=1e-9)
    # More integration lowers the WT term but leaves the aberration floor unchanged.
    more_integ = replace(aber, epoch_s=aber.epoch_s * 100.0)
    assert aberration_displacement_floor(more_integ) == pytest.approx(sigma_aber, rel=1e-12)
    assert aberration_displacement_floor(base) == 0.0  # inert default


# --- D9 per-element vs post-beamforming echo SNR -----------------------------------


def test_per_element_vs_post_beamforming_switch():
    """D9: treating the echo SNR as post-beamforming removes the sqrt(n_eff) aperture gain."""
    base = AcquisitionParams.demo_motor_optimistic()
    pe = detection_budget(_mech(3.9071e-9), base)                       # per-element (default)
    pb = detection_budget(_mech(3.9071e-9), replace(base, echo_snr_is_per_element=False))
    # The aperture gain sqrt(256) = 16 -> 24.08 dB swing.
    assert pe.snr_db - pb.snr_db == pytest.approx(20.0 * math.log10(16.0), rel=1e-6)


# --- inert defaults reproduce the historical baseline; honest preset is worse ------


def test_inert_defaults_reproduce_minus_13_db_baseline():
    """demo_motor_optimistic (all honest terms inert) reproduces the ~-13.03 dB verdict."""
    opt = AcquisitionParams.demo_motor_optimistic()
    assert opt.ensemble_count == 6000
    assert opt.effective_n_elements == 256
    assert aberration_displacement_floor(opt) == 0.0
    assert phase_displacement_floor(opt) == pytest.approx(1.7521e-8, rel=1e-3)
    b = detection_budget(_mech(3.9071e-9), opt)
    assert b.snr_db == pytest.approx(-13.03, abs=0.05)


def test_honest_demo_is_materially_worse_than_baseline():
    """The honest terms together drop the same source well below the -13 dB baseline."""
    hon = detection_budget(_mech(3.9071e-9), AcquisitionParams.demo_motor())
    opt = detection_budget(_mech(3.9071e-9), AcquisitionParams.demo_motor_optimistic())
    assert hon.snr_db < opt.snr_db - 15.0


# --- the composed budget + the binding denominator ---------------------------------


def test_budget_surviving_dz_is_bare_gains_in_floor():
    """Surviving displacement = bare Delta z * content survival (NOT re-amplified)."""
    acq = AcquisitionParams.demo_motor_optimistic()
    mech = _mech(3.0e-9, survival=0.8)
    b = detection_budget(mech, acq)
    assert b.surviving_dz_m == pytest.approx(3.0e-9 * 0.8)
    assert b.ensemble_count == 6000
    assert b.integration_gain == pytest.approx(acq.integration_gain)
    assert b.beamforming_gain == pytest.approx(acq.beamforming_gain)
    assert b.n_elements == 256
    assert b.floor_m == pytest.approx(phase_displacement_floor(acq))


def test_budget_echo_snr_limited_by_default():
    """With no residual clutter or reverberation the echo-SNR (phase) floor binds."""
    acq = AcquisitionParams.demo_motor_optimistic()
    b = detection_budget(_mech(3.0e-9, survival=0.8), acq)
    assert b.limiting_denominator == "echo_snr"
    assert b.floor_m == pytest.approx(phase_displacement_floor(acq))


def test_budget_clutter_limited_when_residual_dominates():
    """A residual clutter floor above the phase floor flips the binding denominator."""
    acq = AcquisitionParams.demo_motor_optimistic()
    phase_floor = phase_displacement_floor(acq)
    b = detection_budget(
        _mech(3.0e-9, survival=0.8), acq, residual_clutter_m=phase_floor * 5.0
    )
    assert b.limiting_denominator == "clutter"
    assert b.floor_m == pytest.approx(phase_floor * 5.0)


def test_budget_reverberation_flips_denominator():
    """D8: reverberation as a multiple of the phase floor can bind the clutter denominator."""
    acq = replace(AcquisitionParams.demo_motor_optimistic(), reverberation_ratio=3.0)
    b = detection_budget(_mech(3.0e-9, survival=0.8), acq)
    assert b.limiting_denominator == "clutter"
    assert b.floor_m == pytest.approx(3.0 * phase_displacement_floor(
        replace(acq, reverberation_ratio=0.0)
    ))


def test_budget_highpass_removes_subcutoff_content():
    """D2: content below the clutter high-pass is filtered out with the bulk motion."""
    acq = AcquisitionParams.demo_motor()  # clutter_highpass_hz = 1.0
    removed = detection_budget(_mech(3.0e-9), acq, content_freq_hz=0.5)
    kept = detection_budget(_mech(3.0e-9), acq, content_freq_hz=20.0)
    assert removed.surviving_dz_m == 0.0
    assert removed.snr_db == float("-inf")
    assert kept.surviving_dz_m == pytest.approx(3.0e-9)


def test_budget_snr_db_and_detectable():
    """SNR is 20 log10 of the displacement ratio; detectable iff it clears the floor."""
    acq = AcquisitionParams.demo_motor_optimistic()
    floor = phase_displacement_floor(acq)
    dz = 10.0 * floor  # survival = 1.0, no re-amplification
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
        replace(base, echo_correlation=0.0)      # must be in (0, 1]
    with pytest.raises(ValueError):
        replace(base, echo_correlation=1.5)
    with pytest.raises(ValueError):
        replace(base, aperture_coherence=0.0)    # must be in (0, 1]
    with pytest.raises(ValueError):
        replace(base, aberration_phase_rad=-1.0)
    with pytest.raises(ValueError):
        replace(base, clutter_highpass_hz=-1.0)
    with pytest.raises(ValueError):
        replace(base, reverberation_ratio=-1.0)
    with pytest.raises(ValueError):
        detection_budget(_mech(1e-9), base, residual_clutter_m=-1.0)
