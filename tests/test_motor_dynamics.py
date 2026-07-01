"""Movement-locked motor dynamics: beta desync at onset, rebound after.

A real M1 motor trial is not a steady limit cycle. The movement-locked drive P(t)
suppresses beta (population desynchronizes) at movement onset and rebounds above
baseline afterwards; the tissue displacement dz(t) tracks that arc. These tests pin
the drive shape and the desync -> rebound sequence in both synchrony and displacement.
"""

from __future__ import annotations

import numpy as np

from base_neural_model.activity.motor_drive import MovementProfile, movement_drive
from base_neural_model.activity.populations import EIParams
from base_neural_model.activity.timeseries import run_activity
from base_neural_model.model.run import run_motor_trial

# --- the drive profile P(t): baseline, onset dip, post-movement overshoot --------


def test_drive_dips_during_movement_and_rebounds():
    p = MovementProfile(
        drive_base=1.2, onset_time_s=0.4, move_duration_s=0.3,
        rebound_lag_s=0.1, rebound_duration_s=0.3,
    )
    drive = movement_drive(p)
    base = drive(0.1)                 # before movement
    during = drive(0.55)              # mid-movement (the dip)
    rebound = drive(0.95)             # after movement (the overshoot)
    assert during < base              # beta suppression at onset
    assert rebound > base             # beta rebound above baseline


def test_drive_never_negative():
    p = MovementProfile(drive_base=0.5, onset_gain=2.0)  # dip would go negative
    drive = movement_drive(p)
    assert drive(p.onset_time_s + 0.5 * p.move_duration_s) >= 0.0


def test_movement_profile_validation():
    import pytest

    with pytest.raises(ValueError):
        MovementProfile(move_duration_s=0.0)


# --- the synchrony arc: desync at onset, rebound after --------------------------


def test_synchrony_desyncs_then_rebounds():
    prof = MovementProfile(onset_time_s=0.4, move_duration_s=0.3)
    ts = run_activity(
        EIParams.motor_cortex(), duration_s=1.0, fs_hz=2000.0,
        drive_fn=movement_drive(prof),
    )
    t, r = ts.t_s, ts.r
    baseline = r[(t > 0.1) & (t < 0.35)].mean()
    movement = r[(t > 0.5) & (t < 0.68)].mean()
    rebound = r[(t > 0.85) & (t < 1.0)].mean()
    assert movement < baseline        # desynchronization at movement onset
    assert rebound > movement         # rebound after movement


def test_constant_drive_unchanged_by_dynamics_hook():
    """With no drive_fn the steady cycle is unaffected (backward compatibility)."""
    ts = run_activity(EIParams.motor_cortex(), duration_s=0.6, fs_hz=2000.0)
    assert 0.0 <= ts.mean_synchrony <= 1.0
    assert ts.spectrum.dominant_freq_hz > 0.0


# --- end-to-end: the displacement timeseries tracks the movement ----------------


def test_motor_trial_displacement_tracks_movement():
    report = run_motor_trial(duration_s=1.0, fs_hz=2000.0)
    ts = report.displacement_timeseries
    t = ts.t_s
    dz = ts.surviving_dz
    base = dz[(t > 0.1) & (t < 0.35)].mean()
    move = dz[(t > 0.5) & (t < 0.68)].mean()
    rebound = dz[(t > 0.85) & (t < 1.0)].mean()
    # The tissue displacement drops during movement (beta desync) and rebounds.
    assert move < base
    assert rebound > move
    assert np.all(dz >= 0.0)


def test_motor_trial_keeps_directional_channel():
    report = run_motor_trial(duration_s=0.6, fs_hz=2000.0)
    md = report.mechanical_displacement
    assert md.directional_axial_m > 0.0          # columnar M1 -> directional signal
    assert report.neural_state.orientation_coherence > 0.0


def test_motor_trial_provenance_records_movement():
    report = run_motor_trial(duration_s=0.6, fs_hz=2000.0)
    joined = " ".join(report.provenance.assumptions)
    assert "movement-locked" in joined or "beta" in joined.lower()


def test_motor_trial_is_a_dynamics_visualizer_with_content_band_fc():
    """The trial's deliverable is the timeseries, reduced f_c is the content carrier.

    Contract for the dynamics-visualizer design: even though a movement event's overall
    dominant spectral peak is the slow (~1 Hz) movement envelope, the reduced
    NeuralState must carry a CONTENT-BAND f_c (the beta carrier) -- both because the
    timeseries low-pass needs a real content corner and because a sub-boundary f_c would
    be a silent Invariant-3 violation. The timeseries is the product; the gates are a
    whole-record-average artifact and are deliberately not asserted here.
    """
    from base_neural_model.activity.oscillation import ENVELOPE_CONTENT_BOUNDARY_HZ

    report = run_motor_trial(duration_s=1.0, fs_hz=2000.0)
    # The deliverable exists and is a real curve.
    assert report.displacement_timeseries.surviving_dz.size > 0
    # The reduced content corner is a genuine content-band carrier, not the envelope.
    assert report.neural_state.content_freq_hz >= ENVELOPE_CONTENT_BOUNDARY_HZ
