"""Model layer - run_neural_model produces both deliverables, consistently.

The end-to-end call must return (a) the activity-driven dz(t) timeseries and (b) the
static dz(neural_state), with the static value equal to the timeseries evaluated at
the state's mean synchrony, and provenance decomposing to the cited constant and the
activity sources.
"""

from __future__ import annotations

import pytest

from base_neural_model.forward.detection import AcquisitionParams
from base_neural_model.model import run_motor_demo, run_neural_model


@pytest.fixture(scope="module")
def report():
    # Short record for speed; the dynamics are stationary on the limit cycle.
    return run_neural_model(duration_s=0.5, fs_hz=2000.0)


def test_run_returns_both_deliverables(report):
    assert report.displacement_timeseries.dz.size > 0          # (a) dz(t)
    assert report.mechanical_displacement.value_m >= 0.0        # (b) static dz


def test_static_matches_timeseries_at_mean_synchrony(report):
    """Deliverable (b) is the chain at the state's synchrony; recompute and compare."""
    from base_neural_model.mechanics.neuron_constants import (
        get_single_neuron_displacement,
    )
    from base_neural_model.mechanics.transduction import mechanical_displacement

    d = get_single_neuron_displacement()
    state = report.neural_state
    params = state.to_mechanics_params(_central_with_constant(d))
    recomputed = mechanical_displacement(
        d, _default_voxel(), params, state.synchrony_fraction
    )
    assert recomputed.value_m == pytest.approx(
        report.mechanical_displacement.value_m, rel=1e-9
    )


def test_gates_are_booleans(report):
    assert isinstance(report.passes_amplitude_gate, bool)
    assert isinstance(report.passes_content_gate, bool)
    assert isinstance(report.passes_dilatation_gate, bool)
    assert isinstance(report.all_gates_pass, bool)


def test_provenance_decomposes_to_inputs(report):
    assert report.provenance.source
    assert any(
        "single-neuron" in a or "AP membrane" in a
        for a in report.provenance.assumptions
    )


def test_neural_state_drives_displacement(report):
    """A real (oscillatory) neural state yields a positive content-surviving signal."""
    assert report.neural_state.synchrony_fraction > 0.0
    assert report.displacement_timeseries.peak_dz_m > 0.0


# --- the acoustic detection layer (Gate A / Stage 1) ----------------------------


def test_detection_absent_without_acquisition(report):
    """Source-side run: no acquisition -> no detection budget, scalar-floor scoring."""
    assert report.detection is None


def test_acquisition_attaches_detection_budget():
    """Supplying the (honest) demo acquisition attaches a budget scored against the floor.

    ``demo_motor`` now carries the honest acoustic terms, so the effective integration is
    the burst-limited ~x10 (N_ens = 100), not the raw x77 (which is demo_motor_optimistic).
    """
    r = run_neural_model(
        duration_s=0.5, fs_hz=2000.0, acquisition=AcquisitionParams.demo_motor()
    )
    assert r.detection is not None
    d = r.detection
    assert d.raw_ensemble_count == 6000
    assert d.ensemble_count == 100                       # burst + decorrelation capped
    assert d.integration_gain == pytest.approx(10.0, rel=1e-6)
    assert d.effective_n_elements == 154                 # aperture coherence 0.6
    assert d.aberration_floor_m > 0.0                    # residual-aberration floor present
    assert d.snr_exceeds_safety is True                  # 30 dB > 28 dB transcranial ceiling
    assert d.floor_m > 0.0
    # The optimistic baseline still gives the historical x77 (the before/after audit).
    opt = run_neural_model(
        duration_s=0.5, fs_hz=2000.0,
        acquisition=AcquisitionParams.demo_motor_optimistic(),
    )
    assert opt.detection.integration_gain == pytest.approx(77.46, rel=1e-3)


def test_motor_demo_content_band_is_far_under_but_envelope_is_large():
    """The honest flagship Gate A: the direct beta signal is orders under; fUS is not.

    With the honest source physics (sub-nm Delta r, viscoelastic transfer, carrier depth,
    coherent fraction) and the honest acoustic terms (burst-limited integration, aperture
    decoherence, residual aberration, safety-capped echo SNR), the direct neuromechanical
    beta content sits ~50+ dB under the through-skull floor -- not the prior -13 dB
    'within an order'. The band-separated decomposition makes the trade explicit: the slow
    hemodynamic (vascular/CBV) envelope is orders LARGER (the fUS signal), but it is the
    envelope, not the specific beta carrier the phase-displacement readout targets.
    """
    r = run_motor_demo()
    d = r.detection
    assert d is not None
    # The content-band (direct neuromechanical) verdict is deep under the floor.
    assert d.snr_db < -40.0
    # The mechanism decomposition is attached and band-separated.
    m = r.mechanisms
    assert m is not None
    # The hemodynamic (vascular) envelope dwarfs the direct beta term by orders.
    assert m.vascular_axial_m > 100.0 * m.direct_axial_m
    assert m.envelope_band_axial_m > d.floor_m          # the envelope clears the floor
    assert d.surviving_dz_m < d.floor_m                 # the beta content does not
    # Both receive-side gains are still load-bearing on the (honest) floor.
    raw_floor = d.floor_m * d.integration_gain * d.beamforming_gain
    assert d.surviving_dz_m < raw_floor


def test_clutter_limited_run_reports_clutter():
    """A large residual clutter floor flips the binding denominator in a full run."""
    r = run_motor_demo(residual_clutter_m=1e-5)
    assert r.detection is not None
    assert r.detection.limiting_denominator == "clutter"


# --- helpers (mirror run.py defaults) -------------------------------------------


def _central_with_constant(d):
    from dataclasses import replace

    from base_neural_model.base.types import MechanicsParams

    return replace(MechanicsParams.central(), membrane_disp_m=d.value_m)


def _default_voxel():
    from base_neural_model.model.run import DEFAULT_VOXEL

    return DEFAULT_VOXEL
