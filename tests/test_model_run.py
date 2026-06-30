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
    """Supplying an acquisition attaches a budget and scores against the derived floor."""
    r = run_neural_model(
        duration_s=0.5, fs_hz=2000.0, acquisition=AcquisitionParams.demo_motor()
    )
    assert r.detection is not None
    d = r.detection
    assert d.integration_gain == pytest.approx(77.46, rel=1e-3)
    assert d.surviving_dz_m > 0.0
    assert d.floor_m > 0.0
    assert d.limiting_denominator == "echo_snr"


def test_motor_demo_lands_within_an_order_of_the_floor():
    """The flagship Gate A: integration lifts the source within ~an order of the floor.

    The spec's success criterion is 'within one to two orders of magnitude of the
    detection floor'. With the demo's conservative echo SNR and skull loss the motor
    source, after the ~x77 integration gain, lands within a single order of the derived
    through-skull floor -- the engineering-sized gap, not a wall.
    """
    r = run_motor_demo()
    d = r.detection
    assert d is not None
    # Within one order: SNR in dB is better than -20 dB (a factor of 10 in displacement).
    assert d.snr_db > -20.0
    # The integration gain is load-bearing: without it the bare source is far under.
    bare = d.surviving_dz_m / d.integration_gain
    assert bare < d.floor_m  # the un-integrated source alone does not clear the floor


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
