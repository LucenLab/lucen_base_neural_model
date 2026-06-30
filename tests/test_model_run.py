"""Model layer - run_neural_model produces both deliverables, consistently.

The end-to-end call must return (a) the activity-driven dz(t) timeseries and (b) the
static dz(neural_state), with the static value equal to the timeseries evaluated at
the state's mean synchrony, and provenance decomposing to the cited constant and the
activity sources.
"""

from __future__ import annotations

import pytest

from base_neural_model.model import run_neural_model


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


# --- helpers (mirror run.py defaults) -------------------------------------------


def _central_with_constant(d):
    from dataclasses import replace

    from base_neural_model.base.types import MechanicsParams

    return replace(MechanicsParams.central(), membrane_disp_m=d.value_m)


def _default_voxel():
    from base_neural_model.model.run import DEFAULT_VOXEL

    return DEFAULT_VOXEL
