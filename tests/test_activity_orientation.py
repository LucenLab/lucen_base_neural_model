"""Activity layer - directional coherence emerges from the dynamics (Option 2).

Orientation coherence Q is no longer only a hardcoded mechanics input: a structurally
aligned (columnar) population expresses its alignment through the temporal synchrony,
so the activity layer produces Q_eff = Q_struct * synchrony. These tests pin that
emergence and its flow into the mechanics.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from base_neural_model.activity.orientation import (
    effective_orientation_coherence,
    expression_factor,
)
from base_neural_model.activity.populations import EIParams
from base_neural_model.activity.reduce import reduce_to_state
from base_neural_model.activity.timeseries import run_activity

# --- the emergence relation Q_eff = Q_struct * f_express(r) ----------------------


def test_expression_factor_limits():
    assert expression_factor(0.0) == 0.0
    assert expression_factor(1.0) == 1.0


def test_effective_q_zero_when_isotropic():
    """No structural alignment -> no directional coherence at any synchrony."""
    for r in (0.0, 0.5, 1.0):
        assert effective_orientation_coherence(0.0, r) == 0.0


def test_effective_q_zero_when_incoherent():
    """Aligned structure but incoherent firing -> directional sum still washes out."""
    assert effective_orientation_coherence(1.0, 0.0) == 0.0


def test_effective_q_is_product():
    assert effective_orientation_coherence(0.8, 0.5) == pytest.approx(0.4)


def test_effective_q_rises_with_both_factors():
    base = effective_orientation_coherence(0.5, 0.5)
    assert effective_orientation_coherence(0.9, 0.5) > base   # more structure
    assert effective_orientation_coherence(0.5, 0.9) > base   # more synchrony


def test_effective_q_rejects_out_of_range():
    with pytest.raises(ValueError):
        effective_orientation_coherence(1.5, 0.5)


# --- the reduce step emits (or withholds) Q on the NeuralState ------------------


def test_isotropic_population_emits_no_orientation():
    ts = run_activity(EIParams.central(), duration_s=0.4, fs_hz=2000.0)
    state = reduce_to_state(ts)
    assert state.orientation_coherence is None


def test_columnar_population_emits_orientation():
    ei = replace(EIParams.central(), structural_alignment=0.9)
    ts = run_activity(ei, duration_s=0.4, fs_hz=2000.0)
    state = reduce_to_state(ts)
    assert state.orientation_coherence is not None
    # Q_eff equals Q_struct * mean synchrony.
    assert state.orientation_coherence == pytest.approx(
        0.9 * state.synchrony_fraction
    )
    assert 0.0 <= state.orientation_coherence <= 1.0


def test_orientation_provenance_recorded():
    ei = replace(EIParams.central(), structural_alignment=0.8)
    ts = run_activity(ei, duration_s=0.4, fs_hz=2000.0)
    state = reduce_to_state(ts)
    assert any("orientation coherence" in a for a in state.provenance.assumptions)


# --- end-to-end: activity-generated Q drives the mechanics ----------------------


def test_columnar_population_drives_directional_signal():
    from base_neural_model.base.types import MechanicsParams
    from base_neural_model.model.run import run_neural_model

    ei = replace(EIParams.central(), structural_alignment=0.9)
    mech = replace(MechanicsParams.central(), anisotropy=0.6, mean_axis_projection=1.0)
    report = run_neural_model(ei, mechanics=mech, duration_s=0.4, fs_hz=2000.0)
    md = report.mechanical_displacement
    # The directional term is non-zero and the Q the mechanics used came from activity.
    assert md.directional_axial_m > 0.0
    assert md.orientation_coherence == pytest.approx(
        report.neural_state.orientation_coherence
    )
    assert md.value_m > md.isotropic_axial_m


def test_structural_alignment_validation():
    with pytest.raises(ValueError):
        replace(EIParams.central(), structural_alignment=1.5)
