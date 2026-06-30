"""The motor-cortex (M1) preset and what makes it distinct from a generic column.

M1 differs from the generic cortical model in physically specific ways: a beta-band
rhythm (not gamma), strongly columnar layer-5 pyramidal/Betz cells (high structural
alignment -> an active directional channel), large sparse somata, and a layer-5
voxel depth. These tests pin those differences and the self-consistency of the
preset.
"""

from __future__ import annotations

import pytest

from base_neural_model.activity.populations import EIParams
from base_neural_model.activity.reduce import reduce_to_state
from base_neural_model.activity.timeseries import run_activity
from base_neural_model.base.types import MechanicsParams, VoxelGeometry
from base_neural_model.model.run import run_motor_cortex, run_neural_model

# --- the activity preset: a beta-band, columnar population ----------------------


def test_motor_rhythm_is_beta_band():
    """M1's E/I limit cycle lands in the beta band (~13-30 Hz), not gamma."""
    ts = run_activity(EIParams.motor_cortex(), duration_s=1.0, fs_hz=2000.0)
    f_c = ts.spectrum.dominant_freq_hz
    assert 13.0 <= f_c <= 30.0


def test_motor_population_is_columnar():
    assert EIParams.motor_cortex().structural_alignment > 0.5


def test_motor_state_emits_orientation_coherence():
    ts = run_activity(EIParams.motor_cortex(), duration_s=0.5, fs_hz=2000.0)
    state = reduce_to_state(ts)
    assert state.orientation_coherence is not None
    assert state.orientation_coherence > 0.0


# --- the mechanics preset: large, anisotropic Betz cells ------------------------


def test_motor_cells_are_large_and_anisotropic():
    m = MechanicsParams.motor_cortex()
    generic = MechanicsParams.central()
    assert m.cell_radius_m > generic.cell_radius_m      # large layer-5 somata
    assert m.anisotropy > 0.0                           # elongated, columnar
    assert m.content_freq_hz <= 30.0                    # beta content corner


def test_motor_preset_is_volume_fraction_consistent():
    """The sparse large-cell count must agree with the asserted volume fraction -
    using the generic 10k count would imply f_cell > 1 (cells overfilling)."""
    m = MechanicsParams.motor_cortex()
    v = VoxelGeometry.motor_cortex_layer5()
    # Should not raise: the count is set for consistency with the Betz-cell radius.
    m.check_volume_fraction_consistency(v, rel_tol=0.15)


def test_motor_voxel_at_layer5_depth():
    v = VoxelGeometry.motor_cortex_layer5()
    assert 1.0e-3 <= v.depth_m <= 2.5e-3   # M1 layer 5 below the pial surface


# --- end-to-end: the directional channel is material in M1 ----------------------


def test_motor_run_has_directional_contribution():
    """The columnar alignment, expressed through synchrony, makes the directional
    channel contribute a real share of the M1 signal - unlike the generic model."""
    report = run_motor_cortex(duration_s=0.5, fs_hz=2000.0)
    md = report.mechanical_displacement
    assert md.directional_axial_m > 0.0
    assert md.orientation_coherence > 0.0
    # The total is the isotropic plus the (positive) directional term.
    assert md.value_m == pytest.approx(
        md.isotropic_axial_m + md.directional_axial_m
    )


def test_generic_run_has_no_directional_contribution():
    """The generic model stays isotropic - the directional channel is M1-specific."""
    report = run_neural_model(duration_s=0.5, fs_hz=2000.0)
    md = report.mechanical_displacement
    assert md.directional_axial_m == 0.0
    assert report.neural_state.orientation_coherence is None


def test_motor_provenance_names_m1_features():
    report = run_motor_cortex(duration_s=0.4, fs_hz=2000.0)
    joined = " ".join(report.provenance.assumptions)
    assert "beta" in joined.lower()
    assert "columnar" in joined.lower() or "Q_struct" in joined
