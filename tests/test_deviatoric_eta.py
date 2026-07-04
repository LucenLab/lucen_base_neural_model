"""The deviatoric (shape-change) channel is gated by its OWN net-realization
fraction ``deviatoric_eta``, decoupled from the volumetric poroelastic ``dilatation_eta``.

The directional term is a constant-volume shape change; the drainage fraction ``eta``
(how much of a VOLUME change survives ECS-reservoir redistribution) physically does not
apply to it. These tests pin that decoupling, so a future change to the volumetric eta
cannot silently move the shape channel, and vice versa.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from base_neural_model import mechanical_displacement
from base_neural_model.base.types import MechanicsParams, VoxelGeometry
from base_neural_model.mechanics.neuron_constants import get_single_neuron_displacement


def _m1_params(d_single, **over):
    """M1 preset with the cited Delta r pinned and an active orientation coherence."""
    base = replace(
        MechanicsParams.motor_cortex(),
        membrane_disp_m=d_single.value_m,
        orientation_coherence=0.7,  # activity layer would fill this in; fix it here
    )
    return replace(base, **over)


def test_directional_term_does_not_move_with_volumetric_eta():
    """The deviatoric axial term is invariant to ``dilatation_eta`` (the category fix).

    Halving the volumetric eta must halve the ISOTROPIC term but leave the DIRECTIONAL
    term untouched -- they are now governed by different fractions.
    """
    d = get_single_neuron_displacement()
    geom = VoxelGeometry.motor_cortex_layer5()
    s = 0.8

    hi = mechanical_displacement(d, geom, _m1_params(d, dilatation_eta=0.5), s)
    lo = mechanical_displacement(d, geom, _m1_params(d, dilatation_eta=0.25), s)

    # Isotropic term is linear in dilatation_eta -> halves.
    assert lo.isotropic_axial_m == pytest.approx(0.5 * hi.isotropic_axial_m, rel=1e-9)
    # Directional term carries deviatoric_eta (unchanged here) -> identical.
    assert lo.directional_axial_m == pytest.approx(hi.directional_axial_m, rel=1e-9)
    assert lo.directional_axial_m > 0.0  # non-degenerate: the channel is active


def test_directional_term_scales_with_deviatoric_eta():
    """The deviatoric axial term IS linear in ``deviatoric_eta`` (its own knob)."""
    d = get_single_neuron_displacement()
    geom = VoxelGeometry.motor_cortex_layer5()
    s = 0.8

    full = mechanical_displacement(d, geom, _m1_params(d, deviatoric_eta=1.0), s)
    half = mechanical_displacement(d, geom, _m1_params(d, deviatoric_eta=0.5), s)

    assert half.directional_axial_m == pytest.approx(
        0.5 * full.directional_axial_m, rel=1e-9
    )
    # The isotropic (volume) term does not see deviatoric_eta.
    assert half.isotropic_axial_m == pytest.approx(full.isotropic_axial_m, rel=1e-9)


def test_deviatoric_eta_default_is_one_and_undiscounted():
    """Default ``deviatoric_eta = 1.0``: shape change fully realized (undrained)."""
    assert MechanicsParams.central().deviatoric_eta == 1.0
    assert MechanicsParams.motor_cortex().deviatoric_eta == 1.0


def test_deviatoric_eta_is_validated():
    """Out-of-range ``deviatoric_eta`` raises in the chain (a [0, 1] fraction)."""
    d = get_single_neuron_displacement()
    geom = VoxelGeometry.motor_cortex_layer5()
    with pytest.raises(ValueError, match="deviatoric_eta"):
        mechanical_displacement(d, geom, _m1_params(d, deviatoric_eta=1.5), 0.8)


def test_motor_preset_ships_the_cited_displacement():
    """The M1 preset carries the cited whole-cell Delta r (0.4 nm), not the old 1.5 nm.

    A direct ``mechanical_displacement(preset)`` call must not trip the Invariant-2
    guard (d_single.value_m == params.membrane_disp_m).
    """
    d = get_single_neuron_displacement()
    assert MechanicsParams.motor_cortex().membrane_disp_m == pytest.approx(d.value_m)


def test_isotropic_model_unaffected_by_deviatoric_eta():
    """With beta = 0 (central preset) there is no directional term, so deviatoric_eta
    is inert -- the isotropic chain is byte-for-byte unchanged."""
    d = get_single_neuron_displacement()
    geom = VoxelGeometry.motor_cortex_layer5()
    base = replace(MechanicsParams.central(), membrane_disp_m=d.value_m)
    a = mechanical_displacement(d, geom, replace(base, deviatoric_eta=1.0), 0.8)
    b = mechanical_displacement(d, geom, replace(base, deviatoric_eta=0.2), 0.8)
    assert a.axial_displacement_m == pytest.approx(b.axial_displacement_m, rel=1e-12)
    assert a.directional_axial_m == 0.0
