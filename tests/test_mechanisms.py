"""The band-separated mechanism-combination model (S2).

Pins that the direct neuromechanical (beta content) term co-exists with the slow osmotic
and vascular (envelope) terms; that the vascular/CBV term dwarfs the direct term (the
fUS-vs-neuromechanical trade); that the osmotic strain is saturated (S7) and its net
dilatation suppressed by a small eta; and the band bookkeeping.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from base_neural_model.base.types import MechanicsParams, VoxelGeometry
from base_neural_model.mechanics.mechanisms import (
    MechanismParams,
    decompose_mechanisms,
)
from base_neural_model.mechanics.neuron_constants import get_single_neuron_displacement
from base_neural_model.mechanics.transduction import mechanical_displacement


def _direct():
    d = get_single_neuron_displacement()
    geom = VoxelGeometry.motor_cortex_layer5()
    params = replace(MechanicsParams.motor_cortex(), membrane_disp_m=d.value_m)
    return mechanical_displacement(d, geom, params, 0.83), geom, params


def test_vascular_envelope_dwarfs_direct_beta_content():
    """The hemodynamic (vascular/CBV) envelope term is orders larger than direct beta."""
    direct, geom, params = _direct()
    dec = decompose_mechanisms(
        direct, geom, confinement_kappa=params.confinement_kappa,
        saturation_strain=params.saturation_strain,
    )
    assert dec.content_band_axial_m == direct.axial_displacement_m  # beta content = direct
    assert dec.vascular_axial_m > 1000.0 * dec.direct_axial_m       # ~um vs ~0.1 nm
    assert dec.envelope_band_axial_m == pytest.approx(
        dec.osmotic_axial_m + dec.vascular_axial_m
    )
    assert dec.total_axial_m == pytest.approx(
        dec.direct_axial_m + dec.envelope_band_axial_m
    )


def test_vascular_net_dilatation_is_micron_scale():
    """1% CBV over a 0.3 mm gate with eta~1 -> ~um displacement (the fUS signal)."""
    direct, geom, params = _direct()
    dec = decompose_mechanisms(
        direct, geom, confinement_kappa=0.5,
        mech_params=MechanismParams(vascular_strain=1e-2, vascular_eta=1.0),
        saturation_strain=None,
    )
    # eta * kappa * L * strain = 1 * 0.5 * 3e-4 * 1e-2 = 1.5e-6 m.
    assert dec.vascular_axial_m == pytest.approx(1.5e-6, rel=1e-6)


def test_osmotic_saturated_and_redistribution_suppressed():
    """The large osmotic strain is saturation-capped (S7) and its net dilatation small."""
    direct, geom, _ = _direct()
    mp = MechanismParams(osmotic_strain=2e-2, osmotic_eta=0.05,
                         vascular_strain=0.0, vascular_eta=0.0)
    capped = decompose_mechanisms(direct, geom, confinement_kappa=0.5,
                                  mech_params=mp, saturation_strain=1e-2)
    uncapped = decompose_mechanisms(direct, geom, confinement_kappa=0.5,
                                    mech_params=mp, saturation_strain=None)
    # Saturation reduces the osmotic term (2e-2 strain above the 1e-2 cap).
    assert capped.osmotic_axial_m < uncapped.osmotic_axial_m
    # Even uncapped, the small eta keeps the net dilatation well below the vascular scale.
    assert uncapped.osmotic_axial_m < 0.5 * 1.5e-6


def test_mechanism_params_validate():
    with pytest.raises(ValueError):
        MechanismParams(osmotic_strain=-1.0)
    with pytest.raises(ValueError):
        MechanismParams(vascular_eta=1.5)
