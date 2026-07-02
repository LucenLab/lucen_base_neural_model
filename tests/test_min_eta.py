"""The minimum detectable dilatation fraction eta* (base_neural_model.model.min_eta).

eta* is the smallest dilatation fraction that clears all three gates at a synchrony --
the number the Stage-1 bench experiment must beat. These tests pin the inverse and the
closed-form structure it rests on:

* the source term is linear in eta (so the bisection is valid and monotone);
* the located eta* sits exactly on the pass boundary (bracketing);
* eta* falls as synchrony rises -- more coherence lowers the dilatation needed;
* the binding constraint is identified: the hard Gate-3 floor, a linear amplitude gate
  above it, or infeasible (the eta-independent pedestal bar is unmet at this synchrony);
* the eta-independent pedestal bar genuinely makes low-synchrony points infeasible.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from base_neural_model import min_detectable_eta
from base_neural_model.base.types import MechanicsParams, VoxelGeometry
from base_neural_model.mechanics.transduction import mechanical_displacement
from base_neural_model.model.gates import DEFAULT_ETA_FLOOR

_FLOOR_M = 1e-9
_LOW_JITTER_S = 0.2e-3


@pytest.fixture
def d_single():
    """Fixed test-scale single-neuron displacement for the eta* MACHINERY tests.

    Pinned to the pre-revision 2 nm and decoupled from the honest cited constant
    (~0.4 nm), so eta*(s) monotonicity, boundary-bracketing and infeasibility regimes
    run in a non-degenerate feasible window. The honest Delta r drives the demo verdict.
    """
    from dataclasses import replace

    from base_neural_model import get_single_neuron_displacement

    return replace(get_single_neuron_displacement(), value_m=2e-9)


def _params(*, d_single, **overrides) -> MechanicsParams:
    base = replace(
        MechanicsParams.central(),
        membrane_disp_m=d_single.value_m,
        jitter_sigma_s=_LOW_JITTER_S,
    )
    return replace(base, **overrides) if overrides else base


# --- The closed-form premise: Delta z linear in eta ------------------------------


def test_source_term_is_linear_in_eta(d_single, voxel):
    """Delta z / eta is constant -- the inverse's monotonicity premise."""
    params = _params(d_single=d_single)
    s = 0.5
    ratios = []
    for eta in (0.1, 0.3, 0.7, 1.0):
        dz = mechanical_displacement(d_single, voxel, replace(params, dilatation_eta=eta),
                                 s).axial_displacement_m
        ratios.append(dz / eta)
    assert all(r == pytest.approx(ratios[0], rel=1e-9) for r in ratios)


# --- Locating eta* ----------------------------------------------------------------


def test_eta_star_on_the_pass_boundary(d_single, voxel):
    """Just below eta* fails; at and just above it passes (bracketing)."""
    params = _params(d_single=d_single)
    s = 0.4
    res = min_detectable_eta(d_single, voxel, params, s, floor_m=_FLOOR_M)
    assert res.feasible and res.binding_constraint == "amplitude"

    def passes(eta: float) -> bool:
        from base_neural_model import (
            passes_content_survival_gate,
            passes_dilatation_gate,
            passes_stage1_gate,
        )

        sweep = [mechanical_displacement(d_single, voxel,
                                     replace(params, dilatation_eta=eta), s)]
        return (
            passes_stage1_gate(sweep, unaberrated_floor_m=_FLOOR_M)
            and passes_content_survival_gate(sweep, estimate_floor_m=_FLOOR_M)
            and passes_dilatation_gate(sweep)
        )

    e = res.eta_star
    assert passes(e)
    assert not passes(e * 0.95)            # less dilatation fails
    assert passes(min(1.0, e * 1.05))      # more dilatation passes


def test_eta_star_decreases_with_synchrony(d_single, voxel):
    """More synchrony lowers the dilatation needed: eta*(s) is non-increasing."""
    params = _params(d_single=d_single)
    ss = [0.1, 0.2, 0.4, 0.7, 1.0]
    stars = [
        min_detectable_eta(d_single, voxel, params, s, floor_m=_FLOOR_M).eta_star
        for s in ss
    ]
    assert all(e is not None for e in stars), stars
    assert all(b <= a + 1e-9 for a, b in zip(stars, stars[1:], strict=False)), stars
    assert stars[-1] < stars[0]


def test_eta_star_is_eta_independent_of_caller_value(d_single, voxel):
    """The caller's params.dilatation_eta is overridden -- result is invariant to it."""
    s = 0.4
    lo_p = _params(d_single=d_single, dilatation_eta=0.01)
    hi_p = _params(d_single=d_single, dilatation_eta=0.99)
    a = min_detectable_eta(d_single, voxel, lo_p, s, floor_m=_FLOOR_M)
    b = min_detectable_eta(d_single, voxel, hi_p, s, floor_m=_FLOOR_M)
    assert a.eta_star == pytest.approx(b.eta_star)


# --- The binding constraint -------------------------------------------------------


def test_gate3_floor_binds_when_amplitude_clears_at_the_floor(d_single):
    """A favorable voxel/material clears amplitude at eta_floor -> Gate 3 binds."""
    big_voxel = VoxelGeometry(extent_axial_m=1e-3, extent_lateral_m=1e-3,
                              neuron_count=50_000, depth_m=2e-2)
    favorable = MechanicsParams(
        membrane_disp_m=d_single.value_m, cell_radius_m=6e-6, cell_volume_fraction=0.25,
        confinement_kappa=1.0, dilatation_eta=0.5, jitter_sigma_s=_LOW_JITTER_S,
        content_freq_hz=100.0,
    )
    res = min_detectable_eta(d_single, big_voxel, favorable, 1.0, floor_m=_FLOOR_M)
    assert res.feasible
    assert res.binding_constraint == "gate3_floor"
    assert res.eta_star == pytest.approx(DEFAULT_ETA_FLOOR)


def test_infeasible_at_low_synchrony_from_pedestal_bar(d_single, voxel):
    """Low synchrony: the eta-independent pedestal bar is unmet -> no eta passes."""
    params = _params(d_single=d_single)
    res = min_detectable_eta(d_single, voxel, params, 0.02, floor_m=_FLOOR_M)
    assert res.feasible is False
    assert res.eta_star is None
    assert res.binding_constraint == "infeasible"


def test_relaxed_floor_makes_gate3_bind(d_single, voxel):
    """A lenient amplitude floor lets eta_floor clear the gates -> Gate 3 binds."""
    res = min_detectable_eta(d_single, voxel, _params(d_single=d_single), 1.0,
                             floor_m=1e-10)
    assert res.binding_constraint == "gate3_floor"
    assert res.eta_star == pytest.approx(DEFAULT_ETA_FLOOR)


# --- Guard rails and provenance ---------------------------------------------------


@pytest.mark.parametrize("bad_s", [-0.1, 1.5])
def test_rejects_synchrony_out_of_range(d_single, voxel, bad_s):
    with pytest.raises(ValueError):
        min_detectable_eta(d_single, voxel, _params(d_single=d_single), bad_s,
                           floor_m=_FLOOR_M)


@pytest.mark.parametrize("bad_floor", [0.0, 1.5])
def test_rejects_eta_floor_out_of_range(d_single, voxel, bad_floor):
    with pytest.raises(ValueError):
        min_detectable_eta(d_single, voxel, _params(d_single=d_single), 0.5,
                           floor_m=_FLOOR_M, eta_floor=bad_floor)


def test_provenance_records_the_inverse(d_single, voxel):
    res = min_detectable_eta(d_single, voxel, _params(d_single=d_single), 0.4,
                             floor_m=_FLOOR_M)
    joined = " ".join(res.provenance.assumptions)
    assert "min detectable eta" in joined
    assert "linear in eta" in joined
