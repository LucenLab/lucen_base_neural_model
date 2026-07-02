"""The inverse problem: how far spatial resolution can be pushed (resolution.py).

The forward gates ask "does (eta, s) pass at the baseline voxel?". ``min_resolution``
inverts that: holding (eta, s) fixed, how small can the voxel get -- how fine the
spatial resolution -- before a gate fails? These tests pin the model and the inverse:

* ``scaled_voxel`` shrinks isotropically at *fixed cell density* (``f_cell`` invariant,
  ``N = round(rho^3 * N0)``);
* the two chain terms move oppositely with ``rho`` (coherent ``~ rho``, pedestal
  ``~ rho^(-1/2)``) -- the squeeze that makes a finite resolution limit exist;
* the located limit sits exactly on the pass boundary (bracketing);
* min-N falls as either fulcrum (eta, s) rises -- a favorable corner buys finer
  resolution -- and the search reports infeasibility below the Gate-3 eta floor;
* the limiting gate is identified, and switches from the estimate-floor bar to the
  pedestal bar as the estimate floor is relaxed.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from base_neural_model import min_resolution, scaled_voxel
from base_neural_model.base.types import MechanicsParams
from base_neural_model.mechanics.transduction import mechanical_displacement
from base_neural_model.model.gates import DEFAULT_ETA_FLOOR
from base_neural_model.model.resolution import DEFAULT_MIN_NEURON_FLOOR

_FLOOR_M = 1e-9
_LOW_JITTER_S = 0.2e-3


@pytest.fixture
def d_single():
    """Fixed test-scale single-neuron displacement for the resolution MACHINERY tests.

    Pinned to the pre-revision 2 nm and decoupled from the honest cited constant
    (~0.4 nm), so the shrink/boundary/density-floor assertions run in a non-degenerate
    feasible window. The honest Delta r drives the demo verdict, not this machinery test.
    """
    from dataclasses import replace

    from base_neural_model import get_single_neuron_displacement

    return replace(get_single_neuron_displacement(), value_m=2e-9)


def _params(eta: float, *, d_single) -> MechanicsParams:
    return replace(
        MechanicsParams.central(),
        membrane_disp_m=d_single.value_m,
        dilatation_eta=eta,
        jitter_sigma_s=_LOW_JITTER_S,
    )


# --- 1. scaled_voxel: isotropic shrink at fixed cell density ---------------------


@pytest.mark.parametrize("rho", [1.0, 0.8, 0.5, 0.25, 0.1])
def test_scaled_voxel_preserves_cell_density(voxel, rho):
    """f_cell = N*V_cell/V_voxel is invariant under the shrink (fixed density).

    Invariance holds exactly in the continuum; the neuron count is rounded to an
    integer (you cannot have a fractional cell), so a small density drift is allowed
    -- 2% comfortably covers the rounding at these scales while still catching any
    genuine density error (which would be order-unity, not sub-percent).
    """
    r = 8e-6
    scaled = scaled_voxel(voxel, rho)
    assert scaled.cell_volume_fraction(r) == pytest.approx(
        voxel.cell_volume_fraction(r), rel=2e-2
    )


@pytest.mark.parametrize("rho", [0.8, 0.5, 0.25])
def test_scaled_voxel_scales_count_as_volume(voxel, rho):
    """Neuron count scales as rho^3 (volume); extents scale as rho (linear)."""
    scaled = scaled_voxel(voxel, rho)
    assert scaled.neuron_count == max(1, round(voxel.neuron_count * rho**3))
    assert scaled.extent_axial_m == pytest.approx(voxel.extent_axial_m * rho)
    assert scaled.extent_lateral_m == pytest.approx(voxel.extent_lateral_m * rho)
    assert scaled.depth_m == voxel.depth_m  # path length, not a voxel extent


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_scaled_voxel_rejects_out_of_range_rho(voxel, bad):
    with pytest.raises(ValueError):
        scaled_voxel(voxel, bad)


def test_chain_terms_move_oppositely_with_rho(d_single, voxel):
    """Coherent ~ rho; incoherent pedestal ~ rho^(-1/2): the resolution squeeze.

    This is *why* a finite resolution limit exists -- pushing finer lowers the signal
    and raises the source's own noise floor at the same time.
    """
    params = _params(0.5, d_single=d_single)
    s = 0.5
    rho_a, rho_b = 0.8, 0.2
    a = mechanical_displacement(d_single, scaled_voxel(voxel, rho_a), params, s)
    b = mechanical_displacement(d_single, scaled_voxel(voxel, rho_b), params, s)

    # Coherent signal scales linearly in rho.
    assert b.axial_displacement_m / a.axial_displacement_m == pytest.approx(
        rho_b / rho_a, rel=1e-6
    )
    # Pedestal scales as rho^(-1/2): smaller voxel -> larger pedestal.
    assert b.incoherent_pedestal_m > a.incoherent_pedestal_m
    assert b.incoherent_pedestal_m / a.incoherent_pedestal_m == pytest.approx(
        (rho_b / rho_a) ** -0.5, rel=1e-6
    )


# --- 2. min_resolution: locating the finest feasible resolution ------------------


def test_baseline_favorable_corner_can_push_below_baseline(d_single, voxel):
    """A favorable (eta, s) lets the voxel shrink well below the 10k baseline."""
    lim = min_resolution(d_single, voxel, _params(1.0, d_single=d_single), 1.0,
                         floor_m=_FLOOR_M)
    assert lim.feasible
    assert lim.min_scale_rho < 1.0
    assert lim.min_neuron_count < voxel.neuron_count
    assert lim.resolution_gain > 1.0  # 1 / rho
    # The gates would allow far finer here, so the density floor is what binds.
    assert lim.limiting_gate == "density_floor"
    assert lim.min_neuron_count == DEFAULT_MIN_NEURON_FLOOR


def test_min_resolution_infeasible_below_eta_floor(d_single, voxel):
    """Below the Gate-3 eta floor, no resolution passes -> infeasible, gate3."""
    dead = _params(DEFAULT_ETA_FLOOR / 2.0, d_single=d_single)
    lim = min_resolution(d_single, voxel, dead, 1.0, floor_m=_FLOOR_M)
    assert lim.feasible is False
    assert lim.limiting_gate == "gate3"
    assert lim.min_neuron_count == voxel.neuron_count  # reports baseline


def test_located_limit_sits_on_the_pass_boundary(d_single, voxel):
    """Just below the located rho fails; at and just above it passes (bracketing).

    Uses a (eta, s) where a forward gate -- not the density floor -- binds, so the
    pass boundary is a real gate crossing (at the density floor the gates still pass,
    which would break the 'just below fails' half of the bracket).
    """
    params = _params(0.5, d_single=d_single)
    s = 0.5
    lim = min_resolution(d_single, voxel, params, s, floor_m=_FLOOR_M)
    assert lim.feasible
    assert lim.limiting_gate != "density_floor"  # this corner is gate-bound

    def passes(rho: float) -> bool:
        from base_neural_model import (
            passes_content_survival_gate,
            passes_dilatation_gate,
            passes_stage1_gate,
        )

        sweep = [mechanical_displacement(d_single, scaled_voxel(voxel, rho), params, s)]
        return (
            passes_stage1_gate(sweep, unaberrated_floor_m=_FLOOR_M)
            and passes_content_survival_gate(sweep, estimate_floor_m=_FLOOR_M)
            and passes_dilatation_gate(sweep)
        )

    rho = lim.min_scale_rho
    assert passes(rho)
    assert not passes(rho * 0.9)             # a finer voxel fails
    assert passes(min(1.0, rho * 1.1))       # a coarser one still passes


# --- 3. Monotonicity: a better fulcrum buys finer resolution ---------------------


def test_finer_resolution_with_higher_eta(d_single, voxel):
    """min-N is non-increasing in eta: more dilatation -> can shrink further."""
    s = 0.5
    etas = [0.1, 0.25, 0.5, 0.75, 1.0]
    counts = [
        min_resolution(d_single, voxel, _params(e, d_single=d_single), s,
                       floor_m=_FLOOR_M).min_neuron_count
        for e in etas
    ]
    assert all(b <= a for a, b in zip(counts, counts[1:], strict=False)), counts
    assert counts[-1] < counts[0]


def test_finer_resolution_with_higher_synchrony(d_single, voxel):
    """min-N is non-increasing in synchrony at fixed eta."""
    eta = 0.5
    ss = np.linspace(0.3, 1.0, 8)
    counts = [
        min_resolution(d_single, voxel, _params(eta, d_single=d_single), float(s),
                       floor_m=_FLOOR_M).min_neuron_count
        for s in ss
    ]
    assert all(b <= a for a, b in zip(counts, counts[1:], strict=False)), counts
    assert counts[-1] < counts[0]


# --- 4. The limiting gate is identified and switches with the bars ---------------


def test_limiting_gate_switches_to_pedestal_when_estimate_floor_relaxed(d_single, voxel):
    """Tie the estimate floor to the amplitude floor -> estimate-floor bound;
    relax it far below -> the rho^(-1/2) pedestal becomes the binding constraint.

    This makes the "sqrt(N) noise eats the signal" regime reachable and labelled. The
    pedestal here binds at a handful of neurons, well below the density floor, so the
    floor is lowered to 1 to expose the gate under test (the default floor would mask
    it as "density_floor" -- which is the subject of its own test below).
    """
    params = _params(0.5, d_single=d_single)
    s = 0.5

    tied = min_resolution(d_single, voxel, params, s, floor_m=_FLOOR_M,
                          estimate_floor_m=_FLOOR_M)
    assert tied.limiting_gate == "gate2_estimate_floor"

    relaxed = min_resolution(d_single, voxel, params, s, floor_m=_FLOOR_M,
                             estimate_floor_m=1e-12, min_neuron_floor=1)
    assert relaxed.limiting_gate == "gate2_pedestal"
    # Relaxing the estimate floor can only allow an equal-or-finer resolution.
    assert relaxed.min_scale_rho <= tied.min_scale_rho + 1e-9


# --- 5. The mandatory minimum-density floor --------------------------------------


def test_density_floor_caps_resolution_when_gates_would_allow_finer(d_single, voxel):
    """The gate-limited finest voxel can fall below the density floor; it is capped.

    At a favorable corner with a relaxed estimate floor the gates would allow a few-
    neuron voxel, but the mandatory floor stops the shrink at DEFAULT_MIN_NEURON_FLOOR
    and tags ``density_floor``. The reported limit is the binding one of the two.
    """
    lim = min_resolution(d_single, voxel, _params(1.0, d_single=d_single), 1.0,
                         floor_m=_FLOOR_M, estimate_floor_m=1e-12)
    assert lim.feasible
    assert lim.limiting_gate == "density_floor"
    assert lim.min_neuron_count == DEFAULT_MIN_NEURON_FLOOR


def test_density_floor_is_never_undershot(d_single, voxel):
    """No (eta, s) ever returns a feasible voxel below the density floor."""
    for eta in (0.2, 0.5, 1.0):
        for s in (0.3, 0.6, 1.0):
            lim = min_resolution(d_single, voxel, _params(eta, d_single=d_single),
                                 s, floor_m=_FLOOR_M)
            if lim.feasible:
                assert lim.min_neuron_count >= DEFAULT_MIN_NEURON_FLOOR


def test_raising_the_density_floor_only_coarsens_the_limit(d_single, voxel):
    """A higher density floor can only raise (never lower) the reported min-N."""
    params = _params(1.0, d_single=d_single)
    s = 1.0
    low = min_resolution(d_single, voxel, params, s, floor_m=_FLOOR_M,
                         min_neuron_floor=100)
    high = min_resolution(d_single, voxel, params, s, floor_m=_FLOOR_M,
                          min_neuron_floor=2_000)
    assert high.min_neuron_count >= low.min_neuron_count
    assert high.min_neuron_count == 2_000  # gates allow finer -> floor binds at 2000


@pytest.mark.parametrize("bad", [0, -5])
def test_min_neuron_floor_rejects_nonpositive(d_single, voxel, bad):
    with pytest.raises(ValueError):
        min_resolution(d_single, voxel, _params(0.5, d_single=d_single), 0.5,
                       floor_m=_FLOOR_M, min_neuron_floor=bad)


def test_min_neuron_floor_above_baseline_is_rejected(d_single, voxel):
    """A floor above the baseline count is contradictory (shrink yet grow)."""
    with pytest.raises(ValueError):
        min_resolution(d_single, voxel, _params(0.5, d_single=d_single), 0.5,
                       floor_m=_FLOOR_M, min_neuron_floor=voxel.neuron_count + 1)


def test_provenance_records_the_inverse_assumptions(d_single, voxel):
    """The ResolutionLimit carries its model assumptions (Invariant 6)."""
    lim = min_resolution(d_single, voxel, _params(0.5, d_single=d_single), 0.5,
                         floor_m=_FLOOR_M)
    joined = " ".join(lim.provenance.assumptions)
    assert "fixed cell density" in joined
    assert "three-gate" in joined
