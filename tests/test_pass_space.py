"""The (eta, s) measurable pass-space -- reverse-engineering the feasibility region.

The Sobol and verdict-flip layers prove *which* factors decide the Module-1 verdict:
the two fulcra are eta (net-dilatation fraction) and content-band synchrony s, with
geometry and the cited constant pinned out. This suite takes that result and maps the
*region*: for which (eta, s) does the source term clear all three kill gates, and how
does the margin climb through the four named confidence tiers ::

    pessimistic  ->  central  ->  optimistic  ->  unarguable

The object under test is the (eta, s) plane. ``s`` is the swept synchrony axis and
``eta`` is swept directly via ``replace(MechanicsParams.central(), dilatation_eta=...)``
(a clean axis -- routing through ``log10_permeability`` only matters when permeability
itself is the variable). A point PASSES iff it clears all three gates of
``base_neural_model.model.gates`` -- the fullest honest neural-model verdict:

* Gate 1 (amplitude): coherent ``Delta z`` within reach of the unaberrated floor;
* Gate 2 (content survival): jitter-surviving ``Delta z`` above its incoherent
  pedestal and above the estimate floor;
* Gate 3 (dilatation): ``eta`` at or above ``eta_floor``.

The chain is monotone up in both fulcra, so the pass-set is an upper-right region with
a single decreasing crossover frontier ``s*(eta)``; N (neuron count) is *not* an axis
(the coherent term is N-independent) and is shown here to be a control, not a fulcrum.

All numbers are derived from the existing chain -- this suite adds no production code.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from base_neural_model import (
    displacement_sweep,
    mechanical_displacement,
    passes_content_survival_gate,
    passes_dilatation_gate,
    passes_stage1_gate,
)
from base_neural_model.base.bands import Band
from base_neural_model.base.provenance import Provenance
from base_neural_model.base.types import (
    MechanicsParams,
    NeuronDisplacement,
    VoxelGeometry,
)
from base_neural_model.model.gates import DEFAULT_ETA_FLOOR

# A fixed unaberrated floor and matching estimate floor for the plane (metres).
_FLOOR_M = 1e-9
# Low jitter so Gate 2 is genuinely alive across the plane (not auto-extinguished);
# the jitter low-pass is exercised separately in test_envelope_gates.py.
_LOW_JITTER_S = 0.2e-3


def _passes_all_gates(
    sweep: list,
    *,
    floor_m: float = _FLOOR_M,
    estimate_floor_m: float = _FLOOR_M,
    reach_orders: float = 2.0,
) -> bool:
    """The shared three-gate verdict: a sweep PASSES iff all three gates pass.

    This is the pass criterion the whole suite maps. ``passes_*`` each scan the
    sweep for a satisfying synchrony, so on a single-point sweep this is the verdict
    *at that (eta, s) cell*, and on a full grid it is the "passes somewhere" verdict.
    """
    return (
        passes_stage1_gate(sweep, unaberrated_floor_m=floor_m, reach_orders=reach_orders)
        and passes_content_survival_gate(sweep, estimate_floor_m=estimate_floor_m)
        and passes_dilatation_gate(sweep)
    )


def _params_at(
    eta: float, *, jitter_s: float = _LOW_JITTER_S, d_single=None
) -> MechanicsParams:
    """Central-column params with the dilatation fraction set to ``eta``."""
    central = MechanicsParams.central()
    membrane = d_single.value_m if d_single is not None else central.membrane_disp_m
    return replace(
        MechanicsParams.central(),
        membrane_disp_m=membrane,
        dilatation_eta=eta,
        jitter_sigma_s=jitter_s,
    )


def _cell_passes(d_single, voxel, eta: float, s: float, **kw) -> bool:
    """Three-gate verdict at a single (eta, s) point (single-element sweep)."""
    params = _params_at(eta, d_single=d_single)
    sweep = displacement_sweep(d_single, voxel, params, np.array([s]))
    return _passes_all_gates(sweep, **kw)


# --- 1. The pass-set is an upper-right monotone region ---------------------------


def test_pass_set_is_upper_right_monotone_region(d_single, voxel):
    """If (s, eta) passes, every (s' >= s, eta' >= eta) on the grid passes too.

    The shape of the measurable space, locked: the chain is monotone up in both
    fulcra (and Gate 3 is monotone up in eta), so the pass-set has no holes and no
    re-entrant boundary -- it is a single upper-right region.
    """
    s_grid = np.linspace(0.0, 1.0, 21)
    eta_grid = np.linspace(0.0, 1.0, 21)

    # passes[i, j] for eta_grid[i], s_grid[j].
    passes = np.array(
        [[_cell_passes(d_single, voxel, eta, s) for s in s_grid] for eta in eta_grid]
    )

    # At least one pass and one fail, or the test proves nothing.
    assert passes.any() and not passes.all()

    n_eta, n_s = passes.shape
    for i in range(n_eta):
        for j in range(n_s):
            if passes[i, j]:
                # Everything up (higher eta) and right (higher s) must also pass.
                assert passes[i:, j:].all(), (
                    f"pass at (eta={eta_grid[i]:.3f}, s={s_grid[j]:.3f}) but a "
                    "higher-eta/higher-s cell fails -> non-monotone pass region"
                )


def test_pass_frontier_is_a_clean_threshold_per_eta(d_single, voxel):
    """Along each eta row, the pass region is a single suffix in s (one crossing).

    Equivalent to: once a row starts passing as s increases, it never stops. A clean
    crossover frontier s*(eta) exists row by row.
    """
    s_grid = np.linspace(0.0, 1.0, 41)
    for eta in (0.1, 0.25, 0.5, 1.0):
        row = np.array([_cell_passes(d_single, voxel, eta, s) for s in s_grid])
        if not row.any():
            continue
        first_pass = int(np.argmax(row))
        # No failure may appear at or after the first pass.
        assert row[first_pass:].all(), f"row eta={eta} has a re-entrant boundary"


# --- 2. The crossover synchrony s*(eta) is monotone decreasing -------------------


def _crossover_s(d_single, voxel, eta: float, s_grid: np.ndarray) -> float | None:
    """Smallest s on the grid whose (eta, s) cell passes all three gates."""
    for s in s_grid:
        if _cell_passes(d_single, voxel, eta, float(s)):
            return float(s)
    return None


def test_crossover_synchrony_decreases_with_eta(d_single, voxel):
    """More dilatation -> less synchrony needed: s*(eta) is non-increasing."""
    s_grid = np.linspace(0.0, 1.0, 201)
    etas = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    crossovers = [_crossover_s(d_single, voxel, e, s_grid) for e in etas]

    # Every one of these eta values admits a crossover within [0, 1].
    assert all(c is not None for c in crossovers), crossovers
    # Strictly the doc's claim: the frontier falls as eta rises.
    assert all(
        b <= a + 1e-9 for a, b in zip(crossovers, crossovers[1:], strict=False)
    ), f"s*(eta) not monotone decreasing: {list(zip(etas, crossovers, strict=True))}"
    # And it genuinely moves (not a flat line): high eta needs far less s than low eta.
    assert crossovers[-1] < crossovers[0]


def test_measurable_space_has_an_eta_floor_at_gate3(d_single, voxel):
    """Below eta_floor nothing passes at any s -- the floor of the measurable space.

    Gate 3 vetoes outright for eta < eta_floor, so the space is closed from below at
    an eta no smaller than DEFAULT_ETA_FLOOR regardless of how high synchrony goes.
    """
    s_grid = np.linspace(0.0, 1.0, 201)

    # At/below the Gate-3 floor: no synchrony rescues it.
    assert _crossover_s(d_single, voxel, DEFAULT_ETA_FLOOR, s_grid) is None
    assert _crossover_s(d_single, voxel, DEFAULT_ETA_FLOOR / 2.0, s_grid) is None

    # Somewhere above the floor a pass becomes reachable (the space is non-empty).
    assert _crossover_s(d_single, voxel, 1.0, s_grid) is not None

    # The lowest eta that admits any pass is at least the Gate-3 floor.
    first_feasible = next(
        e for e in np.linspace(0.0, 1.0, 101)
        if _crossover_s(d_single, voxel, float(e), s_grid) is not None
    )
    assert first_feasible >= DEFAULT_ETA_FLOOR - 1e-9


# --- 3. The four confidence tiers: pessimistic -> ... -> unarguable --------------

# Section-5 envelope geometry (300 um gate), shared across the tier tests.
_TIER_GATE_M = 300e-6


def _tier_voxel() -> VoxelGeometry:
    return VoxelGeometry(
        extent_axial_m=_TIER_GATE_M,
        extent_lateral_m=1e-3,
        neuron_count=20_000,
        depth_m=2e-2,
    )


def _tier_neuron(params: MechanicsParams) -> NeuronDisplacement:
    prov = Provenance(
        source="pass-space tier column",
        assumptions=("cited Delta r for this tier",),
        band=Band.CONTENT_FAST,
    )
    return NeuronDisplacement(
        value_m=params.membrane_disp_m, band=Band.CONTENT_FAST, provenance=prov
    )


# (name, params, synchrony) per tier. Pessimistic/central/optimistic reuse the
# section-5 columns (test_envelope_gates.py); "unarguable" pushes both fulcra and the
# material to the favorable corner so Delta z sits ~2 orders above the 1 nm floor.
_PESSIMISTIC = MechanicsParams(
    membrane_disp_m=1.0e-9, cell_radius_m=10e-6, cell_volume_fraction=0.10,
    confinement_kappa=1.0 / 3.0, dilatation_eta=0.1, jitter_sigma_s=3.0e-3,
    content_freq_hz=100.0,
)
_CENTRAL = MechanicsParams(
    membrane_disp_m=1.5e-9, cell_radius_m=8e-6, cell_volume_fraction=0.15,
    confinement_kappa=0.5, dilatation_eta=0.5, jitter_sigma_s=1.0e-3,
    content_freq_hz=100.0,
)
_OPTIMISTIC = MechanicsParams(
    membrane_disp_m=3.0e-9, cell_radius_m=6e-6, cell_volume_fraction=0.25,
    confinement_kappa=1.0, dilatation_eta=1.0, jitter_sigma_s=0.5e-3,
    content_freq_hz=100.0,
)
_UNARGUABLE = MechanicsParams(
    membrane_disp_m=3.0e-9, cell_radius_m=6e-6, cell_volume_fraction=0.25,
    confinement_kappa=1.0, dilatation_eta=1.0, jitter_sigma_s=0.3e-3,
    content_freq_hz=100.0,
)
_TIERS = (
    ("pessimistic", _PESSIMISTIC, 0.1),
    ("central", _CENTRAL, 0.4),
    ("optimistic", _OPTIMISTIC, 0.8),
    ("unarguable", _UNARGUABLE, 1.0),
)


def _tier_margin(params: MechanicsParams, s: float) -> float:
    """Coherent axial displacement at the tier corner, as a multiple of the floor."""
    r = mechanical_displacement(_tier_neuron(params), _tier_voxel(), params, s)
    return r.axial_displacement_m / _FLOOR_M


def test_tier_margins_form_a_strict_ladder():
    """pessimistic < central < optimistic < unarguable, in margin above the floor.

    This is the literal "ranges from pessimistic to central to optimistic to
    unarguable proof" -- each tier sits strictly further above the detectability
    floor than the last.
    """
    margins = {name: _tier_margin(p, s) for name, p, s in _TIERS}
    assert margins["pessimistic"] < margins["central"]
    assert margins["central"] < margins["optimistic"]
    assert margins["optimistic"] < margins["unarguable"]
    # Anchor the endpoints to the section-5 worked numbers (sub-floor vs ~2 orders up).
    assert margins["pessimistic"] < 0.1          # ~0.03x floor: well under
    assert margins["unarguable"] > 100.0         # ~2+ orders above floor


def test_tier_gate_verdicts_climb_from_fail_to_unarguable():
    """The named tiers map onto the gate verdict: fail -> pass -> pass-at-a-hard-bar.

    * pessimistic fails the three-gate verdict even at the generous default reach;
    * central and optimistic pass at the nominal floor;
    * a *raised* floor (8 nm) that central no longer clears is still cleared by
      optimistic and unarguable -- "unarguable" proof survives a demanding bar.

    The verdict is the gates' own "passes at some synchrony" semantics: each tier
    sweep is the full s grid, so a tier passes if any synchrony clears the bar (its
    grid peak surviving displacement is what the hard floor is compared to).
    """
    grid = np.linspace(0.0, 1.0, 21)
    sweeps = {
        name: displacement_sweep(_tier_neuron(p), _tier_voxel(), p, grid)
        for name, p, _ in _TIERS
    }

    # Pessimistic dies; the rest pass at the nominal floor.
    assert not _passes_all_gates(sweeps["pessimistic"])
    assert _passes_all_gates(sweeps["central"])
    assert _passes_all_gates(sweeps["optimistic"])
    assert _passes_all_gates(sweeps["unarguable"])

    # A demanding floor at a tight bar separates the merely-feasible from the
    # unarguable: central's grid peak (~5 nm surviving) drops out, while optimistic
    # and unarguable (~100+ nm) clear an 8 nm floor with room to spare.
    hard = dict(floor_m=8e-9, estimate_floor_m=8e-9, reach_orders=0.0)
    assert not _passes_all_gates(sweeps["central"], **hard)
    assert _passes_all_gates(sweeps["optimistic"], **hard)
    assert _passes_all_gates(sweeps["unarguable"], **hard)


# --- 4. Tier <-> (eta, s) localization -------------------------------------------


def test_tiers_localize_in_the_eta_s_plane(d_single, voxel):
    """Each named tier lands where the plane (sect.1) says it should.

    Re-evaluated on the common (central-material) plane used for the map so the
    comparison is apples-to-apples: the tier's own (eta, s) coordinate is fed in and
    checked against the same three-gate verdict.
    """
    # pessimistic eta is at/below the Gate-3 floor region -> outside the pass set
    # even at full synchrony.
    assert not _cell_passes(d_single, voxel, eta=0.05, s=1.0)

    # central (eta=0.5, s=0.4) sits inside the pass set.
    assert _cell_passes(d_single, voxel, eta=0.5, s=0.4)

    # optimistic / unarguable (eta=1.0, high s) sit inside, and stay inside when the
    # floor is raised 3x -- "deep inside" the measurable space.
    deep = dict(floor_m=3e-9, estimate_floor_m=3e-9, reach_orders=0.0)
    assert _cell_passes(d_single, voxel, eta=1.0, s=0.8)
    assert _cell_passes(d_single, voxel, eta=1.0, s=0.8, **deep)
    assert _cell_passes(d_single, voxel, eta=1.0, s=1.0, **deep)


# --- 5. Both fulcra are necessary: the space is closed on both axes --------------


def test_dilatation_veto_closes_the_low_eta_edge(d_single, voxel):
    """Drop eta below the floor with synchrony pinned high: Gate 3 vetoes regardless."""
    params = _params_at(DEFAULT_ETA_FLOOR / 2.0, d_single=d_single)
    sweep = displacement_sweep(d_single, voxel, params, np.array([1.0]))  # s = 1, maximal
    assert not passes_dilatation_gate(sweep)
    assert not _passes_all_gates(sweep)


def test_synchrony_veto_closes_the_low_s_edge(d_single, voxel):
    """Drop s -> 0 with eta pinned high: the coherent term vanishes, Gate 1/2 fail."""
    params = _params_at(1.0, d_single=d_single)  # maximal dilatation
    at_zero = mechanical_displacement(d_single, voxel, params, 0.0)
    assert at_zero.axial_displacement_m == 0.0          # coherent term gone
    assert at_zero.incoherent_pedestal_m > 0.0          # only the noise pedestal left

    sweep = [at_zero]
    assert not passes_stage1_gate(sweep, unaberrated_floor_m=_FLOOR_M)
    assert not passes_content_survival_gate(sweep, estimate_floor_m=_FLOOR_M)
    assert not _passes_all_gates(sweep)


def test_neither_fulcrum_alone_suffices(d_single, voxel):
    """High eta with zero s fails, and high s with sub-floor eta fails: both needed."""
    # high eta, zero synchrony
    assert not _cell_passes(d_single, voxel, eta=1.0, s=0.0)
    # high synchrony, sub-floor eta
    assert not _cell_passes(d_single, voxel, eta=DEFAULT_ETA_FLOOR / 2.0, s=1.0)
    # but both together pass -> the corner is genuinely a joint requirement
    assert _cell_passes(d_single, voxel, eta=1.0, s=1.0)


# --- 6. N is a control, not a fulcrum (why it isn't an axis) ---------------------


def test_neuron_count_does_not_move_the_gate1_verdict(d_single, voxel):
    """The coherent Delta z is N-independent: the Gate-1 axis ignores neuron count.

    This is *why* the measurable space is (eta, s) and not (N, s) -- the amplitude
    fulcrum does not respond to N at all.
    """
    params = _params_at(0.5, d_single=d_single)
    s = 0.6
    small = mechanical_displacement(d_single, voxel, params, s)
    big_voxel = replace(voxel, neuron_count=voxel.neuron_count * 100)
    big = mechanical_displacement(d_single, big_voxel, params, s)

    coherent = small.axial_displacement_m
    assert big.axial_displacement_m == pytest.approx(coherent, rel=1e-12)


def test_neuron_count_only_shifts_the_gate2_pedestal(d_single, voxel):
    """More neurons lower the incoherent pedestal (1/sqrt(N)), easing Gate 2 only.

    N enters solely through the pedestal that Gate 2 must clear; raising N strictly
    lowers it (cleaner source), enlarging the Gate-2 pass region without touching the
    Gate-1 amplitude. Confirms N's role is a control on the noise floor, not a fulcrum.
    """
    params = _params_at(0.5, d_single=d_single)
    s = 0.3
    small = mechanical_displacement(d_single, voxel, params, s)
    big_voxel = replace(voxel, neuron_count=voxel.neuron_count * 4)
    big = mechanical_displacement(d_single, big_voxel, params, s)

    # Pedestal falls as 1/sqrt(N): 4x neurons -> half the pedestal.
    assert big.incoherent_pedestal_m < small.incoherent_pedestal_m
    assert big.incoherent_pedestal_m == pytest.approx(
        small.incoherent_pedestal_m / 2.0, rel=1e-9
    )
    # ... while the coherent signal both gates' amplitude depends on is unchanged.
    coherent = small.axial_displacement_m
    assert big.axial_displacement_m == pytest.approx(coherent, rel=1e-12)
