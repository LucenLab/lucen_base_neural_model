"""Inverse problem: how far can spatial resolution be pushed and still pass?

The forward question Module 1 answers is "at the baseline voxel, does (eta, s) clear
the three gates?". This module asks the reverse: holding (eta, s) and the cited
constants fixed, **how small can the voxel get -- how fine the spatial resolution --
before the source term fails a gate?**

Resolution is modelled as an isotropic shrink of the voxel at *fixed cell density*
(source physics doc voxel; ``f_cell = N * V_cell / V_voxel`` is held constant). A
linear scale ``rho in (0, 1]`` shrinks both in-plane extents and the range gate::

    L = rho * L0,  W = rho * W0,  V = rho^3 * V0,  N = round(rho^3 * N0)

so ``rho = (N / N0)^(1/3)`` and the neuron count is the natural readout. Two terms of
the transduction chain move in opposite directions as ``rho`` falls (resolution rises):

* coherent signal ``Delta z = eta*kappa*L*f_cell*s*(3 dr/r) ~ rho``  -- falls linearly;
* incoherent pedestal ``Delta z_incoh ~ L/sqrt(N) ~ rho^(-1/2)``     -- rises.

Pushing resolution therefore lowers the signal *and* raises the source's own noise
floor. Each gate becomes a monotone constraint in ``rho`` (Gate 1 and the Gate-2
estimate-floor bar are linear floors; the Gate-2 pedestal bar is a ``rho^(3/2)`` floor;
Gate 3 is ``rho``-independent and acts as a precondition). The passing set is an upper
interval ``[rho_min, 1]``, so the finest feasible resolution is found by one monotone
bisection -- evaluated against the real :func:`mechanical_displacement` chain, not the
closed form, so it cannot drift from the model.
"""

from __future__ import annotations

from dataclasses import dataclass

from base_neural_model.base.provenance import Provenance, extend
from base_neural_model.base.types import (
    MechanicsParams,
    NeuronDisplacement,
    VoxelGeometry,
)
from base_neural_model.mechanics.transduction import mechanical_displacement
from base_neural_model.model.gates import (
    DEFAULT_ETA_FLOOR,
    DEFAULT_REACH_ORDERS,
    passes_content_survival_gate,
    passes_dilatation_gate,
    passes_stage1_gate,
)

# Minimum neuron count per voxel for the model's population statistics to hold. Below
# roughly this many cells the incoherent pedestal's 1/sqrt(N) scaling (a central-limit
# statement needing many independent firers) and the continuum treatment of f_cell / the
# homogenized strain stop being faithful. The gate-limited finest voxel (pure
# amplitude/noise arithmetic) can fall far below this -- a 2-neuron "voxel" -- so this
# floor is imposed unconditionally: the reported limit is the BINDING one of the gate
# limit and this floor. It does not improve the signal; it keeps the answer inside the
# regime where the forward model holds, hence conservative and buildable.
#
# This is a FLAT count, deliberately not a density-derived floor: under the fixed-density
# shrink (N = density * V, held constant), a density floor scales as rho^3 in lockstep
# with the realized count and therefore never binds -- it would be a no-op. A flat count
# binds precisely because it does not shrink with the voxel. The genuinely *physical*
# resolution limit is acoustic, not cellular (ultrasound lambda/2 ~= 385 um at 2 MHz
# already exceeds the 300 um axial gate), and belongs in Module 2; this floor is the
# Module-1 statistical-validity guardrail only.
DEFAULT_MIN_NEURON_FLOOR: int = 100

# Which constraint pins the resolution limit. "none" => gates never bind down to the
# density floor; "gate1_amplitude"/"gate2_*" => a forward gate binds; "gate3" => eta
# below floor, infeasible at any resolution; "density_floor" => the gates would allow a
# finer voxel, but the minimum-density floor is what stops the shrink.
LimitingGate = str


@dataclass(frozen=True)
class ResolutionLimit:
    """The finest spatial resolution that still passes all three gates.

    ``min_scale_rho`` is the smallest linear voxel-shrink factor in ``[rho_lo, 1]``
    whose source term clears every gate; ``min_neuron_count`` is the neuron count at
    that scale (``round(rho^3 * N0)``) -- the headline "how few neurons" number.
    ``voxel`` is the shrunk geometry. ``limiting_gate`` names the gate that stops any
    further shrink. ``feasible`` is ``False`` when even the baseline (``rho = 1``)
    fails, in which case the other fields describe the baseline, not a real limit.
    """

    min_scale_rho: float
    min_neuron_count: int
    voxel: VoxelGeometry
    limiting_gate: LimitingGate
    feasible: bool
    baseline_neuron_count: int
    provenance: Provenance

    @property
    def resolution_gain(self) -> float:
        """Linear resolution improvement over baseline (= 1 / rho >= 1)."""
        return 1.0 / self.min_scale_rho

    @property
    def voxel_edge_reduction(self) -> float:
        """Voxel edge as a fraction of baseline (= rho <= 1)."""
        return self.min_scale_rho


def scaled_voxel(base: VoxelGeometry, rho: float) -> VoxelGeometry:
    """Isotropically shrink ``base`` by linear factor ``rho`` at fixed cell density.

    Both voxel extents scale by ``rho``; the neuron count scales by ``rho^3`` (volume)
    so the implied ``f_cell = N * V_cell / V_voxel`` is invariant. ``depth_m`` is the
    propagation path length, not a voxel extent, so it is left unchanged (matching
    :attr:`VoxelGeometry.volume_m3`, which deliberately ignores it).
    """
    if not 0.0 < rho <= 1.0:
        raise ValueError(f"rho must lie in (0, 1], got {rho!r}")
    return VoxelGeometry(
        extent_axial_m=base.extent_axial_m * rho,
        extent_lateral_m=base.extent_lateral_m * rho,
        neuron_count=max(1, round(base.neuron_count * rho**3)),
        depth_m=base.depth_m,
    )


def _passes_all(
    d_single: NeuronDisplacement,
    voxel: VoxelGeometry,
    params: MechanicsParams,
    s: float,
    *,
    floor_m: float,
    estimate_floor_m: float,
    reach_orders: float,
) -> bool:
    """Three-gate verdict at one (voxel, s) point (single-element sweep)."""
    sweep = [mechanical_displacement(d_single, voxel, params, s)]
    return (
        passes_stage1_gate(sweep, unaberrated_floor_m=floor_m, reach_orders=reach_orders)
        and passes_content_survival_gate(sweep, estimate_floor_m=estimate_floor_m)
        and passes_dilatation_gate(sweep)
    )


def _limiting_gate(
    d_single: NeuronDisplacement,
    voxel: VoxelGeometry,
    params: MechanicsParams,
    s: float,
    *,
    floor_m: float,
    estimate_floor_m: float,
    reach_orders: float,
) -> LimitingGate:
    """Name the gate that is tightest (closest to failing) at this voxel.

    Evaluated at the located limit, this is the gate a further shrink would break
    first: the one whose single-point margin is smallest, normalized within each
    gate's own units.
    """
    out = mechanical_displacement(d_single, voxel, params, s)
    surviving = out.axial_displacement_m * out.content_band_survival

    # Slack in each gate's own terms (>= 1 means passing with that much headroom).
    inf = float("inf")
    g1_threshold = floor_m / (10.0**reach_orders)
    g1_slack = out.axial_displacement_m / g1_threshold if g1_threshold > 0 else inf
    g2_floor_slack = surviving / estimate_floor_m if estimate_floor_m > 0 else inf
    ped = out.incoherent_pedestal_m
    g2_ped_slack = surviving / ped if ped > 0 else inf

    candidates = {
        "gate1_amplitude": g1_slack,
        "gate2_estimate_floor": g2_floor_slack,
        "gate2_pedestal": g2_ped_slack,
    }
    return min(candidates, key=candidates.get)


def min_resolution(
    d_single: NeuronDisplacement,
    base_voxel: VoxelGeometry,
    params: MechanicsParams,
    synchrony_fraction: float,
    *,
    floor_m: float,
    estimate_floor_m: float | None = None,
    reach_orders: float = DEFAULT_REACH_ORDERS,
    min_neuron_floor: int = DEFAULT_MIN_NEURON_FLOOR,
    rho_lo: float = 1e-3,
    tol: float = 1e-4,
    max_iter: int = 100,
) -> ResolutionLimit:
    """Finest spatial resolution (smallest voxel / fewest neurons) that still passes.

    Bisects the linear shrink factor ``rho`` for the smallest value whose source term
    clears all three gates at this ``(eta, s)``. The passing set is an upper interval
    in ``rho`` (the chain is monotone up in ``rho`` for the signal and the pedestal
    worsens as ``rho`` falls), so a single bisection locates the edge.

    A minimum-count floor of ``min_neuron_floor`` neurons is imposed unconditionally
    (see :data:`DEFAULT_MIN_NEURON_FLOOR`): the search never shrinks below the voxel
    holding that many cells, so the returned limit is the BINDING one of the gate limit
    and the count floor. When the gates would have allowed a finer voxel but the floor
    stops the shrink, ``limiting_gate = "density_floor"``. This keeps the reported
    resolution inside the regime where the model's population statistics hold; it is a
    conservatism guardrail, not a signal gain. (A *density*-derived floor is deliberately
    not used: under the fixed-density shrink it scales as ``rho^3`` with the realized
    count and never binds -- see :data:`DEFAULT_MIN_NEURON_FLOOR`.)

    ``estimate_floor_m`` defaults to ``floor_m``. If even the baseline (``rho = 1``)
    fails -- e.g. ``eta`` is below the Gate-3 floor, or the baseline voxel already
    fails at this synchrony -- the result is marked ``feasible = False`` and reports
    the baseline geometry with ``limiting_gate = "gate3"`` (eta veto) or the tightest
    forward gate.
    """
    if min_neuron_floor < 1:
        raise ValueError(f"min_neuron_floor must be >= 1, got {min_neuron_floor!r}")
    est = floor_m if estimate_floor_m is None else estimate_floor_m
    n0 = base_voxel.neuron_count
    if min_neuron_floor > n0:
        raise ValueError(
            f"min_neuron_floor ({min_neuron_floor}) exceeds the baseline neuron "
            f"count ({n0}); the voxel cannot shrink and grow at once"
        )
    gate_kw = dict(floor_m=floor_m, estimate_floor_m=est, reach_orders=reach_orders)

    # The density floor as a shrink factor: rho_floor = (N_floor / N0)^(1/3). The search
    # is clamped here so it never explores statistically-meaningless voxels.
    rho_floor = (min_neuron_floor / n0) ** (1.0 / 3.0)
    search_lo = max(rho_lo, rho_floor)

    base_prov = extend(
        d_single.provenance,
        "resolution inverse: isotropic voxel shrink at fixed cell density "
        "(N = round(rho^3 * N0), f_cell invariant); rho = (N/N0)^(1/3)",
        f"three-gate pass criterion at floor = {floor_m} m, estimate floor = {est} m, "
        f"reach_orders = {reach_orders}",
        f"minimum-count floor = {min_neuron_floor} neurons "
        f"(rho_floor = {rho_floor:.4g}), imposed so the limit stays inside the "
        "population-statistics regime",
        f"baseline voxel N0 = {n0}, L0 = {base_voxel.extent_axial_m} m, "
        f"synchrony s = {synchrony_fraction}",
    )

    # Infeasible if the baseline itself fails: no shrink can help (shrinking only
    # lowers signal and raises the pedestal).
    if not _passes_all(d_single, base_voxel, params, synchrony_fraction, **gate_kw):
        # Distinguish the eta veto (Gate 3) from a forward-gate failure.
        eta_dead = params.dilatation_eta < DEFAULT_ETA_FLOOR
        gate = "gate3" if eta_dead else _limiting_gate(
            d_single, base_voxel, params, synchrony_fraction, **gate_kw
        )
        return ResolutionLimit(
            min_scale_rho=1.0,
            min_neuron_count=n0,
            voxel=base_voxel,
            limiting_gate=gate,
            feasible=False,
            baseline_neuron_count=n0,
            provenance=extend(base_prov, "infeasible: baseline (rho=1) fails this gate"),
        )

    # Does the source still pass at the density floor? If so, the gates never bind
    # within the allowed range and the density floor is what stops the shrink.
    floor_vox = scaled_voxel(base_voxel, search_lo)
    if _passes_all(d_single, floor_vox, params, synchrony_fraction, **gate_kw):
        gate = "density_floor" if search_lo == rho_floor else "none"
        note = (
            f"gates pass to the density floor -> bound by {min_neuron_floor} neurons"
            if gate == "density_floor"
            else f"passes to the search floor rho_lo = {search_lo}"
        )
        return ResolutionLimit(
            min_scale_rho=search_lo,
            min_neuron_count=floor_vox.neuron_count,
            voxel=floor_vox,
            limiting_gate=gate,
            feasible=True,
            baseline_neuron_count=n0,
            provenance=extend(base_prov, note),
        )

    # The gates bind above the density floor -> bisect for the smallest passing rho.
    lo, hi = search_lo, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if _passes_all(d_single, scaled_voxel(base_voxel, mid), params,
                       synchrony_fraction, **gate_kw):
            hi = mid
        else:
            lo = mid
        if (hi - lo) <= tol:
            break

    rho_min = hi  # the passing side of the bracket
    vox = scaled_voxel(base_voxel, rho_min)
    gate = _limiting_gate(d_single, vox, params, synchrony_fraction, **gate_kw)
    return ResolutionLimit(
        min_scale_rho=rho_min,
        min_neuron_count=vox.neuron_count,
        voxel=vox,
        limiting_gate=gate,
        feasible=True,
        baseline_neuron_count=n0,
        provenance=extend(
            base_prov,
            f"min passing rho = {rho_min:.4g} -> N = {vox.neuron_count} "
            f"({vox.neuron_count / n0:.2%} of baseline)",
        ),
    )
