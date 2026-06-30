r"""Inverse problem: the minimum detectable dilatation fraction eta*.

The deepest source-side unknown is eta, the net-dilatation fraction -- the share of the
per-cell volume change that survives as net tissue dilatation rather than internal water
redistribution at conserved volume. The forward gates ask "does this (eta, s) pass?".
This module inverts that into the number the Stage-1 bench experiment actually has to
beat: **eta\***, the smallest eta that clears all three gates at a given synchrony.

That makes the project's go/no-go criterion an explicit, falsifiable threshold:

    measure eta on the bench (direct displacement imaging of driven cortex);
    if eta < eta*, the displacement readout is falsified at that synchrony, and the
    stiffness-modulation observable is the fallback (source physics doc 2.3 / gate 3).

Why the inverse is well-posed (closed form). The coherent source term is linear in eta::

    Delta z = eta * kappa * L * f_cell * s * (3 dr/r)   ->   Delta z ∝ eta

so the verdict scalar is monotone increasing in eta and a single bisection locates the
threshold. Three of the four constraints give an eta floor; one does not:

* Gate 1 (amplitude) and Gate 2 (vs the estimate floor) are linear lower bounds on eta;
* Gate 3 is the hard floor ``eta >= eta_floor``, independent of everything;
* Gate 2 (vs the incoherent pedestal) is eta-INDEPENDENT -- the pedestal carries the
  same ``eta * kappa * L`` prefactor as the coherent term, so eta cancels. The pedestal
  bar therefore cannot be cleared by raising eta; only synchrony or N moves it. If the
  pedestal bar fails at a synchrony, NO eta passes there and the point is infeasible --
  reported as such rather than as a spurious threshold.

eta* is thus ``max(gate-1 bound, gate-2-floor bound, eta_floor)`` where the pedestal bar
permits any pass at all. The bisection evaluates the real :func:`mechanical_displacement`
chain so it stays truthful to the model.
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

# Which constraint sets eta*. "gate3_floor" => the hard eta_floor binds; "amplitude" =>
# a linear gate (Gate 1 or the Gate-2 estimate floor) binds above it; "infeasible" =>
# even eta=1 fails (the eta-independent pedestal bar is unmet at this synchrony).
BindingConstraint = str


@dataclass(frozen=True)
class EtaThreshold:
    """The minimum detectable dilatation fraction at one synchrony.

    ``eta_star`` is the smallest eta in ``[eta_floor, 1]`` whose source term clears all
    three gates at ``synchrony_fraction``. ``feasible`` is ``False`` when even ``eta=1``
    fails -- i.e. the eta-independent pedestal bar is unmet, so no eta rescues this
    synchrony; ``eta_star`` is then ``None``. ``binding_constraint`` names what sets the
    threshold. ``margin_at_one`` is how far above the bar ``eta=1`` sits (the headroom a
    perfect dilatation fraction would have), useful for ranking synchronies.
    """

    eta_star: float | None
    synchrony_fraction: float
    feasible: bool
    binding_constraint: BindingConstraint
    eta_floor: float
    provenance: Provenance


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
    """Three-gate verdict at one (params, s) point (single-element sweep)."""
    sweep = [mechanical_displacement(d_single, voxel, params, s)]
    return (
        passes_stage1_gate(sweep, unaberrated_floor_m=floor_m, reach_orders=reach_orders)
        and passes_content_survival_gate(sweep, estimate_floor_m=estimate_floor_m)
        and passes_dilatation_gate(sweep)
    )


def _with_eta(params: MechanicsParams, eta: float) -> MechanicsParams:
    from dataclasses import replace

    return replace(params, dilatation_eta=eta)


def min_detectable_eta(
    d_single: NeuronDisplacement,
    voxel: VoxelGeometry,
    params: MechanicsParams,
    synchrony_fraction: float,
    *,
    floor_m: float,
    estimate_floor_m: float | None = None,
    reach_orders: float = DEFAULT_REACH_ORDERS,
    eta_floor: float = DEFAULT_ETA_FLOOR,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> EtaThreshold:
    """Smallest dilatation fraction eta that passes all three gates at this synchrony.

    Bisects eta in ``[eta_floor, 1]`` for the threshold (the verdict is monotone up in
    eta). ``params.dilatation_eta`` is overridden by the swept eta, so the caller's value
    is irrelevant. ``estimate_floor_m`` defaults to ``floor_m``.

    Returns an :class:`EtaThreshold`. If even ``eta=1`` fails the point is infeasible
    (``feasible=False``, ``eta_star=None``, ``binding_constraint="infeasible"``) -- this
    happens when the eta-independent pedestal bar is unmet at this synchrony. If the
    Gate-3 floor itself already passes, ``eta_star = eta_floor`` and the binding
    constraint is ``"gate3_floor"``; otherwise a linear gate binds above the floor
    (``"amplitude"``).
    """
    if not 0.0 <= synchrony_fraction <= 1.0:
        raise ValueError(
            f"synchrony_fraction must be in [0, 1], got {synchrony_fraction!r}"
        )
    if not 0.0 < eta_floor <= 1.0:
        raise ValueError(f"eta_floor must lie in (0, 1], got {eta_floor!r}")
    est = floor_m if estimate_floor_m is None else estimate_floor_m
    s = synchrony_fraction
    gate_kw = dict(floor_m=floor_m, estimate_floor_m=est, reach_orders=reach_orders)

    out_at_one = mechanical_displacement(d_single, voxel, _with_eta(params, 1.0), s)
    surviving_at_one = out_at_one.axial_displacement_m * out_at_one.content_band_survival
    margin_at_one = surviving_at_one / est if est > 0 else float("inf")

    base_prov = extend(
        d_single.provenance,
        "min detectable eta*: smallest dilatation fraction passing all three gates "
        "at the given synchrony; Delta z is linear in eta so a bisection locates it",
        f"three-gate pass criterion at floor = {floor_m} m, estimate floor = {est} m, "
        f"reach_orders = {reach_orders}, eta_floor = {eta_floor}",
        f"synchrony s = {s}, voxel N = {voxel.neuron_count}, "
        f"L = {voxel.extent_axial_m} m",
    )

    def result(eta_star, feasible, binding, *note):
        return EtaThreshold(
            eta_star=eta_star,
            synchrony_fraction=s,
            feasible=feasible,
            binding_constraint=binding,
            eta_floor=eta_floor,
            provenance=extend(base_prov, *note) if note else base_prov,
        )

    # Infeasible: even a perfect dilatation fraction fails -> the eta-independent
    # pedestal bar (or a degenerate s) is unmet; no eta rescues this synchrony.
    if not _passes_all(d_single, voxel, _with_eta(params, 1.0), s, **gate_kw):
        return result(
            None, False, "infeasible",
            "even eta=1 fails: the eta-independent Gate-2 pedestal bar is unmet at "
            f"this synchrony (eta=1 surviving displacement = {margin_at_one:.3g}x the "
            "estimate floor, yet below the incoherent pedestal)",
        )

    # The Gate-3 floor already passes -> the hard floor is the binding threshold.
    if _passes_all(d_single, voxel, _with_eta(params, eta_floor), s, **gate_kw):
        return result(
            eta_floor, True, "gate3_floor",
            f"eta_floor = {eta_floor} already clears the amplitude gates; the Gate-3 "
            "dilatation floor is the binding threshold",
        )

    # A linear amplitude gate binds above the floor -> bisect for the crossing.
    lo, hi = eta_floor, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if _passes_all(d_single, voxel, _with_eta(params, mid), s, **gate_kw):
            hi = mid
        else:
            lo = mid
        if (hi - lo) <= tol:
            break

    return result(
        hi, True, "amplitude",
        f"eta* = {hi:.4g} set by a linear amplitude gate (Gate 1 / Gate-2 estimate "
        f"floor) above the Gate-3 floor; eta=1 has {margin_at_one:.3g}x headroom",
    )
