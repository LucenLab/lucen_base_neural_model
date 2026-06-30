r"""Inverse problems for the directional channel: the minimum beta* and Q*.

The directional (deviatoric / orientation) channel adds an axial term on top of the
volume-change monopole::

    Delta z = Delta z_iso * [1 + beta * (S_dev / kappa) * g(Q, mu)] ,
              g(Q, mu) = (2/3) * Q * P2(mu)

where ``beta`` is the per-cell anisotropy (deviatoric fraction of the eigenstrain) and
``Q`` is the orientation order parameter (how aligned the population's axes are).
``min_eta`` inverts the gates for the dilatation fraction; these two functions invert
them for the *directional* unknowns - the numbers a measurement of cell shape (beta)
or columnar alignment (Q) would have to beat for the directional channel to carry the
verdict.

**Well-posedness.** The bracket is *linear* in ``beta`` and linear in ``g``, and
``g`` is monotone increasing in ``Q`` for a fixed director. So when the directional
term is positive - the cells expand *toward* the beam, ``g(Q, mu) > 0`` (i.e.
``|mu| > 1/sqrt(3)``) - the total ``Delta z`` is monotone increasing in both ``beta``
and ``Q``, and a single bisection locates the threshold, exactly as for ``eta``.

When ``g(Q, mu) <= 0`` - the cells expand *across* the beam - the directional term
*subtracts* from the axial signal: raising ``beta`` or ``Q`` makes the displacement
*smaller*. No ``beta`` or ``Q`` rescues a failing point by adding signal, so the
problem is reported infeasible-by-direction (``binding_constraint =
"infeasible_direction"``) rather than returning a spurious threshold. This is the
directional analogue of ``min_eta``'s eta-independent-pedestal infeasibility.

Both solvers evaluate the real :func:`mechanical_displacement` chain, so they stay
truthful to the model. A solver tells you *what value is required*, not what the
tissue actually has - it turns "is the directional channel big enough?" into an
explicit, falsifiable bench target.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from base_neural_model.base.provenance import Provenance, extend
from base_neural_model.base.types import (
    MechanicsParams,
    NeuronDisplacement,
    VoxelGeometry,
)
from base_neural_model.mechanics.orientation import axial_orientation_factor
from base_neural_model.mechanics.transduction import mechanical_displacement
from base_neural_model.model.gates import (
    DEFAULT_REACH_ORDERS,
    passes_content_survival_gate,
    passes_dilatation_gate,
    passes_stage1_gate,
)

# "directional" => a positive directional term bisected to the threshold; "isotropic"
# => the volume-change signal already passes at the lower bound (no directional help
# needed); "infeasible_direction" => g(Q, mu) <= 0, the orientation subtracts signal,
# so no beta/Q rescues; "infeasible" => even the maximum directional term fails.
BindingConstraint = str


@dataclass(frozen=True)
class DirectionalThreshold:
    """The minimum directional unknown (beta* or Q*) that clears all three gates.

    ``value_star`` is the smallest value in ``[lo, 1]`` of the swept directional
    quantity whose source term passes at ``synchrony_fraction``; ``None`` when the
    point is infeasible. ``quantity`` names what was swept (``"anisotropy"`` or
    ``"orientation_coherence"``). ``binding_constraint`` names the regime (see above).
    """

    value_star: float | None
    quantity: str
    synchrony_fraction: float
    feasible: bool
    binding_constraint: BindingConstraint
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


def _min_directional(
    quantity: str,
    d_single: NeuronDisplacement,
    voxel: VoxelGeometry,
    params: MechanicsParams,
    synchrony_fraction: float,
    *,
    floor_m: float,
    estimate_floor_m: float | None,
    reach_orders: float,
    lo: float,
    tol: float,
    max_iter: int,
) -> DirectionalThreshold:
    """Shared bisection for the beta and Q directional inverses.

    ``quantity`` is the :class:`MechanicsParams` field to sweep in ``[lo, 1]``
    (``"anisotropy"`` or ``"orientation_coherence"``).
    """
    if not 0.0 <= synchrony_fraction <= 1.0:
        raise ValueError(
            f"synchrony_fraction must be in [0, 1], got {synchrony_fraction!r}"
        )
    if not 0.0 <= lo <= 1.0:
        raise ValueError(f"lo must lie in [0, 1], got {lo!r}")
    est = floor_m if estimate_floor_m is None else estimate_floor_m
    s = synchrony_fraction
    gate_kw = dict(floor_m=floor_m, estimate_floor_m=est, reach_orders=reach_orders)

    def with_value(v: float) -> MechanicsParams:
        return replace(params, **{quantity: v})

    base_prov = extend(
        d_single.provenance,
        f"min directional {quantity}*: smallest {quantity} passing all three gates at "
        "the given synchrony; Delta z is linear in the directional term so a bisection "
        "locates it when that term is positive",
        f"three-gate pass criterion at floor = {floor_m} m, estimate floor = {est} m, "
        f"reach_orders = {reach_orders}",
        f"synchrony s = {s}, voxel N = {voxel.neuron_count}, "
        f"L = {voxel.extent_axial_m} m, mu = {params.mean_axis_projection}",
    )

    def result(value_star, feasible, binding, *note):
        return DirectionalThreshold(
            value_star=value_star,
            quantity=quantity,
            synchrony_fraction=s,
            feasible=feasible,
            binding_constraint=binding,
            provenance=extend(base_prov, *note) if note else base_prov,
        )

    # Direction check: the orientation factor at the UPPER bound of the sweep. For Q
    # this is g at Q=1; for beta the sweep does not change g, so evaluate g at the
    # params' own Q. If g <= 0 the directional term cannot add signal.
    q_for_g = 1.0 if quantity == "orientation_coherence" else params.orientation_coherence
    g_max = axial_orientation_factor(q_for_g, params.mean_axis_projection)
    if g_max <= 0.0:
        return result(
            None, False, "infeasible_direction",
            f"g(Q, mu) = {g_max:.3g} <= 0: the population's axes expand across the beam "
            f"(mu = {params.mean_axis_projection}), so the directional term subtracts "
            f"from the axial signal -- raising {quantity} cannot rescue this point",
        )

    # The lower bound already passes -> no directional contribution is needed; the
    # volume-change (isotropic) signal alone clears the gates.
    if _passes_all(d_single, voxel, with_value(lo), s, **gate_kw):
        return result(
            lo, True, "isotropic",
            f"the isotropic signal already clears the gates at {quantity} = {lo}; no "
            "directional contribution is required",
        )

    # The maximum directional term still fails -> infeasible even fully directional.
    if not _passes_all(d_single, voxel, with_value(1.0), s, **gate_kw):
        return result(
            None, False, "infeasible",
            f"even {quantity} = 1 (the maximum directional term) fails the gates at "
            "this synchrony; the directional channel cannot carry the verdict here",
        )

    # A passing threshold exists between lo and 1 -> bisect (monotone in the term).
    a, b = lo, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        if _passes_all(d_single, voxel, with_value(mid), s, **gate_kw):
            b = mid
        else:
            a = mid
        if (b - a) <= tol:
            break

    return result(
        b, True, "directional",
        f"{quantity}* = {b:.4g}: the smallest {quantity} whose directional term lifts "
        "the signal across the gates (g > 0, so the verdict is monotone in it)",
    )


def min_anisotropy(
    d_single: NeuronDisplacement,
    voxel: VoxelGeometry,
    params: MechanicsParams,
    synchrony_fraction: float,
    *,
    floor_m: float,
    estimate_floor_m: float | None = None,
    reach_orders: float = DEFAULT_REACH_ORDERS,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> DirectionalThreshold:
    """Smallest per-cell anisotropy beta* that passes all three gates at this synchrony.

    The directional bench target on **cell shape**: how elongated/deviatoric the
    per-cell eigenstrain must be (at the params' orientation coherence ``Q`` and
    director ``mu``) for the directional channel to lift the signal across the gates.
    Bisects ``anisotropy`` in ``[0, 1]``. Returns a :class:`DirectionalThreshold`;
    ``binding_constraint`` is ``"isotropic"`` if no directional help is needed,
    ``"infeasible_direction"`` if the orientation subtracts signal (``g <= 0``), or
    ``"infeasible"`` if even ``beta=1`` fails.
    """
    return _min_directional(
        "anisotropy", d_single, voxel, params, synchrony_fraction,
        floor_m=floor_m, estimate_floor_m=estimate_floor_m, reach_orders=reach_orders,
        lo=0.0, tol=tol, max_iter=max_iter,
    )


def min_orientation_coherence(
    d_single: NeuronDisplacement,
    voxel: VoxelGeometry,
    params: MechanicsParams,
    synchrony_fraction: float,
    *,
    floor_m: float,
    estimate_floor_m: float | None = None,
    reach_orders: float = DEFAULT_REACH_ORDERS,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> DirectionalThreshold:
    """Smallest orientation order Q* that passes all three gates at this synchrony.

    The directional bench target on **columnar alignment**: how aligned the
    population's cell axes must be (at the params' anisotropy ``beta`` and director
    ``mu``) for the directional channel to lift the signal across the gates. Bisects
    ``orientation_coherence`` in ``[0, 1]``. Returns a :class:`DirectionalThreshold`
    with the same ``binding_constraint`` regimes as :func:`min_anisotropy`.
    """
    return _min_directional(
        "orientation_coherence", d_single, voxel, params, synchrony_fraction,
        floor_m=floor_m, estimate_floor_m=estimate_floor_m, reach_orders=reach_orders,
        lo=0.0, tol=tol, max_iter=max_iter,
    )
