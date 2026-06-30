"""Per-factor verdict-flip sweep (source doc, PART 2, Module 1 (iv)).

The complement to the Sobol analysis: for each input, hold the others at nominal and
find the value that flips the verdict across the contrast bar, then tabulate which
factors can flip it at all. The expected result -- the same claim the Sobol layer
demonstrates from the other direction -- is that only eta (via permeability) and
content-band synchrony s can move the verdict; geometry and the cited constant
cannot flip it within their literature ranges.

The verdict predicate is injectable. Its default is the Gate-1 amplitude proxy
(jitter-surviving coherent displacement vs an unaberrated floor), consistent with
:func:`base_neural_model.model.passes_stage1_gate`. The factor space can be the
mechanics-only :meth:`SobolProblem.default` or the activity-spanning
:meth:`SobolProblem.with_activity`, in which the E/I drive (not bare synchrony) is
the factor that flips the verdict - so the flip test spans activity -> mechanics.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from base_neural_model.base.provenance import Provenance
from base_neural_model.base.types import NeuronDisplacement, VoxelGeometry
from base_neural_model.model.sensitivity import (
    DEFAULT_BOUNDS,
    DEFAULT_FACTORS,
    SobolProblem,
    model_verdict_scalar,
)

# Nominal (central-column) point in the swept factor space, in the same units the
# factor names use (nm, um, thousands, log10 for permeability).
DEFAULT_NOMINAL: dict[str, float] = {
    "poisson_ratio": 0.47,
    "log10_permeability": -12.5,
    "synchrony": 0.4,
    "membrane_disp_nm": 2.0,
    "cell_radius_um": 8.0,
    "cell_volume_fraction": 0.15,
    "neuron_count_k": 20.0,
}

# Nominal for the activity-spanning factor space (SobolProblem.with_activity): the
# bare ``synchrony`` factor is replaced by the upstream E/I ``drive_e`` (held just
# above threshold) and the firing heterogeneity ``phase_spread_hz``.
DEFAULT_NOMINAL_WITH_ACTIVITY: dict[str, float] = {
    "poisson_ratio": 0.47,
    "log10_permeability": -12.5,
    "membrane_disp_nm": 2.0,
    "cell_radius_um": 8.0,
    "cell_volume_fraction": 0.15,
    "neuron_count_k": 20.0,
    "drive_e": 1.4,
    "phase_spread_hz": 4.0,
}

# Verdict predicate: maps a verdict scalar (metres) to pass/fail against a bar.
VerdictPredicate = Callable[[float], bool]


def gate1_predicate(contrast_bar_m: float) -> VerdictPredicate:
    """The default predicate: ``verdict scalar >= contrast_bar_m`` (Gate-1 proxy)."""
    if contrast_bar_m <= 0.0:
        raise ValueError(f"contrast_bar_m must be positive, got {contrast_bar_m!r}")
    return lambda y: y >= contrast_bar_m


@dataclass(frozen=True)
class VerdictFlip:
    """Whether one factor can flip the verdict, and where it does so."""

    factor: str
    nominal: float
    flip_value: float | None   # value at which the verdict flips, or None
    flips: bool
    lo: float
    hi: float


@dataclass(frozen=True, eq=False)
class VerdictFlipTable:
    """The per-factor flip table -- the expected outcome: only eta and s flip."""

    rows: tuple[VerdictFlip, ...]
    flipping_factors: tuple[str, ...]
    provenance: Provenance


def _evaluate(
    factor: str,
    value: float,
    nominal: dict[str, float],
    names: tuple[str, ...],
    d_single: NeuronDisplacement,
    voxel: VoxelGeometry,
) -> float:
    """Verdict scalar with one factor set to ``value`` and the rest at nominal."""
    row = dict(nominal)
    row[factor] = value
    return model_verdict_scalar(row, names, d_single, voxel)


def find_flip(
    factor: str,
    lo: float,
    hi: float,
    nominal: dict[str, float],
    names: tuple[str, ...],
    d_single: NeuronDisplacement,
    voxel: VoxelGeometry,
    predicate: VerdictPredicate,
    *,
    tol: float = 1e-4,
    max_iter: int = 100,
) -> VerdictFlip:
    """Locate the value of ``factor`` in ``[lo, hi]`` where the verdict flips.

    The verdict scalar is monotone in each individual factor (confirmed by
    ``tests/test_monotonicity.py`` for the chain factors), so the predicate changes
    sign at most once across ``[lo, hi]``. If the predicate agrees at both ends the
    factor cannot flip the verdict in range (``flips=False``); otherwise a bisection
    pins the crossing to ``tol`` (relative to the bracket width).
    """
    nominal_value = nominal[factor]
    pass_lo = predicate(_evaluate(factor, lo, nominal, names, d_single, voxel))
    pass_hi = predicate(_evaluate(factor, hi, nominal, names, d_single, voxel))

    if pass_lo == pass_hi:
        return VerdictFlip(
            factor=factor, nominal=nominal_value, flip_value=None, flips=False,
            lo=lo, hi=hi,
        )

    a, b = lo, hi
    width = hi - lo
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        pass_mid = predicate(_evaluate(factor, mid, nominal, names, d_single, voxel))
        if pass_mid == pass_lo:
            a = mid
        else:
            b = mid
        if (b - a) <= tol * width:
            break

    return VerdictFlip(
        factor=factor, nominal=nominal_value, flip_value=0.5 * (a + b), flips=True,
        lo=lo, hi=hi,
    )


def verdict_flip_table(
    d_single: NeuronDisplacement,
    voxel: VoxelGeometry,
    predicate: VerdictPredicate,
    *,
    problem: SobolProblem | None = None,
    nominal: dict[str, float] | None = None,
    tol: float = 1e-4,
) -> VerdictFlipTable:
    """Build the per-factor flip table over the source factor space.

    For each factor, holds the rest at ``nominal`` and calls :func:`find_flip`. The
    ``flipping_factors`` field is the headline result: the doc predicts only
    ``log10_permeability`` (eta) and ``synchrony`` appear there.
    """
    problem = problem or SobolProblem(names=DEFAULT_FACTORS, bounds=DEFAULT_BOUNDS)
    nominal = nominal or dict(DEFAULT_NOMINAL)
    if set(nominal) != set(problem.names):
        raise ValueError("nominal keys must match the problem factor names")

    rows: list[VerdictFlip] = []
    for name, (lo, hi) in zip(problem.names, problem.bounds, strict=True):
        rows.append(
            find_flip(name, lo, hi, nominal, problem.names, d_single, voxel,
                      predicate, tol=tol)
        )

    flipping = tuple(r.factor for r in rows if r.flips)
    provenance = Provenance(
        source="Per-factor verdict-flip sweep over the Module-1 source factor space",
        assumptions=(
            "verdict predicate is the injected Gate-1 amplitude proxy (jitter-"
            "surviving Delta z vs a fixed floor) while Modules 2/3 are stubs",
            "each factor swept alone with the rest held at the central-column "
            "nominal; the verdict scalar is monotone per factor so a single "
            "bisection locates the flip",
            "expected: only eta (log10_permeability) and synchrony flip the verdict",
        ),
        band=d_single.band,
    )
    return VerdictFlipTable(
        rows=tuple(rows), flipping_factors=flipping, provenance=provenance
    )
