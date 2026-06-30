"""Population summation: neural activity -> effective voxel displacement.

The summed displacement of ``N`` neurons interpolates between two limits:

* fully coherent (``s = 1``): displacements add in phase, ``d_sum = N * d_1`` -
  the optimistic ceiling (tens of nm);
* fully incoherent (``s = 0``): random phases, RMS scaling, ``d_sum = sqrt(N) * d_1`` -
  partial cancellation drives the sum toward the floor.

Treating synchrony ``s`` as the fraction of neurons firing coherently, with the
remainder incoherent, the contract this module realizes is::

    d_sum(s) = d_1 * [ s*N + sqrt((1 - s)*N) ]

This MUST reproduce ``N*d_1`` at ``s=1`` and ``sqrt(N)*d_1`` at ``s=0`` (spec
section 3.2 - the acceptance boundary, enforced by the coherence-limits test).
"""

from __future__ import annotations

import math

import numpy as np

from lucen.base.bands import require_content_fast
from lucen.base.provenance import extend
from lucen.base.types import NeuronDisplacement, SummedDisplacement, VoxelGeometry

_INTERPOLATION_ASSUMPTION = (
    "synchrony = fraction of neurons firing coherently; coherent fraction adds in "
    "phase (linear), incoherent remainder adds in RMS (sqrt): "
    "d_sum(s) = d_1 * [s*N + sqrt((1-s)*N)]"
)


def _summed_value_m(d_single_m: float, n: int, s: float) -> float:
    """Core scalar contract. Realizes ``d_1 * [s*N + sqrt((1-s)*N)]``.

    At ``s=1`` this is exactly ``N*d_1``; at ``s=0`` exactly ``sqrt(N)*d_1``.
    """
    coherent_term = s * n
    incoherent_term = math.sqrt((1.0 - s) * n)
    return d_single_m * (coherent_term + incoherent_term)


def summed_displacement(
    d_single: NeuronDisplacement,
    geom: VoxelGeometry,
    synchrony_fraction: float,
) -> SummedDisplacement:
    """Compute summed voxel displacement at one synchrony value.

    Precondition: ``d_single.band is CONTENT_FAST`` (enforced; raises otherwise).
    Postcondition: ``result.value_m`` equals ``N*d_1`` at ``s=1`` and
    ``sqrt(N)*d_1`` at ``s=0`` (spec section 3.2).
    """
    require_content_fast(d_single.band, context="summed_displacement")

    if not 0.0 <= synchrony_fraction <= 1.0:
        raise ValueError(
            f"synchrony_fraction must be in [0, 1], got {synchrony_fraction!r}"
        )
    if geom.neuron_count <= 0:
        raise ValueError(f"neuron_count must be positive, got {geom.neuron_count!r}")

    value_m = _summed_value_m(
        d_single.value_m, geom.neuron_count, synchrony_fraction
    )

    provenance = extend(
        d_single.provenance,
        _INTERPOLATION_ASSUMPTION,
        f"voxel neuron_count N = {geom.neuron_count}",
    )
    return SummedDisplacement(
        value_m=value_m,
        synchrony_fraction=synchrony_fraction,
        band=d_single.band,
        provenance=provenance,
    )


def synchrony_sweep(
    d_single: NeuronDisplacement,
    geom: VoxelGeometry,
    synchrony_grid: np.ndarray,
) -> list[SummedDisplacement]:
    """Vectorized sweep over synchrony - the primary Module 1 product.

    Coherence (synchrony fraction) is the swept independent variable (Invariant 4):
    the output is a *curve*, never a single point. ``synchrony_grid`` is e.g.
    ``np.linspace(0, 1, 51)`` and must lie within ``[0, 1]``.
    """
    require_content_fast(d_single.band, context="synchrony_sweep")

    grid = np.asarray(synchrony_grid, dtype=float)
    if grid.ndim != 1:
        raise ValueError(f"synchrony_grid must be 1-D, got shape {grid.shape}")
    if grid.size == 0:
        raise ValueError("synchrony_grid is empty")
    if grid.min() < 0.0 or grid.max() > 1.0:
        raise ValueError("synchrony_grid values must lie within [0, 1]")

    return [summed_displacement(d_single, geom, float(s)) for s in grid]
