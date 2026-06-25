"""Forward + return path through the skull via k-Wave (spec section 4.2).

Forward focus -> voxel interaction -> echo return, returning the effective
displacement recoverable at the array, carrying one-way transmission and residual
aberration forward for the Module 3 noise model.
"""

from __future__ import annotations

from lucen.base.types import (
    ArrayGeometry,
    PropagatedDisplacement,
    SummedDisplacement,
    VoxelGeometry,
)
from lucen.propagation.correction import AberrationCorrection
from lucen.propagation.skull import SkullAcousticMap


def propagate_round_trip(
    source: SummedDisplacement,
    geom: VoxelGeometry,
    skull: SkullAcousticMap,
    correction: AberrationCorrection,
    array_geometry: ArrayGeometry,
    freq_hz: float,
) -> PropagatedDisplacement:
    """Forward focus -> voxel interaction -> echo return, via k-Wave.

    ``freq_hz`` is the interrogation frequency (e.g. 2e6). Returns the effective
    displacement recoverable at the array, with the one-way skull transmission and
    residual aberration attached for the noise model. The returned band must equal
    ``source.band`` (the content band is preserved through propagation), and
    provenance must carry the skull, correction method, and frequency forward.

    Stub - body is implementation work (spec build step 4).
    """
    raise NotImplementedError(
        "propagate_round_trip: k-Wave forward/return propagation not yet "
        "implemented (spec section 4.2)"
    )
