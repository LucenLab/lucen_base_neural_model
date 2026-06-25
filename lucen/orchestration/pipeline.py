"""The end-to-end feasibility pipeline (spec section 6.1).

Wires M1 -> M2 -> M3 across the synchrony sweep and produces the deliverable: the
contrast-vs-synchrony curve and the crossover synchrony (the minimum synchrony
that clears the separability bar, or ``None`` if the wall is real).
"""

from __future__ import annotations

import numpy as np

from lucen.base.types import ArrayGeometry, FeasibilityCurve, VoxelGeometry
from lucen.detection.clutter import BulkMotion
from lucen.detection.noise import NoiseModel
from lucen.propagation.skull import SkullAcousticMap


def run_feasibility_sweep(
    geom: VoxelGeometry,
    skull: SkullAcousticMap,
    array_geometry: ArrayGeometry,
    noise: NoiseModel,
    bulk: BulkMotion,
    synchrony_grid: np.ndarray,
    interrogation_freq_hz: float,
    n_frames_per_unit: int,
    separability_threshold_db: float,
    correction_method: str = "time_reversal",
) -> FeasibilityCurve:
    """Wire M1 -> M2 -> M3 across the synchrony sweep.

    Produces the deliverable :class:`FeasibilityCurve` and its crossover synchrony
    (the minimum synchrony clearing the separability bar, or ``None`` if the wall
    is real). Every quantity reaching the curve must carry a populated provenance
    chain (Invariant 6).

    Stub - body is implementation work (spec build step 5); requires Modules 2 & 3.
    """
    raise NotImplementedError(
        "run_feasibility_sweep: M1->M2->M3 wiring not yet implemented "
        "(spec section 6.1); requires Modules 2 and 3"
    )
