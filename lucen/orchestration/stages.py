"""In-silico Stage 0/1/2 runners (spec section 6.2).

Each runner maps directly onto one of the document's kill-gates. The ordering is
the falsification discipline applied to the build itself: each stage can kill the
project for less compute than the stage after it, and Stage 1 - pure arithmetic
over a cited constant - can kill it for almost nothing.

Stage 1 depends only on Module 1 (:mod:`lucen.source`), which is implemented;
Stage 2 additionally requires Modules 2 and 3.
"""

from __future__ import annotations

import numpy as np

from lucen.base.types import ArrayGeometry, VoxelGeometry
from lucen.detection.clutter import BulkMotion
from lucen.detection.noise import NoiseModel
from lucen.propagation.skull import SkullAcousticMap


def stage0_budget(
    geom: VoxelGeometry,
    unaberrated_floor_m: float,
) -> bool:
    """Stage 0 - first-principles SNR budget (spec section 6.2).

    Confirms the order-of-magnitude gap is 1-2 orders, not 4-6 - the arithmetic
    that Module 1 then makes precise. Returns ``True`` if the prior holds.

    Stub - body is implementation work (spec build step 5).
    """
    raise NotImplementedError(
        "stage0_budget: first-principles SNR budget not yet implemented "
        "(spec section 6.2)"
    )


def stage1_signal(
    geom: VoxelGeometry,
    synchrony_grid: np.ndarray,
    unaberrated_floor_m: float,
) -> bool:
    """Stage 1 - Module 1 against the unaberrated floor (spec section 6.2).

    Is the source term real? Runs the synchrony sweep
    (:func:`lucen.source.synchrony_sweep`) and applies the Stage-1 gate
    (:func:`lucen.source.passes_stage1_gate`). Depends only on Module 1.

    Stub - body is implementation work (spec build step 5). The underlying Module 1
    pieces it composes are already implemented and tested.
    """
    raise NotImplementedError(
        "stage1_signal: Stage-1 runner not yet wired (spec section 6.2); "
        "composes the implemented lucen.source sweep + gate"
    )


def stage2_through_skull(
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
) -> bool:
    """Stage 2 - Modules 1+2+3 against a skull-mimic (spec section 6.2).

    Does tagged content-band contrast survive to the separability bar after clutter
    filtering? This is the number that replaces the whole prior. Requires Modules 2
    and 3.

    Stub - body is implementation work (spec build step 5).
    """
    raise NotImplementedError(
        "stage2_through_skull: Stage-2 runner not yet implemented "
        "(spec section 6.2); requires Modules 2 and 3"
    )
