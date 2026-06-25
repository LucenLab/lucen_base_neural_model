"""The contrast verdict (spec section 5.2).

The full detection chain for one synchrony value: phase estimate -> coherent
integration -> clutter filter -> contrast, setting ``separable`` against the
active/inactive bar (MEG-class separability threshold).
"""

from __future__ import annotations

from lucen.base.types import DetectionResult, PropagatedDisplacement
from lucen.detection.clutter import BulkMotion
from lucen.detection.noise import NoiseModel


def detect(
    propagated: PropagatedDisplacement,
    noise: NoiseModel,
    bulk: BulkMotion,
    n_frames_per_unit: int,
    separability_threshold_db: float,
) -> DetectionResult:
    """Full chain: phase estimate -> integrate -> clutter filter -> contrast.

    Sets ``separable = (contrast_db >= separability_threshold_db)``. ``contrast_db``
    is ``20*log10(signal/noise_floor)`` (use :func:`lucen.base.units.db20`), and
    the result carries ``propagated.synchrony_fraction`` through unchanged.

    Stub - body is implementation work (spec build step 3).
    """
    raise NotImplementedError(
        "detect: phase->integrate->clutter->contrast chain not yet implemented "
        "(spec section 5.2)"
    )
