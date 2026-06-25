"""Minimal aberration correction (spec section 4.2).

A *minimal* correction lives inside the propagation loop from the start, because
the uncorrected floor (micrometre range) and the corrected floor differ by enough
to flip the verdict - an uncorrected contrast number is not an honest one.
``method="none"`` is a valid baseline for the uncorrected-vs-corrected comparison.

The deferred metamaterial inverse-design correction (spec section 4.4) is the same
``compute_correction`` interface with ``method="metamaterial_inverse_design"``; it
is intentionally NOT part of this build - correction recovers surviving signal, it
cannot manufacture signal that was never there, so it is only built once Modules
1-3 show the signal clears the floor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lucen.base.types import ArrayGeometry
from lucen.propagation.skull import SkullAcousticMap


@dataclass(frozen=True, eq=False)
class AberrationCorrection:
    """A computed wavefront correction and the residual it leaves.

    ``eq=False`` for the ``ndarray`` field (same correction as elsewhere).
    """

    phase_screen_rad: np.ndarray    # per-element phase correction, radians
    method: str                     # "phase_screen" | "time_reversal" | "none"
    residual_rad: float             # residual wavefront error after correction


def compute_correction(
    skull: SkullAcousticMap,
    array_geometry: ArrayGeometry,
    method: str = "time_reversal",
) -> AberrationCorrection:
    """Compute a minimal aberration correction for the skull + array.

    ``method="none"`` returns an identity (zero-phase) correction and is the
    uncorrected baseline. ``"phase_screen"`` and ``"time_reversal"`` are the
    minimal in-loop corrections. ``"metamaterial_inverse_design"`` is deferred
    (spec section 4.4) and is out of scope for this build.

    Stub - body is implementation work (spec build step 4).
    """
    raise NotImplementedError(
        f"compute_correction(method={method!r}): aberration correction not yet "
        f"implemented (spec section 4.2)"
    )
