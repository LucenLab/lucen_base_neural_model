"""Clutter filter: bulk-motion rejection (spec section 5.1, step 3).

Strips coherent whole-field bulk motion (cardiac, respiratory) - orders of
magnitude above the neural signal - and keeps the fast, spatially patchy neural
component. Bulk motion is the dominant confound, so the filter is scored on a
bulk-motion amplitude that is genuinely orders above the signal: if it only works
when bulk motion is unrealistically small, that must surface as a fail, not be
hidden by a soft test case (spec section 5.3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BulkMotion:
    """Whole-field bulk-motion confound parameters.

    Amplitudes are in metres and are orders of magnitude above the neural signal
    (um-mm vs nm) - this is the honesty constraint on the dominant confound.
    """

    cardiac_amplitude_m: float      # um-mm: orders above the neural signal
    cardiac_freq_hz: float          # ~1
    respiratory_amplitude_m: float
    respiratory_freq_hz: float


def apply_clutter_filter(
    voxel_timeseries: np.ndarray,
    method: str = "svd",
) -> np.ndarray:
    """Remove coherent whole-field bulk motion, keep the fast patchy component.

    ``voxel_timeseries`` is displacement(t) at the voxel, with bulk motion
    superimposed. ``method`` is ``"svd"`` or ``"highpass"``. Returns the residual
    neural displacement estimate (metres, SI) and is expected to report the
    achieved suppression ratio alongside (bulk motion is the dominant confound).

    Stub - body is implementation work (spec build step 3).
    """
    raise NotImplementedError(
        f"apply_clutter_filter(method={method!r}): bulk-motion rejection not yet "
        f"implemented (spec section 5.1)"
    )
