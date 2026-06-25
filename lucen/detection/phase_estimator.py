"""Echo phase -> displacement estimate (spec section 5.1, step 1).

The first detection sub-step: convert the measured echo phase shift into a
displacement estimate at the interrogation frequency.
"""

from __future__ import annotations

import numpy as np


def estimate_displacement_from_phase(
    echo_phase_rad: np.ndarray,
    freq_hz: float,
    sound_speed_m_s: float,
) -> np.ndarray:
    """Convert echo phase shift (radians) to displacement (metres).

    Uses the round-trip phase-to-displacement relation at ``freq_hz`` in a medium
    of speed ``sound_speed_m_s``. Returns displacement in metres (SI - Invariant 1),
    same shape as ``echo_phase_rad``.

    Stub - body is implementation work (spec build step 3).
    """
    raise NotImplementedError(
        "estimate_displacement_from_phase: phase->displacement not yet "
        "implemented (spec section 5.1)"
    )
