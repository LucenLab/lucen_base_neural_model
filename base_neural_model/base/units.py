"""SI units and the reporting boundary (Design Invariant 1).

Internally, *every* quantity is SI: displacement in metres, frequency in Hz,
distance in metres, time in seconds, pressure in Pa, density in kg/m^3, sound
speed in m/s. A single nm/um slip moves the feasibility verdict by three orders
of magnitude - it fabricates or destroys the entire case. So conversion to any
human-readable unit happens *only* here, at the reporting boundary, and never in
module logic.

This module also defines the physically-sane displacement bounds that the
unit-consistency test (spec section 7.1) asserts on every quantity crossing a
module boundary - the cheap tripwire for the nm/um class of error.
"""

from __future__ import annotations

import math

# --- SI scale factors (metres per unit) ----------------------------------------
NM_PER_M: float = 1e9
UM_PER_M: float = 1e6
MM_PER_M: float = 1e3

# --- Physically-sane displacement bounds (metres) -------------------------------
# Any displacement crossing a module boundary must satisfy
# ``DISPLACEMENT_MIN_M < d < DISPLACEMENT_MAX_M``. Sub-picometre or super-
# centimetre "displacements" in this pipeline are always a units bug, never real.
DISPLACEMENT_MIN_M: float = 1e-12
DISPLACEMENT_MAX_M: float = 1e-2


def is_sane_displacement_m(value_m: float) -> bool:
    """True iff ``value_m`` is a finite displacement within the sane SI band.

    The guard behind the unit-consistency invariant (spec section 7.1).
    """
    return math.isfinite(value_m) and DISPLACEMENT_MIN_M < value_m < DISPLACEMENT_MAX_M


# --- Reporting-boundary converters (SI -> human-readable) -----------------------
# Use these ONLY when formatting output for a human. Never feed their result back
# into a computation.


def m_to_nm(value_m: float) -> float:
    """Metres -> nanometres. Reporting boundary only."""
    return value_m * NM_PER_M


def m_to_um(value_m: float) -> float:
    """Metres -> micrometres. Reporting boundary only."""
    return value_m * UM_PER_M


def m_to_mm(value_m: float) -> float:
    """Metres -> millimetres. Reporting boundary only."""
    return value_m * MM_PER_M


def db20(ratio: float) -> float:
    """Amplitude ratio -> decibels, ``20 * log10(ratio)``.

    The contrast convention used throughout: displacement is an amplitude, so
    contrast in dB is ``20*log10(signal/noise_floor)`` (spec section 2,
    ``DetectionResult.contrast_db``). Raises on a non-positive ratio - a zero or
    negative amplitude ratio is never physically meaningful here.
    """
    if ratio <= 0.0:
        raise ValueError(f"db20 requires a positive amplitude ratio, got {ratio!r}")
    return 20.0 * math.log10(ratio)
