"""Temporal bands and the envelope-vs-content guard (Design Invariant 3).

Every displacement quantity carries a band label. The model must never let a
slow-envelope (rhythm/timing) figure stand in for a fast-content figure: that
category error silently swaps "when speech happens" for "what the speech is" and
corrupts the verdict. The guard here makes the error a raised exception rather
than a comment - it is a typed constraint.
"""

from __future__ import annotations

from enum import Enum


class Band(Enum):
    """Temporal band of a displacement quantity.

    Guards against the envelope-vs-content category error (Invariant 3).
    """

    CONTENT_FAST = "content_fast"    # the band Lucen must detect
    ENVELOPE_SLOW = "envelope_slow"  # rhythm; timing only, NOT content
    BROADBAND = "broadband"          # unseparated; disallowed past Module 1


class BandError(ValueError):
    """Raised when a displacement quantity is in the wrong temporal band for the
    operation requested - e.g. feeding an envelope figure into the population
    summation that must operate on the content band."""


def require_content_fast(band: Band, *, context: str = "operation") -> None:
    """Assert that ``band is Band.CONTENT_FAST``, else raise ``BandError``.

    Call this at every entry point that must operate on the content band (the
    single-neuron displacement, the population summation). ``context`` names the
    caller so the failure message points at the offending boundary.
    """
    if band is not Band.CONTENT_FAST:
        raise BandError(
            f"{context} requires a {Band.CONTENT_FAST.value!r} displacement, "
            f"but received {band.value!r}. The fast/content band is the target, "
            f"never the slow envelope (Invariant 3)."
        )
