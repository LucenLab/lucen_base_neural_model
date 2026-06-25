"""Within-unit coherent integration gain (spec section 5.1, step 2).

The sqrt(N) SNR gain from coherently integrating the frames that make up one
speech unit (tens-to-hundreds of frames -> ~tenfold gain), applied before any
trial averaging.
"""

from __future__ import annotations


def coherent_integration_gain(n_frames: int) -> float:
    """``sqrt(n_frames)`` SNR gain within one speech unit, pre-trial-averaging.

    ``n_frames`` is the number of frames per speech unit. Returns the amplitude
    SNR gain factor.

    Stub - body is implementation work (spec build step 3).
    """
    raise NotImplementedError(
        "coherent_integration_gain: sqrt(N) integration gain not yet implemented "
        "(spec section 5.1)"
    )
