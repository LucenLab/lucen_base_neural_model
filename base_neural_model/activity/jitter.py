"""Spike-timing jitter sigma_t from the population's synchrony and rhythm.

Firing-time jitter is the standard deviation of individual spike times about the
population's mean phase. It is the load-bearing low-pass on the content band: the
transduction chain attenuates the coherent signal by
``exp(-2*pi^2*f_c^2*sigma_t^2)`` (the characteristic function of Gaussian jitter at
the content corner), so jitter, not amplitude, is what extinguishes content first.

Jitter and synchrony are two views of the same coherence: a tightly synchronized
population (Kuramoto ``r -> 1``) has little phase spread, hence small ``sigma_t``; an
incoherent one (``r -> 0``) has jitter approaching a full oscillation period. We map
the order parameter to a phase spread via the von Mises / wrapped-normal relation
``r = exp(-sigma_phi^2 / 2)`` (so ``sigma_phi = sqrt(-2 ln r)``) and convert the phase
spread to a time at the rhythm's period ``T = 1/f``.
"""

from __future__ import annotations

import math


def jitter_from_synchrony(synchrony: float, freq_hz: float) -> float:
    """Spike-timing jitter ``sigma_t`` (s) from synchrony ``r`` at rhythm ``freq_hz``.

    Inverts the wrapped-normal order parameter ``r = exp(-sigma_phi^2 / 2)`` for the
    phase spread ``sigma_phi`` (radians), then converts to time via the oscillation
    period: ``sigma_t = sigma_phi / (2*pi*f)``. Perfect synchrony (``r = 1``) gives
    zero jitter; ``r -> 0`` gives a jitter approaching a quarter-to-half period.

    ``synchrony`` is clipped just inside (0, 1): exactly 1 would give zero jitter (a
    degenerate delta low-pass) and exactly 0 an infinite spread, neither physical.
    """
    if not 0.0 <= synchrony <= 1.0:
        raise ValueError(f"synchrony must lie in [0, 1], got {synchrony!r}")
    if freq_hz <= 0.0:
        raise ValueError(f"freq_hz must be positive, got {freq_hz!r}")

    r = min(max(synchrony, 1e-6), 1.0 - 1e-9)
    sigma_phi = math.sqrt(-2.0 * math.log(r))   # radians of phase spread
    return sigma_phi / (2.0 * math.pi * freq_hz)


def content_band_survival(sigma_t: float, f_c: float) -> float:
    """Jitter low-pass ``exp(-2*pi^2*f_c^2*sigma_t^2)`` at the content corner.

    The fraction of the content-band coherent signal that outlives Gaussian
    firing-time jitter - 1 at zero jitter, falling steeply as jitter grows. Shared
    with the transduction chain so the activity and mechanics agree on the low-pass.
    """
    if sigma_t < 0.0:
        raise ValueError(f"sigma_t must be >= 0, got {sigma_t!r}")
    if f_c <= 0.0:
        raise ValueError(f"f_c must be positive, got {f_c!r}")
    return math.exp(-2.0 * math.pi**2 * f_c**2 * sigma_t**2)
