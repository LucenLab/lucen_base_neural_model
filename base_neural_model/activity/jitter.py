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

# Absolute (synchrony-independent) spike-timing jitter floor, seconds. Real cortical
# spike timing has an irreducible standard deviation of order a few milliseconds set by
# channel noise, synaptic-latency variability, and conduction jitter -- present even in
# a perfectly phase-locked population. Measured single-neuron reliability puts this at
# ~1-5 ms (e.g. Mainen & Sejnowski 1995, Science 268:1503, ~1-2 ms to repeated current
# injection in vitro; in vivo cortical spike jitter is a few ms). It does NOT shrink as
# synchrony rises, which is why it must be added independently of the phase-spread term.
ABSOLUTE_JITTER_FLOOR_S: float = 3.0e-3


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


def total_jitter(
    synchrony: float,
    freq_hz: float,
    *,
    floor_s: float = ABSOLUTE_JITTER_FLOOR_S,
) -> float:
    r"""Total spike-timing jitter ``sigma_t`` (s): phase-spread AND an absolute floor.

    Two independent contributions to firing-time spread add in quadrature:

    * the **synchrony-implied** spread ``sigma_phi / (2 pi f)`` from population phase
      disagreement (:func:`jitter_from_synchrony`), which *does* shrink as the
      population phase-locks; and
    * the **absolute floor** ``floor_s`` (channel/synaptic/conduction jitter), which
      does not.

    ``sigma_t = sqrt(sigma_synchrony^2 + floor_s^2)``.

    This is the correction that makes the content corner ``f_c`` load-bearing. With the
    synchrony-only jitter, ``sigma_t = sigma_phi / (2 pi f_c)`` makes the low-pass
    ``exp(-2 pi^2 f_c^2 sigma_t^2) = exp(-sigma_phi^2) = synchrony`` -- the ``f_c``
    cancels and survival collapses to ``s`` regardless of frequency, so the content
    band is inert. The absolute floor breaks that cancellation: at fixed ``floor_s``
    the survival ``exp(-2 pi^2 f_c^2 floor_s^2)`` falls with ``f_c`` (quadratically),
    so higher-frequency content is genuinely harder to detect and where the content
    band sits finally matters.
    """
    if floor_s < 0.0:
        raise ValueError(f"floor_s must be >= 0, got {floor_s!r}")
    sigma_sync = jitter_from_synchrony(synchrony, freq_hz)
    return math.hypot(sigma_sync, floor_s)


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
