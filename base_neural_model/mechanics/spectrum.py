"""Content-band spectrum of the activity-driven displacement dz(t).

The displacement timeseries (deliverable (a)) carries lexical information only in its
content band; this module computes its one-sided power spectrum and the fraction of
power that sits at/above the content/envelope boundary. It reuses the same boundary
as the activity oscillation analysis so the two layers agree on what "content" means.

A displacement that exists but whose power has been low-passed into the slow envelope
carries timing, not content - the spectrum is how that shows up as a number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from base_neural_model.activity.oscillation import ENVELOPE_CONTENT_BOUNDARY_HZ
from base_neural_model.mechanics.timeseries import DisplacementTimeseries


@dataclass(frozen=True, eq=False)
class ContentBandSpectrum:
    """One-sided power spectrum of dz(t) and its content/envelope split."""

    freqs_hz: np.ndarray          # one-sided FFT frequency axis, Hz
    power: np.ndarray             # one-sided power spectrum of dz(t) (mean removed)
    content_band_power: float     # integrated power at >= the boundary
    envelope_band_power: float    # integrated power below the boundary
    boundary_hz: float            # the content/envelope boundary used

    @property
    def content_fraction(self) -> float:
        """Fraction of (non-DC) displacement power that sits in the content band."""
        total = self.content_band_power + self.envelope_band_power
        return self.content_band_power / total if total > 0.0 else 0.0


def displacement_spectrum(
    ts: DisplacementTimeseries,
    *,
    boundary_hz: float = ENVELOPE_CONTENT_BOUNDARY_HZ,
) -> ContentBandSpectrum:
    """Compute the content-band power spectrum of the displacement trajectory.

    Operates on the content-surviving displacement (after the jitter low-pass), mean
    removed so the spectrum reflects the modulation rather than the operating point.
    """
    t = np.asarray(ts.t_s, dtype=float)
    signal = ts.surviving_dz
    if t.size < 4:
        raise ValueError(f"need >= 4 samples for a spectrum, got {t.size}")

    dt = float(np.mean(np.diff(t)))
    if dt <= 0.0:
        raise ValueError("time vector must be increasing")
    fs = 1.0 / dt

    signal = signal - signal.mean()
    n = signal.size
    spectrum = np.abs(np.fft.rfft(signal)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    nonzero = freqs > 0.0
    content_power = float(spectrum[nonzero & (freqs >= boundary_hz)].sum())
    envelope_power = float(spectrum[nonzero & (freqs < boundary_hz)].sum())

    return ContentBandSpectrum(
        freqs_hz=freqs,
        power=spectrum,
        content_band_power=content_power,
        envelope_band_power=envelope_power,
        boundary_hz=boundary_hz,
    )
