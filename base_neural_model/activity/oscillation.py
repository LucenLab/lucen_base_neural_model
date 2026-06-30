"""Oscillation analysis of the E/I activity: dominant frequency and band power.

Reads the excitatory activity trajectory E(t) from the neural-mass model and
extracts the rhythm the mechanics cares about: the dominant oscillation frequency
(which becomes the content-band corner ``f_c``), the power in the content band, and
the content-vs-envelope split (reusing :mod:`base_neural_model.base.bands`).

The content/envelope boundary is the standard speech distinction: the slow envelope
(rhythm/timing, ``< ~16 Hz``) versus the fast content band (the carrier of lexical
information). Only the content band feeds the mechanical transduction (Invariant 3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Content/envelope boundary (Hz). Below this is the slow speech envelope (timing);
# at/above it is the fast content band the mechanics target.
ENVELOPE_CONTENT_BOUNDARY_HZ: float = 16.0


@dataclass(frozen=True)
class OscillationSpectrum:
    """The activity's frequency content and its content/envelope split."""

    freqs_hz: np.ndarray          # one-sided FFT frequency axis, Hz
    power: np.ndarray             # one-sided power spectrum of E(t) (mean removed)
    dominant_freq_hz: float       # peak frequency in the content band
    content_band_power: float     # integrated power at >= the envelope boundary
    envelope_band_power: float    # integrated power below the envelope boundary

    @property
    def content_fraction(self) -> float:
        """Fraction of (non-DC) oscillation power that sits in the content band."""
        total = self.content_band_power + self.envelope_band_power
        return self.content_band_power / total if total > 0.0 else 0.0


def analyze_oscillation(
    t: np.ndarray,
    e: np.ndarray,
    *,
    boundary_hz: float = ENVELOPE_CONTENT_BOUNDARY_HZ,
) -> OscillationSpectrum:
    """Compute the one-sided power spectrum of E(t) and its content/envelope split.

    The mean (DC) is removed first so the spectrum reflects the oscillation, not the
    operating point. The dominant content-band frequency is the peak above the
    envelope boundary; it becomes the content corner ``f_c`` the mechanics consume.
    """
    t = np.asarray(t, dtype=float)
    e = np.asarray(e, dtype=float)
    if t.ndim != 1 or t.size < 4:
        raise ValueError(f"need a 1-D time series of length >= 4, got shape {t.shape}")
    if e.shape != t.shape:
        raise ValueError(f"e and t must match shape, got {e.shape} vs {t.shape}")

    dt = float(np.mean(np.diff(t)))
    if dt <= 0.0:
        raise ValueError("time vector must be increasing")
    fs = 1.0 / dt

    signal = e - e.mean()
    n = signal.size
    spectrum = np.abs(np.fft.rfft(signal)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    # Split power at the envelope/content boundary, ignoring the DC bin.
    nonzero = freqs > 0.0
    content_mask = nonzero & (freqs >= boundary_hz)
    envelope_mask = nonzero & (freqs < boundary_hz)
    content_power = float(spectrum[content_mask].sum())
    envelope_power = float(spectrum[envelope_mask].sum())

    if content_mask.any() and spectrum[content_mask].max() > 0.0:
        dominant = float(freqs[content_mask][np.argmax(spectrum[content_mask])])
    else:
        # No content-band oscillation: fall back to the overall non-DC peak.
        dominant = (
            float(freqs[nonzero][np.argmax(spectrum[nonzero])])
            if nonzero.any()
            else 0.0
        )

    return OscillationSpectrum(
        freqs_hz=freqs,
        power=spectrum,
        dominant_freq_hz=dominant,
        content_band_power=content_power,
        envelope_band_power=envelope_power,
    )
