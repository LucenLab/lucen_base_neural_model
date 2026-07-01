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

# Content/envelope boundary (Hz). Below this is slow sub-rhythm modulation (the
# movement envelope / mu-band timing); at/above it is the beta content band the
# mechanics target. Set at the sensorimotor beta lower edge (~13 Hz, Kilavik et al.
# 2013) so it sits BELOW the modelled low-beta fundamental (~15 Hz) and never amputates
# it -- the earlier 16 Hz value was a speech envelope/content line that wrongly excluded
# a legitimate low-beta carrier, forcing f_c onto a harmonic. The content corner itself
# is the dominant oscillatory peak (the fundamental), taken as the global non-DC peak.
ENVELOPE_CONTENT_BOUNDARY_HZ: float = 13.0

# Minimum share of (non-DC) oscillation power the content band must hold for its peak to
# count as a real content carrier. Above FFT-leakage noise (a strong slow envelope leaks
# float-scale residue into the content band) but far below any genuine carrier, so it
# only rejects the no-real-carrier case. Not a physical threshold -- a numerical guard.
_CONTENT_PRESENCE_FRACTION: float = 1e-6


@dataclass(frozen=True)
class OscillationSpectrum:
    """The activity's frequency content and its content/envelope split."""

    freqs_hz: np.ndarray          # one-sided FFT frequency axis, Hz
    power: np.ndarray             # one-sided power spectrum of E(t) (mean removed)
    dominant_freq_hz: float       # dominant non-DC peak overall (may be in the envelope)
    content_band_power: float     # integrated power at >= the envelope boundary
    envelope_band_power: float    # integrated power below the envelope boundary
    # Peak frequency of the CONTENT band (>= boundary) only, or None when there is no
    # content-band power. This -- not ``dominant_freq_hz`` -- is the content corner f_c
    # the mechanics may consume; it is None exactly when the signal has no fast carrier
    # (e.g. an envelope-dominated movement event), so a caller cannot mistake a slow
    # envelope peak for the content carrier.
    content_dominant_freq_hz: float | None = None

    @property
    def content_fraction(self) -> float:
        """Fraction of (non-DC) oscillation power that sits in the content band."""
        total = self.content_band_power + self.envelope_band_power
        return self.content_band_power / total if total > 0.0 else 0.0

    @property
    def dominant_is_content(self) -> bool:
        """True iff the overall dominant peak lies in the content band.

        False when the strongest oscillation is the slow envelope (e.g. a movement
        event whose desync/rebound structure dominates the spectrum), which is exactly
        the case that must not be reduced to a CONTENT_FAST state.
        """
        return (
            self.content_dominant_freq_hz is not None
            and self.dominant_freq_hz == self.content_dominant_freq_hz
        )


def analyze_oscillation(
    t: np.ndarray,
    e: np.ndarray,
    *,
    boundary_hz: float = ENVELOPE_CONTENT_BOUNDARY_HZ,
) -> OscillationSpectrum:
    """Compute the one-sided power spectrum of E(t) and its content/envelope split.

    The mean (DC) is removed first so the spectrum reflects the oscillation, not the
    operating point. Two peaks are reported and they are deliberately distinct:

    * ``dominant_freq_hz`` -- the strongest non-DC peak *overall*. For a beta rhythm
      this is the beta carrier, but for an envelope-dominated signal (e.g. a movement
      event) it can lie *below* the boundary, in the slow envelope.
    * ``content_dominant_freq_hz`` -- the strongest peak *within the content band*
      (>= ``boundary_hz``), or ``None`` when the content band carries no power. This is
      the content corner ``f_c`` the mechanics may consume; it is ``None`` precisely
      when there is no fast carrier to consume, so the slow envelope cannot be
      mislabelled as content (Invariant 3).
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

    # The overall dominant peak is the global non-DC peak. For a beta rhythm this is
    # the beta carrier (the fundamental, ~76% of the power); for an envelope-dominated
    # signal (a movement event) it can fall below the boundary. We report it honestly
    # as "dominant overall" and compute the content-band peak SEPARATELY, so a slow
    # envelope peak is never silently handed back as the content corner.
    dominant = (
        float(freqs[nonzero][np.argmax(spectrum[nonzero])]) if nonzero.any() else 0.0
    )
    # The content corner f_c: the strongest peak in the content band only, or None when
    # the content band carries no *meaningful* power (no fast carrier exists to consume).
    # The threshold is a small fraction of total oscillation power, not > 0: FFT leakage
    # from a strong slow envelope leaves float-noise residue in the content band, and
    # treating that as a carrier would put a spurious high-frequency f_c on an otherwise
    # envelope-only signal. A carrier must hold at least this share of the power.
    total_power = content_power + envelope_power
    content_present = (
        content_mask.any()
        and total_power > 0.0
        and content_power / total_power >= _CONTENT_PRESENCE_FRACTION
    )
    if content_present:
        content_dominant: float | None = float(
            freqs[content_mask][np.argmax(spectrum[content_mask])]
        )
    else:
        content_dominant = None

    return OscillationSpectrum(
        freqs_hz=freqs,
        power=spectrum,
        dominant_freq_hz=dominant,
        content_band_power=content_power,
        envelope_band_power=envelope_power,
        content_dominant_freq_hz=content_dominant,
    )
