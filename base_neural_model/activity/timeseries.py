"""ActivityTimeseries: the dynamical activity trajectory + the run_activity driver.

Bundles one integration of the E/I neural-mass model into a single immutable record:
the time grid, the E(t) and I(t) trajectories, and the instantaneous synchrony r(t).
``run_activity`` is the one call that integrates the ODEs and assembles it.

The instantaneous synchrony is built from the content-band oscillation: the analytic
amplitude envelope of E(t) sets a time-varying entraining coupling, which maps
through the same mean-field order parameter used for the scalar synchrony
(:mod:`.synchrony`). So r(t) tracks the rhythm - high while the limit cycle is strong
and regular, lower where it weakens.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.signal import hilbert

from base_neural_model.activity.neural_mass import integrate_ei
from base_neural_model.activity.oscillation import (
    OscillationSpectrum,
    analyze_oscillation,
)
from base_neural_model.activity.populations import EIParams
from base_neural_model.activity.synchrony import mean_field_order_parameter
from base_neural_model.base.provenance import Provenance, extend


@dataclass(frozen=True, eq=False)
class ActivityTimeseries:
    """One E/I integration: ``t``, ``E(t)``, ``I(t)``, instantaneous synchrony ``r(t)``.

    ``eq=False`` because the ndarray fields make a generated ``__eq__`` ambiguous and
    unhashable under NumPy (the repo's standard correction for array contracts).
    """

    t_s: np.ndarray               # uniform time grid, seconds
    e: np.ndarray                 # excitatory activation in [0, 1]
    i: np.ndarray                 # inhibitory activation in [0, 1]
    r: np.ndarray                 # instantaneous synchrony (Kuramoto order param), [0,1]
    spectrum: OscillationSpectrum  # frequency content of E(t) + content/envelope split
    params: EIParams
    provenance: Provenance

    @property
    def fs_hz(self) -> float:
        """Sample rate of the uniform time grid, Hz."""
        return 1.0 / float(np.mean(np.diff(self.t_s)))

    @property
    def mean_synchrony(self) -> float:
        """Time-averaged synchrony - the scalar ``s`` the mechanics consume."""
        return float(np.mean(self.r))

    @property
    def mean_firing_rate_hz(self) -> float:
        """Population mean firing rate (Hz): mean E activation x the rhythm frequency.

        In the Wilson-Cowan fraction-of-active-cells convention E is a population
        activation, not a rate; multiplying the mean activation by the dominant
        oscillation frequency gives an order-of-magnitude population firing rate.
        """
        return float(np.mean(self.e)) * self.spectrum.dominant_freq_hz


def _instantaneous_synchrony(
    e: np.ndarray,
    *,
    phase_spread_hz: float,
    coupling_gain: float,
) -> np.ndarray:
    """Instantaneous synchrony r(t) from the rhythm envelope AND the engaged activity.

    Population synchrony in this mean-field reduction reflects how strongly the
    population is *collectively engaged in the coherent rhythm*. Two factors set it:

    * the Hilbert envelope of the mean-removed E(t) - the instantaneous oscillation
      amplitude;
    * the local mean activation relative to the record's mean - a gain that drops when
      the population is driven off its resting set-point (e.g. the movement-related
      *suppression* of the beta rhythm: as drive falls and activation collapses, the
      coherent rhythm the population shares weakens, so synchrony desynchronizes).

    Their product sets the time-varying coupling ``K(t)``, mapped through the
    mean-field order parameter to r(t) in [0, 1]. With a constant drive the activation
    gain is ~1 everywhere and this reduces to the pure-envelope behaviour; under a
    movement-locked drive it is what makes synchrony dip at onset and rebound after.

    Both the rhythm envelope and the activation gain are smoothed over roughly one
    oscillation period. Population synchrony is a phase-locking measure that varies on
    a *slower* timescale than the rhythm itself; without the smoothing the raw Hilbert
    envelope of the non-sinusoidal E(t) wobbles within each cycle and r(t) would dip
    to zero every period - an artifact, not a real per-cycle desynchronization.
    """
    mean_e = e.mean()
    signal = e - mean_e
    # Smooth the envelope over ~one period so r(t) varies on the synchrony timescale,
    # not the rhythm's (the order parameter is a windowed phase-locking measure).
    window = max(1, e.size // 25)
    envelope = _smooth(np.abs(hilbert(signal)), window=window)
    # Local activation envelope: a slow measure of how engaged the population is,
    # normalized to its own mean so a steady cycle gives gain ~1.
    activation = _smooth(np.abs(e), window=window)
    gain = activation / activation.mean() if activation.mean() > 0 else np.ones_like(e)
    coupling = coupling_gain * envelope * gain
    return np.array(
        [mean_field_order_parameter(float(k), phase_spread_hz) for k in coupling]
    )


def _smooth(x: np.ndarray, *, window: int) -> np.ndarray:
    """Moving-average smooth with a centred box window (edge-padded)."""
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(np.pad(x, window // 2, mode="edge"), kernel, mode="same")[
        window // 2 : window // 2 + x.size
    ]


def run_activity(
    params: EIParams | None = None,
    *,
    duration_s: float = 8.0,
    fs_hz: float = 2000.0,
    phase_spread_hz: float = 4.0,
    coupling_gain: float = 300.0,
    drive_fn: Callable[[float], float] | None = None,
) -> ActivityTimeseries:
    """Integrate the E/I model and assemble the :class:`ActivityTimeseries`.

    ``phase_spread_hz`` is the intrinsic firing-frequency heterogeneity ``gamma`` and
    ``coupling_gain`` (synaptic gain per unit oscillation amplitude) scales the rhythm
    envelope into the entraining coupling; the default places the central limit
    cycle's synchrony at ``mean r ~ 0.65`` with r(t) tracking the rhythm.

    ``duration_s`` defaults to 8 s: long enough that (a) the FFT resolves ``f_c`` to
    ~0.1 Hz (resolution = 1/duration), so the reported corner is not a 1 Hz-bin
    rounding of the true harmonic, and (b) the synchrony ``r(t)`` sits at its sustained
    steady-state value rather than the start-up ramp. A short (~1 s) window catches the
    envelope-coupling ramp and under-reports synchrony (~0.76 vs the steady ~0.96); a
    sustained brain-reading epoch is steady-state, so the longer default is the honest
    operating point. Constant-drive integration at 2 kHz.

    ``drive_fn`` is an optional time-varying excitatory drive ``P(t)`` (e.g. the
    movement-locked motor profile, :func:`motor_drive.movement_drive`); when given it
    overrides the constant tonic drive, so r(t) tracks the imposed event.
    """
    params = params or EIParams.central()
    t, e, i = integrate_ei(
        params, duration_s=duration_s, fs_hz=fs_hz, drive_fn=drive_fn
    )
    # Split the spectrum at the rhythm's own envelope/content lower edge, so a gamma
    # preset (band ~30-80 Hz) is not forced through the beta boundary. The motor presets
    # carry the beta band, leaving their boundary at ~13 Hz unchanged.
    spectrum = analyze_oscillation(t, e, boundary_hz=params.rhythm_band.f_lo_hz)
    r = _instantaneous_synchrony(
        e, phase_spread_hz=phase_spread_hz, coupling_gain=coupling_gain
    )

    provenance = extend(
        params.provenance,
        f"E/I neural-mass integrated {duration_s} s at {fs_hz} Hz (scipy RK45)"
        + ("" if drive_fn is None else "; movement-locked drive P(t) imposed"),
        f"dominant content-band oscillation f_c = {spectrum.dominant_freq_hz:.3g} Hz; "
        f"content power fraction {spectrum.content_fraction:.3g}",
        "instantaneous synchrony r(t) = mean-field Kuramoto order parameter of the "
        "Hilbert-envelope coupling; mean r is the synchrony fraction s the mechanics "
        "consume",
    )
    return ActivityTimeseries(
        t_s=t, e=e, i=i, r=r, spectrum=spectrum, params=params, provenance=provenance
    )
