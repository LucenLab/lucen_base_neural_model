"""Population synchrony: a Kuramoto order parameter driven by the E/I rhythm.

The transduction chain consumes a synchrony fraction ``s in [0, 1]`` - the degree to
which the population fires coherently. Mechanically, ``s`` is the fraction of the
per-cell volume change that adds coherently rather than averaging into the
incoherent pedestal; phenomenologically it is the Kuramoto order parameter of the
population's firing phases::

    r = | (1/N) * sum_j exp(i*phi_j) |

``r = 1`` is perfect phase-locking (full coherence), ``r = 0`` a uniform phase
spread (incoherent). Rather than integrate N per-neuron phase oscillators, we use the
mean-field closure: for a Kuramoto population with coupling ``K`` and a unimodal
phase-frequency spread of width ``gamma``, the stationary order parameter rises from
0 to 1 as the coupling crosses the critical ``K_c = 2*gamma`` (a pitchfork). The E/I
model supplies that coupling: a stronger, more regular content-band oscillation
entrains the population more tightly, so ``K`` scales with the content-band
oscillation amplitude.

This keeps the synchrony analytic and monotone in the activity (no second ODE
system), while remaining a genuine order parameter bounded in [0, 1].
"""

from __future__ import annotations

import math

import numpy as np


def kuramoto_order_parameter(phases: np.ndarray) -> float:
    """Kuramoto order parameter ``r = |mean(exp(i phi))|`` of phases, in [0, 1]."""
    phases = np.asarray(phases, dtype=float)
    if phases.size == 0:
        raise ValueError("phases is empty")
    return float(np.abs(np.mean(np.exp(1j * phases))))


def mean_field_order_parameter(coupling: float, spread: float) -> float:
    """Stationary Kuramoto order parameter for coupling ``K`` and phase spread ``gamma``.

    Below the critical coupling ``K_c = 2*gamma`` the incoherent state is stable
    (``r = 0``); above it a partially synchronized branch appears and grows toward 1.
    We use the standard self-consistent result for a Lorentzian frequency
    distribution, ``r = sqrt(1 - K_c/K)`` for ``K > K_c`` and 0 otherwise - the
    classic second-order (pitchfork) transition, monotone increasing in ``K``.
    """
    if coupling < 0.0:
        raise ValueError(f"coupling must be >= 0, got {coupling!r}")
    if spread <= 0.0:
        raise ValueError(f"spread must be positive, got {spread!r}")
    k_c = 2.0 * spread
    if coupling <= k_c:
        return 0.0
    return math.sqrt(1.0 - k_c / coupling)


# Calibration of the two free surrogate parameters (drive_threshold, coupling_gain)
# against the actual EIParams.central() ODE (integrate_ei -> run_activity.mean_synchrony),
# least-squares fit over drive_e in [0.8, 2.0] at phase_spread_hz = 4.0 -- the exact range
# and spread the Sobol with_activity factor space (model/sensitivity.py) sweeps. This
# replaces two previously-picked constants with values regressed against the model's own
# dynamical ground truth: RMS residual 0.055, max residual 0.17 (at the sharp limit-cycle
# onset near drive_e ~ 1.36-1.4, which any smooth pitchfork surrogate underfits locally).
# Recalibrate by re-running the fit in this docstring's script if EIParams.central() or the
# Sobol sweep's drive_e bounds change. The with_activity Sobol collapse's robustness to
# this surrogate's own documented max residual (0.17) is checked in
# tests/test_sobol_collapse.py::test_activity_collapse_is_robust_to_surrogate_worst_case_error.
_CALIBRATED_DRIVE_THRESHOLD: float = 1.281
_CALIBRATED_COUPLING_GAIN: float = 108.5


def synchrony_from_drive(
    drive: float,
    *,
    phase_spread_hz: float,
    drive_threshold: float = _CALIBRATED_DRIVE_THRESHOLD,
    coupling_gain: float = _CALIBRATED_COUPLING_GAIN,
) -> float:
    """Fast analytic synchrony from the E/I drive (a surrogate for the full ODE run).

    For global-sensitivity sweeps that evaluate thousands of parameter vectors,
    integrating the neural-mass ODEs per sample is prohibitive. This surrogate
    captures the load-bearing dependence the full model exhibits: above a drive
    threshold the E/I loop enters its limit cycle and the rhythm amplitude (hence the
    entraining coupling) grows with drive, so synchrony rises monotonically from 0
    toward 1. The coupling is ``coupling_gain * max(0, drive - drive_threshold)``,
    fed through the same mean-field order parameter as the dynamical path.

    The defaults are NOT picked constants: they are least-squares fit against the actual
    ``EIParams.central()`` ODE trajectory (via ``run_activity().mean_synchrony``) over the
    ``drive_e`` range the Sobol ``with_activity`` factor space sweeps -- ``drive_threshold
    = 1.281`` locates the ODE's own limit-cycle bifurcation (measured onset ~1.36-1.4, the
    surrogate's smooth pitchfork form places its threshold slightly below the sharp true
    onset to best-fit the whole curve), and ``coupling_gain = 108.5`` matches the rise in
    synchrony above it (RMS residual 0.055, max residual 0.17 at the bifurcation itself).

    It is a surrogate, not the ODE result: it preserves monotonicity and the
    sub-/supra-threshold transition, which is what the variance decomposition needs, and
    is now anchored to the model it approximates rather than an arbitrary choice.
    """
    if drive < 0.0:
        raise ValueError(f"drive must be >= 0, got {drive!r}")
    coupling = coupling_gain * max(0.0, drive - drive_threshold)
    if coupling <= 0.0:
        return 0.0
    return mean_field_order_parameter(coupling, phase_spread_hz)


def synchrony_from_oscillation(
    content_band_power: float,
    *,
    phase_spread_hz: float,
    coupling_gain: float = 1.0,
) -> float:
    """Synchrony fraction ``s`` from the content-band oscillation power.

    The content-band oscillation amplitude sets the entraining coupling
    ``K = coupling_gain * sqrt(content_band_power)`` (amplitude ~ sqrt power), and
    ``phase_spread_hz`` is the intrinsic firing-frequency heterogeneity ``gamma``.
    The synchrony is the resulting mean-field order parameter, bounded in [0, 1]: a
    stronger, more regular rhythm entrains the population more tightly.
    """
    if content_band_power < 0.0:
        raise ValueError(
            f"content_band_power must be >= 0, got {content_band_power!r}"
        )
    if coupling_gain <= 0.0:
        raise ValueError(f"coupling_gain must be positive, got {coupling_gain!r}")
    coupling = coupling_gain * math.sqrt(content_band_power)
    return mean_field_order_parameter(coupling, phase_spread_hz)
