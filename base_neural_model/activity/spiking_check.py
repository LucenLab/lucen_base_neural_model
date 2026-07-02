r"""Spiking cross-check of the Kuramoto -> wrapped-normal jitter proxy (S5).

The content-band low-pass ``exp(-2 pi^2 f_c^2 sigma_t^2)`` rests on the firing-time
jitter ``sigma_t``, which the model does NOT simulate: :mod:`base_neural_model.activity.
jitter` infers it from the Kuramoto order parameter ``r`` via the wrapped-normal relation
``r = exp(-sigma_phi^2 / 2)`` (so ``sigma_phi = sqrt(-2 ln r)``), then converts to time at
the rhythm period. Wilson-Cowan is a 2-variable mean-field and has no spike-timing
distribution, so that relation is *asserted*, not measured.

This module supplies the missing spiking check: a population of noisy phase oscillators
(a spiking-phase model near the synchronization transition, the regime the exact
Montbrio-Pazo-Roxin QIF reduction also describes) is integrated, and BOTH its order
parameter ``r`` and its measured phase spread ``sigma_phi`` are read off directly. If the
measured spread matches ``sqrt(-2 ln r)`` across coupling strengths, the proxy is
validated from a spiking population rather than assumed.

It is a **validation, not a number-mover**: at the beta corner the jitter low-pass is
nearly inert anyway (with a ~3 ms absolute floor, ``exp(-2 pi^2 (20 Hz)^2 (3 ms)^2)`` is
~0.93, ~0.6 dB), so ``sigma_t`` is not where the beta verdict is lost -- this check only
confirms the proxy is faithful, and is where a *gamma*/speech readout would lean on it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpikingJitterCheck:
    """Result of one noisy-phase-population run: measured r vs the proxy phase spread."""

    order_parameter_r: float          # measured Kuramoto r = <|mean e^{i phi}|>_t
    measured_phase_spread_rad: float  # measured circular std of phases about the mean
    proxy_phase_spread_rad: float     # sqrt(-2 ln r): the wrapped-normal proxy
    coupling: float

    @property
    def relative_error(self) -> float:
        """|measured - proxy| / proxy: how well the wrapped-normal proxy holds here."""
        if self.proxy_phase_spread_rad <= 0.0:
            return 0.0
        return abs(self.measured_phase_spread_rad - self.proxy_phase_spread_rad) / (
            self.proxy_phase_spread_rad
        )


def simulate_phase_population(
    *,
    coupling: float,
    n_oscillators: int = 400,
    freq_hz: float = 20.0,
    freq_spread_hz: float = 2.0,
    noise: float = 4.0,
    duration_s: float = 2.0,
    dt_s: float = 1.0e-3,
    seed: int = 0,
) -> SpikingJitterCheck:
    r"""Integrate a noisy Kuramoto population and read off r and its phase spread.

    ``dphi_i = omega_i dt + (K/N) sum_j sin(phi_j - phi_i) dt + sqrt(2 noise) dW`` with
    Lorentzian-like heterogeneous ``omega_i`` (mean ``2 pi freq``, spread ``freq_spread``)
    and white phase noise. A larger ``coupling`` -> tighter synchrony -> larger ``r`` and
    smaller phase spread. After discarding a transient, ``r`` is the time-averaged order
    parameter and ``measured_phase_spread`` the time-averaged circular standard deviation
    of the phases about their mean -- the quantity the wrapped-normal proxy predicts as
    ``sqrt(-2 ln r)``.
    """
    if n_oscillators < 2:
        raise ValueError(f"n_oscillators must be >= 2, got {n_oscillators!r}")
    if duration_s <= 0.0 or dt_s <= 0.0:
        raise ValueError("duration_s and dt_s must be positive")

    rng = np.random.default_rng(seed)
    omega = 2.0 * math.pi * (freq_hz + freq_spread_hz * rng.standard_normal(n_oscillators))
    phi = rng.uniform(-math.pi, math.pi, n_oscillators)

    n_steps = int(duration_s / dt_s)
    burn = n_steps // 4
    r_samples: list[float] = []
    spread_samples: list[float] = []
    sqrt_noise = math.sqrt(2.0 * noise * dt_s)
    for step in range(n_steps):
        z = np.exp(1j * phi).mean()
        r_t = abs(z)
        mean_phase = np.angle(z)
        # Mean-field Kuramoto coupling toward the population phase.
        dphi = omega * dt_s + coupling * r_t * np.sin(mean_phase - phi) * dt_s
        phi = phi + dphi + sqrt_noise * rng.standard_normal(n_oscillators)
        if step >= burn:
            r_samples.append(r_t)
            # Circular std about the mean phase: sqrt(-2 ln R) of the deviations.
            dev = np.angle(np.exp(1j * (phi - mean_phase)))
            r_dev = abs(np.exp(1j * dev).mean())
            spread_samples.append(math.sqrt(max(-2.0 * math.log(max(r_dev, 1e-9)), 0.0)))

    r_mean = float(np.mean(r_samples))
    spread_mean = float(np.mean(spread_samples))
    proxy = math.sqrt(max(-2.0 * math.log(min(max(r_mean, 1e-9), 1.0 - 1e-12)), 0.0))
    return SpikingJitterCheck(
        order_parameter_r=r_mean,
        measured_phase_spread_rad=spread_mean,
        proxy_phase_spread_rad=proxy,
        coupling=coupling,
    )
