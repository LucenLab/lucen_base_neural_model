"""Activity layer - synchrony is a bounded order parameter, monotone in coupling.

The Kuramoto order parameter must stay in [0, 1], be zero below the critical coupling
and rise above it, and increase with the entraining oscillation strength. These pin
the synchrony the mechanics consume.
"""

from __future__ import annotations

import numpy as np
import pytest

from base_neural_model.activity.synchrony import (
    kuramoto_order_parameter,
    mean_field_order_parameter,
    synchrony_from_oscillation,
)


def test_order_parameter_of_aligned_phases_is_one():
    assert kuramoto_order_parameter(np.zeros(100)) == pytest.approx(1.0)


def test_order_parameter_of_uniform_phases_is_near_zero():
    phases = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    assert kuramoto_order_parameter(phases) < 1e-6


def test_mean_field_zero_below_critical():
    # K_c = 2*spread; below it the incoherent state holds (r = 0).
    assert mean_field_order_parameter(coupling=1.0, spread=4.0) == 0.0


def test_mean_field_monotone_above_critical():
    spread = 4.0
    rs = [mean_field_order_parameter(k, spread) for k in (8.5, 12.0, 20.0, 50.0)]
    assert all(0.0 <= r <= 1.0 for r in rs)
    assert all(np.diff(rs) > 0)  # rises with coupling


def test_synchrony_rises_with_oscillation_power():
    weak = synchrony_from_oscillation(1.0, phase_spread_hz=2.0, coupling_gain=10.0)
    strong = synchrony_from_oscillation(100.0, phase_spread_hz=2.0, coupling_gain=10.0)
    assert 0.0 <= weak <= strong <= 1.0


def test_synchrony_bounded(activity_timeseries):
    r = activity_timeseries.r
    assert r.min() >= 0.0 and r.max() <= 1.0
    assert 0.0 <= activity_timeseries.mean_synchrony <= 1.0
