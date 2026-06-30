"""Activity layer - the E/I neural-mass ODEs integrate to a sane oscillatory regime.

The dynamical core must integrate stably, stay in the [0, 1] activation range, sit in
the fast/content band, and respond monotonically to drive (more drive -> more
activity). These are the no-sign-error guarantees on the neural-mass model.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from base_neural_model.activity.neural_mass import integrate_ei
from base_neural_model.activity.oscillation import analyze_oscillation
from base_neural_model.activity.populations import EIParams


def test_integration_returns_uniform_grid(ei_params):
    t, e, i = integrate_ei(ei_params, duration_s=0.5, fs_hz=2000.0)
    assert t.shape == e.shape == i.shape
    assert t.size == 1000  # 0.5 s * 2000 Hz
    dt = np.diff(t)
    assert np.allclose(dt, dt[0])  # uniform


def test_activations_stay_in_unit_range(ei_params):
    _, e, i = integrate_ei(ei_params, duration_s=0.5, fs_hz=2000.0)
    assert e.min() >= 0.0 and e.max() <= 1.0
    assert i.min() >= 0.0 and i.max() <= 1.0


def test_central_regime_oscillates_in_content_band(ei_params):
    t, e, _ = integrate_ei(ei_params, duration_s=1.0, fs_hz=2000.0)
    spec = analyze_oscillation(t, e)
    # The central E/I parameters produce a limit cycle in the fast/content band.
    assert spec.dominant_freq_hz >= 16.0
    assert spec.content_band_power > 0.0


def test_higher_drive_raises_mean_activity(ei_params):
    """More excitatory drive must not lower the mean activity (monotone response)."""
    low = ei_params
    high = replace(ei_params, drive_e=ei_params.drive_e + 0.5)
    _, e_low, _ = integrate_ei(low, duration_s=1.0, fs_hz=2000.0)
    _, e_high, _ = integrate_ei(high, duration_s=1.0, fs_hz=2000.0)
    assert e_high.mean() >= e_low.mean()


def test_invalid_time_constant_rejected():
    import pytest

    with pytest.raises(ValueError):
        replace(EIParams.central(), tau_e_s=0.0)
