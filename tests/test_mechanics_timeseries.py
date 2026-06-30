"""Mechanics layer - the activity-driven displacement timeseries dz(t) (deliverable a).

dz(t) must be non-negative, track the synchrony drive monotonically, be SI-sane, and
have a well-defined content-band spectrum. This is the dynamic product of the model.
"""

from __future__ import annotations

import numpy as np

from base_neural_model.base.types import MechanicsParams
from base_neural_model.base.units import is_sane_displacement_m
from base_neural_model.mechanics.neuron_constants import (
    get_single_neuron_displacement,
)
from base_neural_model.mechanics.spectrum import displacement_spectrum
from base_neural_model.mechanics.timeseries import displacement_timeseries


def _params(d_single):
    from dataclasses import replace

    return replace(MechanicsParams.central(), membrane_disp_m=d_single.value_m)


def test_displacement_timeseries_shape_matches_activity(
    activity_timeseries, neural_state, voxel
):
    d = get_single_neuron_displacement()
    ts = displacement_timeseries(activity_timeseries, neural_state, voxel, _params(d), d)
    assert ts.dz.shape == activity_timeseries.r.shape
    assert ts.t_s.shape == activity_timeseries.t_s.shape


def test_displacement_non_negative(activity_timeseries, neural_state, voxel):
    d = get_single_neuron_displacement()
    ts = displacement_timeseries(activity_timeseries, neural_state, voxel, _params(d), d)
    assert np.all(ts.dz >= 0.0)


def test_displacement_tracks_synchrony_monotonically(
    activity_timeseries, neural_state, voxel
):
    """dz is linear in synchrony, so the dz ordering matches the r ordering."""
    d = get_single_neuron_displacement()
    ts = displacement_timeseries(activity_timeseries, neural_state, voxel, _params(d), d)
    order_r = np.argsort(ts.synchrony)
    dz_sorted = ts.dz[order_r]
    assert np.all(np.diff(dz_sorted) >= -1e-30)  # non-decreasing with synchrony


def test_peak_displacement_is_si_sane(activity_timeseries, neural_state, voxel):
    d = get_single_neuron_displacement()
    ts = displacement_timeseries(activity_timeseries, neural_state, voxel, _params(d), d)
    assert is_sane_displacement_m(ts.peak_dz_m)


def test_spectrum_power_non_negative(activity_timeseries, neural_state, voxel):
    d = get_single_neuron_displacement()
    ts = displacement_timeseries(activity_timeseries, neural_state, voxel, _params(d), d)
    spec = displacement_spectrum(ts)
    assert spec.content_band_power >= 0.0
    assert spec.envelope_band_power >= 0.0
    assert 0.0 <= spec.content_fraction <= 1.0
