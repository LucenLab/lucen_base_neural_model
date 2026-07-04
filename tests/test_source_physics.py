"""Honest source-physics terms: viscoelastic transfer, coherent-term factors, bursty
beta drive, and the spiking jitter cross-check (S1, S3, S4, S5, S6, S7).

Pins each new source term's direction and magnitude on top of the transduction chain,
which reduces exactly to the prior lossless/quasi-static/fully-coherent model at the
inert defaults.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from base_neural_model.activity.motor_drive import bursty_beta_drive
from base_neural_model.activity.populations import EIParams
from base_neural_model.activity.reduce import reduce_to_state
from base_neural_model.activity.spiking_check import simulate_phase_population
from base_neural_model.activity.timeseries import run_activity
from base_neural_model.base.types import MechanicsParams, VoxelGeometry
from base_neural_model.mechanics.neuron_constants import get_single_neuron_displacement
from base_neural_model.mechanics.transduction import mechanical_displacement
from base_neural_model.mechanics.viscoelastic import (
    ViscoelasticParams,
    viscoelastic_factor,
)


def _md(**overrides):
    d = get_single_neuron_displacement()
    geom = VoxelGeometry.motor_cortex_layer5()
    params = replace(MechanicsParams.central(), membrane_disp_m=d.value_m,
                     content_freq_hz=20.0, **overrides)
    return mechanical_displacement(d, geom, params, 1.0)


# --- S1: viscoelastic transfer function --------------------------------------------


def test_viscoelastic_factor_falls_with_frequency_and_is_bounded():
    """|H(f)| in (0, 1], monotonically falling with frequency (stiffer at higher f)."""
    fs = [1.0, 5.0, 20.0, 50.0, 100.0]
    hs = [viscoelastic_factor(f) for f in fs]
    assert all(0.0 < h <= 1.0 for h in hs)
    assert all(b < a for a, b in zip(hs, hs[1:], strict=False))  # strictly decreasing
    assert viscoelastic_factor(20.0) < 1.0                       # attenuated at beta


def test_viscoelastic_params_validate():
    with pytest.raises(ValueError):
        ViscoelasticParams(relaxation_time_s=0.0)
    with pytest.raises(ValueError):
        ViscoelasticParams(exponent=1.5)


def test_motor_preset_viscoelastic_factor_matches_live_computation():
    """The motor preset's hand-set viscoelastic_factor must match viscoelastic_factor()
    at the preset's own content_freq_hz, so the two cannot silently drift apart."""
    m = MechanicsParams.motor_cortex()
    live = viscoelastic_factor(m.content_freq_hz, ViscoelasticParams())
    assert m.viscoelastic_factor == pytest.approx(live, abs=1e-4)


# --- S1/S3/S6/S7: the coherent-term attenuation factors in the chain ----------------


def test_inert_factors_reproduce_lossless_chain():
    """At the defaults the transfer factor is 1.0 and the coherent strain is unchanged."""
    md = _md()
    assert md.source_transfer_factor == 1.0


def test_transfer_factors_compose_multiplicatively():
    """viscoelastic x carrier x coherent-fraction multiply the coherent axial term."""
    base = _md()
    hon = _md(viscoelastic_factor=0.6, carrier_modulation_depth=0.5,
              correlation_coherent_fraction=0.7)
    assert hon.source_transfer_factor == pytest.approx(0.6 * 0.5 * 0.7, rel=1e-9)
    assert hon.axial_displacement_m == pytest.approx(
        base.axial_displacement_m * 0.6 * 0.5 * 0.7, rel=1e-6
    )


def test_saturation_nonbinding_at_direct_strain_but_binds_when_large():
    """S7: the tiny direct strain is barely capped; a large strain saturates."""
    uncapped = _md()
    capped = _md(saturation_strain=1e-2)
    # Direct coherent strain (~2e-5) << 1e-2 cap -> <1% reduction.
    assert capped.axial_displacement_m == pytest.approx(uncapped.axial_displacement_m, rel=0.05)
    # A cap at/below the strain scale bites hard.
    hard = _md(saturation_strain=1e-6)
    assert hard.axial_displacement_m < 0.2 * uncapped.axial_displacement_m


def test_chain_rejects_out_of_range_factors():
    with pytest.raises(ValueError):
        _md(viscoelastic_factor=0.0)
    with pytest.raises(ValueError):
        _md(carrier_modulation_depth=1.5)
    with pytest.raises(ValueError):
        _md(correlation_coherent_fraction=0.0)
    with pytest.raises(ValueError):
        _md(saturation_strain=0.0)


# --- S4: bursty beta drive + in-burst synchrony -------------------------------------


def test_bursty_drive_is_intermittent():
    """High drive at the burst center, sub-threshold trough between; period = dur/occupancy."""
    drv = bursty_beta_drive(occupancy=0.2, burst_duration_s=0.2, drive_high=1.3,
                            drive_trough=0.6)
    assert drv(0.1) == pytest.approx(1.3, abs=1e-6)   # burst center
    assert drv(0.6) == pytest.approx(0.6, abs=1e-3)   # between bursts (period 1.0 s)
    assert drv(1.1) == pytest.approx(1.3, abs=1e-6)   # next burst


def test_bursty_drive_validates():
    with pytest.raises(ValueError):
        bursty_beta_drive(occupancy=0.0)
    with pytest.raises(ValueError):
        bursty_beta_drive(burst_duration_s=0.0)
    with pytest.raises(ValueError):
        bursty_beta_drive(drive_high=0.5, drive_trough=1.0)


def test_in_burst_percentile_exceeds_trough_diluted_mean():
    """A bursty trajectory has p90 (in-burst) synchrony well above its whole-record mean."""
    act = run_activity(EIParams.motor_cortex(), duration_s=6.0, fs_hz=2000.0,
                       drive_fn=bursty_beta_drive(occupancy=0.2, burst_duration_s=0.2))
    mean_state = reduce_to_state(act)
    p90_state = reduce_to_state(act, synchrony_percentile=90.0)
    assert p90_state.synchrony_fraction > mean_state.synchrony_fraction + 0.2


def test_reduce_percentile_validates():
    act = run_activity(EIParams.motor_cortex(), duration_s=2.0, fs_hz=2000.0,
                       drive_fn=bursty_beta_drive())
    with pytest.raises(ValueError):
        reduce_to_state(act, synchrony_percentile=0.0)
    with pytest.raises(ValueError):
        reduce_to_state(act, synchrony_percentile=101.0)


# --- S5: spiking jitter cross-check -------------------------------------------------


def test_spiking_population_validates_wrapped_normal_proxy():
    """Measured phase spread matches sqrt(-2 ln r) across coupling: the proxy is faithful."""
    for K in (20.0, 40.0, 80.0):
        c = simulate_phase_population(coupling=K, duration_s=2.0, seed=1)
        assert c.relative_error < 0.05          # proxy within 5% of the measured spread
    # Tighter coupling -> larger r, smaller spread.
    weak = simulate_phase_population(coupling=20.0, duration_s=2.0, seed=1)
    strong = simulate_phase_population(coupling=80.0, duration_s=2.0, seed=1)
    assert strong.order_parameter_r > weak.order_parameter_r
    assert strong.measured_phase_spread_rad < weak.measured_phase_spread_rad
