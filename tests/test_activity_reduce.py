"""Activity layer - reducing the timeseries yields a sane, content-band NeuralState.

The bridge to the mechanics must produce SI-sane scalars (s in [0,1], positive jitter
and content corner), label the band CONTENT_FAST (Invariant 3), and carry non-null
provenance (Invariant 6).
"""

from __future__ import annotations

import math

import pytest

from base_neural_model.activity.jitter import (
    ABSOLUTE_JITTER_FLOOR_S,
    jitter_from_synchrony,
    total_jitter,
)
from base_neural_model.activity.reduce import reduce_to_state
from base_neural_model.base.bands import Band


def test_reduced_state_is_si_sane(neural_state):
    assert 0.0 <= neural_state.synchrony_fraction <= 1.0
    assert neural_state.jitter_sigma_s > 0.0
    assert neural_state.content_freq_hz > 0.0
    assert neural_state.mean_firing_rate_hz >= 0.0


def test_reduced_state_band_is_content_fast(neural_state):
    assert neural_state.band is Band.CONTENT_FAST


def test_reduced_state_provenance_non_null(neural_state):
    assert neural_state.provenance.source
    assert len(neural_state.provenance.assumptions) >= 1


def test_jitter_is_phase_spread_plus_absolute_floor(neural_state):
    """The reduced jitter is total_jitter: phase spread AND the absolute floor.

    The reduced sigma_t must combine the synchrony-implied phase spread with the
    synchrony-independent absolute floor in quadrature. This is the correction that
    keeps the content corner f_c load-bearing (a pure phase-spread sigma_t makes the
    jitter low-pass cancel f_c and collapse to synchrony).
    """
    s = neural_state.synchrony_fraction
    f_c = neural_state.content_freq_hz
    expected = total_jitter(s, f_c)
    assert neural_state.jitter_sigma_s == expected

    # It strictly exceeds the synchrony-only jitter by exactly the floor (quadrature),
    # so the decoupling is real and non-trivial.
    sync_only = jitter_from_synchrony(s, f_c)
    assert neural_state.jitter_sigma_s > sync_only
    assert neural_state.jitter_sigma_s == math.hypot(sync_only, ABSOLUTE_JITTER_FLOOR_S)


def test_higher_synchrony_means_less_jitter():
    """More coherence -> tighter timing (monotone): jitter falls as synchrony rises."""
    f_c = 30.0
    assert jitter_from_synchrony(0.9, f_c) < jitter_from_synchrony(0.3, f_c)


def test_total_jitter_adds_floor_in_quadrature():
    """total_jitter = hypot(phase-spread jitter, absolute floor); >= both parts."""
    s, f_c = 0.9, 20.0
    sync_only = jitter_from_synchrony(s, f_c)
    total = total_jitter(s, f_c)
    assert total == math.hypot(sync_only, ABSOLUTE_JITTER_FLOOR_S)
    assert total >= sync_only
    assert total >= ABSOLUTE_JITTER_FLOOR_S
    # With no floor it reduces to the pure phase-spread jitter.
    assert total_jitter(s, f_c, floor_s=0.0) == sync_only


def test_absolute_floor_breaks_the_fc_cancellation():
    """The floor is what makes the jitter low-pass depend on f_c.

    Synchrony-only jitter makes survival exp(-2 pi^2 f_c^2 sigma_t^2) == synchrony at
    ANY f_c (the corner cancels). With the absolute floor the survival must fall as f_c
    rises, so the content corner is load-bearing again.
    """
    s = 0.9
    survivals = []
    for f_c in (10.0, 20.0, 40.0):
        sigma_t = total_jitter(s, f_c)
        survivals.append(math.exp(-2.0 * math.pi**2 * f_c**2 * sigma_t**2))
    # strictly decreasing in f_c (higher content frequency -> harder to detect)
    assert survivals[0] > survivals[1] > survivals[2]

    # Sanity: with floor_s = 0 the same computation is flat at synchrony (the old bug).
    flat = []
    for f_c in (10.0, 20.0, 40.0):
        sigma_t = total_jitter(s, f_c, floor_s=0.0)
        flat.append(math.exp(-2.0 * math.pi**2 * f_c**2 * sigma_t**2))
    assert flat[0] == flat[1] == flat[2]
    assert flat[0] == pytest.approx(s, rel=1e-9)


def test_reduce_carries_activity_provenance(activity_timeseries):
    state = reduce_to_state(activity_timeseries)
    # The E/I source survives into the reduced state's chain.
    assert "Wilson" in state.provenance.source or "E/I" in " ".join(
        state.provenance.assumptions
    )
