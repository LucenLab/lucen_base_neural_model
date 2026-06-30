"""Activity layer - reducing the timeseries yields a sane, content-band NeuralState.

The bridge to the mechanics must produce SI-sane scalars (s in [0,1], positive jitter
and content corner), label the band CONTENT_FAST (Invariant 3), and carry non-null
provenance (Invariant 6).
"""

from __future__ import annotations

from base_neural_model.activity.jitter import jitter_from_synchrony
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


def test_jitter_consistent_with_synchrony(neural_state):
    """The reduced jitter equals jitter_from_synchrony at the reduced (s, f_c)."""
    expected = jitter_from_synchrony(
        neural_state.synchrony_fraction, neural_state.content_freq_hz
    )
    assert neural_state.jitter_sigma_s == expected


def test_higher_synchrony_means_less_jitter():
    """More coherence -> tighter timing (monotone): jitter falls as synchrony rises."""
    f_c = 30.0
    assert jitter_from_synchrony(0.9, f_c) < jitter_from_synchrony(0.3, f_c)


def test_reduce_carries_activity_provenance(activity_timeseries):
    state = reduce_to_state(activity_timeseries)
    # The E/I source survives into the reduced state's chain.
    assert "Wilson" in state.provenance.source or "E/I" in " ".join(
        state.provenance.assumptions
    )
