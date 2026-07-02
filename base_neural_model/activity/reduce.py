"""Reduce a dynamical ActivityTimeseries to the scalar NeuralState the mechanics use.

The bridge between the two layers. The transduction chain and the gates operate on a
handful of scalars - synchrony ``s``, jitter ``sigma_t``, content corner ``f_c``,
mean firing rate - not on a trajectory. :func:`reduce_to_state` collapses the
activity timeseries to exactly those, carrying the provenance forward so the static
displacement still traces back to the E/I parameters that produced it.

Keeping this as an explicit, separate step is what makes the dynamical layer
optional: a caller who already has ``(s, sigma_t, f_c)`` can build a ``NeuralState``
directly and drive the mechanics without integrating any ODEs.
"""

from __future__ import annotations

import numpy as np

from base_neural_model.activity.jitter import total_jitter
from base_neural_model.activity.oscillation import ENVELOPE_CONTENT_BOUNDARY_HZ
from base_neural_model.activity.orientation import effective_orientation_coherence
from base_neural_model.activity.timeseries import ActivityTimeseries
from base_neural_model.base.bands import Band
from base_neural_model.base.provenance import extend
from base_neural_model.base.types import NeuralState


def reduce_to_state(
    activity: ActivityTimeseries,
    *,
    synchrony_percentile: float | None = None,
) -> NeuralState:
    """Collapse an :class:`ActivityTimeseries` to a :class:`NeuralState`.

    * ``synchrony_fraction`` = the time-averaged Kuramoto order parameter ``mean r``,
      or -- when ``synchrony_percentile`` is given -- that percentile of ``r(t)``. The
      percentile is for a **bursty** trajectory (:func:`base_neural_model.activity.
      motor_drive.bursty_beta_drive`): a burst-locked acquisition integrates over the
      burst, so the relevant synchrony is the in-burst (high-percentile) value, NOT the
      trough-diluted mean, which would double-count the burst-intermittency penalty that
      the detection coherence window already charges (D1). ``None`` keeps the mean (the
      default for a steady rhythm);
    * ``content_freq_hz`` = the dominant content-band oscillation frequency ``f_c``;
    * ``jitter_sigma_s`` = the total firing-time jitter at ``f_c``
      (:func:`total_jitter`): the synchrony-implied phase spread combined in quadrature
      with an absolute (synchrony-independent) floor, so the activity and mechanics
      agree on the content-band low-pass. The absolute floor is what keeps ``f_c``
      load-bearing -- without it the low-pass collapses to ``synchrony`` at every
      frequency and the content corner drops out (see :func:`total_jitter`);
    * ``mean_firing_rate_hz`` = the population mean firing rate;
    * ``orientation_coherence`` = the effective directional order ``Q_eff`` the
      mechanics consume, the population's structural alignment ``Q_struct`` expressed
      through the temporal synchrony (:func:`effective_orientation_coherence`).
      ``None`` when the population is isotropic (``Q_struct = 0``), so a non-columnar
      population leaves the mechanics' own (Option-1) value untouched.

    The band is ``CONTENT_FAST``: the reduced state is the content-band drive to the
    mechanics, never the slow envelope (Invariant 3).
    """
    if synchrony_percentile is None:
        s = activity.mean_synchrony
    else:
        if not 0.0 < synchrony_percentile <= 100.0:
            raise ValueError(
                "synchrony_percentile must lie in (0, 100], got "
                f"{synchrony_percentile!r}"
            )
        s = float(np.percentile(activity.r, synchrony_percentile))
    # The content corner must be a CONTENT-BAND carrier, not merely the overall
    # dominant peak: for an envelope-dominated signal (e.g. a movement event whose
    # desync/rebound structure dominates the spectrum) the overall peak lies in the
    # slow envelope, and reducing that to a CONTENT_FAST state would violate Invariant
    # 3 (the content band is the target, never the slow envelope) -- silently, since a
    # ~1 Hz f_c makes the jitter low-pass ~1.0 and defeats the content-survival gate.
    # ``content_dominant_freq_hz`` is None exactly when no fast carrier exists, so we
    # reject that case rather than substitute the envelope peak.
    f_c = activity.spectrum.content_dominant_freq_hz
    if f_c is None or f_c <= 0.0:
        raise ValueError(
            "activity has no content-band oscillation to reduce: the dominant peak "
            f"({activity.spectrum.dominant_freq_hz:.3g} Hz) is in the slow envelope "
            f"(< {ENVELOPE_CONTENT_BOUNDARY_HZ:.0f} Hz boundary), so there is no fast "
            "content carrier. Reducing this to a CONTENT_FAST NeuralState would "
            "mislabel the envelope as content (Invariant 3). This happens when the "
            "activity is an envelope-dominated event (e.g. a movement trial reduced "
            "over a short record) rather than a sustained content-band rhythm."
        )
    sigma_t = total_jitter(s, f_c)

    q_struct = activity.params.structural_alignment
    # Only emit an orientation coherence when the population is structurally aligned;
    # an isotropic population (Q_struct = 0) leaves the directional channel to the
    # mechanics' own setting (None), preserving backward behaviour.
    q_eff = (
        effective_orientation_coherence(q_struct, s) if q_struct > 0.0 else None
    )

    extra = []
    if q_eff is not None:
        extra.append(
            f"orientation coherence Q_eff = Q_struct {q_struct:.3g} x synchrony "
            f"{s:.3g} = {q_eff:.3g} (directional channel emerges from the activity)"
        )
    s_label = (
        "mean r" if synchrony_percentile is None
        else f"p{synchrony_percentile:g} r (in-burst)"
    )
    provenance = extend(
        activity.provenance,
        f"reduced to NeuralState: s = {s_label} = {s:.3g}, f_c = {f_c:.3g} Hz, "
        f"sigma_t = {sigma_t:.3g} s (phase spread at f_c + absolute floor, in "
        f"quadrature), mean rate = {activity.mean_firing_rate_hz:.3g} Hz",
        *extra,
        band=Band.CONTENT_FAST,
    )
    return NeuralState(
        synchrony_fraction=s,
        jitter_sigma_s=sigma_t,
        content_freq_hz=f_c,
        mean_firing_rate_hz=activity.mean_firing_rate_hz,
        band=Band.CONTENT_FAST,
        provenance=provenance,
        orientation_coherence=q_eff,
    )
