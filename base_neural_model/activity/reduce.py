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

from base_neural_model.activity.jitter import jitter_from_synchrony
from base_neural_model.activity.orientation import effective_orientation_coherence
from base_neural_model.activity.timeseries import ActivityTimeseries
from base_neural_model.base.bands import Band
from base_neural_model.base.provenance import extend
from base_neural_model.base.types import NeuralState


def reduce_to_state(activity: ActivityTimeseries) -> NeuralState:
    """Collapse an :class:`ActivityTimeseries` to a :class:`NeuralState`.

    * ``synchrony_fraction`` = the time-averaged Kuramoto order parameter ``mean r``;
    * ``content_freq_hz`` = the dominant content-band oscillation frequency ``f_c``;
    * ``jitter_sigma_s`` = the jitter implied by that synchrony at ``f_c``
      (:func:`jitter_from_synchrony`), so the activity and mechanics agree on the
      content-band low-pass;
    * ``mean_firing_rate_hz`` = the population mean firing rate;
    * ``orientation_coherence`` = the effective directional order ``Q_eff`` the
      mechanics consume, the population's structural alignment ``Q_struct`` expressed
      through the temporal synchrony (:func:`effective_orientation_coherence`).
      ``None`` when the population is isotropic (``Q_struct = 0``), so a non-columnar
      population leaves the mechanics' own (Option-1) value untouched.

    The band is ``CONTENT_FAST``: the reduced state is the content-band drive to the
    mechanics, never the slow envelope (Invariant 3).
    """
    s = activity.mean_synchrony
    f_c = activity.spectrum.dominant_freq_hz
    if f_c <= 0.0:
        raise ValueError(
            "activity has no content-band oscillation (dominant_freq_hz <= 0); "
            "the E/I model is not in an oscillatory regime, so no content corner "
            "can be reduced"
        )
    sigma_t = jitter_from_synchrony(s, f_c)

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
    provenance = extend(
        activity.provenance,
        f"reduced to NeuralState: s = mean r = {s:.3g}, f_c = {f_c:.3g} Hz, "
        f"sigma_t = {sigma_t:.3g} s (from s at f_c), mean rate = "
        f"{activity.mean_firing_rate_hz:.3g} Hz",
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
