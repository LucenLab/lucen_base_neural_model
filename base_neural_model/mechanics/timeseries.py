"""DisplacementTimeseries: the activity-driven tissue displacement dz(t).

Deliverable (a). The activity layer produces an instantaneous synchrony trajectory
``r(t)``; this module maps it to the tissue's net axial displacement trajectory
``dz(t)`` by evaluating the (existing, tested) transduction chain at each instant's
synchrony. The chain is unchanged - only its synchrony argument now varies in time,
sourced from the dynamical model rather than swept by hand.

The jitter and content corner are taken from the reduced :class:`NeuralState` (they
are slow, population-level properties, not per-sample), while synchrony is the
fast-varying drive. The result is the displacement signal the tissue actually
undergoes as the rhythm waxes and wanes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from base_neural_model.activity.timeseries import ActivityTimeseries
from base_neural_model.base.provenance import Provenance, extend
from base_neural_model.base.types import (
    MechanicsParams,
    NeuralState,
    NeuronDisplacement,
    VoxelGeometry,
)
from base_neural_model.mechanics.transduction import mechanical_displacement


@dataclass(frozen=True, eq=False)
class DisplacementTimeseries:
    """The tissue displacement trajectory dz(t) driven by the activity synchrony r(t).

    ``eq=False`` for the ndarray fields (the repo's array-contract convention).
    """

    t_s: np.ndarray               # time grid, seconds (shared with the activity)
    dz: np.ndarray                # net axial displacement dz(t), metres
    pedestal: np.ndarray          # incoherent pedestal dz(t), metres
    synchrony: np.ndarray         # the driving synchrony r(t), [0, 1]
    content_band_survival: float  # the jitter low-pass applied (constant over t)
    provenance: Provenance

    @property
    def surviving_dz(self) -> np.ndarray:
        """Content-band displacement after the jitter low-pass: ``dz(t) * survival``."""
        return self.dz * self.content_band_survival

    @property
    def peak_dz_m(self) -> float:
        """Peak content-surviving displacement over the record, metres."""
        return float(np.max(self.surviving_dz))


def displacement_timeseries(
    activity: ActivityTimeseries,
    state: NeuralState,
    geom: VoxelGeometry,
    params: MechanicsParams,
    d_single: NeuronDisplacement,
) -> DisplacementTimeseries:
    """Map the activity synchrony trajectory r(t) to a tissue displacement dz(t).

    Evaluates :func:`mechanical_displacement` at each sample's synchrony, with the
    jitter and content corner stamped from the reduced ``state`` (so the content-band
    survival is consistent with the dynamics). ``params.membrane_disp_m`` must equal
    the cited ``d_single.value_m`` - the constant enters the chain in one place
    (Invariant 2).
    """
    chain_params = state.to_mechanics_params(params)

    r = activity.r
    dz = np.empty(r.shape, dtype=float)
    pedestal = np.empty(r.shape, dtype=float)
    survival = 0.0
    for idx, s in enumerate(r):
        md = mechanical_displacement(d_single, geom, chain_params, float(s))
        dz[idx] = md.axial_displacement_m
        pedestal[idx] = md.incoherent_pedestal_m
        survival = md.content_band_survival  # constant in s; same every iteration

    provenance = extend(
        state.provenance,
        "displacement timeseries dz(t): transduction chain evaluated at each "
        "instant's synchrony r(t); jitter sigma_t and content corner f_c from the "
        "reduced NeuralState (population-level, constant over the record)",
        f"content-band survival applied = {survival:.3g} "
        "(exp(-2 pi^2 f_c^2 sigma_t^2))",
    )
    return DisplacementTimeseries(
        t_s=activity.t_s,
        dz=dz,
        pedestal=pedestal,
        synchrony=r,
        content_band_survival=survival,
        provenance=provenance,
    )
