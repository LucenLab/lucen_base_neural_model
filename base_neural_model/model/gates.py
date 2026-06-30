"""The three Module 1 kill gates (Design Invariant 5; source physics doc 5).

The module exposes three gates, all of which must pass:

* **Gate 1 - amplitude.** Some plausible synchrony puts the coherent axial
  displacement within ~1-2 orders of the unaberrated floor. Failure is the
  source term collapsing below detectability before propagation - the project
  wall, found for the cost of an arithmetic sweep.
* **Gate 2 - content survival.** Under physiological jitter the content-band
  coherent signal stays above its own incoherent pedestal *and* above the
  displacement-estimate floor. Failure is content low-passed into the envelope -
  a signal that exists but carries no lexical information.
* **Gate 3 - dilatation.** The dilatation fraction ``eta`` is plausibly
  non-negligible. This gate cannot be *closed* by the model; it makes the
  project's sensitivity to ``eta`` visible and routes the verdict to the Stage-1
  bench measurement.
"""

from __future__ import annotations

import math

from base_neural_model.base.types import MechanicalDisplacement

# "Within reach" = within this many orders of magnitude of the unaberrated floor.
# The spec frames the survivable gap as ~1-2 orders; we take the generous edge (2)
# as the default bar so the gate only kills when the gap is genuinely hopeless.
DEFAULT_REACH_ORDERS: float = 2.0

# Below this dilatation fraction the displacement readout is effectively dead -
# nearly all the per-cell volume change is internal water redistribution at
# conserved bulk volume rather than net tissue dilatation (source doc 2.3).
DEFAULT_ETA_FLOOR: float = 0.05


def passes_stage1_gate(
    sweep: list[MechanicalDisplacement],
    unaberrated_floor_m: float,
    reach_orders: float = DEFAULT_REACH_ORDERS,
) -> bool:
    """Gate 1 (amplitude). The cheapest falsification.

    Returns ``True`` iff ANY synchrony value in ``sweep`` puts the coherent axial
    displacement within ``reach_orders`` decades of ``unaberrated_floor_m`` (i.e.
    ``Delta z >= floor / 10**reach_orders``). ``False`` => the content-bearing
    tissue displacement is below the floor => the source term fails on its own.
    """
    if not sweep:
        raise ValueError("passes_stage1_gate received an empty sweep")
    if unaberrated_floor_m <= 0.0:
        raise ValueError(
            f"unaberrated_floor_m must be positive, got {unaberrated_floor_m!r}"
        )

    reach_threshold_m = unaberrated_floor_m / (10.0**reach_orders)
    best = max(s.axial_displacement_m for s in sweep)
    return best >= reach_threshold_m


def passes_content_survival_gate(
    sweep: list[MechanicalDisplacement],
    estimate_floor_m: float,
) -> bool:
    """Gate 2 (content survival).

    For the content-band signal to carry lexical information at some synchrony,
    the *jitter-surviving* coherent displacement must clear two bars at that
    synchrony: it must sit above the source's own incoherent pedestal (section 4)
    and above the displacement-estimate floor.

    Returns ``True`` iff at least one synchrony value satisfies both, with the
    coherent term first attenuated by ``content_band_survival`` (the jitter
    low-pass, section 3). ``False`` => content is low-passed into the envelope.
    """
    if not sweep:
        raise ValueError("passes_content_survival_gate received an empty sweep")
    if estimate_floor_m <= 0.0:
        raise ValueError(
            f"estimate_floor_m must be positive, got {estimate_floor_m!r}"
        )

    for s in sweep:
        surviving = s.axial_displacement_m * s.content_band_survival
        above_pedestal = surviving > s.incoherent_pedestal_m
        above_floor = surviving >= estimate_floor_m
        if above_pedestal and above_floor:
            return True
    return False


def passes_dilatation_gate(
    sweep: list[MechanicalDisplacement],
    eta_floor: float = DEFAULT_ETA_FLOOR,
) -> bool:
    """Gate 3 (dilatation). The model cannot close this gate; it exposes it.

    Returns ``True`` iff the dilatation fraction ``eta`` used in the sweep is at
    or above ``eta_floor`` - i.e. plausibly non-negligible. ``eta`` is empirical
    and is exactly what the Stage-1 bench (direct displacement imaging of driven
    cortical activity in the content band) measures. A failure here flags that a
    null Stage-1 result falsifies the *displacement* readout specifically,
    leaving the stiffness-modulation observable as the fallback.
    """
    if not sweep:
        raise ValueError("passes_dilatation_gate received an empty sweep")

    eta = sweep[0].dilatation_eta
    if any(not math.isclose(s.dilatation_eta, eta) for s in sweep):
        raise ValueError("sweep mixes dilatation_eta values; expected one per sweep")
    return eta >= eta_floor
