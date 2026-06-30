"""Stage-1 kill criterion (Design Invariant 5).

The cheapest possible falsification: pure arithmetic over a cited constant. Before
the skull is even added, does *any* plausible synchrony put the summed content-band
displacement within reach of the unaberrated detection floor?

* Pass: there is a synchrony where summed displacement lands within ~1-2 orders of
  the unaberrated floor (tens-of-nm regime) - Lucen stays in "engineering gap"
  territory.
* Fail (kill): even at high synchrony the summed displacement sits far under the
  unaberrated floor. The source term fails; the project hits a wall here, found for
  the cost of an arithmetic sweep (spec section 3.4).
"""

from __future__ import annotations

from lucen.base.types import SummedDisplacement

# "Within reach" = within this many orders of magnitude of the unaberrated floor.
# The spec frames the survivable gap as ~1-2 orders; we take the generous edge (2)
# as the default bar so the gate only kills when the gap is genuinely hopeless.
DEFAULT_REACH_ORDERS: float = 2.0


def passes_stage1_gate(
    sweep: list[SummedDisplacement],
    unaberrated_floor_m: float,
    reach_orders: float = DEFAULT_REACH_ORDERS,
) -> bool:
    """Stage-1 kill criterion.

    Returns ``True`` iff ANY synchrony value in ``sweep`` puts the summed
    displacement within ``reach_orders`` decades of ``unaberrated_floor_m`` (i.e.
    ``d_sum >= floor / 10**reach_orders``). ``False`` => the content-bearing signal
    is below the floor before the skull is added => source term fails => wall.
    """
    if not sweep:
        raise ValueError("passes_stage1_gate received an empty sweep")
    if unaberrated_floor_m <= 0.0:
        raise ValueError(
            f"unaberrated_floor_m must be positive, got {unaberrated_floor_m!r}"
        )

    reach_threshold_m = unaberrated_floor_m / (10.0**reach_orders)
    best = max(s.value_m for s in sweep)
    return best >= reach_threshold_m
