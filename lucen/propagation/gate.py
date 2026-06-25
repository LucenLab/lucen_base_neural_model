"""Stage-2 kill criterion (Design Invariant 5).

With minimal correction, does enough of the source displacement survive the round
trip that Module 3 can still separate active from inactive?

* Pass: corrected surviving displacement keeps the contrast above the detection
  floor; quantify the correction recovery factor (corrected vs uncorrected) - the
  single biggest SNR lever and the justification for the metamaterial work later.
* Fail (kill): even with idealized correction, skull attenuation + residual
  aberration push the recoverable displacement below the floor - the non-invasive
  case fails here (spec section 4.3).
"""

from __future__ import annotations

from lucen.base.types import PropagatedDisplacement


def passes_stage2_gate(
    propagated: PropagatedDisplacement,
    detection_floor_m: float,
) -> bool:
    """Stage-2 kill criterion.

    Returns ``True`` iff the displacement recoverable at the array after the
    corrected round trip stays at or above ``detection_floor_m``. ``False`` => the
    non-invasive (through-skull) case fails.

    Stub - body is implementation work (spec build step 4).
    """
    raise NotImplementedError(
        "passes_stage2_gate: Stage-2 criterion not yet implemented "
        "(spec section 4.3)"
    )
