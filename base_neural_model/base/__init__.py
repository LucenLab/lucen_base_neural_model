"""Shared foundation for the Lucen forward model.

Nothing in the three modules computes correctly without these: SI unit handling,
the temporal-band guard, provenance propagation, and the typed data contracts that
cross every module boundary (spec section 8, build step 1).
"""

from lucen.base.bands import Band, require_content_fast
from lucen.base.provenance import Provenance, extend, merge
from lucen.base.types import (
    ArrayGeometry,
    DetectionResult,
    FeasibilityCurve,
    NeuronDisplacement,
    PropagatedDisplacement,
    SummedDisplacement,
    VoxelGeometry,
)

__all__ = [
    "Band",
    "require_content_fast",
    "Provenance",
    "merge",
    "extend",
    "NeuronDisplacement",
    "VoxelGeometry",
    "SummedDisplacement",
    "PropagatedDisplacement",
    "DetectionResult",
    "FeasibilityCurve",
    "ArrayGeometry",
]
