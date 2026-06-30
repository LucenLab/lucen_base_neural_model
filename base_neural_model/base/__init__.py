"""Shared foundation for base_neural_model.

Nothing in the activity or mechanics layers computes correctly without these: SI
unit handling, the temporal-band guard, provenance propagation, and the typed data
contracts that cross the activity -> mechanics -> gates boundary.
"""

from base_neural_model.base.bands import Band, require_content_fast
from base_neural_model.base.provenance import Provenance, extend, merge
from base_neural_model.base.types import (
    MechanicalDisplacement,
    MechanicsParams,
    NeuralState,
    NeuronDisplacement,
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
    "MechanicsParams",
    "MechanicalDisplacement",
    "NeuralState",
]
