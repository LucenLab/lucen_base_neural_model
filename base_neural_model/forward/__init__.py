"""Forward acoustic / detection layer -- the source/sensing seam.

The first package that crosses the boundary the source model (``activity``,
``mechanics``, ``model``) deliberately stops at. It composes the net axial source
displacement Delta z with a conventional phase-sensitive ultrafast-ultrasound
acquisition -- within-epoch coherent integration and the through-skull Walker-Trahey
displacement floor -- to answer the Gate-A / Stage-1 detectability question. No
Lucen-specific innovation is modelled here (the spec excludes it at this gate).
"""

from base_neural_model.forward.detection import (
    AcquisitionParams,
    DetectionBudget,
    detection_budget,
    integration_gain,
    phase_displacement_floor,
    phase_to_displacement_m_per_rad,
)

__all__ = [
    "AcquisitionParams",
    "DetectionBudget",
    "detection_budget",
    "integration_gain",
    "phase_displacement_floor",
    "phase_to_displacement_m_per_rad",
]
