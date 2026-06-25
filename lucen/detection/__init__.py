"""Module 3 - Detection and the contrast verdict.

Converts the returned displacement into a *measured* one against the real noise
floor and decides separability, in three ordered sub-steps: phase->displacement
estimation, within-unit coherent integration (the sqrt(N) ~ tenfold gain across a
speech unit), and clutter filtering (strip cardiac/respiratory bulk motion). The
output is the contrast number and a boolean (spec section 5).

NOTE: this module is scaffolding. All functions raise ``NotImplementedError`` -
bodies are the implementation work scoped by the spec (build step 3).
"""

from lucen.detection.clutter import BulkMotion, apply_clutter_filter
from lucen.detection.integration import coherent_integration_gain
from lucen.detection.noise import NoiseModel
from lucen.detection.phase_estimator import estimate_displacement_from_phase
from lucen.detection.verdict import detect

__all__ = [
    "estimate_displacement_from_phase",
    "coherent_integration_gain",
    "BulkMotion",
    "apply_clutter_filter",
    "NoiseModel",
    "detect",
]
