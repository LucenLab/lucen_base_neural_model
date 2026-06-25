"""Module 2 - Acoustic propagation through skull.

Carries the source displacement's acoustic consequence in through the temporal-
window skull, lets it refract and attenuate, and carries the echo back out -
returning the effective displacement recoverable at the array. Built on
``k-wave-python``; a *minimal* aberration correction lives inside the loop from
the start, because the uncorrected and corrected floors differ by enough to flip
the verdict (spec section 4).

NOTE: this module is scaffolding. All functions raise ``NotImplementedError`` -
bodies are the implementation work scoped by the spec (build step 4). The k-Wave
dependency is imported lazily inside :mod:`lucen.propagation.medium` so importing
this package never requires ``k-wave-python`` to be installed.
"""

from lucen.propagation.correction import AberrationCorrection, compute_correction
from lucen.propagation.gate import passes_stage2_gate
from lucen.propagation.propagate import propagate_round_trip
from lucen.propagation.skull import SkullAcousticMap, load_skull_map

__all__ = [
    "SkullAcousticMap",
    "load_skull_map",
    "AberrationCorrection",
    "compute_correction",
    "propagate_round_trip",
    "passes_stage2_gate",
]
