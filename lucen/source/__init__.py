"""Module 1 - Population summation (the deciding build).

Turns neural activity into an effective voxel displacement field with synchrony
as the swept parameter. This is the only module whose result is genuinely unknown
for speech cortex, so it is built and run first (spec section 8, build step 2).
The single-neuron displacement is imported as a cited constant, not simulated.
"""

from lucen.source.gate import passes_stage1_gate
from lucen.source.neuron_constants import get_single_neuron_displacement
from lucen.source.population import summed_displacement, synchrony_sweep

__all__ = [
    "get_single_neuron_displacement",
    "summed_displacement",
    "synchrony_sweep",
    "passes_stage1_gate",
]
