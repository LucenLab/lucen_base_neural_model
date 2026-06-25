"""Orchestration - running the in-silico stages.

Wires Module 1 -> Module 2 -> Module 3 across the synchrony sweep to produce the
deliverable feasibility curve, and provides the Stage 0/1/2 runners that map
directly onto the spec's kill-gates (spec section 6).

NOTE: the pipeline and stage runners are scaffolding and raise
``NotImplementedError`` until Modules 2 and 3 land. Module 1 (:mod:`lucen.source`)
is already runnable on its own.
"""

from lucen.orchestration.pipeline import run_feasibility_sweep
from lucen.orchestration.stages import (
    stage0_budget,
    stage1_signal,
    stage2_through_skull,
)

__all__ = [
    "run_feasibility_sweep",
    "stage0_budget",
    "stage1_signal",
    "stage2_through_skull",
]
