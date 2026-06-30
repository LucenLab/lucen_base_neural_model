"""Lucen forward model.

A three-module forward simulation that computes one number: the through-skull,
content-band voxel displacement contrast versus the post-clutter noise floor, as a
*curve* over neural synchrony, with the crossover synchrony named.

See ``lucen_forward_model_spec.md`` for the full technical specification and the
design invariants that govern every module.

Package layout
--------------
``lucen.base``          Shared foundation: SI units, temporal bands, provenance,
                        and the typed data contracts that cross module boundaries.
``lucen.source``        Module 1 - population summation (the deciding build).
``lucen.propagation``   Module 2 - acoustic propagation through skull (k-Wave).
``lucen.detection``     Module 3 - detection and the contrast verdict.
``lucen.orchestration`` Wires M1 -> M2 -> M3 and runs the synchrony sweep.
"""

__version__ = "0.1.0"
