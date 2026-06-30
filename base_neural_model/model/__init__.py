"""Model layer - the end-to-end neural model, the gates, and the inverse analyses.

Wires the dynamical activity layer to the mechanics chain and scores the result
against the three kill gates (amplitude, content survival, dilatation). Also hosts
the inverse problems (minimum detectable eta, finest spatial resolution) and the
global analyses (Sobol sensitivity, per-factor verdict flip) that show the verdict
collapses onto eta and synchrony.
"""

from base_neural_model.model.gates import (
    passes_content_survival_gate,
    passes_dilatation_gate,
    passes_stage1_gate,
)
from base_neural_model.model.min_directional import (
    DirectionalThreshold,
    min_anisotropy,
    min_orientation_coherence,
)
from base_neural_model.model.min_eta import EtaThreshold, min_detectable_eta
from base_neural_model.model.resolution import (
    DEFAULT_MIN_NEURON_FLOOR,
    ResolutionLimit,
    min_resolution,
    scaled_voxel,
)
from base_neural_model.model.run import (
    NeuralModelReport,
    run_motor_cortex,
    run_motor_trial,
    run_neural_model,
)
from base_neural_model.model.sensitivity import (
    SobolIndices,
    SobolProblem,
    model_verdict_scalar,
    run_model_sobol,
)
from base_neural_model.model.verdict_flip import (
    VerdictFlip,
    VerdictFlipTable,
    find_flip,
    gate1_predicate,
    verdict_flip_table,
)

__all__ = [
    # the end-to-end neural model (both deliverables)
    "run_neural_model",
    "run_motor_cortex",
    "run_motor_trial",
    "NeuralModelReport",
    # the three kill gates
    "passes_stage1_gate",
    "passes_content_survival_gate",
    "passes_dilatation_gate",
    # inverse: minimum detectable dilatation fraction (the bench bar)
    "EtaThreshold",
    "min_detectable_eta",
    # inverse: minimum directional unknowns (beta*, Q*) -- directional bench targets
    "DirectionalThreshold",
    "min_anisotropy",
    "min_orientation_coherence",
    # inverse: finest spatial resolution
    "ResolutionLimit",
    "min_resolution",
    "scaled_voxel",
    "DEFAULT_MIN_NEURON_FLOOR",
    # Sobol global sensitivity
    "SobolProblem",
    "SobolIndices",
    "run_model_sobol",
    "model_verdict_scalar",
    # per-factor verdict flip
    "VerdictFlip",
    "VerdictFlipTable",
    "verdict_flip_table",
    "find_flip",
    "gate1_predicate",
]
