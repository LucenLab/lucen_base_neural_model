"""base_neural_model - a comprehensive neural model: activity -> tissue mechanics.

A two-layer model of what neural activity is and what mechanical signal it produces
in tissue, stopping before any sensing/ultrasound:

* ``base_neural_model.activity``  - the dynamical neural-mass layer: E/I mean-field
  ODEs producing firing, oscillations, and population synchrony over time, reduced to
  a ``NeuralState`` (synchrony s, jitter sigma_t, content corner f_c, mean rate).
* ``base_neural_model.mechanics`` - the transduction chain: membrane Delta r ->
  per-cell volume change -> tissue strain -> net axial dilatation Delta z, with the
  Eshelby confinement factor kappa and the poroelastic dilatation fraction eta.
* ``base_neural_model.model``     - the end-to-end model, the three kill gates, and
  the inverse / global-sensitivity analyses.

Two deliverables: an activity-driven displacement timeseries ``dz(t)`` with its
content-band spectrum, and the static tissue displacement from a neural state with
its full decomposition. See ``docs/architecture.md``.

The design invariants (SI units, the cited single-neuron constant, the content-band
guard, every gate a kill criterion, propagating provenance) are enforced as code in
``base_neural_model.base``.
"""

from base_neural_model.base import (
    Band,
    MechanicalDisplacement,
    MechanicsParams,
    NeuralState,
    NeuronDisplacement,
    Provenance,
    VoxelGeometry,
    extend,
    merge,
    require_content_fast,
)
from base_neural_model.forward import (
    AcquisitionParams,
    AxisSensitivity,
    BudgetCurve,
    DetectionBudget,
    acoustic_axis_ranking,
    detection_budget,
    integration_gain,
    phase_displacement_floor,
    phase_to_displacement_m_per_rad,
    sweep_axis,
)
from base_neural_model.mechanics import (
    DisplacementTimeseries,
    MatrixParams,
    PoroelasticParams,
    build_mechanics_params,
    consolidation_number,
    consolidation_time,
    displacement_spectrum,
    displacement_sweep,
    displacement_timeseries,
    drainage_completeness,
    eshelby_kappa,
    eshelby_kappa_from,
    eta_from_permeability,
    get_single_neuron_displacement,
    mechanical_displacement,
    poroelastic_eta,
)
from base_neural_model.model import (
    DEFAULT_MIN_NEURON_FLOOR,
    DirectionalThreshold,
    EtaThreshold,
    NeuralModelReport,
    ResolutionLimit,
    SobolIndices,
    SobolProblem,
    find_flip,
    gate1_predicate,
    min_anisotropy,
    min_detectable_eta,
    min_orientation_coherence,
    min_resolution,
    model_verdict_scalar,
    passes_content_survival_gate,
    passes_dilatation_gate,
    passes_stage1_gate,
    run_model_sobol,
    run_motor_cortex,
    run_motor_demo,
    run_motor_trial,
    run_neural_model,
    scaled_voxel,
    verdict_flip_table,
)

__version__ = "0.2.0"

__all__ = [
    # base contracts
    "Band",
    "require_content_fast",
    "Provenance",
    "extend",
    "merge",
    "NeuronDisplacement",
    "VoxelGeometry",
    "MechanicsParams",
    "MechanicalDisplacement",
    "NeuralState",
    # mechanics
    "get_single_neuron_displacement",
    "mechanical_displacement",
    "displacement_sweep",
    "MatrixParams",
    "eshelby_kappa",
    "eshelby_kappa_from",
    "PoroelasticParams",
    "poroelastic_eta",
    "eta_from_permeability",
    "consolidation_number",
    "consolidation_time",
    "drainage_completeness",
    "build_mechanics_params",
    "DisplacementTimeseries",
    "displacement_timeseries",
    "displacement_spectrum",
    # forward acoustic / detection layer (the source/sensing seam)
    "AcquisitionParams",
    "DetectionBudget",
    "detection_budget",
    "integration_gain",
    "phase_displacement_floor",
    "phase_to_displacement_m_per_rad",
    # forward acoustic budget sweep
    "BudgetCurve",
    "sweep_axis",
    "AxisSensitivity",
    "acoustic_axis_ranking",
    # model + gates + inverse
    "run_neural_model",
    "run_motor_cortex",
    "run_motor_trial",
    "run_motor_demo",
    "NeuralModelReport",
    "passes_stage1_gate",
    "passes_content_survival_gate",
    "passes_dilatation_gate",
    "EtaThreshold",
    "min_detectable_eta",
    "DirectionalThreshold",
    "min_anisotropy",
    "min_orientation_coherence",
    "ResolutionLimit",
    "min_resolution",
    "scaled_voxel",
    "DEFAULT_MIN_NEURON_FLOOR",
    "SobolProblem",
    "SobolIndices",
    "run_model_sobol",
    "model_verdict_scalar",
    "verdict_flip_table",
    "find_flip",
    "gate1_predicate",
]
