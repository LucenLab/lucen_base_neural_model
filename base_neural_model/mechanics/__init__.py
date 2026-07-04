"""Mechanics layer - the transduction chain from neural activity to tissue displacement.

Turns a neural state (synchrony, jitter, content corner) into the net axial voxel
displacement the tissue undergoes, through the volume-change transduction chain
(membrane Delta r -> per-cell dV -> tissue strain eps_V -> axial Delta z). The
single-neuron membrane displacement is a cited constant (Invariant 2); the two
load-bearing unknowns kappa and eta come from the Eshelby and poroelastic sub-models.

The deliverable is NOT a sum of membrane displacements (that quantity cancels); it is
the net axial dilatation. Two products: a static ``mechanical_displacement`` at one
synchrony, and a ``displacement_timeseries`` driven by the activity layer's
synchrony trajectory.
"""

from base_neural_model.base.types import MechanicsParams
from base_neural_model.mechanics.derive import build_mechanics_params
from base_neural_model.mechanics.eshelby import (
    MatrixParams,
    eshelby_kappa,
    eshelby_kappa_from,
    eshelby_provenance,
    eshelby_sphere_components,
    eshelby_spheroid_trace,
    interaction_verdict_shift_db,
    mori_tanaka_kappa_clamped,
    mori_tanaka_kappa_traction_free,
)
from base_neural_model.mechanics.neuron_constants import (
    get_single_neuron_displacement,
)
from base_neural_model.mechanics.orientation import (
    axial_orientation_factor,
    deviatoric_eshelby_response,
    directional_axial_strain,
)
from base_neural_model.mechanics.poroelastic import (
    PoroelasticParams,
    consolidation_number,
    consolidation_time,
    drainage_completeness,
    eta_from_permeability,
    poroelastic_eta,
    poroelastic_provenance,
)
from base_neural_model.mechanics.spectrum import (
    ContentBandSpectrum,
    displacement_spectrum,
)
from base_neural_model.mechanics.timeseries import (
    DisplacementTimeseries,
    displacement_timeseries,
)
from base_neural_model.mechanics.transduction import (
    displacement_sweep,
    mechanical_displacement,
)

__all__ = [
    "MechanicsParams",
    "get_single_neuron_displacement",
    "mechanical_displacement",
    "displacement_sweep",
    # kappa from the Eshelby tensor (matrix Poisson ratio)
    "MatrixParams",
    "eshelby_kappa",
    "eshelby_kappa_from",
    "eshelby_sphere_components",
    "eshelby_spheroid_trace",
    "eshelby_provenance",
    "mori_tanaka_kappa_traction_free",
    "mori_tanaka_kappa_clamped",
    "interaction_verdict_shift_db",
    # eta from the poroelastic drainage state
    "PoroelasticParams",
    "poroelastic_eta",
    "eta_from_permeability",
    "consolidation_number",
    "consolidation_time",
    "drainage_completeness",
    "poroelastic_provenance",
    # directional (deviatoric / orientation) channel
    "axial_orientation_factor",
    "deviatoric_eshelby_response",
    "directional_axial_strain",
    # composing kappa/eta into a ready-to-sweep MechanicsParams
    "build_mechanics_params",
    # activity-driven displacement timeseries + its content-band spectrum
    "DisplacementTimeseries",
    "displacement_timeseries",
    "ContentBandSpectrum",
    "displacement_spectrum",
]
