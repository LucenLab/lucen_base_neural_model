"""Activity layer - the dynamical neural-mass model that drives the mechanics.

A Wilson-Cowan E/I mean-field model integrated over time produces firing,
oscillations, and population synchrony; the layer reduces that trajectory to a
``NeuralState`` (synchrony s, jitter sigma_t, content corner f_c, mean rate) that the
mechanics consume. Build order within the layer: populations -> neural_mass (ODEs)
-> oscillation -> synchrony -> jitter -> timeseries -> reduce.
"""

from base_neural_model.activity.jitter import (
    content_band_survival,
    jitter_from_synchrony,
)
from base_neural_model.activity.motor_drive import (
    MovementProfile,
    movement_drive,
)
from base_neural_model.activity.neural_mass import integrate_ei
from base_neural_model.activity.orientation import (
    effective_orientation_coherence,
    expression_factor,
)
from base_neural_model.activity.oscillation import (
    ENVELOPE_CONTENT_BOUNDARY_HZ,
    OscillationSpectrum,
    analyze_oscillation,
)
from base_neural_model.activity.populations import EIParams
from base_neural_model.activity.reduce import reduce_to_state
from base_neural_model.activity.synchrony import (
    kuramoto_order_parameter,
    mean_field_order_parameter,
    synchrony_from_drive,
    synchrony_from_oscillation,
)
from base_neural_model.activity.timeseries import ActivityTimeseries, run_activity

__all__ = [
    # population parameters
    "EIParams",
    # the dynamical core
    "integrate_ei",
    # oscillation analysis
    "analyze_oscillation",
    "OscillationSpectrum",
    "ENVELOPE_CONTENT_BOUNDARY_HZ",
    # synchrony
    "kuramoto_order_parameter",
    "mean_field_order_parameter",
    "synchrony_from_drive",
    "synchrony_from_oscillation",
    # jitter + the shared content-band low-pass
    "jitter_from_synchrony",
    "content_band_survival",
    # directional coherence emerging from the activity (Option 2)
    "effective_orientation_coherence",
    "expression_factor",
    # movement-locked motor dynamics
    "MovementProfile",
    "movement_drive",
    # the timeseries + driver
    "ActivityTimeseries",
    "run_activity",
    # the bridge to the mechanics
    "reduce_to_state",
]
