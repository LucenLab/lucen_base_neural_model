"""run_neural_model: the end-to-end neural model producing both deliverables.

One call wires the whole model:

    EIParams --(neural-mass ODEs)--> ActivityTimeseries
             --(reduce)-----------> NeuralState  (s, sigma_t, f_c, rate)
             --(transduction)------> DisplacementTimeseries dz(t)   [deliverable (a)]
                                  +-> MechanicalDisplacement       [deliverable (b)]
             --(spectrum)---------> ContentBandSpectrum of dz(t)
             --(gates)------------> the three kill verdicts

Deliverable (a) is the activity-driven displacement timeseries ``dz(t)`` with its
content-band spectrum; deliverable (b) is the static tissue displacement evaluated at
the neural state's mean synchrony, with the full transduction decomposition. Both
share the same reduced ``NeuralState`` and the same cited single-neuron constant, so
the dynamic and static pictures are consistent by construction.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from base_neural_model.activity.motor_drive import MovementProfile, movement_drive
from base_neural_model.activity.populations import EIParams
from base_neural_model.activity.reduce import reduce_to_state
from base_neural_model.activity.timeseries import ActivityTimeseries, run_activity
from base_neural_model.base.provenance import Provenance, merge
from base_neural_model.base.types import (
    MechanicalDisplacement,
    MechanicsParams,
    NeuralState,
    VoxelGeometry,
)
from base_neural_model.mechanics.neuron_constants import (
    get_single_neuron_displacement,
)
from base_neural_model.mechanics.spectrum import (
    ContentBandSpectrum,
    displacement_spectrum,
)
from base_neural_model.mechanics.timeseries import (
    DisplacementTimeseries,
    displacement_timeseries,
)
from base_neural_model.mechanics.transduction import mechanical_displacement
from base_neural_model.model.gates import (
    passes_content_survival_gate,
    passes_dilatation_gate,
    passes_stage1_gate,
)

# A representative mm-scale cortical voxel (matches the repo's central voxel).
DEFAULT_VOXEL = VoxelGeometry(
    extent_axial_m=3e-4, extent_lateral_m=1e-3, neuron_count=10_000, depth_m=2e-2
)

# Unaberrated displacement floor the amplitude gate scores against (metres).
DEFAULT_FLOOR_M: float = 1e-9


@dataclass(frozen=True, eq=False)
class NeuralModelReport:
    """The end-to-end neural-model deliverable: both displacement products + verdicts.

    ``eq=False`` because the timeseries fields carry ndarrays.
    """

    activity: ActivityTimeseries          # the dynamical trajectory
    neural_state: NeuralState             # reduced (s, sigma_t, f_c, rate)
    displacement_timeseries: DisplacementTimeseries  # deliverable (a): dz(t)
    spectrum: ContentBandSpectrum         # content-band spectrum of dz(t)
    mechanical_displacement: MechanicalDisplacement  # deliverable (b): static dz(state)
    passes_amplitude_gate: bool
    passes_content_gate: bool
    passes_dilatation_gate: bool
    provenance: Provenance

    @property
    def all_gates_pass(self) -> bool:
        """True iff all three kill gates pass for this neural state."""
        return (
            self.passes_amplitude_gate
            and self.passes_content_gate
            and self.passes_dilatation_gate
        )


def run_neural_model(
    params: EIParams | None = None,
    *,
    geom: VoxelGeometry = DEFAULT_VOXEL,
    mechanics: MechanicsParams | None = None,
    duration_s: float = 1.0,
    fs_hz: float = 2000.0,
    unaberrated_floor_m: float = DEFAULT_FLOOR_M,
    drive_fn: Callable[[float], float] | None = None,
) -> NeuralModelReport:
    """Run the full activity -> mechanics model and return both deliverables.

    Defaults integrate the central E/I limit cycle and feed it through the central
    mechanics chain on a representative cortical voxel. The cited single-neuron
    displacement is pinned into the mechanics params (Invariant 2). The amplitude and
    content-survival gates score the static state; the dilatation gate scores eta.

    ``drive_fn`` is an optional time-varying excitatory drive ``P(t)`` (e.g. a
    movement-locked motor profile); when given, the activity tracks the imposed event
    rather than a steady limit cycle.
    """
    params = params or EIParams.central()

    d_single = get_single_neuron_displacement()
    base_mechanics = mechanics or MechanicsParams.central()
    # Pin the cited constant into the chain in exactly one place (Invariant 2).
    base_mechanics = replace(base_mechanics, membrane_disp_m=d_single.value_m)

    activity = run_activity(
        params, duration_s=duration_s, fs_hz=fs_hz, drive_fn=drive_fn
    )
    state = reduce_to_state(activity)

    # Deliverable (a): the activity-driven displacement timeseries + its spectrum.
    ts = displacement_timeseries(activity, state, geom, base_mechanics, d_single)
    spectrum = displacement_spectrum(ts)

    # Deliverable (b): the static displacement at the state's mean synchrony, with the
    # state's jitter/content corner stamped on, so it equals the timeseries at s.
    static_params = state.to_mechanics_params(base_mechanics)
    static = mechanical_displacement(
        d_single, geom, static_params, state.synchrony_fraction
    )

    # Gates score the static state (a one-element sweep).
    sweep = [static]
    amp = passes_stage1_gate(sweep, unaberrated_floor_m=unaberrated_floor_m)
    content = passes_content_survival_gate(sweep, estimate_floor_m=unaberrated_floor_m)
    dilat = passes_dilatation_gate(sweep)

    provenance = merge(
        ts.provenance,
        static.provenance,
        source="base_neural_model end-to-end (activity -> mechanics, no sensing)",
    )
    return NeuralModelReport(
        activity=activity,
        neural_state=state,
        displacement_timeseries=ts,
        spectrum=spectrum,
        mechanical_displacement=static,
        passes_amplitude_gate=amp,
        passes_content_gate=content,
        passes_dilatation_gate=dilat,
        provenance=provenance,
    )


def run_motor_cortex(
    *,
    duration_s: float = 1.0,
    fs_hz: float = 2000.0,
    unaberrated_floor_m: float = DEFAULT_FLOOR_M,
) -> NeuralModelReport:
    """Run the model configured for primary motor cortex (M1), steady (resting beta).

    Wires the M1 presets: a beta-band E/I rhythm with strongly columnar layer-5
    structural alignment (:meth:`EIParams.motor_cortex`), large anisotropic Betz-cell
    mechanics (:meth:`MechanicsParams.motor_cortex`), and a voxel at M1 layer-5 depth
    (:meth:`VoxelGeometry.motor_cortex_layer5`). The directional channel is material
    here: the columnar alignment, expressed through the beta-band synchrony, adds a
    directional term on top of the volume-change signal - the feature a generic
    isotropic cortical patch does not have.
    """
    return run_neural_model(
        EIParams.motor_cortex(),
        geom=VoxelGeometry.motor_cortex_layer5(),
        mechanics=MechanicsParams.motor_cortex(),
        duration_s=duration_s,
        fs_hz=fs_hz,
        unaberrated_floor_m=unaberrated_floor_m,
    )


def run_motor_trial(
    profile: MovementProfile | None = None,
    *,
    duration_s: float = 1.0,
    fs_hz: float = 2000.0,
    unaberrated_floor_m: float = DEFAULT_FLOOR_M,
) -> NeuralModelReport:
    """Run an M1 **movement trial**: the motor presets with a movement-locked drive.

    Unlike :func:`run_motor_cortex` (a steady resting-beta cycle), this imposes a
    movement-locked drive ``P(t)`` (:func:`base_neural_model.activity.motor_drive.
    movement_drive`): beta is suppressed (synchrony desynchronizes) at movement onset
    and rebounds above baseline afterwards. The displacement timeseries ``dz(t)`` -
    and, through the orientation coherence, the directional channel - therefore track
    the movement event rather than a constant level. The reduced ``NeuralState``
    summarizes the whole trial (its time-averaged synchrony and the f_c it sits at).
    """
    drive = movement_drive(profile)
    return run_neural_model(
        EIParams.motor_cortex(),
        geom=VoxelGeometry.motor_cortex_layer5(),
        mechanics=MechanicsParams.motor_cortex(),
        duration_s=duration_s,
        fs_hz=fs_hz,
        unaberrated_floor_m=unaberrated_floor_m,
        drive_fn=drive,
    )
