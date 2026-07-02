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

from base_neural_model.activity.motor_drive import (
    MovementProfile,
    bursty_beta_drive,
    movement_drive,
    sustained_imagery_drive,
)
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
from base_neural_model.forward.detection import (
    AcquisitionParams,
    DetectionBudget,
    detection_budget,
)
from base_neural_model.forward.safety import echo_snr_within_safety
from base_neural_model.mechanics.mechanisms import (
    MechanismDecomposition,
    MechanismParams,
    decompose_mechanisms,
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
# N = 21,000 is the count that reconciles with MechanicsParams.central()'s asserted
# f_cell = 0.15 at r = 8 um (f_cell = N * V_cell / V_voxel = 0.150), and it sits in the
# literature density range (cortical ~40-100k neurons/mm^3, Herculano-Houzel 2009 ->
# ~12-30k in this 0.3 nL voxel). The earlier N=10,000 implied f_cell~0.072, a silent 2x
# inconsistency with the central preset (see check_volume_fraction_consistency).
DEFAULT_VOXEL = VoxelGeometry(
    extent_axial_m=3e-4, extent_lateral_m=1e-3, neuron_count=21_000, depth_m=2e-2
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
    # Acoustic detection verdict (Gate A / Stage 1). Populated only when an
    # AcquisitionParams is supplied; None keeps the model strictly source-side, and
    # the gates then score against the scalar unaberrated_floor_m as before.
    detection: DetectionBudget | None = None
    # Band-separated source-mechanism decomposition (S2). Populated only when
    # mechanism_params is supplied: the direct neuromechanical (beta content) term plus
    # the slow osmotic + vascular (envelope) terms, so the content-vs-hemodynamic trade
    # is explicit. ``detection`` scores the content-band (direct) term.
    mechanisms: MechanismDecomposition | None = None

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
    duration_s: float = 8.0,
    fs_hz: float = 2000.0,
    unaberrated_floor_m: float = DEFAULT_FLOOR_M,
    acquisition: AcquisitionParams | None = None,
    residual_clutter_m: float = 0.0,
    synchrony_percentile: float | None = None,
    mechanism_params: MechanismParams | None = None,
    drive_fn: Callable[[float], float] | None = None,
) -> NeuralModelReport:
    """Run the full activity -> mechanics model and return both deliverables.

    Defaults integrate the central E/I limit cycle and feed it through the central
    mechanics chain on a representative cortical voxel. The cited single-neuron
    displacement is pinned into the mechanics params (Invariant 2). The amplitude and
    content-survival gates score the static state; the dilatation gate scores eta.

    ``acquisition`` opts in to the acoustic detection layer (Gate A / Stage 1): when
    given, the static displacement is composed with a conventional phase-sensitive
    ultrafast acquisition into a :class:`~base_neural_model.forward.detection.
    DetectionBudget`, and the amplitude / content-survival gates score against the
    *derived* through-skull floor (``budget.floor_m``) with the integration gain
    folded into the signal -- instead of the scalar ``unaberrated_floor_m``.
    ``residual_clutter_m`` is the post-clutter-filter residual that competes with the
    echo-SNR floor for the binding denominator. When ``acquisition is None`` the model
    stays strictly source-side and the scalar-floor path is unchanged.

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
    # ``synchrony_percentile`` scores a bursty trajectory at its in-burst synchrony (S4);
    # None keeps the whole-record mean (the steady-rhythm default).
    state = reduce_to_state(activity, synchrony_percentile=synchrony_percentile)

    # Deliverable (a): the activity-driven displacement timeseries + its spectrum.
    # Split the spectrum at the rhythm's own envelope/content edge (gamma splits at
    # ~30 Hz, beta at ~13 Hz), consistent with the activity-side split in run_activity.
    ts = displacement_timeseries(activity, state, geom, base_mechanics, d_single)
    spectrum = displacement_spectrum(ts, boundary_hz=params.rhythm_band.f_lo_hz)

    # Deliverable (b): the static displacement at the state's mean synchrony, with the
    # state's jitter/content corner stamped on, so it equals the timeseries at s.
    static_params = state.to_mechanics_params(base_mechanics)
    static = mechanical_displacement(
        d_single, geom, static_params, state.synchrony_fraction
    )

    # Acoustic detection layer (Gate A / Stage 1): when an acquisition is supplied, the
    # derived through-skull floor already carries ALL the receive-side coherent gain
    # (sqrt(n_elements) beamforming + 1/sqrt(N_ens) integration), so the gates score the
    # BARE static displacement directly against that full floor -- the same ratio as
    # DetectionBudget.snr_db. (Dividing the floor by the integration gain here would
    # re-apply the temporal sqrt(N_ens) that is already in it -- the double count.)
    detection: DetectionBudget | None = None
    if acquisition is None:
        amp_floor_m = unaberrated_floor_m
        content_floor_m = unaberrated_floor_m
    else:
        # The content corner is passed so the clutter high-pass (D2) removes any carrier
        # below its cutoff; the safety check (D6) flags an echo SNR above the transcranial
        # MI/thermal ceiling.
        detection = detection_budget(
            static,
            acquisition,
            residual_clutter_m=residual_clutter_m,
            content_freq_hz=state.content_freq_hz,
            snr_exceeds_safety=not echo_snr_within_safety(acquisition),
        )
        amp_floor_m = detection.floor_m
        content_floor_m = detection.floor_m

    # Band-separated source-mechanism decomposition (S2): the direct neuromechanical
    # (beta content) term plus the slow osmotic + vascular (envelope/hemodynamic) terms.
    mechanisms: MechanismDecomposition | None = None
    if mechanism_params is not None:
        mechanisms = decompose_mechanisms(
            static,
            geom,
            confinement_kappa=static_params.confinement_kappa,
            mech_params=mechanism_params,
            saturation_strain=static_params.saturation_strain,
        )

    # Gates score the static state (a one-element sweep).
    sweep = [static]
    amp = passes_stage1_gate(sweep, unaberrated_floor_m=amp_floor_m)
    content = passes_content_survival_gate(sweep, estimate_floor_m=content_floor_m)
    dilat = passes_dilatation_gate(sweep)

    prov_source = (
        "base_neural_model end-to-end (activity -> mechanics, no sensing)"
        if acquisition is None
        else "base_neural_model end-to-end (activity -> mechanics -> acoustic detection)"
    )
    provenance = merge(
        ts.provenance,
        static.provenance,
        source=prov_source,
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
        detection=detection,
        mechanisms=mechanisms,
    )


def run_motor_cortex(
    *,
    low_beta: bool = False,
    duration_s: float = 8.0,
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

    ``low_beta`` selects the low-beta E/I preset (:meth:`EIParams.motor_cortex_low_beta`,
    ~13-17 Hz) instead of the default high-beta one (~20-25 Hz); the mechanics and voxel
    are identical, only the rhythm differs (Kilavik et al. 2013).
    """
    ei = (
        EIParams.motor_cortex_low_beta() if low_beta else EIParams.motor_cortex()
    )
    return run_neural_model(
        ei,
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
    and rebounds above baseline afterwards.

    **This is a dynamics visualizer: the deliverable is the timeseries, not a verdict.**
    The product is ``report.displacement_timeseries`` -- the ``dz(t)`` curve showing
    baseline -> desync dip -> rebound -- and the directional channel it carries
    (``report.mechanical_displacement.directional_axial_m``, the columnar M1 signal).

    The reduced ``NeuralState`` on the report is **not** a summary of the event: it is a
    whole-record *average* synchrony with the content-band carrier ``f_c``, and it
    exists only because the transduction chain needs a content corner to stamp the
    jitter low-pass onto the timeseries. Because the trial is an event (synchrony dips
    and rebounds), that time-average describes no instant the trial actually passed
    through. **Consequently the pass/fail gate fields (``passes_amplitude_gate``,
    ``passes_content_gate``, ``passes_dilatation_gate``) are not meaningful for a
    movement trial** -- they score a fictional whole-record steady state. Read the
    detectability verdict from the steady-state entry points (:func:`run_motor_cortex`,
    :func:`run_motor_demo`), not from a movement trial.
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


def run_motor_demo(
    *,
    acquisition: AcquisitionParams | None = None,
    residual_clutter_m: float = 0.0,
    mechanism_params: MechanismParams | None = None,
    burst_occupancy: float = 0.2,
    burst_duration_s: float = 0.2,
    synchrony_percentile: float = 90.0,
    duration_s: float = 8.0,
    fs_hz: float = 2000.0,
) -> NeuralModelReport:
    """Run the **flagship-demo Gate A**, with the honest source and acoustic physics.

    The single call that *is* the demo's Stage-1 / Gate-A source sweep, rewritten to the
    honest defaults. It wires the M1 presets (beta-band E/I, columnar layer-5 mechanics
    with the honest source-transfer factors, layer-5 voxel) with a **bursty** beta drive
    (:func:`base_neural_model.activity.motor_drive.bursty_beta_drive`) -- sensorimotor
    beta is transient, not sustained, even under held demand -- and scores it through the
    honest acoustic detection layer (:meth:`AcquisitionParams.demo_motor`: burst-limited
    coherent integration, aperture decoherence, residual aberration, clutter high-pass,
    reverberation, and a safety-flagged echo SNR).

    The bursty trajectory is reduced at its **in-burst** synchrony
    (``synchrony_percentile``, default p90): a burst-locked acquisition integrates over
    the burst, so the intermittency penalty is charged once, on the acoustic side, as the
    coherence-window cap (``coherence_time_s`` = burst duration) -- NOT twice by also
    diluting the synchrony with the between-burst troughs. ``mechanism_params`` (default
    :class:`~base_neural_model.mechanics.mechanisms.MechanismParams`) adds the
    band-separated osmotic + vascular envelope terms so ``report.mechanisms`` exposes the
    content-band (direct beta) vs envelope (hemodynamic/fUS) trade; ``report.detection``
    scores the content-band term. Pass ``residual_clutter_m`` for the clutter-limited
    regime. See :meth:`AcquisitionParams.demo_motor_optimistic` for the prior baseline.
    """
    acq = acquisition or AcquisitionParams.demo_motor()
    drive = bursty_beta_drive(occupancy=burst_occupancy, burst_duration_s=burst_duration_s)
    return run_neural_model(
        EIParams.motor_cortex(),
        geom=VoxelGeometry.motor_cortex_layer5(),
        mechanics=MechanicsParams.motor_cortex(),
        duration_s=duration_s,
        fs_hz=fs_hz,
        acquisition=acq,
        residual_clutter_m=residual_clutter_m,
        synchrony_percentile=synchrony_percentile,
        mechanism_params=mechanism_params or MechanismParams(),
        drive_fn=drive,
    )
