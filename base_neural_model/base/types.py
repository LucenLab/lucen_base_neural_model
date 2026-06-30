"""Shared data contracts for the neural model.

These are the typed objects that cross the activity -> mechanics -> gates boundary -
the actual interface. Get them right and the layers can be built and tested
independently.

* ``NeuronDisplacement`` - the cited single-neuron membrane displacement constant.
* ``VoxelGeometry`` - the tissue voxel whose displacement we compute.
* ``MechanicsParams`` - the cell/material inputs to the transduction chain, each a
  quantity a reviewer is meant to contest in isolation.
* ``NeuralState`` - the reduced summary of the dynamical activity layer (synchrony,
  jitter, content corner, mean rate) that the mechanics consumes. It is the bridge
  between the activity timeseries and the static displacement chain.
* ``MechanicalDisplacement`` - the deliverable of the transduction chain: the net
  axial dilatation across the gate, NOT a sum of membrane displacements (that
  quantity physically cancels). It carries the decomposed chain so the number traces
  back to its factors.

The activity and displacement *timeseries* contracts (which hold NumPy arrays) live
in their owning modules (``activity.timeseries``, ``mechanics.timeseries``) and use
``frozen=True, eq=False`` so the array fields do not generate an ambiguous,
unhashable ``__eq__``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from base_neural_model.base.bands import Band
from base_neural_model.base.provenance import Provenance


@dataclass(frozen=True)
class NeuronDisplacement:
    """Single-neuron membrane displacement - a CITED CONSTANT (Invariant 2).

    No module derives this; it enters the system as one number with literature
    provenance. ``band`` MUST be ``CONTENT_FAST`` to feed the population model.
    """

    value_m: float              # metres; ~1e-9 to 3e-9
    band: Band
    provenance: Provenance


@dataclass(frozen=True)
class VoxelGeometry:
    """The voxel whose source displacement we compute."""

    extent_axial_m: float       # ~3e-4 (range-gate length)
    extent_lateral_m: float     # ~1e-3 (focal width)
    neuron_count: int           # ~1e4-1e5 in a mm-scale voxel
    depth_m: float              # cortical depth, for propagation path length

    @classmethod
    def motor_cortex_layer5(cls) -> VoxelGeometry:
        """A voxel placed at primary motor cortex (M1) layer 5 (somatotopic depth).

        Layer 5 - where the large pyramidal / Betz output cells sit - is deep in the
        ~3 mm M1 cortical sheet, ~1.5-2 mm below the pial surface. M1 is notably the
        thickest cortical area. The depth is set to the layer-5 band so the geometry
        reflects where the motor output cells actually are.

        The neuron count is set for **consistency with the large layer-5 somata**: at
        the Betz-cell radius (~20 um, :meth:`MechanicsParams.motor_cortex`) a 0.3x1x1
        mm voxel holds far fewer cells than the generic 10k at ~15% volume fraction -
        about 1300. Large output cells are sparse; using the generic count here would
        imply a volume fraction above 1 (cells overfilling the voxel), the exact
        nm/um-class inconsistency ``check_volume_fraction_consistency`` guards against.
        """
        return cls(
            extent_axial_m=3e-4,
            extent_lateral_m=1e-3,
            neuron_count=1_300,   # sparse large layer-5 cells (consistent with r~20um)
            depth_m=1.8e-3,       # ~1.8 mm: M1 layer 5 below the pial surface
        )

    @property
    def volume_m3(self) -> float:
        """Voxel volume, metres^3.

        Assumes a box of ``axial x lateral x lateral`` (the source physics doc's
        0.3 x 1 x 1 mm voxel). ``depth_m`` is the propagation path length, NOT a
        third voxel extent, so it is deliberately not used here.
        """
        return self.extent_axial_m * self.extent_lateral_m**2

    def cell_volume_fraction(self, cell_radius_m: float) -> float:
        """Cell volume fraction ``f_cell = N * V_cell / V_voxel`` implied by the
        neuron count and a cell radius.

        This makes the ``neuron_count`` / ``cell_volume_fraction`` redundancy
        explicit: a caller can derive ``f_cell`` from the geometry instead of
        setting it independently, or compare the two to catch an inconsistency.
        """
        if cell_radius_m <= 0.0:
            raise ValueError(f"cell_radius_m must be positive, got {cell_radius_m!r}")
        cell_volume_m3 = (4.0 / 3.0) * math.pi * cell_radius_m**3
        return self.neuron_count * cell_volume_m3 / self.volume_m3


@dataclass(frozen=True)
class MechanicsParams:
    """Cell- and material-scale inputs to the transduction chain.

    Each field is a quantity a reviewer is meant to contest in isolation. The cited
    membrane displacement ``Delta r`` enters here and nowhere else (Invariant 2).
    ``kappa`` and ``eta`` are the two load-bearing unknowns; ``eta`` (the
    net-dilatation fraction) is the deepest source-side risk.
    """

    membrane_disp_m: float       # Delta r: radial membrane displacement, metres
    cell_radius_m: float         # r: characteristic cell radius, metres
    cell_volume_fraction: float  # f_cell = n * V_cell, dimensionless in [0, 1]
    confinement_kappa: float     # kappa: axial partition of dilatation, [1/3, 1]
    dilatation_eta: float        # eta: net-dilatation fraction, [0, 1]
    jitter_sigma_s: float        # sigma_t: firing-time jitter SD, seconds
    content_freq_hz: float       # f_c: content-band corner frequency, Hz
    # --- Directional (deviatoric) channel (orientation.py) ----------------------
    # Per-cell expansion is not a perfect sphere: a deviatoric part along the cell
    # axis adds a DIRECTIONAL term that survives only if the population's axes are
    # aligned. These default to the isotropic (monopole-only) model, so any
    # MechanicsParams that omits them reduces exactly to the volume-change chain.
    anisotropy: float = 0.0              # beta: per-cell deviatoric fraction, [0, 1]
    orientation_coherence: float = 0.0   # Q: orientation order parameter, [0, 1]
    mean_axis_projection: float = 1.0    # mu = (director . beam axis), [-1, 1]
    matrix_poisson_ratio: float = 0.2    # nu for the deviatoric Eshelby response

    @classmethod
    def central(cls) -> MechanicsParams:
        """Central-column defaults (source physics doc, section 5 table).

        The directional channel defaults to isotropic (``anisotropy = 0``), so the
        central params reproduce the pure volume-change model; set ``anisotropy`` and
        ``orientation_coherence`` to activate it.
        """
        return cls(
            membrane_disp_m=1.5e-9,
            cell_radius_m=8e-6,
            cell_volume_fraction=0.15,
            confinement_kappa=0.5,
            dilatation_eta=0.5,
            jitter_sigma_s=1.0e-3,
            content_freq_hz=100.0,
        )

    @classmethod
    def motor_cortex(cls) -> MechanicsParams:
        """Primary motor cortex (M1) layer-5 cell/material parameters.

        The directional channel is the point of the motor variant. M1 layer-5
        pyramidal cells - the giant Betz cells especially - are large and strongly
        elongated along the apical-dendrite (columnar) axis, so the per-cell
        eigenstrain has a large **deviatoric** part (high ``anisotropy``) along an
        axis the population shares. With the beam interrogating roughly along the
        cortical column, ``mean_axis_projection`` is near 1, so the directional term
        adds to the axial signal. The orientation coherence ``Q`` itself is supplied
        by the activity layer (``EIParams.motor_cortex`` sets the structural
        alignment); it is left at 0 here so a static, activity-free construction stays
        isotropic until the dynamics fill it in.

        ``content_freq_hz`` is set to the beta band (the M1 content rhythm); a larger
        ``cell_radius_m`` reflects the large layer-5 somata.
        """
        return cls(
            membrane_disp_m=1.5e-9,
            cell_radius_m=20e-6,        # large layer-5 / Betz somata (vs ~8 um generic)
            cell_volume_fraction=0.15,
            confinement_kappa=0.5,
            dilatation_eta=0.5,
            jitter_sigma_s=1.0e-3,
            content_freq_hz=20.0,       # beta-band content corner
            anisotropy=0.6,             # elongated, columnar cells -> deviatoric strain
            mean_axis_projection=1.0,   # beam ~ along the cortical column
            matrix_poisson_ratio=0.2,
        )

    def check_volume_fraction_consistency(
        self, geom: VoxelGeometry, *, rel_tol: float = 0.1
    ) -> None:
        """Opt-in: assert the asserted ``cell_volume_fraction`` agrees with the
        value the geometry implies (``f_cell = N * V_cell / V_voxel``).

        ``cell_volume_fraction`` and ``neuron_count`` are partly-redundant knobs
        (source physics doc, section 2.2): the coherent strain depends on
        ``f_cell``, while ``neuron_count`` sets only the incoherent pedestal. They
        are kept independent so a reviewer can contest each in isolation - this
        check is therefore opt-in, not enforced in the constructor. Raises
        ``ValueError`` if the two disagree by more than ``rel_tol`` (default 10%).
        """
        derived = geom.cell_volume_fraction(self.cell_radius_m)
        if not math.isclose(self.cell_volume_fraction, derived, rel_tol=rel_tol):
            raise ValueError(
                f"cell_volume_fraction={self.cell_volume_fraction} disagrees with "
                f"the value implied by the geometry ({derived:.4g}) at "
                f"r={self.cell_radius_m} m, N={geom.neuron_count}, "
                f"V_voxel={geom.volume_m3:.4g} m^3 (rel_tol={rel_tol}). "
                "Reconcile f_cell, neuron_count, or cell_radius_m."
            )


@dataclass(frozen=True)
class MechanicalDisplacement:
    """The transduction-chain deliverable at one synchrony value.

    The net axial dilatation across the gate, content-band -- NOT a sum of membrane
    displacements. Carries the decomposed transduction chain so the deliverable
    traces back to its factors (source physics doc, section 6).
    """

    axial_displacement_m: float    # Delta z_coh: TOTAL axial term (iso + directional)
    volumetric_strain: float       # eps_V(s), pre-confinement, for tracing
    incoherent_pedestal_m: float   # Delta z_incoh: source-intrinsic in-band noise
    synchrony_fraction: float      # s: 0.0 (incoherent) .. 1.0 (fully coherent)
    jitter_sigma_s: float          # sigma_t, sets content-band survival
    content_band_survival: float   # s(f_c) = exp(-2*pi^2*f_c^2*sigma_t^2)
    confinement_kappa: float       # kappa in [1/3, 1]
    dilatation_eta: float          # eta in [0, 1] -- the load-bearing unknown
    band: Band                     # MUST be CONTENT_FAST
    provenance: Provenance
    # Directional decomposition of the axial term. The isotropic (monopole) part is
    # the original volume-change signal; the directional part is the deviatoric
    # contribution that survives only with orientation coherence. They default so an
    # isotropic construction has all the signal in the isotropic part.
    isotropic_axial_m: float = 0.0     # the volume-change (monopole) axial term
    directional_axial_m: float = 0.0   # the deviatoric (orientation) axial term, signed
    orientation_coherence: float = 0.0  # Q used, [0, 1]

    @property
    def value_m(self) -> float:
        """The source term, as one obvious field (= ``axial_displacement_m``)."""
        return self.axial_displacement_m


@dataclass(frozen=True)
class NeuralState:
    """Reduced summary of the dynamical activity layer (the activity -> mechanics bridge).

    The dynamical neural-mass model produces firing, oscillations, and synchrony
    *over time* (``activity.ActivityTimeseries``); ``reduce_to_state`` collapses that
    trajectory to the scalars the static transduction chain consumes. A caller can
    also build a ``NeuralState`` directly to drive the mechanics without integrating
    the ODEs.

    ``band`` MUST be ``CONTENT_FAST``: the mechanics target the content band, never
    the slow envelope (Invariant 3).
    """

    synchrony_fraction: float    # s in [0, 1]: TEMPORAL population coherence (Kuramoto r)
    jitter_sigma_s: float        # sigma_t: spike-timing jitter SD, seconds
    content_freq_hz: float       # f_c: content-band corner frequency, Hz
    mean_firing_rate_hz: float   # population mean firing rate, Hz
    band: Band                   # MUST be CONTENT_FAST
    provenance: Provenance
    # DIRECTIONAL coherence (Q in [0, 1]): how aligned the cells' displacement axes
    # are in 3-D - independent of temporal synchrony. ``None`` means the activity
    # layer does not yet produce it (Option 1: it is hardcoded on MechanicsParams);
    # a value here (Option 2) lets the dynamics drive it and is stamped onto the chain.
    orientation_coherence: float | None = None

    def to_mechanics_params(self, base: MechanicsParams) -> MechanicsParams:
        """Stamp this state's activity-derived fields onto ``base``.

        ``synchrony_fraction`` (temporal) is passed separately to the chain (it is the
        swept argument), but ``jitter_sigma_s`` and ``content_freq_hz`` are activity
        outputs that belong on :class:`MechanicsParams`. If this state carries an
        ``orientation_coherence`` (Option 2, the activity layer producing directional
        alignment), it is stamped too; otherwise the base params' own value is left
        untouched (Option 1, hardcoded). The cell/material factors a reviewer contests
        (kappa, eta, beta, f_cell, Delta r) are left untouched.
        """
        from dataclasses import replace

        updates: dict = {
            "jitter_sigma_s": self.jitter_sigma_s,
            "content_freq_hz": self.content_freq_hz,
        }
        if self.orientation_coherence is not None:
            updates["orientation_coherence"] = self.orientation_coherence
        return replace(base, **updates)
