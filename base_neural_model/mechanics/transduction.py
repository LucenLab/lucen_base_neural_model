"""Transduction chain: neural activity -> net axial voxel displacement.

The placeholder this replaces summed single-neuron *membrane* displacements as
collinear translations. That is a category error: surface displacements of
randomly-arranged, radially-expanding cells have zero net translation (the
dipole moment cancels); the surviving term is the **monopole**, the volume
change. The interrogation beam reads the net axial *dilatation* across the gate,
not a sum of cellular excursions.

The corrected chain runs through the volume change at each link (source physics
doc, section 2)::

    membrane Delta r
      -> per-cell fractional volume change   3*Delta r / r          (section 2.1)
      -> tissue volumetric strain  eps_V = f_cell * s * (3*Delta r / r)  (2.2)
      -> axial displacement        Delta z = eta * kappa * L * eps_V       (2.3)

Two limits the corrected coherence invariant must reproduce (section 6):

* fully coherent (``s = 1``): ``eps_V = f_cell * (3*Delta r / r)`` - the analog
  of the placeholder's ``N*d_1`` ceiling, but a *strain*, not a heave of tens of
  micrometres;
* fully incoherent (``s = 0``): the coherent term vanishes and a ``1/sqrt(N)``
  fluctuating pedestal remains - and that pedestal is *noise*, not signal.

Firing-time jitter ``sigma_t`` low-passes the population signal; the content band
is its first casualty (section 3). The output is therefore a content-band curve,
carrying its jitter-determined survival explicitly, never a scalar.

The monopole above is orientation-free (a volume change has no direction), which is
why the isotropic chain survives a randomly-arranged population. But a real cell does
not expand as a perfect sphere: its eigenstrain has a *deviatoric* (directional) part
along the cell axis. That **directional channel** (``orientation.py``) adds an axial
term that survives only when the population's axes are *aligned* - a coherence axis
distinct from temporal synchrony ``s``. It is governed by the anisotropy fraction
``beta`` and the orientation order parameter ``Q``, and vanishes (recovering the
monopole-only model) when either is zero. The total axial displacement is the
isotropic plus the directional term.
"""

from __future__ import annotations

import math

import numpy as np

from base_neural_model.base.bands import require_content_fast
from base_neural_model.base.provenance import extend
from base_neural_model.base.types import (
    MechanicalDisplacement,
    MechanicsParams,
    NeuronDisplacement,
    VoxelGeometry,
)
from base_neural_model.mechanics.orientation import (
    directional_axial_strain,
    orientation_provenance,
)

_CHAIN_ASSUMPTIONS = (
    "source term is the net axial dilatation across the gate, NOT a sum of "
    "membrane displacements (the collinear sum cancels; the monopole survives)",
    "per-cell fractional volume change = 3*Delta r / r (uniform radial expansion "
    "of a spherical cell)",
    "coherent tissue strain eps_V = f_cell * s * (3*Delta r / r); the (1-s) "
    "incoherent remainder adds in RMS as an in-band noise pedestal, not signal",
    "axial displacement Delta z = eta * kappa * L * eps_V; kappa partitions "
    "dilatation axially (Eshelby inclusion), eta is the net-dilatation fraction "
    "and is the load-bearing source-side unknown routed to Stage 1",
    "content-band survival s(f_c) = exp(-2*pi^2*f_c^2*sigma_t^2): firing jitter "
    "is a low-pass on the population signal, extinguishing content before envelope",
)


def _fractional_volume_change(params: MechanicsParams) -> float:
    """Per-cell fractional volume change ``3*Delta r / r`` (section 2.1)."""
    return 3.0 * params.membrane_disp_m / params.cell_radius_m


def _volumetric_strain_coh(params: MechanicsParams, s: float) -> float:
    """Coherent voxel volumetric strain ``f_cell * s * (3*Delta r / r)`` (2.2)."""
    return params.cell_volume_fraction * s * _fractional_volume_change(params)


def _volumetric_strain_incoh(params: MechanicsParams, n: int, s: float) -> float:
    """Incoherent strain pedestal ``f_cell * sqrt((1-s)/N) * (3*Delta r / r)`` (2.2).

    The cells that fire but are not phase-locked contribute a fluctuating strain
    whose RMS scales as ``sqrt(count)``; per voxel that is the ``1/sqrt(N)``
    floor. This lands in the detection band as *noise*, not signal.
    """
    return (
        params.cell_volume_fraction
        * math.sqrt((1.0 - s) / n)
        * _fractional_volume_change(params)
    )


def _axial_from_strain(
    eta: float, kappa: float, gate_len_m: float, eps_v: float
) -> float:
    """Net axial displacement ``eta * kappa * L * eps_V`` across the gate (2.3)."""
    return eta * kappa * gate_len_m * eps_v


def _saturate(eps: float, saturation_strain: float | None) -> float:
    """Soft cap on the coherent strain ``eps / (1 + |eps|/eps_sat)`` (S7).

    A Michaelis-Menten-like saturation: ECS shrinkage and membrane interaction bound how
    large the coherent volumetric strain can grow. For ``eps << eps_sat`` it returns
    ``~eps`` (the direct neuromechanical strain sits far below any plausible cap, so this
    is nearly inert -- which is itself the finding); for ``eps >> eps_sat`` it saturates
    toward ``eps_sat``. ``None`` disables the cap (the prior strictly-linear model).
    """
    if saturation_strain is None:
        return eps
    return eps / (1.0 + abs(eps) / saturation_strain)


def _content_band_survival(sigma_t: float, f_c: float) -> float:
    """Jitter low-pass ``exp(-2*pi^2*f_c^2*sigma_t^2)`` at the corner (section 3).

    This is the characteristic function of Gaussian firing-time jitter evaluated
    at the content-band corner: 1 at the slow envelope, low in the content band.
    """
    return math.exp(-2.0 * math.pi**2 * f_c**2 * sigma_t**2)


def mechanical_displacement(
    d_single: NeuronDisplacement,
    geom: VoxelGeometry,
    params: MechanicsParams,
    synchrony_fraction: float,
) -> MechanicalDisplacement:
    """Compute the net axial source displacement at one synchrony value.

    Realizes the boxed relations of source physics doc section 2-3. The cited
    membrane displacement ``d_single`` feeds the volume relation via
    ``params.membrane_disp_m`` (which it must equal) - never a summation
    (Invariant 2). The range gate ``L`` is ``geom.extent_axial_m``.

    Precondition: ``d_single.band is CONTENT_FAST`` (enforced; raises otherwise).
    Postcondition: ``volumetric_strain`` equals ``f_cell*(3*Delta r/r)`` at
    ``s=1`` and the ``1/sqrt(N)`` incoherent floor at ``s=0`` (corrected
    coherence invariant, section 6).
    """
    require_content_fast(d_single.band, context="mechanical_displacement")

    if not 0.0 <= synchrony_fraction <= 1.0:
        raise ValueError(
            f"synchrony_fraction must be in [0, 1], got {synchrony_fraction!r}"
        )
    if geom.neuron_count <= 0:
        raise ValueError(f"neuron_count must be positive, got {geom.neuron_count!r}")
    if params.cell_radius_m <= 0.0:
        raise ValueError(f"cell_radius_m must be positive, got {params.cell_radius_m!r}")
    if d_single.value_m != params.membrane_disp_m:
        raise ValueError(
            "params.membrane_disp_m must equal the cited d_single.value_m "
            f"({d_single.value_m!r}); the cited constant enters only here "
            f"(Invariant 2), got {params.membrane_disp_m!r}"
        )

    if not 0.0 < params.viscoelastic_factor <= 1.0:
        raise ValueError(
            f"viscoelastic_factor must lie in (0, 1], got {params.viscoelastic_factor!r}"
        )
    if not 0.0 <= params.carrier_modulation_depth <= 1.0:
        raise ValueError(
            "carrier_modulation_depth must lie in [0, 1], got "
            f"{params.carrier_modulation_depth!r}"
        )
    if not 0.0 < params.correlation_coherent_fraction <= 1.0:
        raise ValueError(
            "correlation_coherent_fraction must lie in (0, 1], got "
            f"{params.correlation_coherent_fraction!r}"
        )
    if params.saturation_strain is not None and params.saturation_strain <= 0.0:
        raise ValueError(
            f"saturation_strain must be positive or None, got {params.saturation_strain!r}"
        )
    if not 0.0 <= params.deviatoric_eta <= 1.0:
        raise ValueError(
            f"deviatoric_eta must lie in [0, 1], got {params.deviatoric_eta!r}"
        )

    n = geom.neuron_count
    s = synchrony_fraction
    gate_len_m = geom.extent_axial_m

    eps_coh_raw = _volumetric_strain_coh(params, s)
    eps_incoh = _volumetric_strain_incoh(params, n, s)

    # Honest source-physics attenuations of the coherent term, each once and each
    # reducing to unity at its default: the viscoelastic transfer |H(f_c)| (S1, the
    # tissue is not a static spring), the beta-carrier modulation depth (S3, only the
    # rate-modulated fraction of the per-spike swelling lives at the carrier), and the
    # mutually-phase-coherent fraction (S6, only cells within a correlation length add
    # coherently). Then the strain saturation cap (S7). The incoherent pedestal is left
    # as the raw in-band noise floor (it is not the binding detection floor).
    source_transfer = (
        params.viscoelastic_factor
        * params.carrier_modulation_depth
        * params.correlation_coherent_fraction
    )
    eps_coh = _saturate(eps_coh_raw * source_transfer, params.saturation_strain)

    # Isotropic (monopole / volume-change) axial term -- the original signal.
    axial_iso = _axial_from_strain(
        params.dilatation_eta, params.confinement_kappa, gate_len_m, eps_coh
    )
    axial_incoh = _axial_from_strain(
        params.dilatation_eta, params.confinement_kappa, gate_len_m, eps_incoh
    )

    # Directional (deviatoric / orientation) axial term. It uses the deviatoric
    # Eshelby response in place of kappa and the population orientation factor; it is
    # zero when beta = 0 or the orientation is random (Q = 0), recovering the
    # isotropic model exactly. It carries the same gate length as the monopole, but its
    # OWN net-realization fraction ``deviatoric_eta`` (default 1.0), NOT the volumetric
    # ``dilatation_eta``: the deviatoric part is a constant-volume shape change, so the
    # poroelastic drainage discount that gates the VOLUME term does not apply to it.
    eps_dir = directional_axial_strain(
        eps_coh,
        anisotropy=params.anisotropy,
        order_parameter=params.orientation_coherence,
        director_projection=params.mean_axis_projection,
        nu=params.matrix_poisson_ratio,
    )
    axial_dir = params.deviatoric_eta * gate_len_m * eps_dir

    # The beam reads the magnitude of the total axial coherent displacement.
    axial_total = abs(axial_iso + axial_dir)

    survival = _content_band_survival(params.jitter_sigma_s, params.content_freq_hz)

    chain_provenance = [*_CHAIN_ASSUMPTIONS]
    if params.anisotropy > 0.0 and params.orientation_coherence > 0.0:
        chain_provenance.extend(
            orientation_provenance(
                anisotropy=params.anisotropy,
                order_parameter=params.orientation_coherence,
                director_projection=params.mean_axis_projection,
                nu=params.matrix_poisson_ratio,
            )
        )
    provenance = extend(
        d_single.provenance,
        *chain_provenance,
        f"voxel neuron_count N = {n}",
        f"range gate L = {gate_len_m} m",
        f"kappa = {params.confinement_kappa}, eta = {params.dilatation_eta}, "
        f"f_cell = {params.cell_volume_fraction}, r = {params.cell_radius_m} m",
        f"sigma_t = {params.jitter_sigma_s} s, f_c = {params.content_freq_hz} Hz",
        f"honest source-physics coherent-term attenuation = {source_transfer:.4g} "
        f"(viscoelastic |H| {params.viscoelastic_factor}, carrier depth "
        f"{params.carrier_modulation_depth}, coherent fraction "
        f"{params.correlation_coherent_fraction}; S1/S3/S6), saturation "
        f"{params.saturation_strain} (S7): coherent strain {eps_coh_raw:.4g} -> "
        f"{eps_coh:.4g}",
        f"axial decomposition: isotropic {axial_iso:.4g} m (volumetric, eta = "
        f"{params.dilatation_eta}) + directional {axial_dir:.4g} m (deviatoric shape "
        f"change, deviatoric_eta = {params.deviatoric_eta}, NOT gated by the poroelastic "
        f"drainage eta; Q = {params.orientation_coherence})",
    )
    return MechanicalDisplacement(
        axial_displacement_m=axial_total,
        volumetric_strain=eps_coh,
        incoherent_pedestal_m=axial_incoh,
        synchrony_fraction=s,
        jitter_sigma_s=params.jitter_sigma_s,
        content_band_survival=survival,
        confinement_kappa=params.confinement_kappa,
        dilatation_eta=params.dilatation_eta,
        band=d_single.band,
        provenance=provenance,
        isotropic_axial_m=axial_iso,
        directional_axial_m=axial_dir,
        orientation_coherence=params.orientation_coherence,
        source_transfer_factor=source_transfer,
    )


def displacement_sweep(
    d_single: NeuronDisplacement,
    geom: VoxelGeometry,
    params: MechanicsParams,
    synchrony_grid: np.ndarray,
) -> list[MechanicalDisplacement]:
    """Vectorized sweep over synchrony - the primary Module 1 product.

    Coherence (synchrony fraction) is the swept independent variable (Invariant
    4): the output is a *curve*, never a single point. ``synchrony_grid`` is e.g.
    ``np.linspace(0, 1, 51)`` and must lie within ``[0, 1]``.
    """
    require_content_fast(d_single.band, context="displacement_sweep")

    grid = np.asarray(synchrony_grid, dtype=float)
    if grid.ndim != 1:
        raise ValueError(f"synchrony_grid must be 1-D, got shape {grid.shape}")
    if grid.size == 0:
        raise ValueError("synchrony_grid is empty")
    if grid.min() < 0.0 or grid.max() > 1.0:
        raise ValueError("synchrony_grid values must lie within [0, 1]")

    return [mechanical_displacement(d_single, geom, params, float(s)) for s in grid]
