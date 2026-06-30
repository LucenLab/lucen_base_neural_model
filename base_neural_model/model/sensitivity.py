"""Global Sobol sensitivity -- the central in-silico test (source doc, PART 2).

The deliverable of Module 1 is not a number but *uncertainty compression*: the
model must DEMONSTRATE that the verdict collapses onto eta (net-dilatation fraction)
and content-band synchrony s, and NOT onto geometry or cited constants. This module
runs a variance-based global sensitivity analysis (Sobol first-order S1 and
total-order ST, Saltelli sampling) over the source inputs and turns that assertion
into a passing measurement: ST concentrates on the eta-driver (permeability k) and
on s, while Delta r, r, f_cell and N get near-zero total order.

SALib does the sampling and the index estimation; this module supplies the model
output -- the scalar verdict each parameter vector produces through the
:func:`build_mechanics_params` -> :func:`displacement_sweep` chain.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

from base_neural_model.activity.synchrony import synchrony_from_drive
from base_neural_model.base.bands import Band
from base_neural_model.base.provenance import Provenance
from base_neural_model.base.types import NeuronDisplacement, VoxelGeometry
from base_neural_model.mechanics.derive import build_mechanics_params
from base_neural_model.mechanics.eshelby import MatrixParams
from base_neural_model.mechanics.poroelastic import (
    PERMEABILITY_MAX,
    PERMEABILITY_MIN,
    PoroelasticParams,
)
from base_neural_model.mechanics.transduction import mechanical_displacement

# The canonical swept factor space. The two unmeasured fulcra are eta (via
# log10_permeability, ~3 decades, genuinely unconstrained at ms/focal scale) and
# content-band synchrony s (the swept axis). The remaining factors -- the matrix
# Poisson ratio (kappa-driver) plus the cited constant and the geometry -- are
# "measured-and-transferable": the literature pins each to a NARROW range. The
# collapse the Sobol test demonstrates is a consequence of that asymmetry: each
# factor's bound is its literature uncertainty, so variance flows to the wide,
# unmeasured axes. This is the doc's central argument made quantitative -- NOT a
# claim that eta/s have privileged partial derivatives in a multiplicative chain.
DEFAULT_FACTORS: tuple[str, ...] = (
    "poisson_ratio",      # -> kappa  (measured-transferable: undrained nu_u, Su 2023)
    "log10_permeability",  # -> eta  (UNMEASURED: ~3 decades, the net-dilatation axis)
    "synchrony",          # s         (the swept content-band axis)
    "membrane_disp_nm",   # cited constant (measured-transferable: ~1-3 nm)
    "cell_radius_um",     # geometry (measured-transferable)
    "cell_volume_fraction",  # geometry (measured-transferable)
    "neuron_count_k",     # geometry (measured-transferable)
)

DEFAULT_BOUNDS: tuple[tuple[float, float], ...] = (
    (0.45, 0.49),                              # nu: undrained fast band, tight (Su 2023)
    (np.log10(PERMEABILITY_MIN), np.log10(PERMEABILITY_MAX)),  # log10 k: ~3 decades, WIDE
    (0.0, 1.0),                                # synchrony: full sweep, WIDE
    (1.8, 2.2),                                # membrane_disp_nm: cited ~2 nm, tight
    (7.0, 9.0),                                # cell_radius_um: tight
    (0.13, 0.17),                              # cell_volume_fraction: tight
    (15.0, 25.0),                              # neuron_count_k (thousands): tight
)

# Fixed poroelastic context for the eta sweep (the parts not swept as factors).
_FIXED_PORO = {
    "porosity": 0.2,
    "storage_modulus_Pa": 1e9,   # ~ water-like bulk modulus
    "gate_time_s": 1e-3,         # fast/content gate
}
_FIXED_MISC = {
    "jitter_sigma_s": 1e-3,
    "content_freq_hz": 100.0,
}


@dataclass(frozen=True)
class SobolProblem:
    """A named, bounded factor space for the Sobol analysis."""

    names: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.names) != len(self.bounds):
            raise ValueError("names and bounds must have equal length")
        for lo, hi in self.bounds:
            if not lo < hi:
                raise ValueError(f"each bound must be lo < hi, got ({lo}, {hi})")

    @classmethod
    def default(cls) -> SobolProblem:
        """The canonical source factor space (eta/kappa/s + geometry controls)."""
        return cls(names=DEFAULT_FACTORS, bounds=DEFAULT_BOUNDS)

    @classmethod
    def with_activity(cls) -> SobolProblem:
        """The factor space spanning the ACTIVITY layer as well as the mechanics.

        Replaces the bare ``synchrony`` factor with the upstream activity drivers
        ``drive_e`` (the E/I excitatory drive) and ``phase_spread_hz`` (the firing
        heterogeneity), from which synchrony is derived through the activity layer's
        drive->synchrony surrogate. The verdict-collapse test then spans the whole
        model: variance still flows to the net-dilatation (eta) axis and the drive
        that sets synchrony, not to geometry or the cited constant.
        """
        names = tuple(n for n in DEFAULT_FACTORS if n != "synchrony") + (
            "drive_e",
            "phase_spread_hz",
        )
        bounds = tuple(
            b for n, b in zip(DEFAULT_FACTORS, DEFAULT_BOUNDS, strict=True)
            if n != "synchrony"
        ) + (
            (0.8, 2.0),   # drive_e: spans sub- to supra-threshold (the activity axis)
            (2.0, 8.0),   # phase_spread_hz: firing heterogeneity gamma
        )
        return cls(names=names, bounds=bounds)

    def to_salib(self) -> dict:
        return {
            "num_vars": len(self.names),
            "names": list(self.names),
            "bounds": [list(b) for b in self.bounds],
        }


@dataclass(frozen=True, eq=False)
class SobolIndices:
    """Sobol indices over a :class:`SobolProblem` (ndarrays -> eq=False).

    ``eq=False`` mirrors the array-contract convention in ``base.types``: a frozen
    dataclass holding ndarrays cannot have a generated ``__eq__``/``__hash__``.
    """

    names: tuple[str, ...]
    S1: np.ndarray
    ST: np.ndarray
    n_base: int
    provenance: Provenance

    def dominant(self, *, total_order: bool = True, top: int = 2) -> tuple[str, ...]:
        """The ``top`` factors by index magnitude (ST by default)."""
        idx = self.ST if total_order else self.S1
        order = np.argsort(idx)[::-1]
        return tuple(self.names[i] for i in order[:top])

    def as_dict(self, *, total_order: bool = True) -> dict[str, float]:
        idx = self.ST if total_order else self.S1
        return {name: float(v) for name, v in zip(self.names, idx, strict=True)}


def model_verdict_scalar(
    sample_row: Mapping[str, float] | np.ndarray,
    names: tuple[str, ...],
    d_single: NeuronDisplacement,
    voxel: VoxelGeometry,
) -> float:
    """Model output for one parameter vector: the Gate-1 proxy verdict scalar.

    Maps a swept parameter vector through the kappa/eta sub-models and the
    transduction chain, then reduces the result to one number the variance
    decomposition can act on: the jitter-surviving coherent axial displacement,
    ``axial_displacement_m * content_band_survival`` (the Gate-1 amplitude proxy,
    used because Modules 2/3 are not yet implemented). Synchrony is itself a swept
    factor, so the scalar is evaluated at that row's synchrony, not a sweep peak.
    """
    if not isinstance(sample_row, Mapping):
        sample_row = dict(zip(names, np.asarray(sample_row, dtype=float), strict=True))

    # Synchrony is either a direct factor (mechanics-only space) or derived from the
    # activity drive (the with_activity space), so the verdict spans both layers.
    if "synchrony" in sample_row:
        synchrony = float(sample_row["synchrony"])
    else:
        synchrony = synchrony_from_drive(
            float(sample_row["drive_e"]),
            phase_spread_hz=float(sample_row["phase_spread_hz"]),
        )

    matrix = MatrixParams(poisson_ratio=float(sample_row["poisson_ratio"]))
    permeability = 10.0 ** float(sample_row["log10_permeability"])
    poro = PoroelasticParams(
        permeability_m4_per_Ns=permeability,
        porosity=_FIXED_PORO["porosity"],
        poro_diffusivity_m2_per_s=permeability * _FIXED_PORO["storage_modulus_Pa"],
        gate_time_s=_FIXED_PORO["gate_time_s"],
        drainage_length_m=voxel.extent_axial_m,
    )
    membrane_disp_m = float(sample_row["membrane_disp_nm"]) * 1e-9
    params = build_mechanics_params(
        matrix=matrix,
        poro=poro,
        membrane_disp_m=membrane_disp_m,
        cell_radius_m=float(sample_row["cell_radius_um"]) * 1e-6,
        cell_volume_fraction=float(sample_row["cell_volume_fraction"]),
        jitter_sigma_s=_FIXED_MISC["jitter_sigma_s"],
        content_freq_hz=_FIXED_MISC["content_freq_hz"],
    )

    # The cited constant enters the chain only via membrane_disp_m; feed a matching
    # NeuronDisplacement so the chain's equality guard holds for the swept Delta r.
    neuron = NeuronDisplacement(
        value_m=membrane_disp_m, band=Band.CONTENT_FAST, provenance=d_single.provenance
    )
    geom = VoxelGeometry(
        extent_axial_m=voxel.extent_axial_m,
        extent_lateral_m=voxel.extent_lateral_m,
        neuron_count=max(1, int(round(float(sample_row["neuron_count_k"]) * 1000))),
        depth_m=voxel.depth_m,
    )
    out = mechanical_displacement(neuron, geom, params, synchrony)
    return out.axial_displacement_m * out.content_band_survival


def run_model_sobol(
    d_single: NeuronDisplacement,
    voxel: VoxelGeometry,
    *,
    problem: SobolProblem | None = None,
    n_base: int = 1024,
    seed: int | None = 0,
) -> SobolIndices:
    """Run the Sobol analysis over the source factor space.

    Saltelli-samples ``N = n_base * (2k + 2)`` parameter vectors (``n_base`` must be
    a power of two for the scrambled Sobol sequence), evaluates
    :func:`model_verdict_scalar` on each, and estimates S1/ST with SALib. The
    returned indices are the demonstration that the verdict collapses onto the
    eta-driver (``log10_permeability``) and ``synchrony``.
    """
    problem = problem or SobolProblem.default()
    salib_problem = problem.to_salib()

    samples = sobol_sample.sample(salib_problem, n_base, seed=seed)
    Y = np.array(
        [model_verdict_scalar(row, problem.names, d_single, voxel) for row in samples]
    )

    result = sobol_analyze.analyze(salib_problem, Y, seed=seed)
    s1 = np.clip(np.asarray(result["S1"], dtype=float), 0.0, None)  # noise -> 0
    st = np.clip(np.asarray(result["ST"], dtype=float), 0.0, None)

    provenance = Provenance(
        source="Sobol global sensitivity (SALib Saltelli sampling) over the "
        "Module-1 source factor space",
        assumptions=(
            "verdict scalar = jitter-surviving coherent axial displacement "
            "(Delta z * content-band survival), the Gate-1 amplitude proxy used "
            "while Modules 2/3 are stubs",
            "kappa derived from the matrix Poisson ratio (Eshelby); eta derived "
            "from the poroelastic drainage number via swept log-permeability",
            f"Saltelli base sample n = {n_base}; total evaluations "
            f"N = n*(2k+2) with k = {len(problem.names)} factors",
            "S1/ST clipped at 0 to absorb estimator sampling noise",
        ),
        band=d_single.band,
    )
    return SobolIndices(
        names=problem.names, S1=s1, ST=st, n_base=n_base, provenance=provenance
    )
