"""Eshelby confinement factor kappa, DERIVED from the matrix Poisson ratio.

The transduction chain (``population.py``) reads ``kappa`` as a bare scalar: the
fraction of a per-cell dilatational eigenstrain that survives as net constrained
dilatation once the surrounding tissue resists it. The source-physics doc (PART 2,
Module 1) requires that scalar to come from the analytical Eshelby tensor (Mura
formulation), not be asserted free, because kappa is a *strong function of the
matrix Poisson ratio* and the fast/content band sits at the incompressible end
(undrained nu -> 0.5) where confinement is fiercest.

The physics, for a dilatational (purely volumetric) eigenstrain eps* in an
ellipsoidal inclusion embedded in an isotropic matrix: the constrained strain
inside the inclusion is ``eps^c = S : eps*`` where ``S`` is the Eshelby tensor.
For a hydrostatic eigenstrain ``eps*_ij = (eps*/3) delta_ij`` the constrained
*volumetric* strain is::

    eps^c_kk / eps* = S_1111 + 2 S_1122          (sphere)

which is exactly the fraction of the per-cell volume change realized as net
tissue dilatation -- our ``kappa``. For the sphere the boxed closed forms give::

    S_1111 = (7 - 5 nu) / [15 (1 - nu)]
    S_1122 = (5 nu - 1) / [15 (1 - nu)]
    kappa  = S_1111 + 2 S_1122 = (1 + nu) / [3 (1 - nu)]

with the two limits the doc names:

* ``nu -> 0`` (fully compressible matrix, the free/unconfined limit): kappa = 1/3.
* ``nu -> 1/2`` (incompressible matrix, the fast-band case): kappa -> 1 -- a
  dilatational eigenstrain in a sphere then produces uniform hydrostatic stress
  and the volume change is forced entirely into the constrained response.

``nu = 1/2`` is a ``1/(1 - nu)`` singularity; it is approached as a limit, never
evaluated, so the matrix Poisson ratio is guarded strictly below 0.5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# kappa is the axial/volumetric partition of dilatation; physically [1/3, 1].
KAPPA_FREE: float = 1.0 / 3.0   # nu -> 0, unconfined free inclusion
KAPPA_CONFINED: float = 1.0     # nu -> 1/2, incompressible matrix


@dataclass(frozen=True)
class MatrixParams:
    """Surrounding-matrix elastic state that sets the confinement of a cell.

    ``poisson_ratio`` is the *undrained* matrix Poisson ratio in the fast band
    (Su et al. 2023: nu_u > 0.49 in the short-time regime). ``aspect_ratio`` is
    the inclusion semi-axis ratio a3/a1: 1 is a sphere, < 1 an oblate/penny
    inclusion, > 1 a prolate one.
    """

    poisson_ratio: float
    aspect_ratio: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.poisson_ratio < 0.5:
            raise ValueError(
                "poisson_ratio must lie in [0, 0.5) -- nu = 0.5 is the "
                f"incompressible singularity, approached as a limit; got "
                f"{self.poisson_ratio!r}"
            )
        if self.aspect_ratio <= 0.0:
            raise ValueError(f"aspect_ratio must be positive, got {self.aspect_ratio!r}")


def eshelby_sphere_components(nu: float) -> tuple[float, float]:
    """Return ``(S_1111, S_1122)`` for a spherical inclusion (Mura, boxed forms).

    ``S_1111 = (7 - 5 nu) / [15 (1 - nu)]``, ``S_1122 = (5 nu - 1) / [15 (1 - nu)]``.
    """
    if not 0.0 <= nu < 0.5:
        raise ValueError(f"nu must lie in [0, 0.5), got {nu!r}")
    denom = 15.0 * (1.0 - nu)
    s1111 = (7.0 - 5.0 * nu) / denom
    s1122 = (5.0 * nu - 1.0) / denom
    return s1111, s1122


def _shape_confinement_weight(aspect_ratio: float) -> float:
    """Monotone shape weight in ``(0, 1]`` toward the confined limit.

    The sphere (``a = 1``) is the reference and carries weight 1 (the closed-form
    trace stands alone). Departing from spherical -- either oblate (penny, ``a < 1``)
    or prolate (``a > 1``) -- presents a flatter or more elongated inclusion that the
    matrix confines *less* uniformly, so the realized dilatation shifts toward the
    free 1/3 partition. A symmetric, monotone-in-|log a| weight captures that without
    over-claiming a spheroidal closed form: ``w = sech(ln a)`` = 1 at the sphere,
    decaying smoothly and staying in ``(0, 1]``. The matrix-Poisson dependence (the
    load-bearing physics, exact for the sphere) is untouched; only the shape's pull
    toward the free limit is modeled.
    """
    if aspect_ratio <= 0.0:
        raise ValueError(f"aspect_ratio must be positive, got {aspect_ratio!r}")
    return 1.0 / math.cosh(math.log(aspect_ratio))


def eshelby_spheroid_trace(nu: float, aspect_ratio: float) -> float:
    """Constrained-dilatation fraction (the ``kappa`` trace) for an inclusion.

    For a sphere (``aspect_ratio = 1``) this is exactly ``S_1111 + 2 S_1122 =
    (1 + nu)/[3(1 - nu)]`` -- the doc's boxed result, the load-bearing case (the
    chain models cells as uniformly-expanding spheres). For a non-spherical
    inclusion the trace is pulled from the sphere value toward the free 1/3 partition
    by a monotone shape weight, keeping the result in ``[1/3, 1]`` and reducing to
    the sphere result at ``aspect_ratio = 1``.
    """
    if not 0.0 <= nu < 0.5:
        raise ValueError(f"nu must lie in [0, 0.5), got {nu!r}")
    sphere_trace = (1.0 + nu) / (3.0 * (1.0 - nu))
    weight = _shape_confinement_weight(aspect_ratio)
    # Interpolate between the free 1/3 partition and the sphere trace by the weight.
    return KAPPA_FREE + (sphere_trace - KAPPA_FREE) * weight


def eshelby_kappa(nu: float, aspect_ratio: float = 1.0) -> float:
    """Confinement factor ``kappa`` derived from the matrix Poisson ratio.

    The fraction of a per-cell dilatational eigenstrain realized as net constrained
    tissue dilatation -- the ``kappa`` the transduction chain consumes. For a sphere
    this is ``(1 + nu) / [3 (1 - nu)]``: 1/3 at nu = 0 (free), -> 1 at nu -> 1/2
    (incompressible, the fast-band case).

    Postcondition: the result lies in ``[1/3, 1]``. A value outside that band is a
    physics bug (bad nu/aspect input), raised rather than silently clamped.
    """
    trace = eshelby_spheroid_trace(nu, aspect_ratio)
    if not (KAPPA_FREE - 1e-9) <= trace <= (KAPPA_CONFINED + 1e-9):
        raise ValueError(
            f"derived kappa={trace!r} fell outside the physical band "
            f"[{KAPPA_FREE}, {KAPPA_CONFINED}] at nu={nu}, aspect_ratio="
            f"{aspect_ratio}; check the matrix parameters"
        )
    return min(KAPPA_CONFINED, max(KAPPA_FREE, trace))


def eshelby_kappa_from(matrix: MatrixParams) -> float:
    """``eshelby_kappa`` from a validated :class:`MatrixParams`."""
    return eshelby_kappa(matrix.poisson_ratio, matrix.aspect_ratio)


# --- Finite volume fraction: inclusion-inclusion interaction (Mori-Tanaka bounds) -----
# eshelby_kappa is the SINGLE-inclusion (dilute) limit: each cell's local strain field is
# undisturbed by its neighbours. At the model's operating cell_volume_fraction ~ 0.15
# (base.types presets) that limit is at the conventional dilute/interacting boundary, so
# the neglected interaction is a live question, not obviously zero. These two functions
# BOUND the correction instead of asserting it away.
#
# The problem is the HOMOGENEOUS-eigenstrain case (cells share the matrix stiffness and
# carry a swelling eigenstrain), NOT the inhomogeneity/stiffness-mismatch case whose
# famous Mori-Tanaka result kappa_d/(1 - f(1 - kappa_d)) is a DIFFERENT problem. For the
# eigenstrain case the standard Mori-Tanaka closure (each inclusion sees the average
# matrix strain, eps_incl = <eps>_matrix + S:eps*) yields two forms that differ only by the
# macroscopic boundary condition and BRACKET the real embedded voxel:
#   * traction-free (<sigma> = 0, free to swell):  kappa = f + (1 - f) kappa_d   [UP]
#   * displacement-clamped (<eps> = 0, rigid):     kappa = (1 - f) kappa_d       [DOWN]
# Both are self-consistent and both reduce to kappa_d at f = 0 (Voigt/Reuss-style bounds).


def mori_tanaka_kappa_traction_free(kappa_dilute: float, f: float) -> float:
    """Upper Mori-Tanaka kappa bound: a traction-free (freely swelling) voxel.

    ``kappa = f + (1 - f) * kappa_dilute`` -- each inclusion sees the average matrix
    strain of a composite under zero macroscopic stress, so neighbours swelling into the
    shared matrix RELIEVE a cell's constraint and it realizes MORE of its eigenstrain.
    Monotone up from ``kappa_dilute`` at ``f = 0`` to 1 at ``f = 1`` (all-inclusion,
    freely swelling -> full eigenstrain realized). The upper edge of the interaction
    bracket for the homogeneous eigenstrain problem.
    """
    if not 0.0 <= kappa_dilute <= 1.0 + 1e-9:
        raise ValueError(f"kappa_dilute must lie in [0, 1], got {kappa_dilute!r}")
    if not 0.0 <= f < 1.0:
        raise ValueError(f"f (volume fraction) must lie in [0, 1), got {f!r}")
    return f + (1.0 - f) * kappa_dilute


def mori_tanaka_kappa_clamped(kappa_dilute: float, f: float) -> float:
    """Lower Mori-Tanaka kappa bound: a displacement-clamped (rigidly embedded) voxel.

    ``kappa = (1 - f) * kappa_dilute`` -- each inclusion sees the average matrix strain of
    a composite held at zero macroscopic strain, so a cell swelling must push the matrix
    into compensating compression and realizes LESS of its eigenstrain. Monotone down from
    ``kappa_dilute`` at ``f = 0`` to 0 at ``f = 1`` (all-inclusion, rigidly held -> no net
    dilatation). The lower edge of the interaction bracket for the homogeneous eigenstrain
    problem.
    """
    if not 0.0 <= kappa_dilute <= 1.0 + 1e-9:
        raise ValueError(f"kappa_dilute must lie in [0, 1], got {kappa_dilute!r}")
    if not 0.0 <= f < 1.0:
        raise ValueError(f"f (volume fraction) must lie in [0, 1), got {f!r}")
    return (1.0 - f) * kappa_dilute


def interaction_verdict_shift_db(f: float) -> float:
    """Max |dB| shift the interaction bracket imposes on the (kappa-linear) verdict.

    The source term is linear in kappa (``Delta z = eta kappa L eps_V``), so a kappa
    factor ``r`` shifts the dB verdict by ``20 log10(r)``. The widest excursion is the
    clamped lower bound's factor ``(1 - f)``, whose ``|20 log10(1 - f)|`` is INDEPENDENT
    of ``kappa_dilute`` -- so this single number bounds the interaction effect at any kappa
    anchor. ~1.4 dB at ``f = 0.15`` (the operating fraction), ~3.1 dB at ``f = 0.30``:
    negligible against the model's ~-65 dB verdict and the ~60 dB (3-decade) eta span, so
    the dilute kappa is used rather than a Mori-Tanaka midpoint (which would inject the
    boundary-condition ambiguity for a sub-1.4 dB effect).
    """
    if not 0.0 <= f < 1.0:
        raise ValueError(f"f (volume fraction) must lie in [0, 1), got {f!r}")
    if f == 0.0:
        return 0.0
    return abs(20.0 * math.log10(1.0 - f))


def eshelby_provenance(nu: float, aspect_ratio: float = 1.0) -> tuple[str, ...]:
    """Assumption strings the sweep layer appends when kappa is Eshelby-derived."""
    return (
        "kappa is the constrained-dilatation fraction S_1111 + 2 S_1122 of a "
        "dilatational eigenstrain in an Eshelby inclusion (Mura formulation), "
        "NOT a free scalar",
        f"matrix Poisson ratio nu = {nu} (undrained/fast-band value; Su et al. "
        "2023 report nu_u > 0.49 in the short-time regime -> kappa near confined)",
        f"inclusion aspect ratio a3/a1 = {aspect_ratio} (1 = spherical cell)",
        "isotropic matrix; single-inclusion (dilute) kappa, with inclusion-inclusion "
        "interaction BOUNDED (Mori-Tanaka traction-free/clamped closures, "
        "mori_tanaka_kappa_*) and shown to shift the kappa-linear verdict by "
        "<= |20 log10(1-f)| ~ 1.4 dB at the operating f ~ 0.15 -- negligible vs the "
        "eta/synchrony drivers, so the dilute kappa is used",
    )
