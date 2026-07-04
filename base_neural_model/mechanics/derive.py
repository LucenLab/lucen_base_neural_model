"""Compose Eshelby kappa and poroelastic eta into a MechanicsParams.

The transduction chain (``transduction.py``) consumes ``kappa`` and ``eta`` as bare
scalars on :class:`MechanicsParams`. This thin, cycle-free bridge derives those two
load-bearing unknowns from their physical sub-models -- :func:`eshelby_kappa` from
the matrix Poisson ratio and :func:`poroelastic_eta` from the drainage state -- and
assembles the rest of the chain inputs into a ready-to-sweep ``MechanicsParams``.

It lives here, not in ``base_neural_model.base.types``, so the Eshelby/poroelastic
helpers stay free of any import of the contract types (no cycle): the dependency
points one way, derive -> {eshelby, poroelastic, types}.
"""

from __future__ import annotations

from dataclasses import replace

from base_neural_model.base.types import MechanicsParams
from base_neural_model.mechanics.eshelby import MatrixParams, eshelby_kappa_from
from base_neural_model.mechanics.poroelastic import PoroelasticParams, poroelastic_eta

# Fast-band undrained matrix Poisson ratio (Su et al. 2023: nu_u > 0.49 in the
# short-time regime) -- the same regime the Sobol sweep's nu in [0.45, 0.49] probes
# and the value MechanicsParams.confinement_kappa's provenance note (base.types)
# names as the intended Eshelby-derivation regime for kappa.
FAST_BAND_UNDRAINED_NU: float = 0.49


def build_mechanics_params(
    *,
    matrix: MatrixParams,
    poro: PoroelasticParams,
    membrane_disp_m: float,
    cell_radius_m: float,
    cell_volume_fraction: float,
    jitter_sigma_s: float,
    content_freq_hz: float,
) -> MechanicsParams:
    """Build a :class:`MechanicsParams` with kappa and eta derived from sub-models.

    ``confinement_kappa`` comes from the Eshelby tensor (matrix Poisson ratio);
    ``dilatation_eta`` comes from the poroelastic drainage state. Every other field
    is a direct chain input. The derived kappa/eta land in exactly the fields
    ``mechanical_displacement`` already validates, so the existing chain and all its
    tests are untouched.
    """
    return MechanicsParams(
        membrane_disp_m=membrane_disp_m,
        cell_radius_m=cell_radius_m,
        cell_volume_fraction=cell_volume_fraction,
        confinement_kappa=eshelby_kappa_from(matrix),
        dilatation_eta=poroelastic_eta(poro),
        jitter_sigma_s=jitter_sigma_s,
        content_freq_hz=content_freq_hz,
    )


def motor_cortex_with_aspect_ratio(
    aspect_ratio: float,
    *,
    matrix_nu: float = FAST_BAND_UNDRAINED_NU,
) -> MechanicsParams:
    """:meth:`MechanicsParams.motor_cortex` with a non-spherical confinement kappa.

    Every field is identical to :meth:`MechanicsParams.motor_cortex` EXCEPT
    ``confinement_kappa``, which is derived from the Eshelby spheroid trace
    (:func:`eshelby.eshelby_kappa_from`) at the given ``aspect_ratio`` (semi-axis
    ratio a3/a1: 1 = sphere, the ``motor_cortex()`` default; <1 oblate, >1 prolate)
    instead of the preset's bare ``confinement_kappa = 0.5`` literal.

    The motor preset already treats M1 layer-5 / Betz cells as large and strongly
    elongated along the apical-dendrite axis for the DIRECTIONAL (deviatoric)
    channel (``anisotropy = 0.6``, ``motor_cortex()``'s docstring); this function
    lets the same elongation shape the VOLUMETRIC confinement kappa too, via a
    prolate ``aspect_ratio > 1``, rather than leaving kappa at the generic
    spherical-cell value while only the directional channel reflects the cell
    shape. At ``aspect_ratio = 1.0`` this reproduces ``motor_cortex()``'s
    ``confinement_kappa = 0.5`` only if ``matrix_nu`` is chosen to match (it is
    NOT pinned to 0.5 by default; the default ``matrix_nu`` is the fast-band
    undrained regime the kappa provenance note on ``MechanicsParams`` names, see
    ``base.types``), so this is an ADDITIONAL, opt-in derivation path -- it does
    not change what ``MechanicsParams.motor_cortex()`` itself returns.

    ``matrix_nu`` is the volumetric-confinement matrix Poisson ratio (independent
    of ``matrix_poisson_ratio``, which governs only the deviatoric/directional
    channel -- see the provenance note on ``MechanicsParams.confinement_kappa``).
    """
    base = MechanicsParams.motor_cortex()
    kappa = eshelby_kappa_from(MatrixParams(poisson_ratio=matrix_nu, aspect_ratio=aspect_ratio))
    return replace(base, confinement_kappa=kappa)
