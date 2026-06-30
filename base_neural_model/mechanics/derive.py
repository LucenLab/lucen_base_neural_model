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

from base_neural_model.base.types import MechanicsParams
from base_neural_model.mechanics.eshelby import MatrixParams, eshelby_kappa_from
from base_neural_model.mechanics.poroelastic import PoroelasticParams, poroelastic_eta


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
