r"""Viscoelastic transfer function: the tissue is not a massless static spring (S1).

The transduction chain (:mod:`base_neural_model.mechanics.transduction`) maps an
oscillating neural volume change to a displacement *statically* -- the strain-to-
displacement relation ``Delta z = eta kappa L eps_V`` is evaluated at every instant with
no mechanical dynamics. But brain tissue is **viscoelastic** with a strongly
frequency-dependent complex shear modulus (MR elastography: the storage modulus ``G'``
and loss modulus ``G''`` both rise across 10-100+ Hz as a power law, with whole-brain
group-mean exponents of 0.88 and 1.07 respectively over 10-50 Hz, Testu et al. 2017,
J Mech Behav Biomed Mater). A source oscillating at the beta content corner (~20 Hz)
therefore drives the tissue through its *dynamic* modulus, not its DC stiffness: the
in-phase displacement is attenuated and phase-lagged relative to the lossless static
value.

This module supplies that missing factor. A **fractional (springpot) low-pass** models
the relaxation the oscillating dilatation source sees::

    H(omega) = 1 / (1 + (i omega tau) ** a)

with relaxation time ``tau`` and power-law exponent ``a in (0, 1]`` (a = 1 recovers a
Debye/standard relaxation; a < 1 is the broad, power-law spectrum brain actually shows).
``|H|`` is the amplitude the coherent axial term keeps and ``arg H`` its phase lag. At
``omega tau << 1`` (relaxation faster than the rhythm) ``|H| -> 1`` and the quasi-static
chain is recovered; at the beta corner with a few-ms relaxation ``|H|`` is a modest
sub-unity factor -- a real, contestable loss, not orders of magnitude.

Like ``eta`` and ``kappa`` upstream, the relaxation time is a reviewer-contestable input
measured away from the operating point; the module's job is to make the frequency-
dependence *representable*, not to assert its exact value. The power-law exponent,
however, is a directly measured quantity (Testu et al. 2017): this single-exponent
springpot uses one shared ``a`` for both storage and loss response, so it is fit to the
measured storage-modulus exponent (0.88) -- the more direct driver of the in-phase
amplitude ``|H|`` this factor attenuates -- with the measured loss-modulus exponent
(1.07) noted as the same order, supporting a single shared exponent as a defensible
one-parameter compromise rather than an independent two-exponent model. The applied
factor is stored on :class:`~base_neural_model.base.types.MechanicsParams` as
``viscoelastic_factor`` (default 1.0 -> off), so an isotropic static construction is
unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Brain viscoelastic relaxation time: ms-scale, contestable, and measured away from the
# beta operating point (like C_v for eta).
DEFAULT_RELAXATION_TIME_S: float = 3.0e-3  # ~3 ms relaxation
# Power-law exponent: the measured whole-brain storage-modulus exponent (Testu et al.
# 2017, J Mech Behav Biomed Mater; group mean 0.88 over 10-50 Hz shear waves). The
# measured loss-modulus exponent is 1.07 -- close enough in order that this single-
# exponent springpot's shared ``a`` reasonably represents both.
DEFAULT_SPRINGPOT_EXPONENT: float = 0.88


@dataclass(frozen=True)
class ViscoelasticParams:
    """Fractional (springpot) viscoelastic relaxation of the tissue matrix."""

    relaxation_time_s: float = DEFAULT_RELAXATION_TIME_S
    exponent: float = DEFAULT_SPRINGPOT_EXPONENT

    def __post_init__(self) -> None:
        if self.relaxation_time_s <= 0.0:
            raise ValueError(
                f"relaxation_time_s must be positive, got {self.relaxation_time_s!r}"
            )
        if not 0.0 < self.exponent <= 1.0:
            raise ValueError(f"exponent must lie in (0, 1], got {self.exponent!r}")


def _springpot_response(f_hz: float, params: ViscoelasticParams) -> complex:
    """Complex transfer ``H(omega) = 1 / (1 + (i omega tau)^a)`` at ``f_hz``."""
    if f_hz <= 0.0:
        raise ValueError(f"f_hz must be positive, got {f_hz!r}")
    omega = 2.0 * math.pi * f_hz
    x = (omega * params.relaxation_time_s) ** params.exponent
    # (i)^a = exp(i a pi/2); (i omega tau)^a = x * (cos + i sin) of a*pi/2.
    a_phase = params.exponent * math.pi / 2.0
    denom = complex(1.0 + x * math.cos(a_phase), x * math.sin(a_phase))
    return 1.0 / denom


def viscoelastic_factor(f_hz: float, params: ViscoelasticParams | None = None) -> float:
    """Amplitude the coherent term keeps at ``f_hz``: ``|H(omega)| in (0, 1]`` (S1).

    1 at low frequency (relaxation faster than the rhythm), falling as the frequency
    approaches / exceeds the relaxation rate. This is the factor stored as
    ``MechanicsParams.viscoelastic_factor`` and applied to the coherent axial term.
    """
    params = params or ViscoelasticParams()
    return abs(_springpot_response(f_hz, params))


def viscoelastic_phase_rad(f_hz: float, params: ViscoelasticParams | None = None) -> float:
    """Phase lag ``arg H(omega)`` (radians, <= 0) the displacement acquires at ``f_hz``."""
    params = params or ViscoelasticParams()
    return math.atan2(_springpot_response(f_hz, params).imag,
                      _springpot_response(f_hz, params).real)


def viscoelastic_provenance(f_hz: float, params: ViscoelasticParams) -> tuple[str, ...]:
    """Assumption strings appended when the viscoelastic factor is applied."""
    mag = viscoelastic_factor(f_hz, params)
    return (
        "tissue is viscoelastic, NOT a massless static spring (S1): a fractional "
        "(springpot) transfer H(w)=1/(1+(i w tau)^a) attenuates and phase-lags the "
        "coherent term at the content corner",
        f"relaxation tau = {params.relaxation_time_s * 1e3:.3g} ms (contestable, "
        "measured away from the beta operating point like C_v for eta); exponent a = "
        f"{params.exponent} (Testu et al. 2017, J Mech Behav Biomed Mater: whole-brain "
        "group-mean power-law exponents 0.88 storage / 1.07 loss modulus, 10-50 Hz) "
        f"-> |H(f_c={f_hz:.3g} Hz)| = {mag:.3g}",
    )
