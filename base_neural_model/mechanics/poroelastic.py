"""Poroelastic sub-model for the net-dilatation fraction eta.

``eta`` is the load-bearing source-side unknown: what fraction of a per-cell volume
change becomes *net tissue dilatation* the beam can read, versus *internal water
redistribution* at conserved bulk volume that it cannot. The literature pins eta
only loosely (~0.01 .. ~0.5, ceiling 1) because every relevant poroelastic
parameter was measured at seconds-to-minutes while eta is a millisecond question
(source doc, PART 1).

The legitimate purpose of this module is NOT to compute eta's true value -- the
doc is explicit that no parameter sweep can clear the project, only kill it. It is
to SWEEP eta across the literature range in a physically structured way, so the
sensitivity and verdict-flip layers can show the verdict collapses onto eta.

The controlling quantity is the dimensionless drainage (consolidation) number::

    theta = C_v * t / L^2

where ``C_v`` is the poroelastic diffusivity, ``t`` the interrogation gate time,
and ``L`` the drainage length (the range gate). Drainage on the fast/content
timescale is possible only when ``theta >~ 1``:

* ``theta << 1`` (fast band, short gate): drainage incomplete, net dilatation
  suppressed -- eta near the floor ~0.01.
* ``theta >~ 1`` (slow / long length): drainage complete, eta toward its ceiling.

``eta = 1`` is the *unconfined optimistic ceiling* -- what 2D culture in an infinite
bath would measure (the wrong eta), because that preparation removes the ECS-
reservoir competition and confinement entirely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Literature permeability spread, m^4/(N s) (~4 orders; cat brain ~1.6e-11).
PERMEABILITY_MIN: float = 1e-14
PERMEABILITY_MAX: float = 1e-11

# eta is the net-dilatation fraction; physically [0, 1].
ETA_FLOOR_DEFAULT: float = 0.01   # drainage-incomplete suppressed floor
ETA_CEILING_DEFAULT: float = 1.0  # unconfined optimistic ceiling (2D-culture limit)


@dataclass(frozen=True)
class PoroelasticParams:
    """Inputs to the poroelastic eta sweep (all reviewer-contestable in isolation)."""

    permeability_m4_per_Ns: float    # k, literature 1e-14 .. 1e-11
    porosity: float                  # alpha (ECS volume fraction), (0, 1); ~0.2
    poro_diffusivity_m2_per_s: float  # C_v, the consolidation diffusivity
    gate_time_s: float               # t, the fast/content interrogation gate time
    drainage_length_m: float         # L, drainage length (the range gate)
    vascular_source: float = 0.0     # optional fast vascular-sourced water term, [0, 1]
    osmotic_forcing: float = 0.0     # optional osmotic forcing term, [0, 1]
    eta_ceiling: float = ETA_CEILING_DEFAULT
    eta_floor: float = ETA_FLOOR_DEFAULT

    def __post_init__(self) -> None:
        if self.permeability_m4_per_Ns <= 0.0:
            raise ValueError(
                f"permeability must be positive, got {self.permeability_m4_per_Ns!r}"
            )
        if not 0.0 < self.porosity < 1.0:
            raise ValueError(f"porosity must lie in (0, 1), got {self.porosity!r}")
        if self.poro_diffusivity_m2_per_s <= 0.0:
            raise ValueError(
                f"poro_diffusivity must be positive, got "
                f"{self.poro_diffusivity_m2_per_s!r}"
            )
        if self.gate_time_s <= 0.0:
            raise ValueError(f"gate_time_s must be positive, got {self.gate_time_s!r}")
        if self.drainage_length_m <= 0.0:
            raise ValueError(
                f"drainage_length_m must be positive, got {self.drainage_length_m!r}"
            )
        if not 0.0 <= self.vascular_source <= 1.0:
            raise ValueError(
                f"vascular_source must lie in [0, 1], got {self.vascular_source!r}"
            )
        if not 0.0 <= self.osmotic_forcing <= 1.0:
            raise ValueError(
                f"osmotic_forcing must lie in [0, 1], got {self.osmotic_forcing!r}"
            )
        if not 0.0 < self.eta_floor <= self.eta_ceiling <= 1.0:
            raise ValueError(
                "require 0 < eta_floor <= eta_ceiling <= 1, got "
                f"floor={self.eta_floor!r}, ceiling={self.eta_ceiling!r}"
            )


def consolidation_number(C_v: float, t: float, L: float) -> float:
    """Drainage number ``theta = C_v t / L^2`` (drainage completeness group)."""
    if C_v <= 0.0 or t <= 0.0 or L <= 0.0:
        raise ValueError(f"C_v, t, L must be positive, got {C_v!r}, {t!r}, {L!r}")
    return C_v * t / (L * L)


def consolidation_time(C_v: float, L: float) -> float:
    """Characteristic consolidation time ``tau = L^2 / C_v`` (source doc, tau~L^2/C_v)."""
    if C_v <= 0.0 or L <= 0.0:
        raise ValueError(f"C_v, L must be positive, got {C_v!r}, {L!r}")
    return L * L / C_v


def drainage_completeness(theta: float) -> float:
    """Monotone map ``theta -> [0, 1]``, 0 at ``theta << 1`` -> 1 at ``theta >~ 1``.

    A saturating ``1 - exp(-theta)``: at the fast band (theta << 1) it is ~theta
    (drainage barely begun), and it approaches 1 once theta >~ a few (drainage
    essentially complete). The exact functional form is a modeling choice; what is
    load-bearing is the two limits and monotonicity, which any saturating form
    shares.
    """
    if theta < 0.0:
        raise ValueError(f"theta must be non-negative, got {theta!r}")
    return 1.0 - math.exp(-theta)


def poroelastic_eta(params: PoroelasticParams) -> float:
    """Net-dilatation fraction ``eta`` from the poroelastic drainage state.

    Interpolates between ``eta_floor`` (drainage incomplete -> redistribution-
    dominated) and ``eta_ceiling`` (drainage complete -> unconfined limit) by the
    drainage completeness, then lifts toward the ceiling by any fast vascular-
    sourced or osmotically-forced water that bypasses ECS drainage. The result is
    validated to ``[0, 1]``.
    """
    theta = consolidation_number(
        params.poro_diffusivity_m2_per_s, params.gate_time_s, params.drainage_length_m
    )
    completeness = drainage_completeness(theta)

    # Base eta from drainage completeness, between floor and ceiling.
    eta = params.eta_floor + (params.eta_ceiling - params.eta_floor) * completeness

    # Fast bypass: vascular-sourced / osmotically-forced water adds net dilatation
    # that does not wait on ECS drainage. Pushes the residual gap to the ceiling.
    bypass = max(params.vascular_source, params.osmotic_forcing)
    eta = eta + (params.eta_ceiling - eta) * bypass

    if not 0.0 <= eta <= 1.0:
        raise ValueError(
            f"derived eta={eta!r} fell outside [0, 1] at theta={theta}; "
            "check the poroelastic parameters"
        )
    return eta


def eta_from_permeability(
    permeability_m4_per_Ns: float,
    *,
    porosity: float,
    storage_modulus_Pa: float,
    gate_time_s: float,
    drainage_length_m: float,
    vascular_source: float = 0.0,
    osmotic_forcing: float = 0.0,
    eta_ceiling: float = ETA_CEILING_DEFAULT,
    eta_floor: float = ETA_FLOOR_DEFAULT,
) -> float:
    """``eta`` mapped straight from permeability ``k`` -- the headline literature sweep.

    The poroelastic diffusivity is ``C_v = k * M`` with ``M`` a storage/Biot
    modulus (Pa); sweeping ``k`` over the literature range
    ``[1e-14, 1e-11] m^4/(N s)`` then traces eta from its suppressed floor (low k,
    no drainage in the gate) toward its ceiling (high k). This is the one-call
    generator the sensitivity layer sweeps.
    """
    C_v = permeability_m4_per_Ns * storage_modulus_Pa
    return poroelastic_eta(
        PoroelasticParams(
            permeability_m4_per_Ns=permeability_m4_per_Ns,
            porosity=porosity,
            poro_diffusivity_m2_per_s=C_v,
            gate_time_s=gate_time_s,
            drainage_length_m=drainage_length_m,
            vascular_source=vascular_source,
            osmotic_forcing=osmotic_forcing,
            eta_ceiling=eta_ceiling,
            eta_floor=eta_floor,
        )
    )


def poroelastic_provenance(params: PoroelasticParams) -> tuple[str, ...]:
    """Assumption strings the sweep layer appends when eta is poroelastically derived."""
    theta = consolidation_number(
        params.poro_diffusivity_m2_per_s, params.gate_time_s, params.drainage_length_m
    )
    return (
        "eta is the net-dilatation fraction set by drainage completeness "
        "theta = C_v t / L^2, NOT a free scalar and NOT eta's true value -- the "
        "sub-model exists to SWEEP eta across the literature range",
        f"drainage number theta = {theta:.3g} (theta << 1 -> drainage incomplete -> "
        "eta suppressed toward the floor; theta >~ 1 -> eta toward ceiling)",
        f"eta_ceiling = {params.eta_ceiling} is the UNCONFINED optimistic limit -- "
        "what 2D culture in an infinite bath would wrongly measure",
        f"eta_floor = {params.eta_floor}: undrained incompressibility + ECS-reservoir "
        "redistribution (Su et al. 2023; Ostby et al. 2009) at the fast band",
        "C_v / permeability is measured ~3 orders of magnitude away in time; treat "
        "any eta > 0.1 as an extrapolation, not a measurement",
    )
