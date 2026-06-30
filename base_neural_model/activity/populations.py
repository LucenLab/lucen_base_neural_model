"""Excitatory/inhibitory population parameters for the neural-mass model.

The dynamical activity layer is a Wilson-Cowan-style mean-field model of two coupled
neural populations - excitatory (E) and inhibitory (I). :class:`EIParams` collects
the cited-range parameters that set their dynamics: the synaptic coupling weights
between and within the populations, the membrane time constants, the sigmoid
response gains and thresholds, and the external drive.

All quantities are SI where a unit applies: time constants and the drive timescale in
seconds, firing activities are dimensionless population activations in [0, 1] (the
Wilson-Cowan fraction-of-active-cells convention). The defaults sit in the standard
oscillatory regime where the E/I loop produces a limit cycle - the gamma-band-like
rhythm whose synchrony the mechanics consumes.
"""

from __future__ import annotations

from dataclasses import dataclass

from base_neural_model.base.bands import Band
from base_neural_model.base.provenance import Provenance


@dataclass(frozen=True)
class EIParams:
    """Wilson-Cowan E/I mean-field parameters (cited oscillatory regime).

    Coupling weights ``w_*`` are dimensionless synaptic gains; ``tau_*`` are membrane
    time constants (s); ``gain_*`` and ``theta_*`` set the slope and midpoint of the
    sigmoid response; ``drive_e``/``drive_i`` are the tonic external inputs.
    """

    w_ee: float          # E -> E recurrent excitation
    w_ei: float          # I -> E inhibition (enters E equation with a minus sign)
    w_ie: float          # E -> I excitation
    w_ii: float          # I -> I recurrent inhibition
    tau_e_s: float       # excitatory membrane time constant, seconds
    tau_i_s: float       # inhibitory membrane time constant, seconds
    gain_e: float        # excitatory sigmoid slope
    gain_i: float        # inhibitory sigmoid slope
    theta_e: float       # excitatory sigmoid threshold (midpoint)
    theta_i: float       # inhibitory sigmoid threshold (midpoint)
    drive_e: float       # tonic external drive to E
    drive_i: float       # tonic external drive to I
    provenance: Provenance
    # Anatomical columnar alignment Q_struct in [0, 1]: how aligned the population's
    # cell axes are in 3-D (the structural half of the directional channel). 0 =
    # isotropic (no directional signal regardless of synchrony, the default), 1 =
    # perfectly columnar (e.g. layer-5 pyramidal cells). The activity expresses this
    # through the temporal synchrony to produce the orientation coherence Q_eff.
    structural_alignment: float = 0.0

    def __post_init__(self) -> None:
        if self.tau_e_s <= 0.0 or self.tau_i_s <= 0.0:
            raise ValueError(
                f"time constants must be positive, got tau_e={self.tau_e_s!r}, "
                f"tau_i={self.tau_i_s!r}"
            )
        for name in ("w_ee", "w_ei", "w_ie", "w_ii"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"coupling weight {name} must be >= 0")
        if self.gain_e <= 0.0 or self.gain_i <= 0.0:
            raise ValueError("sigmoid gains must be positive")
        if not 0.0 <= self.structural_alignment <= 1.0:
            raise ValueError(
                f"structural_alignment must lie in [0, 1], got "
                f"{self.structural_alignment!r}"
            )

    @classmethod
    def central(cls) -> EIParams:
        """Central-column E/I parameters in the standard oscillatory (limit-cycle) regime.

        Coupling and time constants follow the classic Wilson-Cowan gamma-generating
        set: strong recurrent excitation balanced by faster, stronger E->I->E
        inhibition, with membrane time constants in the few-millisecond range that
        place the limit-cycle frequency in the fast/content band.
        """
        return cls(
            w_ee=16.0,
            w_ei=12.0,
            w_ie=15.0,
            w_ii=3.0,
            tau_e_s=6e-3,   # ~6 ms -> limit cycle in the tens-of-Hz content band
            tau_i_s=15e-3,  # slower inhibition closes the E/I loop with a phase lag
            gain_e=1.0,
            gain_i=1.0,
            theta_e=4.0,
            theta_i=3.7,
            drive_e=1.2,    # tonic drive into the oscillatory regime
            drive_i=0.0,
            provenance=Provenance(
                source=(
                    "Wilson & Cowan 1972 (excitatory/inhibitory mean-field dynamics); "
                    "gamma-generating E/I parameter regime, cortical content band"
                ),
                assumptions=(
                    "two coupled populations (E, I) as a mean field; activity is the "
                    "fraction of active cells in [0, 1], not a rate in Hz",
                    "sigmoid response S(x) = 1/(1 + exp(-gain (x - theta)))",
                    "membrane time constants tau_e ~ 6 ms, tau_i ~ 15 ms place the "
                    "E/I limit cycle in the fast/content band",
                    "tonic drive holds the loop in its oscillatory (limit-cycle) "
                    "regime; raising it raises mean activity and oscillation frequency",
                ),
                band=Band.CONTENT_FAST,
            ),
        )

    @classmethod
    def motor_cortex(cls) -> EIParams:
        """Primary motor cortex (M1) E/I parameters: a BETA-band limit cycle.

        M1's signature sensorimotor rhythm is beta (~13-30 Hz), not the gamma of the
        generic column. Slightly slower membrane time constants place the E/I limit
        cycle in the beta band; the rest of the coupling regime is retained. The
        defining structural difference is set here too: M1 layer-5 pyramidal cells
        (including the giant Betz cells) are large and strongly **columnar**, so the
        structural alignment ``Q_struct`` is high - which, expressed through the
        movement-locked synchrony, drives the directional channel that a generic
        isotropic patch lacks.
        """
        return cls(
            w_ee=16.0,
            w_ei=12.0,
            w_ie=15.0,
            w_ii=3.0,
            tau_e_s=9e-3,    # ~9 ms -> limit cycle in the beta band (~20 Hz)
            tau_i_s=22e-3,   # proportionally slower inhibition, beta E/I loop
            gain_e=1.0,
            gain_i=1.0,
            theta_e=4.0,
            theta_i=3.7,
            drive_e=1.2,
            drive_i=0.0,
            structural_alignment=0.9,  # layer-5 pyramidal columns: strongly aligned
            provenance=Provenance(
                source=(
                    "Wilson & Cowan E/I mean field tuned to the M1 beta rhythm; "
                    "layer-5 pyramidal (Betz) columnar alignment (Murphy et al. 2016 "
                    "M1 cytoarchitecture; beta sensorimotor rhythm, e.g. Baker 2007)"
                ),
                assumptions=(
                    "M1 sensorimotor rhythm is beta (~13-30 Hz); tau_e ~ 9 ms places "
                    "the E/I limit cycle there",
                    "layer-5 pyramidal / Betz cells are large and strongly columnar "
                    "-> high structural alignment Q_struct ~ 0.9 (vs ~0 for an "
                    "isotropic patch); the directional channel is material in M1",
                    "structural alignment is expressed as orientation coherence "
                    "through the temporal synchrony (Q_eff = Q_struct * s)",
                    "beta content carries movement-related modulation (the motor "
                    "'content' band), distinct from the slow movement envelope",
                ),
                band=Band.CONTENT_FAST,
            ),
        )
