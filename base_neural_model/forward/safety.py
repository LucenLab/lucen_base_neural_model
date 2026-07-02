r"""Acoustic-output safety ceiling on the assumed echo SNR (D6).

The detection budget takes ``echo_snr_linear`` as a free knob and the budget sweep
ranges it over ~20-40 dB. But transmit intensity is not free: diagnostic ultrasound is
bounded by the FDA/IEC **mechanical index** (MI <= 1.9) and **thermal** limits (derated
ISPTA.3, and a cranial thermal index), and those bounds bite *hardest transcranially*
because the skull absorbs strongly and heats. The received echo power -- hence the echo
SNR -- scales with the transmit intensity that reaches the focus, so the favourable end
of the echo-SNR axis may simply not be reachable within safety.

This module turns that into a **ceiling** on the achievable per-element echo SNR and a
check the budget can flag. The ceiling is a reviewer-contestable input (like the
source-side ``eta``): the physics it encodes -- that the transcranial transmit derating
lowers the safe echo SNR -- is the load-bearing part, not the exact calibration. The
default ``reference_surface_echo_snr_db`` is the per-element free-field echo SNR a weak
(nm-displacement) scatterer returns at the MI-limited *surface* pressure; the transmit
side then pays the one-way skull loss, so the safe post-transmit ceiling falls one skull
derating below it. At the demo's 12 dB one-way skull loss the ceiling is ~28 dB, so the
demo's assumed 30 dB per-element SNR **exceeds** what is safely achievable transcranially
-- an honest flag the prior budget could not raise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from base_neural_model.forward.detection import AcquisitionParams

# FDA/IEC diagnostic acoustic-output limits (Track 3).
DEFAULT_MECHANICAL_INDEX_MAX: float = 1.9        # MI limit
DEFAULT_ISPTA_DERATED_MW_CM2: float = 720.0      # derated ISPTA.3 (general); cranial TIC applies
# Per-element free-field echo SNR (dB) a weak scatterer returns at the MI-limited surface
# pressure. Contestable; the transcranial transmit derating is subtracted from it.
DEFAULT_REFERENCE_SURFACE_ECHO_SNR_DB: float = 40.0


@dataclass(frozen=True)
class SafetyLimits:
    """Acoustic-output safety limits and the reference calibration (all contestable)."""

    mechanical_index_max: float = DEFAULT_MECHANICAL_INDEX_MAX
    ispta_derated_mw_cm2: float = DEFAULT_ISPTA_DERATED_MW_CM2
    reference_surface_echo_snr_db: float = DEFAULT_REFERENCE_SURFACE_ECHO_SNR_DB

    def __post_init__(self) -> None:
        if self.mechanical_index_max <= 0.0:
            raise ValueError(
                f"mechanical_index_max must be positive, got {self.mechanical_index_max!r}"
            )
        if self.ispta_derated_mw_cm2 <= 0.0:
            raise ValueError(
                f"ispta_derated_mw_cm2 must be positive, got {self.ispta_derated_mw_cm2!r}"
            )


def max_per_element_echo_snr_db(
    skull_transmit_derating_db: float,
    limits: SafetyLimits | None = None,
) -> float:
    """Safe ceiling on the per-element echo SNR (dB) through a transmit skull derating.

    The MI-limited surface echo SNR minus the one-way (transmit-side) skull loss: at the
    surface the pressure is capped by MI, and the focus sees that pressure reduced by the
    transmit path through bone, lowering the returned echo power one derating below the
    surface figure. Falls with skull derating -- the transcranial penalty.
    """
    if skull_transmit_derating_db < 0.0:
        raise ValueError(
            f"skull_transmit_derating_db must be >= 0, got {skull_transmit_derating_db!r}"
        )
    limits = limits or SafetyLimits()
    return limits.reference_surface_echo_snr_db - skull_transmit_derating_db


def echo_snr_within_safety(
    acq: AcquisitionParams,
    limits: SafetyLimits | None = None,
) -> bool:
    """True iff ``acq``'s assumed per-element echo SNR is achievable within safety.

    Compares the assumed per-element echo SNR (``10 log10(echo_snr_linear)``) against
    :func:`max_per_element_echo_snr_db`, using the acquisition's own one-way skull loss
    as the transmit derating (a symmetric transmit/receive assumption). ``False`` means
    the budget is assuming more transmit power than MI/thermal limits allow transcranially.
    """
    assumed_db = 10.0 * math.log10(acq.echo_snr_linear)
    ceiling_db = max_per_element_echo_snr_db(acq.skull_loss_db_oneway, limits)
    return assumed_db <= ceiling_db


def safety_provenance(
    acq: AcquisitionParams,
    limits: SafetyLimits | None = None,
) -> str:
    """One provenance line stating the safe ceiling and whether the acquisition clears it."""
    limits = limits or SafetyLimits()
    assumed_db = 10.0 * math.log10(acq.echo_snr_linear)
    ceiling_db = max_per_element_echo_snr_db(acq.skull_loss_db_oneway, limits)
    ok = assumed_db <= ceiling_db
    return (
        f"acoustic-output safety (MI<={limits.mechanical_index_max}, derated ISPTA "
        f"{limits.ispta_derated_mw_cm2} mW/cm^2): safe per-element echo SNR ceiling "
        f"= {ceiling_db:.4g} dB (surface {limits.reference_surface_echo_snr_db} dB - "
        f"{acq.skull_loss_db_oneway} dB transmit skull loss); assumed "
        f"{assumed_db:.4g} dB {'WITHIN' if ok else 'EXCEEDS'} safety [D6]"
    )
