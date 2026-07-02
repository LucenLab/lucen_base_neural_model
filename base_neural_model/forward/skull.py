r"""Frequency-dependent skull attenuation coupling center frequency to skull loss (D7).

:class:`~base_neural_model.forward.detection.AcquisitionParams` carries ``center_freq_hz``
and ``skull_loss_db_oneway`` as *independent* fields, so nothing stops a sweep from
setting 10 MHz with the demo's 12 dB loss -- unphysical, because bone absorption rises
steeply with frequency. This module supplies the coupling: the one-way skull loss as a
function of centre frequency, so a frequency sweep drags the loss with it.

Bone attenuation follows a power law in frequency::

    loss_oneway_dB = alpha * (f_MHz ** n) * L_cm

with an absorption coefficient ``alpha`` (dB per cm per MHz^n) and exponent ``n ~ 1.5-2``
for skull. The default ``alpha`` and thickness are calibrated to the demo's temporal
window so ``skull_loss_from_freq(2e6)`` reproduces the 12 dB one-way baseline; then the
loss grows with frequency while the displacement sensitivity ``lambda/4pi`` *improves*
with frequency -- the genuine two-sided trade the flat-loss model cannot express.
:func:`frequency_trade_curve` (in :mod:`base_neural_model.forward.budget`) resolves it.
"""

from __future__ import annotations

# Calibrated to a thin temporal-window: alpha * (2 MHz)^1.5 * 0.4 cm = ~12 dB one-way.
DEFAULT_ABSORPTION_DB_PER_CM_MHZ_N: float = 10.6   # dB / (cm * MHz^n) for temporal bone
DEFAULT_FREQ_EXPONENT: float = 1.5                 # bone attenuation power-law exponent
DEFAULT_BONE_THICKNESS_M: float = 4.0e-3           # ~4 mm temporal-window bone


def skull_loss_from_freq(
    center_freq_hz: float,
    *,
    bone_thickness_m: float = DEFAULT_BONE_THICKNESS_M,
    absorption_db_per_cm_mhz_n: float = DEFAULT_ABSORPTION_DB_PER_CM_MHZ_N,
    freq_exponent: float = DEFAULT_FREQ_EXPONENT,
) -> float:
    """One-way skull insertion loss (dB) at ``center_freq_hz`` (D7).

    ``alpha * (f_MHz ** n) * L_cm``. With the defaults, 2 MHz gives ~12 dB (the demo
    baseline); 3 MHz ~22 dB, 1 MHz ~4.2 dB -- the steep frequency penalty that makes a
    higher readout frequency a losing trade through bone despite its better sensitivity.
    """
    if center_freq_hz <= 0.0:
        raise ValueError(f"center_freq_hz must be positive, got {center_freq_hz!r}")
    if bone_thickness_m <= 0.0:
        raise ValueError(f"bone_thickness_m must be positive, got {bone_thickness_m!r}")
    if absorption_db_per_cm_mhz_n < 0.0:
        raise ValueError(
            f"absorption_db_per_cm_mhz_n must be >= 0, got {absorption_db_per_cm_mhz_n!r}"
        )
    f_mhz = center_freq_hz / 1e6
    length_cm = bone_thickness_m * 100.0
    return absorption_db_per_cm_mhz_n * (f_mhz**freq_exponent) * length_cm
