"""Acoustic-output safety ceiling on the assumed echo SNR (D6).

Pins that the safe per-element echo SNR ceiling falls with transcranial skull derating,
that the demo's assumed 30 dB exceeds it at the 12 dB skull loss, and that a thin-skull
regime clears it.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from base_neural_model.forward.detection import AcquisitionParams
from base_neural_model.forward.safety import (
    SafetyLimits,
    echo_snr_within_safety,
    max_per_element_echo_snr_db,
)


def test_ceiling_falls_with_skull_derating():
    """The safe echo-SNR ceiling is the surface figure minus the transmit skull loss."""
    assert max_per_element_echo_snr_db(0.0) == pytest.approx(40.0)   # no bone: surface figure
    assert max_per_element_echo_snr_db(12.0) == pytest.approx(28.0)  # 12 dB one-way transmit loss
    # Monotone: more bone -> lower safe ceiling.
    assert max_per_element_echo_snr_db(20.0) < max_per_element_echo_snr_db(4.0)


def test_demo_echo_snr_exceeds_safety_transcranially():
    """The demo assumes 30 dB per-element, above the 28 dB safe ceiling at 12 dB skull."""
    acq = AcquisitionParams.demo_motor()  # echo_snr_linear = 1e3 -> 30 dB, skull 12 dB
    assert echo_snr_within_safety(acq) is False


def test_thin_skull_clears_safety():
    """At a thin (4 dB) temporal window the 30 dB assumption is within safety (36 dB ceiling)."""
    acq = replace(AcquisitionParams.demo_motor(), skull_loss_db_oneway=4.0)
    assert echo_snr_within_safety(acq) is True


def test_safety_limits_validate():
    with pytest.raises(ValueError):
        SafetyLimits(mechanical_index_max=0.0)
    with pytest.raises(ValueError):
        SafetyLimits(ispta_derated_mw_cm2=-1.0)
