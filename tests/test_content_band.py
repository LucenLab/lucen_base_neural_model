"""Content-band inverse optimization: the 2-D shared band and the 1-D per-preset bands.

Pins the optimizer's contract on the corrected (absolute-jitter-floor) physics: the
in-band survival factor genuinely depends on where the band sits, the maximin shared
band is bounded by the worst per-preset ceiling, and no solution collapses out of the
admissible beta box.
"""

from __future__ import annotations

import numpy as np
import pytest

from base_neural_model.activity import run_activity
from base_neural_model.activity.motor_drive import sustained_imagery_drive
from base_neural_model.activity.populations import EIParams
from base_neural_model.base.types import VoxelGeometry
from base_neural_model.forward.content_band import (
    BETA_HI_HZ,
    BETA_LO_HZ,
    MIN_BANDWIDTH_HZ,
    band_on_fundamental,
    inband_survival,
    optimize_individual_bands,
    optimize_shared_band,
)
from base_neural_model.forward.detection import AcquisitionParams
from base_neural_model.mechanics import MechanicsParams
from base_neural_model.model.run import run_neural_model

_DRIVE = sustained_imagery_drive(1.3)
_ACQ = AcquisitionParams.demo_motor()


def _run(low_beta: bool):
    ei = EIParams.motor_cortex_low_beta() if low_beta else EIParams.motor_cortex()
    return run_activity(ei, duration_s=8.0, drive_fn=_DRIVE)


def _coherent_dz(low_beta: bool) -> float:
    ei = EIParams.motor_cortex_low_beta() if low_beta else EIParams.motor_cortex()
    r = run_neural_model(
        ei, geom=VoxelGeometry.motor_cortex_layer5(),
        mechanics=MechanicsParams.motor_cortex(), acquisition=_ACQ, drive_fn=_DRIVE,
    )
    return r.mechanical_displacement.axial_displacement_m


@pytest.fixture(scope="module")
def runs():
    return {"high-beta": _run(False), "low-beta": _run(True)}


@pytest.fixture(scope="module")
def cdz():
    return {"high-beta": _coherent_dz(False), "low-beta": _coherent_dz(True)}


def test_inband_survival_depends_on_band(runs):
    """The corrected physics makes G frequency-sensitive: a lower band survives better.

    This is the whole point of the model revision -- under the old synchrony-only
    jitter G would be identical (== synchrony) for every band.
    """
    ts = runs["high-beta"]
    low = inband_survival(ts, 13.0, 16.0)
    high = inband_survival(ts, 25.0, 30.0)
    assert 0.0 < high < low <= 1.0


def test_inband_survival_empty_band_is_zero(runs):
    # A window with no spectral content returns 0 (strictly dominated).
    assert inband_survival(runs["high-beta"], 13.01, 13.02) == 0.0


def test_individual_band_is_admissible_and_detectable_ceiling(runs, cdz):
    b = optimize_individual_bands(runs["high-beta"], _ACQ, coherent_dz_m=cdz["high-beta"])
    assert BETA_LO_HZ <= b.f_lo_hz < b.f_hi_hz <= BETA_HI_HZ
    assert b.f_hi_hz - b.f_lo_hz >= MIN_BANDWIDTH_HZ
    assert BETA_LO_HZ <= b.f_c_hz <= BETA_HI_HZ
    assert 0.0 < b.inband_survival <= 1.0
    assert np.isfinite(b.snr_db)


def test_shared_band_worst_is_bounded_by_min_individual(runs, cdz):
    """The maximin shared band cannot beat the worst preset's own optimum."""
    shared = optimize_shared_band(runs, _ACQ, coherent_dz_m=cdz)
    ind = {
        k: optimize_individual_bands(runs[k], _ACQ, coherent_dz_m=cdz[k]).snr_db
        for k in runs
    }
    assert shared.worst_snr_db <= min(ind.values()) + 1e-9
    # The worst-case value is realized by (at least) one preset at the shared band.
    realized_worst = min(b.snr_db for b in shared.per_preset.values())
    assert shared.worst_snr_db == pytest.approx(realized_worst, abs=1e-9)


def test_shared_band_is_in_beta_box(runs, cdz):
    shared = optimize_shared_band(runs, _ACQ, coherent_dz_m=cdz)
    assert BETA_LO_HZ <= shared.f_lo_hz < shared.f_hi_hz <= BETA_HI_HZ


def test_band_on_fundamental_sits_on_the_rhythm(runs, cdz):
    """The reported band is centred on the fundamental (the beta carrier), not leakage.

    After the retune the fundamental IS the beta peak (~20 Hz high-beta, ~15 Hz
    low-beta), so pinning the band there means f_c equals the dominant oscillatory
    frequency and the band captures the bulk of the power.
    """
    for k in runs:
        pin = band_on_fundamental(runs[k], _ACQ, coherent_dz_m=cdz[k])
        f0 = runs[k].spectrum.dominant_freq_hz
        assert pin.f_lo_hz <= f0 <= pin.f_hi_hz          # fundamental is inside the band
        assert pin.f_c_hz == pytest.approx(f0, abs=1.0)  # in-band peak IS the fundamental
        assert pin.content_fraction > 0.5                 # arch fundamental dominates


def test_free_optimum_is_degenerate_not_above_fundamental(runs, cdz):
    """The free optimum reads the lowest admissible band (leakage), <= the fundamental.

    This documents WHY band_on_fundamental exists: jitter physics makes the free search
    prefer the lowest frequency, so its peak never sits above the rhythm's fundamental.
    """
    for k in runs:
        free = optimize_individual_bands(runs[k], _ACQ, coherent_dz_m=cdz[k])
        f0 = runs[k].spectrum.dominant_freq_hz
        assert free.f_c_hz <= f0 + 1.0  # never targets above the fundamental


def test_high_beta_outscores_low_beta_at_any_shared_band(runs, cdz):
    """High-beta is the favorable target: it should not be the harder one to serve."""
    shared = optimize_shared_band(runs, _ACQ, coherent_dz_m=cdz)
    assert (
        shared.per_preset["high-beta"].snr_db
        >= shared.per_preset["low-beta"].snr_db - 1e-9
    )
