"""Acoustic SNR budget sweep: the curve and the axis ranking.

Pins the budget-sweep behaviour: the SNR-vs-skull-loss curve falls at the expected
-2 dB per one-way dB (two-way x 20log10), the 0 dB crossing is located, the binding
denominator flips along a clutter sweep, and the acoustic axis ranking leads with the
two strategic levers (skull loss and epoch).
"""

from __future__ import annotations

import numpy as np
import pytest

from base_neural_model.base.bands import Band
from base_neural_model.base.provenance import Provenance
from base_neural_model.base.types import MechanicalDisplacement
from base_neural_model.forward.budget import (
    acoustic_axis_ranking,
    sweep_axis,
)
from base_neural_model.forward.detection import AcquisitionParams, detection_budget


def _mech(dz_m: float = 3.5e-9, survival: float = 0.81) -> MechanicalDisplacement:
    prov = Provenance(
        source="test source displacement for the budget sweep",
        assumptions=("synthetic motor-demo-scale dz",),
        band=Band.CONTENT_FAST,
    )
    return MechanicalDisplacement(
        axial_displacement_m=dz_m,
        volumetric_strain=0.0,
        incoherent_pedestal_m=0.0,
        synchrony_fraction=0.81,
        jitter_sigma_s=1e-3,
        content_band_survival=survival,
        confinement_kappa=0.5,
        dilatation_eta=0.5,
        band=Band.CONTENT_FAST,
        provenance=prov,
    )


def test_skull_loss_sweep_falls_2db_per_oneway_db():
    """SNR falls 2 dB per one-way skull dB (two-way; in displacement = x20log10)."""
    mech = _mech()
    acq = AcquisitionParams.demo_motor()
    loss = np.linspace(0.0, 20.0, 11)
    curve = sweep_axis(mech, acq, "skull_loss_db_oneway", loss)
    slope = np.diff(curve.snr_db) / np.diff(loss)
    assert np.allclose(slope, -2.0, atol=1e-6)


def test_skull_loss_crossing_located():
    """The 0 dB crossing (signal == floor) is interpolated on the monotone curve."""
    mech = _mech()
    acq = AcquisitionParams.demo_motor()
    curve = sweep_axis(mech, acq, "skull_loss_db_oneway", np.linspace(0.0, 24.0, 25))
    xc = curve.crossing_value
    assert xc is not None
    # Re-evaluate at the crossing: SNR should be ~0 dB there.
    from dataclasses import replace
    b = detection_budget(mech, replace(acq, skull_loss_db_oneway=xc))
    assert b.snr_db == pytest.approx(0.0, abs=1e-6)


def test_no_crossing_returns_none():
    """A curve that never crosses 0 dB reports no crossing."""
    mech = _mech(dz_m=1e-6)  # huge source -> always detectable
    acq = AcquisitionParams.demo_motor()
    curve = sweep_axis(mech, acq, "skull_loss_db_oneway", np.linspace(0.0, 10.0, 6))
    assert np.all(curve.snr_db > 0)
    assert curve.crossing_value is None


def test_clutter_sweep_flips_denominator():
    """Sweeping residual clutter above the phase floor flips the binding denominator."""
    mech = _mech()
    acq = AcquisitionParams.demo_motor()
    from base_neural_model.forward.detection import phase_displacement_floor
    pf = phase_displacement_floor(acq)
    # Below the phase floor -> echo_snr; above -> clutter.
    lo = sweep_axis(mech, acq, "skull_loss_db_oneway", np.array([12.0]),
                    residual_clutter_m=pf * 0.1)
    hi = sweep_axis(mech, acq, "skull_loss_db_oneway", np.array([12.0]),
                    residual_clutter_m=pf * 10.0)
    assert lo.limiting_denominator == ("echo_snr",)
    assert hi.limiting_denominator == ("clutter",)


def test_axis_ranking_leads_with_skull_and_epoch():
    """The acoustic verdict collapses onto skull loss and epoch (the levers)."""
    mech = _mech()
    acq = AcquisitionParams.demo_motor()
    ranking = acoustic_axis_ranking(mech, acq)
    top_two = {ranking[0].axis_name, ranking[1].axis_name}
    assert top_two == {"skull_loss_db_oneway", "epoch_s"}
    # Each top axis swings the verdict by tens of dB.
    assert ranking[0].db_span > 20.0


def test_sweep_rejects_bad_axis():
    with pytest.raises(ValueError):
        sweep_axis(_mech(), AcquisitionParams.demo_motor(), "not_a_field",
                   np.array([1.0]))
