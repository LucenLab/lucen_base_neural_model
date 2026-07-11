"""Acoustic SNR budget sweep: the curve and the axis ranking.

Pins the budget-sweep behaviour: the SNR-vs-skull-loss curve falls at the expected
-2 dB per one-way dB (two-way x 20log10), the 0 dB crossing is located, the binding
denominator flips along a clutter sweep, and the acoustic axis ranking leads with skull
loss then echo SNR (epoch, the motor advantage, is material but now sqrt(N)-scaled).
"""

from __future__ import annotations

from dataclasses import replace

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
    """SNR falls 2 dB per one-way skull dB (two-way; in displacement = x20log10).

    Pinned on the inert (optimistic) base so the SNR-limited Walker-Trahey term is the
    whole floor; the honest ``demo_motor`` adds an integration-independent aberration
    floor that flattens this slope at low skull loss (that is tested separately).
    """
    mech = _mech()
    acq = AcquisitionParams.demo_motor_optimistic()
    loss = np.linspace(0.0, 20.0, 11)
    curve = sweep_axis(mech, acq, "skull_loss_db_oneway", loss)
    slope = np.diff(curve.snr_db) / np.diff(loss)
    assert np.allclose(slope, -2.0, atol=1e-6)


def test_skull_loss_crossing_located():
    """The 0 dB crossing (signal == floor) is interpolated on the monotone curve."""
    mech = _mech()
    acq = AcquisitionParams.demo_motor_optimistic()
    curve = sweep_axis(mech, acq, "skull_loss_db_oneway", np.linspace(0.0, 24.0, 25))
    xc = curve.crossing_value
    assert xc is not None
    # Re-evaluate at the crossing: SNR should be ~0 dB there.
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


def test_axis_ranking_leads_with_skull_and_echo_snr():
    """The acoustic verdict collapses onto skull loss (steepest) then echo SNR.

    With the corrected single temporal sqrt(N_ens) (not N), the epoch swings the verdict
    as 10*log10(N) not 20*log10(N); on the honest preset epoch's leverage is reduced
    *further* because the coherence window (a ~200 ms beta burst) caps how much of a
    longer epoch actually integrates (D1), so an epoch swept past the coherence window
    barely moves the verdict. Skull loss still dominates (a squared, two-way,
    exponential-in-dB term) and echo SNR is #2.
    """
    mech = _mech()
    acq = AcquisitionParams.demo_motor()
    ranking = acoustic_axis_ranking(mech, acq)
    assert ranking[0].axis_name == "skull_loss_db_oneway"
    top_two = {ranking[0].axis_name, ranking[1].axis_name}
    assert top_two == {"skull_loss_db_oneway", "echo_snr_linear"}
    assert ranking[0].db_span > 20.0
    # Epoch is now only weakly material: the coherence window caps the integration it buys.
    epoch = next(a for a in ranking if a.axis_name == "epoch_s")
    assert 0.0 < epoch.db_span < 10.0


def test_sweep_rejects_bad_axis():
    with pytest.raises(ValueError):
        sweep_axis(_mech(), AcquisitionParams.demo_motor(), "not_a_field",
                   np.array([1.0]))
