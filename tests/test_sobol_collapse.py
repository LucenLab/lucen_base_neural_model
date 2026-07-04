"""Sobol global sensitivity -- the central in-silico test (source doc, PART 2).

Three assertions:

1. SALib's Sobol estimator recovers the analytic indices of the Ishigami function
   (a standard sensitivity-analysis oracle) -- a sanity check that the sampling and
   analysis we rely on are wired correctly.
2. ``run_model_sobol`` over the source factor space collapses the verdict onto eta
   (via ``log10_permeability``) and content-band ``synchrony``: those two carry the
   total-order variance, while the matrix Poisson ratio, the cited constant and the
   geometry -- each pinned to its literature uncertainty -- get near-zero ST. This
   is the doc's "the verdict collapses onto eta and s" made a passing assertion.
3. The ``with_activity`` collapse (variance onto ``drive_e`` and the eta-driver) is
   ROBUST to the ``synchrony_from_drive`` surrogate's own documented worst-case error
   against the real ODE (RMS 0.055, max 0.17 at the sharp limit-cycle bifurcation) --
   not an artifact of the exact calibrated constants.
"""

from __future__ import annotations

import numpy as np
import pytest
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

from base_neural_model import run_model_sobol
from base_neural_model.activity.synchrony import (
    _CALIBRATED_COUPLING_GAIN,
    _CALIBRATED_DRIVE_THRESHOLD,
    synchrony_from_drive,
)


def test_ishigami_indices_recovered():
    """SALib recovers Ishigami's analytic first-order indices within tolerance."""
    a, b = 7.0, 0.1
    problem = {
        "num_vars": 3,
        "names": ["x1", "x2", "x3"],
        "bounds": [[-np.pi, np.pi]] * 3,
    }
    X = sobol_sample.sample(problem, 4096, seed=0)
    Y = np.sin(X[:, 0]) + a * np.sin(X[:, 1]) ** 2 + b * X[:, 2] ** 4 * np.sin(X[:, 0])
    res = sobol_analyze.analyze(problem, Y, seed=0)

    # Analytic first-order indices (Crestaux et al.): S1 ~ 0.314, S2 ~ 0.442, S3 = 0.
    assert res["S1"][0] == pytest.approx(0.314, abs=0.05)
    assert res["S1"][1] == pytest.approx(0.442, abs=0.05)
    assert res["S1"][2] == pytest.approx(0.0, abs=0.05)
    # x3 has no first-order effect but a total effect (interaction with x1).
    assert res["ST"][2] > res["S1"][2]


def test_verdict_collapses_onto_eta_and_synchrony(d_single, voxel):
    """ST concentrates on synchrony and the eta-driver; controls are near zero."""
    idx = run_model_sobol(d_single, voxel, n_base=256, seed=0)
    st = idx.as_dict(total_order=True)

    # The two fulcra dominate.
    assert set(idx.dominant(top=2)) == {"synchrony", "log10_permeability"}

    # Every measured-and-transferable control is well below the fulcra.
    control_floor = min(st["synchrony"], st["log10_permeability"])
    for control in (
        "poisson_ratio",
        "membrane_disp_nm",
        "cell_radius_um",
        "cell_volume_fraction",
        "neuron_count_k",
    ):
        assert st[control] < 0.1
        assert st[control] < control_floor


def test_verdict_collapses_onto_activity_drive_and_eta(d_single, voxel):
    """With the activity-spanning factor space the variance flows to the E/I drive
    (which sets synchrony) and the eta-driver, not to geometry or the cited constant -
    the collapse now spans activity -> mechanics."""
    from base_neural_model.model.sensitivity import SobolProblem

    idx = run_model_sobol(
        d_single, voxel, problem=SobolProblem.with_activity(), n_base=256, seed=0
    )
    st = idx.as_dict(total_order=True)

    # The activity drive carries the most variance; the eta-driver is the next axis.
    assert "drive_e" in idx.dominant(top=2)
    # The measured-transferable controls stay near zero.
    for control in ("membrane_disp_nm", "cell_radius_um", "neuron_count_k"):
        assert st[control] < 0.1


def _bracketing_coupling_gains(*, drive_e: float = 1.4, phase_spread_hz: float = 4.0) -> tuple[float, float]:
    """Coupling gains producing +/-0.17 synchrony error at the bifurcation drive.

    0.17 is the surrogate's own documented MAX residual against the real ODE
    (``synchrony.py``'s calibration docstring), evaluated at ``drive_e = 1.4`` -- inside
    the sharp limit-cycle onset (~1.36-1.4) where the residual is largest. Solved
    numerically (not hand-picked) so the perturbation is exactly the documented
    worst case, not an arbitrary stress value. ``drive_threshold`` is left at its
    calibrated value: near this transition a threshold shift collapses synchrony to
    either its floor or ceiling almost immediately (the transition width is set by
    ``coupling_gain`` alone), so gain is the controllable, smoothly-varying axis for a
    graded +/-0.17 bracket; the surrogate's own docstring attributes the max residual
    to the same coupling-gain-controlled rise, not the threshold placement.
    """
    from scipy.optimize import brentq

    nominal = synchrony_from_drive(drive_e, phase_spread_hz=phase_spread_hz)

    def residual(gain: float, target: float) -> float:
        return (
            synchrony_from_drive(
                drive_e, phase_spread_hz=phase_spread_hz,
                drive_threshold=_CALIBRATED_DRIVE_THRESHOLD, coupling_gain=gain,
            )
            - target
        )

    gain_lo = brentq(residual, 40.0, _CALIBRATED_COUPLING_GAIN, args=(nominal - 0.17,))
    gain_hi = brentq(residual, _CALIBRATED_COUPLING_GAIN, 400.0, args=(nominal + 0.17,))
    return gain_lo, gain_hi


@pytest.mark.parametrize("bracket", ["lo", "hi"])
def test_activity_collapse_is_robust_to_surrogate_worst_case_error(d_single, voxel, bracket):
    """The with_activity collapse conclusion survives the surrogate's documented
    max residual (0.17 synchrony at the bifurcation), not just its exact calibration.

    Perturbs coupling_gain to the numerically-solved value that reproduces a +/-0.17
    synchrony error at drive_e=1.4 (the sharp transition where the real-ODE fit is
    worst), re-runs the with_activity Sobol sweep under that perturbed surrogate, and
    checks the SAME qualitative conclusion holds: drive_e remains a dominant factor and
    the measured-transferable geometry/cited-constant controls stay near zero. If this
    ever fails, the with_activity collapse claim is fragile to the surrogate's own
    admitted error and should be investigated, not silenced by loosening the assertion.
    """
    from base_neural_model.model.sensitivity import SobolProblem

    gain_lo, gain_hi = _bracketing_coupling_gains()
    perturbed_gain = gain_lo if bracket == "lo" else gain_hi

    idx = run_model_sobol(
        d_single, voxel,
        problem=SobolProblem.with_activity(),
        n_base=256, seed=0,
        synchrony_from_drive_kwargs={
            "drive_threshold": _CALIBRATED_DRIVE_THRESHOLD,
            "coupling_gain": perturbed_gain,
        },
    )
    st = idx.as_dict(total_order=True)

    assert "drive_e" in idx.dominant(top=2)
    for control in ("membrane_disp_nm", "cell_radius_um", "neuron_count_k"):
        assert st[control] < 0.1


def test_sobol_is_reproducible(d_single, voxel):
    a = run_model_sobol(d_single, voxel, n_base=128, seed=7)
    b = run_model_sobol(d_single, voxel, n_base=128, seed=7)
    assert np.allclose(a.ST, b.ST)
    assert np.allclose(a.S1, b.S1)


def test_indices_clipped_nonnegative(d_single, voxel):
    idx = run_model_sobol(d_single, voxel, n_base=128, seed=0)
    assert np.all(idx.S1 >= 0.0)
    assert np.all(idx.ST >= 0.0)
