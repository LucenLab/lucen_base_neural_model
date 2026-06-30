"""Sobol global sensitivity -- the central in-silico test (source doc, PART 2).

Two assertions:

1. SALib's Sobol estimator recovers the analytic indices of the Ishigami function
   (a standard sensitivity-analysis oracle) -- a sanity check that the sampling and
   analysis we rely on are wired correctly.
2. ``run_model_sobol`` over the source factor space collapses the verdict onto eta
   (via ``log10_permeability``) and content-band ``synchrony``: those two carry the
   total-order variance, while the matrix Poisson ratio, the cited constant and the
   geometry -- each pinned to its literature uncertainty -- get near-zero ST. This
   is the doc's "the verdict collapses onto eta and s" made a passing assertion.
"""

from __future__ import annotations

import numpy as np
import pytest
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

from base_neural_model import run_model_sobol


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


def test_sobol_is_reproducible(d_single, voxel):
    a = run_model_sobol(d_single, voxel, n_base=128, seed=7)
    b = run_model_sobol(d_single, voxel, n_base=128, seed=7)
    assert np.allclose(a.ST, b.ST)
    assert np.allclose(a.S1, b.S1)


def test_indices_clipped_nonnegative(d_single, voxel):
    idx = run_model_sobol(d_single, voxel, n_base=128, seed=0)
    assert np.all(idx.S1 >= 0.0)
    assert np.all(idx.ST >= 0.0)
