"""Poroelastic eta sub-model: drainage limits, monotonicity, and bounds.

eta is parameterized by the drainage number theta = C_v t / L^2 so it can be SWEPT
across the literature range (source doc, PART 2, Module 1 (ii)) -- not to compute
its true value. These tests pin: monotonicity in theta; the fast-band suppression
(theta << 1 -> near the floor) versus drained ceiling (theta >> 1 -> near ceiling);
eta in [0, 1] across the permeability spread; and the guard rails.
"""

from __future__ import annotations

import numpy as np
import pytest

from base_neural_model.mechanics.poroelastic import (
    PERMEABILITY_MAX,
    PERMEABILITY_MIN,
    PoroelasticParams,
    consolidation_number,
    consolidation_time,
    drainage_completeness,
    eta_from_permeability,
    poroelastic_eta,
)


def _params(**overrides) -> PoroelasticParams:
    base = dict(
        permeability_m4_per_Ns=1e-12,
        porosity=0.2,
        poro_diffusivity_m2_per_s=1e-3,
        gate_time_s=1e-3,
        drainage_length_m=3e-4,
    )
    base.update(overrides)
    return PoroelasticParams(**base)


def test_consolidation_number_and_time():
    assert consolidation_number(2.0, 3.0, 2.0) == pytest.approx(2.0 * 3.0 / 4.0)
    assert consolidation_time(4.0, 2.0) == pytest.approx(4.0 / 4.0)


def test_drainage_completeness_limits_and_monotone():
    assert drainage_completeness(0.0) == pytest.approx(0.0)
    assert drainage_completeness(1e-6) < 1e-5            # theta << 1 -> ~0
    assert drainage_completeness(20.0) == pytest.approx(1.0, abs=1e-6)  # theta >> 1
    thetas = np.linspace(0.0, 10.0, 50)
    vals = np.array([drainage_completeness(t) for t in thetas])
    assert np.all(np.diff(vals) >= 0.0)


def test_eta_suppressed_in_fast_band():
    """theta << 1 (low diffusivity, short gate): eta near the floor."""
    p = _params(poro_diffusivity_m2_per_s=1e-6)  # theta ~ 1e-6*1e-3/(9e-8) ~ 0.011
    eta = poroelastic_eta(p)
    assert eta < 0.1
    assert eta >= p.eta_floor


def test_eta_approaches_ceiling_when_drained():
    """theta >> 1: eta near the ceiling."""
    p = _params(poro_diffusivity_m2_per_s=1e2)  # theta enormous
    assert poroelastic_eta(p) == pytest.approx(p.eta_ceiling, abs=1e-6)


def test_eta_monotone_increasing_in_diffusivity():
    cvs = np.logspace(-7, 2, 30)
    etas = np.array([poroelastic_eta(_params(poro_diffusivity_m2_per_s=c)) for c in cvs])
    assert np.all(np.diff(etas) >= -1e-12)


def test_eta_in_bounds_across_permeability_sweep():
    ks = np.logspace(np.log10(PERMEABILITY_MIN), np.log10(PERMEABILITY_MAX), 40)
    etas = np.array([
        eta_from_permeability(
            k, porosity=0.2, storage_modulus_Pa=1e9, gate_time_s=1e-3,
            drainage_length_m=3e-4,
        )
        for k in ks
    ])
    assert np.all(etas >= 0.0) and np.all(etas <= 1.0)
    assert np.all(np.diff(etas) >= -1e-12)  # monotone up in k


def test_vascular_bypass_lifts_eta():
    """A fast vascular/osmotic source lifts eta toward the ceiling at fixed drainage."""
    base = _params(poro_diffusivity_m2_per_s=1e-5)
    lifted = _params(poro_diffusivity_m2_per_s=1e-5, vascular_source=0.8)
    assert poroelastic_eta(lifted) > poroelastic_eta(base)


@pytest.mark.parametrize(
    "overrides",
    [
        {"permeability_m4_per_Ns": 0.0},
        {"porosity": 0.0},
        {"porosity": 1.0},
        {"poro_diffusivity_m2_per_s": -1.0},
        {"gate_time_s": 0.0},
        {"drainage_length_m": 0.0},
        {"vascular_source": 1.5},
        {"eta_floor": 0.6, "eta_ceiling": 0.5},
    ],
)
def test_params_guard_rails(overrides):
    with pytest.raises(ValueError):
        _params(**overrides)
