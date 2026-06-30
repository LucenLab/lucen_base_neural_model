"""Shared fixtures for the invariant tests."""

from __future__ import annotations

import numpy as np
import pytest

from base_neural_model import get_single_neuron_displacement
from base_neural_model.base.types import MechanicsParams, VoxelGeometry


@pytest.fixture
def d_single():
    """The cited single-neuron content-band displacement."""
    return get_single_neuron_displacement()


@pytest.fixture
def source_params(d_single):
    """Central-column transduction-chain params (source physics doc, section 5).

    ``membrane_disp_m`` is pinned to the cited ``d_single`` so the constant
    enters the chain in exactly one place (Invariant 2).
    """
    from dataclasses import replace

    return replace(MechanicsParams.central(), membrane_disp_m=d_single.value_m)


@pytest.fixture
def voxel():
    """A representative mm-scale speech-cortex voxel."""
    return VoxelGeometry(
        extent_axial_m=3e-4,
        extent_lateral_m=1e-3,
        neuron_count=10_000,
        depth_m=2e-2,
    )


@pytest.fixture
def synchrony_grid():
    """The standard synchrony sweep grid, s in [0, 1]."""
    return np.linspace(0.0, 1.0, 51)


@pytest.fixture
def matrix_params():
    """A fast-band (undrained) matrix: spherical inclusion, nu near incompressible."""
    from base_neural_model.mechanics.eshelby import MatrixParams

    return MatrixParams(poisson_ratio=0.47, aspect_ratio=1.0)


@pytest.fixture
def poro_params(voxel):
    """A central poroelastic context for the eta sub-model (drainage at the gate)."""
    from base_neural_model.mechanics.poroelastic import PoroelasticParams

    k = 10 ** -12.5
    return PoroelasticParams(
        permeability_m4_per_Ns=k,
        porosity=0.2,
        poro_diffusivity_m2_per_s=k * 1e9,
        gate_time_s=1e-3,
        drainage_length_m=voxel.extent_axial_m,
    )


@pytest.fixture
def sobol_problem():
    """The canonical source factor space (eta/kappa/s + geometry controls)."""
    from base_neural_model.model.sensitivity import SobolProblem

    return SobolProblem.default()


# --- Activity-layer fixtures (the dynamical neural-mass model) -------------------


@pytest.fixture
def ei_params():
    """Central E/I neural-mass parameters in the oscillatory (limit-cycle) regime."""
    from base_neural_model.activity.populations import EIParams

    return EIParams.central()


@pytest.fixture
def activity_timeseries(ei_params):
    """One integration of the central E/I model (1 s at 2 kHz) with synchrony r(t)."""
    from base_neural_model.activity.timeseries import run_activity

    return run_activity(ei_params, duration_s=1.0, fs_hz=2000.0)


@pytest.fixture
def neural_state(activity_timeseries):
    """The reduced NeuralState (s, sigma_t, f_c, rate) the mechanics consume."""
    from base_neural_model.activity.reduce import reduce_to_state

    return reduce_to_state(activity_timeseries)
