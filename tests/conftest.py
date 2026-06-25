"""Shared fixtures for the invariant tests."""

from __future__ import annotations

import numpy as np
import pytest

from lucen.base.types import VoxelGeometry
from lucen.source import get_single_neuron_displacement


@pytest.fixture
def d_single():
    """The cited single-neuron content-band displacement."""
    return get_single_neuron_displacement()


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
