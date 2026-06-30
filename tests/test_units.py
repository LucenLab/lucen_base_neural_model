"""Invariant 1 / spec section 7.1 - unit consistency.

Any displacement crossing a module boundary is in metres and within physically
sane bounds. Catches the nm/um class of error that silently flips the verdict.
"""

from __future__ import annotations

import math

import pytest

from base_neural_model import displacement_sweep
from base_neural_model.base.types import MechanicsParams, VoxelGeometry
from base_neural_model.base.units import (
    DISPLACEMENT_MAX_M,
    DISPLACEMENT_MIN_M,
    db20,
    is_sane_displacement_m,
    m_to_nm,
    m_to_um,
)


@pytest.mark.parametrize(
    "value_m, expected",
    [
        (1e-9, True),     # 1 nm - a plausible single-neuron displacement
        (9e-8, True),     # 90 nm - the optimistic-column axial source displacement
        (1e-15, False),   # sub-picometre - always a units bug
        (1e-1, False),    # 10 cm - always a units bug
        (0.0, False),
        (-1e-9, False),
        (float("nan"), False),
        (float("inf"), False),
    ],
)
def test_is_sane_displacement_bounds(value_m, expected):
    assert is_sane_displacement_m(value_m) is expected


def test_sane_bounds_are_ordered():
    assert DISPLACEMENT_MIN_M < DISPLACEMENT_MAX_M


def test_reporting_converters_are_pure_scaling():
    assert m_to_nm(1e-9) == pytest.approx(1.0)
    assert m_to_um(1e-6) == pytest.approx(1.0)


def test_db20_matches_definition():
    assert db20(10.0) == pytest.approx(20.0)
    assert db20(1.0) == pytest.approx(0.0)
    assert db20(100.0) == pytest.approx(40.0)


def test_db20_rejects_nonpositive_ratio():
    with pytest.raises(ValueError):
        db20(0.0)
    with pytest.raises(ValueError):
        db20(-3.0)


def test_voxel_volume_matches_doc_voxel():
    """The 0.3 x 1 x 1 mm voxel of the source physics doc is 3e-10 m^3.

    ``depth_m`` (propagation path length) must NOT enter the volume.
    """
    g = VoxelGeometry(
        extent_axial_m=3e-4, extent_lateral_m=1e-3, neuron_count=10_000, depth_m=2e-2
    )
    assert g.volume_m3 == pytest.approx(3e-10, rel=1e-12)


def test_derived_cell_volume_fraction_matches_definition():
    """f_cell = N * (4/3 pi r^3) / V_voxel, the explicit redundancy resolver."""
    g = VoxelGeometry(
        extent_axial_m=3e-4, extent_lateral_m=1e-3, neuron_count=20_000, depth_m=2e-2
    )
    r = 8e-6
    expected = g.neuron_count * (4.0 / 3.0) * math.pi * r**3 / g.volume_m3
    assert g.cell_volume_fraction(r) == pytest.approx(expected, rel=1e-12)
    # N=20k at r=8um reconciles with the central column's asserted f_cell=0.15.
    assert g.cell_volume_fraction(r) == pytest.approx(0.15, rel=0.05)


def test_derived_cell_volume_fraction_rejects_nonpositive_radius():
    g = VoxelGeometry(
        extent_axial_m=3e-4, extent_lateral_m=1e-3, neuron_count=10_000, depth_m=2e-2
    )
    with pytest.raises(ValueError):
        g.cell_volume_fraction(0.0)
    with pytest.raises(ValueError):
        g.cell_volume_fraction(-8e-6)


def test_volume_fraction_consistency_check_passes_when_reconciled():
    """Opt-in consistency check: N=20k at r=8um reconciles with f_cell=0.15."""
    g = VoxelGeometry(
        extent_axial_m=3e-4, extent_lateral_m=1e-3, neuron_count=20_000, depth_m=2e-2
    )
    # Must not raise: derived f_cell ~ 0.143 is within 10% of the asserted 0.15.
    MechanicsParams.central().check_volume_fraction_consistency(g)


def test_volume_fraction_consistency_check_raises_on_mismatch():
    """The standard N=10k voxel implies f_cell ~ 0.072, far from the asserted 0.15."""
    g = VoxelGeometry(
        extent_axial_m=3e-4, extent_lateral_m=1e-3, neuron_count=10_000, depth_m=2e-2
    )
    with pytest.raises(ValueError, match="cell_volume_fraction"):
        MechanicsParams.central().check_volume_fraction_consistency(g)


def test_volume_fraction_consistency_tolerance_is_tunable():
    """A loose tolerance accepts the same pair a tight one rejects."""
    g = VoxelGeometry(
        extent_axial_m=3e-4, extent_lateral_m=1e-3, neuron_count=10_000, depth_m=2e-2
    )
    params = MechanicsParams.central()
    with pytest.raises(ValueError):
        params.check_volume_fraction_consistency(g, rel_tol=0.1)
    # ~0.072 vs 0.15 is within a factor of ~2.1; a generous tolerance accepts it.
    params.check_volume_fraction_consistency(g, rel_tol=1.5)


def test_shipped_default_voxel_is_consistent_with_central_preset():
    """Regression guard: the SHIPPED default voxel must reconcile with central().

    The guard above tests the mechanism with local voxels; this pins the actual
    constants the model ships. DEFAULT_VOXEL (N=21,000) paired with
    MechanicsParams.central() (f_cell=0.15 at r=8 um) must pass at the tight default
    tolerance - the silent N=10,000 mismatch that inflated every generic displacement
    2x cannot recur without failing here.
    """
    from base_neural_model.model.run import DEFAULT_VOXEL

    MechanicsParams.central().check_volume_fraction_consistency(DEFAULT_VOXEL)


def test_module1_outputs_are_sane_si(d_single, voxel, source_params, synchrony_grid):
    """Every displacement Module 1 emits crosses the boundary in sane SI metres.

    The two terms vanish at opposite ends: the coherent axial term is zero at
    ``s=0`` (no coherent signal) and the incoherent pedestal is zero at ``s=1``
    (every cell locked). Each is checked for sanity only where it is nonzero.
    """
    sweep = displacement_sweep(d_single, voxel, source_params, synchrony_grid)
    for s in sweep:
        assert math.isfinite(s.axial_displacement_m)
        assert math.isfinite(s.incoherent_pedestal_m)
        if s.synchrony_fraction > 0.0:
            assert is_sane_displacement_m(s.axial_displacement_m), (
                f"axial displacement {s.axial_displacement_m} m at synchrony "
                f"{s.synchrony_fraction} is outside sane SI bounds"
            )
        if s.synchrony_fraction < 1.0:
            assert is_sane_displacement_m(s.incoherent_pedestal_m), (
                f"incoherent pedestal {s.incoherent_pedestal_m} m at synchrony "
                f"{s.synchrony_fraction} is outside sane SI bounds"
            )
