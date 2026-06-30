"""Per-factor verdict-flip sweep (source doc, PART 2, Module 1 (iv)).

The complement to the Sobol test: holding all but one factor at nominal, which
factor can flip the verdict across the contrast bar? The expected answer -- the same
claim from the other direction -- is that only eta (via ``log10_permeability``) and
``synchrony`` flip it; the literature-pinned controls cannot move the verdict across
the bar within their ranges.
"""

from __future__ import annotations

from base_neural_model import gate1_predicate, verdict_flip_table
from base_neural_model.model.sensitivity import model_verdict_scalar
from base_neural_model.model.verdict_flip import (
    DEFAULT_FACTORS,
    DEFAULT_NOMINAL,
    find_flip,
)


def _nominal_scalar(d_single, voxel) -> float:
    return model_verdict_scalar(DEFAULT_NOMINAL, DEFAULT_FACTORS, d_single, voxel)


def test_only_eta_and_synchrony_flip(d_single, voxel):
    """A bar set above the nominal verdict can be cleared only by eta or s."""
    y0 = _nominal_scalar(d_single, voxel)
    table = verdict_flip_table(d_single, voxel, gate1_predicate(1.5 * y0))
    assert set(table.flipping_factors) == {"synchrony"} or set(
        table.flipping_factors
    ) <= {"synchrony", "log10_permeability"}
    # No geometry/cited-constant factor is ever in the flipping set.
    for control in (
        "poisson_ratio",
        "membrane_disp_nm",
        "cell_radius_um",
        "cell_volume_fraction",
        "neuron_count_k",
    ):
        assert control not in table.flipping_factors


def test_both_fulcra_flip_with_a_lower_bar(d_single, voxel):
    """A bar below nominal: both eta (k) and s can flip it."""
    y0 = _nominal_scalar(d_single, voxel)
    table = verdict_flip_table(d_single, voxel, gate1_predicate(0.7 * y0))
    assert "synchrony" in table.flipping_factors
    assert "log10_permeability" in table.flipping_factors


def test_flip_value_lies_in_bounds(d_single, voxel):
    """A located flip value sits within the factor's swept range."""
    y0 = _nominal_scalar(d_single, voxel)
    table = verdict_flip_table(d_single, voxel, gate1_predicate(0.7 * y0))
    for row in table.rows:
        if row.flips:
            assert row.lo <= row.flip_value <= row.hi


def test_find_flip_reports_no_flip_when_endpoints_agree(d_single, voxel):
    """If the predicate agrees at both ends, the factor cannot flip the verdict."""
    # An unreachably high bar: no synchrony value clears it -> no flip.
    impossible = gate1_predicate(1e3)
    row = find_flip(
        "synchrony", 0.0, 1.0, DEFAULT_NOMINAL, DEFAULT_FACTORS,
        d_single, voxel, impossible,
    )
    assert row.flips is False
    assert row.flip_value is None


def test_find_flip_locates_crossing_to_tol(d_single, voxel):
    """On a flipping factor, the located value brackets the predicate crossing."""
    y0 = _nominal_scalar(d_single, voxel)
    pred = gate1_predicate(0.7 * y0)
    row = find_flip(
        "synchrony", 0.0, 1.0, DEFAULT_NOMINAL, DEFAULT_FACTORS,
        d_single, voxel, pred, tol=1e-4,
    )
    assert row.flips is True
    # Verdict scalar is monotone up in s: just below the flip fails, just above passes.
    nominal = dict(DEFAULT_NOMINAL)
    below = dict(nominal, synchrony=max(0.0, row.flip_value - 1e-2))
    above = dict(nominal, synchrony=min(1.0, row.flip_value + 1e-2))
    assert not pred(model_verdict_scalar(below, DEFAULT_FACTORS, d_single, voxel))
    assert pred(model_verdict_scalar(above, DEFAULT_FACTORS, d_single, voxel))
