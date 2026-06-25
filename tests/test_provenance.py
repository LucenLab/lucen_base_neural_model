"""Invariant 6 / spec section 7.4 - provenance non-null.

No quantity reaches the deliverable without a populated provenance chain. Tested
end-to-end through Module 1 now; the full-pipeline assertion is marked xfail until
orchestration (Modules 2 & 3) lands.
"""

from __future__ import annotations

import numpy as np
import pytest

from lucen.base.bands import Band
from lucen.base.provenance import Provenance, extend, merge
from lucen.source import get_single_neuron_displacement, synchrony_sweep


def test_cited_constant_carries_provenance():
    d = get_single_neuron_displacement()
    assert d.provenance.source
    assert len(d.provenance.assumptions) >= 1
    assert d.provenance.band is Band.CONTENT_FAST


def test_summation_extends_provenance(d_single, voxel, synchrony_grid):
    """Every summed quantity carries the cited provenance plus the model's own."""
    sweep = synchrony_sweep(d_single, voxel, synchrony_grid)
    base_n = len(d_single.provenance.assumptions)
    for s in sweep:
        assert s.provenance.source == d_single.provenance.source
        # Summation adds the interpolation form + the neuron count.
        assert len(s.provenance.assumptions) > base_n
        assert s.provenance.band is Band.CONTENT_FAST


def test_extend_appends_and_relabels():
    p = Provenance(source="src", assumptions=("a",), band=Band.CONTENT_FAST)
    q = extend(p, "b", band=Band.BROADBAND)
    assert q.source == "src"
    assert q.assumptions == ("a", "b")
    assert q.band is Band.BROADBAND


def test_merge_dedupes_and_requires_same_band():
    p = Provenance(source="s1", assumptions=("a", "b"), band=Band.CONTENT_FAST)
    q = Provenance(source="s2", assumptions=("b", "c"), band=Band.CONTENT_FAST)
    m = merge(p, q)
    assert m.assumptions == ("a", "b", "c")  # order-preserving, de-duplicated
    assert m.band is Band.CONTENT_FAST


def test_merge_rejects_band_mismatch():
    p = Provenance(source="s1", assumptions=("a",), band=Band.CONTENT_FAST)
    q = Provenance(source="s2", assumptions=("b",), band=Band.ENVELOPE_SLOW)
    with pytest.raises(ValueError):
        merge(p, q)


@pytest.mark.xfail(
    reason="orchestration (Modules 2 & 3) not yet implemented", strict=True
)
def test_feasibility_curve_provenance_chain_populated():
    """Full-pipeline provenance: enable once run_feasibility_sweep is implemented."""
    from lucen.orchestration import run_feasibility_sweep

    run_feasibility_sweep(  # will raise NotImplementedError -> xfail
        geom=None,
        skull=None,
        array_geometry=None,
        noise=None,
        bulk=None,
        synchrony_grid=np.linspace(0, 1, 5),
        interrogation_freq_hz=2e6,
        n_frames_per_unit=100,
        separability_threshold_db=10.0,
    )
