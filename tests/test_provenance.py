"""Invariant 6 - provenance non-null.

No quantity reaches a deliverable without a populated provenance chain, tested
end-to-end from the cited constant through the activity -> mechanics model.
"""

from __future__ import annotations

import pytest

from base_neural_model import displacement_sweep, get_single_neuron_displacement
from base_neural_model.base.bands import Band
from base_neural_model.base.provenance import Provenance, extend, merge


def test_cited_constant_carries_provenance():
    d = get_single_neuron_displacement()
    assert d.provenance.source
    assert len(d.provenance.assumptions) >= 1
    assert d.provenance.band is Band.CONTENT_FAST


def test_source_chain_extends_provenance(d_single, voxel, source_params, synchrony_grid):
    """Every source quantity carries the cited provenance plus the chain's own.

    The load-bearing unknowns (eta, kappa, sigma_t) must be named in provenance
    so a reviewer can contest each factor in isolation (source physics doc 6).
    """
    sweep = displacement_sweep(d_single, voxel, source_params, synchrony_grid)
    base_n = len(d_single.provenance.assumptions)
    for s in sweep:
        assert s.provenance.source == d_single.provenance.source
        # The chain adds the volume relation, packing, kappa/eta, and jitter.
        assert len(s.provenance.assumptions) > base_n
        assert s.provenance.band is Band.CONTENT_FAST
        joined = " ".join(s.provenance.assumptions)
        assert "eta" in joined
        assert "kappa" in joined
        assert "sigma_t" in joined


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


def test_end_to_end_model_provenance_chain_populated():
    """The full activity -> mechanics report decomposes to its cited inputs."""
    from base_neural_model.model import run_neural_model

    report = run_neural_model(duration_s=0.3, fs_hz=2000.0)
    assert report.provenance.source
    # The chain carries both the activity (E/I) and the cited single-neuron sources.
    joined = report.provenance.source + " ".join(report.provenance.assumptions)
    assert "Wilson" in joined or "E/I" in joined
    assert any("single-neuron" in a or "AP membrane" in a
               for a in report.provenance.assumptions)
