"""Invariant 3 / spec section 7.2 - band guard.

Any attempt to feed an ENVELOPE_SLOW or BROADBAND displacement into the source
transduction chain raises. The envelope-vs-content category error is
unrepresentable.
"""

from __future__ import annotations

import pytest

from base_neural_model import mechanical_displacement
from base_neural_model.base.bands import Band, BandError, require_content_fast
from base_neural_model.base.provenance import Provenance
from base_neural_model.base.types import NeuronDisplacement


def _neuron_with_band(band: Band) -> NeuronDisplacement:
    prov = Provenance(source="test", assumptions=("test",), band=band)
    return NeuronDisplacement(value_m=2e-9, band=band, provenance=prov)


def test_require_content_fast_passes_content():
    # Must not raise.
    require_content_fast(Band.CONTENT_FAST, context="test")


@pytest.mark.parametrize("band", [Band.ENVELOPE_SLOW, Band.BROADBAND])
def test_require_content_fast_rejects_non_content(band):
    with pytest.raises(BandError):
        require_content_fast(band, context="test")


@pytest.mark.parametrize("band", [Band.ENVELOPE_SLOW, Band.BROADBAND])
def test_source_displacement_rejects_non_content_band(band, voxel, source_params):
    """The category error is unrepresentable at the source-chain boundary."""
    bad = _neuron_with_band(band)
    with pytest.raises(BandError):
        mechanical_displacement(bad, voxel, source_params, synchrony_fraction=1.0)


def test_source_displacement_accepts_content_band(voxel, source_params):
    good = _neuron_with_band(Band.CONTENT_FAST)
    result = mechanical_displacement(good, voxel, source_params, synchrony_fraction=1.0)
    assert result.band is Band.CONTENT_FAST
