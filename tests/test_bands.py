"""Invariant 3 / spec section 7.2 - band guard.

Any attempt to feed an ENVELOPE_SLOW or BROADBAND displacement into the population
summation raises. The envelope-vs-content category error is unrepresentable.
"""

from __future__ import annotations

import pytest

from lucen.base.bands import Band, BandError, require_content_fast
from lucen.base.provenance import Provenance
from lucen.base.types import NeuronDisplacement
from lucen.source import summed_displacement


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
def test_summed_displacement_rejects_non_content_band(band, voxel):
    """The category error is unrepresentable at the population-summation boundary."""
    bad = _neuron_with_band(band)
    with pytest.raises(BandError):
        summed_displacement(bad, voxel, synchrony_fraction=1.0)


def test_summed_displacement_accepts_content_band(voxel):
    good = _neuron_with_band(Band.CONTENT_FAST)
    result = summed_displacement(good, voxel, synchrony_fraction=1.0)
    assert result.band is Band.CONTENT_FAST
