"""The cited single-neuron displacement (Design Invariant 2).

This figure is a CITED CONSTANT, never a simulated quantity. The micromechanical
simulation that would derive it is deliberately out of scope - the displacement of
a membrane during an action potential has been measured optically in vivo, so it
enters the system as one number with a literature provenance string attached.

The exact value is the one thing a reviewer is *meant* to be able to contest. It
is isolated here, behind provenance, precisely so that contesting it is a one-line
change and nothing downstream hides an alternative assumption.
"""

from __future__ import annotations

from base_neural_model.base.bands import Band, require_content_fast
from base_neural_model.base.provenance import Provenance
from base_neural_model.base.types import NeuronDisplacement

# --- The cited figure -----------------------------------------------------------
# Both source papers measure MAMMALIAN neurons and report displacement at TWO spatial
# scales that are the same underlying motion, not competing measurements: a LOCAL peak
# at one point on the membrane (Yang: regions individually +/-; Ling: soma local peaks
# +1.4/-1.8 nm, neurite +0.9/-0.7 nm), and a WHOLE-CELL spatially-averaged magnitude
# after summing those same regions WITH their sign (Yang: "~0.2 nm... averaged value
# over the entire neuron"; Ling: "mean absolute displacement" 0.4 nm for soma). Because
# different regions of one cell move in opposite directions at the same instant (Yang:
# "region 1... negative... region 2... positive"; Ling: "Most of the soma rose up, while
# the left boundary fell"), the whole-cell average is smaller than the local peak by
# intra-cell cancellation alone -- it is NOT a different or more-conservative
# measurement of a different phenomenon, and the local 1-3 nm range some other citations
# quote is NOT from giant/squid axons (a prior version of this comment misattributed it).
# This chain's Delta r must be the WHOLE-CELL figure (0.2-0.4 nm): the coherent-strain
# step downstream (mechanics.transduction, s = population/inter-cell synchrony) discounts
# only cell-to-cell phase coherence, so the input radius change must already have any
# within-cell (soma-vs-neurite, sign-heterogeneous) cancellation folded in, or that
# cancellation would never be applied at all. The prior 2 nm value was a local-peak-scale
# figure used at the whole-cell chain position -- the actual error this revision fixes.
CITED_DISPLACEMENT_MIN_M: float = 0.2e-9  # whole-cell mammalian lower edge (ACS Nano 2018)
CITED_DISPLACEMENT_MAX_M: float = 0.4e-9  # whole-cell mammalian upper edge (PNAS 2020, QPI)
_CITED_DISPLACEMENT_M: float = 0.4e-9     # metres (~0.4 nm), content band, mammalian, whole-cell

_CITED_PROVENANCE = Provenance(
    source="Yang et al. 2018 (ACS Nano) & Ling et al. 2020 (PNAS) - optical "
    "action-potential membrane displacement, WHOLE-CELL spatially-averaged magnitude "
    "(mammalian neurons)",
    assumptions=(
        "single-neuron AP membrane displacement is ~0.2-0.4 nm in mammalian cells "
        "when averaged WITH SIGN over the whole cell footprint; the same papers report "
        "larger LOCAL peaks at individual points (Ling: soma +1.4/-1.8 nm) that are the "
        "identical motion before intra-cell (soma-vs-neurite, opposite-sign) "
        "cancellation -- using the local-peak figure here would double-count what the "
        "downstream synchrony factor s (inter-cell coherence) already discounts",
        "the measured optical displacement is content-band (fast), not the slow "
        "rhythmic envelope",
        "the figure transfers from the measured preparation to speech-/motor-cortex "
        "pyramidal neurons without order-of-magnitude correction",
        "displacement adds linearly per neuron up to the coherent-strain saturation "
        "cap (see mechanics.transduction saturation_strain); no per-neuron saturation "
        "below it",
    ),
    band=Band.CONTENT_FAST,
)


def get_single_neuron_displacement() -> NeuronDisplacement:
    """Return the cited single-neuron content-band displacement.

    Attaches provenance and guarantees ``band is CONTENT_FAST`` (Invariant 3);
    raises ``BandError`` via :func:`require_content_fast` if the cited figure is
    ever mislabelled as a non-content band.
    """
    require_content_fast(
        _CITED_PROVENANCE.band, context="get_single_neuron_displacement"
    )
    return NeuronDisplacement(
        value_m=_CITED_DISPLACEMENT_M,
        band=Band.CONTENT_FAST,
        provenance=_CITED_PROVENANCE,
    )
