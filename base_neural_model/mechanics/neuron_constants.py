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
# Optical / full-field interferometric measurements of action-potential-correlated
# membrane displacement report nanometre-scale motion. We take a defensible
# mid-range content-band value; the provenance below records the source and the
# assumptions a reviewer would challenge to move it.
_CITED_DISPLACEMENT_M: float = 2.0e-9  # metres (~2 nm), content band

_CITED_PROVENANCE = Provenance(
    source="Ling et al. 2020, Light: Sci. Appl. - optical action-potential "
    "membrane displacement (nanometre scale)",
    assumptions=(
        "single-neuron AP membrane displacement is ~1-3 nm in vivo",
        "the measured optical displacement is content-band (fast), not the slow "
        "rhythmic envelope",
        "the figure transfers from the measured preparation to speech-cortex "
        "pyramidal neurons without order-of-magnitude correction",
        "displacement adds linearly per neuron before the synchrony model is "
        "applied (no per-neuron saturation in the relevant regime)",
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
