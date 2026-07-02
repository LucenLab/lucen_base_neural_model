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
# membrane displacement in MAMMALIAN neurons report SUB-nanometre motion (~0.2-0.4 nm);
# the larger 0.5-5 nm figures are from giant (squid) axons, and 2.4 nm / 9 nm are
# bilayer / whole-cell model estimates. The prior 2 nm was the optimistic top of that
# spread. We take a defensible mammalian mid-to-upper value; the provenance records the
# sources and the assumptions a reviewer would challenge to move it.
CITED_DISPLACEMENT_MIN_M: float = 0.2e-9  # sub-nm mammalian lower edge (ACS Nano 2018)
CITED_DISPLACEMENT_MAX_M: float = 0.4e-9  # mammalian upper edge (PNAS 2020, QPI)
_CITED_DISPLACEMENT_M: float = 0.4e-9     # metres (~0.4 nm), content band, mammalian

_CITED_PROVENANCE = Provenance(
    source="Yang et al. 2018 (ACS Nano) & Ling et al. 2020 (PNAS / Light: Sci. Appl.) "
    "- optical action-potential membrane displacement; sub-nm in mammalian neurons",
    assumptions=(
        "single-neuron AP membrane displacement is ~0.2-0.4 nm in mammalian cells "
        "(0.5-5 nm only in giant/squid axons; 2 nm was the optimistic prior value)",
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
