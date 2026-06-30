r"""Orientation coherence emerging from the activity (the directional Option 2).

The mechanics' directional channel (``mechanics/orientation.py``) reads an
orientation order parameter ``Q in [0, 1]`` - how aligned the population's
displacement axes are in 3-D. Option 1 hardcoded ``Q`` on ``MechanicsParams``. Here
it instead **emerges from the activity**, the directional analogue of how temporal
synchrony emerges from the E/I rhythm.

Two ingredients combine, one structural and one dynamical:

* **structural alignment** ``Q_struct in [0, 1]`` - an anatomical property of the
  population: how columnar the cells are. Pyramidal cells in a cortical column share
  an apical-dendrite axis (high ``Q_struct``); a mixed/interneuron-rich patch is
  closer to isotropic (low ``Q_struct``). This is set by the population, not the
  dynamics, and is the natural place the motor cortex differs (Betz/layer-5
  pyramidal cells are large and strongly columnar).

* **dynamical expression** - even structurally aligned axes only contribute
  *coherently* when the cells co-activate in phase. When firing is temporally
  scattered, aligned cells push along their shared axis but at random *times*, so the
  net directional sum washes out just as the monopole's coherent term does. The
  expressed fraction therefore rises with the temporal synchrony ``r``: a column that
  is anatomically aligned but firing incoherently expresses little directional
  signal; the same column firing in tight synchrony expresses it fully.

The effective order parameter the mechanics consume is the product

    Q_eff = Q_struct * f_express(r) ,   f_express(r) = r ,

a deliberately simple, monotone gating (the load-bearing claims are the two limits:
``Q_eff = 0`` when either the structure is isotropic *or* the firing is incoherent,
and ``Q_eff = Q_struct`` at full synchrony). Temporal synchrony ``s`` and directional
coherence ``Q`` remain distinct axes - they are merely coupled through the shared
requirement of in-phase co-activation.
"""

from __future__ import annotations


def expression_factor(synchrony: float) -> float:
    """Fraction of structural alignment expressed at temporal synchrony ``r``.

    ``f_express(r) = r``: aligned axes contribute to the directional sum only insofar
    as the cells fire in phase. 0 at incoherent firing, 1 at full synchrony. Kept
    deliberately simple; what is load-bearing is monotonicity and the two limits.
    """
    if not 0.0 <= synchrony <= 1.0:
        raise ValueError(f"synchrony must lie in [0, 1], got {synchrony!r}")
    return synchrony


def effective_orientation_coherence(
    structural_alignment: float, synchrony: float
) -> float:
    """Effective orientation order ``Q_eff = Q_struct * f_express(r)``, in [0, 1].

    ``structural_alignment`` is the anatomical columnar alignment ``Q_struct`` (1 for
    a perfectly columnar pyramidal population, 0 for isotropic); ``synchrony`` is the
    temporal Kuramoto order ``r``. The product is the orientation coherence the
    mechanics' directional channel consumes - zero unless the population is *both*
    structurally aligned *and* firing coherently.
    """
    if not 0.0 <= structural_alignment <= 1.0:
        raise ValueError(
            f"structural_alignment must lie in [0, 1], got {structural_alignment!r}"
        )
    return structural_alignment * expression_factor(synchrony)
