"""Shared data contracts (spec section 2).

These are the typed objects that cross module boundaries - the actual interface.
Get them right and the three modules can be built independently.

Two deliberate corrections to the spec's literal listing:

* ``FeasibilityCurve`` holds NumPy arrays. A ``frozen=True`` dataclass with an
  ``ndarray`` field generates an ``__eq__``/``__hash__`` that NumPy makes
  ambiguous (``arr == arr`` is array-valued, not bool) and unhashable. So such
  contracts use ``frozen=True, eq=False`` - still immutable, with identity
  equality. Scalar-only contracts keep plain ``frozen=True``.
* ``ArrayGeometry`` is referenced in the spec's Module 2 signatures as a forward
  string but never defined. It is defined here so those signatures resolve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lucen.base.bands import Band
from lucen.base.provenance import Provenance


@dataclass(frozen=True)
class NeuronDisplacement:
    """Single-neuron membrane displacement - a CITED CONSTANT (Invariant 2).

    No module derives this; it enters the system as one number with literature
    provenance. ``band`` MUST be ``CONTENT_FAST`` to feed the population model.
    """

    value_m: float              # metres; ~1e-9 to 3e-9
    band: Band
    provenance: Provenance


@dataclass(frozen=True)
class VoxelGeometry:
    """The voxel whose summed displacement we compute."""

    extent_axial_m: float       # ~3e-4 (range-gate length)
    extent_lateral_m: float     # ~1e-3 (focal width)
    neuron_count: int           # ~1e4-1e5 in a mm-scale voxel
    depth_m: float              # cortical depth, for propagation path length


@dataclass(frozen=True)
class ArrayGeometry:
    """Transducer array geometry (correction #2; referenced but undefined in spec).

    Carried into Module 2's correction and propagation calls. Minimal but
    sufficient to parameterize aberration correction and the return path.
    """

    element_pitch_m: float      # centre-to-centre element spacing, metres
    element_count: int          # number of elements across the aperture
    center_freq_hz: float       # array centre frequency, Hz

    @property
    def aperture_m(self) -> float:
        """Total aperture width, metres."""
        return self.element_pitch_m * self.element_count


@dataclass(frozen=True)
class SummedDisplacement:
    """Module 1 -> Module 2 hand-off. The source term."""

    value_m: float              # summed voxel displacement, metres
    synchrony_fraction: float   # 0.0 (incoherent) .. 1.0 (fully coherent)
    band: Band
    provenance: Provenance


@dataclass(frozen=True)
class PropagatedDisplacement:
    """Module 2 -> Module 3 hand-off.

    Source term after the round trip through skull + correction, as it arrives
    back at the array.
    """

    value_m: float                  # effective displacement recoverable at the array
    one_way_transmission: float     # fraction of amplitude surviving skull, one way
    residual_aberration_rad: float  # residual wavefront error after correction
    synchrony_fraction: float
    band: Band
    provenance: Provenance


@dataclass(frozen=True)
class DetectionResult:
    """Module 3 output. The verdict for one synchrony value."""

    signal_m: float             # integrated, post-clutter displacement estimate
    noise_floor_m: float        # post-filter noise floor
    contrast_db: float          # 20*log10(signal/noise_floor)
    separable: bool             # contrast clears the active/inactive bar
    synchrony_fraction: float


@dataclass(frozen=True, eq=False)
class FeasibilityCurve:
    """The deliverable. Contrast vs synchrony across the full sweep.

    ``eq=False`` because the array fields make a generated ``__eq__`` ambiguous
    and unhashable under NumPy (see module docstring).
    """

    synchrony: np.ndarray               # shape (K,)
    contrast_db: np.ndarray             # shape (K,)
    separable: np.ndarray               # bool, shape (K,)
    crossover_synchrony: float | None   # min synchrony that clears the bar, or None
