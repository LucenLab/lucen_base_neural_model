"""k-Wave medium assembly (spec section 1, ``module2_propagation/medium.py``).

Assembles a ``k-wave-python`` medium from a :class:`SkullAcousticMap`. This is the
only module that touches ``k-wave-python`` directly, and it imports it *lazily*
inside the function body so that importing :mod:`lucen.propagation` - or running
Modules 1 and 3, or the unit/contract tests - never requires the heavy k-Wave
dependency to be installed. Install it with the optional extra::

    uv sync --extra propagation
"""

from __future__ import annotations

from typing import Any

from lucen.propagation.skull import SkullAcousticMap


def _import_kwave() -> Any:
    """Import ``kwave`` lazily, with an actionable error if it is missing.

    Keeps the k-Wave dependency optional: only the propagation path needs it.
    """
    try:
        import kwave  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "k-wave-python is required for Module 2 propagation but is not "
            "installed. Install the optional extra: uv sync --extra propagation"
        ) from exc
    return kwave


def build_kwave_medium(skull: SkullAcousticMap) -> Any:
    """Assemble a k-Wave ``kWaveMedium`` from the skull acoustic maps.

    Returns the medium object k-Wave's propagator consumes (sound speed, density,
    and power-law absorption fields).

    Stub - body is implementation work (spec build step 4).
    """
    _import_kwave()
    raise NotImplementedError(
        "build_kwave_medium: kWaveMedium assembly not yet implemented "
        "(spec section 1, module2_propagation/medium.py)"
    )
