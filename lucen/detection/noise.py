"""Noise floor model (spec section 5.2).

The displacement-equivalent noise floor the measured signal is compared against,
including how it improves under coherent integration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NoiseModel:
    """Displacement-equivalent noise floor and its integration behaviour."""

    thermal_floor_m: float              # base displacement-equivalent noise, metres
    aberration_phase_noise_rad: float   # residual-aberration phase noise, radians

    def floor_after_integration(self, n_samples: int) -> float:
        """Effective floor (metres) after ``sqrt(n_samples)`` coherent integration.

        Stub - body is implementation work (spec build step 3).
        """
        raise NotImplementedError(
            "NoiseModel.floor_after_integration: integrated floor not yet "
            "implemented (spec section 5.2)"
        )
