"""Shared rhythm-band operating points for the inverse-design plots.

The forward presets (``EIParams.central`` / ``motor_cortex`` / ``motor_cortex_low_beta``)
each settle at a characteristic content corner ``f_c`` and, through the synchrony, a
firing-time jitter ``sigma_t``. The inverse plots (min_eta, pass_space, min_directional,
resolution) used to be drawn at a single idealized low-jitter point, which switched the
Gate-2 jitter low-pass essentially off and hid the band dependence. This module reads the
*real* (f_c, sigma_t) each preset lands at, straight from the live model, so the inverse
figures can be split by band and cannot drift from the dynamics.

Each band is (label, colour, f_c_hz, sigma_t_s). The values are computed once at import.
"""

from __future__ import annotations

from dataclasses import dataclass

from base_neural_model.activity import run_activity
from base_neural_model.activity.populations import EIParams
from base_neural_model.activity.reduce import reduce_to_state


@dataclass(frozen=True)
class BandPoint:
    """A rhythm band's real operating point on the jitter low-pass."""

    label: str
    colour: str
    f_c_hz: float
    sigma_t_s: float


def _operating_point(ei_factory, label: str, colour: str) -> BandPoint:
    """Run the preset live and read back the (f_c, sigma_t) it settles at."""
    state = reduce_to_state(run_activity(ei_factory(), duration_s=1.2, fs_hz=2000.0))
    return BandPoint(
        label=label,
        colour=colour,
        f_c_hz=state.content_freq_hz,
        sigma_t_s=state.jitter_sigma_s,
    )


# Computed once from the live presets; same colours as the summary plots.
BANDS: tuple[BandPoint, ...] = (
    _operating_point(EIParams.central, "gamma (generic)", "#8e44ad"),
    _operating_point(EIParams.motor_cortex, "high-beta (M1)", "#e67e22"),
    _operating_point(EIParams.motor_cortex_low_beta, "low-beta (M1)", "#16a085"),
)

# The idealized reference point the inverse plots historically used: a tight jitter that
# leaves the content essentially intact, so the curve isolates the amplitude trade.
IDEALIZED = BandPoint("idealized (no jitter)", "0.4", 100.0, 0.2e-3)
