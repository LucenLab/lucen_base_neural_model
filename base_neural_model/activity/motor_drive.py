r"""Movement-locked drive for the motor-cortex dynamics.

The steady presets (``EIParams.motor_cortex``) hold M1 on a constant beta limit
cycle. A real motor trial is not steady: M1 beta is **suppressed** at movement onset
(movement-related beta desynchronization - the resting rhythm breaks down as the
population is engaged in movement) and then **rebounds** above baseline after the
movement ends (the post-movement beta rebound). This module builds the time-varying
excitatory drive ``P(t)`` that imposes that profile, to be passed to
:func:`integrate_ei` via its ``drive_fn`` hook.

The drive is a baseline with a movement-locked **dip then overshoot** around the
onset time::

    P(t) = drive_base
           - onset_gain   * bump(t; t_onset, dur_move)        # beta suppression at onset
           + rebound_gain * bump(t; t_onset + dur_move + lag, dur_rebound)  # rebound

where ``bump`` is a smooth (Gaussian) pulse. Lowering the drive during movement
weakens the beta limit cycle so the population **desynchronizes** (the order parameter
dips); the post-movement overshoot drives it **above baseline** so beta rebounds. The
shape is a modeling choice; what is load-bearing is the sequence - baseline -> onset
desync -> rebound - and that the population synchrony, hence the directional channel,
is now tied to the movement event rather than constant.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class MovementProfile:
    """Parameters of a movement-locked drive ``P(t)`` around a single onset.

    Times in seconds, drives in the same dimensionless units as ``EIParams.drive_e``.
    """

    drive_base: float = 1.2       # tonic baseline drive (the resting beta cycle)
    onset_time_s: float = 0.4     # movement onset within the record
    move_duration_s: float = 0.3  # movement duration (the desync window)
    onset_gain: float = 0.9       # drive DIP during movement (suppresses/desyncs beta)
    rebound_gain: float = 0.4     # drive overshoot after movement (beta rebound)
    rebound_lag_s: float = 0.1    # gap between movement end and the rebound
    rebound_duration_s: float = 0.3

    def __post_init__(self) -> None:
        for name in (
            "move_duration_s", "rebound_duration_s",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)!r}")
        if self.drive_base < 0.0:
            raise ValueError("drive_base must be >= 0")


def _bump(t: float, center: float, width: float) -> float:
    """Smooth unit-height Gaussian pulse centred at ``center`` with s.d. ``width``."""
    return math.exp(-0.5 * ((t - center) / width) ** 2)


def movement_drive(profile: MovementProfile | None = None) -> Callable[[float], float]:
    """Build the movement-locked drive ``P(t)`` (a callable seconds -> drive).

    Returns a function suitable for :func:`integrate_ei`'s ``drive_fn``: a baseline
    with a drive **dip** at movement onset (which suppresses/desynchronizes beta) and
    a post-movement **overshoot** (the beta rebound). With the default profile and a
    ~1 s record, the trial shows resting beta -> movement desync -> rebound.
    """
    p = profile or MovementProfile()
    move_center = p.onset_time_s + 0.5 * p.move_duration_s
    rebound_center = (
        p.onset_time_s + p.move_duration_s + p.rebound_lag_s
        + 0.5 * p.rebound_duration_s
    )

    def drive(t: float) -> float:
        suppression = p.onset_gain * _bump(t, move_center, 0.5 * p.move_duration_s)
        rebound = p.rebound_gain * _bump(t, rebound_center, 0.5 * p.rebound_duration_s)
        return max(0.0, p.drive_base - suppression + rebound)

    return drive
