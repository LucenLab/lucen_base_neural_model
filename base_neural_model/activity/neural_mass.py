"""Wilson-Cowan E/I neural-mass ODEs, integrated with scipy.

The dynamical core of the activity layer. Two coupled mean-field populations evolve
as::

    tau_e * dE/dt = -E + S_e( w_ee*E - w_ei*I + drive_e )
    tau_i * dI/dt = -I + S_i( w_ie*E - w_ii*I + drive_i )

with a logistic sigmoid ``S(x) = 1 / (1 + exp(-gain*(x - theta)))``. In the central
parameter regime (:meth:`EIParams.central`) the E/I loop sits on a limit cycle: E(t)
oscillates in the fast/content band, and that oscillation is what downstream
synchrony, band power, and the mechanical signal are read from.

The integration is a single ``scipy.integrate.solve_ivp`` call (default RK45). The
returned trajectories are sampled on a uniform time grid so the FFT-based oscillation
analysis (:mod:`.oscillation`) can run directly on them.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.integrate import solve_ivp

from base_neural_model.activity.populations import EIParams


def _sigmoid(x: np.ndarray | float, gain: float, theta: float) -> np.ndarray | float:
    """Logistic response ``1 / (1 + exp(-gain*(x - theta)))`` in [0, 1]."""
    return 1.0 / (1.0 + np.exp(-gain * (x - theta)))


def _rhs(
    t: float,
    y: np.ndarray,
    p: EIParams,
    drive_fn: Callable[[float], float] | None,
) -> list[float]:
    """Wilson-Cowan right-hand side ``[dE/dt, dI/dt]`` at state ``y = [E, I]``.

    ``drive_fn``, when given, supplies a time-varying excitatory drive ``P(t)`` that
    overrides the constant ``p.drive_e`` - the hook the movement-locked motor drive
    uses (:mod:`.motor_drive`). When ``None`` the constant tonic drive is used and the
    behaviour is identical to the steady limit cycle.
    """
    e, i = y
    drive_e_in = p.drive_e if drive_fn is None else drive_fn(t)
    drive_e = _sigmoid(p.w_ee * e - p.w_ei * i + drive_e_in, p.gain_e, p.theta_e)
    drive_i = _sigmoid(p.w_ie * e - p.w_ii * i + p.drive_i, p.gain_i, p.theta_i)
    de = (-e + drive_e) / p.tau_e_s
    di = (-i + drive_i) / p.tau_i_s
    return [de, di]


def integrate_ei(
    params: EIParams,
    *,
    duration_s: float = 1.0,
    fs_hz: float = 2000.0,
    e0: float = 0.1,
    i0: float = 0.1,
    settle_s: float = 0.2,
    drive_fn: Callable[[float], float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the E/I mean-field ODEs and return ``(t, E, I)`` on a uniform grid.

    ``duration_s`` is the kept record length and ``fs_hz`` the sample rate (choose
    >= ~10x the expected oscillation so the FFT resolves it). ``settle_s`` of
    transient is integrated and discarded first so the kept record sits on the limit
    cycle / fixed point, not the start-up transient. ``e0``/``i0`` are the initial
    activations in [0, 1].

    ``drive_fn`` is an optional time-varying excitatory drive ``P(t)`` (seconds ->
    drive); when given it overrides the constant ``params.drive_e``, letting a caller
    impose a movement-locked profile (:mod:`.motor_drive`). Its time origin is the
    kept-record origin (the ``settle_s`` transient is shifted off), so ``t = 0`` in
    ``drive_fn`` is the start of the returned record.

    Uses ``scipy.integrate.solve_ivp`` (RK45) with a dense uniform evaluation grid.
    """
    if duration_s <= 0.0:
        raise ValueError(f"duration_s must be positive, got {duration_s!r}")
    if fs_hz <= 0.0:
        raise ValueError(f"fs_hz must be positive, got {fs_hz!r}")
    if settle_s < 0.0:
        raise ValueError(f"settle_s must be >= 0, got {settle_s!r}")
    for name, v in (("e0", e0), ("i0", i0)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1], got {v!r}")

    total_s = settle_s + duration_s
    n_samples = int(round(duration_s * fs_hz))
    t_eval = settle_s + np.arange(n_samples) / fs_hz

    # Shift drive_fn's origin to the kept record (t=0 at the end of settling).
    shifted_drive = (
        None if drive_fn is None else (lambda t: drive_fn(t - settle_s))
    )

    sol = solve_ivp(
        _rhs,
        (0.0, total_s),
        y0=[e0, i0],
        t_eval=t_eval,
        args=(params, shifted_drive),
        method="RK45",
        rtol=1e-7,
        atol=1e-9,
        max_step=1.0 / fs_hz,
    )
    if not sol.success:
        raise RuntimeError(f"E/I integration failed: {sol.message}")

    # Re-zero the kept record's time origin so t starts at 0.
    t = sol.t - settle_s
    e = sol.y[0]
    i = sol.y[1]
    return t, e, i
