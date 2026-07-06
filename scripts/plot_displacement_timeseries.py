"""Deliverable (a): the activity-driven displacement timeseries dz(t) across a movement.

A steady rhythm gives a near-flat dz(t); the M1 story that moves is a MOVEMENT TRIAL. The
motor drive imposes the sensorimotor signature -- resting beta, suppression (desynchrony)
at movement onset, then a rebound above baseline -- and because the tissue displacement
tracks synchrony, dz(t) shows the same arc. Panel 1 overlays dz(t) (raw and after the
jitter low-pass) with the synchrony that drives it; the movement window is shaded. Panel 2
is the content-band spectrum of the steady rhythm and its beta line.

Run::

    uv run python scripts/plot_displacement_timeseries.py            # writes dz_t.png
    uv run python scripts/plot_displacement_timeseries.py --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _style as S
import numpy as np

from base_neural_model import run_motor_cortex, run_motor_trial
from base_neural_model.activity.motor_drive import MovementProfile

_ONSET_S = 0.4
_MOVE_S = 0.3


def build_figure():
    import matplotlib.pyplot as plt

    S.apply_style()
    profile = MovementProfile(onset_time_s=_ONSET_S, move_duration_s=_MOVE_S)
    trial = run_motor_trial(profile)
    ts = trial.displacement_timeseries
    steady = run_motor_cortex()
    sp = steady.spectrum

    fig = plt.figure(figsize=(13.5, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], left=0.06, right=0.98,
                          top=0.80, bottom=0.14, wspace=0.24)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    S.suptitle(
        fig,
        "Displacement timeseries dz(t)  |  motor cortex (M1)",
        "A movement trial: resting beta -> suppression at onset -> rebound. The tissue "
        "displacement tracks the synchrony.",
    )

    # --- Panel 1: dz(t) (raw + surviving) with synchrony on a twin-free overlay -----
    t = ts.t_s
    ax0.axvspan(_ONSET_S, _ONSET_S + _MOVE_S, color=S.OSMOTIC, alpha=0.10, zorder=0)
    ax0.text(_ONSET_S + _MOVE_S / 2, ts.dz.max() * 1e9 * 1.02, "movement",
             ha="center", va="bottom", fontsize=8.5, color=S.OSMOTIC, fontweight="bold")

    ax0.plot(t, ts.dz * 1e9, color=S.MUTED, lw=1.0, label="dz(t) raw")
    ax0.plot(t, ts.surviving_dz * 1e9, color=S.DIRECT, lw=2.0,
             label="dz(t) after jitter low-pass")
    ax0.set_ylabel("axial displacement  (nm)")
    ax0.set_xlabel("time  (s)")
    ax0.set_title("1. dz(t) through a movement trial")
    ax0.set_ylim(bottom=0)
    ax0.legend(loc="lower right", ncol=2)
    S.tidy(ax0, xgrid=False)

    # Annotate the three phases directly.
    ymax = ts.dz.max() * 1e9
    for tt, label, yfrac in [(0.15, "resting\nbeta", 0.90),
                             (_ONSET_S + _MOVE_S / 2, "beta\nsuppressed", 0.55),
                             (1.15, "beta\nrebound", 0.72)]:
        i = np.argmin(np.abs(t - tt))
        ax0.annotate(label, xy=(tt, ts.surviving_dz[i] * 1e9),
                     xytext=(tt, ymax * yfrac),
                     fontsize=7.8, color=S.INK_2, ha="center", va="center",
                     arrowprops=dict(arrowstyle="->", color=S.MUTED, lw=0.8))

    # --- Panel 2: content-band spectrum of the steady rhythm ------------------------
    f = sp.freqs_hz
    keep = f <= 60
    ax1.fill_between(f[keep], sp.power[keep] / sp.power[keep].max(),
                     color=S.DIRECT, alpha=0.18, zorder=2)
    ax1.plot(f[keep], sp.power[keep] / sp.power[keep].max(), color=S.DIRECT, lw=1.8,
             zorder=3)
    ax1.axvline(sp.boundary_hz, color=S.INK_2, lw=1.0, ls="--")
    ax1.text(sp.boundary_hz + 1, 0.92, f"content edge\n{sp.boundary_hz:.0f} Hz",
             fontsize=8, color=S.INK_2, va="top")
    # mark the beta peak
    ipk = np.argmax(sp.power[keep])
    fpk = f[keep][ipk]
    ax1.annotate(f"beta {fpk:.0f} Hz", xy=(fpk, 1.0), xytext=(fpk + 8, 0.75),
                 fontsize=8.5, color=S.INK, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=S.INK_2))
    ax1.set_xlim(0, 60)
    ax1.set_ylim(0, 1.08)
    ax1.set_xlabel("frequency  (Hz)")
    ax1.set_ylabel("normalized power")
    ax1.set_title("2. Content-band spectrum (steady beta)")
    S.tidy(ax1, xgrid=False)

    S.footer(fig, "Recomputed live: dz(t) from run_motor_trial(); spectrum from "
                  "run_motor_cortex(). Displacement is the surviving content-band dz.")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("plots/dz_t.png"))
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = build_figure()
    S.save(fig, args.output, dpi=args.dpi)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
