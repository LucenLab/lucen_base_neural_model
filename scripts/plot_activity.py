"""The dynamical layer: the M1 E/I neural-mass rhythm, its synchrony, and its spectrum.

The activity layer integrates a Wilson-Cowan excitatory/inhibitory mean field; for the
motor preset it settles on a beta limit cycle. Panel 1 shows the E and I traces over a
short window (so the cycles are legible) plus the Kuramoto synchrony r(t) it produces;
panel 2 is the power spectrum of E(t), with the beta peak and content-band edge marked.

Run::

    uv run python scripts/plot_activity.py            # writes activity.png
    uv run python scripts/plot_activity.py --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _style as S

from base_neural_model.model import run_motor_cortex


def build_figure():
    import matplotlib.pyplot as plt

    S.apply_style()
    report = run_motor_cortex()
    a = report.activity
    ns = report.neural_state
    sp = a.spectrum

    fig = plt.figure(figsize=(13.5, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], left=0.06, right=0.98,
                          top=0.80, bottom=0.14, wspace=0.24)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    S.suptitle(
        fig,
        "Activity layer  |  motor cortex (M1)",
        f"Wilson-Cowan E/I mean field on a beta limit cycle "
        f"({ns.content_freq_hz:.0f} Hz), synchrony s = {ns.synchrony_fraction:.2f}.",
    )

    # --- Panel 1: E/I traces + synchrony over a 0.6 s window ------------------------
    t = a.t_s
    win = (t >= 0.4) & (t <= 1.0)
    ax0.plot(t[win], a.e[win], color=S.DIRECT, lw=1.7, label="E (excitatory)")
    ax0.plot(t[win], a.i[win], color=S.OSMOTIC, lw=1.4, label="I (inhibitory)")
    ax0.plot(t[win], a.r[win], color=S.VIOLET, lw=1.8, label="synchrony r(t)")
    ax0.set_ylim(0, 1.08)
    ax0.set_xlabel("time  (s)")
    ax0.set_ylabel("activity  /  synchrony")
    ax0.set_title("1. The E/I beta limit cycle and synchrony")
    ax0.legend(loc="upper right", ncol=3)
    S.tidy(ax0, xgrid=False)

    # --- Panel 2: power spectrum of E(t) -------------------------------------------
    f = sp.freqs_hz
    keep = f <= 80
    p = sp.power[keep] / sp.power[keep].max()
    ax1.fill_between(f[keep], p, color=S.DIRECT, alpha=0.18, zorder=2)
    ax1.plot(f[keep], p, color=S.DIRECT, lw=1.8, zorder=3)
    fpk = sp.dominant_freq_hz
    ax1.annotate(f"beta {fpk:.0f} Hz", xy=(fpk, 1.0), xytext=(fpk + 12, 0.78),
                 fontsize=9, color=S.INK, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=S.INK_2))
    ax1.text(0.98, 0.55, f"content fraction\n{sp.content_fraction * 100:.0f}%",
             transform=ax1.transAxes, ha="right", va="top", fontsize=8.5,
             color=S.INK_2)
    ax1.set_xlim(0, 80)
    ax1.set_ylim(0, 1.08)
    ax1.set_xlabel("frequency  (Hz)")
    ax1.set_ylabel("normalized power")
    ax1.set_title("2. E(t) power spectrum")
    S.tidy(ax1, xgrid=False)

    S.footer(fig, "Recomputed live from run_motor_cortex().activity. Synchrony is the "
                  "Kuramoto order parameter r(t) the reduction reads as s.")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("plots/activity.png"))
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
