"""The acoustic SNR budget: where the detection wall is, and which axis carries it.

The detection layer turns the source displacement into a through-skull detectability SNR;
this sweep turns that single point into curves. Panel 1 sweeps the skull two-way loss at a
few echo-SNR levels, marks the 0 dB wall (signal == floor) and the honest demo operating
point. Panel 2 ranks the acquisition axes by how far each moves the verdict over its
literature span -- the acoustic-side complement to the source-side (eta, s) collapse. It
leads with skull loss, then echo SNR; the epoch is only weakly material because the honest
coherence window caps how much a longer epoch integrates.

Every number is read from the live detection/budget chain, so the figure cannot drift.

Run::

    uv run python scripts/plot_budget.py            # writes budget.png
    uv run python scripts/plot_budget.py --show
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import _style as S
import numpy as np

from base_neural_model import (
    AcquisitionParams,
    acoustic_axis_ranking,
    run_motor_demo,
    sweep_axis,
)

# Echo-SNR levels (free-field, dB) to draw as separate skull-loss curves, dark->light.
_ECHO_SNR_DB = (20.0, 30.0, 40.0)
_ECHO_HUES = ("#9ec5f4", "#3987e5", "#184f95")  # blue sequential ramp, light->dark


def build_figure():
    import matplotlib.pyplot as plt

    S.apply_style()
    report = run_motor_demo()
    mech = report.mechanical_displacement
    base_acq = AcquisitionParams.demo_motor()
    demo = report.detection

    fig = plt.figure(figsize=(13.5, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], left=0.065, right=0.985,
                          top=0.80, bottom=0.14, wspace=0.22)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    S.suptitle(
        fig,
        "Acoustic SNR budget  |  motor cortex (M1)",
        "Source dz held fixed; conventional phase-sensitive ultrafast ultrasound "
        "through the skull.",
    )

    # --- Panel 1: SNR vs two-way skull loss, one curve per echo-SNR level -----------
    oneway = np.linspace(0.0, 16.0, 65)
    twoway = 2.0 * oneway
    for snr_db, hue in zip(_ECHO_SNR_DB, _ECHO_HUES, strict=True):
        acq = replace(base_acq, echo_snr_linear=10.0 ** (snr_db / 10.0))
        curve = sweep_axis(mech, acq, "skull_loss_db_oneway", oneway)
        ax0.plot(twoway, curve.snr_db, lw=2.2, color=hue)
        # direct label at the right end of each curve
        ax0.text(twoway[-1] + 0.3, curve.snr_db[-1], f"{snr_db:.0f} dB",
                 va="center", ha="left", fontsize=8.5, color=hue, fontweight="bold")

    ax0.axhline(0.0, color=S.INK_2, lw=1.2, ls="--")
    ax0.text(0.5, 1.5, "detection wall  (signal = floor)", fontsize=8, color=S.INK_2)

    # The live demo operating point.
    ax0.plot([2.0 * base_acq.skull_loss_db_oneway], [demo.snr_db], "*",
             color=S.ACCENT, ms=18, zorder=6, markeredgecolor=S.SURFACE,
             markeredgewidth=1.2)
    ax0.annotate(
        f"honest demo\n{demo.snr_db:.1f} dB  ({demo.limiting_denominator})",
        xy=(2.0 * base_acq.skull_loss_db_oneway, demo.snr_db),
        xytext=(2.0 * base_acq.skull_loss_db_oneway - 15.0, demo.snr_db + 14.0),
        fontsize=8.5, color=S.ACCENT, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=S.ACCENT))

    ax0.text(twoway[-1] + 0.3, -26, "echo\nSNR", fontsize=7.5, color=S.MUTED,
             ha="left", va="center")
    ax0.set_xlabel("skull two-way loss  (dB)")
    ax0.set_ylabel("detectability SNR  (dB)")
    ax0.set_title("1. The wall: SNR vs skull loss")
    ax0.set_xlim(0, 34)
    S.tidy(ax0, xgrid=False)

    # --- Panel 2: acoustic axis ranking (dB swing over the literature span) ---------
    ranking = acoustic_axis_ranking(mech, base_acq)
    # role colours: skull loss = floor-red (the wall), the rest recede to blues/muted
    axis_colour = {
        "skull_loss_db_oneway": S.FLOOR,
        "echo_snr_linear": S.DIRECT,
        "epoch_s": S.OSMOTIC,
        "frame_rate_hz": S.MUTED,
    }
    pretty = {
        "skull_loss_db_oneway": "skull loss",
        "echo_snr_linear": "echo SNR",
        "epoch_s": "epoch length",
        "frame_rate_hz": "frame rate",
    }
    names = [pretty.get(a.axis_name, a.axis_name) for a in ranking]
    spans = [a.db_span for a in ranking]
    cols = [axis_colour.get(a.axis_name, S.MUTED) for a in ranking]
    y = np.arange(len(names))[::-1]
    ax1.barh(y, spans, color=cols, height=0.6, zorder=3)
    for yi, a in zip(y, ranking, strict=True):
        ax1.text(a.db_span + 0.6, yi, f"{a.snr_db_lo:+.0f} -> {a.snr_db_hi:+.0f} dB",
                 va="center", fontsize=8.5, color=S.INK_2)
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=9.5, color=S.INK_2)
    ax1.set_xlabel("verdict swing over literature span  (dB)")
    ax1.set_xlim(0, max(spans) * 1.35)
    ax1.set_title("2. Which axis carries the verdict")
    S.tidy(ax1, ygrid=False)
    S.caption(ax1, "skull loss dominates (squared, exponential-in-dB);\n"
                   "epoch is weak -- the coherence window caps it.",
              loc="lower right")

    S.footer(fig, "Recomputed live from run_motor_demo() + sweep_axis / "
                  "acoustic_axis_ranking. Demo skull two-way loss "
                  f"{2 * base_acq.skull_loss_db_oneway:.0f} dB.")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("plots/budget.png"))
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
