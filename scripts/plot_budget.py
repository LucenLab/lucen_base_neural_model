"""Visualize the acoustic SNR budget -- the Gate-B 'where's the wall' figure.

The detection layer turns the source displacement into a through-skull detectability
SNR; this sweep turns that single point into a curve. Panel 1 sweeps the **skull
two-way loss** at a few echo-SNR levels and marks the 0 dB wall (where the
integration-amplified signal equals the through-skull floor) and the demo operating
point. Panel 2 ranks the four acquisition axes by how much each moves the verdict over
its literature span -- the acoustic-side 'which axis carries the verdict' complement to
the source-side Sobol collapse (eta, s); it leads with skull loss and epoch.

Every number is read back from the live detection/budget chain so the figure cannot
drift from the model.

Run::

    uv sync --group viz
    uv run python scripts/plot_budget.py             # writes budget.png
    uv run python scripts/plot_budget.py --show
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from base_neural_model import (
    AcquisitionParams,
    acoustic_axis_ranking,
    run_motor_demo,
    sweep_axis,
)

# Echo-SNR levels (free-field, dB) to draw as separate skull-loss curves.
_ECHO_SNR_DB = (20.0, 30.0, 40.0)


def build_figure():
    import matplotlib.pyplot as plt

    # The live motor-demo source displacement and its acquisition.
    report = run_motor_demo()
    mech = report.mechanical_displacement
    base_acq = AcquisitionParams.demo_motor()

    fig = plt.figure(figsize=(13, 5.0), constrained_layout=True)
    fig.suptitle(
        "Acoustic SNR budget   (motor source dz held fixed; conventional "
        "phase-sensitive ultrafast US)",
        fontsize=13, fontweight="bold",
    )
    ax0, ax1 = fig.subplots(1, 2)

    # --- Panel 1: SNR vs two-way skull loss, one curve per echo-SNR level -------
    oneway = np.linspace(0.0, 16.0, 65)
    twoway = 2.0 * oneway
    for snr_db in _ECHO_SNR_DB:
        acq = replace(base_acq, echo_snr_linear=10.0 ** (snr_db / 10.0))
        curve = sweep_axis(mech, acq, "skull_loss_db_oneway", oneway)
        line, = ax0.plot(twoway, curve.snr_db, lw=2,
                         label=f"echo SNR = {snr_db:.0f} dB")
        xc = curve.crossing_value
        if xc is not None:
            ax0.plot([2.0 * xc], [0.0], "o", color=line.get_color(), ms=6, zorder=5)

    ax0.axhline(0.0, color="k", lw=1.2, ls="--", alpha=0.7)
    ax0.text(0.4, 0.8, "detection wall (signal = floor)", fontsize=8, alpha=0.7)

    # The live demo operating point (its echo SNR + skull loss).
    demo = report.detection
    ax0.plot([2.0 * base_acq.skull_loss_db_oneway], [demo.snr_db], "*",
             color="#1f3b73", ms=16, zorder=6, label="demo operating point")
    ax0.annotate(
        f"demo: {demo.snr_db:.1f} dB\n(2-way loss "
        f"{2 * base_acq.skull_loss_db_oneway:.0f} dB, {demo.limiting_denominator})",
        xy=(2.0 * base_acq.skull_loss_db_oneway, demo.snr_db),
        xytext=(2.0 * base_acq.skull_loss_db_oneway - 14.0, demo.snr_db - 10.0),
        fontsize=8, color="#1f3b73", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#1f3b73"),
    )

    ax0.set_xlabel("skull two-way loss  (dB)")
    ax0.set_ylabel("detectability SNR  (dB)")
    ax0.set_title("1. The wall: SNR vs skull loss (per echo-SNR level)")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(alpha=0.3)

    # --- Panel 2: acoustic axis ranking (dB swing over the literature span) -----
    ranking = acoustic_axis_ranking(mech, base_acq)
    names = [a.axis_name for a in ranking]
    spans = [a.db_span for a in ranking]
    colours = ["#c0392b", "#e67e22", "#2980b9", "#16a085"][: len(names)]
    y = np.arange(len(names))[::-1]
    ax1.barh(y, spans, color=colours, alpha=0.85)
    for yi, a in zip(y, ranking, strict=True):
        ax1.text(a.db_span + 0.4, yi,
                 f"{a.snr_db_lo:+.0f} -> {a.snr_db_hi:+.0f} dB",
                 va="center", fontsize=8)
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel("verdict swing over literature span  (dB)")
    ax1.set_title("2. Which axis carries the verdict (acoustic-side collapse)")
    ax1.grid(axis="x", alpha=0.3)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("plots/budget.png"))
    parser.add_argument("--show", action="store_true", help="also open a window")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = build_figure()
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.output.resolve()}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
