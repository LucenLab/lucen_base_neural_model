"""The three co-existing tissue-displacement mechanisms, band-separated (the trade).

The direct neuromechanical beta signal is the *smallest* of the activity-driven mechanical
changes. Two slow envelope mechanisms co-exist and are far larger: osmotic/ECS shrinkage,
and the neurovascular blood-volume (CBV) change behind functional ultrasound.
The load-bearing point this figure makes: the only mechanism that clears the through-skull
floor is vascular -- but it lives in the slow envelope band, which a phase-displacement
readout removes with its ~1 Hz clutter high-pass. So the thing that is detectable is
ordinary fUS (power-Doppler of blood), not the specific fast beta carrier.

Left: each mechanism's axial displacement vs the floor (log), coloured by band. Right: the
same three placed on the content/envelope band axis, showing the 1 Hz high-pass that keeps
the beta carrier and drops the envelope.

Run::

    uv run python scripts/plot_mechanisms.py            # writes mechanisms.png
    uv run python scripts/plot_mechanisms.py --show
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import _style as S
import numpy as np

from base_neural_model.model import run_motor_demo


def build_figure():
    import matplotlib.pyplot as plt

    S.apply_style()
    report = run_motor_demo()
    m = report.mechanisms
    floor = report.detection.floor_m

    # (label, value_m, band, colour, freq_hz, detail)
    mechs = [
        ("direct\nneuromechanical", m.direct_axial_m, "content", S.DIRECT, 20.0,
         "the beta carrier"),
        ("osmotic / ECS", m.osmotic_axial_m, "envelope", S.OSMOTIC, 0.3,
         "K+ clearance, ECS shrink"),
        ("neurovascular / CBV", m.vascular_axial_m, "envelope", S.VASCULAR, 0.1,
         "blood volume = fUS"),
    ]

    fig = plt.figure(figsize=(13.5, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], left=0.07, right=0.985,
                          top=0.80, bottom=0.13, wspace=0.24)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    S.suptitle(
        fig,
        "Three co-existing mechanisms  |  motor cortex (M1)",
        "Only the vascular envelope clears the floor -- but that is functional "
        "ultrasound (fUS), not the fast beta carrier the phase readout targets.",
    )

    # --- Panel 1: displacement vs floor, log, coloured by mechanism -----------------
    y = np.arange(len(mechs))[::-1]
    vals = np.array([mm[1] for mm in mechs])
    cols = [mm[3] for mm in mechs]
    ax0.barh(y, vals * 1e9, color=cols, height=0.6, zorder=3)
    ax0.set_xscale("log")
    for yi, mm in zip(y, mechs, strict=True):
        snr = 20 * math.log10(mm[1] / floor)
        ax0.text(mm[1] * 1e9 * 1.3, yi, f"{mm[1] * 1e9:.3g} nm   ({snr:+.0f} dB)",
                 va="center", ha="left", fontsize=8.5, fontweight="bold", color=S.INK)
        ax0.text(mm[1] * 1e9 * 1.3, yi - 0.3, mm[5], va="center", ha="left",
                 fontsize=7.3, color=S.MUTED)
    ax0.set_yticks(y)
    ax0.set_yticklabels([mm[0] for mm in mechs], fontsize=9, color=S.INK_2)
    ax0.set_xlabel("axial displacement  (nm, log scale)")
    ax0.set_xlim(1e-2, 1e4)
    ax0.axvline(floor * 1e9, color=S.FLOOR, lw=1.6, ls="--", zorder=2)
    ax0.text(floor * 1e9 * 1.1, 2.35, f"through-skull\nfloor {floor * 1e9:.0f} nm",
             color=S.FLOOR, fontsize=8, fontweight="bold", va="center", ha="left")
    ax0.set_title("1. Displacement vs the detection floor")
    S.tidy(ax0, ygrid=False)

    # --- Panel 2: the band axis + the 1 Hz clutter high-pass ------------------------
    hp = 1.0  # clutter high-pass, Hz
    ax1.set_xscale("log")
    ax1.set_xlim(0.03, 100)
    ax1.set_ylim(-0.7, len(mechs) + 0.15)

    # Shade the removed (envelope) and kept (content) sides of the high-pass.
    ax1.axvspan(0.03, hp, color=S.FLOOR, alpha=0.06, zorder=0)
    ax1.axvspan(hp, 100, color=S.GOOD, alpha=0.06, zorder=0)
    ax1.axvline(hp, color=S.INK_2, lw=1.4, ls="--", zorder=2)
    ax1.text(hp * 1.15, len(mechs) + 0.02, "1 Hz clutter high-pass",
             fontsize=8.5, color=S.INK_2, fontweight="bold", va="top", rotation=0)
    ax1.text(0.045, -0.42, "REMOVED  (slow envelope ->\nfUS uses power-Doppler here)",
             fontsize=7.8, color=S.FLOOR, va="bottom", fontweight="bold")
    ax1.text(1.4, -0.55, "KEPT (beta carrier)", fontsize=7.8, color=S.GOOD,
             va="bottom", ha="left", fontweight="bold")

    for yi, mm in zip(range(len(mechs))[::-1], mechs, strict=True):
        detectable = mm[1] > floor
        edge = S.INK if detectable else "none"
        ax1.scatter([mm[4]], [yi], s=360, color=mm[3], zorder=4,
                    edgecolors=edge, linewidths=1.6)
        # label to the LEFT for the kept (right-side) marker so it never runs off-panel
        kept = mm[4] >= hp
        note = "survives" if kept else "removed"
        dx = -8 if kept else 8
        ha = "right" if kept else "left"
        ax1.annotate(f"{mm[0].replace(chr(10), ' ')}\n{mm[1] * 1e9:.3g} nm - {note}",
                     xy=(mm[4], yi), xytext=(dx, 16), textcoords="offset points",
                     fontsize=8, color=S.INK_2, va="bottom", ha=ha)
    ax1.set_yticks([])
    ax1.set_xlabel("mechanism frequency  (Hz, log scale)")
    ax1.set_title("2. Which band survives the phase-displacement readout")
    S.tidy(ax1, ygrid=False)
    S.caption(ax1,
              "detectable = displacement > floor (ringed marker).\n"
              "vascular is detectable but sits in the removed band.",
              loc="lower right")

    S.footer(fig, "Recomputed live from run_motor_demo().mechanisms; the direct term "
                  "carries the honest source factors and the fixed deviatoric_eta.")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("plots/mechanisms.png"))
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
