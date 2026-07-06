"""The M1 headline: neural activity -> NeuralState -> predicted nanometre displacement.

One consolidated view of a steady motor-cortex run, following the pipeline end to end:

1. **Activity** -- the E/I beta limit cycle and the synchrony r(t) it settles at.
2. **The bridge** -- the reduced NeuralState (s, f_c, sigma_t, Q, rate) as a readable
   card, and the jitter low-pass exp(-2 pi^2 f^2 sigma^2) with this run's content corner
   marked, so the surviving fraction is read off the curve.
3. **Decomposition** -- the predicted axial Delta z split into its isotropic (monopole)
   and directional (deviatoric Betz) shares, with the incoherent pedestal for scale.

Uses run_motor_cortex() (steady resting beta) so the rhythm is clean; the honest Gate-A
verdict with its bursty in-burst scoring lives in plot_budget / plot_transduction_chain.

Run::

    uv run python scripts/plot_neural_summary.py            # writes neural_summary.png
    uv run python scripts/plot_neural_summary.py --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _style as S
import numpy as np

from base_neural_model.model import run_motor_cortex


def build_figure():
    import matplotlib.pyplot as plt

    S.apply_style()
    report = run_motor_cortex()
    a = report.activity
    ns = report.neural_state
    md = report.mechanical_displacement

    fig = plt.figure(figsize=(14, 7.6))
    gs = fig.add_gridspec(
        2, 3, left=0.055, right=0.985, top=0.82, bottom=0.09,
        height_ratios=[1.0, 1.0], width_ratios=[1.25, 1.0, 1.0],
        hspace=0.42, wspace=0.30,
    )
    ax_rhythm = fig.add_subplot(gs[0, 0])
    ax_sync = fig.add_subplot(gs[1, 0], sharex=ax_rhythm)
    ax_card = fig.add_subplot(gs[0, 1])
    ax_surv = fig.add_subplot(gs[1, 1])
    ax_dz = fig.add_subplot(gs[:, 2])

    S.suptitle(
        fig,
        "Neural model summary  |  motor cortex (M1)",
        "activity  ->  NeuralState  ->  predicted axial displacement.  "
        f"Steady resting beta at {ns.content_freq_hz:.0f} Hz.",
    )

    # --- 1a. The E/I beta rhythm (show a 0.6 s window so the cycles are legible) -----
    t = a.t_s
    win = (t >= 0.4) & (t <= 1.0)
    ax_rhythm.plot(t[win], a.e[win], color=S.DIRECT, lw=1.6, label="E (excitatory)")
    ax_rhythm.plot(t[win], a.i[win], color=S.OSMOTIC, lw=1.4, label="I (inhibitory)")
    ax_rhythm.set_ylabel("population activity")
    ax_rhythm.set_title("1. The E/I beta limit cycle")
    ax_rhythm.legend(loc="upper right", ncol=2)
    ax_rhythm.tick_params(labelbottom=False)
    S.tidy(ax_rhythm, xgrid=False)

    # --- 1b. Synchrony r(t), the Kuramoto order parameter --------------------------
    ax_sync.plot(t[win], a.r[win], color=S.VIOLET, lw=1.8)
    ax_sync.axhline(ns.synchrony_fraction, color=S.INK_2, lw=1.0, ls="--")
    ax_sync.text(t[win][-1], ns.synchrony_fraction, f" s = {ns.synchrony_fraction:.2f}",
                 va="center", ha="left", fontsize=8.5, color=S.INK_2, fontweight="bold")
    ax_sync.set_ylim(0, 1.05)
    ax_sync.set_xlabel("time  (s)")
    ax_sync.set_ylabel("synchrony  r(t)")
    ax_sync.set_title("Population synchrony (Kuramoto r)")
    S.tidy(ax_sync, xgrid=False)

    # --- 2a. The NeuralState bridge card -------------------------------------------
    ax_card.axis("off")
    ax_card.set_title("2. The reduced NeuralState")
    rows = [
        ("synchrony  s", f"{ns.synchrony_fraction:.2f}", "temporal coherence"),
        ("content corner  f_c", f"{ns.content_freq_hz:.0f} Hz", "beta rhythm"),
        ("jitter  sigma_t", f"{ns.jitter_sigma_s * 1e3:.1f} ms", "firing-time spread"),
        ("orientation  Q", f"{ns.orientation_coherence:.2f}", "columnar alignment"),
        ("mean rate", f"{ns.mean_firing_rate_hz:.1f} Hz", "population firing"),
    ]
    y0 = 0.92
    for i, (name, val, note) in enumerate(rows):
        y = y0 - i * 0.185
        ax_card.text(0.02, y, name, fontsize=9.5, color=S.INK_2, va="center")
        ax_card.text(0.62, y, val, fontsize=12, color=S.INK, fontweight="bold",
                     va="center", ha="left")
        ax_card.text(0.62, y - 0.058, note, fontsize=7.3, color=S.MUTED,
                     va="center", ha="left")
        if i < len(rows):
            ax_card.axhline(y - 0.093, xmin=0.02, xmax=0.98, color=S.GRID, lw=0.8)

    # --- 2b. Jitter survival curve --------------------------------------------------
    f = np.linspace(0, 60, 400)
    survival = np.exp(-2.0 * np.pi**2 * f**2 * ns.jitter_sigma_s**2)
    ax_surv.plot(f, survival, color=S.DIRECT, lw=2)
    fc, sv = ns.content_freq_hz, md.content_band_survival
    ax_surv.axvline(fc, color=S.INK_2, lw=1.0, ls="--")
    ax_surv.plot([fc], [sv], "o", color=S.DIRECT, ms=8, zorder=5,
                 markeredgecolor=S.SURFACE, markeredgewidth=1.5)
    ax_surv.annotate(f"f_c = {fc:.0f} Hz\nsurvival {sv:.2f}",
                     xy=(fc, sv), xytext=(fc + 9, sv - 0.22),
                     fontsize=8.5, color=S.INK, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=S.INK_2))
    ax_surv.set_xlim(0, 60)
    ax_surv.set_ylim(0, 1.05)
    ax_surv.set_xlabel("frequency  (Hz)")
    ax_surv.set_ylabel("content-band survival")
    ax_surv.set_title("Jitter low-pass  exp(-2π²f²σ²)")
    S.tidy(ax_surv, xgrid=False)

    # --- 3. The Delta z decomposition (stacked) -------------------------------------
    iso = md.isotropic_axial_m * 1e9
    dirc = md.directional_axial_m * 1e9
    ped = md.incoherent_pedestal_m * 1e9
    total = md.axial_displacement_m * 1e9

    ax_dz.bar(0, iso, width=0.62, color=S.ISO, label="isotropic (monopole)", zorder=3)
    ax_dz.bar(0, dirc, bottom=iso + 0.003, width=0.62, color=S.DIR,
              label="directional (Betz)", zorder=3)
    ax_dz.bar(1, ped, width=0.62, color=S.PEDESTAL, label="incoherent pedestal", zorder=3)
    ax_dz.set_xticks([0, 1])
    ax_dz.set_xticklabels(["coherent\nsignal", "noise\npedestal"], fontsize=9)
    ax_dz.set_ylabel("axial displacement  (nm)")
    ax_dz.set_title("3. Predicted Delta z decomposition")
    ax_dz.legend(loc="upper right", fontsize=8)

    # direct labels on the stack
    ax_dz.text(0, iso / 2, f"{iso:.3f}", ha="center", va="center",
               fontsize=8.5, color=S.SURFACE, fontweight="bold")
    ax_dz.text(0, iso + dirc / 2, f"{dirc:.3f}", ha="center", va="center",
               fontsize=8.5, color=S.SURFACE, fontweight="bold")
    ax_dz.text(0, total + 0.012, f"total {total:.3f} nm", ha="center", va="bottom",
               fontsize=9.5, color=S.INK, fontweight="bold")
    ax_dz.text(1, ped + 0.012, f"{ped:.3f} nm", ha="center", va="bottom",
               fontsize=8.5, color=S.INK_2)
    ax_dz.set_ylim(0, total * 1.28)
    S.tidy(ax_dz, xgrid=False)

    S.footer(fig, "Recomputed live from run_motor_cortex() (steady resting beta). The "
                  "directional term uses deviatoric_eta (shape change, undrained), not "
                  "the volumetric dilatation eta.")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path,
                        default=Path("plots/neural_summary.png"))
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
