"""The transduction chain: how a 0.4 nm membrane motion becomes a ~0.27 nm voxel signal.

This is the figure that answers the intuition "0.4 nm x ~1300 neurons should be huge."
It is not. The naive coherent sum (N x dr ~ 520 nm) counts a piston-stroke the geometry
forbids: radially-swelling cells produce a *volume change*, not a heave, and a volume
change reads as a tiny tissue *strain*. Panel 1 walks the chain factor by factor on a log
axis, so every order of magnitude is visibly accounted for. Panel 2 places the surviving
signal against the through-skull detection floor -- the ~60 dB gap the whole model exists
to establish honestly.

Every number is read from the live motor-demo chain, so the figure cannot drift.

Run::

    uv sync --group viz
    uv run python scripts/plot_transduction_chain.py     # writes transduction_chain.png
    uv run python scripts/plot_transduction_chain.py --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _style as S
import numpy as np

from base_neural_model.base.types import MechanicsParams, VoxelGeometry
from base_neural_model.mechanics.neuron_constants import get_single_neuron_displacement
from base_neural_model.model import run_motor_demo


def _chain_stages():
    """The ordered (label, value_m, note) cascade, read from the live model."""
    report = run_motor_demo()
    md = report.mechanical_displacement
    p = MechanicsParams.motor_cortex()
    geom = VoxelGeometry.motor_cortex_layer5()

    dr = get_single_neuron_displacement().value_m  # the cited whole-cell Delta r
    r = p.cell_radius_m
    n = geom.neuron_count
    f_cell = p.cell_volume_fraction
    s = md.synchrony_fraction
    transfer = md.source_transfer_factor
    eta, kappa, L = p.dilatation_eta, p.confinement_kappa, geom.extent_axial_m

    frac_vol = 3.0 * dr / r                       # 3 dr / r
    naive = n * dr                                # the (wrong) coherent-piston sum
    # Displacement-scale cascade: express each strain stage x the gate length so the
    # units stay in metres and the bars are directly comparable to dr and the floor.
    eps_iso = f_cell * s * frac_vol               # coherent volumetric strain
    dz_before_transfer = eta * kappa * L * eps_iso
    dz_after_transfer = dz_before_transfer * transfer
    total = md.axial_displacement_m               # + directional term, honest factors

    stages = [
        ("naive N x dr\n(coherent piston -- forbidden)", naive, S.MUTED,
         f"N = {n} cells x {dr * 1e9:.1f} nm"),
        ("one cell's dr\n(the cited constant)", dr, S.INK_2,
         "whole-cell mammalian AP swelling"),
        ("x 3dr/r  (volume, not heave)", L * frac_vol, S.DIRECT,
         f"3dr/r = {frac_vol * 1e6:.0f} ppm  across the {L * 1e6:.0f} um gate"),
        ("x f_cell . s  (packing + sync)", L * eps_iso, S.DIRECT,
         f"f_cell {f_cell} . s {s:.2f}"),
        ("x eta.kappa  (net axial dilatation)", dz_before_transfer, S.DIRECT,
         f"eta {eta} . kappa {kappa}"),
        ("x honest source factors", dz_after_transfer, S.DIRECT,
         f"viscoelastic . carrier . coherent = {transfer:.2f}"),
        ("+ directional (Betz) term", total, S.VIOLET,
         "monopole + deviatoric shape channel"),
    ]
    return report, stages, total


def build_figure():
    import matplotlib.pyplot as plt

    S.apply_style()
    report, stages, total = _chain_stages()
    floor = report.detection.floor_m
    snr = report.detection.snr_db

    fig = plt.figure(figsize=(13.5, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], left=0.06, right=0.985,
                          top=0.80, bottom=0.13, wspace=0.28)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    S.suptitle(
        fig,
        "The transduction chain  |  motor cortex (M1)",
        "0.4 nm per-neuron membrane swelling  ->  ~0.27 nm voxel signal.  "
        "The naive 'N x dr' counts a coherent piston-stroke the geometry forbids.",
    )

    # --- Panel 1: the cascade, log axis so every order of magnitude is visible -------
    labels = [s[0] for s in stages]
    values = np.array([s[1] for s in stages])
    colours = [s[2] for s in stages]
    notes = [s[3] for s in stages]
    y = np.arange(len(stages))[::-1]

    ax0.barh(y, values * 1e9, color=colours, height=0.62, zorder=3)
    ax0.set_xscale("log")
    for yi, v, note in zip(y, values, notes, strict=True):
        ax0.text(v * 1e9 * 1.25, yi, f"{v * 1e9:.3g} nm",
                 va="center", ha="left", fontsize=8.5, fontweight="bold", color=S.INK)
        ax0.text(v * 1e9 * 1.25, yi - 0.32, note,
                 va="center", ha="left", fontsize=7.2, color=S.MUTED)
    ax0.set_yticks(y)
    ax0.set_yticklabels(labels, fontsize=8.5, color=S.INK_2)
    ax0.set_xlabel("axial displacement  (nm, log scale)")
    ax0.set_xlim(1e-2, 2e3)
    S.tidy(ax0, ygrid=False)

    # Mark the floor as a vertical limit line so the cascade lands relative to it.
    ax0.axvline(floor * 1e9, color=S.FLOOR, lw=1.6, ls="--", zorder=2)
    ax0.text(floor * 1e9 * 1.1, 1.5, f"detection\nfloor {floor * 1e9:.0f} nm",
             color=S.FLOOR, fontsize=8, fontweight="bold", va="center", ha="left")

    # --- Panel 2: signal vs floor -- the gap, as two bars + the dB read -------------
    xs = [0, 1]
    ax1.bar(0, total * 1e9, width=0.6, color=S.DIRECT, zorder=3)
    ax1.bar(1, floor * 1e9, width=0.6, color=S.FLOOR, zorder=3)
    ax1.set_yscale("log")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(["direct beta\nsignal", "through-skull\nfloor"], fontsize=9)
    ax1.set_ylabel("axial displacement  (nm, log scale)")
    ax1.set_ylim(5e-2, 1e3)
    for x, v in [(0, total), (1, floor)]:
        ax1.text(x, v * 1e9 * 1.15, f"{v * 1e9:.0f} nm" if v * 1e9 >= 10
                 else f"{v * 1e9:.2f} nm",
                 ha="center", va="bottom", fontsize=9, fontweight="bold", color=S.INK)
    S.tidy(ax1, xgrid=False)

    # The dB gap annotation, spanning the two bars.
    ytxt = (total * floor) ** 0.5 * 1e9
    ax1.annotate("", xy=(0.5, floor * 1e9 * 0.9), xytext=(0.5, total * 1e9 * 1.1),
                 arrowprops=dict(arrowstyle="<->", color=S.INK_2, lw=1.3))
    ax1.text(0.5, ytxt, f"  {snr:+.1f} dB\n  ~{abs(snr):.0f} dB under",
             ha="left", va="center", fontsize=10, fontweight="bold", color=S.INK)
    ax1.set_title("Signal vs floor")
    S.caption(ax1, "echo-SNR limited: sensing tweaks\ndo not close a ~50 dB physics gap",
              loc="upper right")

    S.footer(fig, "Recomputed live from run_motor_demo(); the M1 preset at the cited "
                  "0.4 nm whole-cell Delta r. Bars are axial displacement.")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path,
                        default=Path("plots/transduction_chain.png"))
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
