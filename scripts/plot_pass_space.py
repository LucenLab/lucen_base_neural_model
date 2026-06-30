"""Render the (eta, s) measurable pass-space as one explainable figure.

This is the visual companion to ``tests/test_pass_space.py``: it recomputes every
value live from the Module-1 chain (never hardcoded), so the picture cannot drift
from the model. Three panels tell the whole story:

1. **The (eta, s) plane** -- a heatmap of the coherent axial swelling as a multiple of
   the detectability floor, with the three-gate PASS region outlined and the crossover
   frontier ``s*(eta)`` drawn on top. This is the measurable space itself.
2. **The crossover frontier ``s*(eta)``** -- the minimum synchrony needed to pass at
   each dilatation fraction. The boundary line of panel 1, read precisely.
3. **The four confidence tiers** -- pessimistic -> central -> optimistic -> unarguable,
   as margins above the floor on a log axis, against the 1 nm bar.

Run::

    uv sync --group viz                 # one-time: install matplotlib
    uv run python scripts/plot_pass_space.py            # writes pass_space.png
    uv run python scripts/plot_pass_space.py --show     # also open a window
    uv run python scripts/plot_pass_space.py -o out.png # custom output path
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from base_neural_model import (
    displacement_sweep,
    get_single_neuron_displacement,
    mechanical_displacement,
    passes_content_survival_gate,
    passes_dilatation_gate,
    passes_stage1_gate,
)
from base_neural_model.base.bands import Band
from base_neural_model.base.provenance import Provenance
from base_neural_model.base.types import (
    MechanicsParams,
    NeuronDisplacement,
    VoxelGeometry,
)
from base_neural_model.model.gates import DEFAULT_ETA_FLOOR

# --- the shared model context (mirrors tests/test_pass_space.py) -----------------

FLOOR_M = 1e-9            # unaberrated detectability floor (and estimate floor)
LOW_JITTER_S = 0.2e-3    # low jitter so Gate 2 is alive across the plane

_D1 = get_single_neuron_displacement()
_VOXEL = VoxelGeometry(
    extent_axial_m=3e-4, extent_lateral_m=1e-3, neuron_count=10_000, depth_m=2e-2
)


def _params_at(eta: float) -> MechanicsParams:
    """Central-column params with the dilatation fraction set to ``eta``."""
    return replace(
        MechanicsParams.central(),
        membrane_disp_m=_D1.value_m,
        dilatation_eta=eta,
        jitter_sigma_s=LOW_JITTER_S,
    )


def _all_gates(sweep, *, floor=FLOOR_M, est=FLOOR_M, reach=2.0) -> bool:
    """The three-gate verdict: amplitude AND content survival AND dilatation."""
    return (
        passes_stage1_gate(sweep, unaberrated_floor_m=floor, reach_orders=reach)
        and passes_content_survival_gate(sweep, estimate_floor_m=est)
        and passes_dilatation_gate(sweep)
    )


def _cell_passes(eta: float, s: float) -> bool:
    """Three-gate verdict at a single (eta, s) point."""
    return _all_gates(displacement_sweep(_D1, _VOXEL, _params_at(eta), np.array([s])))


def _margin(eta: float, s: float) -> float:
    """Coherent axial swelling at (eta, s), as a multiple of the floor."""
    out = mechanical_displacement(_D1, _VOXEL, _params_at(eta), s)
    return out.axial_displacement_m / FLOOR_M


def _crossover_s(eta: float, s_fine: np.ndarray) -> float | None:
    """Smallest s passing all three gates at this eta (None if none does)."""
    for s in s_fine:
        if _cell_passes(eta, float(s)):
            return float(s)
    return None


# --- the four confidence tiers (section-5 envelope columns + an unarguable corner)

_TIER_VOXEL = VoxelGeometry(
    extent_axial_m=300e-6, extent_lateral_m=1e-3, neuron_count=20_000, depth_m=2e-2
)
# (label, params, synchrony, colour)
_TIERS = (
    ("pessimistic",
     MechanicsParams(1.0e-9, 10e-6, 0.10, 1.0 / 3.0, 0.1, 3.0e-3, 100.0), 0.1, "#c0392b"),
    ("central",
     MechanicsParams(1.5e-9, 8e-6, 0.15, 0.5, 0.5, 1.0e-3, 100.0), 0.4, "#e67e22"),
    ("optimistic",
     MechanicsParams(3.0e-9, 6e-6, 0.25, 1.0, 1.0, 0.5e-3, 100.0), 0.8, "#27ae60"),
    ("unarguable",
     MechanicsParams(3.0e-9, 6e-6, 0.25, 1.0, 1.0, 0.3e-3, 100.0), 1.0, "#16a085"),
)


def _tier_margin(params: MechanicsParams, s: float) -> float:
    prov = Provenance(source="plot tier", assumptions=("x",), band=Band.CONTENT_FAST)
    neuron = NeuronDisplacement(
        value_m=params.membrane_disp_m, band=Band.CONTENT_FAST, provenance=prov
    )
    out = mechanical_displacement(neuron, _TIER_VOXEL, params, s)
    return out.axial_displacement_m / FLOOR_M


# --- the figure ------------------------------------------------------------------


def build_figure():
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    n = 121
    eta_grid = np.linspace(0.0, 1.0, n)
    s_grid = np.linspace(0.0, 1.0, n)
    SS, EE = np.meshgrid(s_grid, eta_grid)  # SS[i,j]=s, EE[i,j]=eta

    margin = np.vectorize(_margin)(EE, SS)
    passes = np.vectorize(_cell_passes)(EE, SS)

    s_fine = np.linspace(0.0, 1.0, 401)
    cross_eta = np.linspace(0.0, 1.0, 81)
    cross_s = np.array([_crossover_s(e, s_fine) for e in cross_eta], dtype=float)

    fig = plt.figure(figsize=(15, 5.2), constrained_layout=True)
    fig.suptitle(
        "The measurable (eta, s) space for ultrasonic neural-voxel swelling  "
        "(Module 1, three-gate verdict, floor = 1 nm)",
        fontsize=13, fontweight="bold",
    )
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    # --- Panel 1: the (eta, s) plane -------------------------------------------
    # Heatmap of swelling margin (log colour), PASS boundary, frontier, tier points.
    floor_clip = np.clip(margin, 1e-3, None)
    im = ax0.pcolormesh(
        s_grid, eta_grid, floor_clip,
        norm=LogNorm(vmin=1e-2, vmax=max(1e2, floor_clip.max())),
        cmap="viridis", shading="auto",
    )
    cbar = fig.colorbar(im, ax=ax0, pad=0.02)
    cbar.set_label("swelling  Delta z / floor  (x, log)")

    # The PASS region outline (where all three gates pass). This white contour *is*
    # the crossover frontier s*(eta) -- the boundary panel 2 plots precisely -- so we
    # draw it once here and label it rather than overlaying a redundant curve.
    valid = np.isfinite(cross_s)
    ax0.contour(
        s_grid, eta_grid, passes.astype(float), levels=[0.5],
        colors="white", linewidths=2.0,
    )
    # Proxy handle so the white boundary appears in the legend (contour sets carry
    # no legend label of their own in current matplotlib).
    ax0.plot([], [], color="white", lw=2.0, label="PASS boundary = s*(eta)")
    # The Gate-3 eta floor: nothing passes below it.
    ax0.axhline(DEFAULT_ETA_FLOOR, color="red", ls=":", lw=1.4)
    ax0.text(
        0.5, DEFAULT_ETA_FLOOR + 0.012, f"Gate-3 eta floor = {DEFAULT_ETA_FLOOR}",
        color="red", fontsize=8, ha="center", va="bottom",
    )
    ax0.text(
        0.97, 0.95, "PASS\n(all 3 gates)", color="white", fontsize=10,
        ha="right", va="top", fontweight="bold",
    )
    ax0.text(0.03, 0.30, "FAIL", color="white", fontsize=10, ha="left", va="top")
    ax0.set_xlabel("synchrony  s")
    ax0.set_ylabel("dilatation fraction  eta")
    ax0.set_title("1. The pass-region: more of either knob never hurts")
    ax0.legend(loc="lower left", fontsize=8, framealpha=0.85)

    # --- Panel 2: the crossover frontier s*(eta) -------------------------------
    ax1.plot(cross_eta[valid], cross_s[valid], "o-", color="#2c3e50", lw=2, ms=4)
    ax1.fill_between(cross_eta[valid], cross_s[valid], 1.0, color="#a3d9a5", alpha=0.6)
    ax1.fill_between(cross_eta[valid], 0.0, cross_s[valid], color="#f5b7b1", alpha=0.6)
    ax1.axvline(DEFAULT_ETA_FLOOR, color="red", ls=":", lw=1.4)
    ax1.text(0.55, 0.82, "PASS", color="#1e8449", fontsize=11, fontweight="bold")
    ax1.text(0.12, 0.10, "FAIL", color="#922b21", fontsize=11, fontweight="bold")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("dilatation fraction  eta")
    ax1.set_ylabel("minimum synchrony  s*  to pass")
    ax1.set_title("2. Crossover frontier: more dilatation buys lower synchrony")
    ax1.grid(alpha=0.3)

    # --- Panel 3: the four confidence tiers ------------------------------------
    labels, margins, colours = [], [], []
    for name, params, s, colour in _TIERS:
        labels.append(name)
        margins.append(_tier_margin(params, s))
        colours.append(colour)
    y = np.arange(len(labels))
    ax2.barh(y, margins, color=colours, height=0.6)
    ax2.axvline(1.0, color="black", ls="--", lw=1.4)
    ax2.text(
        1.25, len(labels) - 0.5, "detectability\nfloor (1x)",
        fontsize=8, va="bottom", ha="left",
    )
    ax2.set_xscale("log")
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("swelling margin above floor  (x, log)")
    ax2.set_title("3. Confidence tiers: 0.03x -> 113x the floor")
    for yi, m in zip(y, margins, strict=True):
        ax2.text(m * 1.15, yi, f"{m:.2f}x", va="center", fontsize=9)
    ax2.set_xlim(1e-2, 1e3)
    ax2.invert_yaxis()
    ax2.grid(alpha=0.3, axis="x")

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("plots/pass_space.png"),
        help="output image path (default: pass_space.png)",
    )
    parser.add_argument("--show", action="store_true", help="also open a window")
    parser.add_argument("--dpi", type=int, default=150, help="raster DPI")
    args = parser.parse_args()

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")  # headless render
    import matplotlib.pyplot as plt

    fig = build_figure()
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.output.resolve()}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
