"""Visualize the inverse problem: how far spatial resolution can be pushed.

Built on ``min_resolution`` (base_neural_model.model.resolution) -- the inverse of the
pass-space question. For each (eta, s) it solves for the smallest voxel (fewest
neurons, at fixed cell density) that still clears all three gates, and reports that as
a neuron count and
an equivalent voxel-edge shrink. The headline question -- "instead of 10k, can we hit
5k (or finer)?" -- is read straight off these surfaces, whose contour ladder runs from
the 10k baseline down to the finest feasible voxel.

Two limits set that finest voxel, and the figure shows both:

* the **gate** limit -- shrinking lowers the coherent signal (~rho) AND raises the
  source's own sqrt(N) noise pedestal (~rho^-1/2), so a finite gate limit exists; and
* the **minimum-density floor** (``DEFAULT_MIN_NEURON_FLOOR``) -- imposed
  unconditionally so the answer stays inside the regime where the model's population
  statistics hold. Wherever the gates would allow a finer voxel, this floor is what
  binds (hatched in panel 1, "density floor" in panels 2-3). The reported limit is the
  binding one of the two: a conservative, buildable resolution, not a gate artifact.

Run::

    uv sync --group viz
    uv run python scripts/plot_resolution.py           # writes resolution.png
    uv run python scripts/plot_resolution.py --show
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from base_neural_model import (
    DEFAULT_MIN_NEURON_FLOOR,
    get_single_neuron_displacement,
    min_resolution,
)
from base_neural_model.base.types import MechanicsParams, VoxelGeometry

FLOOR_M = 1e-9
LOW_JITTER_S = 0.2e-3

_D1 = get_single_neuron_displacement()
_BASE_VOXEL = VoxelGeometry(
    extent_axial_m=3e-4, extent_lateral_m=1e-3, neuron_count=10_000, depth_m=2e-2
)
_N0 = _BASE_VOXEL.neuron_count
# Baseline voxel edge (lateral) in micrometres, for the resolution readout.
_EDGE0_UM = _BASE_VOXEL.extent_lateral_m * 1e6

# Gates colour key for the limiting-gate panel.
_GATE_CODES = {
    "none": 0,
    "gate1_amplitude": 1,
    "gate2_estimate_floor": 2,
    "gate2_pedestal": 3,
    "gate3": 4,
    "density_floor": 5,
}
_GATE_LABELS = [
    "unbounded", "Gate1 ampl.", "Gate2 est-floor", "Gate2 pedestal", "Gate3 eta",
    "density floor",
]


def _params(eta: float) -> MechanicsParams:
    return replace(
        MechanicsParams.central(),
        membrane_disp_m=_D1.value_m,
        dilatation_eta=eta,
        jitter_sigma_s=LOW_JITTER_S,
    )


def _limit(eta: float, s: float, *, estimate_floor_m: float = FLOOR_M):
    return min_resolution(
        _D1, _BASE_VOXEL, _params(eta), s,
        floor_m=FLOOR_M, estimate_floor_m=estimate_floor_m,
    )


def build_figure():
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, LogNorm

    n = 81
    eta_grid = np.linspace(0.0, 1.0, n)
    s_grid = np.linspace(0.05, 1.0, n)  # s=0 never passes; skip the trivial column

    # Surface 1: min neuron count over (eta, s) at the nominal (tied) floor.
    min_n = np.full((n, n), np.nan)
    gate_code = np.full((n, n), np.nan)
    for i, eta in enumerate(eta_grid):
        for j, s in enumerate(s_grid):
            lim = _limit(eta, float(s))
            if lim.feasible:
                min_n[i, j] = lim.min_neuron_count
            gate_code[i, j] = _GATE_CODES[lim.limiting_gate]

    fig = plt.figure(figsize=(16, 5.2), constrained_layout=True)
    fig.suptitle(
        "How far can spatial resolution be pushed?  Minimum neurons (finest voxel) "
        "that still passes all 3 gates,  floor = 1 nm",
        fontsize=13, fontweight="bold",
    )
    ax0, ax1, ax2 = fig.subplots(1, 3)

    # --- Panel 1: min-N over (eta, s) ------------------------------------------
    masked = np.ma.masked_invalid(min_n)
    vmax = np.nanmax(min_n) if np.isfinite(min_n).any() else _N0
    im = ax0.pcolormesh(
        s_grid, eta_grid, masked,
        norm=LogNorm(vmin=max(1, np.nanmin(min_n)), vmax=max(vmax, _N0)),
        cmap="viridis_r", shading="auto",
    )
    ax0.set_facecolor("0.85")  # infeasible (eta below floor) shown grey
    cbar = fig.colorbar(im, ax=ax0, pad=0.02)
    cbar.set_label("min neurons to pass  (log; fewer = finer resolution)")

    # Contour ladder from the 10k baseline down toward the finest feasible voxel.
    # Log-spaced decade lines (the feasible range spans several decades), plus the 5k
    # target and the global minimum N (the finest feasible voxel on the plane) drawn as
    # its own contour -- all read from the surface, nothing hardcoded. With the
    # mandatory minimum-density floor in force, that minimum is the floor itself
    # wherever the gates would allow finer.
    finest_n = int(np.nanmin(min_n))
    decade_levels = [lv for lv in (10, 100, 1_000) if finest_n < lv < _N0]
    levels = sorted({finest_n, 5_000, _N0, *decade_levels})

    # Colour the baseline red, the finest-voxel line cyan, the rest white.
    colours = []
    for lv in levels:
        if lv == _N0:
            colours.append("red")
        elif lv == finest_n:
            colours.append("cyan")
        else:
            colours.append("white")
    cs = ax0.contour(
        s_grid, eta_grid, min_n, levels=levels, colors=colours, linewidths=1.5,
    )

    def _fmt(v: float) -> str:
        if v == _N0:
            return "10k baseline"
        if v == finest_n:
            return f"floor = {finest_n}"
        if v == 5_000:
            return "5k"
        return f"{int(v / 1000)}k" if v >= 1_000 else f"{int(v)}"

    ax0.clabel(cs, fmt={lv: _fmt(lv) for lv in levels}, fontsize=7.5)

    # Hatch the region where the density floor (not a gate) binds: the gates would
    # allow a finer voxel here, so this plateau is the conservatism guardrail at work.
    floor_region = (gate_code == _GATE_CODES["density_floor"]).astype(float)
    ax0.contourf(
        s_grid, eta_grid, floor_region, levels=[0.5, 1.5],
        colors="none", hatches=["//"],
    )
    ax0.contour(
        s_grid, eta_grid, floor_region, levels=[0.5], colors="cyan", linewidths=1.2,
    )

    ax0.text(0.5, 0.02, "grey = infeasible (eta < Gate-3 floor)   //// = density-floor "
             "bound (gates allow finer)", fontsize=6.8, color="0.2",
             ha="center", va="bottom")
    ax0.set_xlabel("synchrony  s")
    ax0.set_ylabel("dilatation fraction  eta")
    ax0.set_title(
        f"1. Finest feasible voxel (density floor = {DEFAULT_MIN_NEURON_FLOOR} cells)"
    )

    # --- Panel 2: which gate bounds the limit ----------------------------------
    # A diagnostic of the underlying gate structure, so the density floor is lifted
    # (min_neuron_floor=1) -- otherwise it would mask everything in the favorable
    # region. The estimate floor is also relaxed 1000x so the pedestal regime (the
    # sqrt(N) noise bar) is visible and not hidden by the estimate-floor bar.
    gate_relaxed = np.full((n, n), np.nan)
    for i, eta in enumerate(eta_grid):
        for j, s in enumerate(s_grid):
            lim = min_resolution(
                _D1, _BASE_VOXEL, _params(eta), float(s),
                floor_m=FLOOR_M, estimate_floor_m=FLOOR_M * 1e-3, min_neuron_floor=1,
            )
            gate_relaxed[i, j] = _GATE_CODES[lim.limiting_gate]

    cmap = plt.get_cmap("Set2", len(_GATE_LABELS))
    norm = BoundaryNorm(np.arange(-0.5, len(_GATE_LABELS) + 0.5), cmap.N)
    im2 = ax1.pcolormesh(s_grid, eta_grid, gate_relaxed, cmap=cmap, norm=norm,
                         shading="auto")
    cbar2 = fig.colorbar(im2, ax=ax1, ticks=range(len(_GATE_LABELS)), pad=0.02)
    cbar2.ax.set_yticklabels(_GATE_LABELS, fontsize=8)
    ax1.set_xlabel("synchrony  s")
    ax1.set_ylabel("dilatation fraction  eta")
    ax1.set_title("2. Which gate would bind (density floor lifted, est. floor relaxed)")

    # --- Panel 3: min-N vs s at a few eta, with 10k/5k guides ------------------
    s_line = np.linspace(0.05, 1.0, 60)
    for eta, colour in ((0.25, "#8e44ad"), (0.5, "#2980b9"), (1.0, "#16a085")):
        counts = []
        for s in s_line:
            lim = _limit(eta, float(s))
            counts.append(lim.min_neuron_count if lim.feasible else np.nan)
        ax2.plot(s_line, counts, lw=2, color=colour, label=f"eta = {eta}")

    ax2.axhline(_N0, color="red", ls="--", lw=1.4)
    ax2.axhline(5_000, color="black", ls=":", lw=1.4)
    ax2.axhline(DEFAULT_MIN_NEURON_FLOOR, color="cyan", ls="-", lw=1.6)
    ax2.text(0.06, _N0 * 1.05, "10k baseline", color="red", fontsize=8)
    ax2.text(0.06, 5_000 * 1.05, "5k target", color="black", fontsize=8)
    ax2.text(0.06, DEFAULT_MIN_NEURON_FLOOR * 1.05,
             f"density floor ({DEFAULT_MIN_NEURON_FLOOR})", color="#0aa", fontsize=8)
    ax2.set_yscale("log")
    ax2.set_xlabel("synchrony  s")
    ax2.set_ylabel("min neurons to pass  (log)")
    ax2.set_title("3. Curves bottom out at the density floor (cyan), not at zero")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(alpha=0.3, which="both")

    # Twin axis: translate neuron count -> voxel edge (um) at fixed density.
    # edge = edge0 * (N / N0)^(1/3).
    ax2b = ax2.twinx()
    ax2b.set_yscale("log")
    ax2b.set_ylim(*ax2.get_ylim())
    yt = np.array([100, 1_000, 5_000, 10_000])
    yt = yt[(yt >= ax2.get_ylim()[0]) & (yt <= ax2.get_ylim()[1])]
    ax2b.set_yticks(yt)
    ax2b.set_yticklabels([f"{_EDGE0_UM * (v / _N0) ** (1 / 3):.0f}" for v in yt],
                         fontsize=8)
    ax2b.set_ylabel("equiv. voxel edge (um)", fontsize=9)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("plots/resolution.png"))
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
