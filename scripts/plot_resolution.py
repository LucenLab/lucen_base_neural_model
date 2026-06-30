"""Visualize the inverse problem: how far spatial resolution can be pushed.

Built on ``min_resolution`` (base_neural_model.model.resolution) -- the inverse of the
pass-space question. For each (eta, s) it solves for the smallest voxel (fewest
neurons, at fixed cell density) that still clears all three gates, and reports that as
a neuron count and an equivalent voxel-edge shrink. The headline question -- "can we
shrink the voxel to capture fewer neurons but resolve finer detail, and what stops us?"
-- is read straight off these surfaces, whose contour ladder runs from the 21k baseline
down to the finest feasible voxel.

Two CELLULAR limits set that finest voxel (panels 1-2):

* the **gate** limit -- shrinking lowers the coherent signal (~rho) AND raises the
  source's own sqrt(N) noise pedestal (~rho^-1/2), so a finite gate limit exists; and
* the **minimum-density floor** (``DEFAULT_MIN_NEURON_FLOOR``) -- imposed
  unconditionally so the answer stays inside the regime where the model's population
  statistics hold. Wherever the gates would allow a finer voxel, this floor is what
  binds (hatched in panel 1, "density floor" in panels 2-3). The reported limit is the
  binding one of the two: a conservative, buildable resolution, not a gate artifact.

Panel 3 then sets the cellular shrink against the **acoustic** diffraction limit
(lambda/2, a Module-2 constraint), split by rhythm band. The interesting finding: over
much of the favorable regime the cellular signal would permit a voxel FINER than
ultrasound can resolve, so the true resolution ceiling there is acoustic, not cellular
-- shrinking to capture fewer neurons buys real resolution only down to lambda/2.

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
from _bands import BANDS

from base_neural_model import (
    DEFAULT_MIN_NEURON_FLOOR,
    get_single_neuron_displacement,
    min_resolution,
)
from base_neural_model.base.types import MechanicsParams, VoxelGeometry

FLOOR_M = 1e-9
LOW_JITTER_S = 0.2e-3

# Acoustic diffraction limit: the finest voxel ULTRASOUND can resolve is ~lambda/2 =
# c / (2 f). Soft-tissue speed of sound ~1540 m/s. This is a Module-2 limit, drawn here
# to show where the *cellular* (Module-1) shrink runs past what the beam can resolve --
# below these lines, the signal would permit a finer voxel than the wavelength allows.
_C_SOUND_M_S = 1540.0
_ACOUSTIC_MHZ = (2.0, 3.0, 5.0)  # representative imaging frequencies


def _acoustic_half_lambda_um(f_mhz: float) -> float:
    """Finest acoustically-resolvable voxel edge (lambda/2) in um at f (MHz)."""
    return (_C_SOUND_M_S / (f_mhz * 1e6)) / 2.0 * 1e6


_D1 = get_single_neuron_displacement()
_BASE_VOXEL = VoxelGeometry(
    extent_axial_m=3e-4, extent_lateral_m=1e-3, neuron_count=21_000, depth_m=2e-2
)
_N0 = _BASE_VOXEL.neuron_count
# Baseline voxel edge (lateral) in micrometres, for the resolution readout.
_EDGE0_UM = _BASE_VOXEL.extent_lateral_m * 1e6


def _edge_um(n: int) -> float:
    """Voxel lateral edge (um) for a neuron count, at fixed density: edge ~ N^(1/3)."""
    return _EDGE0_UM * (n / _N0) ** (1.0 / 3.0)


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


def _params(eta: float, *, band=None) -> MechanicsParams:
    p = replace(
        MechanicsParams.central(),
        membrane_disp_m=_D1.value_m,
        dilatation_eta=eta,
        jitter_sigma_s=LOW_JITTER_S,
    )
    if band is not None:
        p = replace(p, content_freq_hz=band.f_c_hz, jitter_sigma_s=band.sigma_t_s)
    return p


def _limit(eta: float, s: float, *, estimate_floor_m: float = FLOOR_M, band=None):
    return min_resolution(
        _D1, _BASE_VOXEL, _params(eta, band=band), s,
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

    # Contour ladder from the baseline down toward the finest feasible voxel.
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
            return f"{_N0 // 1000}k baseline"
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

    # --- Panel 3: finest voxel EDGE vs s, by band, against the acoustic limit ---
    # The voxel-shrink question made direct: smaller voxel = fewer neurons = finer
    # spatial resolution. We plot the finest feasible voxel *edge* (um) the cellular
    # signal permits, per rhythm band, at a favorable eta=1.0 -- and overlay the
    # ultrasound diffraction limit (lambda/2). Where the cellular curve drops BELOW an
    # acoustic line, the signal would allow a finer voxel than the beam can resolve: the
    # binding constraint flips from cellular (Module 1) to acoustic (Module 2).
    eta_fixed = 1.0
    s_line = np.linspace(0.05, 1.0, 60)
    for band in BANDS:
        edges, gates = [], []
        for s in s_line:
            lim = _limit(eta_fixed, float(s), band=band)
            edges.append(_edge_um(lim.min_neuron_count) if lim.feasible else np.nan)
            gates.append(lim.limiting_gate)
        ax2.plot(s_line, edges, lw=2.4, color=band.colour,
                 label=f"{band.label}: f_c={band.f_c_hz:.0f}Hz")

    # The cellular statistical floor as an edge (the density-floor plateau).
    floor_edge = _edge_um(DEFAULT_MIN_NEURON_FLOOR)
    ax2.axhline(floor_edge, color="0.3", ls="-", lw=1.4)
    ax2.text(0.97, floor_edge * 1.02,
             f"cellular density floor ({DEFAULT_MIN_NEURON_FLOOR} cells, "
             f"{floor_edge:.0f} um)", color="0.2", fontsize=7.5, ha="right", va="bottom")
    # The baseline voxel edge.
    ax2.axhline(_EDGE0_UM, color="red", ls="--", lw=1.2)
    ax2.text(0.05, _EDGE0_UM * 0.96, f"baseline {_EDGE0_UM:.0f} um (N={_N0:,})",
             color="red", fontsize=7.5, va="top")

    # The acoustic diffraction limits (Module 2): below these the beam cannot resolve.
    for f_mhz in _ACOUSTIC_MHZ:
        y = _acoustic_half_lambda_um(f_mhz)
        ax2.axhline(y, color="#2980b9", ls=":", lw=1.3)
        ax2.text(0.05, y * 1.02, f"acoustic limit @ {f_mhz:.0f} MHz "
                 f"(lambda/2={y:.0f} um)", color="#2980b9", fontsize=7, va="bottom")

    ax2.set_yscale("log")
    ax2.set_ylim(floor_edge * 0.8, _EDGE0_UM * 1.3)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("synchrony  s")
    ax2.set_ylabel("finest feasible voxel edge  (um, log)")
    ax2.set_title(f"3. Voxel shrink vs the acoustic limit (eta={eta_fixed})")
    ax2.legend(loc="upper right", fontsize=7.5, title="cellular limit, by band",
               framealpha=0.95)
    ax2.grid(alpha=0.3, which="both")

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
