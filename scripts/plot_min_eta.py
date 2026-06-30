"""Visualize the minimum detectable dilatation fraction eta*(s) -- the bench bar.

Built on ``min_detectable_eta`` (base_neural_model.model.min_eta), the inverse that
turns the project's go/no-go into a single falsifiable number: for each synchrony s,
the smallest dilatation fraction eta that clears all three gates. The bench experiment
(direct displacement imaging of driven cortex) measures eta; this curve is the bar it
must beat. Anything below the curve -- or to the left of where the curve even exists --
falsifies the displacement readout and routes the project to the stiffness-modulation
fallback.

Panel 1: eta*(s) with the pass region (above the curve) and the two ways to fail
shaded -- below the curve (dilatation too low) and the infeasible band at low s (the
eta-independent pedestal bar is unmet, so no eta passes). The Gate-3 floor and the
four tiers' eta values are drawn for context.

Panel 2: the bar at each rhythm's REAL operating point -- eta*(s) computed at the
(f_c, sigma_t) the gamma / high-beta / low-beta presets actually settle at, against the
idealized low-jitter reference. The jitter low-pass in force lifts the curve and the
bands fan out, so the bench bar depends on which rhythm is being imaged (high-beta is
the hardest: slowest timing relative to its frequency).

Run::

    uv sync --group viz
    uv run python scripts/plot_min_eta.py            # writes min_eta.png
    uv run python scripts/plot_min_eta.py --show
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from _bands import BANDS, IDEALIZED

from base_neural_model import get_single_neuron_displacement, min_detectable_eta
from base_neural_model.base.types import MechanicsParams, VoxelGeometry
from base_neural_model.model.gates import DEFAULT_ETA_FLOOR

FLOOR_M = 1e-9
LOW_JITTER_S = 0.2e-3

_D1 = get_single_neuron_displacement()
_VOXEL = VoxelGeometry(
    extent_axial_m=3e-4, extent_lateral_m=1e-3, neuron_count=21_000, depth_m=2e-2
)
_PARAMS = replace(
    MechanicsParams.central(), membrane_disp_m=_D1.value_m, jitter_sigma_s=LOW_JITTER_S
)

# The four tiers' dilatation fractions, for context guides (same columns as the other
# figures): pessimistic 0.1 -> central 0.5 -> optimistic/unarguable 1.0.
_TIER_ETAS = (
    ("pessimistic", 0.1, "#c0392b"),
    ("central", 0.5, "#e67e22"),
    ("optimistic", 1.0, "#27ae60"),
)


def _eta_star_curve(
    s_grid: np.ndarray, *, floor_m: float = FLOOR_M,
    f_c_hz: float | None = None, sigma_t_s: float | None = None,
):
    """eta*(s) over the grid; NaN where infeasible (no eta passes).

    With ``f_c_hz`` / ``sigma_t_s`` given, the curve is computed at that real rhythm
    operating point (so the Gate-2 jitter low-pass is in force); otherwise it uses the
    idealized low-jitter params, isolating the amplitude trade.
    """
    params = _PARAMS
    if f_c_hz is not None and sigma_t_s is not None:
        params = replace(params, content_freq_hz=f_c_hz, jitter_sigma_s=sigma_t_s)
    out = np.full(s_grid.shape, np.nan)
    for i, s in enumerate(s_grid):
        res = min_detectable_eta(_D1, _VOXEL, params, float(s), floor_m=floor_m)
        if res.feasible:
            out[i] = res.eta_star
    return out


def build_figure():
    import matplotlib.pyplot as plt

    s_grid = np.linspace(0.0, 1.0, 201)
    eta_star = _eta_star_curve(s_grid)
    feasible = ~np.isnan(eta_star)
    # Left edge of feasibility (smallest s with any passing eta).
    s_feasible_min = float(s_grid[feasible][0]) if feasible.any() else 1.0

    fig = plt.figure(figsize=(13, 5.2), constrained_layout=True)
    fig.suptitle(
        "Minimum detectable dilatation fraction  eta*(s)  -- the Stage-1 bench bar "
        "(floor = 1 nm)",
        fontsize=13, fontweight="bold",
    )
    ax0, ax1 = fig.subplots(1, 2)

    # --- Panel 1: eta*(s), pass region, the two failure modes ------------------
    ax0.plot(s_grid[feasible], eta_star[feasible], color="#1f3b73", lw=2.5,
             label="eta*(s) -- bench bar", zorder=5)

    # PASS region: above the curve, within the feasible synchrony band.
    ax0.fill_between(s_grid[feasible], eta_star[feasible], 1.0,
                     color="#a3d9a5", alpha=0.6, label="PASS (measure eta above bar)")
    # FAIL by dilatation: below the curve.
    ax0.fill_between(s_grid[feasible], 0.0, eta_star[feasible],
                     color="#f5b7b1", alpha=0.6, label="FAIL: eta below bar")
    # Infeasible band: no eta passes (pedestal bar unmet).
    if s_feasible_min > 0:
        ax0.axvspan(0.0, s_feasible_min, color="0.8", alpha=0.7)
        ax0.text(s_feasible_min / 2, 0.5, "infeasible\nat any eta\n(pedestal bar)",
                 fontsize=7.5, color="0.25", ha="center", va="center")

    # Gate-3 floor and tier eta guides.
    ax0.axhline(DEFAULT_ETA_FLOOR, color="red", ls=":", lw=1.3)
    ax0.text(0.98, DEFAULT_ETA_FLOOR + 0.015, f"Gate-3 eta floor = {DEFAULT_ETA_FLOOR}",
             color="red", fontsize=7.5, ha="right", va="bottom")
    for name, eta, colour in _TIER_ETAS:
        ax0.axhline(eta, color=colour, ls="--", lw=1.0, alpha=0.7)
        # Place each label just below its line at mid-x, so it never collides with the
        # title, the infeasible-band text, or the other tier labels at the axis edge.
        ax0.text(0.50, eta - 0.018, f"{name} eta={eta}", color=colour, fontsize=7.5,
                 ha="center", va="top")

    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.set_xlabel("synchrony  s")
    ax0.set_ylabel("dilatation fraction  eta")
    ax0.set_title("1. The bench must measure eta above this curve to pass")
    ax0.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # --- Panel 2: the bar at each rhythm's REAL operating point -----------------
    # The idealized reference (tight jitter, content essentially intact) vs each band's
    # real (f_c, sigma_t): the jitter low-pass in force lifts the bar, and the bands fan
    # out -- the same dilatation that passes in one rhythm can fail in another purely
    # because more content was low-passed away.
    ideal = _eta_star_curve(s_grid, f_c_hz=IDEALIZED.f_c_hz,
                            sigma_t_s=IDEALIZED.sigma_t_s)
    feas_i = ~np.isnan(ideal)
    ax1.plot(s_grid[feas_i], ideal[feas_i], lw=1.8, color="0.4", ls="--",
             label=f"{IDEALIZED.label}", zorder=3)

    for band in BANDS:
        curve = _eta_star_curve(s_grid, f_c_hz=band.f_c_hz, sigma_t_s=band.sigma_t_s)
        feas = ~np.isnan(curve)
        ax1.plot(s_grid[feas], curve[feas], lw=2.4, color=band.colour,
                 label=f"{band.label}: f_c={band.f_c_hz:.0f}Hz, "
                 f"sigma_t={band.sigma_t_s*1e3:.1f}ms", zorder=5)

    ax1.axhline(DEFAULT_ETA_FLOOR, color="red", ls=":", lw=1.2)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("synchrony  s")
    ax1.set_ylabel("minimum detectable eta*")
    ax1.set_title("2. The real jitter low-pass lifts the bar, by band")
    ax1.legend(loc="upper right", fontsize=7.5, title="rhythm operating point")
    ax1.grid(alpha=0.3)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("plots/min_eta.png"))
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
