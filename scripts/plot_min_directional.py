"""Visualize the directional inverse solvers: the minimum beta*(s) and Q*(s).

The directional analogue of ``plot_min_eta``. ``min_anisotropy`` and
``min_orientation_coherence`` invert the gates for the directional unknowns - how
elongated the per-cell eigenstrain (``beta``) or how columnar the population
(``Q``) must be to lift the signal across the detection bar. These are bench targets
on cell shape and alignment, not claims about the tissue.

The directional channel has a richer regime structure than eta because it depends on
the beam/column alignment ``mu = (director . beam)``:

* ``mu`` near +-1 (axes along the beam): ``g(Q, mu) > 0`` -> the directional term
  ADDS signal, so a threshold beta*/Q* exists;
* ``|mu| < 1/sqrt(3) ~ 0.577``: ``g(Q, mu) <= 0`` -> the term SUBTRACTS signal, so no
  beta/Q can rescue (``infeasible_direction``).

Two columns (beta* and Q*), each with two panels: the threshold vs synchrony at full
alignment, and the threshold vs ``mu`` at fixed synchrony showing the regime
boundaries. Every value is read back from the live solver, so the figure cannot drift.

Run::

    uv sync --group viz
    uv run python scripts/plot_min_directional.py        # -> plots/min_directional.png
    uv run python scripts/plot_min_directional.py --show
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from base_neural_model import (
    get_single_neuron_displacement,
    min_anisotropy,
    min_orientation_coherence,
)
from base_neural_model.base.types import MechanicsParams, VoxelGeometry

FLOOR_M = 1.5e-9          # a bar the isotropic signal alone does not clear
LOW_JITTER_S = 0.2e-3     # tight timing so survival does not dominate the picture
FIXED_S = 0.7             # synchrony for the mu-sweep panels
GZERO_MU = 1.0 / np.sqrt(3.0)  # the g(Q, mu) = 0 boundary (P2(mu) = 0)

_D1 = get_single_neuron_displacement()
_VOXEL = VoxelGeometry(
    extent_axial_m=3e-4, extent_lateral_m=1e-3, neuron_count=10_000, depth_m=2e-2
)
# A weak source (low eta) so the directional channel is what carries the verdict.
_BASE = replace(
    MechanicsParams.central(),
    membrane_disp_m=_D1.value_m,
    dilatation_eta=0.1,
    jitter_sigma_s=LOW_JITTER_S,
    anisotropy=0.6,            # the Q solver sweeps Q at this beta
    orientation_coherence=0.7,  # the beta solver sweeps beta at this Q
    mean_axis_projection=1.0,
)


def _star_vs_synchrony(solver, s_grid, *, mu=1.0):
    """value*(s) at alignment mu; NaN where infeasible (no value passes)."""
    out = np.full(s_grid.shape, np.nan)
    params = replace(_BASE, mean_axis_projection=mu)
    for i, s in enumerate(s_grid):
        res = solver(_D1, _VOXEL, params, float(s), floor_m=FLOOR_M)
        if res.feasible:
            out[i] = res.value_star
    return out


def _star_vs_mu(solver, mu_grid, *, s=FIXED_S):
    """value*(mu) at synchrony s; NaN where infeasible, with the regime per point."""
    out = np.full(mu_grid.shape, np.nan)
    regimes = []
    for i, mu in enumerate(mu_grid):
        res = solver(_D1, _VOXEL, replace(_BASE, mean_axis_projection=mu), s,
                     floor_m=FLOOR_M)
        regimes.append(res.binding_constraint)
        if res.feasible:
            out[i] = res.value_star
    return out, regimes


def _panel_vs_synchrony(ax, solver, label):
    s_grid = np.linspace(0.0, 1.0, 161)
    star = _star_vs_synchrony(solver, s_grid)
    feasible = ~np.isnan(star)

    if feasible.any():
        ax.plot(s_grid[feasible], star[feasible], color="#1f3b73", lw=2.5,
                label=f"{label}*(s) -- bench bar", zorder=5)
        ax.fill_between(s_grid[feasible], star[feasible], 1.0, color="#a3d9a5",
                        alpha=0.5, label=f"PASS ({label} above bar)")
        ax.fill_between(s_grid[feasible], 0.0, star[feasible], color="#f5b7b1",
                        alpha=0.5, label=f"FAIL ({label} below bar)")
        s_min = float(s_grid[feasible][0])
        if s_min > 0:
            ax.axvspan(0.0, s_min, color="0.85", alpha=0.7)
            ax.text(s_min / 2, 0.5, "infeasible\n(even max\ndirectional)",
                    fontsize=7, color="0.3", ha="center", va="center")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("synchrony  s")
    ax.set_ylabel(f"minimum  {label}*")
    ax.set_title(f"{label}*(s)  (aligned to beam, mu=1)")
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    ax.grid(alpha=0.3)


def _panel_vs_mu(ax, solver, label):
    mu_grid = np.linspace(0.0, 1.0, 161)
    star, regimes = _star_vs_mu(solver, mu_grid)
    feasible = ~np.isnan(star)

    # Shade the g <= 0 band (cells across the beam: directional subtracts signal).
    ax.axvspan(0.0, GZERO_MU, color="#f5b7b1", alpha=0.4)
    ax.text(GZERO_MU / 2, 0.5, "g <= 0\ndirectional\nsubtracts\n(no rescue)",
            fontsize=7, color="#922b21", ha="center", va="center")
    ax.axvline(GZERO_MU, color="#c0392b", ls="--", lw=1.3)
    ax.text(GZERO_MU + 0.01, 0.96, f"g=0 at mu={GZERO_MU:.2f}", color="#c0392b",
            fontsize=7.5, rotation=90, va="top")

    if feasible.any():
        ax.plot(mu_grid[feasible], star[feasible], color="#1f3b73", lw=2.5,
                label=f"{label}*(mu)", zorder=5)
    # Mark where the threshold exists but is infeasible (g>0 yet too weak).
    infeasible_g_pos = np.array(
        [r == "infeasible" for r in regimes]
    ) & (mu_grid > GZERO_MU)
    if infeasible_g_pos.any():
        ax.fill_between(mu_grid, 0, 1, where=infeasible_g_pos, color="#fdebd0",
                        alpha=0.6, label="infeasible (g>0, too weak)")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("beam/column alignment  mu = (director . beam)")
    ax.set_ylabel(f"minimum  {label}*")
    ax.set_title(f"{label}*(mu)  at s={FIXED_S}: alignment decides feasibility")
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    ax.grid(alpha=0.3)


def build_figure():
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    fig.suptitle(
        "Directional inverse design  --  the cell-shape (beta*) and alignment (Q*) "
        f"a {FLOOR_M*1e9:.1f} nm bar requires  (eta = {_BASE.dilatation_eta})",
        fontsize=14, fontweight="bold",
    )
    (a, b), (c, d) = fig.subplots(2, 2)

    _panel_vs_synchrony(a, min_anisotropy, "beta")
    _panel_vs_mu(c, min_anisotropy, "beta")
    _panel_vs_synchrony(b, min_orientation_coherence, "Q")
    _panel_vs_mu(d, min_orientation_coherence, "Q")

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("plots/min_directional.png")
    )
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
