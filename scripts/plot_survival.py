"""Visualize the content-band survival -- the jitter low-pass Gate 2 applies.

Firing-time jitter acts as a low-pass on the population signal: the survival factor
``survival(sigma_t, f_c) = exp(-2*pi^2*f_c^2*sigma_t^2)`` is the fraction of the
content-band coherent displacement that outlives the jitter. It is 1 at the slow
envelope and falls steeply in the fast content band -- which is *why* content is the
first casualty (the signal can exist yet carry no lexical information). This term is
implicit in the pass-space and resolution figures; here it is the subject.

Every value is read back from the live chain (the ``content_band_survival`` field of
``MechanicalDisplacement``) so the plot cannot drift from the model.

Run::

    uv sync --group viz
    uv run python scripts/plot_survival.py             # writes survival.png
    uv run python scripts/plot_survival.py --show
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from base_neural_model import get_single_neuron_displacement
from base_neural_model.base.types import MechanicsParams, VoxelGeometry
from base_neural_model.mechanics.transduction import mechanical_displacement

_D1 = get_single_neuron_displacement()
_VOXEL = VoxelGeometry(
    extent_axial_m=3e-4, extent_lateral_m=1e-3, neuron_count=10_000, depth_m=2e-2
)

# Content-band corners to draw (Hz). 100 Hz is the model's central content corner.
_CORNERS_HZ = (50.0, 100.0, 200.0)

# The four confidence tiers carry these jitter values (s) -- mark them as the
# operating points where the survival actually lands (same columns as plot_pass_space).
_TIER_JITTER = (
    ("pessimistic", 3.0e-3, "#c0392b"),
    ("central", 1.0e-3, "#e67e22"),
    ("optimistic", 0.5e-3, "#27ae60"),
    ("unarguable", 0.3e-3, "#16a085"),
)


def _survival(sigma_t_s: float, f_c_hz: float) -> float:
    """Survival at (sigma_t, f_c), read back from the live transduction chain."""
    params = replace(
        MechanicsParams.central(),
        membrane_disp_m=_D1.value_m,
        jitter_sigma_s=sigma_t_s,
        content_freq_hz=f_c_hz,
    )
    # Synchrony is irrelevant to the survival factor; any value in [0, 1] works.
    return mechanical_displacement(_D1, _VOXEL, params, 1.0).content_band_survival


def build_figure():
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig = plt.figure(figsize=(13, 5.0), constrained_layout=True)
    fig.suptitle(
        "Content-band survival under firing jitter   "
        "survival = exp(-2 pi^2 f_c^2 sigma_t^2)   (the Gate-2 low-pass)",
        fontsize=13, fontweight="bold",
    )
    ax0, ax1 = fig.subplots(1, 2)

    # --- Panel 1: survival vs jitter sigma_t, one curve per content corner -----
    sigma_ms = np.linspace(0.0, 4.0, 400)
    for f_c in _CORNERS_HZ:
        surv = [_survival(s * 1e-3, f_c) for s in sigma_ms]
        ax0.plot(sigma_ms, surv, lw=2, label=f"f_c = {f_c:.0f} Hz")

    # Mark each tier's jitter as a vertical guide + the survival it reaches at 100 Hz.
    for name, sigma_t, colour in _TIER_JITTER:
        x = sigma_t * 1e3
        y = _survival(sigma_t, 100.0)
        ax0.axvline(x, color=colour, ls=":", lw=1.3, alpha=0.8)
        ax0.plot([x], [y], "o", color=colour, ms=7, zorder=5)
        ax0.annotate(
            f"{name}\n{y:.2f}",
            xy=(x, y), xytext=(x + 0.12, y + 0.04),
            fontsize=8, color=colour, fontweight="bold",
        )

    # Anchor the LIVE operating point: the (sigma_t, survival) a real model run sits
    # at, so the inverse sweep is tied to what the simulation actually produced.
    from base_neural_model.model import run_neural_model

    state = run_neural_model().neural_state
    x_live = state.jitter_sigma_s * 1e3
    y_live = _survival(state.jitter_sigma_s, state.content_freq_hz)
    ax0.plot([x_live], [y_live], "*", color="#1f3b73", ms=16, zorder=6,
             label=f"live run (f_c={state.content_freq_hz:.0f} Hz)")
    ax0.annotate(
        f"{y_live*100:.0f}% survives\nat f_c={state.content_freq_hz:.0f} Hz",
        xy=(x_live, y_live), xytext=(x_live + 0.3, y_live + 0.12),
        fontsize=8.5, color="#1f3b73", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#1f3b73"),
    )

    ax0.set_xlim(0, 4)
    ax0.set_ylim(0, 1.02)
    ax0.set_xlabel("firing-time jitter  sigma_t  (ms)")
    ax0.set_ylabel("content-band survival  (fraction)")
    ax0.set_title("1. Survival vs jitter (tiers at f_c=100 Hz; the live run marked)")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(alpha=0.3)

    # --- Panel 2: (sigma_t, f_c) survival heatmap ------------------------------
    sig = np.linspace(0.05, 4.0, 200) * 1e-3   # avoid the trivial sigma=0 line
    fcs = np.linspace(10.0, 300.0, 200)
    SIG, FC = np.meshgrid(sig, fcs)
    surv = np.vectorize(_survival)(SIG, FC)

    im = ax1.pcolormesh(
        sig * 1e3, fcs, surv,
        norm=LogNorm(vmin=1e-3, vmax=1.0), cmap="magma", shading="auto",
    )
    cbar = fig.colorbar(im, ax=ax1, pad=0.02)
    cbar.set_label("survival (fraction, log)")

    # Iso-survival contours make the low-pass shape legible.
    cs = ax1.contour(
        sig * 1e3, fcs, surv, levels=[0.05, 0.2, 0.5, 0.8],
        colors="white", linewidths=1.0,
    )
    ax1.clabel(cs, fmt="%.2f", fontsize=7)

    # The content band (the target) and each tier's jitter line.
    ax1.axhline(100.0, color="cyan", ls="--", lw=1.2)
    ax1.text(3.6, 108, "content corner  f_c=100 Hz", color="cyan", fontsize=8, ha="right")
    for _name, sigma_t, colour in _TIER_JITTER:
        ax1.axvline(sigma_t * 1e3, color=colour, ls=":", lw=1.1, alpha=0.7)

    ax1.set_xlabel("firing-time jitter  sigma_t  (ms)")
    ax1.set_ylabel("content-band corner  f_c  (Hz)")
    ax1.set_title("2. Faster content + more jitter both crush survival")

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("plots/survival.png"))
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
