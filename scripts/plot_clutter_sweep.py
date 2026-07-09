"""Physiological-clutter sweep -- where residual clutter overtakes the echo-SNR floor.

The honest demo is echo-SNR-limited at a ~232 nm through-skull floor. Physiological
clutter (cardiac/respiratory/vasomotion tissue motion left in the beta content band
after the ~1 Hz SVD clutter high-pass) competes with that floor for the binding
denominator. This sweep varies the residual clutter from 1 nm to 100 um and shows two
regimes: below the echo-SNR floor the verdict is pinned (clutter invisible); above it the
verdict goes clutter-limited and degrades 1:1. The crossover tells you how much residual
clutter the readout could tolerate before biology, not the instrument, becomes the wall.

Run::

    uv sync --group viz
    uv run python scripts/plot_clutter_sweep.py            # writes clutter_sweep.png
    uv run python scripts/plot_clutter_sweep.py --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from base_neural_model.model import run_motor_demo


def build_figure():
    import matplotlib.pyplot as plt

    clutter_m = np.array([1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 2.32e-7, 3e-7,
                          1e-6, 3e-6, 1e-5, 3e-5, 1e-4])
    base = run_motor_demo()
    sig = base.detection.surviving_dz_m
    echo_floor = base.detection.phase_floor_m

    floors, snrs = [], []
    for c in clutter_m:
        d = run_motor_demo(residual_clutter_m=float(c)).detection
        floors.append(d.floor_m)
        snrs.append(d.snr_db)
    floors = np.array(floors)
    snrs = np.array(snrs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle("Physiological clutter sweep  |  motor cortex (M1) honest demo",
                 fontsize=14, fontweight="bold")
    cl = clutter_m * 1e9

    ax1.loglog(cl, floors * 1e9, "o-", color="#c0392b", lw=2, label="binding floor")
    ax1.axhline(echo_floor * 1e9, ls="--", color="#2c6fbb",
                label=f"echo-SNR floor ({echo_floor * 1e9:.0f} nm)")
    ax1.axhline(sig * 1e9, ls=":", color="#27ae60", label=f"signal ({sig * 1e9:.3g} nm)")
    ax1.axvline(echo_floor * 1e9, ls="--", color="grey", alpha=0.6)
    ax1.set_xlabel("residual physiological clutter (nm)")
    ax1.set_ylabel("displacement (nm, log)")
    ax1.set_title("1. Binding floor vs clutter")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")

    ax2.semilogx(cl, snrs, "o-", color="#c0392b", lw=2)
    ax2.axhline(0, ls="--", color="k", alpha=0.5)
    ax2.axhline(base.detection.snr_db, ls=":", color="#2c6fbb",
                label=f"echo-limited plateau ({base.detection.snr_db:.1f} dB)")
    ax2.set_xlabel("residual physiological clutter (nm)")
    ax2.set_ylabel("detectability SNR (dB)")
    ax2.set_title("2. Verdict vs clutter")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    print("crossover at echo floor %.0f nm; verdict pinned at %.1f dB below it"
          % (echo_floor * 1e9, base.detection.snr_db))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("plots/clutter_sweep.png"))
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = build_figure()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    print("wrote", args.output.resolve())
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
