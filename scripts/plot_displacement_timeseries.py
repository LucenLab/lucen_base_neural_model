"""Visualize deliverable (a): the activity-driven tissue displacement dz(t).

The headline dynamic product. The activity layer's synchrony trajectory r(t) drives
the transduction chain, producing the net axial tissue displacement dz(t); this
figure reads it back from the live model (:func:`base_neural_model.model.
run_neural_model`) and shows dz(t) alongside its driving synchrony, plus the
content-band power spectrum of dz(t).

Run::

    uv sync --group viz
    uv run python scripts/plot_displacement_timeseries.py        # writes dz_t.png
    uv run python scripts/plot_displacement_timeseries.py --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

from base_neural_model.activity.oscillation import ENVELOPE_CONTENT_BOUNDARY_HZ
from base_neural_model.base.units import m_to_nm
from base_neural_model.model import run_neural_model


def build_figure():
    import matplotlib.pyplot as plt

    report = run_neural_model()
    ts = report.displacement_timeseries
    spec = report.spectrum
    state = report.neural_state
    md = report.mechanical_displacement

    dz_nm = m_to_nm(ts.surviving_dz)
    coherent_nm = m_to_nm(md.value_m)
    surviving_nm = m_to_nm(md.value_m * md.content_band_survival)
    survival_pct = 100.0 * md.content_band_survival

    fig = plt.figure(figsize=(14, 4.6), constrained_layout=True)
    fig.suptitle(
        "Activity-driven tissue displacement   "
        f"s = {state.synchrony_fraction:.2f}   f_c = {state.content_freq_hz:.1f} Hz   "
        f"predicted dz = {surviving_nm:.2f} nm  "
        f"(coherent {coherent_nm:.2f} nm x {survival_pct:.0f}% jitter survival)",
        fontsize=12, fontweight="bold",
    )
    ax0, ax1, ax2 = fig.subplots(1, 3)

    # --- Panel 1: dz(t), content-surviving displacement ------------------------
    ax0.plot(ts.t_s * 1e3, dz_nm, lw=1.4, color="#2c7fb8")
    ax0.axhline(surviving_nm, color="#e67e22", ls="--", lw=1.2,
                label=f"predicted dz = {surviving_nm:.2f} nm")
    ax0.set_xlim(0, min(200, ts.t_s[-1] * 1e3))
    ax0.set_xlabel("time (ms)")
    ax0.set_ylabel("displacement dz(t)  (nm)")
    ax0.set_title("1. Activity-driven displacement (deliverable a)")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(alpha=0.3)

    # --- Panel 2: the driving synchrony r(t) -----------------------------------
    ax1.plot(ts.t_s * 1e3, ts.synchrony, lw=1.4, color="#16a085")
    ax1.set_ylim(0, 1.02)
    ax1.set_xlim(0, min(200, ts.t_s[-1] * 1e3))
    ax1.set_xlabel("time (ms)")
    ax1.set_ylabel("driving synchrony  r(t)")
    ax1.set_title("2. dz(t) tracks the synchrony drive")
    ax1.grid(alpha=0.3)

    # --- Panel 3: content-band spectrum of dz(t) -------------------------------
    nonzero = spec.freqs_hz > 0
    ax2.semilogy(spec.freqs_hz[nonzero], spec.power[nonzero], lw=1.3, color="#756bb1")
    ax2.axvline(ENVELOPE_CONTENT_BOUNDARY_HZ, color="#c0392b", ls=":", lw=1.3)
    ax2.text(ENVELOPE_CONTENT_BOUNDARY_HZ + 1, ax2.get_ylim()[1] * 0.3,
             "envelope | content", color="#c0392b", fontsize=8, rotation=90,
             va="top")
    ax2.set_xlim(0, 120)
    ax2.set_xlabel("frequency (Hz)")
    ax2.set_ylabel("power of dz(t)")
    ax2.set_title(f"3. dz spectrum (content fraction {spec.content_fraction:.2f})")
    ax2.grid(alpha=0.3, which="both")

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("plots/dz_t.png"))
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
