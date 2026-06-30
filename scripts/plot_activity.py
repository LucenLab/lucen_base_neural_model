"""Visualize the dynamical activity layer: E/I rhythm, synchrony r(t), band power.

The neural-mass model produces the rhythm and the population synchrony that drive the
mechanical signal. This figure reads them straight back from the live activity layer
(:func:`base_neural_model.activity.run_activity`) so the plot cannot drift from the
model: the E(t)/I(t) limit cycle, the instantaneous Kuramoto synchrony r(t), and the
one-sided power spectrum of E(t) with the content/envelope split marked.

Run::

    uv sync --group viz
    uv run python scripts/plot_activity.py              # writes activity.png
    uv run python scripts/plot_activity.py --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

from base_neural_model.activity import run_activity
from base_neural_model.activity.oscillation import ENVELOPE_CONTENT_BOUNDARY_HZ
from base_neural_model.activity.populations import EIParams

# The three cited rhythm presets, by band.
_PRESETS = {
    "gamma": (EIParams.central, "generic cortex, gamma"),
    "high-beta": (EIParams.motor_cortex, "M1, high-beta"),
    "low-beta": (EIParams.motor_cortex_low_beta, "M1, low-beta"),
}


def build_figure(*, preset: str = "gamma"):
    import matplotlib.pyplot as plt

    ei_factory, label = _PRESETS[preset]
    ts = run_activity(ei_factory())
    spec = ts.spectrum

    fig = plt.figure(figsize=(14, 4.6), constrained_layout=True)
    fig.suptitle(
        f"Dynamical activity layer ({label})   "
        f"f_c = {spec.dominant_freq_hz:.1f} Hz   "
        f"mean synchrony s = {ts.mean_synchrony:.2f}",
        fontsize=13, fontweight="bold",
    )
    ax0, ax1, ax2 = fig.subplots(1, 3)

    # --- Panel 1: the E/I limit cycle ------------------------------------------
    ax0.plot(ts.t_s * 1e3, ts.e, lw=1.4, color="#c0392b", label="E (excitatory)")
    ax0.plot(ts.t_s * 1e3, ts.i, lw=1.4, color="#2c7fb8", label="I (inhibitory)")
    ax0.set_xlim(0, min(200, ts.t_s[-1] * 1e3))
    ax0.set_xlabel("time (ms)")
    ax0.set_ylabel("population activation")
    ax0.set_title("1. E/I neural-mass limit cycle")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(alpha=0.3)

    # --- Panel 2: instantaneous synchrony r(t) ---------------------------------
    ax1.plot(ts.t_s * 1e3, ts.r, lw=1.4, color="#16a085")
    ax1.axhline(ts.mean_synchrony, color="#e67e22", ls="--", lw=1.2,
                label=f"mean s = {ts.mean_synchrony:.2f}")
    ax1.set_ylim(0, 1.02)
    ax1.set_xlim(0, min(200, ts.t_s[-1] * 1e3))
    ax1.set_xlabel("time (ms)")
    ax1.set_ylabel("synchrony  r(t)  (Kuramoto order parameter)")
    ax1.set_title("2. Population synchrony tracks the rhythm")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.3)

    # --- Panel 3: power spectrum of E(t) with the content/envelope split -------
    nonzero = spec.freqs_hz > 0
    ax2.semilogy(spec.freqs_hz[nonzero], spec.power[nonzero], lw=1.3, color="#756bb1")
    ax2.axvline(ENVELOPE_CONTENT_BOUNDARY_HZ, color="#c0392b", ls=":", lw=1.3)
    ax2.text(ENVELOPE_CONTENT_BOUNDARY_HZ + 1, ax2.get_ylim()[1] * 0.3,
             "envelope | content", color="#c0392b", fontsize=8, rotation=90,
             va="top")
    ax2.axvline(spec.dominant_freq_hz, color="#16a085", ls="--", lw=1.2,
                label=f"f_c = {spec.dominant_freq_hz:.1f} Hz")
    ax2.set_xlim(0, 120)
    ax2.set_xlabel("frequency (Hz)")
    ax2.set_ylabel("power of E(t)")
    ax2.set_title(f"3. Spectrum (content fraction {spec.content_fraction:.2f})")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3, which="both")

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("plots/activity.png"))
    parser.add_argument("--preset", choices=tuple(_PRESETS), default="gamma",
                        help="rhythm preset (gamma | high-beta | low-beta)")
    parser.add_argument("--show", action="store_true", help="also open a window")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = build_figure(preset=args.preset)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.output.resolve()}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
