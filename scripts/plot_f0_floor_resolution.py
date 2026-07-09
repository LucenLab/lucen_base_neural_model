"""Floor vs readout frequency f0, with spatial resolution overlaid -- the frequency verdict.

The 232 nm through-skull floor is not fixed: it is a function of the readout centre
frequency f0. The Walker-Trahey displacement CRLB improves as ``lambda/4pi ~ 1/f0``
(higher frequency -> finer phase-to-displacement), while bone attenuation worsens as
``~f0^1.5`` (skull.py), so the floor is U-shaped with an interior minimum. This script
finds that minimum, reads off the spatial resolution (~wavelength) there, and asks the
decisive question: is there ANY frequency where the floor drops to the source signal --
i.e. does real-time baseline recovery exist?

It also exercises the ``aberration_jitter_m`` term (the in-band, frequency-independent
residual-skull-wavefront floor): because that floor is flat in f0, it sets a frequency-
independent wall the U-shape sits on. Sweeping a few jitter levels shows when the model
is genuinely echo-SNR-limited (U-shape visible) vs aberration-limited (floor flat).

Run::

    uv sync --group viz
    uv run python scripts/plot_f0_floor_resolution.py       # writes f0_floor_resolution.png
    uv run python scripts/plot_f0_floor_resolution.py --show
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from base_neural_model.forward.budget import frequency_trade_curve
from base_neural_model.forward.detection import AcquisitionParams, DEFAULT_SOUND_SPEED_MPS
from base_neural_model.forward.skull import skull_loss_from_freq
from base_neural_model.model import run_motor_demo, run_motor_demo_optimistic

C = DEFAULT_SOUND_SPEED_MPS

# Aberration-jitter levels to overlay (in-band residual-wavefront displacement floor, m).
JITTERS = {"0 (WT only)": 0.0, "10 nm": 1e-8, "50 nm": 5e-8, "200 nm": 2e-7}


def build_figure():
    import matplotlib.pyplot as plt

    demo = run_motor_demo()
    mech = demo.mechanical_displacement
    sig = demo.detection.surviving_dz_m
    sig_opt = run_motor_demo_optimistic().detection.surviving_dz_m

    base = AcquisitionParams.demo_motor()                          # N_ens = 100 integrated
    rt = replace(base, coherence_time_s=1.0 / base.frame_rate_hz)  # N_ens -> 1 (real time)

    freqs = np.logspace(np.log10(0.3e6), np.log10(5.0e6), 80)
    fmhz = freqs / 1e6

    def floor_of(acq, jitter_m):
        a = replace(acq, aberration_jitter_m=jitter_m)
        return frequency_trade_curve(mech, a, freqs).floor_m

    curves = {k: floor_of(base, j) for k, j in JITTERS.items()}
    rt_floor = floor_of(rt, 5e-8)

    wt = curves["0 (WT only)"]
    i = int(np.argmin(wt))
    lam_min = C / freqs[i]

    fig, ax = plt.subplots(figsize=(12.5, 6.6))
    fig.suptitle("Floor vs readout frequency f0, with spatial resolution  |  M1 honest demo",
                 fontsize=14, fontweight="bold")
    cols = {"0 (WT only)": "#2c6fbb", "10 nm": "#8e44ad", "50 nm": "#c0392b", "200 nm": "#e67e22"}
    for k, fl in curves.items():
        ax.loglog(fmhz, fl * 1e9, lw=2, color=cols[k], label=f"floor, aberration-jitter = {k}")
    ax.loglog(fmhz, rt_floor * 1e9, lw=1.6, ls="--", color="#c0392b",
              label="floor, REAL-TIME (N_ens=1), 50 nm jitter")
    ax.axhline(sig * 1e9, ls=":", color="#27ae60", lw=1.6, label=f"honest signal {sig * 1e9:.3g} nm")
    ax.axhline(sig_opt * 1e9, ls=":", color="#16a085", lw=1.4,
               label=f"optimistic signal {sig_opt * 1e9:.3g} nm")

    ax.plot(fmhz[i], wt[i] * 1e9, "o", color="#2c6fbb", ms=9, zorder=6)
    ax.annotate("floor min %.0f nm @ %.2f MHz\n(best case: full integration,\nzero jitter)"
                % (wt[i] * 1e9, fmhz[i]),
                xy=(fmhz[i], wt[i] * 1e9), xytext=(0.31, 1.5e4),
                fontsize=9, fontweight="bold", color="#2c6fbb",
                arrowprops=dict(arrowstyle="->", color="#2c6fbb"))

    ax.set_xlabel("readout centre frequency f0 (MHz)")
    ax.set_ylabel("displacement floor (nm, log)")
    ax.set_ylim(3e-3, 1e7)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower left", framealpha=0.95)

    ax2 = ax.twinx()
    ax2.loglog(fmhz, C / freqs * 1e3, color="grey", lw=1.4, alpha=0.8)
    ax2.set_ylabel("spatial resolution ~ wavelength lambda (mm)", color="grey")
    ax2.tick_params(axis="y", colors="grey")
    ax2.annotate("resolution here:\nlambda = %.1f mm (axial ~%.1f mm)" % (lam_min * 1e3, lam_min * 1e3 / 2),
                 xy=(fmhz[i], lam_min * 1e3), xytext=(1.4, 4.0),
                 fontsize=9, color="dimgrey", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="grey"))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    print("WT-only floor min %.4g nm at %.3g MHz (skull one-way %.3g dB); resolution lambda %.3g mm; "
          "gap to honest signal %.1f dB"
          % (wt[i] * 1e9, fmhz[i], skull_loss_from_freq(freqs[i]), lam_min * 1e3,
             20 * np.log10(sig / wt[i])))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("plots/f0_floor_resolution.png"))
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
