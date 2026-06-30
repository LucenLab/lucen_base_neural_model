"""The neural-model summary: what one simulation did, end to end.

A single figure that follows one ``run_neural_model`` from neural activity to a
predicted nanometre voxel displacement, so the activity -> mechanics translation is
auditable at a glance. Four panels:

1. **Activity** - the E/I limit cycle and the synchrony r(t) it produces.
2. **The bridge** - the reduced NeuralState (s, f_c, sigma_t, coherent-neuron count)
   and the arrow to the predicted displacement, with the number on it. This is the
   panel that links the two layers.
3. **Jitter survival** - ``exp(-2 pi^2 f^2 sigma^2)`` across frequency, with this
   run's content corner ``f_c`` marked and the surviving fraction read off. The
   direct bridge between the neural dynamics and the jitter low-pass.
4. **Displacement decomposition** - the predicted Delta z broken into its parts
   (isotropic vs directional; coherent vs the incoherent pedestal; before vs after
   the jitter low-pass), all in nm.

Every number is read back from the live ``NeuralModelReport`` - the figure cannot
drift from the model, and it reports the ACTUAL computed values, not targets.

Run::

    uv sync --group viz
    uv run python scripts/plot_neural_summary.py          # -> plots/neural_summary.png
    uv run python scripts/plot_neural_summary.py --motor  # the M1 preset
    uv run python scripts/plot_neural_summary.py --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from base_neural_model.base.units import m_to_nm
from base_neural_model.model.run import (
    DEFAULT_VOXEL,
    run_motor_cortex,
    run_neural_model,
)


def _survival(freq_hz: np.ndarray, sigma_t_s: float) -> np.ndarray:
    """The jitter low-pass exp(-2 pi^2 f^2 sigma_t^2) across frequency."""
    return np.exp(-2.0 * np.pi**2 * freq_hz**2 * sigma_t_s**2)


def build_figure(*, motor: bool, low_beta: bool = False):
    import matplotlib.pyplot as plt

    if motor:
        report = run_motor_cortex(low_beta=low_beta)
        from base_neural_model.base.types import VoxelGeometry

        voxel = VoxelGeometry.motor_cortex_layer5()
        band = "low-beta" if low_beta else "high-beta"
        title_region = f"Motor cortex (M1, {band})"
    else:
        report = run_neural_model()
        voxel = DEFAULT_VOXEL
        title_region = "Generic cortex"

    state = report.neural_state
    md = report.mechanical_displacement
    act = report.activity

    s = state.synchrony_fraction
    f_c = state.content_freq_hz
    sigma_t = state.jitter_sigma_s
    survival = md.content_band_survival

    # The three requested data points -------------------------------------------
    coherent_neurons = int(round(s * voxel.neuron_count))   # illustrative count
    dz_total_nm = m_to_nm(md.value_m)
    dz_surviving_nm = m_to_nm(md.value_m * survival)
    survival_pct = 100.0 * survival

    fig = plt.figure(figsize=(15, 8.6), constrained_layout=True)
    fig.suptitle(
        f"Neural-model summary - {title_region}:  "
        f"activity  ->  NeuralState  ->  predicted dz = {dz_surviving_nm:.2f} nm",
        fontsize=14, fontweight="bold",
    )
    (ax_act, ax_bridge), (ax_surv, ax_decomp) = fig.subplots(2, 2)

    # --- Panel 1: the activity that produced the state -------------------------
    t_ms = act.t_s * 1e3
    ax_act.plot(t_ms, act.e, lw=1.2, color="#c0392b", label="E")
    ax_act.plot(t_ms, act.i, lw=1.2, color="#2c7fb8", label="I")
    ax_act.plot(t_ms, act.r, lw=1.6, color="#16a085", label="synchrony r(t)")
    ax_act.axhline(s, color="#e67e22", ls="--", lw=1.1, label=f"mean s = {s:.2f}")
    ax_act.set_xlim(0, min(250, t_ms[-1]))
    ax_act.set_ylim(0, 1.02)
    ax_act.set_xlabel("time (ms)")
    ax_act.set_ylabel("activation / synchrony")
    ax_act.set_title(f"1. Activity: E/I limit cycle at f_c = {f_c:.0f} Hz")
    ax_act.legend(loc="upper right", fontsize=8, ncol=2)
    ax_act.grid(alpha=0.3)

    # --- Panel 2: the bridge (NeuralState -> predicted dz) ----------------------
    ax_bridge.axis("off")
    lines_state = [
        ("synchrony  s", f"{s:.2f}"),
        ("content corner  f_c", f"{f_c:.0f} Hz"),
        ("jitter  sigma_t", f"{sigma_t * 1e3:.2f} ms"),
        ("mean firing rate", f"{state.mean_firing_rate_hz:.1f} Hz"),
        (
            "orientation coherence  Q",
            "-" if state.orientation_coherence is None
            else f"{state.orientation_coherence:.2f}",
        ),
        (f"coherent neurons  (s x N={voxel.neuron_count:,})", f"~{coherent_neurons:,}"),
    ]
    y = 0.96
    ax_bridge.text(0.0, y, "NeuralState", fontsize=12, fontweight="bold",
                   transform=ax_bridge.transAxes)
    y -= 0.10
    for label, val in lines_state:
        ax_bridge.text(0.02, y, label, fontsize=10, transform=ax_bridge.transAxes)
        ax_bridge.text(0.62, y, val, fontsize=10, fontweight="bold",
                       transform=ax_bridge.transAxes)
        y -= 0.085

    # The arrow + the headline predicted displacement.
    ax_bridge.annotate(
        "", xy=(0.5, y - 0.02), xytext=(0.5, y + 0.05),
        xycoords=ax_bridge.transAxes,
        arrowprops=dict(arrowstyle="-|>", color="#1f3b73", lw=2.5),
    )
    y -= 0.10
    ax_bridge.text(0.5, y, "transduction chain", fontsize=9, style="italic",
                   ha="center", color="#1f3b73", transform=ax_bridge.transAxes)
    y -= 0.11
    ax_bridge.text(0.5, y, f"Predicted dz = {dz_surviving_nm:.2f} nm", fontsize=15,
                   fontweight="bold", ha="center", color="#1f3b73",
                   transform=ax_bridge.transAxes,
                   bbox=dict(boxstyle="round", fc="#eaf0fb", ec="#1f3b73"))
    y -= 0.09
    ax_bridge.text(0.5, y, f"(coherent {dz_total_nm:.2f} nm x {survival_pct:.0f}% "
                   "jitter survival)", fontsize=9, ha="center", color="0.3",
                   transform=ax_bridge.transAxes)
    ax_bridge.set_title("2. The activity -> mechanics bridge", fontsize=11)

    # --- Panel 3: jitter survival vs frequency, anchored at f_c -----------------
    freqs = np.linspace(0.0, 120.0, 600)
    surv = _survival(freqs, sigma_t)
    ax_surv.plot(freqs, surv * 100, lw=2.2, color="#8e44ad")
    ax_surv.axvline(f_c, color="#16a085", ls="--", lw=1.4)
    ax_surv.plot([f_c], [survival_pct], "o", color="#16a085", ms=9, zorder=5)
    ax_surv.annotate(
        f"f_c = {f_c:.0f} Hz\n{survival_pct:.0f}% survives",
        xy=(f_c, survival_pct), xytext=(f_c + 12, min(survival_pct + 18, 90)),
        fontsize=9, fontweight="bold", color="#16a085",
        arrowprops=dict(arrowstyle="->", color="#16a085"),
    )
    ax_surv.fill_between(freqs, 0, surv * 100, where=freqs <= f_c,
                         color="#d2b4de", alpha=0.4)
    ax_surv.set_xlim(0, 120)
    ax_surv.set_ylim(0, 102)
    ax_surv.set_xlabel("frequency (Hz)")
    ax_surv.set_ylabel("coherent displacement surviving (%)")
    ax_surv.set_title(
        f"3. Jitter low-pass exp(-2 pi^2 f^2 sigma^2),  sigma_t = {sigma_t*1e3:.1f} ms"
    )
    ax_surv.grid(alpha=0.3)

    # --- Panel 4: displacement decomposition (nm) ------------------------------
    iso_nm = m_to_nm(md.isotropic_axial_m)
    dir_nm = m_to_nm(md.directional_axial_m)
    pedestal_nm = m_to_nm(md.incoherent_pedestal_m)
    bars = [
        ("isotropic\n(volume)", iso_nm, "#2c7fb8"),
        ("directional\n(orientation)", dir_nm, "#27ae60"),
        ("coherent\ntotal", dz_total_nm, "#1f3b73"),
        ("after jitter\nlow-pass", dz_surviving_nm, "#8e44ad"),
        ("incoherent\npedestal", pedestal_nm, "#c0392b"),
    ]
    labels = [b[0] for b in bars]
    vals = [b[1] for b in bars]
    colours = [b[2] for b in bars]
    xpos = np.arange(len(bars))
    ax_decomp.bar(xpos, vals, color=colours, alpha=0.85)
    for x, v in zip(xpos, vals, strict=True):
        ax_decomp.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9,
                       fontweight="bold")
    ax_decomp.set_xticks(xpos)
    ax_decomp.set_xticklabels(labels, fontsize=8.5)
    ax_decomp.set_ylabel("displacement (nm)")
    ax_decomp.set_title("4. Predicted dz, decomposed")
    ax_decomp.grid(alpha=0.3, axis="y")

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("plots/neural_summary.png")
    )
    parser.add_argument("--motor", action="store_true",
                        help="use the motor-cortex (M1) preset")
    parser.add_argument("--low-beta", action="store_true",
                        help="with --motor, use the low-beta M1 preset (~13-17 Hz)")
    parser.add_argument("--show", action="store_true", help="also open a window")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = build_figure(motor=args.motor, low_beta=args.low_beta)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.output.resolve()}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
