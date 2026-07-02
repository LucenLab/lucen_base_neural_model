# base_neural_model

A comprehensive model of **what neural activity is, what mechanical signal it produces
in tissue, and whether a conventional ultrasound system can detect it** — a dynamical
neural-activity layer feeding a biophysical transduction chain, then composed with an
acoustic detection layer at the explicit source/sensing seam.

Three layers, one pipeline:

1. **Activity** — a Wilson–Cowan excitatory/inhibitory **neural-mass model**
   integrated over time (`scipy`), producing firing, oscillations, and population
   **synchrony**, reduced to a `NeuralState` (synchrony `s`, jitter `σ_t`, content
   corner `f_c`, mean rate).
2. **Mechanics** — the transduction chain membrane Δr → per-cell volume change →
   tissue strain ε_V → **net axial dilatation Δz**, with the Eshelby confinement
   factor κ and the poroelastic dilatation fraction η.
3. **Forward** — the source/sensing seam: Δz composed with a conventional
   phase-sensitive ultrafast-ultrasound acquisition (within-epoch √N integration +
   the through-skull Walker–Trahey detection floor) into a Gate-A / Stage-1
   detectability verdict. The first three layers stop *before* sensing; this one
   crosses the boundary, and is opt-in so they stay untouched without it.

It produces the **source deliverables** — an activity-driven displacement **timeseries
`dz(t)`** with its content-band spectrum, and the **static tissue displacement from a
neural state** with its full decomposition — and, when an acquisition is supplied, the
**detectability verdict** (SNR vs the derived through-skull floor).

```python
from base_neural_model.model import run_neural_model, run_motor_demo

r = run_neural_model()
r.neural_state.synchrony_fraction       # s, from the Kuramoto order parameter
r.displacement_timeseries.peak_dz_m     # (a) the dz(t) timeseries peak
r.mechanical_displacement.value_m       # (b) the static dz(neural_state)
r.all_gates_pass                        # the three kill gates

# Opt into the acoustic detection layer (Gate A / Stage 1):
r = run_motor_demo()                    # bursty motor imagery → honest detectability
r.detection.snr_db                      # content-band (direct beta) SNR (≈ −65 dB honest)
r.mechanisms.envelope_band_axial_m      # the large slow hemodynamic (fUS) envelope, S2
r.detection.limiting_denominator        # "echo_snr" | "clutter"
```

See [`docs/architecture.md`](docs/architecture.md) — **start here.**

## Design invariants

1. **SI units everywhere.** Metres, Hz, seconds, Pa, kg/m³, m/s. Convert to
   human-readable units only at the reporting boundary
   ([`base/units.py`](base_neural_model/base/units.py)). A single nm/µm slip moves the
   verdict by three orders of magnitude.
2. **Single-neuron displacement is a cited constant**, never simulated
   ([`mechanics/neuron_constants.py`](base_neural_model/mechanics/neuron_constants.py)).
3. **The content band is the target, never the slow envelope.** A typed `Band` guard
   ([`base/bands.py`](base_neural_model/base/bands.py)) makes the category error a
   raised exception.
4. **Synchrony is produced by the activity layer** (the Kuramoto order parameter),
   then swept through the mechanics — never asserted by hand.
5. **Every gate scores against a kill criterion** (`passes_*_gate`).
6. **Provenance propagates** ([`base/provenance.py`](base_neural_model/base/provenance.py))
   from the cited constant and the E/I parameters to every deliverable.
7. **Source and sensing are separated.** `activity`/`mechanics`/`model` are pure source
   physics; all ultrasound/detection lives in `forward/` and is opt-in
   (`run_neural_model(acquisition=…)`). Absent an acquisition, the source model is
   unchanged — no sensing assumption leaks into the source verdict.

## Layout

```text
base_neural_model/
  base/          SI units, temporal bands, provenance, the typed data contracts
  activity/      the dynamical neural-mass layer (E/I ODEs → synchrony, NeuralState)
  mechanics/     the transduction chain: neural state → net axial tissue displacement
  model/         the end-to-end model, the three kill gates, the inverse analyses
  forward/       the source/sensing seam: √N integration + the through-skull
                 detection floor (detection.py) and the acoustic budget sweep (budget.py)
tests/           invariant + layer tests
scripts/         live figures (matplotlib) recomputed from the model
docs/            subsystem-by-subsystem architecture docs
```

## Architecture docs

- [`docs/architecture.md`](docs/architecture.md) — the map: the two-layer pipeline,
  invariants, build order. **Start here.**
- [`docs/base.md`](docs/base.md) — foundation: units, bands, provenance, contracts.
- [`docs/activity.md`](docs/activity.md) — the dynamical neural-mass layer.
- [`docs/mechanics.md`](docs/mechanics.md) — the transduction chain (κ, η, the
  directional channel).
- [`docs/model.md`](docs/model.md) — the gates, the inverse and global analyses.
- [`docs/forward.md`](docs/forward.md) — the source/sensing seam: within-epoch √N
  integration + the through-skull Walker–Trahey detection floor (Gate A / Stage 1).
- [`docs/motor_cortex.md`](docs/motor_cortex.md) — the motor-cortex (M1) variant:
  beta rhythm, columnar Betz-cell alignment, movement trials.

## Setup

This project is managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync               # create .venv + install base_neural_model + dev tools
uv sync --no-dev      # runtime deps only (numpy, scipy, SALib, scikit-fem)
uv sync --group viz   # + matplotlib for the scripts/ figures
```

## Running

```bash
uv run pytest               # the invariant + layer tests
uv run ruff check .         # lint
```

The end-to-end model:

```python
from base_neural_model.activity import run_activity, reduce_to_state
from base_neural_model.model import run_neural_model

ts = run_activity()                 # integrate the E/I neural-mass model
state = reduce_to_state(ts)         # → NeuralState (s, σ_t, f_c, rate)
report = run_neural_model()         # both deliverables + the three gates
```

Figures (after `uv sync --group viz`) render into [`plots/`](plots/). The headline
is the consolidated **neural-model summary** — one figure following activity →
`NeuralState` → predicted Δz, with the coherent-neuron count, the jitter-survival
curve, and the displacement decomposition:

```bash
uv run python scripts/plot_neural_summary.py            # generic cortex
uv run python scripts/plot_neural_summary.py --motor    # motor cortex (M1)
```

See [`plots/README.md`](plots/README.md) for all the figures.

## What this build establishes

From a dynamical model of neural activity, it produces the tissue mechanical
displacement that activity generates — both as a timeseries `dz(t)` and as a static
displacement from a reduced neural state — decomposed to its biophysical factors
(Eshelby κ, poroelastic η, content-band survival), scored against three kill gates,
and shown (via Sobol sensitivity and verdict-flip) to collapse onto the net-dilatation
fraction η and the activity drive that sets synchrony.

The opt-in `forward/` layer then carries that source displacement across the
source/sensing seam to the **Gate-A / Stage-1 detectability question**, with the honest
source and acoustic physics as the default. Detectability is the bare surviving Δz over the
through-skull Walker–Trahey floor. The two receive-side coherent gains still live in that
floor (spatial √n_elements beamforming, temporal 1/√N_ens integration), but honestly: the
integration is **burst-limited** (√N_ens ≈ ×10, not ×77 — sensorimotor beta is transient,
and ultrafast frames decorrelate), the aperture is **partly incoherent** through the skull,
and the floor also carries a **residual-aberration** term, an **echo-correlation** penalty,
and a **safety-capped** echo SNR. On the honest source side the cited Δr is the sub-nm
mammalian value (~0.4 nm), the tissue is **viscoelastic** (not a static spring), and only
the rate-modulated, mutually-coherent fraction lives at the beta carrier. The result: the
direct-neuromechanical **content-band verdict is ≈ −65 dB** — the specific fast readout is
far under the floor, not "within an order." The band-separated **mechanism decomposition**
(`run_motor_demo().mechanisms`) makes the trade explicit: the slow **hemodynamic (CBV)
envelope** is orders larger (≈ +13 dB) — but that is ordinary functional ultrasound, not the
beta carrier. `demo_motor_optimistic()` preserves the prior ≈ −13 dB baseline for the
before/after audit. The detection layer is opt-in: without an acquisition the build is
purely a neural source model, not a detector.
