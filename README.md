# base_neural_model

A comprehensive model of **what neural activity is and what mechanical signal it
produces in tissue** — a dynamical neural-activity layer feeding a biophysical
transduction chain, stopping *before* any sensing, ultrasound, or detection.

Two layers, one pipeline:

1. **Activity** — a Wilson–Cowan excitatory/inhibitory **neural-mass model**
   integrated over time (`scipy`), producing firing, oscillations, and population
   **synchrony**, reduced to a `NeuralState` (synchrony `s`, jitter `σ_t`, content
   corner `f_c`, mean rate).
2. **Mechanics** — the transduction chain membrane Δr → per-cell volume change →
   tissue strain ε_V → **net axial dilatation Δz**, with the Eshelby confinement
   factor κ and the poroelastic dilatation fraction η.

It produces **two deliverables**: an activity-driven displacement **timeseries
`dz(t)`** with its content-band spectrum, and the **static tissue displacement from a
neural state** with its full decomposition.

```python
from base_neural_model.model import run_neural_model

r = run_neural_model()
r.neural_state.synchrony_fraction       # s, from the Kuramoto order parameter
r.displacement_timeseries.peak_dz_m     # (a) the dz(t) timeseries peak
r.mechanical_displacement.value_m       # (b) the static dz(neural_state)
r.all_gates_pass                        # the three kill gates
```

See [`docs/architecture.md`](docs/architecture.md) — **start here.**

## Design invariants (non-negotiable, enforced as code)

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
   from the cited constant and the E/I parameters to both deliverables.

## Layout

```text
base_neural_model/
  base/          SI units, temporal bands, provenance, the typed data contracts
  activity/      the dynamical neural-mass layer (E/I ODEs → synchrony, NeuralState)
  mechanics/     the transduction chain: neural state → net axial tissue displacement
  model/         the end-to-end model, the three kill gates, the inverse analyses
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
fraction η and the activity drive that sets synchrony. It deliberately stops before
any sensing modality: it is the neural model, not a detector.
