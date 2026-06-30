# base_neural_model — Architecture

`base_neural_model` is a comprehensive model of **what neural activity is and what
mechanical signal it produces in tissue** — stopping *before* any sensing,
ultrasound, propagation, or detection. It has two layers wired into one pipeline:

```
   EIParams (E/I neural-mass parameters, cited)
        │
        ▼  scipy ODE integration
   ┌─────────────────────────────────────────┐
   │ activity/  — dynamical neural-mass layer │  Wilson–Cowan E/I mean field
   │ firing, oscillations, synchrony over time│  Kuramoto order parameter r(t)
   └─────────────────────────────────────────┘
        │  ActivityTimeseries (t, E, I, r) ── reduce ──► NeuralState (s, σ_t, f_c, rate)
        ▼
   ┌─────────────────────────────────────────┐
   │ mechanics/ — transduction chain          │  Δr → ΔV → ε_V → Δz
   │ neural state → tissue displacement       │  Eshelby κ, poroelastic η
   └─────────────────────────────────────────┘
        │  DisplacementTimeseries dz(t)  +  MechanicalDisplacement dz(state)
        ▼
   ┌─────────────────────────────────────────┐
   │ model/     — gates + inverse analyses    │  3 kill gates; min_eta; resolution;
   │ verdict + sensitivity                    │  Sobol; verdict-flip
   └─────────────────────────────────────────┘
        │
        ▼   two deliverables
   (a) dz(t) timeseries + content-band spectrum     (b) static dz(neural_state)
```

| Subsystem | Package | Doc |
|---|---|---|
| Foundation | [`base/`](../base_neural_model/base/) | [base.md](base.md) |
| Activity (dynamical) | [`activity/`](../base_neural_model/activity/) | [activity.md](activity.md) |
| Mechanics (transduction) | [`mechanics/`](../base_neural_model/mechanics/) | [mechanics.md](mechanics.md) |
| Model (gates + analyses) | [`model/`](../base_neural_model/model/) | [model.md](model.md) |
| Forward (acoustic detection) | [`forward/`](../base_neural_model/forward/) | [forward.md](forward.md) |

The first four layers stop *before* any sensing. `forward/` is the explicit
source/sensing seam: it composes the source Δz with a conventional phase-sensitive
ultrasound acquisition (within-epoch √N integration + the through-skull Walker–Trahey
floor) to answer the Gate-A / Stage-1 detectability question. Opt-in via
`run_neural_model(acquisition=…)`; the source layers are untouched when it is absent.

## The two deliverables

`run_neural_model()` ([`model/run.py`](../base_neural_model/model/run.py)) returns a
`NeuralModelReport` carrying both:

- **(a) the activity-driven displacement timeseries** `dz(t)` with its content-band
  spectrum — the tissue displacement signal as the rhythm waxes and wanes;
- **(b) the static tissue displacement from a neural state** `dz(neural_state)` with
  the full transduction decomposition (coherent term, incoherent pedestal, κ, η,
  content-band survival).

Both share the same reduced `NeuralState` and the same cited single-neuron constant,
so the dynamic and static pictures are consistent by construction.

```python
from base_neural_model.model import run_neural_model
r = run_neural_model()
r.neural_state.synchrony_fraction          # s, from the Kuramoto order parameter
r.displacement_timeseries.peak_dz_m        # (a) dz(t) peak
r.mechanical_displacement.value_m          # (b) static dz
r.all_gates_pass                           # the three kill gates
```

## The design invariants (enforced as code)

1. **SI units everywhere** ([`base/units.py`](../base_neural_model/base/units.py));
   convert only at the reporting boundary. Tripwire `is_sane_displacement_m`.
2. **The single-neuron displacement is a cited constant**, never simulated
   ([`mechanics/neuron_constants.py`](../base_neural_model/mechanics/neuron_constants.py)).
3. **The content band is the target, never the slow envelope.** A typed `Band` guard
   ([`base/bands.py`](../base_neural_model/base/bands.py)) makes the category error a
   raised exception. The activity layer's content/envelope split and the mechanics'
   content-band survival both honor it.
4. **Synchrony is the swept variable of the mechanics**, but it is now *produced* by
   the activity layer (the Kuramoto order parameter), not asserted by hand. The
   activity layer's own independent variable is the E/I drive.
5. **Every gate scores against a kill criterion** ([`model/gates.py`](../base_neural_model/model/gates.py)).
6. **Provenance propagates** ([`base/provenance.py`](../base_neural_model/base/provenance.py))
   from the cited constant and the E/I parameters all the way to both deliverables.

## Build order (cheapest falsification first)

1. **`base/`** — units, types, bands, provenance.
2. **`mechanics/`** — the transduction chain (the deciding physics): membrane Δr →
   net axial dilatation Δz, with Eshelby κ and poroelastic η, plus the three gates
   and the inverse analyses (min_eta, resolution, sensitivity, verdict-flip).
3. **`activity/`** — the dynamical neural-mass layer that produces synchrony,
   oscillations, and band power, reduced to the `NeuralState` the mechanics consume.
4. **`mechanics/timeseries` + `mechanics/spectrum`** — deliverable (a), dz(t).
5. **`model/run`** — wire activity → mechanics → gates into both deliverables.

## Testing as executable invariants

The suite ([`tests/`](../tests/)) is organized by invariant/concept:

| Test file | What it pins |
|---|---|
| `test_units`, `test_bands` | SI sanity + the content-band guard |
| `test_coherence_limits`, `test_monotonicity` | transduction limits + no sign errors |
| `test_eshelby_kappa`, `test_poroelastic_eta` | the κ and η sub-models |
| `test_min_eta`, `test_resolution`, `test_pass_space` | the inverse problems |
| `test_sobol_collapse`, `test_verdict_flip` | the verdict collapses onto η + the activity drive |
| `test_provenance` | provenance non-null, end to end |
| `test_activity_neural_mass`, `_synchrony`, `_reduce` | the dynamical layer |
| `test_mechanics_timeseries`, `test_model_run` | both deliverables |

Run with `uv run pytest`.
