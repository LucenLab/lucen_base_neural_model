# Model layer — `base_neural_model/model/`

The end-to-end model: it wires the dynamical activity layer to the mechanics chain,
scores the result against the three kill gates, and hosts the inverse and
global-sensitivity analyses. This is where the two deliverables are assembled.

## `run_neural_model` — both deliverables

[`run.py`](../base_neural_model/model/run.py) is the headline call:

```
EIParams ─run_activity─► ActivityTimeseries ─reduce─► NeuralState
                                │                          │
                                ├─ displacement_timeseries ─┤─► dz(t)      [deliverable (a)]
                                │       + displacement_spectrum ─► content-band spectrum
                                └─ mechanical_displacement ─┴─► dz(state)  [deliverable (b)]
                                                            └─► 3 gates → NeuralModelReport
```

The static displacement (b) is evaluated at the state's mean synchrony with the
state's jitter and content corner stamped on, so it equals the timeseries (a)
evaluated at that synchrony — the dynamic and static pictures agree by construction.
The cited single-neuron constant is pinned into the mechanics params in exactly one
place (Invariant 2).

## The three kill gates — `gates.py`

All three must pass (a stage that cannot fail proves nothing, Invariant 5):

- **Gate 1 — amplitude.** Some synchrony puts the coherent axial displacement within
  ~1–2 orders of the unaberrated floor.
- **Gate 2 — content survival.** The jitter-surviving coherent signal clears its own
  incoherent pedestal *and* the estimate floor.
- **Gate 3 — dilatation.** The net-dilatation fraction η is plausibly non-negligible
  (η ≥ η_floor). The model cannot *close* this gate; it exposes the η sensitivity.

## Inverse problems

- **`min_eta.py` — η\*.** The smallest η that clears all three gates at a synchrony —
  the falsifiable bench bar. Closed-form-monotone, located by bisection on the real
  chain.
- **`min_directional.py` — β\* and Q\*.** The directional analogue: the smallest
  per-cell anisotropy `β` (`min_anisotropy`) or orientation order `Q`
  (`min_orientation_coherence`) that lifts the signal across the gates. Δz is linear
  in the directional term, so the same bisection applies *when the orientation helps*
  (cells expand toward the beam, `g(Q,μ) > 0`). When the axes are across the beam
  (`g ≤ 0`) the directional term *subtracts* signal — raising β/Q cannot rescue — and
  the solver reports `infeasible_direction`. The four regimes are: `isotropic` (no
  directional help needed), `directional` (a threshold exists), `infeasible` (even the
  maximum directional term fails), `infeasible_direction` (`g ≤ 0`). These are the
  bench targets for **cell shape** and **columnar alignment** — what a measurement
  would have to beat — not claims about the tissue's actual values.
- **`resolution.py` — finest voxel.** Holding (η, s) fixed, how small the voxel can
  shrink (at fixed cell density) before a gate fails. The signal falls ~ρ while the
  pedestal rises ~ρ^(−1/2), so there is a finite resolution floor.

## Global analyses — the verdict collapse

- **`sensitivity.py` — Sobol.** Variance-based global sensitivity (SALib Saltelli) over
  the factor space. `SobolProblem.default()` sweeps the mechanics factors (η via
  log-permeability, synchrony, geometry, the cited constant); `SobolProblem.with_activity()`
  replaces bare synchrony with the upstream **E/I drive** and firing heterogeneity, so
  the collapse spans activity → mechanics. The result: total-order variance concentrates
  on the activity drive and the η axis, while geometry and the cited constant stay near
  zero.
- **`verdict_flip.py` — per-factor flip.** For each factor, hold the rest at nominal and
  find the value that flips the verdict. Only η and the synchrony driver (drive) flip;
  the literature-pinned controls cannot.

## Tests

`test_model_run.py` (both deliverables, static = timeseries at matched synchrony,
gates, provenance), plus the re-homed `test_min_eta`, `test_resolution`,
`test_pass_space`, `test_sobol_collapse` (including the activity-spanning collapse),
`test_verdict_flip`.
