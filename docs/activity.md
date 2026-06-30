# Activity layer — `base_neural_model/activity/`

The dynamical front end: a Wilson–Cowan excitatory/inhibitory **neural-mass model**
integrated over time, producing firing, oscillations, and population **synchrony**,
then reduced to the `NeuralState` the mechanics consume. This is the "what neural
activity *is*" half of the model.

## The chain within the layer

```
EIParams (cited E/I regime)            populations.py
   │  scipy.integrate.solve_ivp
   ▼
E(t), I(t)                              neural_mass.py   (the dynamical core)
   │
   ├─ FFT → dominant freq, band power   oscillation.py   → content corner f_c
   ├─ Hilbert envelope → coupling K(t)  timeseries.py
   │     → Kuramoto order parameter     synchrony.py     → synchrony r(t), s
   └─ r, f_c → spike-timing jitter      jitter.py        → σ_t
   ▼
ActivityTimeseries (t, E, I, r)        timeseries.py
   │  reduce
   ▼
NeuralState (s, σ_t, f_c, mean_rate)   reduce.py         (the bridge to mechanics)
```

## What each module computes

- **`populations.py` — `EIParams`.** The Wilson–Cowan parameters: coupling weights
  `w_ee, w_ei, w_ie, w_ii`, membrane time constants `τ_e, τ_i`, sigmoid gains and
  thresholds, and the external drive. `EIParams.central()` sits in the standard
  gamma-generating limit-cycle regime, with cited provenance. Activity is the
  Wilson–Cowan *fraction of active cells* in [0, 1], not a rate in Hz.
- **`neural_mass.py` — the ODEs.** `τ Ė = −E + S(w·E − w·I + drive)` (likewise I),
  with a logistic `S`, integrated by `scipy.integrate.solve_ivp` (RK45) on a uniform
  grid after discarding a start-up transient. The central regime yields a content-band
  limit cycle.
- **`oscillation.py` — the rhythm.** One-sided FFT power spectrum of E(t), the
  dominant content-band frequency (→ `f_c`), and the content-vs-envelope power split
  at the 16 Hz boundary (the speech envelope/content line).
- **`synchrony.py` — the order parameter.** The Kuramoto `r = |⟨e^{iφ}⟩|`, and its
  mean-field stationary value `r = √(1 − K_c/K)` above the critical coupling
  `K_c = 2γ` (the second-order transition). A `synchrony_from_drive` surrogate maps the
  E/I drive straight to synchrony for the global-sensitivity sweeps without
  re-integrating the ODEs per sample.
- **`jitter.py` — spike-timing jitter.** Inverts the wrapped-normal order parameter
  `r = exp(−σ_φ²/2)` for the phase spread, then converts to time at the rhythm period:
  `σ_t = σ_φ/(2πf)`. Tighter synchrony → smaller jitter. Also exports the shared
  content-band survival `exp(−2π²f_c²σ_t²)`.
- **`timeseries.py` — `ActivityTimeseries` + `run_activity`.** One integration
  bundled with the instantaneous synchrony `r(t)` (from the Hilbert envelope of E(t)).
- **`reduce.py` — `reduce_to_state`.** Collapses the trajectory to a `NeuralState`
  `(s = mean r, σ_t, f_c, mean_rate)` — the activity → mechanics bridge. Keeping it a
  separate step makes the dynamical layer optional: a caller can build a `NeuralState`
  directly and skip the ODEs.

## Invariants this layer honors

- **SI** — time in seconds, frequency in Hz; activations dimensionless in [0, 1].
- **Content band** — the reduced `NeuralState` is labeled `CONTENT_FAST`; the
  oscillation split keeps the slow envelope separate from the content carrier.
- **Provenance** — the E/I source and assumptions propagate from `EIParams` through
  the timeseries and into the reduced state.

## Tests

`test_activity_neural_mass.py` (ODEs integrate, stay in [0,1], oscillate in band,
respond monotonically to drive), `test_activity_synchrony.py` (order parameter bounds
and monotonicity), `test_activity_reduce.py` (the reduced state is SI-sane,
content-band, provenanced, with jitter consistent with synchrony).
