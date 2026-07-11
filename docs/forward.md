# Forward acoustic / detection layer — `base_neural_model/forward/`

**Status: ✅ implemented and tested. The source/sensing seam.**

Every other layer (`activity`, `mechanics`, `model`) stops *before* any ultrasound, by
design — they answer "what mechanical displacement does neural activity produce?" and
nothing about detecting it. `forward/` is the first module that crosses that boundary.
It takes the net axial source displacement Δz the mechanics chain produces and answers
the **Gate A / Stage 1** question both Lucen specs make the cheapest falsification:

> Does the content-bearing displacement, after within-epoch coherent integration, clear
> the phase-sensitive ultrasound detection floor **through the skull**?

No Lucen-specific innovation is modelled here — the spec excludes it at this gate
(no metamaterial correction, no crossed-beam, no custom array). It is a conservative
baseline using conventional research-system characteristics.

```
base_neural_model/forward/
  detection.py   AcquisitionParams, integration gain, the Walker-Trahey floor,
                 and the DetectionBudget that composes them onto a source Delta z
  budget.py      sweep the acoustic axes around one DetectionBudget: SNR-vs-skull-loss
                 curves + the acoustic-side "which axis carries the verdict" ranking
```

## The receive-side coherent gains — two, orthogonal, each counted once

The module recomputes **no** source physics. Detectability is one ratio — the **bare**
surviving `Δz · survival` over the noise floor on estimating it — and **all** the
receive-side coherent gain lives in that floor, in two orthogonal places so that no gain
is ever counted twice. (This is the fix for a prior double-count: the signal was being
amplified by `√N_ens` **and** the floor shrunk by `√N_ens`, applying the one temporal
gain twice — a spurious +37.8 dB. The signal is now bare; the floor carries the averaging.)

### 1. Temporal: within-epoch coherent integration — the demo's load-bearing advantage

Ultrafast imaging collects `N_ens = f_frame · T_epoch` estimates of the same quasi-static
`Δz` per voxel per epoch; averaging them drops the estimate variance as `1/N_ens` (std as
`1/√N_ens` — the Cramér–Rao result for a constant):

```
integration_gain = √(f_frame · T_epoch)     # appears ONLY as 1/√N_ens in the floor
```

This is the **entire reason motor cortex is the favorable bet**: a 1.5 s motor-imagery
epoch at 4 kHz gives N_ens = 6000 (**×77**), where a 50 ms speech unit gives N_ens = 200
(**×14**). The `sustained_imagery_drive`
([`motor_drive.py`](../base_neural_model/activity/motor_drive.py)) supplies the high,
*stable* synchrony this integrates over (the opposite of `movement_drive`'s onset desync).

### 2. Spatial: receive beamforming across the aperture

Delay-and-sum over the `n_elements`-element receive aperture (spec §4.2) coherently sums
the signal (~`n_elements`) and incoherently sums the noise (~`√n_elements`), so the
beamformed **voxel** echo SNR is `n_elements ×` the per-element echo SNR:

```
beamforming_gain = √n_elements          # appears ONCE, inside SNR_echo in the floor
beamformed_echo_snr = n_elements · echo_snr_linear      # 256 × per-element
```

`echo_snr_linear` is therefore the **per-element (pre-beamforming)** echo SNR; the ×256
aperture gain is applied explicitly in `through_skull_echo_snr_linear`, so it can never be
silently baked in twice. This is orthogonal to §1 (that is temporal, across frames).

### 3. The phase-sensitive displacement floor (Walker–Trahey) — where both gains live

Echo-phase displacement estimation has a variance floor set by echo SNR and wavelength;
both receive-side gains fold in here, each once:

```
σ_disp = (λ / 4π) · √(1 / (2 · n_elements · SNR_perelem_ts)) · (1 / √N_ens)     λ = c / f_N
```

The `λ/4π` prefactor inverts the round-trip phase map `δφ = (4π f_N / c)·Δz` — at 2 MHz,
6.13e-5 m/rad, so a 10 nm shift is ~1.6e-4 rad (a sixth of a milliradian — the spec's
number). **Two-way skull loss** is applied inside the floor (the beamformed `SNR_echo`
scaled by `10^(−2·loss_dB/10)`), so the reported number is the *through-bone* floor. The
`√n_elements` (spatial) sits inside `SNR_echo`; the `1/√N_ens` (temporal) is the trailing
factor. This is the **derived replacement for the old hand-set `DEFAULT_FLOOR_M` scalar.**

## The budget and the binding denominator

`detection_budget(mech, acq, residual_clutter_m=…)` returns a `DetectionBudget` carrying
the **bare** surviving signal `Δz · survival` (the `√n_elements` and `1/√N_ens` gains are
in the floor, not the signal), the floor, the `snr_db`, and `limiting_denominator` —
**"echo_snr" vs "clutter"** — which the spec's Gate B asks us to report (a clutter-limited
result reframes the problem as engineering, not a physics wall). It also surfaces
`integration_gain`, `beamforming_gain`, and `n_elements` for transparency. The binding
floor is the worse of the Walker–Trahey echo-SNR floor and the optional
post-clutter-filter residual.

## Wiring into the verdict

`run_neural_model(..., acquisition=AcquisitionParams)` opts in: the amplitude /
content-survival gates score the **bare** static Δz directly against the derived
through-skull floor (which already carries both receive-side gains), the same ratio as
`DetectionBudget.snr_db`. `gates.py` is unchanged. With `acquisition=None` the model stays
strictly source-side and the scalar-floor path is untouched.

`run_motor_demo()` is the single call that *is* the demo's Gate A: M1 presets + sustained
imagery + `AcquisitionParams.demo_motor()`.

## The honest-physics terms (D1–D9)

Beyond the two coherent gains, `AcquisitionParams` carries an honest-physics extension,
each field **inert by default** (a bare construction, or `demo_motor_optimistic()`,
reproduces the prior attenuation-only, full-integration −13 dB baseline) and flipped on in
`demo_motor()`:

- **D1 — effective sample count.** `N_ens` is not `f_frame·T_epoch`: the signal stays
  coherent only over a **beta burst** (`coherence_time_s`, ~200 ms), and ultrafast frames
  **decorrelate** (`frame_decorrelation_time_s`, ~2 ms), so the *independent* looks are
  `coherent_window / decorrelation_time` — for the demo ~100, not 6000 (×10, not ×77).
- **D2 — clutter high-pass.** `clutter_highpass_hz` bounds the coherent window and removes
  any content below its cutoff (a beta carrier survives; a slow envelope does not).
- **D3 — residual-aberration phase noise.** `aberration_phase_rad` maps to a displacement
  floor `(λ/4π)·φ_aber` added **in quadrature**, and it does **not** average with
  integration (it is not an SNR term).
- **D4 — aperture decoherence.** through an uncorrected skull only `aperture_coherence` of
  the array sums coherently, so the beamforming gain uses `n_eff = aperture_coherence·n`.
- **D5 — echo correlation ρ.** `echo_correlation` inflates the per-estimate floor by `1/ρ`.
- **D6 — safety cap.** [`safety.py`](../base_neural_model/forward/safety.py) sets the
  transcranial MI/thermal ceiling on the echo SNR; `snr_exceeds_safety` flags when the
  assumed value (30 dB) is above the ~28 dB safely reachable at 12 dB skull.
- **D7 — frequency-coupled skull loss.** [`skull.py`](../base_neural_model/forward/skull.py)
  couples `skull_loss_db_oneway` to `center_freq_hz` (α ∝ fⁿ); `frequency_trade_curve`
  resolves the sensitivity-vs-attenuation trade (the SNR-optimal readout is ~0.8 MHz, so
  2 MHz is a resolution choice, not an SNR one).
- **D8 — reverberation clutter.** `reverberation_ratio` adds a reverberation floor to the
  clutter denominator.
- **D9 — per-element vs post-beamforming.** `echo_snr_is_per_element` (default `True`) makes
  the B1 commitment explicit: `False` removes the √n aperture gain (a −24 dB swing).

## The honest result

Under the honest terms (`run_motor_demo()`, verified live), the flagship demo reports a
**content-band verdict of ≈ −87.8 dB**, echo-SNR-limited: the bare direct-neuromechanical
beta signal (~0.0094 nm surviving, at the ~0.3 nm mammalian Δr midpoint, in-burst s ≈ 0.83)
against a **~232 nm** through-skull floor — the floor raised from the optimistic 17.5 nm by
the burst-limited integration (×10 not ×77), the aperture decoherence (n_eff = 154), the echo
correlation and the residual-aberration floor. This is **not** "within an order": the direct
beta signal is ~88 dB (~4.4 decades) below the floor. Composing the *same* honest source with
the all-favourable acoustic baseline (`run_motor_demo(acquisition=demo_motor_optimistic())`)
lands at ≈ −65 dB, so the honest *acoustic* terms alone cost ~22 dB; the rest is the honest
source physics. (The acoustic layer in isolation, against a fixed synthetic ~3.9 nm source, is
the −13.03 dB unit-test baseline pinned in `test_detection.py`.)

The **band-separated mechanism decomposition** (S2, `report.mechanisms`) makes the trade
explicit: the slow **hemodynamic (vascular/CBV) envelope** is ~800 nm → **+10.8 dB**, orders
larger than the direct term — but that is the **fUS signal** (cerebral blood volume), an
envelope, not the specific beta carrier, and in a phase-displacement readout it is removed
by the 1 Hz clutter high-pass (fUS instead uses power-Doppler, a different modality). So the
honest reading is: the specific fast neuromechanical readout is far under, while the thing
that *is* detectable is ordinary functional ultrasound.

## The optimistic engineering-gap counterpart — `run_motor_demo_optimistic()`

The honest −87.8 dB is the *pessimistic* end of a range; the spec's Stage 0 claims the gap is
"one to two orders," not four. `run_motor_demo_optimistic()`
([`run.py`](../base_neural_model/model/run.py)) composes the other end: every **contestable**
lever at its favourable-but-**physically-possible** value, with the neural biology and Δr
held fixed. It reports a content-band verdict of **≈ −30 dB (~1.5 orders under floor)** —
squarely inside the spec's engineering gap — via a **67 nm** floor and a **2.1 nm** surviving
signal (vs 232 nm / 0.0094 nm honest).

Each moved lever, and why none is physically impossible:

- **Source** ([`MechanicsParams.motor_cortex_optimistic`](../base_neural_model/base/types.py),
  [`VoxelGeometry.motor_cortex_active_column`](../base_neural_model/base/types.py)):
  `r` 20→8 µm (strain ∝ 1/r at fixed f_cell — finer neuropil surface-to-volume, ×2.5);
  `η` 0.5→1.0 (the undrained fast-timescale ceiling — poroelastic drainage is negligible at
  ms); `L` 0.3→1.5 mm (the coherently-active M1 column, not the range gate — Δz ∝ L; the cost
  is *depth* resolution, not the lateral finger-separation axis).
- **Floor** ([`AcquisitionParams.demo_motor_engineering`](../base_neural_model/forward/detection.py)):
  echo SNR **capped at the 28 dB transcranial safety ceiling** (the same ceiling the honest
  preset now sits at — both are `echo_snr_within_safety` True, so this is not a lever here);
  aperture coherence 0.6→0.9 (the
  metamaterial correction the architecture is built around); and the 2 ms frame-decorrelation
  cap removed so N_ens is the burst-window 800 (thermal-independent frames), resolving the
  D1/echo-SNR-limited tension.

**Held fixed** (not levers): Δr (the cited whole-cell 0.3 nm midpoint — `run_neural_model` re-pins it;
raising it double-counts within-cell cancellation), the honest source-transfer sub-factors
(S1/S3/S6), the neural synchrony/jitter/orientation, and the safety cap. The guardrails
`η ≤ 1`, `κ ≥ 1/3`, and `echo_snr_within_safety` are pinned in
[`test_model_run.py`](../tests/test_model_run.py) and
[`test_detection.py`](../tests/test_detection.py).

The reading: honest and optimistic bracket the *same* signal — ~4.4 orders under at the
pessimistic end, ~1.5 at the physically-achievable-best end. Neither flips the sign; the
fast, effector-specific carrier is under the through-skull floor across the whole plausible
range, and the spread between the two is almost entirely the two D1 integration caps plus the
aberration-correction and safety-cap accounting.

## The budget sweep — where is the wall, and which axis carries the verdict

[`budget.py`](../base_neural_model/forward/budget.py) turns the single Gate-A point into
a curve. Holding the source Δz fixed, `sweep_axis` varies one `AcquisitionParams` axis
and returns a `BudgetCurve` (SNR in dB, surviving Δz, floor, denominator at each point)
with a `crossing_value` — the axis value where SNR hits **0 dB** (signal == floor, the
detection wall).

`acoustic_axis_ranking` is the acoustic-side complement to the source-side Sobol collapse
(`model/sensitivity.py`, which collapses onto η and s): a one-at-a-time local sensitivity
ranking the four acquisition axes by their dB swing over their literature span. It leads
with **skull loss** (steepest — a squared, two-way, exponential-in-dB term), then **echo
SNR**; **epoch** is now only weakly material, because the honest coherence window (D1) caps
how much of a longer epoch actually integrates — sweeping the epoch past the ~200 ms burst
barely moves the verdict. The clean **−2 dB per one-way skull dB** slope holds on the
optimistic base; on the honest preset the integration-independent aberration floor (D3)
flattens it at low skull loss. `frequency_trade_curve` (D7) adds the sensitivity-vs-
attenuation trade with an interior optimum near ~0.8 MHz.

[`scripts/plot_budget.py`](../scripts/plot_budget.py) renders both: SNR-vs-skull-loss at
three echo-SNR levels with the wall and the demo point, plus the axis ranking.

## Tests

[`test_detection.py`](../tests/test_detection.py) pins the honest terms on the inert base:
the raw ×14/×77 vs the coherence/decorrelation-capped `N_ens` (D1), the echo-correlation
floor (D5), the aberration quadrature floor that ignores integration (D3), the aperture-
coherence erosion (D4), the clutter-high-pass content removal (D2), reverberation (D8), the
per-element↔post-beamforming 24 dB swing (D9), and that the inert defaults reproduce the
−13.03 dB baseline while the honest preset is materially worse.
[`test_safety.py`](../tests/test_safety.py) pins the transcranial echo-SNR ceiling (D6).
[`test_model_run.py`](../tests/test_model_run.py) pins the honest acquisition gains and the
honest verdict: the content-band signal far under the floor while the hemodynamic envelope
is large (S2). [`test_budget.py`](../tests/test_budget.py) pins the −2 dB/dB slope (inert
base), the crossing, the clutter flip, the axis ranking (epoch now only weakly material),
and the frequency-trade interior optimum (D7).
