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

## The two acoustic factors

The module recomputes **no** source physics — it composes two acoustic factors onto the
existing `MechanicalDisplacement` (its `axial_displacement_m` and `content_band_survival`):

### 1. Within-epoch coherent integration — the demo's load-bearing advantage

Ultrafast imaging collects `N_ens = f_frame · T_epoch` displacement samples per voxel per
epoch; coherent integration over them gains amplitude SNR as `√N_ens`:

```
integration_gain = √(f_frame · T_epoch)
```

This is the **entire reason motor cortex is the favorable bet**: a 1.5 s motor-imagery
epoch at 4 kHz gives N_ens = 6000 (**×77**), where a 50 ms speech unit gives N_ens = 200
(**×14**). The model previously had no term for this and so scored Gate A against the
*un-integrated* Δz — understating exactly the target the demo leads with. The
`sustained_imagery_drive` ([`motor_drive.py`](../base_neural_model/activity/motor_drive.py))
supplies the high, *stable* synchrony this integrates over (the opposite of
`movement_drive`'s onset desync).

### 2. The phase-sensitive displacement floor (Walker–Trahey)

Echo-phase displacement estimation has a variance floor set by echo SNR and wavelength:

```
σ_disp = (λ / 4π) · √(1 / (2·SNR_echo)) · (1 / √N_ens)        λ = c / f_N
```

The `λ/4π` prefactor inverts the round-trip phase map `δφ = (4π f_N / c)·Δz` — at 2 MHz,
6.13e-5 m/rad, so a 10 nm shift is ~1.6e-4 rad (a sixth of a milliradian — the spec's
number). **Two-way skull loss** is applied inside the floor (`SNR_echo` scaled by
`10^(−2·loss_dB/10)`), so the reported number is the *through-bone* floor. This is the
**derived replacement for the old hand-set `DEFAULT_FLOOR_M` scalar.**

## The budget and the binding denominator

`detection_budget(mech, acq, residual_clutter_m=…)` returns a `DetectionBudget` carrying
the surviving signal `Δz · survival · √N_ens`, the floor, the `snr_db`, and
`limiting_denominator` — **"echo_snr" vs "clutter"** — which the spec's Gate B asks us to
report (a clutter-limited result reframes the problem as engineering, not a physics wall).
The binding floor is the worse of the Walker–Trahey echo-SNR floor and the optional
post-clutter-filter residual.

## Wiring into the verdict

`run_neural_model(..., acquisition=AcquisitionParams)` opts in: the amplitude /
content-survival gates then score against the **derived** through-skull floor with the
integration gain folded into the signal (`gates.py` is unchanged — scoring against
`floor / √N_ens` is equivalent to amplifying the signal). With `acquisition=None` the
model stays strictly source-side and the scalar-floor path is untouched.

`run_motor_demo()` is the single call that *is* the demo's Gate A: M1 presets + sustained
imagery + `AcquisitionParams.demo_motor()`.

## The honest result

At the conservative baseline (30 dB free-field echo SNR, 24 dB two-way skull loss), the
motor source (~3.5 nm static, s≈0.81) lifts to **~220 nm surviving** after ×77
integration against a **~280 nm** through-skull floor — **SNR ≈ −2 dB, echo-SNR-limited.**
That is the spec's success criterion exactly: *"only several decibels below the detection
threshold… within one order of magnitude of detectability"* — an engineering-sized gap,
now produced from first principles rather than asserted. The un-integrated source alone
(~3.5 nm) sits far under the floor, which is precisely why the integration term is
load-bearing.

## The budget sweep — where is the wall, and which axis carries the verdict

[`budget.py`](../base_neural_model/forward/budget.py) turns the single Gate-A point into
a curve. Holding the source Δz fixed, `sweep_axis` varies one `AcquisitionParams` axis
and returns a `BudgetCurve` (SNR in dB, surviving Δz, floor, denominator at each point)
with a `crossing_value` — the axis value where SNR hits **0 dB** (signal == floor, the
detection wall).

`acoustic_axis_ranking` is the acoustic-side complement to the source-side Sobol collapse
(`model/sensitivity.py`, which collapses onto η and s): a one-at-a-time local sensitivity
ranking the four acquisition axes by their dB swing over their literature span. It leads
with **skull loss and epoch** — the two strategic levers (the wall, and the motor
integration advantage). At the demo baseline the wall sits at **~11 dB one-way (~22 dB
two-way) skull loss**, so the demo's 12 dB assumption lands just past it (−2 dB). SNR
falls a clean **−2 dB per one-way skull dB** (two-way × 20 log₁₀).

[`scripts/plot_budget.py`](../scripts/plot_budget.py) renders both: SNR-vs-skull-loss at
three echo-SNR levels with the wall and the demo point, plus the axis ranking.

## Tests

[`test_detection.py`](../tests/test_detection.py) pins the ×14/×77 integration gain, the
λ/4π prefactor and δφ(10 nm) spec cross-check, the floor's 1/√N_ens and 1/√SNR scaling,
its rise with skull dB, and the echo-SNR↔clutter denominator flip.
[`test_model_run.py`](../tests/test_model_run.py) pins the acquisition-wired verdict and
that the motor demo lands within an order of the floor.
[`test_budget.py`](../tests/test_budget.py) pins the −2 dB/dB skull slope, the 0 dB
crossing, the clutter-denominator flip along a sweep, and the axis ranking leading with
skull loss + epoch.
