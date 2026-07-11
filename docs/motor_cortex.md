# Motor cortex (M1) — the directional variant

The model can be aimed at **primary motor cortex (M1)**, where the neural-displacement
question is most concrete: M1 is the site of the largest, most strongly aligned
cortical output cells (layer-5 pyramidal / Betz cells) and a movement-locked beta
rhythm — exactly the conditions under which the **directional channel** (orientation
coherence) matters.

```python
from base_neural_model import run_motor_cortex
r = run_motor_cortex()
r.neural_state.content_freq_hz          # ~19 Hz (beta), not gamma
r.neural_state.orientation_coherence    # Q_eff, emerged from columnar alignment
r.mechanical_displacement.directional_axial_m   # the directional signal share
```

## What makes M1 different (and why it changes the numbers)

| Feature | Generic column | Motor cortex (M1) |
|---|---|---|
| Rhythm | gamma (~28 Hz) | **beta (~19 Hz)** — the sensorimotor rhythm |
| Cell axes | isotropic (random) | **strongly columnar** (`Q_struct ≈ 0.9`) |
| Cell size | ~8 µm | **~20 µm** large layer-5 / Betz somata (sparse) |
| Directional channel | inert | **active** (~13% of the signal) |
| Voxel depth | generic 2 cm | **M1 layer 5**, ~1.8 mm |

The three presets:

- **`EIParams.motor_cortex()`** — slower membrane time constants place the E/I limit
  cycle in the **beta band**; `structural_alignment ≈ 0.9` marks the strongly
  columnar layer-5 population. That structural alignment, expressed through the
  temporal synchrony, *generates* the orientation coherence (the Option-2 path):
  `Q_eff = Q_struct · s`.
- **`MechanicsParams.motor_cortex()`** — large (`r ≈ 20 µm`) Betz-cell somata and a
  substantial per-cell **anisotropy** `β ≈ 0.6` (elongated, columnar cells), with the
  beam interrogating roughly along the cortical column (`mean_axis_projection ≈ 1`).
  The content corner is in the beta band.
- **`VoxelGeometry.motor_cortex_layer5()`** — the voxel at M1 layer-5 depth, with a
  neuron count set **consistent with the large somata** (~1300, not the generic 10k —
  large output cells are sparse; the generic count would imply a volume fraction
  above 1).

## The directional channel is the point

In a generic isotropic patch the directional term is zero — a randomly-oriented
population's vector displacements cancel, leaving only the volume-change monopole. M1
is where that assumption breaks: the columnar layer-5 cells share an axis, so when
they co-fire in the beta rhythm their directional contributions **add** along the
beam. The model now sources that orientation coherence from the activity dynamics
(`activity/orientation.py` → `reduce_to_state` → `NeuralState.orientation_coherence`)
rather than hardcoding it, so temporal synchrony and directional alignment emerge
together from the simulated population.

See [mechanics.md](mechanics.md) for the directional-channel physics
(`g(Q,μ) = (2/3)·Q·P₂(μ)`, the deviatoric Eshelby response) and [activity.md](activity.md)
for how `Q_eff` emerges from the structural alignment.

## Movement trials — the dynamics

`run_motor_cortex()` holds M1 on a steady resting-beta cycle. `run_motor_trial()`
imposes a **movement-locked drive** `P(t)` (`activity/motor_drive.py`): a baseline with
a dip at movement onset and an overshoot afterwards, giving the M1 motor signature —

```
resting beta  →  movement onset: beta SUPPRESSED (synchrony desyncs)  →  beta REBOUND
```

Because the displacement `dz(t)` tracks synchrony, the **tissue displacement signal
itself** shows the arc: it drops during movement and rebounds above baseline after.

```python
from base_neural_model import run_motor_trial, MovementProfile
r = run_motor_trial(MovementProfile(onset_time_s=0.4, move_duration_s=0.3))
# r.displacement_timeseries.surviving_dz  ->  baseline, desync dip, rebound
```

`run_motor_trial` is a **dynamics visualizer**: the deliverable is the timeseries above
(and the directional channel it carries), not a detectability verdict. Its reduced
`NeuralState` is a whole-record *average* — needed only to stamp the content-band
low-pass onto `dz(t)` — so the report's pass/fail gate fields describe a steady state the
event never occupied and are **not** meaningful here. For a detectability verdict use the
steady-state entry points (`run_motor_cortex`, `run_motor_demo`).

A modeling note: in this mean-field reduction, "synchrony" is the rhythm envelope
gated by the population's engaged activation (`activity/timeseries.py`), so the drive
dip that collapses activation is what desynchronizes it. That is a deliberate, stated
proxy — a single 2-D oscillator does not have true population phase desync — chosen so
the desync → rebound *sequence* is faithful even though the microscopic mechanism is
abstracted. Pinned by [`test_motor_dynamics.py`](../tests/test_motor_dynamics.py).

## Honest caveats

- The total M1 displacement comes out **lower** than the generic model, dominated by
  the larger cell radius (`3Δr/r` shrinks with `r`). This is a real consequence of the
  Betz-cell size assumption, not the directional channel — and it is exactly the kind
  of parameter trade-off the model is built to make visible rather than hide.
- The `cell_radius_m ≈ 20 µm` in `MechanicsParams.motor_cortex()` sits **below** the
  true giant-Betz stereological size: modern morphometry puts the Betz soma at a
  mean cell-body volume of ~86,700 µm³ (equivalent-sphere radius ~27 µm), with long
  axes of 60–120 µm (Betz's own 60×120 µm², Brodmann's 53×106 µm²). The preset's 20 µm
  is best read as a *large layer-5 pyramidal output cell* (ordinary L5 somata are
  ~10–12 µm radius; true Betz cells are rarer — only ~10% of layer-Vb pyramids, ~125k
  per hemisphere), not the giant Betz extreme. This is conservative for the headline:
  because coherent displacement scales as `3Δr/r`, raising `r` toward the true-Betz
  ~27 µm would *shrink* the direct beta term further under the floor, and the ~15%
  volume fraction only requires proportionally fewer cells (changing the incoherent
  √(1/N) pedestal, not the verdict, which collapses onto η and s). The shared
  `cell_volume_fraction ≈ 0.15` itself is well-supported: cortical cell-body volume
  fraction is ~10–40% of gray matter (neuronal soma somewhat below the all-cell figure,
  which includes glia).
- `β`, `Q_struct`, and the beam/column alignment `μ` are literature-motivated but
  **unmeasured** at the content band — like η, they are contestable inputs the model
  lets you sweep, not settled values.
- The `motor_cortex()` preset now carries the **honest source-physics factors** (S1/S3/S6/S7:
  viscoelastic transfer, carrier-modulation depth, coherent fraction, saturation) — see
  [mechanics.md](mechanics.md). Combined with the sub-nm cited Δr, the direct M1 beta term is
  ~0.1 nm.
- `run_motor_demo()` is the honest Gate A: it drives M1 with a **bursty** beta rhythm
  (`bursty_beta_drive` — beta is transient, not sustained, even under held demand) and scores
  the **in-burst** synchrony (p90), so the burst-intermittency penalty is charged once, on the
  acoustic side (the coherence-window cap), not twice. It returns both the content-band verdict
  (`report.detection`, ≈ −88 dB) and the band-separated mechanism decomposition
  (`report.mechanisms`: the large hemodynamic/fUS envelope vs the tiny beta carrier).

## Tests

[`test_motor_cortex.py`](../tests/test_motor_cortex.py) pins the beta-band rhythm, the
columnar alignment, the volume-fraction consistency of the sparse large cells, the
layer-5 depth, and that the directional channel contributes in M1 while staying inert
in the generic model.
