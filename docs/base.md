# Foundation — `base_neural_model/base/`

**Status: ✅ implemented and tested.**

The foundation is the shared vocabulary the activity and mechanics layers speak.
Nothing computes correctly without it: SI unit handling, the temporal-band guard,
provenance propagation, and the typed data contracts that cross the
activity → mechanics → gates boundary. Get these right and the layers can be built
independently; get them wrong and a units slip or an envelope/content swap silently
flips the verdict.

```
base_neural_model/base/
  units.py       SI scale factors, sane bounds, reporting converters, db20
  bands.py       Band enum + require_content_fast guard
  provenance.py  Provenance record + extend / merge
  types.py       the typed contracts crossing every module boundary
```

## `units.py` — SI and the reporting boundary (Invariant 1)

Internally *every* quantity is SI. Conversion to a human-readable unit happens
only here, and the result must never flow back into a computation.

- **`is_sane_displacement_m(value_m) -> bool`** — the cheap tripwire. True iff the
  value is finite and within `DISPLACEMENT_MIN_M` (1 pm) … `DISPLACEMENT_MAX_M`
  (10 mm). Sub-picometre or super-centimetre "displacements" in this pipeline are
  always a units bug, never real. The unit-consistency test asserts this on every
  quantity crossing a boundary.
- **`m_to_nm` / `m_to_um` / `m_to_mm`** — reporting-boundary converters only.
- **`db20(ratio) -> float`** — `20·log10(ratio)`, the contrast convention
  (displacement is an amplitude). Raises on a non-positive ratio.

## `bands.py` — the envelope-vs-content guard (Invariant 3)

Every displacement carries a temporal band. The model must never let a
slow-envelope (rhythm/timing — *when* speech happens) figure stand in for a
fast-content (*what* the speech is) figure. That category error is made a raised
exception, not a comment.

- **`Band`** enum: `CONTENT_FAST` (the target), `ENVELOPE_SLOW` (timing only),
  `BROADBAND` (unseparated; disallowed past the activity reduction).
- **`require_content_fast(band, *, context=...)`** — raises `BandError` unless the
  band is `CONTENT_FAST`. Called at every entry point that must operate on the
  content band (the cited constant, the source chain). `context` names the
  offending boundary in the message.

## `provenance.py` — decomposability (Invariant 6)

Provenance travels with every effective quantity so the final contrast number is
traceable back to its inputs: a reviewer can contest a single named assumption
without anything being buried.

- **`Provenance`** (frozen): `source` (literature citation), `assumptions`
  (ordered tuple), `band`. The original `source` is preserved to the end of the
  chain.
- **`extend(prov, *added, band=None)`** — what a *transforming* module calls: append
  the assumptions it introduced, optionally re-label the band.
- **`merge(*provs, source=None)`** — what a module *combining* inputs calls: concat
  + de-dup assumptions, and **require all inputs share a band** — a band mismatch
  here means an envelope quantity leaked into a content path, and is raised.

## `types.py` — the data contracts

These are the typed objects that cross module boundaries — the actual interface.
Two structural conventions:

- Scalar-only contracts are `frozen=True` (immutable, value-equal, hashable).
- Contracts holding NumPy arrays (`ActivityTimeseries`, `DisplacementTimeseries`,
  the spectra) are `frozen=True, eq=False`, because a generated `__eq__` over an
  `ndarray` is ambiguous (`arr == arr` is array-valued) and unhashable. They keep
  identity equality, and they live in their owning modules (`activity.timeseries`,
  `mechanics.timeseries`/`spectrum`) rather than in `base.types`.

### Inputs

| Type | Role |
|---|---|
| `NeuronDisplacement` | the cited single-neuron Δr (Invariant 2); `band` must be `CONTENT_FAST` |
| `VoxelGeometry` | the voxel: axial/lateral extents, `neuron_count` N, `depth_m` (path length) |
| `MechanicsParams` | cell/material chain inputs: Δr, r, f_cell, κ, η, σ_t, f_c |

### The hand-offs

| Type | Produced by → consumed by |
|---|---|
| `NeuralState` | activity (reduce) → mechanics (the s, σ_t, f_c, rate bridge) |
| `MechanicalDisplacement` | the transduction chain's per-state deliverable |

### `VoxelGeometry` derived helpers

Because `neuron_count` and `MechanicsParams.cell_volume_fraction` are partly
redundant (the coherent strain depends on f_cell; N sets only the incoherent
pedestal), the geometry exposes the bridge between them:

- **`volume_m3`** — voxel volume as `axial × lateral²` (the doc's 0.3×1×1 mm
  voxel = 3e-10 m³). `depth_m` is the propagation path length, **not** a third
  voxel extent, so it is deliberately excluded.
- **`cell_volume_fraction(cell_radius_m)`** — `f_cell = N·V_cell / V_voxel` implied
  by the geometry. Lets a caller derive f_cell instead of asserting it.

And the opt-in reconciler on `MechanicsParams`:

- **`check_volume_fraction_consistency(geom, *, rel_tol=0.1)`** — raises if the
  asserted f_cell disagrees with the geometry-derived value beyond `rel_tol`. It
  is **opt-in, not enforced**, so the two stay independently contestable; callers
  who want the guardrail call it explicitly.

## Invariants this subsystem owns

| Invariant | Mechanism | Test |
|---|---|---|
| 1 — SI units | `is_sane_displacement_m`, reporting-only converters | [`test_units.py`](../tests/test_units.py) |
| 3 — band guard | `Band` + `require_content_fast` / `BandError` | [`test_bands.py`](../tests/test_bands.py) |
| 6 — provenance | `Provenance` + `extend` / `merge` | [`test_provenance.py`](../tests/test_provenance.py) |
