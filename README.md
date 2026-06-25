# Lucen — Forward Model

A three-module forward simulation that computes **one number**: the through-skull,
content-band voxel displacement **contrast** versus the post-clutter noise floor — as
a *curve* over neural synchrony, with the crossover synchrony named.

Everything in this repo exists to produce that number honestly and to localize
*which* term kills it if it dies. See [`lucen_forward_model_spec.md`](lucen_forward_model_spec.md)
for the full technical specification.

## Design invariants (non-negotiable)

1. **SI units everywhere.** Metres, Hz, seconds, Pa, kg/m³, m/s. Convert to
   human-readable units only at the reporting boundary ([`lucen/base/units.py`](lucen/base/units.py)).
   A single nm/µm slip moves the verdict by three orders of magnitude.
2. **Single-neuron displacement is a cited constant**, never simulated
   ([`lucen/source/neuron_constants.py`](lucen/source/neuron_constants.py)).
3. **The fast/content band is the target, never the slow envelope.** Enforced as a
   typed guard, not a comment ([`lucen/base/bands.py`](lucen/base/bands.py)).
4. **Synchrony is the swept independent variable.** The output is always a curve.
5. **Every stage scores against a kill criterion** (`passes_*_gate`), not just a plot.
6. **Provenance propagates** ([`lucen/base/provenance.py`](lucen/base/provenance.py)) so the
   final number decomposes back to its inputs.

## Layout

```text
lucen/
  base/          SI units, temporal bands, provenance, the typed data contracts
  source/        Module 1 — population summation (the deciding build) [IMPLEMENTED]
  propagation/   Module 2 — acoustic propagation through skull (k-Wave) [SCAFFOLD]
  detection/     Module 3 — detection and the contrast verdict        [SCAFFOLD]
  orchestration/ wires M1→M2→M3 and runs the synchrony sweep          [SCAFFOLD]
tests/           unit-consistency + invariant tests (spec §7)
```

`base/` and Module 1 (`source/`) are implemented and tested. Modules 2–3 and the
orchestration are typed stubs — every function has its signature, types, docstring,
and a `NotImplementedError` body marking the implementation work.

## Build order (the falsification discipline — spec §8)

Each step can kill the project for less compute than the step after it.

1. `base/` — units, types, bands, provenance. ✅
2. **Module 1 + its gate** — the synchrony sweep against an assumed unaberrated
   floor, *before touching k-Wave*. Pure arithmetic; the cheapest falsification. ✅
3. **Module 3** against synthetic propagated inputs (validate the detection chain
   and clutter filter in isolation).
4. **Module 2** on k-Wave — `method="none"` baseline, then minimal correction, then
   the corrected-vs-uncorrected recovery factor.
5. **Orchestration** — wire the full sweep, produce the feasibility curve, run
   Stages 0–2.
6. Only if favorable: the deferred metamaterial inverse-design correction (spec §4.4).

## Setup

This project is managed with [uv](https://docs.astral.sh/uv/). The Python version is
pinned in [`.python-version`](.python-version) and the exact dependency set is locked in
[`uv.lock`](uv.lock) — both are committed, so every checkout resolves identically.

```bash
uv sync                       # create .venv + install lucen + dev tools (pytest, ruff)
uv sync --extra propagation   # + k-wave-python (Module 2 only; heavy)
uv sync --no-dev              # runtime deps only, no pytest/ruff
```

`uv sync` creates and maintains `.venv` for you — no manual `venv`/`activate` step.
The `dev` group (pytest, ruff) installs by default; `propagation` is an **optional
extra**, imported lazily inside [`lucen/propagation/medium.py`](lucen/propagation/medium.py),
so everything except Module 2 runs without `k-wave-python`.

> Editing dependencies: `uv add <pkg>` / `uv remove <pkg>` updates `pyproject.toml`
> and `uv.lock` together. After changing `pyproject.toml` by hand, run `uv lock` to
> refresh the lockfile.

## Running

Prefix commands with `uv run` to execute inside the project environment (it auto-syncs
first), or activate `.venv` and run them directly.

```bash
uv run pytest               # invariant tests (spec §7); stub-dependent tests xfail/skip
uv run ruff check .         # lint
```

Module 1 is usable directly:

```python
import numpy as np
from lucen.source import get_single_neuron_displacement, synchrony_sweep, passes_stage1_gate
from lucen.base.types import VoxelGeometry

d1 = get_single_neuron_displacement()                     # cited constant + provenance
geom = VoxelGeometry(extent_axial_m=3e-4, extent_lateral_m=1e-3,
                     neuron_count=10_000, depth_m=2e-2)
sweep = synchrony_sweep(d1, geom, np.linspace(0, 1, 51))  # the source-term curve
passes_stage1_gate(sweep, unaberrated_floor_m=1e-7)       # the Stage-1 kill criterion
```

## What this build does and does not establish

**Establishes:** whether, given the cited single-neuron displacement and a
parameterized synchrony, the content-band voxel contrast survives skull attenuation
and aberration to a separability bar after bulk-motion filtering — as a curve over
synchrony, decomposable to its inputs, with the crossover point named.

**Does not establish:** the true synchrony of speech cortex (imported as a sweep, not
measured), single-trial real-time decode, or the crossed-beam upgrade. The simulation
replaces the prior with a number; it does not by itself prove real-time millimetre
speech decoding.
