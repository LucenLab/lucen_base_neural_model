# Mechanics layer — `base_neural_model/mechanics/`

**Status: ✅ implemented and tested. The deciding physics.**

The mechanics layer turns a **neural state** (synchrony, jitter, content corner —
produced by the [activity layer](activity.md)) into the **net axial tissue
displacement across the interrogation range gate, content-band**. The transduction
chain is the load-bearing physics; it can kill the project for the price of
arithmetic over a cited constant. The gates and inverse/global analyses that score it
live in the [model layer](model.md); they are documented here for continuity.

```
base_neural_model/mechanics/
  neuron_constants.py  the cited single-neuron Δr (Invariant 2)
  transduction.py      the transduction chain: activity → net axial dilatation
  eshelby.py           κ DERIVED from the matrix Poisson ratio (Eshelby/Mura)
  poroelastic.py       η as a SWEEP over the drainage number θ = C_v·t/L²
  derive.py            compose κ, η into a ready-to-sweep MechanicsParams
  timeseries.py        dz(t) from an ActivityTimeseries (deliverable a)
  spectrum.py          content-band spectrum of dz(t)
base_neural_model/model/
  gates.py             the three kill gates (Invariant 5)
  sensitivity.py       Sobol global SA — the central in-silico test (SALib)
  verdict_flip.py      per-factor verdict-flip table
```

## The central correction: it is not a sum

The naive "source term" sums single-neuron *membrane* displacements as collinear
scalars across N neurons. That is a **category error**. A firing cell's membrane
moves radially outward by ~1 nm; surface displacements of randomly-arranged,
roughly radially-expanding cells have **zero net translation** (the dipole moment
cancels). The surviving term is the **monopole — the volume change**. Summed as
translations the naive figure reaches tens of *micrometres* of net tissue motion,
three orders of magnitude too large and physically impossible.

The interrogation beam reads a **net axial dilatation across a resolution cell**,
not a sum of cellular excursions. The corrected chain runs through the volume
change at every link, each arrow a geometric factor below unity:

```
membrane Δr  ──3Δr/r──▶  per-cell ΔV/V  ──f_cell·s──▶  tissue strain ε_V
             ──η·κ·L──▶  net axial displacement Δz
```

## The transduction chain, link by link

Implemented in [`transduction.py`](../base_neural_model/mechanics/transduction.py) as small,
individually-testable helpers so each factor is contestable in isolation.

| Step | Relation | Helper |
|---|---|---|
| 2.1 membrane → per-cell volume | `ΔV/V = 3Δr/r` | `_fractional_volume_change` |
| 2.2 coherent tissue strain | `ε_V^coh = f_cell · s · (3Δr/r)` | `_volumetric_strain_coh` |
| 2.2 incoherent pedestal | `ε_V^incoh = f_cell · √((1−s)/N) · (3Δr/r)` | `_volumetric_strain_incoh` |
| 2.3 strain → axial displacement | `Δz = η · κ · L · ε_V` | `_axial_from_strain` |
| 3 jitter low-pass | `s(f_c) = exp(−2π²f_c²σ_t²)` | `_content_band_survival` |

Two factors carry the deepest source-side uncertainty:

- **κ — confinement** ∈ [1/3, 1]. The partition of volumetric strain into the
  axial direction. Now **derived from the matrix Poisson ratio ν** via the
  analytical Eshelby tensor (`eshelby.py`), not asserted free: κ = 1/3 at ν → 0
  (the free inclusion) and κ → 1 at ν → 1/2 (the incompressible fast-band matrix).
  See [κ from the Eshelby tensor](#κ-from-the-eshelby-tensor).
- **η — dilatation fraction** ∈ [0, 1]. **The load-bearing unknown.** What
  fraction of the per-cell volume change becomes *net tissue dilatation* (which
  the beam reads) versus *internal water redistribution* at conserved bulk volume
  (which it does not). Now **parameterized by the poroelastic drainage number**
  θ = C_v·t/L² (`poroelastic.py`) so it can be *swept* across the literature range —
  not to compute its true value. On the fast content timescale θ ≪ 1, drainage is
  incomplete and net dilatation is most suppressed (η → ~0.01); η = 1 is only the
  unconfined ceiling. η is empirical and cannot be closed by the model — it is
  routed to the Stage-1 bench measurement. See
  [η from the poroelastic drainage state](#η-from-the-poroelastic-drainage-state).

### Why N left the signal

In the corrected model the coherent strain depends on the **volume fraction**
f_cell, not the raw count N. N appears only in the incoherent pedestal, which
falls as 1/√N. So adding neurons (at fixed f_cell) leaves the signal unchanged
and *lowers the noise*. The old "more N → more signal" was an artifact of summing
the quantity that physically cancels. See [base.md](base.md) for the
`cell_volume_fraction` derived helper that reconciles N and f_cell.

## κ from the Eshelby tensor

[`eshelby.py`](../base_neural_model/mechanics/eshelby.py) derives κ from the matrix Poisson ratio
instead of accepting it free. For a dilatational (purely volumetric) eigenstrain in
a spherical inclusion in an isotropic matrix, the constrained *volumetric* strain is
the trace `S₁₁₁₁ + 2 S₁₁₂₂` of the Eshelby tensor (Mura formulation), with the boxed
sphere components:

```
S₁₁₁₁ = (7 − 5ν) / [15(1 − ν)]      S₁₁₂₂ = (5ν − 1) / [15(1 − ν)]
κ(ν)  = S₁₁₁₁ + 2 S₁₁₂₂ = (1 + ν) / [3(1 − ν)]
```

| ν | regime | κ |
|---|---|---|
| 0 | fully compressible matrix (free) | 1/3 |
| 0.3 | typical drained | ≈ 0.62 |
| → 0.5 | incompressible — the **undrained fast band** (Su et al. 2023, νᵤ > 0.49) | → 1 |

So the fast/content band, where the matrix is undrained and nearly incompressible,
is exactly where κ is driven toward the confined limit. ν = 1/2 is a `1/(1−ν)`
singularity and is guarded strictly below 0.5 (approached as a limit). A non-unit
inclusion `aspect_ratio` pulls κ from the sphere value toward the free 1/3 partition
by a monotone shape weight, staying in [1/3, 1]. **FEM cross-check:** a real
scikit-fem solve of a spherical inclusion with a dilatational eigenstrain
(`test_eshelby_kappa.py::test_fem_cross_check`) recovers the constrained dilatation
and agrees with the analytical κ.

## The directional channel — orientation coherence

The monopole (volume change) above has **no direction**, which is exactly why it
survives a randomly-arranged population: a pure dilatation cannot cancel. But a real
neuron does not expand as a perfect sphere — its eigenstrain has a **deviatoric**
(directional) part along the cell axis **û**:

```
ε*_ij = ε_iso·δ_ij + β·ε_vol·(û_i û_j − ⅓δ_ij)      ε_vol = 3Δr/r,  β ∈ [0,1]
```

Temporal **synchrony** `s` (do cells fire at the *same time*) and **directional
coherence** (do their axes **û** *point the same way* in 3-D) are **independent
axes** — firing synchronously does not mean firing along the same vector. The beam
reads the **axial** (ẑ) component of the population-averaged strain, so the deviatoric
term contributes only through `⟨(û·ẑ)² − ⅓⟩`. Summarized by a uniaxial **orientation
order parameter** `Q ∈ [0,1]` and the director projection `μ = (director·ẑ)`:

```
axial factor  g(Q,μ) = (2/3)·Q·P₂(μ)       P₂(x) = (3x²−1)/2
ε_dir,axial   = (S₁₁₁₁ − S₁₁₂₂)·β·ε_vol·g(Q,μ)
Δz_total      = |Δz_iso + Δz_dir|
```

| case | g | meaning |
|---|---|---|
| `Q = 0` (random) | 0 | the directional sum cancels → **recovers the volume-only model** |
| `Q = 1, μ = 1` (aligned to beam) | +2/3 | maximal directional survival (adds signal) |
| `Q = 1, μ = 0` (aligned across beam) | −1/3 | sideways expansion → negative axial term |

The deviatoric Eshelby response `S₁₁₁₁ − S₁₁₂₂ = (8 − 10ν)/[15(1−ν)]` is only weakly
ν-dependent (≈0.53 at ν=0, 0.4 at incompressibility) — so unlike the volumetric trace
κ, the **directional channel is not suppressed by the incompressible fast-band
matrix**, making it a genuinely non-redundant signal. Implemented in
[`orientation.py`](../base_neural_model/mechanics/orientation.py); pinned by
[`test_orientation.py`](../tests/test_orientation.py). It is **off by default**
(`anisotropy = 0`), so every isotropic construction reduces exactly to the
volume-change chain. The order parameter `Q` is a hardcoded input on
`MechanicsParams` for now (contestable in isolation, like η); the seam to have the
[activity layer](activity.md) *generate* directional alignment alongside temporal
synchrony is `NeuralState.orientation_coherence`.

## η from the poroelastic drainage state

[`poroelastic.py`](../base_neural_model/mechanics/poroelastic.py) parameterizes η by the
dimensionless drainage (consolidation) number — the quantity that decides whether
water can drain on the interrogation timescale:

```
θ = C_v · t / L²          τ ≈ L² / C_v          completeness = 1 − exp(−θ)
η = η_floor + (η_ceiling − η_floor) · completeness      (lifted by fast vascular/osmotic source)
```

The legitimate purpose is to **sweep** η across the literature range (permeability
k ∈ [1e-14, 1e-11] m⁴/(N·s) mapped through `C_v = k·M`), **not** to compute its true
value — the doc is explicit that no sweep can clear the project, only kill it.

- **θ ≪ 1** (fast band, short gate L): drainage incomplete → redistribution-dominated
  → η near the floor ~0.01 (Su et al. 2023 undrained incompressibility; Østby et al.
  2009 ECS-shrinkage).
- **θ ≳ 1**: drainage complete → η toward its ceiling.
- **η = 1** is the *unconfined optimistic ceiling* — what 2D culture in an infinite
  bath would wrongly measure. Because C_v is measured ~3 orders of magnitude away in
  time, the provenance flags any η > 0.1 as an extrapolation, not a measurement.

[`derive.py`](../base_neural_model/mechanics/derive.py) composes the two sub-models —
`build_mechanics_params(matrix=…, poro=…, …)` calls `eshelby_kappa` and `poroelastic_eta`
and returns a `MechanicsParams` ready for the unchanged `displacement_sweep`.

## The central in-silico test: Sobol collapse

[`sensitivity.py`](../base_neural_model/model/sensitivity.py) is the deliverable the doc names
"uncertainty compression": a variance-based global sensitivity analysis (SALib,
Saltelli sampling, `N = n·(2k+2)`) that **demonstrates** the verdict collapses onto η
and content-band synchrony s — rather than asserting it. The model output is the
Gate-1 amplitude proxy (jitter-surviving Δz; the verdict denominator while Modules 2/3
are stubs, injectable later).

Each factor's bound is its *literature uncertainty*: the measured-and-transferable
factors (matrix ν, the cited Δr, the geometry) are pinned to narrow ranges, while the
**unmeasured** η-driver (k, ~3 decades) and the swept s span their full ranges.
Variance therefore flows to the wide, unmeasured axes — the doc's central argument
made quantitative, not a claim of privileged partial derivatives in a multiplicative
chain. The total-order indices (`run_source_sobol`, pinned by
[`test_sobol_collapse.py`](../tests/test_sobol_collapse.py)):

| factor | role | ST (≈) |
|---|---|---|
| synchrony s | swept content-band axis | 0.65 |
| log10 permeability → η | unmeasured net-dilatation axis | 0.37 |
| matrix ν → κ | measured-transferable | < 0.01 |
| Δr, r, f_cell, N | cited / geometry | < 0.02 |

## The verdict-flip table

[`verdict_flip.py`](../base_neural_model/model/verdict_flip.py) makes the same point from the
other direction: holding all but one factor at the central-column nominal, which
factor can flip the verdict across the contrast bar? A per-factor bisection
(`find_flip`, on the per-factor-monotone proxy) over the literature ranges finds that
**only η (via permeability) and s** flip the verdict; the literature-pinned controls
cannot move it across the bar within their ranges. Pinned by
[`test_verdict_flip.py`](../tests/test_verdict_flip.py).

## The cited constant (Invariant 2)

[`neuron_constants.py`](../base_neural_model/mechanics/neuron_constants.py) holds
`get_single_neuron_displacement()` → a `NeuronDisplacement` of **2 nm**,
content-band, with a provenance string (optical AP membrane displacement,
nanometre scale) and the assumptions a reviewer would challenge to move it. This
figure enters the chain in exactly one place — `MechanicsParams.membrane_disp_m` —
and `mechanical_displacement` raises if the two disagree, so the constant can never
be silently overridden downstream.

## Public API

```python
from base_neural_model import (
    get_single_neuron_displacement,   # the cited Δr
    MechanicsParams,                     # κ, η, σ_t, f_cell, r, f_c  (re-exported)
    mechanical_displacement,             # one synchrony value → MechanicalDisplacement
    displacement_sweep,                    # the curve over s (the primary product)
    passes_stage1_gate,              # Gate 1 — amplitude
    passes_content_survival_gate,    # Gate 2 — content survival
    passes_dilatation_gate,          # Gate 3 — dilatation
    # κ derived from the matrix Poisson ratio (Eshelby/Mura)
    MatrixParams, eshelby_kappa, eshelby_sphere_components,
    # η swept over the poroelastic drainage number θ = C_v·t/L²
    PoroelasticParams, poroelastic_eta, eta_from_permeability, consolidation_number,
    build_mechanics_params,             # compose κ, η → MechanicsParams
    # the central in-silico test + its complement
    SobolProblem, SobolIndices, run_source_sobol,
    VerdictFlip, VerdictFlipTable, verdict_flip_table, gate1_predicate,
)
```

`MechanicsParams.central()` supplies the §5 central-column defaults (Δr=1.5 nm,
r=8 µm, f_cell=0.15, κ=0.5, η=0.5, σ_t=1 ms, f_c=100 Hz). The output
`MechanicalDisplacement` carries the full decomposition: `axial_displacement_m`
(Δz_coh, the deliverable), `volumetric_strain`, `incoherent_pedestal_m`,
`content_band_survival`, plus κ, η, σ_t for tracing. `value_m` aliases
`axial_displacement_m`.

## The three kill gates (Invariant 5)

All three must pass. Implemented in [`gate.py`](../base_neural_model/model/gates.py).

- **Gate 1 — amplitude** (`passes_stage1_gate`). Some plausible synchrony puts
  Δz_coh within ~1–2 orders of the unaberrated floor. Failure = the source
  collapses below detectability *before* the skull — the project wall, found
  cheapest.
- **Gate 2 — content survival** (`passes_content_survival_gate`). The
  jitter-surviving coherent signal stays above its own incoherent pedestal *and*
  above the displacement-estimate floor at some synchrony. Failure = content
  low-passed into the envelope: a signal that exists but carries no lexical
  information.
- **Gate 3 — dilatation** (`passes_dilatation_gate`). η is plausibly
  non-negligible. The model cannot *close* this gate; it makes the project's
  sensitivity to η visible and routes the verdict to Stage 1. A null Stage-1
  result falsifies the *displacement* readout specifically, leaving the
  stiffness-modulation observable as the fallback.

## The honest output: the §5 envelope

The deliverable is not one number but three columns. Pinned by
[`test_envelope_gates.py`](../tests/test_envelope_gates.py):

| | Δr | r | f_cell | κ | η | s | **Δz_coh** |
|---|---|---|---|---|---|---|---|
| pessimistic | 1.0 nm | 10 µm | 0.10 | 1/3 | 0.1 | 0.1 | **≈ 0.03 nm** |
| central | 1.5 nm | 8 µm | 0.15 | 1/2 | 0.5 | 0.4 | **≈ 2.5 nm** |
| optimistic | 3.0 nm | 6 µm | 0.25 | 1 | 1.0 | 0.8 | **≈ 90 nm** |

Against a ~1–10 nm unaberrated floor: optimistic clears it by 1–2 orders (the
fight is then the skull); central sits *at* the floor with no margin; pessimistic
is 2–3 orders *under* it — a wall, found for the cost of an arithmetic sweep. The
dominant killers are η, κ, and content-band s, exactly where a falsification-first
model wants its sensitivity to live.

## Invariants and tests

| Invariant | How the mechanics layer honors it | Test |
|---|---|---|
| 2 — cited constant | Δr enters only the volume relation; equality guard | [`test_coherence_limits.py`](../tests/test_coherence_limits.py) |
| 3 — band guard | `require_content_fast` at every entry point | [`test_bands.py`](../tests/test_bands.py) |
| 4 — swept synchrony | `displacement_sweep` returns a curve | [`test_coherence_limits.py`](../tests/test_coherence_limits.py) |
| 5 — kill criterion | three `passes_*_gate` functions | [`test_envelope_gates.py`](../tests/test_envelope_gates.py) |
| 6 — provenance | chain assumptions (η, κ, σ_t) appended | [`test_provenance.py`](../tests/test_provenance.py) |

Corrected coherence limits (replacing the old N·d₁ / √N·d₁): at `s=1`,
`ε_V = f_cell·(3Δr/r)`; at `s=0`, the coherent term vanishes and a 1/√N pedestal
remains (noise, not signal). A band-survival invariant asserts the content-band
quantity is computed under an explicit σ_t, so the envelope can never substitute
for content.
