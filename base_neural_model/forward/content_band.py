r"""Inverse optimization of the content-band definition for detectability.

The content band is the frequency window ``[f_lo, f_hi]`` the mechanics treat as the
information-bearing carrier: the harmonics inside it are coherently summed, each
attenuated by its own firing-jitter low-pass, and the result is what must clear the
through-skull detection floor. The shipped boundary is a single 16 Hz lower edge
borrowed from the *speech* envelope/content distinction -- the wrong abstraction for a
beta **motor** rhythm, where the information IS the beta oscillation. This module
derives the band from the detection physics instead of asserting it.

This optimizer assumes the **corrected jitter model**
(:func:`base_neural_model.activity.jitter.total_jitter`): firing jitter has an absolute
floor independent of synchrony, so a harmonic at frequency ``f`` survives as
``exp(-2 pi^2 f^2 sigma_t^2)`` with ``sigma_t`` NOT shrinking to cancel ``f``. Under
the older synchrony-only jitter the low-pass collapsed to ``synchrony`` at every
frequency and the band was inert; the floor is what makes ``[f_lo, f_hi]`` matter.

The objective is **detectability** -- the jitter-weighted in-band signal over the
floor -- not raw captured power. The surviving in-band factor is::

    G(f_lo, f_hi) = sum_k  a_k * exp(-2 pi^2 f_k^2 sigma_t^2)

over the FFT bins ``f_k`` in ``[f_lo, f_hi]``, where ``a_k = A_k / sum_j A_j`` is the
in-band amplitude share (``A_k = sqrt(power_k)``). Normalizing the weights to sum to 1
is deliberate: the coherent displacement *amplitude* is set by synchrony (how many
cells swell in phase), which the band does not change; the band only chooses HOW that
fixed amplitude is distributed across surviving vs. jitter-killed harmonics. So ``G``
is a weighted-average survival, bounded in ``[0, 1]``, and widening the band helps only
by shifting weight onto better-surviving (lower-frequency) harmonics.

Because ``exp(-2 pi^2 f^2 sigma_t^2)`` is monotonically gentler at low ``f``, ``G`` is
maximized by admitting the lowest-frequency content available. Left unconstrained that
walks down to the slow envelope fundamental -- which is not a content carrier. The
binding constraint is therefore **physical admissibility**: the band must lie in the
sensorimotor beta range (:data:`BETA_LO_HZ` .. :data:`BETA_HI_HZ`). The optimum is thus
constraint-bound (it sits at the bottom of the admissible beta window and as wide as
the rhythmic power extends), which is itself the finding: for a jitter-limited motor
readout you want the *lowest beta-band* content, not the highest.

Because the free optimum is degenerate (it always walks to the lowest admissible
frequency and reads leakage, not the rhythm), the band you actually **report** is
:func:`band_on_fundamental` -- pinned to the rhythm's own dominant peak, so "detect
high-beta" means detecting the beta carrier at its own frequency. The free searches
below are kept to expose and price that degeneracy, not as the recommended band.

Functions:

* :func:`band_on_fundamental` -- the content band pinned to the rhythm's fundamental
  (the beta carrier). **This is the neuroscientifically meaningful band to report.**
* :func:`optimize_individual_bands` -- the **1-D** free optimum per preset: the
  SNR-maximizing band. A *diagnostic* -- it always sits at the bottom of the beta box
  (jitter prefers low frequency), showing how much SNR the rhythm "leaves on the table"
  by insisting on being read at its own frequency.
* :func:`optimize_shared_band` -- the **2-D** free optimum: one band serving BOTH
  presets, maximizing the *worst* preset's SNR (maximin). Same degeneracy caveat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from base_neural_model.activity.jitter import ABSOLUTE_JITTER_FLOOR_S, total_jitter
from base_neural_model.activity.timeseries import ActivityTimeseries
from base_neural_model.forward.detection import AcquisitionParams, phase_displacement_floor

# Physical admissibility for a motor content carrier: the sensorimotor beta band. Below
# BETA_LO_HZ is the mu-rhythm / envelope (not a content carrier); above BETA_HI_HZ the
# beta harmonics carry negligible power. The optimizer may place the window anywhere in
# this box but cannot collapse onto the ~6-7 Hz envelope fundamental (< BETA_LO_HZ).
BETA_LO_HZ: float = 13.0    # sensorimotor beta lower edge (Kilavik et al. 2013)
BETA_HI_HZ: float = 30.0    # sensorimotor beta upper edge
MIN_BANDWIDTH_HZ: float = 3.0  # a band narrower than this is a line, not a band


@dataclass(frozen=True)
class BandResult:
    """An optimized content band and the detectability it yields."""

    f_lo_hz: float
    f_hi_hz: float
    f_c_hz: float             # dominant (peak-power) frequency inside the band
    content_fraction: float   # in-band power / total non-DC power
    inband_survival: float    # G: jitter-weighted in-band survival factor in [0, 1]
    snr_db: float             # detection SNR this band produces
    surviving_dz_m: float
    floor_m: float

    @property
    def detectable(self) -> bool:
        return self.snr_db >= 0.0


def _inband(
    ts: ActivityTimeseries, f_lo: float, f_hi: float
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """(in-band freqs, in-band amplitude shares a_k, content_fraction) or None if empty."""
    f = ts.spectrum.freqs_hz
    p = ts.spectrum.power
    nz = f > 0.0
    total = float(p[nz].sum())
    band = nz & (f >= f_lo) & (f <= f_hi)
    if not band.any() or p[band].sum() <= 0.0 or total <= 0.0:
        return None
    fk = f[band]
    amp = np.sqrt(p[band])
    a_k = amp / amp.sum()
    frac = float(p[band].sum()) / total
    return fk, a_k, frac


def inband_survival(
    ts: ActivityTimeseries,
    f_lo: float,
    f_hi: float,
    *,
    floor_s: float = ABSOLUTE_JITTER_FLOOR_S,
) -> float:
    r"""Jitter-weighted in-band survival ``G = sum_k a_k exp(-2 pi^2 f_k^2 sigma_t^2)``.

    ``sigma_t`` is the corrected total jitter (:func:`total_jitter`) at each harmonic --
    the synchrony phase-spread plus the absolute floor -- so ``G`` genuinely depends on
    where the band sits. Returns 0.0 for an empty band.
    """
    got = _inband(ts, f_lo, f_hi)
    if got is None:
        return 0.0
    fk, a_k, _ = got
    s = ts.mean_synchrony
    surv_k = np.array(
        [np.exp(-2.0 * np.pi**2 * fh**2 * total_jitter(s, fh, floor_s=floor_s) ** 2)
         for fh in fk]
    )
    return float((a_k * surv_k).sum())


def band_snr_db(
    ts: ActivityTimeseries,
    acq: AcquisitionParams,
    f_lo: float,
    f_hi: float,
    *,
    coherent_dz_m: float,
    floor_s: float = ABSOLUTE_JITTER_FLOOR_S,
) -> float:
    """Detection SNR (dB) if ``[f_lo, f_hi]`` were the content band.

    ``surviving = coherent_dz * G(f_lo, f_hi) * sqrt(N_ens)`` over the Walker-Trahey
    through-skull floor. ``coherent_dz_m`` is the pre-jitter coherent displacement,
    set by synchrony alone (band-invariant). Returns ``-inf`` for an empty band so
    infeasible windows are strictly dominated.
    """
    g = inband_survival(ts, f_lo, f_hi, floor_s=floor_s)
    if g <= 0.0:
        return float("-inf")
    surviving = coherent_dz_m * g * acq.integration_gain
    floor = phase_displacement_floor(acq)
    if floor <= 0.0 or surviving <= 0.0:
        return float("-inf")
    return 20.0 * np.log10(surviving / floor)


def _make_result(
    ts: ActivityTimeseries,
    acq: AcquisitionParams,
    f_lo: float,
    f_hi: float,
    *,
    coherent_dz_m: float,
    floor_s: float,
) -> BandResult:
    got = _inband(ts, f_lo, f_hi)
    if got is None:
        f_c, frac = float("nan"), 0.0
    else:
        fk, _, frac = got
        pk = ts.spectrum.power[(ts.spectrum.freqs_hz >= f_lo) & (ts.spectrum.freqs_hz <= f_hi) & (ts.spectrum.freqs_hz > 0)]
        f_c = float(fk[np.argmax(pk)])
    g = inband_survival(ts, f_lo, f_hi, floor_s=floor_s)
    surviving = coherent_dz_m * g * acq.integration_gain
    floor = phase_displacement_floor(acq)
    snr = band_snr_db(ts, acq, f_lo, f_hi, coherent_dz_m=coherent_dz_m, floor_s=floor_s)
    return BandResult(
        f_lo_hz=f_lo, f_hi_hz=f_hi, f_c_hz=f_c, content_fraction=frac,
        inband_survival=g, snr_db=snr, surviving_dz_m=surviving, floor_m=floor,
    )


def _admissible_edges(
    ts: ActivityTimeseries, *, step_hz: float | None
) -> np.ndarray:
    """Candidate band edges on the run's own FFT grid, within the beta box."""
    f = ts.spectrum.freqs_hz
    step = step_hz if step_hz is not None else float(f[1] - f[0])
    grid = f[(f >= BETA_LO_HZ) & (f <= BETA_HI_HZ) & (f > 0)]
    if grid.size == 0:
        grid = np.arange(BETA_LO_HZ, BETA_HI_HZ + step, step)
    return grid


def _search(
    snr_of_window,
    edges: np.ndarray,
) -> tuple[float, float, float]:
    """Grid-search ``[f_lo, f_hi]`` over admissible edges; return (best_lo, hi, value).

    Exhaustive over the FFT grid (exact to resolution) -- robust against the empty-band
    cliffs and the flat-in-f_c degeneracy that trip gradient methods.
    """
    best = (float("nan"), float("nan"), float("-inf"))
    for lo in edges:
        for hi in edges:
            if hi - lo < MIN_BANDWIDTH_HZ:
                continue
            v = snr_of_window(float(lo), float(hi))
            if v > best[2]:
                best = (float(lo), float(hi), v)
    return best


def optimize_individual_bands(
    ts: ActivityTimeseries,
    acq: AcquisitionParams,
    *,
    coherent_dz_m: float,
    floor_s: float = ABSOLUTE_JITTER_FLOOR_S,
    step_hz: float | None = None,
) -> BandResult:
    r"""1-D problem: the single best content band ``[f_lo, f_hi]`` for ONE preset.

    Exhaustively searches admissible beta-box windows on the run's FFT grid and returns
    the one whose detection SNR is largest. The per-rhythm ceiling.
    """
    edges = _admissible_edges(ts, step_hz=step_hz)
    lo, hi, _ = _search(
        lambda a, b: band_snr_db(ts, acq, a, b, coherent_dz_m=coherent_dz_m, floor_s=floor_s),
        edges,
    )
    if not np.isfinite(lo):
        raise ValueError("no admissible band found in the beta box")
    return _make_result(ts, acq, lo, hi, coherent_dz_m=coherent_dz_m, floor_s=floor_s)


def band_on_fundamental(
    ts: ActivityTimeseries,
    acq: AcquisitionParams,
    *,
    coherent_dz_m: float,
    half_width_hz: float = 3.0,
    floor_s: float = ABSOLUTE_JITTER_FLOOR_S,
) -> BandResult:
    r"""The content band **pinned to the rhythm's own fundamental** -- the one to report.

    This is deliberately NOT the free SNR-maximizing band. Because the jitter low-pass
    ``exp(-2 pi^2 f^2 sigma_t^2)`` is monotonically gentler at low ``f``,
    :func:`optimize_individual_bands` always walks down to the bottom of the admissible
    beta box and reads whatever low-frequency spectral leakage survives best -- an
    SNR-optimal but neuroscientifically meaningless answer (it is not reading the beta
    rhythm, just the lowest crumb the constraint allows). The free optimum is therefore
    kept only as a *diagnostic* of that degeneracy.

    The physically meaningful content band sits ON the rhythm: a narrow window of
    ``+/- half_width_hz`` around the dominant oscillatory peak (the fundamental, which
    for a beta rhythm carries ~three-quarters of the power). This function scores
    detectability there, so "detect high-beta" means "detect the beta carrier at its own
    frequency". Report this; use :func:`optimize_individual_bands` to show the gap to the
    (degenerate) free optimum.
    """
    f_c = ts.spectrum.dominant_freq_hz
    lo = max(0.0, f_c - half_width_hz)
    hi = f_c + half_width_hz
    return _make_result(ts, acq, lo, hi, coherent_dz_m=coherent_dz_m, floor_s=floor_s)


@dataclass(frozen=True)
class SharedBandResult:
    """One content band shared across presets, with each preset's realized metrics."""

    f_lo_hz: float
    f_hi_hz: float
    worst_snr_db: float                # the maximin objective value
    per_preset: dict[str, BandResult]  # each preset's metrics at the shared band


def optimize_shared_band(
    runs: dict[str, ActivityTimeseries],
    acq: AcquisitionParams,
    *,
    coherent_dz_m: dict[str, float],
    floor_s: float = ABSOLUTE_JITTER_FLOOR_S,
    step_hz: float | None = None,
) -> SharedBandResult:
    r"""2-D problem: ONE band ``[f_lo, f_hi]`` serving several presets (maximin).

    Picks the window maximizing the **worst** preset's detection SNR::

        argmax_{f_lo < f_hi}  min_k  SNR_k(f_lo, f_hi)

    The maximin (not a mean) is deliberate: a shared content-band definition is only as
    trustworthy as the rhythm it serves worst. ``runs`` and ``coherent_dz_m`` are keyed
    by preset name; runs should share an FFT grid so candidate edges align.
    """
    if not runs:
        raise ValueError("need at least one preset run")
    names = list(runs)
    ref = runs[names[0]]
    edges = _admissible_edges(ref, step_hz=step_hz)

    def worst(a: float, b: float) -> float:
        return min(
            band_snr_db(runs[k], acq, a, b, coherent_dz_m=coherent_dz_m[k], floor_s=floor_s)
            for k in names
        )

    lo, hi, val = _search(worst, edges)
    if not np.isfinite(lo):
        raise ValueError("no admissible shared band found in the beta box")
    per = {
        k: _make_result(runs[k], acq, lo, hi, coherent_dz_m=coherent_dz_m[k], floor_s=floor_s)
        for k in names
    }
    return SharedBandResult(f_lo_hz=lo, f_hi_hz=hi, worst_snr_db=val, per_preset=per)
