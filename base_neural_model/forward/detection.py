r"""Acoustic detection layer: source displacement -> echo-phase detectability.

This is the FIRST module that crosses the source/sensing boundary the rest of
``base_neural_model`` deliberately stops at. It takes the net axial source
displacement Delta z that the mechanics chain produces (a
:class:`~base_neural_model.base.types.MechanicalDisplacement`) and asks the Gate-A /
Stage-1 question the Lucen specs make the cheapest falsification: **does the
content-bearing displacement, after within-epoch coherent integration, clear the
phase-sensitive ultrasound detection floor through the skull?**

Nothing here recomputes source physics. Detectability is a single ratio -- the bare
surviving source displacement over the noise floor on estimating it -- and ALL of the
receive-side coherent gain lives in that floor, in exactly two orthogonal places so no
gain is ever counted twice:

* **Spatial: receive beamforming across the array (sqrt(n_elements)).** Delay-and-sum
  over ``n_elements`` per-element echoes (spec 4.2) coherently sums the signal (~N) and
  incoherently sums the noise (~sqrt(N)), so the beamformed *voxel* echo SNR is
  ``n_elements`` times the per-element echo SNR. Through an *uncorrected skull* only a
  fraction ``aperture_coherence`` of the aperture sums coherently, so the honest gain
  uses an EFFECTIVE element count ``n_eff = aperture_coherence * n_elements`` (D4).

* **Temporal: within-epoch coherent integration (1/sqrt(N_ens)).** Ultrafast imaging
  collects frames per voxel per epoch, but only *independent* looks at the *same*
  quasi-static Delta z reduce the estimate variance. Two physical limits cap the naive
  ``f_frame * T_epoch`` count (D1): the signal stays coherent only over a **coherence
  window** (a beta burst is ~150-300 ms, not the whole epoch), and successive ultrafast
  frames **decorrelate** on a tissue timescale, so the number of *independent* looks is
  ``coherent_window / frame_decorrelation_time``, not the raw frame count. The effective
  count ``N_ens`` is the smaller of those, and the gain is ``1/sqrt(N_ens)``.

Both fold into the one Walker-Trahey phase floor::

    sigma_wt = (lambda / 4pi) * sqrt(1 / (2 * n_eff * SNR_perelem)) / (rho * sqrt(N_ens))

where ``lambda = c / f_N`` and ``rho`` (``echo_correlation``, D5) is the correlation of
the two echoes a single displacement estimate differences: decorrelation inflates the
per-estimate variance by ``1/rho^2`` (std by ``1/rho``). The ``lambda / 4 pi`` prefactor
is the inverse of the round-trip phase-to-displacement map ``delta phi = (4 pi f_N/c) *
Delta z`` (a 10 nm shift at 2 MHz is ~1.6e-4 rad). On top of the SNR-limited
Walker-Trahey term sits a **residual-aberration phase-noise floor** (D3): the
temporally-fluctuating residual wavefront phase left after correction maps to a
displacement noise ``sigma_aber = (lambda/4pi) * aberration_phase_rad`` that does NOT
average down with integration (it is not an SNR term), and adds in quadrature. Through
bone, two-way skull attenuation lowers the beamformed echo SNR and raises the floor.

The verdict denominator the spec asks us to report -- **echo SNR vs residual clutter**
-- is made explicit: the floor is the worse (larger) of the phase floor and a clutter
floor left after SVD/clutter filtering, which now also carries a **reverberation** term
(D8) and is bounded by the **clutter high-pass** (D2, which both caps the coherence
window and removes content below its cutoff). :class:`DetectionBudget` names which one
dominates.

All the new terms default to their INERT values (``coherence_time_s=None`` -> no cap,
``frame_decorrelation_time_s=None`` -> every frame independent, ``echo_correlation=1``,
``aberration_phase_rad=0``, ``aperture_coherence=1``, ``clutter_highpass_hz=0``,
``reverberation_ratio=0``), so a bare :class:`AcquisitionParams` reproduces the prior
attenuation-only, full-integration behaviour. :meth:`AcquisitionParams.demo_motor`
carries the *honest* values; :meth:`AcquisitionParams.demo_motor_optimistic` recovers the
prior baseline for a before/after audit.

This module is sensing, not source physics, so it lives outside ``mechanics/``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from base_neural_model.base.provenance import Provenance, extend
from base_neural_model.base.types import MechanicalDisplacement

# Soft-tissue longitudinal sound speed (m/s), the standard ultrasound value.
DEFAULT_SOUND_SPEED_MPS: float = 1540.0


@dataclass(frozen=True)
class AcquisitionParams:
    """Conventional phase-sensitive ultrafast-ultrasound acquisition knobs.

    SI units throughout. These are deliberately *off-the-shelf research system*
    parameters (the spec excludes any Lucen-specific innovation at this gate). The
    first block is the original set; the second block is the honest-physics extension
    (D1-D9), each field defaulting to the value that reproduces the prior behaviour so
    a bare construction is unchanged and only the presets opt in.
    """

    center_freq_hz: float          # f_N: readout centre frequency, Hz (~2e6)
    frame_rate_hz: float           # f_frame: ultrafast frame rate, Hz (~4000)
    epoch_s: float                 # T_epoch: imaging epoch, s
    echo_snr_linear: float         # echo SNR (linear power ratio), pre-skull
    skull_loss_db_oneway: float    # one-way skull insertion loss, dB (applied x2)
    n_elements: int = 256          # receive-aperture element count (spec 3.2)
    sound_speed_mps: float = DEFAULT_SOUND_SPEED_MPS
    # --- Honest-physics extension (D1-D9); defaults are all INERT --------------------
    coherence_time_s: float | None = None       # D1: signal coherence window; None -> epoch
    frame_decorrelation_time_s: float | None = None  # D1: tissue decorr time; None -> 1/f_frame
    echo_correlation: float = 1.0                # D5: per-estimate echo correlation rho in (0,1]
    aberration_phase_rad: float = 0.0            # D3: residual temporal aberration phase (rad)
    aperture_coherence: float = 1.0              # D4: coherent aperture fraction in (0,1]
    clutter_highpass_hz: float = 0.0             # D2: SVD/clutter high-pass cutoff, Hz
    reverberation_ratio: float = 0.0             # D8: reverb clutter as a multiple of the phase floor
    echo_snr_is_per_element: bool = True         # D9: False -> echo_snr is already post-beamforming

    def __post_init__(self) -> None:
        for name in (
            "center_freq_hz", "frame_rate_hz", "epoch_s", "echo_snr_linear",
            "sound_speed_mps",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)!r}")
        if self.n_elements < 1:
            raise ValueError(f"n_elements must be >= 1, got {self.n_elements!r}")
        if self.skull_loss_db_oneway < 0.0:
            raise ValueError(
                f"skull_loss_db_oneway must be >= 0, got {self.skull_loss_db_oneway!r}"
            )
        if self.coherence_time_s is not None and self.coherence_time_s <= 0.0:
            raise ValueError(
                f"coherence_time_s must be positive or None, got {self.coherence_time_s!r}"
            )
        if (
            self.frame_decorrelation_time_s is not None
            and self.frame_decorrelation_time_s <= 0.0
        ):
            raise ValueError(
                "frame_decorrelation_time_s must be positive or None, got "
                f"{self.frame_decorrelation_time_s!r}"
            )
        if not 0.0 < self.echo_correlation <= 1.0:
            raise ValueError(
                f"echo_correlation must lie in (0, 1], got {self.echo_correlation!r}"
            )
        if self.aberration_phase_rad < 0.0:
            raise ValueError(
                f"aberration_phase_rad must be >= 0, got {self.aberration_phase_rad!r}"
            )
        if not 0.0 < self.aperture_coherence <= 1.0:
            raise ValueError(
                f"aperture_coherence must lie in (0, 1], got {self.aperture_coherence!r}"
            )
        if self.clutter_highpass_hz < 0.0:
            raise ValueError(
                f"clutter_highpass_hz must be >= 0, got {self.clutter_highpass_hz!r}"
            )
        if self.reverberation_ratio < 0.0:
            raise ValueError(
                f"reverberation_ratio must be >= 0, got {self.reverberation_ratio!r}"
            )

    @classmethod
    def demo_motor(cls) -> AcquisitionParams:
        """The flagship-demo motor-imagery regime, with the **honest** acoustic terms.

        The 2 MHz readout is the skull-forced compromise; the skull loss is a mid-range
        transcranial figure for the thin temporal-window bone. The honest terms
        (contestable inputs, like the source-side eta): the coherent window is a beta
        burst (~200 ms, not the 1.5 s epoch -- sensorimotor beta is transient even under
        sustained demand); ultrafast frames decorrelate on a ~2 ms tissue timescale; a
        residual temporal aberration of ~0.1 mrad survives correction; only ~60% of the
        aperture sums coherently through the skull; an SVD clutter high-pass at ~1 Hz;
        and modest reverberation. See :meth:`demo_motor_optimistic` for the prior
        all-favourable baseline used in the before/after audit.
        """
        return cls(
            center_freq_hz=2e6,
            frame_rate_hz=4000.0,
            epoch_s=1.5,
            echo_snr_linear=1.0e3,       # ~30 dB free-field PER-ELEMENT echo SNR
            skull_loss_db_oneway=12.0,   # temporal-window bone, one-way (~24 dB two-way)
            n_elements=256,              # 256-element temporal-window aperture (spec 3.2)
            coherence_time_s=0.2,        # beta burst window (eLife 80160; ~150-300 ms)
            frame_decorrelation_time_s=2.0e-3,  # ultrafast tissue decorrelation ~2 ms
            echo_correlation=0.95,       # adjacent-frame echo correlation through bone
            aberration_phase_rad=1.0e-4,  # ~0.1 mrad residual temporal phase after correction
            aperture_coherence=0.6,      # coherent aperture fraction through uncorrected skull
            clutter_highpass_hz=1.0,     # SVD clutter high-pass cutoff
            reverberation_ratio=0.5,     # reverb clutter ~ half the echo floor
        )

    @classmethod
    def demo_motor_optimistic(cls) -> AcquisitionParams:
        """The prior all-favourable baseline (attenuation-only skull, full integration).

        Identical readout/aperture/skull to :meth:`demo_motor` but with every
        honest-physics term at its inert default, so it reproduces the historical
        ``-13.03 dB`` verdict. Kept for the additive before/after audit, not as a
        recommended operating point.
        """
        return cls(
            center_freq_hz=2e6,
            frame_rate_hz=4000.0,
            epoch_s=1.5,
            echo_snr_linear=1.0e3,
            skull_loss_db_oneway=12.0,
            n_elements=256,
        )

    @classmethod
    def demo_motor_engineering(cls) -> AcquisitionParams:
        """The **optimistic engineering-gap** acoustic regime -- every floor lever at its
        favourable-but-*physically-achievable* value, so the budget lands in the spec's
        claimed "one to two orders" gap rather than the honest ~3 orders.

        The counterpart to :meth:`demo_motor` (honest) and distinct from
        :meth:`demo_motor_optimistic` (the old all-inert baseline, which assumes an *unsafe*
        30 dB echo SNR and full 6000-frame integration). Three floor terms move relative to
        ``demo_motor``, each defensible and none physically impossible:

        * **Echo SNR capped at the transcranial safety ceiling, not raised.** 28 dB, i.e.
          ``max_per_element_echo_snr_db(12)`` (40 dB MI-limited surface figure minus the
          12 dB one-way transmit skull loss; see :mod:`base_neural_model.forward.safety`).
          This is *lower* than ``demo_motor``'s 30 dB -- respecting the MI/thermal limit is
          the point, so ``echo_snr_within_safety`` is True here where it is False for
          ``demo_motor``. Hardcoded (``10**2.8``) to avoid a circular import with
          ``safety.py``; the invariant is pinned by ``test_detection``.
        * **Aperture correction (D4): 0.6 -> 0.9.** The metamaterial aberration-correcting
          stack -- the architecture's headline feature, excluded from the honest baseline --
          restores most of the coherent aperture through the skull.
        * **Thermal-independent frames (D1): the 2 ms decorrelation cap is removed.** An
          echo-SNR/thermal-limited floor has independent per-frame noise, so ``N_ens`` is
          the frames in the coherent window (800), capped only by the biology below.

        Deliberately unchanged from ``demo_motor``: the **200 ms beta-burst coherence
        window** (``coherence_time_s=0.2`` -- biology, not an engineering lever, so the gap
        is reached without touching neural physics), the echo correlation, the residual
        aberration, the clutter high-pass, and the reverberation.
        """
        return cls(
            center_freq_hz=2e6,
            frame_rate_hz=4000.0,
            epoch_s=1.5,
            echo_snr_linear=10**2.8,     # 28 dB: the transcranial safety ceiling at 12 dB
                                         # skull (safety.max_per_element_echo_snr_db(12)),
                                         # NOT the honest preset's unsafe 30 dB
            skull_loss_db_oneway=12.0,
            n_elements=256,
            coherence_time_s=0.2,        # beta burst KEPT (biology, not an engineering lever)
            # frame_decorrelation_time_s left at None: frames are thermal-independent, so
            # N_ens is the burst-window frame count (800), not the 2 ms-decorrelation 100
            echo_correlation=0.95,
            aberration_phase_rad=1.0e-4,
            aperture_coherence=0.9,      # metamaterial aberration correction (the architecture's point)
            clutter_highpass_hz=1.0,
            reverberation_ratio=0.5,
        )

    @classmethod
    def speech_unit(cls) -> AcquisitionParams:
        """The speech-unit regime: same hardware, a tens-of-ms coherence window.

        A 50 ms speech unit -- shorter than a beta burst, so the coherence window is the
        epoch itself. Honest acoustic terms as :meth:`demo_motor`; only the epoch
        differs, which is the axis that makes speech the harder bet.
        """
        return cls(
            center_freq_hz=2e6,
            frame_rate_hz=4000.0,
            epoch_s=0.05,
            echo_snr_linear=1.0e3,
            skull_loss_db_oneway=12.0,
            n_elements=256,
            coherence_time_s=0.2,
            frame_decorrelation_time_s=2.0e-3,
            echo_correlation=0.95,
            aberration_phase_rad=1.0e-4,
            aperture_coherence=0.6,
            clutter_highpass_hz=1.0,
            reverberation_ratio=0.5,
        )

    @property
    def wavelength_m(self) -> float:
        """Acoustic wavelength ``lambda = c / f_N`` (m)."""
        return self.sound_speed_mps / self.center_freq_hz

    @property
    def effective_n_elements(self) -> int:
        """Coherently-summing element count ``round(aperture_coherence * n_elements)``.

        Through an uncorrected skull only a fraction of the aperture stays phase-coherent
        (D4); the beamforming gain and beamformed echo SNR use this effective count, not
        the physical ``n_elements``. At ``aperture_coherence = 1`` it is ``n_elements``.
        """
        return max(1, round(self.aperture_coherence * self.n_elements))

    @property
    def coherent_window_s(self) -> float:
        """The window over which the signal stays coherent, seconds (D1/D2).

        The smallest of: the imaging epoch, the signal coherence time (a beta burst), and
        the clutter high-pass memory ``1/clutter_highpass_hz`` (echo phase cannot be
        summed coherently for longer than the high-pass retains it). Coherent integration
        happens only inside this window.
        """
        window = self.epoch_s
        if self.coherence_time_s is not None:
            window = min(window, self.coherence_time_s)
        if self.clutter_highpass_hz > 0.0:
            window = min(window, 1.0 / self.clutter_highpass_hz)
        return window

    @property
    def _decorrelation_time_s(self) -> float:
        """Frame decorrelation time, seconds; ``None`` means every frame is independent."""
        if self.frame_decorrelation_time_s is None:
            return 1.0 / self.frame_rate_hz
        return self.frame_decorrelation_time_s

    @property
    def raw_ensemble_count(self) -> int:
        """The NAIVE frame count ``round(f_frame * T_epoch)`` (the pre-D1 optimistic value)."""
        return max(1, round(self.frame_rate_hz * self.epoch_s))

    @property
    def ensemble_count(self) -> int:
        """Effective independent-look count ``N_ens`` for coherent integration (D1).

        The smaller of the frames that fit in the coherent window
        (``f_frame * coherent_window``) and the *independent* looks in it
        (``coherent_window / frame_decorrelation_time``). With the inert defaults
        (no coherence cap, every frame independent) this equals
        :attr:`raw_ensemble_count`.
        """
        frames_in_window = self.frame_rate_hz * self.coherent_window_s
        independent_looks = self.coherent_window_s / self._decorrelation_time_s
        return max(1, round(min(frames_in_window, independent_looks)))

    @property
    def integration_gain(self) -> float:
        """Within-window temporal coherent-integration amplitude gain ``sqrt(N_ens)``."""
        return math.sqrt(self.ensemble_count)

    @property
    def beamforming_gain(self) -> float:
        """Receive-beamforming amplitude gain ``sqrt(n_eff)`` (spatial, per frame; D4)."""
        return math.sqrt(self.effective_n_elements)

    @property
    def beamformed_echo_snr_linear(self) -> float:
        """Free-field beamformed *voxel* echo SNR (linear power ratio; D4+D9).

        When ``echo_snr_is_per_element`` (the default), the per-element SNR is raised to
        the voxel level by the coherent aperture sum: ``n_eff * echo_snr_linear``. When
        ``False``, ``echo_snr_linear`` is already the post-beamforming voxel SNR and the
        aperture gain is NOT applied again (the -37 dB reading of the B1 ambiguity).
        """
        if self.echo_snr_is_per_element:
            return self.effective_n_elements * self.echo_snr_linear
        return self.echo_snr_linear

    @property
    def through_skull_echo_snr_linear(self) -> float:
        """Beamformed voxel echo SNR after two-way skull loss (linear power ratio)."""
        return self.beamformed_echo_snr_linear * 10.0 ** (
            -2.0 * self.skull_loss_db_oneway / 10.0
        )


def integration_gain(acq: AcquisitionParams) -> float:
    """Within-window coherent-integration amplitude gain ``sqrt(N_ens)`` (D1).

    ``N_ens`` is the effective independent-look count (:attr:`AcquisitionParams.
    ensemble_count`), which the coherence window and frame decorrelation cap below the
    naive ``f_frame * T_epoch``.
    """
    return acq.integration_gain


def phase_to_displacement_m_per_rad(acq: AcquisitionParams) -> float:
    """Echo-phase to axial-displacement conversion ``lambda / 4 pi`` (m / rad).

    The inverse of the round-trip phase map ``delta phi = (4 pi f_N / c) Delta z``.
    At 2 MHz this is ~6.13e-5 m/rad, so a 10 nm displacement is ~1.6e-4 rad.
    """
    return acq.wavelength_m / (4.0 * math.pi)


def aberration_displacement_floor(acq: AcquisitionParams) -> float:
    """Residual-aberration phase-noise displacement floor ``(lambda/4pi)*phi_aber`` (D3).

    The temporally-fluctuating residual wavefront phase left after aberration correction
    maps directly to a displacement uncertainty. Unlike the Walker-Trahey term it is NOT
    an SNR quantity, so it does NOT average down with integration; it adds in quadrature
    to the SNR-limited floor. Zero when ``aberration_phase_rad = 0``.
    """
    return phase_to_displacement_m_per_rad(acq) * acq.aberration_phase_rad


def walker_trahey_floor(acq: AcquisitionParams) -> float:
    """SNR-limited through-skull phase floor, metres (Walker-Trahey; D1,D4,D5).

    ``sigma_wt = (lambda/4pi) * sqrt(1/(2*SNR_echo)) / (rho * sqrt(N_ens))`` where
    ``SNR_echo`` is the beamformed voxel echo SNR AFTER two-way skull loss (carrying the
    effective ``n_eff`` aperture gain, D4), ``rho`` is the per-estimate echo correlation
    (D5, variance inflation ``1/rho^2``) and ``N_ens`` the effective independent-look
    count (D1). Falls as ``1/sqrt(N_ens)``, ``1/sqrt(n_eff)`` and ``1/sqrt(SNR)``; rises
    with skull dB and as ``rho`` drops.
    """
    snr = acq.through_skull_echo_snr_linear
    prefactor = phase_to_displacement_m_per_rad(acq)
    return prefactor * math.sqrt(1.0 / (2.0 * snr)) / (acq.echo_correlation * acq.integration_gain)


def phase_displacement_floor(acq: AcquisitionParams) -> float:
    """Total through-skull displacement floor, metres: Walker-Trahey AND aberration.

    The SNR-limited Walker-Trahey term (:func:`walker_trahey_floor`, which carries both
    receive-side coherent gains and the echo correlation) added in quadrature with the
    integration-independent residual-aberration phase-noise floor
    (:func:`aberration_displacement_floor`, D3)::

        sigma_disp = sqrt(sigma_wt^2 + sigma_aber^2)

    With the inert defaults this reduces exactly to the prior Walker-Trahey-only floor.
    """
    return math.hypot(walker_trahey_floor(acq), aberration_displacement_floor(acq))


# Which denominator pins the detection floor. "echo_snr" => the through-skull phase floor
# dominates; "clutter" => residual post-filter clutter (SVD residual + reverberation)
# dominates. The spec asks Gate B to report exactly this (a clutter-limited failure
# reframes the problem as an engineering fight rather than a physics wall).
LimitingDenominator = str


@dataclass(frozen=True)
class DetectionBudget:
    """The Gate-A / Stage-1 detectability verdict for one source displacement.

    Composes the source deliverable (Delta z, content-band survival) with the acoustic
    layer (through-skull phase floor incl. aberration, residual + reverberation clutter).
    The headline is ``snr_db`` and ``limiting_denominator`` (echo-SNR- vs clutter-limited).
    All receive-side coherent gain lives in the floor, so ``surviving_dz_m`` is the bare
    ``Delta z * survival`` and is compared directly against ``floor_m``.
    """

    surviving_dz_m: float          # Delta z * content survival (BARE; gains in the floor)
    phase_floor_m: float           # total through-skull floor (Walker-Trahey + aberration)
    residual_clutter_m: float      # binding clutter floor (SVD residual + reverberation)
    floor_m: float                 # the binding floor = max(phase_floor, clutter)
    limiting_denominator: LimitingDenominator
    integration_gain: float        # sqrt(N_ens): temporal gain, folded into the floor
    ensemble_count: int            # N_ens (effective, D1)
    raw_ensemble_count: int        # naive f_frame*T_epoch (for the before/after audit)
    beamforming_gain: float        # sqrt(n_eff): spatial gain, folded into the floor
    n_elements: int                # physical receive-aperture element count
    effective_n_elements: int      # coherent aperture element count n_eff (D4)
    aberration_floor_m: float      # the residual-aberration component of the floor (D3)
    snr_exceeds_safety: bool        # D6: assumed echo SNR exceeds the safety-achievable ceiling
    provenance: Provenance

    @property
    def snr_linear(self) -> float:
        """Detectability ratio: surviving displacement over the binding floor."""
        return self.surviving_dz_m / self.floor_m if self.floor_m > 0 else float("inf")

    @property
    def snr_db(self) -> float:
        """Detectability in dB (20 log10 of the bare-displacement-over-floor ratio)."""
        r = self.snr_linear
        return 20.0 * math.log10(r) if r > 0 else float("-inf")

    @property
    def detectable(self) -> bool:
        """True iff the surviving displacement clears the binding floor (SNR >= 1)."""
        return self.surviving_dz_m >= self.floor_m


def detection_budget(
    mech: MechanicalDisplacement,
    acq: AcquisitionParams,
    *,
    residual_clutter_m: float = 0.0,
    content_freq_hz: float | None = None,
    snr_exceeds_safety: bool = False,
) -> DetectionBudget:
    """Compose a source displacement with the acoustic layer into a detectability verdict.

    ``mech`` is the existing source deliverable; the surviving signal is the BARE
    ``Delta z * survival`` -- NOT amplified, because both receive-side gains are in the
    floor. The binding floor is the worse of the total phase floor
    (:func:`phase_displacement_floor`) and the clutter floor, where the clutter floor is
    ``max(residual_clutter_m, reverberation_ratio * phase_floor)`` (D8).

    ``content_freq_hz`` (when given) lets the clutter high-pass (D2) remove content that
    falls below ``clutter_highpass_hz``: a beta carrier under the cutoff is filtered out
    with the bulk motion, so the surviving signal is zeroed. ``snr_exceeds_safety`` (D6)
    is recorded when the caller has determined the assumed echo SNR is not achievable
    within acoustic-output safety limits (see :mod:`base_neural_model.forward.safety`).
    """
    if residual_clutter_m < 0.0:
        raise ValueError(
            f"residual_clutter_m must be >= 0, got {residual_clutter_m!r}"
        )

    gain = acq.integration_gain
    bf_gain = acq.beamforming_gain
    # Bare surviving displacement: the receive-side coherent gains are in the floor.
    surviving = mech.axial_displacement_m * mech.content_band_survival

    # D2: content below the clutter high-pass is removed with the bulk motion.
    highpass_removed = (
        content_freq_hz is not None
        and acq.clutter_highpass_hz > 0.0
        and content_freq_hz < acq.clutter_highpass_hz
    )
    if highpass_removed:
        surviving = 0.0

    phase_floor = phase_displacement_floor(acq)
    aber_floor = aberration_displacement_floor(acq)

    # D8: reverberation clutter, expressed as a multiple of the echo phase floor, joins
    # the SVD residual in the clutter denominator.
    reverb_floor = acq.reverberation_ratio * phase_floor
    clutter_floor = max(residual_clutter_m, reverb_floor)

    if clutter_floor > phase_floor:
        floor = clutter_floor
        denom = "clutter"
    else:
        floor = phase_floor
        denom = "echo_snr"

    provenance = extend(
        mech.provenance,
        "acoustic detection layer (conventional phase-sensitive ultrafast US; no "
        "Lucen-specific innovation): source Delta z -> through-skull echo-phase SNR",
        f"bare surviving signal = Delta z x content survival = {surviving * 1e9:.4g} nm "
        + ("(REMOVED: content below the clutter high-pass, D2) " if highpass_removed else "")
        + "(NOT re-amplified; the receive-side gains below are in the floor)",
        f"spatial: coherent aperture n_eff = {acq.effective_n_elements} of "
        f"{acq.n_elements} (aperture_coherence = {acq.aperture_coherence}) -> voxel echo "
        f"SNR gain x{acq.effective_n_elements} (amplitude sqrt = {bf_gain:.4g}) [D4]",
        f"temporal: coherent window = {acq.coherent_window_s * 1e3:.4g} ms "
        f"(epoch {acq.epoch_s} s, coherence {acq.coherence_time_s}, highpass "
        f"{acq.clutter_highpass_hz} Hz); N_ens = {acq.ensemble_count} independent looks "
        f"(decorr {acq._decorrelation_time_s * 1e3:.4g} ms) vs naive "
        f"{acq.raw_ensemble_count}; floor factor 1/sqrt(N_ens) (sqrt = {gain:.4g}) [D1]",
        f"Walker-Trahey floor with echo correlation rho = {acq.echo_correlation} [D5]; "
        f"two-way skull loss = {2.0 * acq.skull_loss_db_oneway:.4g} dB -> through-skull "
        f"echo SNR = {acq.through_skull_echo_snr_linear:.4g} (linear)",
        f"residual-aberration phase-noise floor = {aber_floor * 1e9:.4g} nm "
        f"(phi_aber = {acq.aberration_phase_rad} rad, integration-independent) [D3]; "
        f"total phase floor = {phase_floor * 1e9:.4g} nm",
        f"binding floor = {floor * 1e9:.4g} nm, limited by {denom} (SVD residual "
        f"{residual_clutter_m * 1e9:.4g} nm, reverberation {reverb_floor * 1e9:.4g} nm "
        f"at ratio {acq.reverberation_ratio}) [D8]"
        + ("; ECHO SNR EXCEEDS SAFETY CEILING [D6]" if snr_exceeds_safety else ""),
    )
    return DetectionBudget(
        surviving_dz_m=surviving,
        phase_floor_m=phase_floor,
        residual_clutter_m=clutter_floor,
        floor_m=floor,
        limiting_denominator=denom,
        integration_gain=gain,
        ensemble_count=acq.ensemble_count,
        raw_ensemble_count=acq.raw_ensemble_count,
        beamforming_gain=bf_gain,
        n_elements=acq.n_elements,
        effective_n_elements=acq.effective_n_elements,
        aberration_floor_m=aber_floor,
        snr_exceeds_safety=snr_exceeds_safety,
        provenance=provenance,
    )
