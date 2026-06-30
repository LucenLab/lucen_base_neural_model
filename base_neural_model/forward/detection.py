r"""Acoustic detection layer: source displacement -> echo-phase detectability.

This is the FIRST module that crosses the source/sensing boundary the rest of
``base_neural_model`` deliberately stops at. It takes the net axial source
displacement Delta z that the mechanics chain produces (a
:class:`~base_neural_model.base.types.MechanicalDisplacement`) and asks the Gate-A /
Stage-1 question the Lucen specs make the cheapest falsification: **does the
content-bearing displacement, after within-epoch coherent integration, clear the
phase-sensitive ultrasound detection floor through the skull?**

Nothing here recomputes source physics. The module composes two acoustic factors
onto the existing source deliverable:

* **Within-epoch coherent integration (the demo's load-bearing motor advantage).**
  Ultrafast imaging collects ``N_ens = f_frame * T_epoch`` displacement samples per
  voxel per epoch, and coherent integration over them gains SNR as ``sqrt(N_ens)``
  (echo-phase estimate variance falls as 1/N). A multi-second motor-imagery epoch
  supplies thousands of samples (gain ~x77) where a tens-of-ms speech unit supplies
  hundreds (gain ~x14) -- the entire reason the motor target is the favorable bet.

* **The phase-sensitive displacement floor (Walker-Trahey).** Echo-phase
  displacement estimation has a variance floor set by the echo SNR and wavelength::

      sigma_disp = (lambda / 4 pi) * sqrt(1 / (2 * SNR_echo)) * (1 / sqrt(N_ens))

  where ``lambda = c / f_N``. The ``lambda / 4 pi`` prefactor is the inverse of the
  round-trip phase-to-displacement map ``delta phi = (4 pi f_N / c) * Delta z`` (a
  10 nm shift at 2 MHz is ~1.6e-4 rad, a sixth of a milliradian -- the spec's
  number). Through bone, two-way skull attenuation lowers ``SNR_echo`` and raises the
  floor; that loss is applied INSIDE the floor so the reported number is the
  through-skull floor, not a free-field one.

The verdict denominator the spec asks us to report -- **echo SNR vs residual clutter**
-- is made explicit: the floor is the worse (larger) of the Walker-Trahey echo-SNR
floor and an optional residual-clutter floor left after SVD/clutter filtering, and
:class:`DetectionBudget` names which one dominates.

This module is sensing, not source physics, so it lives outside ``mechanics/`` (the
pure source chain). It is the Module-2 seam the source docs anticipate
(``model/resolution.py`` and ``model/gates.py`` both point here).
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
    parameters (the spec excludes any Lucen-specific innovation at this gate): a
    single readout frequency, a plane/diverging-wave ultrafast frame rate, an epoch
    over which the signal stays coherent, an echo SNR, and the two-way skull loss.

    ``echo_snr_linear`` is the free-field (pre-skull) per-frame echo SNR as a linear
    power ratio; ``skull_loss_db_oneway`` is the one-way skull insertion loss, applied
    twice (pulse-echo) inside :func:`phase_displacement_floor`.
    """

    center_freq_hz: float          # f_N: readout centre frequency, Hz (~2e6)
    frame_rate_hz: float           # f_frame: ultrafast frame rate, Hz (~4000)
    epoch_s: float                 # T_epoch: coherent-integration window, s
    echo_snr_linear: float         # per-frame echo SNR (linear power ratio), pre-skull
    skull_loss_db_oneway: float    # one-way skull insertion loss, dB (applied x2)
    sound_speed_mps: float = DEFAULT_SOUND_SPEED_MPS

    def __post_init__(self) -> None:
        for name in (
            "center_freq_hz", "frame_rate_hz", "epoch_s", "echo_snr_linear",
            "sound_speed_mps",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)!r}")
        if self.skull_loss_db_oneway < 0.0:
            raise ValueError(
                f"skull_loss_db_oneway must be >= 0, got {self.skull_loss_db_oneway!r}"
            )

    @classmethod
    def demo_motor(cls) -> AcquisitionParams:
        """The flagship-demo motor-imagery regime (the favorable source term).

        A 1.5 s motor-imagery epoch at ~4 kHz ultrafast framing -> ~6000 samples ->
        ~x77 coherent-integration gain, the demo's central advantage over speech. The
        2 MHz readout is the skull-forced compromise the spec names; the skull loss is
        a mid-range transcranial figure for the thin temporal-window bone.
        """
        return cls(
            center_freq_hz=2e6,
            frame_rate_hz=4000.0,
            epoch_s=1.5,
            echo_snr_linear=1.0e3,     # ~30 dB free-field per-frame echo SNR
            skull_loss_db_oneway=12.0,  # temporal-window bone, one-way (~24 dB two-way)
        )

    @classmethod
    def speech_unit(cls) -> AcquisitionParams:
        """The speech-unit regime: same hardware, a tens-of-ms coherence window.

        A 50 ms speech unit at ~4 kHz -> ~200 samples -> only ~x14 gain. Identical
        optics and skull to :meth:`demo_motor`; only the epoch differs, which is
        exactly the axis that makes speech the harder bet.
        """
        return cls(
            center_freq_hz=2e6,
            frame_rate_hz=4000.0,
            epoch_s=0.05,
            echo_snr_linear=1.0e3,
            skull_loss_db_oneway=12.0,
        )

    @property
    def wavelength_m(self) -> float:
        """Acoustic wavelength ``lambda = c / f_N`` (m)."""
        return self.sound_speed_mps / self.center_freq_hz

    @property
    def ensemble_count(self) -> int:
        """Coherently-integrated frame count ``N_ens = round(f_frame * T_epoch)``."""
        return max(1, round(self.frame_rate_hz * self.epoch_s))

    @property
    def integration_gain(self) -> float:
        """Coherent-integration amplitude gain ``sqrt(N_ens)``."""
        return math.sqrt(self.ensemble_count)

    @property
    def through_skull_echo_snr_linear(self) -> float:
        """Echo SNR after two-way skull loss (linear power ratio).

        Pulse-echo pays the one-way insertion loss twice, so the power ratio is
        scaled by ``10 ** (-2 * skull_loss_db_oneway / 10)``.
        """
        return self.echo_snr_linear * 10.0 ** (-2.0 * self.skull_loss_db_oneway / 10.0)


def integration_gain(acq: AcquisitionParams) -> float:
    """Within-epoch coherent-integration amplitude gain ``sqrt(f_frame * T_epoch)``.

    The demo's load-bearing motor advantage: a long, stable activation window supplies
    more displacement samples, and coherent integration over the ultrafast frame train
    gains as the square root of the count. ~x14 for a 50 ms speech unit, ~x77 for a
    1.5 s motor-imagery epoch (both at 4 kHz).
    """
    return acq.integration_gain


def phase_to_displacement_m_per_rad(acq: AcquisitionParams) -> float:
    """Echo-phase to axial-displacement conversion ``lambda / 4 pi`` (m / rad).

    The inverse of the round-trip phase map ``delta phi = (4 pi f_N / c) Delta z``.
    At 2 MHz this is ~6.13e-5 m/rad, so a 10 nm displacement is ~1.6e-4 rad.
    """
    return acq.wavelength_m / (4.0 * math.pi)


def phase_displacement_floor(acq: AcquisitionParams) -> float:
    """Through-skull phase-sensitive displacement floor (Walker-Trahey), metres.

    ``sigma_disp = (lambda / 4 pi) * sqrt(1 / (2 * SNR_echo)) * (1 / sqrt(N_ens))``
    with ``SNR_echo`` taken AFTER two-way skull loss, so the returned floor is the
    through-bone detection floor a conventional system actually faces. Falls as
    ``1/sqrt(N_ens)`` (more integration) and ``1/sqrt(SNR_echo)`` (cleaner echo), and
    rises with skull dB.
    """
    snr = acq.through_skull_echo_snr_linear
    prefactor = phase_to_displacement_m_per_rad(acq)
    return prefactor * math.sqrt(1.0 / (2.0 * snr)) / acq.integration_gain


# Which denominator pins the detection floor. "echo_snr" => the Walker-Trahey
# through-skull phase floor dominates; "clutter" => residual post-filter clutter
# dominates. The spec asks Gate B to report exactly this (a clutter-limited failure
# reframes the problem as an engineering fight rather than a physics wall).
LimitingDenominator = str


@dataclass(frozen=True)
class DetectionBudget:
    """The Gate-A / Stage-1 detectability verdict for one source displacement.

    Composes the source deliverable (Delta z, content-band survival) with the acoustic
    layer (integration gain, through-skull phase floor, residual clutter). The headline
    is ``snr_db`` -- the content-surviving, integration-amplified displacement over the
    binding noise floor -- and ``limiting_denominator``, which names whether the floor
    is echo-SNR-limited or clutter-limited.
    """

    surviving_dz_m: float          # Delta z * content survival * integration gain
    phase_floor_m: float           # Walker-Trahey through-skull echo-SNR floor
    residual_clutter_m: float      # post-filter residual clutter floor (0 if none)
    floor_m: float                 # the binding floor = max(phase_floor, clutter)
    limiting_denominator: LimitingDenominator
    integration_gain: float        # sqrt(N_ens) applied to the signal
    ensemble_count: int            # N_ens
    provenance: Provenance

    @property
    def snr_linear(self) -> float:
        """Detectability ratio: surviving displacement over the binding floor."""
        return self.surviving_dz_m / self.floor_m if self.floor_m > 0 else float("inf")

    @property
    def snr_db(self) -> float:
        """Detectability in dB (20 log10 of the displacement ratio)."""
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
) -> DetectionBudget:
    """Compose a source displacement with the acoustic layer into a detectability verdict.

    ``mech`` is the existing source deliverable; its ``axial_displacement_m`` is the
    content-band coherent Delta z and ``content_band_survival`` the jitter low-pass.
    The surviving signal is ``Delta z * survival * sqrt(N_ens)``. The binding floor is
    the worse of the Walker-Trahey through-skull phase floor and ``residual_clutter_m``
    (the post-clutter-filter residual, defaulting to 0 == echo-SNR-limited), and
    ``limiting_denominator`` names which dominates.
    """
    if residual_clutter_m < 0.0:
        raise ValueError(
            f"residual_clutter_m must be >= 0, got {residual_clutter_m!r}"
        )

    gain = acq.integration_gain
    surviving = mech.axial_displacement_m * mech.content_band_survival * gain
    phase_floor = phase_displacement_floor(acq)

    if residual_clutter_m > phase_floor:
        floor = residual_clutter_m
        denom = "clutter"
    else:
        floor = phase_floor
        denom = "echo_snr"

    provenance = extend(
        mech.provenance,
        "acoustic detection layer (conventional phase-sensitive ultrafast US; no "
        "Lucen-specific innovation): source Delta z -> through-skull echo-phase SNR",
        f"within-epoch coherent integration: N_ens = {acq.ensemble_count} frames "
        f"(f_frame = {acq.frame_rate_hz} Hz x T_epoch = {acq.epoch_s} s) -> "
        f"amplitude gain sqrt(N_ens) = {gain:.4g}",
        "phase-sensitive displacement floor (Walker-Trahey): "
        "sigma_disp = (lambda/4pi) sqrt(1/(2 SNR_echo)) / sqrt(N_ens), "
        f"lambda = {acq.wavelength_m * 1e3:.4g} mm at f_N = {acq.center_freq_hz} Hz",
        f"two-way skull loss = {2.0 * acq.skull_loss_db_oneway:.4g} dB -> through-skull "
        f"echo SNR = {acq.through_skull_echo_snr_linear:.4g} (linear); "
        f"phase floor = {phase_floor * 1e9:.4g} nm",
        f"binding floor = {floor * 1e9:.4g} nm, limited by {denom} "
        f"(residual clutter floor = {residual_clutter_m * 1e9:.4g} nm)",
    )
    return DetectionBudget(
        surviving_dz_m=surviving,
        phase_floor_m=phase_floor,
        residual_clutter_m=residual_clutter_m,
        floor_m=floor,
        limiting_denominator=denom,
        integration_gain=gain,
        ensemble_count=acq.ensemble_count,
        provenance=provenance,
    )
