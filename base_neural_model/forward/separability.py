r"""Spatial separability of two adjacent voxels: the PSF/sidelobe leakage layer.

Every module up to here (``detection.py``, ``budget.py``) answers a ONE-voxel question:
does this voxel's own displacement clear the noise floor? The demo's Act 1 claim (build
spec sections 5.1, 6.2) is a TWO-voxel question the model has not yet addressed:
neighbouring digit representations sit "on the order of a few millimetres" apart, and the
live map must show a hotspot that visibly *shifts* between them -- which requires that an
active voxel's own displacement dominates its reading over a neighbour's displacement
leaking in through the beam's off-axis response, at the digit-to-digit spacing the
somatotopy actually provides.

Two independent physical leakage mechanisms, both named explicitly in the technical spec
(section 3.4, "Sidelobe and aberration leakage") and the build spec (section 3.2):

* **Sidelobe/main-lobe leakage.** A focused aperture's lateral response is not a clean
  box of width ``w_lat = F# * lambda`` (technical spec section 3.1) -- it has a smooth
  falloff with finite off-axis sidelobes. For a uniformly-illuminated rectangular
  aperture the classic (Fraunhofer/paraxial-focus) one-way pressure response is a sinc in
  the off-axis angle; intensity leakage is its square. The first sidelobe of an
  unapodized rectangular aperture sits at -13.3 dB; apodization (build spec section 4.1)
  trades main-lobe width for deeper, faster-falling sidelobes -- a real, contestable
  engineering knob, not a derived constant, so it is a parameter here exactly as
  ``aperture_coherence`` and the skull loss are upstream.

* **Grating-lobe leakage.** The build spec is explicit that element pitch is set at or
  below ``lambda/2`` "to suppress grating lobes -- spurious off-axis foci that would leak
  distant cortical activity into the target voxel" (section 3.2). Above ``lambda/2`` a
  full-amplitude grating lobe appears at ``sin(theta_g) = lambda/pitch - sin(theta_0)``
  (the array factor's next principal maximum); below it, no grating lobe exists in the
  visible half-space at broadside. This is a THRESHOLD effect, not a smooth falloff, and
  is kept as a separate term rather than folded into the sidelobe envelope.

Both leakage mechanisms act on the SOURCE displacement (they mix a neighbour's true
signal into a voxel's reading), not on the acoustic noise floor, so they compose with
:class:`~base_neural_model.base.types.MechanicalDisplacement` the same way the incoherent
pedestal does in ``mechanics/transduction.py``: as a contaminating term competing with the
target voxel's own signal, not with the detection floor. A voxel pair is separable when
the leaked neighbour signal is small relative to the reading that must distinguish
"neighbour active, self quiet" from "self active" -- the two-hotspot discrimination Act 1
needs, not a resolution length in isolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from base_neural_model.base.provenance import Provenance, extend
from base_neural_model.base.types import MechanicalDisplacement

# First-sidelobe level of a uniformly-illuminated (unapodized) rectangular aperture at
# paraxial focus, in dB below the main-lobe peak (the classic sinc-intensity result,
# 20*log10(sinc sidelobe) for the first zero-adjacent lobe). Apodization deepens this;
# it is the honest (unapodized) worst case, analogous to aperture_coherence=1 upstream.
UNAPODIZED_SIDELOBE_DB: float = -13.3

# Element pitch relative to wavelength below which no grating lobe exists in the visible
# half-space at broadside steering (build spec section 3.2: "pitch is set at or below
# lambda/2 to suppress grating lobes").
GRATING_LOBE_PITCH_RATIO: float = 0.5


@dataclass(frozen=True)
class BeamAperture:
    """The lateral beam geometry a voxel pair's separability is evaluated against.

    Mirrors the technical spec's aperture parameters (section 3.1) and the build spec's
    array numbers (section 3.2): wavelength and F-number set the main-lobe width;
    ``sidelobe_db`` is the apodization-dependent off-axis floor; ``element_pitch_m`` and
    ``n_elements`` set the grating-lobe threshold and the physical aperture extent.
    """

    wavelength_m: float          # lambda = c / f_N (technical spec section 3.1)
    f_number: float              # F# = focal_depth / aperture_width
    sidelobe_db: float = UNAPODIZED_SIDELOBE_DB  # off-axis floor after apodization
    element_pitch_m: float | None = None  # None -> grating-lobe check skipped
    n_elements: int | None = None         # None -> grating-lobe check skipped

    def __post_init__(self) -> None:
        if self.wavelength_m <= 0.0:
            raise ValueError(f"wavelength_m must be positive, got {self.wavelength_m!r}")
        if self.f_number <= 0.0:
            raise ValueError(f"f_number must be positive, got {self.f_number!r}")
        if self.sidelobe_db >= 0.0:
            raise ValueError(
                f"sidelobe_db must be negative (a level below the main lobe), got "
                f"{self.sidelobe_db!r}"
            )
        if self.element_pitch_m is not None and self.element_pitch_m <= 0.0:
            raise ValueError(
                f"element_pitch_m must be positive or None, got {self.element_pitch_m!r}"
            )
        if self.n_elements is not None and self.n_elements < 1:
            raise ValueError(f"n_elements must be >= 1 or None, got {self.n_elements!r}")

    @property
    def lateral_width_m(self) -> float:
        """Main-lobe (-3ish dB, "the voxel") lateral width ``w_lat = F# * lambda``.

        Technical spec section 3.1: "the lateral focal width... on the order of the
        F-number times the wavelength."
        """
        return self.f_number * self.wavelength_m

    @property
    def has_grating_lobe(self) -> bool:
        """True iff the element pitch exceeds the ``lambda/2`` grating-lobe threshold.

        ``False`` (no grating lobe) whenever ``element_pitch_m`` or ``n_elements`` is not
        given -- the check is opt-in, matching the demo's stated pitch (~0.38 mm at
        2 MHz), which already sits at the threshold.
        """
        if self.element_pitch_m is None:
            return False
        return self.element_pitch_m > GRATING_LOBE_PITCH_RATIO * self.wavelength_m

    @property
    def aperture_width_m(self) -> float | None:
        """Physical receive-aperture extent ``pitch * n_elements``, or ``None``."""
        if self.element_pitch_m is None or self.n_elements is None:
            return None
        return self.element_pitch_m * self.n_elements


def sidelobe_leakage_fraction(
    separation_m: float,
    aperture: BeamAperture,
) -> float:
    r"""Amplitude leakage fraction from the beam's off-axis (sidelobe) response.

    Models the one-way lateral pressure response of a uniformly-illuminated rectangular
    aperture at paraxial focus as ``sinc(pi * x / w_lat)`` (the standard Fraunhofer-focus
    result), where ``x`` is the off-axis distance and ``w_lat`` the main-lobe width
    (:attr:`BeamAperture.lateral_width_m`). At the main lobe (``x=0``) this is 1; at its
    first null (``x = w_lat``) it is 0.

    Past the first null the raw sinc oscillates back up toward its (unapodized) side
    peaks; a real windowed/apodized aperture (technical spec section 5, build spec
    section 4.1) does not -- apodization is specifically the engineering choice that
    keeps every sidelobe at or below a design floor. So past the first null this returns
    the apodized floor ``sidelobe_db`` rather than following the raw sinc's oscillation
    back up: the floor is the honest asymptotic cap a real apodized beam sits at or
    below, and the raw sinc is only trusted inside its own first lobe, where apodization
    has not yet had a chance to act.

    Returns an INTENSITY-domain leakage fraction in ``[0, 1]`` (the fraction of the
    neighbour's displacement signal that appears at the target voxel), so it composes
    linearly with the leaking voxel's own displacement.
    """
    if separation_m < 0.0:
        raise ValueError(f"separation_m must be >= 0, got {separation_m!r}")
    w_lat = aperture.lateral_width_m
    if separation_m == 0.0:
        return 1.0

    if separation_m <= w_lat:
        # Inside the main lobe's first null: the raw sinc IS the response.
        x = math.pi * separation_m / w_lat
        result_db = 20.0 * math.log10(abs(math.sin(x) / x))
    else:
        # Past the first null: the apodized sidelobe floor, not the raw sinc's rebound.
        result_db = aperture.sidelobe_db
    return 10.0 ** (result_db / 20.0)


@dataclass(frozen=True)
class SeparabilityVerdict:
    """Whether two adjacent voxels' readings are dominated by their own activity.

    ``contrast_db`` is the ratio of a fully-active voxel's own signal to the leaked
    signal a fully-active NEIGHBOUR contributes at the SAME reading -- the two-hotspot
    discrimination margin Act 1 needs. Positive and large means "own signal dominates";
    0 dB means the neighbour's leak is exactly as large as the voxel's own signal (no
    discrimination possible); negative means the neighbour's leak would be read as the
    stronger signal.
    """

    separation_m: float
    lateral_width_m: float
    sidelobe_leakage_fraction: float
    has_grating_lobe: bool
    own_signal_m: float
    leaked_signal_m: float
    contrast_db: float
    separable: bool
    provenance: Provenance


def voxel_pair_separability(
    mech: MechanicalDisplacement,
    aperture: BeamAperture,
    separation_m: float,
    *,
    min_contrast_db: float = 6.0,
) -> SeparabilityVerdict:
    """Two-voxel separability verdict at a given digit-to-digit spacing.

    Both voxels are scored at the SAME source displacement ``mech`` (the symmetric case:
    either could be the active one, and the question is whether each reading is
    dominated by its own voxel) -- the honest worst case for a somatotopic map where
    adjacent digit representations have comparable synchrony and geometry (build spec
    section 5.1). ``own_signal_m`` is ``mech``'s own axial displacement; ``leaked_signal_m``
    is that same displacement attenuated by :func:`sidelobe_leakage_fraction` at
    ``separation_m`` -- a fully-active neighbour is the worst-case contaminator.

    ``separable`` requires BOTH: the sidelobe contrast clears ``min_contrast_db`` (default
    6 dB, a conventional discrimination margin -- roughly a factor-of-2 signal ratio) AND
    no grating lobe exists at this aperture (a grating lobe reproduces the neighbour's
    signal at FULL amplitude, which no sidelobe contrast margin can be honest about; the
    build spec's own pitch choice exists specifically to keep this false).

    ``mech.axial_displacement_m == 0`` is degenerate (no signal to discriminate against a
    leak of zero); raises rather than returning a spurious infinite contrast.
    """
    if mech.axial_displacement_m <= 0.0:
        raise ValueError(
            f"mech.axial_displacement_m must be positive to score separability, got "
            f"{mech.axial_displacement_m!r}"
        )
    if min_contrast_db < 0.0:
        raise ValueError(f"min_contrast_db must be >= 0, got {min_contrast_db!r}")

    leak_frac = sidelobe_leakage_fraction(separation_m, aperture)
    own = mech.axial_displacement_m
    leaked = own * leak_frac

    contrast_db = 20.0 * math.log10(own / leaked) if leaked > 0.0 else math.inf
    grating = aperture.has_grating_lobe
    separable = (contrast_db >= min_contrast_db) and not grating

    provenance = extend(
        mech.provenance,
        "voxel-pair separability (Act 1 spatial-resolution claim): a neighbouring "
        "voxel's own displacement, leaked through the beam's off-axis response, "
        "competes with the target voxel's own signal -- NOT with the acoustic noise "
        "floor (a source-side contaminant, like the incoherent pedestal)",
        f"beam lateral width w_lat = F#*lambda = {aperture.lateral_width_m * 1e3:.4g} mm "
        f"(F# = {aperture.f_number}, lambda = {aperture.wavelength_m * 1e3:.4g} mm)",
        f"separation = {separation_m * 1e3:.4g} mm -> sidelobe leakage fraction = "
        f"{leak_frac:.4g} (unapodized sinc main lobe past its first null, capped at the "
        f"apodized sidelobe floor {aperture.sidelobe_db} dB)",
        f"own signal = {own * 1e9:.4g} nm, leaked neighbour signal = {leaked * 1e9:.4g} nm "
        f"(worst case: neighbour scored at the SAME displacement as the target)",
        f"contrast = {contrast_db:.4g} dB vs required {min_contrast_db} dB"
        + (
            "; GRATING LOBE PRESENT (element pitch exceeds lambda/2): full-amplitude "
            "leakage the sidelobe contrast margin does not capture"
            if grating
            else "; no grating lobe (pitch within lambda/2)"
        ),
    )
    return SeparabilityVerdict(
        separation_m=separation_m,
        lateral_width_m=aperture.lateral_width_m,
        sidelobe_leakage_fraction=leak_frac,
        has_grating_lobe=grating,
        own_signal_m=own,
        leaked_signal_m=leaked,
        contrast_db=contrast_db,
        separable=separable,
        provenance=provenance,
    )


def min_separable_distance(
    mech: MechanicalDisplacement,
    aperture: BeamAperture,
    *,
    min_contrast_db: float = 6.0,
    hi_m: float = 0.05,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Smallest voxel-pair spacing clearing ``min_contrast_db`` (bisection on separation).

    The sidelobe leakage fraction is monotone non-increasing in ``separation_m`` (the
    envelope only falls off-axis), so contrast is monotone non-decreasing in separation
    and a single bisection locates the threshold. ``hi_m`` (default 5 cm) is a generous
    upper search bound -- far past any plausible cortical voxel spacing -- so the search
    always brackets a real crossing unless the aperture cannot separate anything within a
    physically sane range, in which case ``hi_m`` is returned as the (unresolved) bound.

    Does not check the grating-lobe condition (that is a threshold on the APERTURE, not
    on separation -- see :attr:`BeamAperture.has_grating_lobe`, checked once, not swept).
    """
    lo, hi = 0.0, hi_m
    if voxel_pair_separability(
        mech, aperture, hi, min_contrast_db=min_contrast_db
    ).contrast_db < min_contrast_db:
        return hi_m
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        verdict = voxel_pair_separability(mech, aperture, mid, min_contrast_db=min_contrast_db)
        if verdict.contrast_db >= min_contrast_db:
            hi = mid
        else:
            lo = mid
        if (hi - lo) <= tol:
            break
    return hi
