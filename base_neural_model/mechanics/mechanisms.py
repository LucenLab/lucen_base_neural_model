r"""Combining the co-existing tissue displacement mechanisms, band-separated (S2).

The transduction chain models ONE source of tissue displacement: the direct
**neuromechanical** deformation of firing neurons (the ~0.4 nm AP membrane displacement
summed into a coherent dilatation at the beta content corner). But that is the *smallest*
of the activity-driven mechanical changes in tissue. Two larger ones co-exist, and the
research literature says they colocalize and linearly correlate with the neural activity
while being spatially/temporally distinct (Kim et al. 2024; direct deformation is itself
the target of interferometric transcranial imaging, bioRxiv 2023.10.05.561052):

* **Osmotic / ECS** -- activity drives K+ clearance and extracellular-space shrinkage of
  a few percent (Ostby et al. 2009). A LARGE fractional volume change, but it is water
  *redistributing* from ECS into cells at ~conserved bulk volume, so its **net
  dilatation** (what the beam reads) is small: ``eta_osmotic`` is low, and the strain is
  above the saturation cap (S7). It is SLOW (~100 ms-1 s) -- an envelope signal, not a
  beta carrier.

* **Neurovascular / CBV** -- neural activity drives vasodilation and a cerebral-blood-
  volume increase; blood flows *into* the voxel from outside, so it genuinely ADDS tissue
  volume (``eta_vascular ~ 1``, real net dilatation) and produces a LARGE displacement
  (~um scale, matching pulsatility). It is the basis of functional ultrasound (fUS) and
  BOLD. It is also SLOW (~1 s) -- again an envelope signal, correlated with the activity
  envelope but carrying no beta content.

The honest combination is therefore **band-separated**: the beta **content** band holds
only the tiny direct neuromechanical carrier; the slow **envelope** band holds the large
osmotic + vascular terms. The consequence is the load-bearing trade this whole module
exists to make explicit: the direct beta signal is orders below detectability, while the
envelope is large -- but detecting the envelope is detecting cerebral blood volume, i.e.
ordinary fUS, not the specific fast neuromechanical readout.

**A consistency caveat on the direct term's own eta.** This module discounts the OSMOTIC
mechanism with ``eta_osmotic ~ 0.05`` because activity-driven cell swelling draws water
from the ECS reservoir at ~conserved bulk volume -- a compartment-conservation argument.
The SAME argument applies to the fast neuromechanical term: an AP-locked membrane
expansion is also fed largely by water crossing the membrane from the ECS, so its NET
tissue dilatation should carry a similarly small eta. Yet the direct chain
(``transduction.py``) runs at ``dilatation_eta = 0.5`` -- ~10x more generous than the
osmotic term, with no compartment discount. This is deliberately left as the poroelastic
sweep's job (``poroelastic.py`` sweeps eta across ~0.01..1 and the verdict collapses onto
it), but it means the ~-88 dB direct-term verdict is, if anything, OPTIMISTIC: a
compartment-consistent fast eta (~0.05-0.1) would push the direct beta signal a further
~10-20 dB down. The vascular term legitimately keeps ``eta ~ 1`` because blood adds
volume from OUTSIDE the closed ECS/cell system -- which is exactly why it, and not the
neuromechanical term, is the robust net-volume signal.

**Detection of each band (proper modeling).** In the phase-sensitive *displacement*
readout the model uses, the SVD clutter high-pass (D2) that removes cardiac/respiratory
bulk motion sits at ~1 Hz: the beta carrier (20-30 Hz) survives it, while the slow
hemodynamic envelope sits at/below it and is *removed* -- which is exactly why fUS does
not use phase-displacement but **power-Doppler of red-blood-cell backscatter**, a
different modality outside this budget. So: the envelope amplitude reported here is the
tissue displacement the hemodynamic change produces; whether a given readout can *use* it
depends on the modality, and the phase-displacement readout with a 1 Hz high-pass cannot.
The content-band verdict (direct beta vs the honest floor) is the one this module feeds
into the demo.
"""

from __future__ import annotations

from dataclasses import dataclass

from base_neural_model.base.provenance import Provenance, extend
from base_neural_model.base.types import MechanicalDisplacement, VoxelGeometry
from base_neural_model.mechanics.transduction import _saturate


@dataclass(frozen=True)
class MechanismParams:
    """Amplitudes and net-dilatation fractions of the osmotic and vascular mechanisms.

    Reviewer-contestable inputs (like ``eta`` upstream). Both mechanisms are SLOW
    (envelope band), so they carry no beta content; only their net-dilatation
    displacement matters, and it is set by the strain, the net-dilatation fraction, and
    the saturation cap.
    """

    osmotic_strain: float = 2.0e-2       # ECS-scale fractional volume change (LARGE)
    osmotic_eta: float = 0.05            # net dilatation fraction (mostly redistribution)
    vascular_strain: float = 1.0e-2      # CBV-scale fractional volume change
    vascular_eta: float = 1.0            # real net dilatation (blood adds volume)

    def __post_init__(self) -> None:
        for name in ("osmotic_strain", "vascular_strain"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)!r}")
        for name in ("osmotic_eta", "vascular_eta"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {getattr(self, name)!r}")


@dataclass(frozen=True)
class MechanismDecomposition:
    """The band-separated displacement from all three co-existing source mechanisms."""

    direct_axial_m: float        # neuromechanical, BETA CONTENT band (the specific signal)
    osmotic_axial_m: float       # osmotic/ECS net dilatation, envelope band
    vascular_axial_m: float      # neurovascular/CBV net dilatation, envelope band
    provenance: Provenance

    @property
    def content_band_axial_m(self) -> float:
        """The beta-content-band displacement -- only the direct neuromechanical term."""
        return self.direct_axial_m

    @property
    def envelope_band_axial_m(self) -> float:
        """The slow envelope-band displacement -- osmotic + vascular (hemodynamic)."""
        return self.osmotic_axial_m + self.vascular_axial_m

    @property
    def total_axial_m(self) -> float:
        """Total tissue displacement across all mechanisms and bands."""
        return self.direct_axial_m + self.envelope_band_axial_m


def decompose_mechanisms(
    direct: MechanicalDisplacement,
    geom: VoxelGeometry,
    *,
    confinement_kappa: float,
    mech_params: MechanismParams | None = None,
    saturation_strain: float | None = None,
) -> MechanismDecomposition:
    """Combine the direct neuromechanical term with the osmotic and vascular mechanisms.

    ``direct`` is the transduction-chain output (the beta-content-band neuromechanical
    displacement). The osmotic and vascular terms are computed as ``eta * kappa * L *
    saturated(strain)`` -- the same net-dilatation-across-the-gate relation as the direct
    chain -- and assigned to the envelope band (they are slow). The saturation cap (S7)
    is applied to their large strains, so the osmotic redistribution self-limits.
    """
    mp = mech_params or MechanismParams()
    length_m = geom.extent_axial_m

    eps_osm = _saturate(mp.osmotic_strain, saturation_strain)
    eps_vasc = _saturate(mp.vascular_strain, saturation_strain)
    osmotic = mp.osmotic_eta * confinement_kappa * length_m * eps_osm
    vascular = mp.vascular_eta * confinement_kappa * length_m * eps_vasc

    provenance = extend(
        direct.provenance,
        "mechanism combination (S2): band-separated over the three co-existing tissue "
        "displacement sources -- direct neuromechanical (beta content) + osmotic + "
        "vascular (both slow, envelope band)",
        f"osmotic: strain {mp.osmotic_strain:.3g} -> saturated {eps_osm:.3g}, "
        f"eta_osmotic {mp.osmotic_eta} (mostly ECS redistribution -> small net "
        f"dilatation) -> {osmotic * 1e9:.4g} nm (envelope band)",
        f"vascular: strain {mp.vascular_strain:.3g} -> saturated {eps_vasc:.3g}, "
        f"eta_vascular {mp.vascular_eta} (blood adds volume -> real net dilatation) -> "
        f"{vascular * 1e9:.4g} nm (envelope band; this is the fUS/CBV signal)",
        f"direct neuromechanical (beta content) = {direct.axial_displacement_m * 1e9:.4g} "
        "nm; content-band detectability uses this term (the envelope terms are removed "
        "by the ~1 Hz clutter high-pass in a phase-displacement readout, D2)",
        band=direct.band,
    )
    return MechanismDecomposition(
        direct_axial_m=direct.axial_displacement_m,
        osmotic_axial_m=osmotic,
        vascular_axial_m=vascular,
        provenance=provenance,
    )
