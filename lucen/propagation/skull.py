"""CT-derived skull acoustic property maps (spec section 4.2).

Maps a CT-derived skull (or a parameterized skull-mimic for Stage 2) to the
acoustic property fields k-Wave needs: sound speed, density, and attenuation,
plus the temporal-window mask marking the thin-bone entry region.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class SkullAcousticMap:
    """3-D acoustic property maps for the skull region.

    ``eq=False`` because the array fields make a generated ``__eq__`` ambiguous
    and unhashable under NumPy (same correction as :class:`FeasibilityCurve`).
    """

    sound_speed_m_s: np.ndarray         # 3-D map, m/s
    density_kg_m3: np.ndarray           # 3-D map, kg/m^3
    attenuation_db_cm_mhz: np.ndarray   # 3-D map, dB / (cm * MHz)
    grid_spacing_m: float               # isotropic voxel spacing, metres
    temporal_window_mask: np.ndarray    # bool, the thin-bone entry region


def load_skull_map(ct_source: str) -> SkullAcousticMap:
    """Build acoustic property maps from a CT-derived skull or skull-mimic.

    Maps Hounsfield units -> (sound speed, density, attenuation). ``ct_source``
    is a path to a CT volume, or a recognized parameterized skull-mimic spec for
    Stage 2.

    Stub - body is implementation work (spec build step 4).
    """
    raise NotImplementedError(
        "load_skull_map: HU -> (c, rho, alpha) mapping not yet implemented "
        "(spec section 4.2)"
    )
