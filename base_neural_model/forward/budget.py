r"""Acoustic SNR budget sweep -- turning the single Gate-A point into a curve.

:mod:`~base_neural_model.forward.detection` produces ONE detectability verdict at one
acquisition. This module sweeps the acoustic budget axes around it so the Gate-A /
Gate-B question becomes a curve and a dominant-axis ranking rather than a single number:

* **Where is the wall?** SNR vs **skull two-way loss** at a few echo-SNR levels -- the
  spec's Gate-B "report which denominator dominates" deliverable, made visual.
* **Which axis carries the verdict?** A one-at-a-time local sensitivity over the four
  acquisition knobs (echo SNR, skull loss, epoch, residual clutter), reported as the dB
  change per axis across its literature span. The source side already collapses onto eta
  and s (``model/sensitivity.py``); this is the acoustic-side complement, and it is
  expected to collapse onto **skull loss** (it enters the floor as a squared, two-way,
  exponential-in-dB term).

The sweep holds the SOURCE displacement fixed (a single ``MechanicalDisplacement``, e.g.
the motor-demo static dz) and varies only the acquisition -- so it isolates the acoustic
budget from the source physics, which is swept separately upstream.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from base_neural_model.base.provenance import Provenance
from base_neural_model.base.types import MechanicalDisplacement
from base_neural_model.forward.detection import AcquisitionParams, detection_budget


@dataclass(frozen=True, eq=False)
class BudgetCurve:
    """A 1-D acoustic sweep: SNR (dB) vs one acquisition axis (ndarrays -> eq=False)."""

    axis_name: str                  # the swept acquisition field
    axis_values: np.ndarray         # the swept values (axis units)
    snr_db: np.ndarray              # detectability in dB at each value
    surviving_dz_m: np.ndarray      # surviving displacement at each value (m)
    floor_m: np.ndarray             # binding floor at each value (m)
    limiting_denominator: tuple[str, ...]  # "echo_snr" | "clutter" at each value
    provenance: Provenance

    @property
    def crossing_value(self) -> float | None:
        """The axis value where SNR crosses 0 dB (signal == floor), if it is crossed.

        Linearly interpolated on the (monotone) SNR-vs-axis curve. ``None`` if the
        curve never crosses 0 dB over the swept range.
        """
        s = self.snr_db
        if np.all(s > 0) or np.all(s < 0):
            return None
        # First index where the sign changes.
        sign = np.sign(s)
        idx = int(np.argmax(sign[:-1] != sign[1:]))
        x0, x1 = self.axis_values[idx], self.axis_values[idx + 1]
        y0, y1 = s[idx], s[idx + 1]
        return float(x0 - y0 * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(x0)


def sweep_axis(
    mech: MechanicalDisplacement,
    base_acq: AcquisitionParams,
    axis_name: str,
    values: np.ndarray,
    *,
    residual_clutter_m: float = 0.0,
) -> BudgetCurve:
    """Sweep one :class:`AcquisitionParams` field, holding the source dz fixed.

    ``axis_name`` is any numeric ``AcquisitionParams`` field (e.g.
    ``"skull_loss_db_oneway"``, ``"echo_snr_linear"``, ``"epoch_s"``). Each value
    produces a :class:`~base_neural_model.forward.detection.DetectionBudget`; the curve
    collects the dB SNR, surviving displacement, floor, and binding denominator.
    """
    vals = np.asarray(values, dtype=float)
    if vals.ndim != 1 or vals.size == 0:
        raise ValueError(f"values must be a non-empty 1-D array, got shape {vals.shape}")
    if not hasattr(base_acq, axis_name):
        raise ValueError(f"{axis_name!r} is not an AcquisitionParams field")

    snr_db = np.empty(vals.shape, dtype=float)
    surviving = np.empty(vals.shape, dtype=float)
    floor = np.empty(vals.shape, dtype=float)
    denom: list[str] = []
    for i, v in enumerate(vals):
        acq = replace(base_acq, **{axis_name: float(v)})
        b = detection_budget(mech, acq, residual_clutter_m=residual_clutter_m)
        snr_db[i] = b.snr_db
        surviving[i] = b.surviving_dz_m
        floor[i] = b.floor_m
        denom.append(b.limiting_denominator)

    provenance = Provenance(
        source="acoustic SNR budget sweep over an AcquisitionParams axis "
        "(source displacement held fixed)",
        assumptions=(
            f"swept axis = {axis_name} over [{vals.min():.4g}, {vals.max():.4g}], "
            f"{vals.size} points",
            f"residual clutter floor = {residual_clutter_m * 1e9:.4g} nm",
            "SNR = surviving displacement (dz * survival * sqrt(N_ens)) / binding floor",
        ),
        band=mech.band,
    )
    return BudgetCurve(
        axis_name=axis_name,
        axis_values=vals,
        snr_db=snr_db,
        surviving_dz_m=surviving,
        floor_m=floor,
        limiting_denominator=tuple(denom),
        provenance=provenance,
    )


@dataclass(frozen=True)
class AxisSensitivity:
    """One acquisition axis's local influence on the dB verdict over its span."""

    axis_name: str
    lo: float
    hi: float
    snr_db_lo: float
    snr_db_hi: float

    @property
    def db_span(self) -> float:
        """Total dB swing of the verdict across this axis's span (absolute)."""
        return abs(self.snr_db_hi - self.snr_db_lo)


# Literature-ish spans for the four acquisition axes (the acoustic-side analogue of the
# Sobol bounds). echo SNR ~20-40 dB free-field; skull one-way ~4-20 dB at the temporal
# window; epoch spans speech-unit to motor-imagery; clutter as a fraction of the phase
# floor handled separately (it is not an AcquisitionParams field).
DEFAULT_AXIS_SPANS: dict[str, tuple[float, float]] = {
    "echo_snr_linear": (10.0 ** (20.0 / 10.0), 10.0 ** (40.0 / 10.0)),  # 20-40 dB
    "skull_loss_db_oneway": (4.0, 20.0),     # thin temporal bone .. thick/aberrated
    "epoch_s": (0.05, 2.0),                  # speech unit .. long motor-imagery epoch
    "frame_rate_hz": (1000.0, 8000.0),       # ultrafast plane/diverging-wave range
}


def acoustic_axis_ranking(
    mech: MechanicalDisplacement,
    base_acq: AcquisitionParams,
    *,
    spans: dict[str, tuple[float, float]] | None = None,
    residual_clutter_m: float = 0.0,
) -> tuple[AxisSensitivity, ...]:
    """Rank the acquisition axes by how much each moves the dB verdict over its span.

    A one-at-a-time local sensitivity: each axis is moved from the low to the high end
    of its literature span (others held at ``base_acq``), and the resulting dB swing is
    the axis's influence. Returned sorted by ``db_span`` descending -- the acoustic-side
    "which axis carries the verdict" ranking, expected to lead with skull loss.
    """
    spans = spans or DEFAULT_AXIS_SPANS
    out: list[AxisSensitivity] = []
    for axis, (lo, hi) in spans.items():
        curve = sweep_axis(
            mech, base_acq, axis, np.array([lo, hi]),
            residual_clutter_m=residual_clutter_m,
        )
        out.append(
            AxisSensitivity(
                axis_name=axis, lo=lo, hi=hi,
                snr_db_lo=float(curve.snr_db[0]),
                snr_db_hi=float(curve.snr_db[1]),
            )
        )
    return tuple(sorted(out, key=lambda a: a.db_span, reverse=True))
