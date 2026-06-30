"""Provenance records and their propagation (Design Invariant 6).

Provenance travels with every effective quantity so the final contrast number is
decomposable back to its inputs: a reviewer can contest a single named assumption
without anything being buried. Each module that transforms a quantity must
``extend`` the provenance with the assumptions it introduced; a module that
combines several inputs must ``merge`` their provenances.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from base_neural_model.base.bands import Band


@dataclass(frozen=True)
class Provenance:
    """Travels with every effective quantity.

    Lets the final number be traced back to its assumptions (Invariant 6).
    """

    source: str                      # e.g. "Ling et al. 2020, optical AP displacement"
    assumptions: tuple[str, ...]
    band: Band

    def with_assumptions(self, *added: str) -> Provenance:
        """Return a copy with ``added`` assumptions appended (band/source kept)."""
        return replace(self, assumptions=self.assumptions + tuple(added))


def extend(prov: Provenance, *added: str, band: Band | None = None) -> Provenance:
    """Propagate provenance through a transforming module.

    Appends the assumptions this module introduced, and optionally re-labels the
    band when the transform changes it. Source is preserved so the original
    literature citation always survives to the end of the chain.
    """
    out = prov.with_assumptions(*added)
    if band is not None:
        out = replace(out, band=band)
    return out


def merge(*provs: Provenance, source: str | None = None) -> Provenance:
    """Combine the provenances of several inputs into one.

    Used where a module consumes multiple provenanced quantities. Assumptions are
    concatenated (order-preserving, de-duplicated). All inputs must share a band -
    a band mismatch here means an envelope quantity leaked into a content-band
    computation (Invariant 3) and is raised as such.
    """
    if not provs:
        raise ValueError("merge requires at least one Provenance")

    bands = {p.band for p in provs}
    if len(bands) != 1:
        raise ValueError(
            f"cannot merge provenances across bands {sorted(b.value for b in bands)}; "
            f"an envelope quantity may have leaked into a content-band path (Invariant 3)"
        )

    seen: set[str] = set()
    merged_assumptions: list[str] = []
    for p in provs:
        for a in p.assumptions:
            if a not in seen:
                seen.add(a)
                merged_assumptions.append(a)

    merged_source = source if source is not None else " + ".join(p.source for p in provs)
    return Provenance(
        source=merged_source,
        assumptions=tuple(merged_assumptions),
        band=next(iter(bands)),
    )
