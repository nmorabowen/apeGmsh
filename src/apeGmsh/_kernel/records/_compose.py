"""apeGmsh._kernel.records._compose — Frozen ``ComposeRecord``.

A :class:`ComposeRecord` is the immutable, ergonomic view of one
composed source-module entry that lives on the broker as part of
:class:`apeGmsh._kernel.record_sets.ComposeSet` (exposed on
``fem.composed_from``).

The record carries the provenance attributes specified by ADR 0038
§"Schema" for the ``/fem/composed_from/{label}/`` H5 sub-group:
``source_path``, ``source_fem_hash``, ``source_neutral_schema_version``,
``translate``, optional ``rotate``, optional ``partition_rank``,
``composed_at`` (ISO timestamp) and an optional free-form ``properties``
mapping.

Phase 3A.1 introduces this record + its companion set so the
neutral-schema bump 2.8.0 → 2.9.0 can round-trip composition metadata.
Phase 3B's ``Compose`` facade is the producer; Phase 3D's H5 reader
adapter consumes it through the ``H5ModelReader`` Protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class ComposeRecord:
    """Immutable provenance entry for one composed source module.

    Parameters
    ----------
    label : str
        The namespace prefix assigned to this source at compose time.
        Used by Phase 3B's tag-rewrite + ``module_label`` parallel
        dataset to mark every row that came from this source.
    source_path : str
        Path to the source ``model.h5`` that contributed this module.
    source_fem_hash : str
        ``fem_hash`` of the source at compose time.  Pre-2.9.0 sources
        are tolerated; the value is opaque to this record.
    source_neutral_schema_version : str
        ``neutral_schema_version`` of the source at compose time
        (e.g. ``"2.9.0"``).  Carried as a string so older sources
        round-trip without parse coercion.
    translate : tuple[float, float, float]
        XYZ translation applied to the source's geometry at compose
        time.  ``(0.0, 0.0, 0.0)`` for an unshifted compose.
    rotate : tuple[float, float, float, float] | None
        Optional quaternion ``(x, y, z, w)`` applied to the source's
        geometry at compose time.  ``None`` when no rotation was
        supplied.
    partition_rank : int | None
        Layer-2 hint per ADR 0038 §"Three-layer rank model" — when the
        source contributed its mesh to a single specific partition
        rank, that integer is stored here.  ``None`` when the source's
        partitions were distributed by the default Layer-1 policy or
        overridden by Layer-3 METIS.
    composed_at : str
        ISO-8601 timestamp captured when this compose entry was
        emitted.  Free-form display string; not parsed.
    properties : Mapping[str, str | int | float] | None
        Optional free-form attribute dict (round-trips through the
        ``/fem/composed_from/{label}/properties/`` sub-attribute
        group).  ``None`` and ``{}`` are equivalent; the canonical
        round-trip uses ``{}`` after read.
    """

    label: str
    source_path: str
    source_fem_hash: str
    source_neutral_schema_version: str
    translate: tuple[float, float, float]
    rotate: tuple[float, float, float, float] | None = None
    partition_rank: int | None = None
    composed_at: str = ""
    # ``frozen=True`` permits ``field(default_factory=...)`` — the
    # convention used elsewhere in this package (see
    # ``_constraints.py`` ``dofs: list[int] = field(default_factory=list)``).
    properties: Mapping[str, str | int | float] = field(default_factory=dict)

    def __repr__(self) -> str:
        rot = "" if self.rotate is None else ", rotate=..."
        rank = (
            "" if self.partition_rank is None
            else f", partition_rank={self.partition_rank}"
        )
        return (
            f"ComposeRecord(label={self.label!r}, "
            f"source_path={self.source_path!r}, "
            f"translate={self.translate}{rot}{rank})"
        )
