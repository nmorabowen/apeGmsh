"""Per-zone schema versioning + a compatibility floor per zone (ADR 0023,
ADR 0113 (#1303)).

Bump cadence (locked per ADR 0023):

- Patch (Z): fix-only; no schema-shape change. Old readers parse identically.
- Minor (Y): additive changes (new dataset/attr/field; old required fields
  remain), or a semantic change that ships a reader shim keyed on a named
  ``*_FROM`` constant. The current reader opens every minor from the zone's
  floor on; a reader older than the file refuses it loudly (INV-4 below —
  there is no forward tolerance).
- Major (X): breaking changes (removed field, renamed dataset, changed dtype).
  Old readers refuse with SchemaVersionError.

Compatibility floor (ADR 0113 (#1303); it retired ADR 0023's two-version
window, which expired files the readers could still parse):

- Reader at X.Z.* with floor X.F.* accepts X.F.* through X.Z.*; the patch
  is ignored. Each writer owns its floor constant beside its version
  constant; :func:`reader_floor` reads it, as :func:`reader_version` reads
  the version. A floor moves only with a major bump.
- Older minors  -> SchemaVersionError (too old: below the floor)
- Newer minors  -> SchemaVersionError (newer than reader understands; refusing
  is safer than silent tolerance -- INV-4, dual of ADR 0021's lineage
  warn-not-raise)
- Different major -> SchemaVersionError (breaking change)

Per-zone version stamps + one envelope (ADR 0023):

- ``/meta/neutral_schema_version``    -> :data:`NEUTRAL_KEY`
- ``/meta/opensees_schema_version``   -> :data:`OPENSEES_KEY`
- ``/meta/results_schema_version``    -> :data:`RESULTS_KEY`
- ``/meta/geometry_schema_version``   -> :data:`GEOMETRY_KEY` (ADR 0112 D2)
- ``/meta/provenance_schema_version`` -> :data:`PROVENANCE_KEY` (ADR 0112 D3)
- ``/meta/schema_version``            -> :data:`ENVELOPE_KEY` (back-compat only)

Files written before Phase 7a (envelope-only) read via the envelope-fallback
path in :func:`read_zone_version`; that lookup returns the envelope value
when the per-zone key is absent. INV-2: new code must not branch on the
envelope; it exists so one-key readers keep working.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional


__all__ = [
    "ENVELOPE_KEY",
    "GEOMETRY",
    "GEOMETRY_KEY",
    "GEOMETRY_SCHEMA_FLOOR",
    "GEOMETRY_SCHEMA_VERSION",
    "NEUTRAL",
    "NEUTRAL_KEY",
    "OPENSEES",
    "OPENSEES_KEY",
    "PROVENANCE",
    "PROVENANCE_KEY",
    "PROVENANCE_ORIGIN_FROM",
    "PROVENANCE_SCHEMA_FLOOR",
    "PROVENANCE_SCHEMA_VERSION",
    "RESULTS",
    "RESULTS_KEY",
    "SchemaVersion",
    "SchemaVersionError",
    "read_zone_version",
    "reader_floor",
    "reader_version",
    "validate_zone_version",
]


# ---------------------------------------------------------------------------
# Zone identifiers (used by error messages + reader_version dispatch)
# ---------------------------------------------------------------------------

#: Neutral zone identifier (broker-written FEMData snapshot).
NEUTRAL: str = "neutral"

#: OpenSees zone identifier (bridge-written ``/opensees/`` group).
OPENSEES: str = "opensees"

#: Results zone identifier (results-runtime ``/stages/`` group).
RESULTS: str = "results"

#: Geometry zone identifier (``/geometry`` root zone, ADR 0112 D2). It
#: lives in the sibling ``<stem>.geometry.h5`` only (V0 ratification Q1).
GEOMETRY: str = "geometry"

#: Provenance zone identifier (``/provenance`` root zone, ADR 0112 D3).
PROVENANCE: str = "provenance"


# ---------------------------------------------------------------------------
# /meta/ attribute keys
# ---------------------------------------------------------------------------

#: Legacy single envelope key. Back-compat only (ADR 0023 INV-2). New code
#: must not branch on this; the per-zone keys are authoritative.
ENVELOPE_KEY: str = "schema_version"

#: Per-zone key for the neutral zone (ADR 0023 §"Three per-zone version stamps").
NEUTRAL_KEY: str = "neutral_schema_version"

#: Per-zone key for the OpenSees bridge zone.
OPENSEES_KEY: str = "opensees_schema_version"

#: Per-zone key for the results zone (introduced by Phase 4 / ADR 0020).
RESULTS_KEY: str = "results_schema_version"

#: Per-zone key for the geometry zone (ADR 0112 D2, #1304).
GEOMETRY_KEY: str = "geometry_schema_version"

#: Per-zone key for the provenance zone (ADR 0112 D3, #1304).
PROVENANCE_KEY: str = "provenance_schema_version"


# ---------------------------------------------------------------------------
# Writer versions of the zones whose writers import them from here
# ---------------------------------------------------------------------------

#: Current version of the ``/geometry`` zone. Its writer (V2b) imports this
#: constant, so reader and writer share one source (``architecture/h5-schema.md``,
#: "/geometry").
GEOMETRY_SCHEMA_VERSION: str = "1.0.0"

#: Current version of the ``/provenance`` zone. Its writers (V2c, V2d)
#: import this constant (``architecture/h5-schema.md``, "/provenance").
PROVENANCE_SCHEMA_VERSION: str = "1.1.0"  # V2d #1378: additive records/origin

#: Floors of the two zones above (ADR 0113 (#1303)). A zone registered in
#: ``_ZONE_KEY`` gets a floor equal to its first version.
GEOMETRY_SCHEMA_FLOOR: str = "1.0.0"
PROVENANCE_SCHEMA_FLOOR: str = "1.0.0"

#: First ``/provenance`` version whose ``records`` table carries the
#: ``origin`` column (#1378), as a ``(major, minor, patch)`` triple.  The
#: reader requires the column from this version on and fills ``"user"``
#: below it.
PROVENANCE_ORIGIN_FROM: tuple[int, int, int] = (1, 1, 0)


# Internal map zone -> per-zone key. Centralised so callers never spell the
# key directly (ADR 0023 / surgical-change discipline).
_ZONE_KEY: dict[str, str] = {
    NEUTRAL: NEUTRAL_KEY,
    OPENSEES: OPENSEES_KEY,
    RESULTS: RESULTS_KEY,
    GEOMETRY: GEOMETRY_KEY,
    PROVENANCE: PROVENANCE_KEY,
}

# Zones born after the per-zone split. The legacy envelope predates them, so
# it never stands in for their version: an absent key means the zone was not
# written, never "use the envelope" (ADR 0023 INV-2).
_NO_ENVELOPE_ZONES: frozenset[str] = frozenset({GEOMETRY, PROVENANCE})


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchemaVersion:
    """Semver triple ``X.Y.Z``.

    Used everywhere by the schema-version logic (ADR 0023). Compares
    field-wise so ordering is meaningful; ``__str__`` round-trips through
    :meth:`parse`.
    """

    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, s: str) -> "SchemaVersion":
        """Parse a semver-shaped string ``"X.Y.Z"`` or ``"X.Y"``.

        Tolerates ``"X.Y"`` (two-part) by treating the patch as 0. The
        envelope key historically stored ``"1.1"`` for some pre-Phase-4
        results files; treating those as ``(1, 1, 0)`` preserves
        back-compat without a separate code path.

        Raises
        ------
        ValueError
            If ``s`` is empty, has more than three parts, or any part is
            not an integer.
        """
        if not s:
            raise ValueError("SchemaVersion.parse: empty string")
        parts = s.split(".")
        if len(parts) < 2 or len(parts) > 3:
            raise ValueError(
                f"SchemaVersion.parse: {s!r} is not semver-shaped "
                f"(expected X.Y or X.Y.Z)"
            )
        try:
            major = int(parts[0])
            minor = int(parts[1])
            patch = int(parts[2]) if len(parts) == 3 else 0
        except ValueError as exc:
            raise ValueError(
                f"SchemaVersion.parse: {s!r} has non-integer parts"
            ) from exc
        return cls(major=major, minor=minor, patch=patch)

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


class SchemaVersionError(ValueError):
    """Raised when an HDF5 file's zone schema is outside the reader's range.

    Carries the file version, the supported range (floor to current) and
    what to do, per ADR 0023 §"Per-zone read validation" and ADR 0113 (#1303).
    """


# ---------------------------------------------------------------------------
# Reader-side known versions (sourced from the writers' constants)
# ---------------------------------------------------------------------------


def reader_version(zone: str) -> SchemaVersion:
    """Return the current code's writer version for ``zone``.

    Sources the constant from the writer module so the reader and writer
    cannot drift (single source of truth, ADR 0023 surgical-change
    discipline). Imports are local so importing this module is cheap and
    doesn't pull h5py.

    Parameters
    ----------
    zone
        One of the keys of ``_ZONE_KEY``: :data:`NEUTRAL`, :data:`OPENSEES`,
        :data:`RESULTS`, :data:`GEOMETRY` or :data:`PROVENANCE`.

    Raises
    ------
    ValueError
        If ``zone`` is not a known zone identifier.
    """
    if zone == NEUTRAL:
        from ...mesh._femdata_h5_io import NEUTRAL_SCHEMA_VERSION
        return SchemaVersion.parse(NEUTRAL_SCHEMA_VERSION)
    if zone == OPENSEES:
        from ..emitter.h5 import SCHEMA_VERSION
        return SchemaVersion.parse(SCHEMA_VERSION)
    if zone == RESULTS:
        from ...results.schema._versions import RESULTS_SCHEMA_VERSION
        return SchemaVersion.parse(RESULTS_SCHEMA_VERSION)
    if zone == GEOMETRY:
        return SchemaVersion.parse(GEOMETRY_SCHEMA_VERSION)
    if zone == PROVENANCE:
        return SchemaVersion.parse(PROVENANCE_SCHEMA_VERSION)
    raise ValueError(
        f"reader_version: unknown zone {zone!r} "
        f"(expected one of {tuple(_ZONE_KEY)!r})"
    )


def reader_floor(zone: str) -> SchemaVersion:
    """Return the oldest version of ``zone`` the current reader opens.

    Mirrors :func:`reader_version`: the floor is a constant the zone's
    writer owns beside its version constant (ADR 0113 (#1303)).

    Raises
    ------
    ValueError
        If ``zone`` is not a known zone identifier.
    """
    if zone == NEUTRAL:
        from ...mesh._femdata_h5_io import NEUTRAL_SCHEMA_FLOOR
        return SchemaVersion.parse(NEUTRAL_SCHEMA_FLOOR)
    if zone == OPENSEES:
        from ..emitter.h5 import SCHEMA_FLOOR
        return SchemaVersion.parse(SCHEMA_FLOOR)
    if zone == RESULTS:
        from ...results.schema._versions import RESULTS_SCHEMA_FLOOR
        return SchemaVersion.parse(RESULTS_SCHEMA_FLOOR)
    if zone == GEOMETRY:
        return SchemaVersion.parse(GEOMETRY_SCHEMA_FLOOR)
    if zone == PROVENANCE:
        return SchemaVersion.parse(PROVENANCE_SCHEMA_FLOOR)
    raise ValueError(
        f"reader_floor: unknown zone {zone!r} "
        f"(expected one of {tuple(_ZONE_KEY)!r})"
    )


# ---------------------------------------------------------------------------
# Read-side helpers
# ---------------------------------------------------------------------------


def read_zone_version(
    meta_attrs: Mapping[str, object],
    zone: str,
    *,
    envelope_fallback: bool = True,
) -> Optional[SchemaVersion]:
    """Read the per-zone version from ``/meta`` attrs.

    Parameters
    ----------
    meta_attrs
        Mapping of attribute name to value (typically ``f["meta"].attrs``).
    zone
        One of the keys of ``_ZONE_KEY`` (see :func:`reader_version`).
    envelope_fallback
        When the per-zone key is absent and this is true (the default),
        return the value of :data:`ENVELOPE_KEY` instead. This is the
        back-compat path for pre-Phase-7a files (single-stamp legacy).
        ADR 0023 §"Single-stamp legacy files". It never applies to
        :data:`GEOMETRY` or :data:`PROVENANCE`, which postdate the
        envelope: for them an absent key returns ``None``.

    Returns
    -------
    SchemaVersion | None
        ``None`` when no version stamp is present at all (legitimate for
        the results zone on bridge-only files, and for files with no
        ``/meta`` group at all).

    Raises
    ------
    ValueError
        If ``zone`` is not a known zone identifier, or if the version
        string is malformed (not semver-shaped).
    """
    if zone not in _ZONE_KEY:
        raise ValueError(
            f"read_zone_version: unknown zone {zone!r} "
            f"(expected one of {tuple(_ZONE_KEY)!r})"
        )
    per_zone_key = _ZONE_KEY[zone]
    raw: object | None = None
    if per_zone_key in meta_attrs:
        raw = meta_attrs[per_zone_key]
    elif (
        envelope_fallback
        and zone not in _NO_ENVELOPE_ZONES
        and ENVELOPE_KEY in meta_attrs
    ):
        raw = meta_attrs[ENVELOPE_KEY]
    if raw is None:
        return None
    s = _decode(raw)
    if not s:
        return None
    return SchemaVersion.parse(s)


def validate_zone_version(
    file_version: SchemaVersion,
    reader: SchemaVersion,
    *,
    zone: str,
) -> None:
    """Floor check (ADR 0113 (#1303); ADR 0023 INV-3 / INV-4).

    With ``floor = reader_floor(zone)``, accepts iff:

    - ``file.major == reader.major``, and
    - ``floor.minor <= file.minor <= reader.minor`` (the patch is ignored).

    Refuses with :class:`SchemaVersionError`, naming both ends of the
    supported range, on:

    - a different major (either direction);
    - ``file.minor < floor.minor`` (too old: below the floor);
    - ``file.minor > reader.minor`` (newer than this reader; INV-4 —
      silent tolerance is worse than refusing).

    Parameters
    ----------
    file_version
        The version read from the file's per-zone (or envelope) key.
    reader
        The reader code's current version for the same zone (from
        :func:`reader_version`).
    zone
        Zone identifier; selects the floor and labels the message.

    Raises
    ------
    SchemaVersionError
        Whenever the file is outside the reader's supported range.
    ValueError
        If ``zone`` is unknown, or ``reader`` is not at or above the zone's
        floor on the same major (a caller passing an impossible reader).
    """
    floor = reader_floor(zone)
    if floor.major != reader.major or floor.minor > reader.minor:
        raise ValueError(
            f"validate_zone_version: reader {reader} is not at or above "
            f"the {zone} floor {floor} on the same major"
        )
    if file_version.major != reader.major:
        advice = (
            _UPGRADE if file_version.major > reader.major else _REGENERATE
        )
        raise SchemaVersionError(
            _range_msg(zone, file_version, floor, reader,
                       cause="different major", advice=advice)
        )
    if file_version.minor < floor.minor:
        raise SchemaVersionError(
            _range_msg(zone, file_version, floor, reader,
                       cause="too old", advice=_REGENERATE)
        )
    if file_version.minor > reader.minor:
        raise SchemaVersionError(
            _range_msg(zone, file_version, floor, reader,
                       cause="newer than this reader", advice=_UPGRADE)
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _decode(raw: object) -> str:
    """Decode an h5py attr value to ``str``.

    Schema-version attrs are written as scalar strings; some legacy files
    may store them as bytes or 0-D numpy arrays. Centralised so callers
    treat the value as a plain string.
    """
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        return raw
    # h5py sometimes returns 0-D numpy arrays for scalar string attrs.
    try:
        import numpy as np
        if isinstance(raw, np.ndarray):
            if raw.shape == ():
                item = raw.item()
                if isinstance(item, bytes):
                    return item.decode("utf-8", errors="replace")
                return str(item)
    except ImportError:  # pragma: no cover - numpy is a hard dep
        pass
    return str(raw)


_UPGRADE = "Upgrade apeGmsh to read this archive."
_REGENERATE = (
    "Regenerate the file from its script with the current apeGmsh."
)


def _range_msg(
    zone: str,
    file_version: SchemaVersion,
    floor: SchemaVersion,
    reader: SchemaVersion,
    *,
    cause: str,
    advice: str,
) -> str:
    """Build the SchemaVersionError text.

    Names the file's version, the cause, the supported range from the
    floor to the reader ("supports 2.10.x\u20132.34.x") and what to do.
    """
    return (
        f"{zone}_schema_version={file_version}: {cause}: this reader "
        f"supports {floor.major}.{floor.minor}.x\u2013"
        f"{reader.major}.{reader.minor}.x. {advice}"
    )
