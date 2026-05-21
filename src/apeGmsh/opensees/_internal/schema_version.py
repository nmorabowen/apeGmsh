"""Per-zone schema versioning + two-version reader window (ADR 0023).

Bump cadence (locked per ADR 0023):

- Patch (Z): fix-only; no schema-shape change. Old readers parse identically.
- Minor (Y): additive changes (new dataset/attr/field; old required fields remain).
  Old readers continue to parse, ignoring new content. The two-version window
  means the previous minor's readers can still open the file.
- Major (X): breaking changes (removed field, renamed dataset, changed dtype).
  Old readers refuse with SchemaVersionError.

Two-version reader window:

- Reader at X.Y.Z accepts X.Y.* and X.(Y-1).*
- Older minors  -> SchemaVersionError (too old; outside window)
- Newer minors  -> SchemaVersionError (newer than reader understands; refusing
  is safer than silent tolerance -- INV-4, dual of ADR 0021's lineage
  warn-not-raise)
- Different major -> SchemaVersionError (breaking change)

Three per-zone version stamps + one envelope (ADR 0023):

- ``/meta/neutral_schema_version``  -> :data:`NEUTRAL_KEY`
- ``/meta/opensees_schema_version`` -> :data:`OPENSEES_KEY`
- ``/meta/results_schema_version``  -> :data:`RESULTS_KEY`
- ``/meta/schema_version``          -> :data:`ENVELOPE_KEY` (back-compat only)

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
    "NEUTRAL",
    "NEUTRAL_KEY",
    "OPENSEES",
    "OPENSEES_KEY",
    "RESULTS",
    "RESULTS_KEY",
    "SchemaVersion",
    "SchemaVersionError",
    "read_zone_version",
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


# Internal map zone -> per-zone key. Centralised so callers never spell the
# key directly (ADR 0023 / surgical-change discipline).
_ZONE_KEY: dict[str, str] = {
    NEUTRAL: NEUTRAL_KEY,
    OPENSEES: OPENSEES_KEY,
    RESULTS: RESULTS_KEY,
}


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
    """Raised when an HDF5 file's zone schema is outside the reader's window.

    Carries an explicit upgrade-path message (file version + supported range)
    per ADR 0023 §"Per-zone read validation."
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
        One of :data:`NEUTRAL`, :data:`OPENSEES`, :data:`RESULTS`.

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
    raise ValueError(
        f"reader_version: unknown zone {zone!r} "
        f"(expected one of {NEUTRAL!r}, {OPENSEES!r}, {RESULTS!r})"
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
        One of :data:`NEUTRAL`, :data:`OPENSEES`, :data:`RESULTS`.
    envelope_fallback
        When the per-zone key is absent and this is true (the default),
        return the value of :data:`ENVELOPE_KEY` instead. This is the
        back-compat path for pre-Phase-7a files (single-stamp legacy).
        ADR 0023 §"Single-stamp legacy files".

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
            f"(expected one of {NEUTRAL!r}, {OPENSEES!r}, {RESULTS!r})"
        )
    per_zone_key = _ZONE_KEY[zone]
    raw: object | None = None
    if per_zone_key in meta_attrs:
        raw = meta_attrs[per_zone_key]
    elif envelope_fallback and ENVELOPE_KEY in meta_attrs:
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
    """Two-version-window check (ADR 0023 INV-3 / INV-4).

    Accepts:

    - ``file.major == reader.major``
    - ``file.minor in {reader.minor, reader.minor - 1}``

    Refuses (with explicit upgrade-path text) on:

    - Different major (any direction).
    - ``file.minor < reader.minor - 1`` (too old; outside the window).
    - ``file.minor > reader.minor`` (newer than reader understands;
      INV-4 — silent tolerance is worse than refusing).

    Parameters
    ----------
    file_version
        The version read from the file's per-zone (or envelope) key.
    reader
        The reader code's current version for the same zone (from
        :func:`reader_version`).
    zone
        Zone identifier for error-message context.

    Raises
    ------
    SchemaVersionError
        Whenever the file is outside the reader's two-version window.
    """
    supported_low = reader.minor - 1
    supported_high = reader.minor
    if file_version.major != reader.major:
        raise SchemaVersionError(
            _window_msg(zone, file_version, reader, supported_low,
                        supported_high, cause="different major")
        )
    if file_version.minor < supported_low:
        raise SchemaVersionError(
            _window_msg(zone, file_version, reader, supported_low,
                        supported_high, cause="too old; outside window")
        )
    if file_version.minor > supported_high:
        raise SchemaVersionError(
            _window_msg(zone, file_version, reader, supported_low,
                        supported_high, cause="newer than this reader")
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


def _window_msg(
    zone: str,
    file_version: SchemaVersion,
    reader: SchemaVersion,
    low: int,
    high: int,
    *,
    cause: str,
) -> str:
    """Build the SchemaVersionError text.

    Includes the file's version, the reader's supported range, and a
    concise cause. Tests assert that both the file version and the
    supported range appear in the message.
    """
    low_clamped = max(low, 0)
    return (
        f"{zone}_schema_version={file_version}: {cause}. "
        f"This reader supports {reader.major}.{low_clamped}.x-"
        f"{reader.major}.{high}.x. "
        f"Upgrade apeGmsh to read this archive, or re-emit the file "
        f"with the current version."
    )
