"""Read a ``<stem>.user.h5`` (ADR 0130, Amendment 1).

A :class:`UserFile` holds no open handle: each call opens and closes the file,
so a writer's replace is never blocked by a reader kept in a variable.

Three levels of refusal. A bad ``/meta`` refuses the file. A zone that is
malformed or outside its version range refuses that zone when it is read, and
the other zones still read. A bad row refuses alone, with a reason.
"""
from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import h5py
import numpy as np

from ...opensees._internal.schema_version import (
    COMBINATIONS,
    ENTITIES,
    MODES,
    OPENSEES,
    SPECTRA,
    SchemaVersionError,
    USER_META,
    USER_RESULTS,
    read_zone_version,
    reader_version,
    validate_zone_version,
)
from ._descriptor import Combination, CombinationTable, Refusal, read_generation, resolve
from ._entities import Entities, read_entities
from ._errors import UserFileError, UserFileRefused, ZoneRefused
from ._fields import EntryRefused, ResultEntry, UserResult, read_entry, read_entry_meta
from ._modes import ModeTable, read_modes
from ._names import text_or_empty
from ._spectra import SpectraIndex, read_spectra

__all__ = ["ResultIndex", "UserFile", "read_model_combinations"]

_GENERATION = re.compile(r"g(\d+)\Z")
_ENTRY = re.compile(r"r(\d+)\Z")


@dataclass(frozen=True)
class ResultIndex:
    """The ``/results`` entries in force, plus each refused one."""

    entries: tuple[ResultEntry, ...]
    refused: tuple[Refusal, ...]

    def by_name(self, name: str) -> ResultEntry:
        for entry in self.entries:
            if entry.name == name:
                return entry
        for ref in self.refused:
            if ref.name == name:
                raise KeyError(f"user result {name!r} is refused: {ref.reason} ({ref.detail})")
        raise KeyError(f"no user result named {name!r}")


@contextmanager
def _zone_guard(zone: str) -> Iterator[None]:
    """Anything h5py raises on a damaged zone refuses that zone (a dangling link,
    a dataset where a group belongs). The package's own refusals pass through."""
    try:
        yield
    except (UserFileError, SchemaVersionError):
        raise
    except (IndexError, KeyError, OSError, TypeError, ValueError) as exc:
        raise ZoneRefused(zone, f"it cannot be read: {exc}") from exc


def _text_attr(attrs: h5py.AttributeManager, key: str) -> Optional[str]:
    if key not in attrs:
        return None
    raw = attrs[key]
    if isinstance(raw, np.ndarray):
        if raw.size != 1:
            return ""
        raw = raw.ravel()[0]
    return text_or_empty(raw)


def _zone_version_checked(f: h5py.File, zone: str, root: str) -> bool:
    """Whether the zone is present, refusing a key and a group that disagree.

    ``/meta`` holds each zone's key if and only if the zone is present.
    """
    meta = f["meta"].attrs
    try:
        version = read_zone_version(meta, zone)
    except ValueError as exc:
        raise ZoneRefused(f"/{root}", f"its version key is malformed: {exc}") from exc
    present = root in f
    if version is None and present:
        raise ZoneRefused(f"/{root}", f"/meta has no version key for zone {zone!r}")
    if version is not None and not present:
        raise ZoneRefused(f"/{root}", f"/meta stamps zone {zone!r} but the file has no /{root}")
    if version is None:
        return False
    validate_zone_version(version, reader_version(zone), zone=zone)
    return True


def _current_generation(zone: h5py.Group) -> h5py.Group:
    if not isinstance(zone, h5py.Group) or "current" not in zone.attrs:
        raise ZoneRefused("/combinations", "it has no @current")
    current = np.asarray(zone.attrs["current"])
    if current.size != 1 or current.dtype.kind != "i":
        raise ZoneRefused("/combinations", f"@current is {current!r}, not an integer")
    link = f"g{int(current.ravel()[0])}"
    if link not in zone or not isinstance(zone[link], h5py.Group):
        raise ZoneRefused("/combinations", f"@current names {link}, which it does not hold")
    return zone[link]


def _entry_name(zone: h5py.Group, link: str) -> str:
    """An entry's ``@name`` for a refusal, ``""`` when even that cannot be read."""
    try:
        group = zone[link]
    except (KeyError, OSError):
        return ""
    if isinstance(group, h5py.Group) and "name" in group.attrs:
        return text_or_empty(group.attrs["name"])
    return ""


def read_model_combinations(model_h5: "str | Path") -> list[Combination]:
    """The rows of ``/opensees/combinations/g0`` in a model archive, or ``[]``.

    The zone rides the opensees version (ADR 0130 A3), so that key is checked
    before it is read, and it may be absent: slice VU5 writes it.
    """
    where = "/opensees/combinations"
    with _zone_guard(where):
        with h5py.File(model_h5, "r") as f:
            if "opensees" not in f or "combinations" not in f["opensees"]:
                return []
            meta = f["meta"].attrs if "meta" in f else {}
            version = read_zone_version(meta, OPENSEES)
            if version is None:
                raise ZoneRefused(where, "the archive has no opensees_schema_version")
            validate_zone_version(version, reader_version(OPENSEES), zone=OPENSEES)
            zone = f[where]
            if not isinstance(zone, h5py.Group) or "g0" not in zone:
                raise ZoneRefused(where, "it has no g0")
            return read_generation(zone["g0"], where)


class UserFile:
    """An opened ``<stem>.user.h5``. Reads ``/meta`` here, the zones on demand."""

    def __init__(self, path: "str | Path") -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise UserFileRefused(f"no such file: {self.path}")
        try:
            f = h5py.File(self.path, "r")
        except OSError as exc:
            raise UserFileRefused(f"{self.path} is not a readable HDF5 file: {exc}") from exc
        with f:
            try:
                self._read_meta(f)
            except (UserFileError, SchemaVersionError):
                raise
            except (KeyError, OSError, TypeError, ValueError) as exc:
                raise UserFileRefused(f"{self.path}: /meta cannot be read: {exc}") from exc

    def _read_meta(self, f: h5py.File) -> None:
        if "meta" not in f or not isinstance(f["meta"], h5py.Group):
            raise UserFileRefused(f"{self.path} has no /meta group")
        attrs = f["meta"].attrs
        try:
            version = read_zone_version(attrs, USER_META)
        except ValueError as exc:
            raise UserFileRefused(f"{self.path}: bad {USER_META} version: {exc}") from exc
        if version is None:
            raise UserFileRefused(
                f"{self.path} has no /meta@user_meta_schema_version: it is not a .user.h5")
        validate_zone_version(version, reader_version(USER_META), zone=USER_META)
        fem_hash = _text_attr(attrs, "fem_hash")
        if not fem_hash:
            raise UserFileRefused(f"{self.path}: /meta@fem_hash is required")
        self.fem_hash: str = fem_hash
        self.model_hash: Optional[str] = _text_attr(attrs, "model_hash") or None
        self.results_hash: Optional[str] = _text_attr(attrs, "results_hash") or None
        self.writer: Optional[str] = _text_attr(attrs, "writer")
        self.units: Optional[str] = _text_attr(attrs, "units")
        revision = np.asarray(attrs["revision"]) if "revision" in attrs else None
        if revision is None or revision.size != 1 or revision.dtype.kind != "i":
            raise UserFileRefused(f"{self.path}: /meta@revision is not an integer")
        self.revision: int = int(revision.ravel()[0])
        self.zones: frozenset[str] = frozenset(k for k in f if k != "meta")

    # ---- /combinations -------------------------------------------------

    def combinations(self, *, model_h5: "Optional[str | Path]" = None) -> CombinationTable:
        """The current generation, judged row by row.

        ``model_h5`` brings its ``/opensees/combinations`` into the namespace:
        a name in both files refuses both rows, and a row here may name a row
        there. Raises :class:`ZoneRefused` or ``SchemaVersionError`` when the
        zone itself cannot be read; the file stays open for the other zones.
        """
        model_rows = read_model_combinations(model_h5) if model_h5 is not None else []
        with _zone_guard("/combinations"), h5py.File(self.path, "r") as f:
            if not _zone_version_checked(f, COMBINATIONS, "combinations"):
                return resolve([], model_rows)
            rows = read_generation(_current_generation(f["combinations"]))
        return resolve(rows, model_rows)

    def generations(self) -> tuple[int, ...]:
        """Every generation the file holds, oldest first: the backup of A4."""
        with _zone_guard("/combinations"), h5py.File(self.path, "r") as f:
            if not _zone_version_checked(f, COMBINATIONS, "combinations"):
                return ()
            found = (_GENERATION.match(k) for k in f["combinations"])
            return tuple(sorted(int(m.group(1)) for m in found if m))

    # ---- /results ------------------------------------------------------

    def results(self) -> ResultIndex:
        """Every entry, judged without reading ``DATA``. A bad one refuses alone."""
        with _zone_guard("/results"), h5py.File(self.path, "r") as f:
            if not _zone_version_checked(f, USER_RESULTS, "results"):
                return ResultIndex((), ())
            zone = f["results"]
            if not isinstance(zone, h5py.Group):
                raise ZoneRefused("/results", "it is not a group")
            links = sorted(
                (k for k in zone if _ENTRY.match(k)), key=lambda k: int(k[1:]))
            entries: list[ResultEntry] = []
            refused: list[Refusal] = []
            for at, link in enumerate(links):
                try:
                    group = zone[link]
                    if not isinstance(group, h5py.Group):
                        raise EntryRefused("bad-param", f"{group.name} is not a group")
                    entries.append(read_entry_meta(link, group))
                except ZoneRefused:  # a non-UTF-8 table cell refuses the zone
                    raise
                except EntryRefused as exc:
                    refused.append(Refusal(_entry_name(zone, link), exc.reason, exc.detail, row=at))
                except (IndexError, KeyError, OSError, TypeError, ValueError) as exc:
                    # What h5py raises on a damaged entry (a broken link, a
                    # dataset where a group belongs, an attribute of the wrong
                    # type) refuses that entry; the others still read.
                    refused.append(Refusal(_entry_name(zone, link), "bad-param", str(exc), row=at))
        counts: dict[str, int] = {}
        for entry in entries:
            counts[entry.name] = counts.get(entry.name, 0) + 1
        for entry in entries:
            if counts[entry.name] > 1:
                refused.append(Refusal(
                    entry.name, "duplicate",
                    f"{counts[entry.name]} user results are named {entry.name!r}",
                    row=links.index(entry.link),
                ))
        return ResultIndex(
            tuple(e for e in entries if counts[e.name] == 1), tuple(refused))

    def read_result(self, name: str) -> UserResult:
        """Load one entry by name, ``DATA`` included."""
        entry = self.results().by_name(name)
        with h5py.File(self.path, "r") as f:
            return read_entry(entry, f["results"][entry.link])

    # ---- /modes, /spectra and /entities --------------------------------

    def modes(self) -> ModeTable:
        """The ``/modes`` table. Raises :class:`ZoneRefused` when the file holds
        none or it is malformed, ``SchemaVersionError`` outside its version range."""
        with _zone_guard("/modes"), h5py.File(self.path, "r") as f:
            if not _zone_version_checked(f, MODES, "modes"):
                raise ZoneRefused("/modes", "this file holds no modal table")
            return read_modes(f["modes"])

    def spectra(self) -> SpectraIndex:
        """Every ``/spectra`` case, values included. A bad case refuses alone."""
        with _zone_guard("/spectra"), h5py.File(self.path, "r") as f:
            if not _zone_version_checked(f, SPECTRA, "spectra"):
                return SpectraIndex((), ())
            zone = f["spectra"]
            if not isinstance(zone, h5py.Group):
                raise ZoneRefused("/spectra", "it is not a group")
            return read_spectra(zone)

    def entities(self) -> Entities:
        """The ``/entities`` cuts, storeys and point sets; empty tables when the
        file holds none. Raises :class:`ZoneRefused` on a malformed zone,
        ``SchemaVersionError`` outside its version range."""
        with _zone_guard("/entities"), h5py.File(self.path, "r") as f:
            if not _zone_version_checked(f, ENTITIES, "entities"):
                return Entities()
            return read_entities(f["entities"])
