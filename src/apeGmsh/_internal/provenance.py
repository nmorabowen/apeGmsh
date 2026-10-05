"""Declaration provenance: where in the user's source each declaration was made.

ADR 0112 D3 (V0 decisions 11-14, ``architecture/h5-schema.md``,
"/provenance").  Every user declaration that reaches a capture choke
point gets one record keyed by its **declaration path**
``<zone>/<family>/<name|#k>``, never by an HDF5 group name or a tag.  A
record points at two frames:

* ``site``: the first frame outside apeGmsh and the standard library,
  the line that made the call (inside a user helper, if there is one);
* ``script``: the outermost ``__main__`` frame of the user code around
  ``site``, the script line that started it.

The store is deduplicated into three tables, ``files (path, sha256,
kind)``, ``sites (file, line, function)`` and ``records (path, site,
script, seq)``, so 1,000 declarations from one loop cost one file row
and one site row.

**One record per user call.**  A user call is identified by its *entry
frame*: the apeGmsh (or stdlib) frame the site frame called.  Every
choke point reached again under the same entry frame (a geometry
``_register`` that also creates its label, a builder that creates ten
physical groups, a verb that stores a def per segment) is a call
apeGmsh synthesised and records nothing; the first capture of the call
is its record.  The module holds a strong reference to the last entry
frame (one slot for the process, not one per store), so a finished
call's frame object cannot be recycled into a false match, and no store
keeps its session alive.  A named path that already has a record keeps its first record
(a label merged into, a physical group appended to).

There is no opt-out (V0 Q6): the overhead invariant is at most 50 ms per
1,000 registrations, held by ``tests/test_provenance_capture.py``.

Sessions own their store through :func:`store_for`; the bridge (V2d)
holds its own :class:`ProvenanceStore` and calls
:meth:`ProvenanceStore.capture` at ``apeSees._register``.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
import sysconfig
import weakref
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import NamedTuple

__all__ = [
    "FileRow",
    "ProvenanceOverflowError",
    "ProvenanceStore",
    "ProvenanceTable",
    "RecordRow",
    "SiteRow",
    "SourceLocation",
    "base_dir_for",
    "capture",
    "decode_columns",
    "encode_columns",
    "store_for",
    "table_for",
]

#: Largest value an ``int32`` column holds (``h5-schema.md``, "Integer
#: policy for the ADR 0112 zones").
INT32_MAX: int = 2**31 - 1


class ProvenanceOverflowError(ValueError):
    """A ``/provenance`` value does not fit ``int32``; nothing was written."""


# ---------------------------------------------------------------------------
# Rows and the immutable table
# ---------------------------------------------------------------------------


class FileRow(NamedTuple):
    """A source file.  ``path`` is absolute POSIX in memory (pseudo-files
    such as ``<string>`` keep their name); ``sha256`` is ``""`` when the
    source cannot be read; ``kind`` is ``script`` or ``module``."""

    path: str
    sha256: str
    kind: str


class SiteRow(NamedTuple):
    """A source line: ``file`` is a row of ``files``, ``line`` 1-based."""

    file: int
    line: int
    function: str


class RecordRow(NamedTuple):
    """One declaration.  ``site`` and ``script`` are rows of ``sites``
    (-1 when no such frame exists); ``seq`` is the 0-based capture order.
    ``origin`` is ``"user"`` for a declaration the user made and
    ``"synthesised"`` for an object apeGmsh created inside a verb the
    user called (schema 1.1.0; a 1.0.0 file reads as ``"user"``)."""

    path: str
    site: int
    script: int
    seq: int
    origin: str = "user"


class SourceLocation(NamedTuple):
    """A site resolved to its file: what go-to-source opens."""

    path: str
    sha256: str
    line: int
    function: str


@dataclass(frozen=True)
class ProvenanceTable:
    """The immutable snapshot a :class:`~apeGmsh.mesh.FEMData.FEMData`
    carries and ``/provenance`` round-trips.  No hash reads it."""

    files: tuple[FileRow, ...] = ()
    sites: tuple[SiteRow, ...] = ()
    records: tuple[RecordRow, ...] = ()

    def record(self, path: str) -> RecordRow:
        """The record of declaration ``path``; ``KeyError`` if none."""
        for row in self.records:
            if row.path == path:
                return row
        raise KeyError(f"no provenance record for declaration {path!r}")

    def location(self, site: int) -> SourceLocation:
        """Resolve a ``sites`` row; ``site`` must be a valid row (not -1)."""
        if not 0 <= site < len(self.sites):
            raise IndexError(
                f"provenance site row {site} out of range "
                f"(0..{len(self.sites) - 1})")
        s = self.sites[site]
        f = self.files[s.file]
        return SourceLocation(f.path, f.sha256, s.line, s.function)


# ---------------------------------------------------------------------------
# Frame classification
# ---------------------------------------------------------------------------

_APEGMSH, _STDLIB, _THIRD_PARTY, _USER = 0, 1, 2, 3

_PKG_ROOT = os.path.normcase(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + os.sep


def _roots(*names: str) -> tuple[str, ...]:
    paths = sysconfig.get_paths()
    out = {os.path.normcase(os.path.abspath(paths[n])) + os.sep
           for n in names if paths.get(n)}
    return tuple(sorted(out))


# ``platstdlib`` is the venv's ``Lib`` on Windows, which holds
# site-packages, so site-packages is tested first.
_STDLIB_ROOTS = _roots("stdlib", "platstdlib") + (
    os.path.normcase(os.path.dirname(os.__file__)) + os.sep,)
_SITE_ROOTS = _roots("purelib", "platlib")

_CLASS_CACHE: dict[str, int] = {}


def _classify(filename: str) -> int:
    cls = _CLASS_CACHE.get(filename)
    if cls is not None:
        return cls
    if filename.startswith("<"):
        # ``<frozen runpy>`` and friends are the stdlib; ``<string>``,
        # ``<stdin>`` and notebook cells are the user's.
        cls = _STDLIB if filename.startswith("<frozen ") else _USER
    else:
        p = os.path.normcase(os.path.abspath(filename))
        if p.startswith(_PKG_ROOT):
            cls = _APEGMSH
        elif p.startswith(_SITE_ROOTS):
            cls = _THIRD_PARTY
        elif p.startswith(_STDLIB_ROOTS):
            cls = _STDLIB
        else:
            cls = _USER
    _CLASS_CACHE[filename] = cls
    return cls


def _frame_class(f: FrameType) -> int:
    """``_classify`` by file, except that a ``__main__`` frame under the
    apeGmsh tree is the user's: a script run in place there is still the
    user's script.  Launchers that run as ``__main__`` stay what they are:
    stdlib (``python -m cProfile|pdb|trace|profile``) or site-packages
    (``pytest``, ``ipykernel``)."""
    cls = _classify(f.f_code.co_filename)
    if cls == _APEGMSH and f.f_globals.get("__name__") == "__main__":
        return _USER
    return cls


def _capture_frames(
    start: FrameType | None,
) -> tuple[FrameType | None, FrameType | None, FrameType | None]:
    """Return ``(site, entry, script)`` walking out from ``start``.

    Generalises ``opensees/_internal/build.py::_stacklevel_outside_package``
    to two frames.  ``site`` is the first frame outside apeGmsh and the
    stdlib, and ``entry`` the frame it called (``None`` when ``start`` is
    already outside).  ``script`` is the outermost ``__main__`` frame of
    the user code around ``site``.  The walk out from ``site`` passes
    through stdlib frames (``contextlib``, ``functools``, ``runpy``) and
    stops at the first apeGmsh or site-packages frame, so a launcher that
    runs as ``__main__`` from site-packages (``pytest``, ``ipykernel``)
    never claims it.  A pseudo-file (``<string>``) ``__main__`` frame
    never overrides a real-file one already found: ``python -m pdb``
    runs the script through a ``<string>`` trampoline in its own
    ``__main__`` globals.  With no real file around it (``-c``,
    ``<stdin>``, a notebook cell) the pseudo-file frame is the script.
    """
    entry: FrameType | None = None
    f = start
    while f is not None and _frame_class(f) < _THIRD_PARTY:
        entry = f
        f = f.f_back
    site = f
    script: FrameType | None = None
    while f is not None:
        cls = _frame_class(f)
        if cls == _USER:
            if f.f_globals.get("__name__") == "__main__" and not (
                    script is not None
                    and f.f_code.co_filename.startswith("<")
                    and not script.f_code.co_filename.startswith("<")):
                script = f
        elif cls != _STDLIB and f is not site:
            break
        f = f.f_back
    return site, entry, script


# ---------------------------------------------------------------------------
# The mutable store
# ---------------------------------------------------------------------------

#: The entry frame of the last captured user call.  Held strongly so its
#: frame object cannot be freed and reallocated to a later call; one slot
#: for the process bounds what it keeps alive to that call's stack.
_LAST_ENTRY: FrameType | None = None


class ProvenanceStore:
    """The run's provenance tables, filled by :meth:`capture`."""

    def __init__(self) -> None:
        self._files: list[FileRow] = []
        self._file_index: dict[str, int] = {}
        self._sites: list[SiteRow] = []
        self._site_index: dict[tuple[int, int, str], int] = {}
        self._records: dict[str, RecordRow] = {}
        self._unnamed: dict[tuple[str, str], int] = {}

    def __len__(self) -> int:
        return len(self._records)

    def has(self, zone: str, family: str, key: str) -> bool:
        """True iff ``<zone>/<family>/<key>`` already has a record.  The
        pre-allocation check of a writer that must fail loud on a
        collision before it creates anything (``apeSees._register``)."""
        return f"{zone}/{family}/{key}" in self._records

    def next_unnamed_key(self, zone: str, family: str) -> str:
        """The ``#k`` key the next unnamed capture in ``family`` takes."""
        return f"#{self._unnamed.get((zone, family), 0) + 1}"

    def capture(self, zone: str, family: str, name: str | None, *,
                on_existing: str = "keep") -> str | None:
        """Record one declaration made by the current user call.

        ``name`` is the user's name for it; ``None`` or ``""`` gives the
        unnamed key ``#k`` (1-based order among the family's unnamed
        records).  Returns the declaration path recorded, or ``None``
        when the call already has a record.

        A named path that already has a record is, for a session,
        the same declaration touched again (a label merged into, a
        physical group appended to): with ``on_existing="keep"`` the
        first record stays and ``None`` is returned (V2c,
        ``h5-schema.md`` "/provenance").  For a writer whose names share
        one key space with synthesised keys (the bridge), the same path
        is a collision: ``on_existing="raise"`` raises ``ValueError``
        instead.  An unnamed key is never an append, so a ``#k`` that
        already has a record (a user named something ``#k``) always
        raises: a record is never overwritten.
        """
        global _LAST_ENTRY
        for part, what in ((zone, "zone"), (family, "family")):
            if not part or "/" in part:
                raise ValueError(
                    f"provenance {what} must be a non-empty segment "
                    f"without '/', got {part!r}")
        if on_existing not in ("keep", "raise"):
            raise ValueError(
                f"provenance on_existing must be 'keep' or 'raise', "
                f"got {on_existing!r}")
        site, entry, script = _capture_frames(sys._getframe(1))
        if site is not None and entry is not None:
            if entry is _LAST_ENTRY:
                return None
            _LAST_ENTRY = entry
        if name:
            path = f"{zone}/{family}/{name}"
            if path in self._records:
                if on_existing == "keep":
                    return None
                raise ValueError(
                    f"provenance: declaration path {path!r} already has a "
                    "record; the name collides with an existing key")
        else:
            k = self._unnamed.get((zone, family), 0) + 1
            path = f"{zone}/{family}/#{k}"
            if path in self._records:
                raise ValueError(
                    f"provenance: the unnamed key {path!r} already has a "
                    "record (a declaration was named like an ordinal key); "
                    "a record is never overwritten")
            self._unnamed[(zone, family)] = k
        self._records[path] = RecordRow(
            path,
            self._site_row(site) if site is not None else -1,
            self._site_row(script) if script is not None else -1,
            len(self._records),
        )
        return path

    def capture_synthesised(self, zone: str, family: str,
                            key: str) -> str | None:
        """Record an object apeGmsh synthesised inside the current user
        call, under its own key (maintainer ruling on #1378, finding 2).

        ``key`` has the form ``<verb>:<owner>/<role>`` (``support:<stage>/hold``,
        ``imposed_displacement:<name>``); it never uses the family's ``#k``
        counter, so the user's unnamed declarations keep their numbers.
        The record's site is the user's verb call, like any other
        capture, but the one-record-per-call rule does not apply: every
        synthesised object of the call gets its record, and none of them
        claims the call's entry frame.  ``origin`` is ``"synthesised"``.
        Returns the path.  A key that already has a record is a caller
        bug (the keys are built to be unique: an ordinal ``#k`` for an
        unnamed verb call, a ``@n`` suffix for a repeated owner) and
        raises ``ValueError`` rather than dropping either record.
        """
        for part, what in ((zone, "zone"), (family, "family")):
            if not part or "/" in part:
                raise ValueError(
                    f"provenance {what} must be a non-empty segment "
                    f"without '/', got {part!r}")
        if not key or ":" not in key:
            raise ValueError(
                "provenance synthesised key must read '<verb>:<owner>[/<role>]', "
                f"got {key!r}")
        path = f"{zone}/{family}/{key}"
        if path in self._records:
            raise ValueError(
                f"provenance: synthesised key {path!r} already has a record; "
                "the caller must give each synthesised object a unique key")
        site, _entry, script = _capture_frames(sys._getframe(1))
        self._records[path] = RecordRow(
            path,
            self._site_row(site) if site is not None else -1,
            self._site_row(script) if script is not None else -1,
            len(self._records),
            "synthesised",
        )
        return path

    def snapshot(self) -> ProvenanceTable:
        """The tables as they stand, frozen."""
        return ProvenanceTable(
            tuple(self._files), tuple(self._sites),
            tuple(self._records.values()))

    def _site_row(self, frame: FrameType) -> int:
        key = (self._file_row(frame), int(frame.f_lineno or 0),
               frame.f_code.co_name)
        row = self._site_index.get(key)
        if row is None:
            row = self._site_index[key] = len(self._sites)
            self._sites.append(SiteRow(*key))
        return row

    def _file_row(self, frame: FrameType) -> int:
        filename = frame.f_code.co_filename
        row = self._file_index.get(filename)
        if row is not None:
            return row
        if filename.startswith("<"):
            path, digest = filename, ""
        else:
            path = Path(os.path.abspath(filename)).as_posix()
            try:
                digest = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
            except OSError:
                digest = ""
        kind = ("script" if frame.f_globals.get("__name__") == "__main__"
                else "module")
        row = self._file_index[filename] = len(self._files)
        self._files.append(FileRow(path, digest, kind))
        return row


# ---------------------------------------------------------------------------
# Session stores
# ---------------------------------------------------------------------------

_STORES: "weakref.WeakKeyDictionary[object, ProvenanceStore]" = (
    weakref.WeakKeyDictionary())


def _is_session(owner: object) -> bool:
    # Deferred: ``apeGmsh._session`` imports gmsh; this module stays a leaf.
    from apeGmsh._session import _SessionBase
    return isinstance(owner, _SessionBase)


def store_for(session: object) -> ProvenanceStore:
    """The provenance store of ``session`` (created on first use)."""
    if not _is_session(session):
        raise TypeError(
            f"provenance belongs to an apeGmsh session, got "
            f"{type(session).__name__}")
    store = _STORES.get(session)
    if store is None:
        store = _STORES[session] = ProvenanceStore()
    return store


def capture(owner: object, zone: str, family: str,
            name: str | None) -> str | None:
    """Capture a declaration for the session ``owner``.

    A composite whose parent is not a session (a test double) has no
    run to record into: nothing is captured and ``None`` returned.
    """
    if not _is_session(owner):
        return None
    return store_for(owner).capture(zone, family, name)


def table_for(session: object) -> ProvenanceTable:
    """The frozen provenance of ``session`` (what ``from_gmsh`` attaches)."""
    return store_for(session).snapshot()


# ---------------------------------------------------------------------------
# Column encoding for /provenance (int32, refuse on overflow)
# ---------------------------------------------------------------------------

_ABS = re.compile(r"^(/|[A-Za-z]:/)")


def base_dir_for(h5_path: str) -> str:
    """``@base_dir`` for an artifact written to ``h5_path``: its directory,
    absolute POSIX."""
    return Path(os.path.dirname(os.path.abspath(h5_path))).as_posix()


def _relative(path: str, base_dir: str) -> str:
    if not _ABS.match(path):
        return path  # pseudo-file
    try:
        rel = os.path.relpath(path, base_dir)
    except ValueError:  # another drive
        return path
    rel = rel.replace(os.sep, "/")
    if rel == ".." or rel.startswith("../") or _ABS.match(rel):
        return path
    return rel


def _absolute(path: str, base_dir: str) -> str:
    if _ABS.match(path) or path.startswith("<"):
        return path
    return f"{base_dir.rstrip('/')}/{path}"


def _check_int32(table: ProvenanceTable) -> None:
    def refuse(what: str, value: int) -> None:
        raise ProvenanceOverflowError(
            f"/provenance {what} = {value} does not fit int32 "
            f"(limit {INT32_MAX}); refusing to write a truncated table "
            f"(architecture/h5-schema.md, 'Integer policy')")

    for what, n in (("files rows", len(table.files)),
                    ("sites rows", len(table.sites)),
                    ("records rows", len(table.records))):
        if n > INT32_MAX:
            refuse(what, n)
    for s in table.sites:
        for what, v in (("sites/file", s.file), ("sites/line", s.line)):
            if not -1 <= v <= INT32_MAX:
                refuse(what, v)
    for r in table.records:
        for what, v in (("records/site", r.site),
                        ("records/script", r.script),
                        ("records/seq", r.seq)):
            if not -1 <= v <= INT32_MAX:
                refuse(what, v)


def encode_columns(table: ProvenanceTable,
                   base_dir: str) -> dict[str, dict[str, list]]:
    """``{table: {column: values}}`` for ``/provenance``, paths made
    relative to ``base_dir`` when under it.  Raises
    :class:`ProvenanceOverflowError` before anything is written."""
    _check_int32(table)
    return {
        "files": {
            "path": [_relative(f.path, base_dir) for f in table.files],
            "sha256": [f.sha256 for f in table.files],
            "kind": [f.kind for f in table.files],
        },
        "sites": {
            "file": [s.file for s in table.sites],
            "line": [s.line for s in table.sites],
            "function": [s.function for s in table.sites],
        },
        "records": {
            "path": [r.path for r in table.records],
            "site": [r.site for r in table.records],
            "script": [r.script for r in table.records],
            "seq": [r.seq for r in table.records],
            "origin": [r.origin for r in table.records],
        },
    }


def decode_columns(columns: dict[str, dict[str, list]],
                   base_dir: str) -> ProvenanceTable:
    """Inverse of :func:`encode_columns`.  A ``records`` table without
    an ``origin`` column (schema 1.0.0) reads every record as
    ``"user"``."""
    f, s, r = columns["files"], columns["sites"], columns["records"]
    origin = r.get("origin") or ["user"] * len(r["path"])
    return ProvenanceTable(
        tuple(FileRow(_absolute(p, base_dir), h, k)
              for p, h, k in zip(f["path"], f["sha256"], f["kind"])),
        tuple(SiteRow(int(a), int(b), c)
              for a, b, c in zip(s["file"], s["line"], s["function"])),
        tuple(RecordRow(p, int(a), int(b), int(c), str(o))
              for p, a, b, c, o in zip(r["path"], r["site"], r["script"],
                                       r["seq"], origin)),
    )
