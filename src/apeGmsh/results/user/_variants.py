"""Read and write a ``<study>.variants.json``: the sets of one study (ADR 0132 D8).

The JSON Schema is ``architecture/schemas/variants.schema.json``; the rules it
cannot express are in ``architecture/h5-schema.md``, section
"``<study>.variants.json``", and this module applies them per row: unique
labels and paths, the name rule's NFC test, confinement to the manifest's folder
after links and junctions resolve, and that the path names an existing model
file. Reserved sibling suffixes are refused case-insensitively.

A malformed envelope refuses the file (:class:`VariantsRefused`); a bad row
refuses alone (:class:`~apeGmsh.results.user.RowRefusal`), and the manifest still
opens. Python refuses a newer minor before it validates anything (ADR 0113 D2).
The app, never Python, opens a same-major newer manifest read-only.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import secrets
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from ._errors import UserFileRefused, UserWriteRefused
from ._names import name_problem
from ._workspace import RowRefusal, finite_number, strict_json
from ._writer import _fsync, _host_token, _replace, _sweep_dead_partials

__all__ = [
    "VARIANTS_FORMAT",
    "VARIANTS_SCHEMA_FLOOR",
    "VARIANTS_SCHEMA_VERSION",
    "Variant",
    "VariantDef",
    "VariantsManifest",
    "VariantsRefused",
    "VariantsWriteRefused",
    "path_problem",
    "read_variants",
    "write_variants",
]

VARIANTS_FORMAT = "apegmsh-variants"
VARIANTS_SCHEMA_VERSION = "1.0.0"
VARIANTS_SCHEMA_FLOOR = "1.0.0"
SUFFIX = ".variants.json"

Param = Union[str, int, float, bool]

_VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\Z")
_TOP_KEYS = frozenset({"format", "version", "variants"})
_ROW_KEYS = frozenset({"label", "path", "params"})
#: The schema's path rules, one per clause of ``$defs/path`` (case-insensitive suffixes).
_MODEL_FILE = re.compile(r"(^|/)[^/]+\.h5\Z", re.IGNORECASE)
#: ADR 0134 D1 adds ``.particles.h5`` (the seeded particle set), as apeGmshViewer's reader/variants.ts does.
_RESERVED = re.compile(r"\.(geometry|results|user|workspace|particles)\.h5\Z", re.IGNORECASE)
_FORBIDDEN = re.compile(r"[\\:\x00-\x1f\x7f]")
_DOT_SEGMENT = re.compile(r"(^|/)\.{1,2}(/|$)")
_BOM = b"\xef\xbb\xbf"
#: The app's path rule compares case-folded on Windows and macOS.
_FOLD_CASE = sys.platform in ("win32", "darwin")


class VariantsRefused(UserFileRefused):
    """The manifest cannot be read at all. ``reason`` is the file-level word."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


class VariantsWriteRefused(UserWriteRefused):
    """A manifest write refused before replacing anything; the old file is whole."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class Variant:
    """One row that stands: ``path`` as written, ``resolved`` inside the manifest's folder."""

    row: int
    label: str
    path: str
    resolved: Path
    params: Mapping[str, Param]


@dataclass(frozen=True)
class VariantDef:
    """A row to write: ``path`` is relative to the manifest's folder, with ``/``."""

    label: str
    path: str
    params: Mapping[str, Param] = field(default_factory=dict)


@dataclass(frozen=True)
class VariantsManifest:
    """An opened manifest: the rows that stand, in display order, and the refused ones."""

    path: Path
    version: str
    variants: tuple[Variant, ...]
    refused: tuple[RowRefusal, ...]

    def by_label(self, label: str) -> Variant:
        for v in self.variants:
            if v.label == label:
                return v
        for r in self.refused:
            if r.name == label:
                raise KeyError(f"variant {label!r} is refused: {r.reason} ({r.detail})")
        raise KeyError(f"no variant labelled {label!r}")


def path_problem(path: Any) -> Optional[str]:
    """Why ``path`` breaks the schema's ``$defs/path`` form, or ``None``."""
    if not isinstance(path, str):
        return f"{path!r} is not a string"
    if not path or path.startswith("/"):
        return "it is empty or starts with /"
    if _FORBIDDEN.search(path):
        return "it holds a backslash, a colon or a control character"
    if _DOT_SEGMENT.search(path) or "//" in path:
        return "a segment is empty, '.' or '..'"
    if re.search(r"[. ]/", path):
        return "a folder segment ends in a dot or a space"
    if not _MODEL_FILE.search(path):
        return "its last segment is not a non-empty stem and .h5"
    if _RESERVED.search(path):
        return "it names a sibling file (.geometry.h5, .results.h5, .user.h5, .workspace.h5 or .particles.h5)"
    return None


def _version_problem(version: Any) -> Optional[str]:
    if not isinstance(version, str) or _VERSION.match(version) is None:
        return "bad-version"
    major, minor = (int(p) for p in version.split(".")[:2])
    current = [int(p) for p in VARIANTS_SCHEMA_VERSION.split(".")]
    floor = [int(p) for p in VARIANTS_SCHEMA_FLOOR.split(".")]
    if major != current[0]:
        return "major"
    if minor > current[1]:
        return "newer"
    if minor < floor[1]:
        return "too-old"
    return None


def _envelope(doc: Any, where: Path) -> tuple[str, list]:
    if not isinstance(doc, dict) or doc.get("format") != VARIANTS_FORMAT:
        raise VariantsRefused("not-a-manifest", f"{where}: format is not {VARIANTS_FORMAT!r}")
    version = doc.get("version")
    reason = _version_problem(version)
    if reason is not None:  # before anything else is validated (ADR 0113 D2)
        raise VariantsRefused(reason, f"{where}: version {version!r}; this reader opens "
                                      f"{VARIANTS_SCHEMA_FLOOR} to {VARIANTS_SCHEMA_VERSION}")
    extra = sorted(set(doc) - _TOP_KEYS)
    if extra:
        raise VariantsRefused("bad-envelope", f"{where}: the key {extra[0]!r} is not allowed")
    rows = doc.get("variants")
    if not isinstance(rows, list) or not rows:
        raise VariantsRefused("bad-envelope", f"{where}: variants is not a non-empty list")
    return str(version), rows


def _row_problem(raw: Any) -> Optional[tuple[str, str]]:
    """The first schema-level problem of one row, as ``(reason, detail)``."""
    if not isinstance(raw, dict):
        return "bad-param", "the row is not an object"
    extra = sorted(set(raw) - _ROW_KEYS)
    if extra:
        return "bad-param", f"the key {extra[0]!r} is not allowed"
    if "label" not in raw or name_problem(raw["label"]) is not None:
        return "empty-name", f"label {raw.get('label')!r}: {name_problem(raw.get('label'))}"
    params = raw.get("params")
    if not isinstance(params, dict):
        return "bad-param", "params is not an object"
    for key, value in params.items():
        if name_problem(key) is not None:
            return "empty-name", f"params key {key!r}: {name_problem(key)}"
        if not (isinstance(value, (str, bool)) or finite_number(value)):
            return "bad-param", f"params[{key!r}] = {value!r} is not a string, finite number or boolean"
    problem = path_problem(raw.get("path"))
    if problem is not None:
        return "bad-path", f"path {raw.get('path')!r}: {problem}"
    return None


def _resolve(folder: Path, path: str, exists: bool = True) -> tuple[Optional[Path], str, str]:
    """``(resolved, "", "")``, or ``(None, reason, detail)``: confinement, then existence.

    The resolved name is judged again: links, junctions and 8.3 short names
    (``SLABUS~1.H5`` for ``slab.user.h5``) resolve to the file's real name.
    ``exists=False`` is the writer's judgement: a reader checks existence per row.
    """
    base = folder.resolve()
    full = (base / path).resolve()
    if full != base and base not in full.parents:
        return None, "escapes", f"path {path!r} resolves to {full}, outside {base}"
    if exists and not full.is_file():
        return None, "missing", f"path {path!r}: no file {full}"
    if path_problem(full.name) is not None:
        return None, "bad-path", f"path {path!r} resolves to {full.name}, not a model file"
    return full, "", ""


def _plain(value: Any) -> Any:
    """A kept number token as the plain number it reads as."""
    if isinstance(value, _Float):
        return float(value)
    if isinstance(value, _Int):
        return int(value)
    return value


def _judge(
    rows: list, folder: Path, exists: bool = True,
) -> tuple[tuple[Variant, ...], tuple[RowRefusal, ...]]:
    labels = Counter(r["label"] for r in rows
                     if isinstance(r, dict) and name_problem(r.get("label")) is None)
    found: dict[int, tuple[str, Any]] = {}
    refusals: dict[int, RowRefusal] = {}
    for row, raw in enumerate(rows):
        label = raw.get("label") if isinstance(raw, dict) else None
        name = label if isinstance(label, str) else ""
        problem = _row_problem(raw)
        if problem is None and labels[label] > 1:
            problem = ("duplicate", f"{labels[label]} rows are labelled {label!r}")
        if problem is None:
            resolved, reason, detail = _resolve(folder, raw["path"], exists)
            if resolved is None:
                problem = (reason, detail)
            else:
                found[row] = (name, resolved)
        if problem is not None:
            refusals[row] = RowRefusal("variants", row, name, *problem)
    keys = {row: _path_key(resolved) for row, (_, resolved) in found.items()}
    counts = Counter(keys.values())
    for row, key in keys.items():
        if counts[key] > 1:
            refusals[row] = RowRefusal("variants", row, found[row][0], "duplicate-path",
                                       f"{counts[key]} rows name {found[row][1]}")
    variants = tuple(
        Variant(row=row, label=rows[row]["label"], path=rows[row]["path"], resolved=resolved,
                params={k: _plain(v) for k, v in rows[row]["params"].items()})
        for row, (_, resolved) in found.items() if row not in refusals)
    return variants, tuple(refusals[r] for r in sorted(refusals))


def _path_key(path: Path) -> str:
    text = str(path)
    return text.casefold() if _FOLD_CASE else text


class _Float(float):
    """A JSON number with a fraction or exponent, keeping its token (``1e400`` stays)."""

    token: str

    def __new__(cls, token: str) -> "_Float":
        obj = super().__new__(cls, token)
        obj.token = token
        return obj


class _Int(int):
    """A JSON integer, keeping its token."""

    token: str

    def __new__(cls, token: str) -> "_Int":
        obj = super().__new__(cls, token)
        obj.token = token
        return obj


def _load(raw: bytes, where: Path) -> Any:
    if raw.startswith(_BOM):
        raw = raw[len(_BOM):]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VariantsRefused("bad-json", f"{where} is not UTF-8: {exc}") from exc
    try:
        return strict_json(text, finite=False, parse_float=_Float, parse_int=_Int)
    except ValueError as exc:
        raise VariantsRefused("bad-json", f"{where}: {exc}") from exc


def _text(value: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:  # a lone surrogate read from a \\ud800 escape
        return json.dumps(value)
    return json.dumps(value, ensure_ascii=False)


def _dump(value: Any, level: int = 0) -> str:
    """JSON text as ``json.dumps(indent=2)`` writes it, but a kept number token is
    written as it was read, so a carried row comes back verbatim."""
    pad, end = "  " * (level + 1), "  " * level
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = [f"{pad}{_text(k)}: {_dump(v, level + 1)}" for k, v in value.items()]
        return "{\n" + ",\n".join(items) + "\n" + end + "}"
    if isinstance(value, list):
        if not value:
            return "[]"
        return "[\n" + ",\n".join(pad + _dump(v, level + 1) for v in value) + "\n" + end + "]"
    if isinstance(value, (_Float, _Int)):
        return value.token
    if isinstance(value, str):
        return _text(value)
    if value is None or isinstance(value, (bool, int)):
        return json.dumps(value)
    if isinstance(value, float):
        return json.dumps(value, allow_nan=False)
    raise TypeError(f"{value!r} is not JSON")


def read_variants(path: "str | Path") -> VariantsManifest:
    """Open ``<study>.variants.json``: the envelope must hold; each row is judged alone."""
    where = Path(path)
    try:
        raw = where.read_bytes()
    except OSError as exc:
        raise VariantsRefused("no-file", f"{where} cannot be read: {exc}") from exc
    version, rows = _envelope(_load(raw, where), where)
    variants, refused = _judge(rows, where.parent)
    return VariantsManifest(path=where, version=version, variants=variants, refused=refused)


def _digest(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _old_rows(target: Path) -> tuple[list, Optional[str]]:
    """The rows on disk and the digest of the one read they came from."""
    if not target.exists():
        return [], None
    if not target.is_file():
        raise VariantsWriteRefused("not-a-file", f"{target} is not a file")
    raw = target.read_bytes()
    try:
        _, rows = _envelope(_load(raw, target), target)
    except VariantsRefused as exc:
        raise VariantsWriteRefused(
            exc.reason if exc.reason in ("newer", "major") else "unreadable",
            f"{target}: {exc.detail}; it was not overwritten") from exc
    return rows, hashlib.sha256(raw).hexdigest()


def _label(raw: Any) -> Optional[str]:
    label = raw.get("label") if isinstance(raw, dict) else None
    return label if isinstance(label, str) else None


def write_variants(
    path: "str | Path", variants: Sequence[VariantDef] = (), *, remove: Sequence[str] = (),
) -> Path:
    """Create or update ``<study>.variants.json`` through a partial and a replace.

    A given row replaces the row of the same label, in place; a new label goes
    after the rows already there; ``remove`` drops rows by label. Every other row
    is carried verbatim, refused ones included (INV-2). A given row must pass
    every rule but existence, which a reader checks per row: a pipeline may list
    its variants before it runs them. An existing manifest that is newer than this
    writer, or that does not read, is never overwritten.
    """
    target = Path(path)
    if not target.name.lower().endswith(SUFFIX) or len(target.name) == len(SUFFIX):
        raise VariantsWriteRefused("bad-name", f"{target}: a manifest is named <study>{SUFFIX}")
    given: dict[str, dict[str, Any]] = {}
    for v in variants:
        if name_problem(v.label) is not None:
            raise VariantsWriteRefused("empty-name", f"label {v.label!r}: {name_problem(v.label)}")
        if v.label in given:
            raise VariantsWriteRefused("duplicate", f"label {v.label!r} is given twice")
        given[v.label] = {"label": v.label, "path": v.path, "params": dict(v.params)}
    removing = set(remove)
    if removing & set(given):
        raise VariantsWriteRefused("bad-param", f"{sorted(removing & set(given))} written and removed")

    old, digest = _old_rows(target)
    held = Counter(label for label in map(_label, old) if label is not None)
    if removing - set(held):
        raise VariantsWriteRefused("bad-param", f"no variant labelled {sorted(removing - set(held))}")
    twice = sorted(label for label in set(given) | removing if held[label] > 1)
    if twice:  # replacing or removing one would drop the other, a refused row
        raise VariantsWriteRefused("duplicate", f"{twice} label more than one row of {target.name}")

    rows: list[Any] = []
    origin: list[Optional[int]] = []  # the old row index, None for a given row
    placed: set[str] = set()
    for i, raw in enumerate(old):
        label = _label(raw)
        if label in removing:
            continue
        if label is not None and label in given:
            rows.append(given[label])
            origin.append(None)
            placed.add(label)
            continue
        rows.append(raw)
        origin.append(i)
    for label, row in given.items():
        if label not in placed:
            rows.append(row)
            origin.append(None)
    if not rows:
        raise VariantsWriteRefused("bad-envelope", "a manifest holds at least one variant")

    _, was_refused = _judge(old, target.parent, exists=False)
    stood = set(range(len(old))) - {r.row for r in was_refused}
    _, refused = _judge(rows, target.parent, exists=False)
    for r in refused:  # a given row, or a row this write would break
        if origin[r.row] is None or origin[r.row] in stood:
            raise VariantsWriteRefused(r.reason, f"row {r.row} ({r.name!r}): {r.detail}")
    doc = {"format": VARIANTS_FORMAT, "version": VARIANTS_SCHEMA_VERSION, "variants": rows}
    try:
        payload = (_dump(doc) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise VariantsWriteRefused("bad-param", str(exc)) from exc

    partial = target.with_name(
        f"{target.name}.{_host_token()}-{os.getpid()}-{secrets.token_hex(4)}.partial")
    try:
        with open(partial, "xb") as out:
            out.write(payload)
        _fsync(partial)
        back = partial.read_bytes()
        if back != payload or _dump(_load(back, partial)) + "\n" != payload.decode("utf-8"):
            raise VariantsWriteRefused("partial-mismatch", f"{partial} did not read back")
        if _digest(target) != digest:
            raise VariantsWriteRefused(
                "revision-moved", f"{target} changed while this write ran; nothing was replaced")
        _replace(partial, target)
    except FileExistsError as exc:
        raise VariantsWriteRefused("partial-mismatch", f"{partial} exists: {exc}") from exc
    except BaseException:
        with contextlib.suppress(OSError):
            partial.unlink(missing_ok=True)
        raise
    _sweep_dead_partials(target)
    return target
