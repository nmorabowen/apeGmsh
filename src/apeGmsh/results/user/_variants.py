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
_RESERVED = re.compile(r"\.(geometry|results|user|workspace)\.h5\Z", re.IGNORECASE)
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
        return "it names a sibling file (.geometry.h5, .results.h5, .user.h5 or .workspace.h5)"
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


def _resolve(folder: Path, path: str) -> tuple[Optional[Path], str, str]:
    """``(resolved, "", "")``, or ``(None, reason, detail)``: confinement, then existence."""
    base = folder.resolve()
    full = (base / path).resolve()
    if full != base and base not in full.parents:
        return None, "escapes", f"path {path!r} resolves to {full}, outside {base}"
    if not full.is_file():
        return None, "missing", f"path {path!r}: no file {full}"
    if path_problem(full.name) is not None:
        return None, "bad-path", f"path {path!r} resolves to {full.name}, not a model file"
    return full, "", ""


def _judge(rows: list, folder: Path) -> tuple[tuple[Variant, ...], tuple[RowRefusal, ...]]:
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
            resolved, reason, detail = _resolve(folder, raw["path"])
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
                params=dict(rows[row]["params"]))
        for row, (_, resolved) in found.items() if row not in refusals)
    return variants, tuple(refusals[r] for r in sorted(refusals))


def _path_key(path: Path) -> str:
    text = str(path)
    return text.casefold() if _FOLD_CASE else text


def _load(raw: bytes, where: Path) -> Any:
    if raw.startswith(_BOM):
        raw = raw[len(_BOM):]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VariantsRefused("bad-json", f"{where} is not UTF-8: {exc}") from exc
    try:
        return strict_json(text, finite=False)
    except ValueError as exc:
        raise VariantsRefused("bad-json", f"{where}: {exc}") from exc


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


def _signature(path: Path) -> Optional[tuple[int, int, str]]:
    if not path.exists():
        return None
    st = path.stat()
    return st.st_size, st.st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest()


def write_variants(path: "str | Path", variants: Sequence[VariantDef]) -> Path:
    """Write ``<study>.variants.json`` through a partial and a replace, and return it.

    Every row must stand (its file must exist inside the manifest's folder). An
    existing manifest that is newer than this writer, or that does not read, is
    never overwritten.
    """
    target = Path(path)
    if not target.name.lower().endswith(SUFFIX) or len(target.name) == len(SUFFIX):
        raise VariantsWriteRefused("bad-name", f"{target}: a manifest is named <study>{SUFFIX}")
    doc: dict[str, Any] = {"format": VARIANTS_FORMAT, "version": VARIANTS_SCHEMA_VERSION,
           "variants": [{"label": v.label, "path": v.path, "params": dict(v.params)}
                        for v in variants]}
    if not doc["variants"]:
        raise VariantsWriteRefused("bad-envelope", "a manifest holds at least one variant")
    _, refused = _judge(doc["variants"], target.parent)
    if refused:
        first = refused[0]
        raise VariantsWriteRefused(first.reason, f"row {first.row} ({first.name!r}): {first.detail}")
    if target.exists():
        try:
            read_variants(target)
        except VariantsRefused as exc:
            raise VariantsWriteRefused(
                exc.reason if exc.reason in ("newer", "major") else "unreadable",
                f"{target}: {exc.detail}; it was not overwritten") from exc
    before = _signature(target)
    try:
        payload = (json.dumps(doc, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    except ValueError as exc:
        raise VariantsWriteRefused("bad-param", str(exc)) from exc

    partial = target.with_name(
        f"{target.name}.{_host_token()}-{os.getpid()}-{secrets.token_hex(4)}.partial")
    try:
        with open(partial, "xb") as out:
            out.write(payload)
        _fsync(partial)
        if partial.read_bytes() != payload or strict_json(payload.decode()) != doc:
            raise VariantsWriteRefused("partial-mismatch", f"{partial} did not read back")
        if _signature(target) != before:
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
