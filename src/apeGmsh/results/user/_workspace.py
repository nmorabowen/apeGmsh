"""Read and write a ``<stem>.workspace.h5``: what a user makes (ADR 0132 D1-D7).

The file holds definitions only: named node and element sets, plot definitions
and saved views, never a result value. It sits beside ``<stem>.h5`` with the
same stem. The layout is ``architecture/h5-schema.md``, section
"``<stem>.workspace.h5``".

Three levels of refusal, as in ``.user.h5``. A file that is not HDF5, has no
version key, is outside the version floor or lacks a table refuses the file
(:class:`WorkspaceRefused`). Malformed columns refuse that table when it is
read (:class:`~apeGmsh.results.user.ZoneRefused`). A bad row refuses alone,
with a reason (:class:`RowRefusal`).

A set's verdict (:meth:`WorkspaceFile.verdicts`) is judged against the paired
``<stem>.h5``. Its identity hashes are ``fem_hash`` for a node set and
``fem_hash`` plus ``model_hash`` for an element set; only when one differs is
each member looked for at its stored coordinates. A member matches when its own
tag sits within ``1e-9`` of the model's bounding-box diagonal **and no other**
item of the set's kind does: a position that fits two items is ambiguous and
counts as unmatched, so two zero-length springs on one node pair read ``lost``
after a declaration swap, never ``moved``.

A save (:func:`write_workspace`) rebuilds the whole file through a partial of
its own, verifies it, compares ``/meta@revision`` immediately before the
replace, and renames an unreadable old file aside instead of overwriting it.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import re
import secrets
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import h5py
import numpy as np

from ... import __version__
from ...opensees._internal.lineage import read_stored_lineage
from ._descriptor import StateAddress, address_problem
from ._errors import UserFileRefused, UserWriteRefused, ZoneRefused
from ._names import name_problem, text_or_empty
from ._writer import _create_partial, _fsync, _replace, _sweep_dead_partials

__all__ = [
    "PLOT_GRAMMAR",
    "VIEW_GRAMMAR",
    "WORKSPACE_SCHEMA_FLOOR",
    "WORKSPACE_SCHEMA_VERSION",
    "PlotDef",
    "RowRefusal",
    "SetDef",
    "SetVerdict",
    "SetVerdicts",
    "ViewDef",
    "WorkspaceFile",
    "WorkspaceRefused",
    "WorkspaceRenamedAside",
    "WorkspaceRow",
    "WorkspaceSet",
    "WorkspaceWriteRefused",
    "paired_model",
    "write_workspace",
]

#: The writer's version of the file, and the oldest one this reader opens (ADR 0113).
WORKSPACE_SCHEMA_VERSION = "1.0.0"
WORKSPACE_SCHEMA_FLOOR = "1.0.0"
VERSION_KEY = "workspace_schema_version"
SUFFIX = ".workspace.h5"
PLOT_GRAMMAR = "apegmsh-plot/1"
VIEW_GRAMMAR = "apegmsh-view/1"
ID_SPACE = "opensees"
KINDS = ("nodes", "elements")
TABLES = ("sets", "plots", "views")
#: A member matches within this fraction of the model's bounding-box diagonal.
TOLERANCE = 1e-9

_STR = h5py.string_dtype(encoding="utf-8")
_VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\Z")
_SET_TEXT = ("name", "kind", "id_space", "fem_hash", "model_hash")
_ROW_TEXT = ("name", "grammar", "body")
#: The refusals that make an old file *unreadable*: renamed aside, never overwritten.
_UNREADABLE = frozenset({"not-hdf5", "no-version", "bad-version", "bad-meta"})
_I4 = np.iinfo(np.int32)
#: The direction the coordinate search sorts along; irrational, so mesh planes
#: and lines rarely collapse onto one projected value.
_DIRECTION = np.array([1.0, 0.6180339887498949, 0.41421356237309515])
_DIRECTION = _DIRECTION / np.linalg.norm(_DIRECTION)


class WorkspaceRefused(UserFileRefused):
    """The file cannot be read as a workspace. ``reason`` is the schema's word."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


class WorkspaceWriteRefused(UserWriteRefused):
    """A workspace write refused before replacing anything; the old file is whole."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


class WorkspaceRenamedAside(UserWarning):
    """An unreadable old workspace was renamed aside before the new one took its name."""


@dataclass(frozen=True)
class RowRefusal:
    """One row that does not stand, and why (``architecture/h5-schema.md``)."""

    table: str
    row: int
    name: str
    reason: str
    detail: str


@dataclass(frozen=True, eq=False)
class WorkspaceSet:
    """One standing row of ``/sets``: members as tags, in the user's order."""

    name: str
    kind: str
    id_space: str
    fem_hash: str
    model_hash: str
    ids: np.ndarray
    xyz: np.ndarray
    row: int


@dataclass(frozen=True)
class WorkspaceRow:
    """One standing row of ``/plots`` or ``/views``: ``body`` as stored, ``data`` parsed."""

    table: str
    row: int
    name: str
    grammar: str
    body: str
    data: Mapping[str, Any]


@dataclass(frozen=True)
class SetVerdict:
    """A set judged against a model (``current``, ``moved``, ``partial``, ``lost``).

    ``members`` says, per member in order, whether it matched; ``checked`` whether
    the per-member check ran (it runs only when an identity hash differs).
    ``max_distance`` is the largest distance of a matched member, the figure the
    app shows a human confirming a ``moved`` set; ``nan`` when nothing was measured.
    """

    name: str
    kind: str
    verdict: str
    matched: int
    total: int
    members: tuple[bool, ...]
    checked: bool
    max_distance: float


@dataclass(frozen=True)
class SetVerdicts:
    """Every standing set's verdict, plus the sets the model refuses (``no-tag-map``)."""

    verdicts: tuple[SetVerdict, ...]
    refused: tuple[RowRefusal, ...]

    def by_name(self, name: str) -> SetVerdict:
        for v in self.verdicts:
            if v.name == name:
                return v
        for r in self.refused:
            if r.name == name:
                raise KeyError(f"set {name!r} is refused: {r.reason} ({r.detail})")
        raise KeyError(f"no set named {name!r}")


@dataclass(frozen=True)
class SetDef:
    """A set to write: ``kind`` is ``nodes`` or ``elements``; ``ids`` are tags, in order.

    The writer takes each member's coordinates (a node's position, an element's
    centroid) and the identity hashes from the paired ``<stem>.h5``.
    """

    name: str
    kind: str
    ids: Sequence[int]


@dataclass(frozen=True)
class PlotDef:
    """A plot to write: ``body`` is one ``apegmsh-plot/1`` object, or its JSON text."""

    name: str
    body: Any


@dataclass(frozen=True)
class ViewDef:
    """A saved view to write: ``body`` is one ``apegmsh-view/1`` shot, or its JSON text."""

    name: str
    body: Any


# ---------------------------------------------------------------------------
# Names and pairing
# ---------------------------------------------------------------------------


def _stem(path: Path) -> Optional[str]:
    """The set's stem, or ``None`` when ``path`` is not named ``<stem>.workspace.h5``."""
    name = path.name
    if not name.lower().endswith(SUFFIX) or len(name) == len(SUFFIX):
        return None
    return name[: -len(SUFFIX)]


def paired_model(path: "str | Path") -> Path:
    """``<stem>.h5`` beside ``<stem>.workspace.h5``, whether it exists or not."""
    target = Path(path)
    stem = _stem(target)
    if stem is None:
        raise WorkspaceRefused("bad-name", f"{target}: a workspace is named <stem>{SUFFIX}")
    return target.with_name(stem + ".h5")


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _scalar_text(attrs: h5py.AttributeManager, key: str) -> Optional[str]:
    if key not in attrs:
        return None
    raw = attrs[key]
    if isinstance(raw, np.ndarray):
        if raw.size != 1:
            return ""
        raw = raw.ravel()[0]
    return text_or_empty(raw)


def _cells(ds: Any, where: str) -> list[bytes]:
    """A string column's cells as raw bytes, so a row is copied verbatim."""
    if not isinstance(ds, h5py.Dataset) or h5py.check_string_dtype(ds.dtype) is None:
        raise ValueError(f"{where} is not a string column")
    if ds.ndim != 1:
        raise ValueError(f"{where} has shape {ds.shape}, not (K,)")
    out: list[bytes] = []
    for raw in ds[...]:
        if isinstance(raw, str):
            raw = raw.encode("utf-8", "surrogateescape")
        out.append(bytes(raw))
    return out


def _read_sets(group: h5py.Group) -> list[dict[str, Any]]:
    cols = {c: _cells(group[c] if c in group else None, f"/sets/{c}") for c in _SET_TEXT}
    k = len(cols["name"])
    if any(len(v) != k for v in cols.values()):
        raise ValueError("the string columns do not share one length")
    for c in ("member_offsets", "ids", "xyz"):
        if c not in group or not isinstance(group[c], h5py.Dataset):
            raise ValueError(f"/sets/{c} is missing")
    offsets = np.asarray(group["member_offsets"][...])
    ids = np.asarray(group["ids"][...])
    xyz = np.asarray(group["xyz"][...])
    if offsets.dtype.kind != "i" or offsets.shape != (k + 1,):
        raise ValueError(f"member_offsets is {offsets.dtype} {offsets.shape}, not i4 ({k + 1},)")
    if ids.dtype.kind != "i" or ids.ndim != 1:
        raise ValueError(f"ids is {ids.dtype} {ids.shape}, not i4 (M,)")
    m = ids.shape[0]
    if xyz.dtype.kind != "f" or xyz.shape != (m, 3):
        raise ValueError(f"xyz is {xyz.dtype} {xyz.shape}, not f8 ({m}, 3)")
    if offsets[0] != 0 or offsets[-1] != m or np.any(np.diff(offsets) < 0):
        raise ValueError(f"member_offsets {offsets.tolist()} do not run from 0 up to {m}")
    if m and (ids.min() < _I4.min or ids.max() > _I4.max):
        raise ValueError("ids do not fit i4")
    rows = []
    for row in range(k):
        a, b = int(offsets[row]), int(offsets[row + 1])
        cell: dict[str, Any] = {c: cols[c][row] for c in _SET_TEXT}
        cell["ids"] = ids[a:b].astype("i4")
        cell["xyz"] = xyz[a:b].astype("f8")
        rows.append(cell)
    return rows


def _read_rows(group: h5py.Group, table: str) -> list[dict[str, Any]]:
    cols = {c: _cells(group[c] if c in group else None, f"/{table}/{c}") for c in _ROW_TEXT}
    k = len(cols["name"])
    if any(len(v) != k for v in cols.values()):
        raise ValueError("the columns do not share one length")
    return [{c: cols[c][row] for c in _ROW_TEXT} for row in range(k)]


class WorkspaceFile:
    """An opened ``<stem>.workspace.h5``. Reads it whole here; it holds no handle.

    ``sets``, ``plots`` and ``views`` are the rows that stand; ``refused`` the
    rows that refuse alone, each with its reason. A malformed table raises
    :class:`~apeGmsh.results.user.ZoneRefused` when it is read.
    """

    def __init__(self, path: "str | Path", *, data: Optional[bytes] = None) -> None:
        """``data`` is the file's bytes, already read: the writer parses and signs one
        read, so a file replaced in between cannot be judged by the other copy."""
        self.path = Path(path)
        if data is None and not self.path.is_file():
            raise WorkspaceRefused("no-file", f"no such file: {self.path}")
        try:
            f = h5py.File(self.path if data is None else io.BytesIO(data), "r")
        except OSError as exc:
            raise WorkspaceRefused("not-hdf5", f"{self.path} is not HDF5: {exc}") from exc
        with f:
            self._read_meta(f)
            self._raw: dict[str, list[dict[str, Any]]] = {}
            self._broken: dict[str, str] = {}
            for table in TABLES:
                if table not in f or not isinstance(f[table], h5py.Group):
                    raise WorkspaceRefused("missing-table", f"{self.path} has no /{table}")
                try:
                    self._raw[table] = (_read_sets(f[table]) if table == "sets"
                                        else _read_rows(f[table], table))
                except (KeyError, OSError, TypeError, ValueError, IndexError) as exc:
                    self._broken[table] = str(exc)
        self._judge()

    def _read_meta(self, f: h5py.File) -> None:
        if "meta" not in f or not isinstance(f["meta"], h5py.Group):
            raise WorkspaceRefused("no-version", f"{self.path} has no /meta")
        attrs = f["meta"].attrs
        version = _scalar_text(attrs, VERSION_KEY)
        if version is None:
            raise WorkspaceRefused("no-version", f"{self.path} has no /meta@{VERSION_KEY}")
        reason = version_problem(version)
        if reason is not None:
            raise WorkspaceRefused(reason, f"{self.path}: {VERSION_KEY} is {version!r}; this "
                                   f"reader opens {WORKSPACE_SCHEMA_FLOOR} to "
                                   f"{WORKSPACE_SCHEMA_VERSION}")
        revision = np.asarray(attrs["revision"]) if "revision" in attrs else None
        if (revision is None or revision.size != 1 or revision.dtype.kind != "i"
                or int(revision.ravel()[0]) < 1):
            raise WorkspaceRefused("bad-meta", f"{self.path}: /meta@revision is not an integer >= 1")
        self.version: str = version
        self.revision: int = int(revision.ravel()[0])
        self.writer: Optional[str] = _scalar_text(attrs, "writer")
        self.save_id: Optional[str] = _scalar_text(attrs, "save_id")

    # ---- the rows ------------------------------------------------------

    def _judge(self) -> None:
        self._sets: tuple[WorkspaceSet, ...] = ()
        self._rows: dict[str, tuple[WorkspaceRow, ...]] = {"plots": (), "views": ()}
        refused: list[RowRefusal] = []
        if "sets" in self._raw:
            self._sets = tuple(_judge_sets(self._raw["sets"], refused))
        for table in ("plots", "views"):
            if table in self._raw:
                self._rows[table] = tuple(_judge_rows(table, self._raw[table], refused))
        self._refused = tuple(refused)

    def _table(self, table: str) -> None:
        if table in self._broken:
            raise ZoneRefused(f"/{table}", self._broken[table])

    @property
    def sets(self) -> tuple[WorkspaceSet, ...]:
        self._table("sets")
        return self._sets

    @property
    def plots(self) -> tuple[WorkspaceRow, ...]:
        self._table("plots")
        return self._rows["plots"]

    @property
    def views(self) -> tuple[WorkspaceRow, ...]:
        self._table("views")
        return self._rows["views"]

    @property
    def refused(self) -> tuple[RowRefusal, ...]:
        """Rows that refuse alone, from every table that reads."""
        return self._refused

    @property
    def broken_tables(self) -> Mapping[str, str]:
        """Tables whose columns are malformed, with the reason."""
        return dict(self._broken)

    def set(self, name: str) -> WorkspaceSet:
        for s in self.sets:
            if s.name == name:
                return s
        raise KeyError(f"no standing set named {name!r}")

    def verdicts(self, model_h5: "Optional[str | Path]" = None) -> SetVerdicts:
        """Judge every standing set against ``model_h5`` (default: the paired ``<stem>.h5``)."""
        model_path = Path(model_h5) if model_h5 is not None else paired_model(self.path)
        if not model_path.is_file():
            raise WorkspaceRefused("unpaired", f"{self.path}: no model archive {model_path}")
        return _verdicts(self.sets, _Model(model_path))


def version_problem(version: str) -> Optional[str]:
    """The file-level refusal for ``version`` under ADR 0113's floor rule, or ``None``."""
    m = _VERSION.match(version)
    if m is None:
        return "bad-version"
    major, minor = int(m.group(1)), int(m.group(2))
    current = [int(p) for p in WORKSPACE_SCHEMA_VERSION.split(".")]
    floor = [int(p) for p in WORKSPACE_SCHEMA_FLOOR.split(".")]
    if major != current[0]:
        return "major"
    if minor > current[1]:
        return "newer"
    if minor < floor[1]:
        return "too-old"
    return None


def _names_in(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], Counter]:
    names = [text_or_empty(r["name"]) for r in rows]
    return names, Counter(n for n in names if name_problem(n) is None)


def _judge_sets(rows: Sequence[Mapping[str, Any]], refused: list[RowRefusal]) -> list[WorkspaceSet]:
    names, counts = _names_in(rows)
    out: list[WorkspaceSet] = []
    for row, raw in enumerate(rows):
        name = names[row]
        text = {c: text_or_empty(raw[c]) for c in _SET_TEXT}
        ids = raw["ids"]
        problem = name_problem(name)
        if problem is not None:
            reason, detail = "empty-name", f"name {name!r}: {problem}"
        elif counts[name] > 1:
            reason, detail = "duplicate", f"{counts[name]} sets are named {name!r}"
        elif text["kind"] not in KINDS:
            reason, detail = "bad-param", f"kind {text['kind']!r} is not nodes or elements"
        elif text["id_space"] != ID_SPACE:
            reason, detail = "bad-param", f"id_space {text['id_space']!r} is not {ID_SPACE!r}"
        elif ids.size == 0:
            reason, detail = "bad-members", "it has no member"
        elif np.unique(ids).size != ids.size:
            reason, detail = "bad-members", "a tag appears twice"
        elif text["kind"] == "elements" and not text["model_hash"]:
            reason, detail = "bad-param", "an element set needs a model_hash"
        else:
            ids_ro, xyz_ro = ids.copy(), raw["xyz"].copy()
            ids_ro.setflags(write=False)
            xyz_ro.setflags(write=False)
            out.append(WorkspaceSet(name=name, kind=text["kind"], id_space=text["id_space"],
                                    fem_hash=text["fem_hash"], model_hash=text["model_hash"],
                                    ids=ids_ro, xyz=xyz_ro, row=row))
            continue
        refused.append(RowRefusal("sets", row, name, reason, detail))
    return out


def _judge_rows(
    table: str, rows: Sequence[Mapping[str, Any]], refused: list[RowRefusal],
) -> list[WorkspaceRow]:
    grammar_wanted = PLOT_GRAMMAR if table == "plots" else VIEW_GRAMMAR
    names, counts = _names_in(rows)
    out: list[WorkspaceRow] = []
    for row, raw in enumerate(rows):
        name = names[row]
        grammar = text_or_empty(raw["grammar"])
        body = text_or_empty(raw["body"])
        problem = name_problem(name)
        if problem is not None:
            reason, detail = "empty-name", f"name {name!r}: {problem}"
        elif counts[name] > 1:
            reason, detail = "duplicate", f"{counts[name]} {table} are named {name!r}"
        elif grammar != grammar_wanted:
            reason, detail = "unknown-grammar", f"grammar {grammar!r} is not {grammar_wanted!r}"
        else:
            data, reason, detail = _body(table, body)
            if data is not None:
                out.append(WorkspaceRow(table, row, name, grammar, body, data))
                continue
        refused.append(RowRefusal(table, row, name, reason, detail))
    return out


def _body(table: str, body: str) -> "tuple[Optional[dict], str, str]":
    """``(data, "", "")`` for a body that stands, else ``(None, reason, detail)``."""
    try:
        data = strict_json(body)
    except ValueError as exc:
        return None, "bad-json", str(exc)
    if not isinstance(data, dict):
        return None, "bad-json", f"the body is a {type(data).__name__}, not one JSON object"
    if table == "plots":
        problem = plot_problem(data)
        if problem is not None:
            return None, "bad-plot", problem
    return data, "", ""


def strict_json(
    text: str, *, finite: bool = True,
    parse_float: Optional[Callable[[str], float]] = None,
    parse_int: Optional[Callable[[str], int]] = None,
) -> Any:
    """``json.loads`` that refuses ``NaN``/``Infinity`` tokens and a repeated key.

    ``finite`` also refuses a number that overflows to infinity (``1e400``); the
    variants manifest passes ``False`` and refuses such a number in its row.
    ``parse_float`` and ``parse_int`` build the numbers (the manifest keeps their
    tokens, so a row it carries is written back as it was read).
    """
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in items:
            if key in out:
                raise ValueError(f"the key {key!r} appears twice")
            out[key] = value
        return out

    def constant(token: str) -> Any:
        raise ValueError(f"{token} is not a JSON number")

    def number(token: str) -> float:
        value = parse_float(token) if parse_float is not None else float(token)
        if finite and not math.isfinite(value):
            raise ValueError(f"{token} overflows to infinity")
        return value

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant,
                          parse_float=number, parse_int=parse_int)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not JSON: {exc}") from exc


# ---------------------------------------------------------------------------
# The plot grammar, apegmsh-plot/1 (closed keys)
# ---------------------------------------------------------------------------

_PLOT_KEYS = {
    "history": ({"kind", "over", "series"}, {"kind", "over", "series"}),
    "xy": ({"kind", "over", "x", "y"}, {"kind", "over", "x", "y"}),
    "profile": ({"kind", "state", "axis", "series"}, {"kind", "state", "axis", "series"}),
}
_SERIES_KEYS = ({"label", "quantity", "target", "reduce", "minus", "scale"},
                {"quantity", "target", "reduce"})
_REDUCE = ("each", "sum", "mean", "min", "max", "absmax")
_GROUPS = ("results", "user_results")
_TARGET_KINDS = ("set", "physical", "label", "mesh_selection")
_SWEEP_AXES = ("time", "step")
_PROFILE_AXES = ("member", "x", "y", "z", "path")


def _closed(obj: Any, allowed: set[str], required: set[str], where: str) -> Optional[str]:
    if not isinstance(obj, dict):
        return f"{where} is not an object"
    extra = sorted(set(obj) - allowed)
    if extra:
        return f"{where}: the key {extra[0]!r} is not allowed"
    missing = sorted(required - set(obj))
    if missing:
        return f"{where}: the key {missing[0]!r} is required"
    return None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _target_problem(target: Any, where: str) -> Optional[str]:
    problem = _closed(target, {"kind", "name"}, {"kind", "name"}, where)
    if problem is not None:
        return problem
    if target["kind"] not in _TARGET_KINDS:
        return f"{where}.kind {target['kind']!r} is not one of {_TARGET_KINDS}"
    problem = name_problem(target["name"])
    return f"{where}.name: {problem}" if problem is not None else None


def _series_problem(series: Any, where: str, reduce_rule: str) -> Optional[str]:
    problem = _closed(series, *_SERIES_KEYS, where)
    if problem is not None:
        return problem
    if "label" in series and name_problem(series["label"]) is not None:
        return f"{where}.label: {name_problem(series['label'])}"
    quantity = series["quantity"]
    problem = _closed(quantity, {"group", "field", "component"},
                      {"group", "field", "component"}, f"{where}.quantity")
    if problem is not None:
        return problem
    if quantity["group"] not in _GROUPS:
        return f"{where}.quantity.group {quantity['group']!r} is not one of {_GROUPS}"
    for key in ("field", "component"):
        if name_problem(quantity[key]) is not None:
            return f"{where}.quantity.{key}: {name_problem(quantity[key])}"
    problem = _target_problem(series["target"], f"{where}.target")
    if problem is not None:
        return problem
    reduce = series["reduce"]
    if reduce not in _REDUCE:
        return f"{where}.reduce {reduce!r} is not one of {_REDUCE}"
    if reduce_rule == "each" and reduce != "each":
        return f"{where}.reduce is {reduce!r}; every profile series reduces 'each'"
    if reduce_rule == "one" and reduce == "each":
        return f"{where}.reduce is 'each'; an xy series reduces to one value"
    if "minus" in series:
        minus = series["minus"]
        problem = _closed(minus, {"target", "reduce"}, {"target", "reduce"}, f"{where}.minus")
        if problem is not None:
            return problem
        problem = _target_problem(minus["target"], f"{where}.minus.target")
        if problem is not None:
            return problem
        if minus["reduce"] not in _REDUCE[1:]:
            return f"{where}.minus.reduce {minus['reduce']!r} must reduce to one value"
    if "scale" in series and not finite_number(series["scale"]):
        return f"{where}.scale {series['scale']!r} is not a finite number"
    return None


def finite_number(value: Any) -> bool:
    """A JSON number (not a boolean) that is a finite double."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:  # an integer literal past the largest double
        return False


def _over_problem(over: Any) -> Optional[str]:
    problem = _closed(over, {"kind", "source", "steps", "axis"}, {"kind", "source", "axis"}, "over")
    if problem is not None:
        return problem
    if over["kind"] != "stage":
        return f"over.kind {over['kind']!r}: 1.0.0 sweeps a stage only"
    if name_problem(over["source"]) is not None:
        return f"over.source: {name_problem(over['source'])}"
    if over["axis"] not in _SWEEP_AXES:
        return f"over.axis {over['axis']!r} is not one of {_SWEEP_AXES}"
    if "steps" in over:
        steps = over["steps"]
        if (not isinstance(steps, list) or len(steps) != 2
                or not all(_is_int(s) and s >= -1 for s in steps)):
            return f"over.steps {steps!r} is not [from, to] of integers >= -1"
    return None


def _series_list_problem(value: Any, where: str, reduce_rule: str) -> Optional[str]:
    if not isinstance(value, list) or not value:
        return f"{where} is not a non-empty list"
    for i, series in enumerate(value):
        problem = _series_problem(series, f"{where}[{i}]", reduce_rule)
        if problem is not None:
            return problem
    return None


def plot_problem(plot: Mapping[str, Any]) -> Optional[str]:
    """Why ``plot`` breaks the ``apegmsh-plot/1`` grammar (naming the key), or ``None``."""
    kind = plot.get("kind")
    if not isinstance(kind, str) or kind not in _PLOT_KEYS:  # a list or object is unhashable
        return f"kind {kind!r} is not one of {tuple(_PLOT_KEYS)}"
    problem = _closed(plot, *_PLOT_KEYS[kind], f"a {kind} plot")
    if problem is not None:
        return problem
    if kind == "profile":
        state = plot["state"]
        problem = _closed(state, {"kind", "source", "step", "mode"},
                          {"kind", "source", "step", "mode"}, "state")
        if problem is not None:
            return problem
        problem = address_problem(
            StateAddress(state["kind"], state["source"], state["step"], state["mode"]))
        if problem is not None:
            return f"state: {problem}"
        if plot["axis"] not in _PROFILE_AXES:
            return f"axis {plot['axis']!r} is not one of {_PROFILE_AXES}"
        return _series_list_problem(plot["series"], "series", "each")
    problem = _over_problem(plot["over"])
    if problem is not None:
        return problem
    if kind == "history":
        return _series_list_problem(plot["series"], "series", "any")
    problem = _series_problem(plot["x"], "x", "one")
    if problem is not None:
        return problem
    return _series_list_problem(plot["y"], "y", "one")


# ---------------------------------------------------------------------------
# The model side: tags, positions, and the per-member check
# ---------------------------------------------------------------------------


class _NoTagMap(Exception):
    pass


class _Items:
    """One kind of model item: tags with positions (``nan`` rows have none)."""

    def __init__(self, tags: np.ndarray, xyz: np.ndarray, tol: float) -> None:
        self.tol = tol
        order = np.argsort(tags, kind="stable")
        self._sorted = tags[order]
        self._xyz_sorted = xyz[order]
        unique, counts = np.unique(tags, return_counts=True)
        self._repeated = unique[counts > 1]
        finite = np.all(np.isfinite(xyz), axis=1)
        pts = xyz[finite]
        proj = pts @ _DIRECTION
        by_proj = np.argsort(proj, kind="stable")
        self._proj = proj[by_proj]
        self._pts = pts[by_proj]
        self._reach = float(np.max(np.abs(proj))) if proj.size else 0.0

    def positions(self, ids: np.ndarray) -> np.ndarray:
        """Each tag's position; ``nan`` for a tag the model lacks, repeats or cannot place."""
        ids = np.asarray(ids, dtype="i8")
        out = np.full((ids.size, 3), np.nan)
        if not self._sorted.size:
            return out
        at = np.clip(np.searchsorted(self._sorted, ids), 0, self._sorted.size - 1)
        found = (self._sorted[at] == ids) & ~np.isin(ids, self._repeated)
        out[found] = self._xyz_sorted[at[found]]
        return out

    def near(self, xyz: np.ndarray) -> np.ndarray:
        """How many placed items lie within ``tol`` of each stored position."""
        counts = np.zeros(len(xyz), dtype="i8")
        if not self._proj.size:
            return counts
        s = xyz @ _DIRECTION
        slack = self.tol * (1.0 + 1e-9) + 16.0 * np.finfo("f8").eps * (np.abs(s) + self._reach)
        lo = np.searchsorted(self._proj, s - slack, side="left")
        hi = np.searchsorted(self._proj, s + slack, side="right")
        for i in range(len(xyz)):
            if not np.all(np.isfinite(xyz[i])) or hi[i] <= lo[i]:
                continue
            d = np.linalg.norm(self._pts[lo[i]:hi[i]] - xyz[i], axis=1)
            counts[i] = int(np.count_nonzero(d <= self.tol))
        return counts


def _node_lookup(ids: np.ndarray, coords: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    out = np.full(wanted.shape + (3,), np.nan)
    if not sorted_ids.size:
        return out
    at = np.clip(np.searchsorted(sorted_ids, wanted), 0, sorted_ids.size - 1)
    found = sorted_ids[at] == wanted
    out[found] = coords[order[at[found]]]
    return out


class _Model:
    """The paired ``<stem>.h5``: its identity hashes now, its items when a check needs them."""

    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            with h5py.File(path, "r") as f:
                meta = f["meta"] if "meta" in f and isinstance(f["meta"], h5py.Group) else None
                fem_hash, model_hash, _ = (
                    read_stored_lineage(meta) if meta is not None else (None, None, None))
                self.has_tag_map = (
                    "opensees" in f and isinstance(f["opensees"], h5py.Group)
                    and "element_meta" in f["opensees"]
                    and isinstance(f["opensees/element_meta"], h5py.Group))
        except OSError as exc:
            raise WorkspaceRefused("unpaired", f"{path} is not a readable model archive: {exc}") from exc
        self.fem_hash: Optional[str] = fem_hash or None
        self.model_hash: Optional[str] = model_hash or None
        self._nodes: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._items: dict[str, _Items] = {}

    def _node_arrays(self, f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
        if self._nodes is None:
            if "nodes" in f and "ids" in f["nodes"] and "coords" in f["nodes"]:
                ids = np.asarray(f["nodes/ids"][...], dtype="i8").ravel()
                coords = np.asarray(f["nodes/coords"][...], dtype="f8").reshape(-1, 3)
            else:
                ids, coords = np.zeros(0, dtype="i8"), np.zeros((0, 3))
            self._nodes = (ids, coords)
        return self._nodes

    def tolerance(self, coords: np.ndarray) -> float:
        finite = coords[np.all(np.isfinite(coords), axis=1)]
        if not finite.size:
            return 0.0
        return TOLERANCE * float(np.linalg.norm(finite.max(axis=0) - finite.min(axis=0)))

    def items(self, kind: str) -> _Items:
        if kind in self._items:
            return self._items[kind]
        with h5py.File(self.path, "r") as f:
            ids, coords = self._node_arrays(f)
            tol = self.tolerance(coords)
            if kind == "nodes":
                items = _Items(ids, coords, tol)
            else:
                if not self.has_tag_map:
                    raise _NoTagMap(f"{self.path} has no /opensees/element_meta")
                tags, centroids = _element_centroids(f, ids, coords)
                items = _Items(tags, centroids, tol)
        self._items[kind] = items
        return items


def _element_centroids(
    f: h5py.File, node_ids: np.ndarray, coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Every ``/opensees/element_meta`` tag and its centroid (``nan`` when it has no nodes)."""
    fem_ids: list[np.ndarray] = []
    fem_cent: list[np.ndarray] = []
    if "elements" in f and isinstance(f["elements"], h5py.Group):
        for alias in f["elements"]:
            grp = f["elements"][alias]
            if not isinstance(grp, h5py.Group) or "ids" not in grp or "connectivity" not in grp:
                continue
            ids = np.asarray(grp["ids"][...], dtype="i8").ravel()
            conn = np.asarray(grp["connectivity"][...], dtype="i8").reshape(ids.size, -1)
            fem_ids.append(ids)
            fem_cent.append(_node_lookup(node_ids, coords, conn).mean(axis=1))
    all_fem = np.concatenate(fem_ids) if fem_ids else np.zeros(0, dtype="i8")
    all_cent = np.concatenate(fem_cent) if fem_cent else np.zeros((0, 3))
    order = np.argsort(all_fem, kind="stable")
    all_fem, all_cent = all_fem[order], all_cent[order]

    tags: list[np.ndarray] = []
    cents: list[np.ndarray] = []
    meta = f["opensees/element_meta"]
    for token in meta:
        grp = meta[token]
        if not isinstance(grp, h5py.Group) or "ids" not in grp:
            raise _NoTagMap(f"/opensees/element_meta/{token} has no ids")
        ids = np.asarray(grp["ids"][...], dtype="i8").ravel()
        fem = (np.asarray(grp["fem_eids"][...], dtype="i8").ravel() if "fem_eids" in grp
               else np.full(ids.size, -1, dtype="i8"))
        if fem.size != ids.size:
            raise _NoTagMap(f"/opensees/element_meta/{token}: fem_eids and ids differ in length")
        cent = np.full((ids.size, 3), np.nan)
        meshed = fem >= 0
        if meshed.any() and all_fem.size:
            at = np.clip(np.searchsorted(all_fem, fem[meshed]), 0, all_fem.size - 1)
            found = all_fem[at] == fem[meshed]
            rows = np.flatnonzero(meshed)[found]
            cent[rows] = all_cent[at[found]]
        if "inline_connectivity" in grp:
            inline = grp["inline_connectivity"][...]
            for row in np.flatnonzero(~meshed):
                nodes = np.asarray(inline[row], dtype="i8").ravel()
                if nodes.size:
                    cent[row] = _node_lookup(node_ids, coords, nodes).mean(axis=0)
        tags.append(ids)
        cents.append(cent)
    return (np.concatenate(tags) if tags else np.zeros(0, dtype="i8"),
            np.concatenate(cents) if cents else np.zeros((0, 3)))


def _hashes_equal(s: WorkspaceSet, model: _Model) -> bool:
    """Every identity hash of the row equals the model's; a missing model hash differs."""
    if model.fem_hash is None or s.fem_hash != model.fem_hash:
        return False
    if s.kind == "elements":
        return model.model_hash is not None and s.model_hash == model.model_hash
    return True


def match_members(items: _Items, ids: np.ndarray, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(matched, distance)`` per member: its own tag within ``tol``, and no other item."""
    own = items.positions(ids)
    distance = np.linalg.norm(np.asarray(xyz, dtype="f8") - own, axis=1)
    with np.errstate(invalid="ignore"):
        matched = (distance <= items.tol) & (items.near(np.asarray(xyz, dtype="f8")) == 1)
    return matched, distance


def _verdicts(sets: Sequence[WorkspaceSet], model: _Model) -> SetVerdicts:
    out: list[SetVerdict] = []
    refused: list[RowRefusal] = []
    for s in sets:
        if s.kind == "elements" and not model.has_tag_map:
            refused.append(RowRefusal("sets", s.row, s.name, "no-tag-map",
                                      f"{model.path} has no /opensees zone to name element tags"))
            continue
        n = int(s.ids.size)
        if _hashes_equal(s, model):
            out.append(SetVerdict(s.name, s.kind, "current", n, n, (True,) * n, False, math.nan))
            continue
        try:
            items = model.items(s.kind)
        except _NoTagMap as exc:
            refused.append(RowRefusal("sets", s.row, s.name, "no-tag-map", str(exc)))
            continue
        matched, distance = match_members(items, s.ids, s.xyz)
        k = int(np.count_nonzero(matched))
        verdict = "moved" if k == n else ("lost" if k == 0 else "partial")
        worst = float(distance[matched].max()) if k else math.nan
        out.append(SetVerdict(s.name, s.kind, verdict, k, n, tuple(bool(m) for m in matched),
                              True, worst))
    return SetVerdicts(tuple(out), tuple(refused))


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _encode(text: str) -> bytes:
    return text.encode("utf-8")


def _body_text(table: str, name: str, body: Any) -> str:
    if isinstance(body, str):
        text = body
    else:
        try:
            text = json.dumps(body, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise WorkspaceWriteRefused("bad-json", f"{table} {name!r}: {exc}") from exc
    data, reason, detail = _body(table, text)
    if data is None:
        raise WorkspaceWriteRefused(reason, f"{table} {name!r}: {detail}")
    return text


def _set_ids(d: SetDef) -> np.ndarray:
    raw = list(d.ids)
    if not raw:
        raise WorkspaceWriteRefused("bad-members", f"set {d.name!r} has no member")
    for tag in raw:
        if isinstance(tag, (bool, np.bool_)) or not isinstance(tag, (int, np.integer)):
            raise WorkspaceWriteRefused("bad-members", f"set {d.name!r}: {tag!r} is not a tag")
        if not _I4.min <= int(tag) <= _I4.max:
            raise WorkspaceWriteRefused("bad-members", f"set {d.name!r}: tag {tag} does not fit i4")
    ids = np.asarray([int(t) for t in raw], dtype="i4")
    if np.unique(ids).size != ids.size:
        raise WorkspaceWriteRefused("bad-members", f"set {d.name!r}: a tag appears twice")
    return ids


def _set_row(d: SetDef, model: _Model) -> dict[str, Any]:
    if d.kind not in KINDS:
        raise WorkspaceWriteRefused("bad-param", f"set {d.name!r}: kind {d.kind!r}")
    ids = _set_ids(d)
    if model.fem_hash is None:
        raise WorkspaceWriteRefused("bad-row", f"{model.path} has no fem_hash to stamp")
    if d.kind == "elements":
        if not model.has_tag_map:
            raise WorkspaceWriteRefused("no-tag-map", f"{model.path} has no /opensees zone")
        if model.model_hash is None:
            raise WorkspaceWriteRefused("bad-param", f"{model.path} has no model_hash to stamp")
    try:
        items = model.items(d.kind)
    except _NoTagMap as exc:
        raise WorkspaceWriteRefused("no-tag-map", str(exc)) from exc
    xyz = items.positions(ids)
    lacking = ids[~np.all(np.isfinite(xyz), axis=1)]
    if lacking.size:
        raise WorkspaceWriteRefused(
            "bad-row", f"set {d.name!r}: the model has no single placed {d.kind[:-1]} "
                       f"{lacking.tolist()[:10]}")
    return {"name": _encode(d.name), "kind": _encode(d.kind), "id_space": _encode(ID_SPACE),
            "fem_hash": _encode(model.fem_hash),
            "model_hash": _encode((model.model_hash or "") if d.kind == "elements" else ""),
            "ids": ids, "xyz": xyz.astype("f8")}


def _given(table: str, rows: Sequence[Any]) -> list[str]:
    names: list[str] = []
    for r in rows:
        problem = name_problem(r.name)
        if problem is not None:
            raise WorkspaceWriteRefused("empty-name", f"{table} name {r.name!r}: {problem}")
        if r.name in names:
            raise WorkspaceWriteRefused("duplicate", f"{table} {r.name!r} is given twice")
        names.append(r.name)
    return names


def _merge(
    old: list[dict[str, Any]], new: dict[str, dict[str, Any]], removing: set[str],
) -> list[dict[str, Any]]:
    """Old rows in place, verbatim, unless replaced or removed; new names after them."""
    out: list[dict[str, Any]] = []
    placed: set[str] = set()
    for raw in old:
        name = text_or_empty(raw["name"])
        if name in removing:
            continue
        if name in new:
            if name not in placed:
                out.append(new[name])
                placed.add(name)
            continue
        out.append(raw)
    out.extend(row for name, row in new.items() if name not in placed)
    return out


def _digest(target: Path) -> Optional[str]:
    """The file's bytes as one sha256, or ``None`` when no file is there."""
    if not target.exists():
        return None
    if not target.is_file():
        raise WorkspaceWriteRefused("not-a-file", f"{target} is not a file")
    return hashlib.sha256(target.read_bytes()).hexdigest()


def _read_previous(target: Path) -> "tuple[Optional[WorkspaceFile], Optional[str]]":
    """``(file, digest)``, ``(None, digest)`` for an unreadable file, ``(None, None)`` for none.

    The bytes are read once: the same read is parsed and signed, so a file that
    lands between two reads can be neither overwritten nor renamed aside.
    """
    if not target.exists():
        return None, None
    if not target.is_file():
        raise WorkspaceWriteRefused("not-a-file", f"{target} is not a file")
    raw = target.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        previous = WorkspaceFile(target, data=raw)
    except WorkspaceRefused as exc:
        if exc.reason in _UNREADABLE:
            return None, digest
        raise WorkspaceWriteRefused(
            exc.reason, f"{target}: {exc.detail}; nothing was replaced") from exc
    if previous.broken_tables:
        table, why = next(iter(previous.broken_tables.items()))
        raise WorkspaceWriteRefused(
            "bad-table", f"{target}: /{table} cannot be copied verbatim ({why}); "
                         "nothing was replaced")
    return previous, digest


def write_workspace(
    path: "str | Path",
    *,
    sets: Sequence[SetDef] = (),
    plots: Sequence[PlotDef] = (),
    views: Sequence[ViewDef] = (),
    remove: Optional[Mapping[str, Iterable[str]]] = None,
    writer: Optional[str] = None,
) -> Path:
    """Create or rebuild ``<stem>.workspace.h5`` atomically and return its path.

    A given row replaces the row of the same name in its table, in place; a new
    name goes after the rows already there. ``remove`` maps a table (``sets``,
    ``plots``, ``views``) to names to drop. Every other row is copied verbatim,
    refused rows included, except a set that now reads ``moved``, whose row
    takes the model's hashes. Refuses (:class:`WorkspaceWriteRefused`) a row that
    would refuse, a tag the model lacks, an old file newer than this writer, and
    a write that lost a race (``revision-moved``). An unreadable old file is
    renamed aside, with a :class:`WorkspaceRenamedAside` warning.
    """
    target = Path(path)
    if _stem(target) is None:
        raise WorkspaceWriteRefused("bad-name", f"{target}: a workspace is named <stem>{SUFFIX}")
    model_path = paired_model(target)
    if not model_path.is_file():
        raise WorkspaceWriteRefused("unpaired", f"{target}: no {model_path.name} beside it")
    removing = {t: set(names) for t, names in (remove or {}).items()}
    for table in removing:
        if table not in TABLES:
            raise WorkspaceWriteRefused("bad-param", f"remove names an unknown table {table!r}")
    given = {"sets": _given("sets", sets), "plots": _given("plots", plots),
             "views": _given("views", views)}
    for table, names in removing.items():
        clash = names & set(given[table])
        if clash:
            raise WorkspaceWriteRefused("bad-param", f"{table} {sorted(clash)} written and removed")

    previous, digest = _read_previous(target)
    unreadable = digest is not None and previous is None
    model = _Model(model_path)
    new: dict[str, dict[str, dict[str, Any]]] = {
        "sets": {d.name: _set_row(d, model) for d in sets},
        "plots": {d.name: {"name": _encode(d.name), "grammar": _encode(PLOT_GRAMMAR),
                           "body": _encode(_body_text("plots", d.name, d.body))} for d in plots},
        "views": {d.name: {"name": _encode(d.name), "grammar": _encode(VIEW_GRAMMAR),
                           "body": _encode(_body_text("views", d.name, d.body))} for d in views},
    }
    old = previous._raw if previous is not None else {t: [] for t in TABLES}
    for table in TABLES:
        held = Counter(text_or_empty(r["name"]) for r in old[table])
        absent = removing.get(table, set()) - set(held)
        if absent:
            raise WorkspaceWriteRefused("bad-param", f"no {table} named {sorted(absent)} to remove")
        # A name two old rows share names neither: replacing or removing one would
        # drop the other, a refused row, with its members (INV-2).
        twice = sorted(n for n in set(given[table]) | removing.get(table, set()) if held[n] > 1)
        if twice:
            raise WorkspaceWriteRefused(
                "duplicate", f"{table} {twice} name more than one row of {target.name}; "
                             "nothing was replaced")
    old_sets = _restamp_moved(previous, model) if previous is not None else []
    final = {
        "sets": _merge(old_sets, new["sets"], removing.get("sets", set())),
        "plots": _merge(old["plots"], new["plots"], removing.get("plots", set())),
        "views": _merge(old["views"], new["views"], removing.get("views", set())),
    }
    revision = 1 if previous is None else previous.revision + 1
    meta = {"revision": revision, "save_id": secrets.token_urlsafe(16),
            "writer": writer if writer is not None else f"apeGmsh {__version__}"}

    partial, out = _create_partial(target)
    try:
        with out:
            _write_file(out, meta, final)
        _verify(partial, meta, final)
        _fsync(partial)
        _refuse_if_moved(target, digest)
        if unreadable:
            aside = _rename_aside(target)
            warnings.warn(
                f"{target} was not a readable workspace; it was renamed aside to {aside.name}",
                WorkspaceRenamedAside, stacklevel=2)
        _replace(partial, target)
    except BaseException:
        with contextlib.suppress(OSError):
            partial.unlink(missing_ok=True)
        raise
    _sweep_dead_partials(target)
    return target


def _restamp_moved(previous: WorkspaceFile, model: _Model) -> list[dict[str, Any]]:
    """The old ``/sets`` rows, verbatim, but a ``moved`` set takes the model's hashes."""
    rows = [dict(r) for r in previous._raw["sets"]]
    standing = {s.name: s for s in previous.sets}  # standing names are unique
    for v in _verdicts(previous.sets, model).verdicts:
        if v.verdict != "moved":
            continue
        s = standing[v.name]
        rows[s.row]["fem_hash"] = _encode(model.fem_hash or "")
        if s.kind == "elements":
            rows[s.row]["model_hash"] = _encode(model.model_hash or "")
    return rows


def _write_file(out: h5py.File, meta: Mapping[str, Any], tables: Mapping[str, list]) -> None:
    g = out.create_group("meta")
    g.attrs[VERSION_KEY] = WORKSPACE_SCHEMA_VERSION
    g.attrs["revision"] = np.int32(meta["revision"])
    g.attrs["writer"] = meta["writer"]
    g.attrs["save_id"] = meta["save_id"]
    sets = out.create_group("sets")
    rows = tables["sets"]
    for c in _SET_TEXT:
        sets.create_dataset(c, data=np.array([r[c] for r in rows], dtype=object), dtype=_STR)
    sizes = [r["ids"].size for r in rows]
    sets.create_dataset("member_offsets", data=np.concatenate(([0], np.cumsum(sizes))).astype("i4"))
    sets.create_dataset("ids", data=np.concatenate([r["ids"] for r in rows] or
                                                   [np.zeros(0, "i4")]).astype("i4"))
    sets.create_dataset("xyz", data=np.concatenate([r["xyz"] for r in rows] or
                                                   [np.zeros((0, 3))]).astype("f8"))
    for table in ("plots", "views"):
        grp = out.create_group(table)
        for c in _ROW_TEXT:
            grp.create_dataset(c, data=np.array([r[c] for r in tables[table]], dtype=object),
                               dtype=_STR)


def _verify(partial: Path, meta: Mapping[str, Any], tables: Mapping[str, list]) -> None:
    """Read the partial back: every table, and ``ids`` and ``xyz`` bit for bit."""
    try:
        back = WorkspaceFile(partial)
    except (WorkspaceRefused, OSError) as exc:
        raise WorkspaceWriteRefused("partial-mismatch", f"{partial} does not read: {exc}") from exc
    if back.broken_tables or back.revision != meta["revision"] or back.save_id != meta["save_id"]:
        raise WorkspaceWriteRefused("partial-mismatch", f"{partial}: /meta did not read back")
    for table in TABLES:
        wrote, read = tables[table], back._raw[table]
        if len(wrote) != len(read):
            raise WorkspaceWriteRefused("partial-mismatch", f"{partial}: /{table} row count")
        for a, b in zip(wrote, read):
            for key, value in a.items():
                same = (value.tobytes() == b[key].tobytes() and value.shape == b[key].shape
                        if isinstance(value, np.ndarray) else value == b[key])
                if not same:
                    raise WorkspaceWriteRefused(
                        "partial-mismatch", f"{partial}: /{table}/{key} did not read back")


def _refuse_if_moved(target: Path, digest: Optional[str]) -> None:
    """The second of two writers refuses here, rather than overwriting.

    The file on disk must still be the bytes this write read, or still be absent.
    That is stricter than ``/meta@revision`` (every save changes ``@revision`` and
    ``@save_id``), and it also holds for an unreadable file about to be renamed
    aside: another writer's save that replaced it is never moved out of the way.
    """
    try:
        now = _digest(target)
    except (OSError, WorkspaceWriteRefused):
        now = "unreadable now"
    if now != digest:
        raise WorkspaceWriteRefused(
            "revision-moved",
            f"{target} {'was created' if digest is None else 'changed'} while this write "
            "ran; nothing was replaced. Write again.")


def _rename_aside(target: Path) -> Path:
    """Move an unreadable file to ``<name>.unreadable-<UTC>[-n]``, never over a file."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    for n in range(1000):
        aside = target.with_name(f"{target.name}.unreadable-{stamp}" + (f"-{n}" if n else ""))
        try:
            if os.name == "nt":
                os.rename(target, aside)  # Windows refuses an existing destination
            else:
                _rename_no_replace(target, aside)
            return aside
        except FileExistsError:
            continue
    raise WorkspaceWriteRefused("not-a-file", f"{target}: no free name to rename it aside")


def _rename_no_replace(src: Path, dst: Path) -> None:
    """POSIX ``rename`` replaces silently, so link (which refuses an existing name) and
    unlink; a file system without hard links falls back to a checked rename."""
    try:
        os.link(src, dst)
    except FileExistsError:
        raise
    except OSError:
        if dst.exists():
            raise FileExistsError(dst) from None
        os.rename(src, dst)
        return
    os.unlink(src)
