"""One ``/results/r<k>`` entry: a small stage on one target (ADR 0130 D7, A5).

Element buckets use the ``.ladruno`` conventions (``COLUMN_MAP``,
``COMP_NAMES``, ``QUADRATURE/GP_PARAM``), so ``parse_blocks`` of the recorder
reader reads a stored bucket. The stage carries its own ``GP_PARAM``, because
the ``.ladruno`` does not always have one to borrow.

``STEP`` and ``TIME`` are ``0..S-1``: the state of a row is in ``STATES``, never
in ``STEP``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import h5py
import numpy as np

from ..readers._ladruno_element_io import gp_param_to_gmsh
from ._descriptor import StateAddress, address_problem, decode_cell
from ._errors import UserFileError
from ._names import name_problem, text_or_empty

__all__ = ["ElementBucket", "EntryRefused", "ResultEntry", "UserResult"]

TARGETS = ("nodes", "elements", "entities", "domain")
BASES = ("results", "model")
_ON = {"nodes": "ON_NODES", "elements": "ON_ELEMENTS",
       "entities": "ON_ENTITIES", "domain": "ON_DOMAIN"}
_STR = h5py.string_dtype(encoding="utf-8")
_COLUMN_MAP = ("LEVELS", "GAUSS_ID", "SECTION_TAG", "FIBER_ID", "NUM_COMP", "MULTIPLICITY")
_I4_MIN, _I4_MAX = -(2 ** 31), 2 ** 31 - 1


class EntryRefused(UserFileError):
    """One entry does not stand. ``reason`` is one of the A3 reasons."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True, eq=False)
class ElementBucket:
    """One bucket: ``DATA (S, n, c)`` split in ``components`` per block.

    Each row of ``components`` (with its ``gauss_ids`` and ``multiplicity``) is
    one ``COLUMN_MAP`` row; a block takes ``len(names) * multiplicity`` columns.

    ``topology``, ``param_domain``, ``family`` and ``order`` place ``gp_param``:
    the frame of a barycentric rule depends on the family, so the recorder's
    read path refuses a group without them. ``order`` has one entry per
    ``gp_param`` column, as the recorder writes it.
    """

    name: str
    ids: np.ndarray
    data: np.ndarray
    components: tuple[tuple[str, ...], ...]
    gauss_ids: tuple[int, ...]
    multiplicity: tuple[int, ...]
    topology: str
    param_domain: str
    family: str
    order: tuple[int, ...]
    gp_param: np.ndarray


@dataclass(frozen=True, eq=False)
class UserResult:
    """A named field on one target. ``states=None`` means state-free, ``S = 1``.

    ``linear=True`` declares that a factored sum of this field is the field of
    the sum (A5). It defaults to False, and a summing rule then refuses.
    """

    name: str
    target: str
    basis: str
    states: Optional[tuple[StateAddress, ...]] = None
    linear: bool = False
    units: Optional[tuple[str, ...]] = None
    element_index_space: Optional[str] = None
    components: tuple[str, ...] = ()
    data: Optional[np.ndarray] = None
    ids: Optional[np.ndarray] = None
    entity_kind: Optional[tuple[str, ...]] = None
    entity_row: Optional[np.ndarray] = None
    buckets: tuple[ElementBucket, ...] = ()


@dataclass(frozen=True)
class ResultEntry:
    """What an entry is, read without its ``DATA``."""

    name: str
    target: str
    basis: str
    per_state: bool
    linear: bool
    link: str
    states: tuple[StateAddress, ...] = ()


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _i4(values: Any, what: str) -> np.ndarray:
    a = np.asarray(values)
    if a.dtype.kind not in "iu":
        raise EntryRefused("bad-param", f"{what} is {a.dtype}, not an integer array")
    if a.size and (int(a.min()) < _I4_MIN or int(a.max()) > _I4_MAX):
        raise EntryRefused("bad-param", f"{what} does not fit i4 (never int64 in this file)")
    return a.astype("i4")


def _f8(values: Any, what: str, s: int, n: Optional[int], c: Optional[int]) -> np.ndarray:
    a = np.asarray(values, dtype="f8")
    if a.ndim != 3 or a.shape[0] != s or (n is not None and a.shape[1] != n) or (
        c is not None and a.shape[2] != c
    ):
        raise EntryRefused(
            "bad-param", f"{what} has shape {a.shape}, expected ({s}, {n or 'n'}, {c or 'c'})")
    return a


def _token_problem(tokens: Sequence[str], what: str) -> None:
    if not tokens:
        raise EntryRefused("bad-param", f"{what} is empty")
    for token in tokens:
        problem = name_problem(token)
        if problem is None and ("," in token or "\n" in token):
            problem = "has a comma or a newline"
        if problem is not None:
            raise EntryRefused("bad-param", f"{what} {token!r}: {problem}")


def _link_problem(name: str) -> Optional[str]:
    problem = name_problem(name)
    if problem is None and ("/" in name or name.strip(".") == ""):
        problem = "cannot be an HDF5 link"
    return problem


def _stack_axes(group: h5py.Group, s: int) -> None:
    group.create_dataset("STEP", data=np.arange(s, dtype="i4"))
    group.create_dataset("TIME", data=np.arange(s, dtype="f8"))


def _write_id(group: h5py.Group, ids: Any, what: str) -> int:
    a = _i4(ids, what)
    if a.ndim != 1:
        raise EntryRefused("bad-param", f"{what} must be one-dimensional, got {a.shape}")
    group.create_dataset("ID", data=a.reshape(-1, 1))
    return a.shape[0]


def _check_order(order: np.ndarray, gp: np.ndarray, where: str) -> None:
    if order.ndim != 1 or order.size != gp.shape[1] or np.any(order < 1):
        raise EntryRefused(
            "bad-param", f"{where}: ORDER {order.tolist()} needs one entry >= 1 "
            f"per GP_PARAM column, and GP_PARAM has shape {gp.shape}")


def _check_frame(gp: np.ndarray, domain: str, topology: str, family: str, where: str) -> None:
    """Refuse what the recorder's read path could not place, at this seam."""
    try:
        gp_param_to_gmsh(gp, param_domain=domain, topology=topology, family=family, where=where)
    except ValueError as exc:
        raise EntryRefused("bad-param", str(exc)) from exc


def _write_bucket(values: h5py.Group, model: h5py.Group, b: ElementBucket, s: int) -> None:
    problem = _link_problem(b.name)
    if problem is not None:
        raise EntryRefused("bad-param", f"bucket {b.name!r}: {problem}")
    k = len(b.components)
    if k == 0 or len(b.gauss_ids) != k or len(b.multiplicity) != k:
        raise EntryRefused("bad-param", f"bucket {b.name}: components, gauss_ids and "
                           "multiplicity must have one row per block")
    width = 0
    for names, mult in zip(b.components, b.multiplicity):
        _token_problem(names, f"bucket {b.name} component")
        if mult < 1:
            raise EntryRefused("bad-param", f"bucket {b.name}: multiplicity {mult}")
        width += len(names) * mult
    group = values.create_group(b.name)
    n = _write_id(group, b.ids, f"bucket {b.name} ids")
    group.create_dataset("DATA", data=_f8(b.data, f"bucket {b.name} DATA", s, n, width))
    _stack_axes(group, s)
    group.attrs["NUM_COLUMNS"] = np.int32(width)
    cmap = group.create_group("COLUMN_MAP")
    rows = {
        "LEVELS": [0] * k, "GAUSS_ID": list(b.gauss_ids), "SECTION_TAG": [-1] * k,
        "FIBER_ID": [-1] * k, "NUM_COMP": [len(c) for c in b.components],
        "MULTIPLICITY": list(b.multiplicity),
    }
    for key in _COLUMN_MAP:
        cmap.create_dataset(key, data=_i4(rows[key], key).reshape(k, 1))
    cmap.attrs["COMP_NAMES"] = "\n".join(",".join(c) for c in b.components)

    gp = np.asarray(b.gp_param, dtype="f8")
    if gp.ndim != 2 or gp.shape[0] < 1:
        raise EntryRefused("bad-param", f"bucket {b.name}: GP_PARAM has shape {gp.shape}")
    for what, text in (("topology", b.topology), ("param_domain", b.param_domain),
                       ("family", b.family)):
        if name_problem(text) is not None:
            raise EntryRefused("bad-param", f"bucket {b.name}: {what} {text!r}")
    order = _i4(b.order, f"bucket {b.name} order")
    _check_order(order, gp, f"bucket {b.name}")
    _check_frame(gp, b.param_domain, b.topology, b.family, f"bucket {b.name}")
    placed = model.create_group(b.name)
    placed.attrs["TOPOLOGY"] = b.topology
    placed.attrs["PARAM_DOMAIN"] = b.param_domain
    placed.attrs["FAMILY"] = b.family
    placed.attrs["ORDER"] = order
    placed.attrs["NUM_GP"] = np.int32(gp.shape[0])
    placed.create_group("QUADRATURE").create_dataset("GP_PARAM", data=gp)


def write_entry(group: h5py.Group, r: UserResult) -> None:
    """Write one entry into the new group ``group``, refusing a bad one."""
    problem = name_problem(r.name)
    if problem is not None:
        raise EntryRefused("empty-name", f"name {r.name!r}: {problem}")
    if r.target not in TARGETS:
        raise EntryRefused("bad-param", f"target {r.target!r} is not one of {TARGETS}")
    if r.basis not in BASES:
        raise EntryRefused("bad-param", f"basis {r.basis!r} is not one of {BASES}")
    if r.states is not None and not r.states:
        raise EntryRefused("bad-param", "states is empty; pass None for a state-free result")
    s = 1 if r.states is None else len(r.states)
    group.attrs["name"] = r.name
    group.attrs["KIND"] = "static"
    group.attrs["target"] = r.target
    group.attrs["per_state"] = np.int8(r.states is not None)
    group.attrs["basis"] = r.basis
    if r.linear:
        group.attrs["linear"] = np.int8(1)
    if r.units is not None:
        _token_problem(r.units, "unit")
        group.attrs["units"] = ",".join(r.units)
    if r.element_index_space is not None:
        if r.target != "elements":
            raise EntryRefused("bad-param", "element_index_space is for elements only")
        group.attrs["element_index_space"] = r.element_index_space
    if r.states is not None:
        _write_states(group.create_group("STATES"), r.states)

    results = group.create_group("RESULTS")
    values = results.create_group(_ON[r.target]).create_group("values")
    if r.target == "elements":
        if not r.buckets:
            raise EntryRefused("bad-param", "an elements result needs a bucket")
        model = group.create_group("MODEL").create_group("ELEMENTS")
        for bucket in r.buckets:
            _write_bucket(values, model, bucket, s)
        return
    _token_problem(r.components, "component")
    values.attrs["COMPONENTS"] = ",".join(r.components)
    c = len(r.components)
    if r.target == "nodes":
        n = _write_id(values, r.ids, "ids")
    elif r.target == "entities":
        row = _i4(r.entity_row, "entity_row")
        kinds = list(r.entity_kind or ())
        if row.ndim != 1 or len(kinds) != row.shape[0]:
            raise EntryRefused("bad-param", "entity_kind and entity_row must have one row per item")
        for kind in kinds:
            if name_problem(kind) is not None:
                raise EntryRefused("bad-param", f"entity_kind {kind!r}")
        n = row.shape[0]
        values.create_dataset("ENTITY_ROW", data=row)
        if kinds:
            values.create_dataset("ENTITY_KIND", data=kinds, dtype=_STR)
        else:
            values.create_dataset("ENTITY_KIND", shape=(0,), dtype=_STR)
    else:
        n = 1
    values.create_dataset("DATA", data=_f8(r.data, "DATA", s, n, c))
    _stack_axes(values, s)


def _write_states(group: h5py.Group, states: Sequence[StateAddress]) -> None:
    for state in states:
        problem = address_problem(state)
        if problem is not None:
            raise EntryRefused("bad-term", problem)
    group.create_dataset("kind", data=[a.kind for a in states], dtype=_STR)
    group.create_dataset("source", data=[a.source for a in states], dtype=_STR)
    group.create_dataset("step", data=np.array([a.step for a in states], dtype="i4"))
    group.create_dataset("mode", data=np.array([a.mode for a in states], dtype="i4"))


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _attr_text(group: h5py.Group, key: str) -> Optional[str]:
    if key not in group.attrs:
        return None
    raw = group.attrs[key]
    if isinstance(raw, np.ndarray):
        if raw.size != 1:
            return ""
        raw = raw.ravel()[0]
    return text_or_empty(raw)


def _attr_flag(group: h5py.Group, key: str) -> Optional[int]:
    if key not in group.attrs:
        return None
    a = np.asarray(group.attrs[key])
    if a.size != 1 or a.dtype.kind not in "iu" or int(a.ravel()[0]) not in (0, 1):
        raise EntryRefused("bad-param", f"@{key} is {a!r}, expected 0 or 1")
    return int(a.ravel()[0])


def _dataset(group: h5py.Group, name: str, kind: str, ndim: int) -> h5py.Dataset:
    if name not in group or not isinstance(group[name], h5py.Dataset):
        raise EntryRefused("bad-param", f"{group.name} has no {name} dataset")
    ds = group[name]
    good = h5py.check_string_dtype(ds.dtype) is not None if kind == "str" else (
        (ds.dtype.kind, ds.dtype.itemsize) == (kind[0], int(kind[1:]))
    )
    if not good or ds.ndim != ndim:
        raise EntryRefused("bad-param", f"{ds.name} is {ds.dtype} {ds.shape}, expected {kind}, rank {ndim}")
    return ds


def _read_states(group: h5py.Group) -> tuple[StateAddress, ...]:
    kinds = _dataset(group, "kind", "str", 1)[...]
    sources = _dataset(group, "source", "str", 1)[...]
    steps = _dataset(group, "step", "i4", 1)[...]
    modes = _dataset(group, "mode", "i4", 1)[...]
    if not (kinds.shape == sources.shape == steps.shape == modes.shape) or kinds.size == 0:
        raise EntryRefused("bad-param", f"{group.name} columns disagree or are empty")
    out = tuple(
        StateAddress(decode_cell(k, "/results", group.name),
                     decode_cell(src, "/results", group.name), int(st), int(m))
        for k, src, st, m in zip(kinds, sources, steps, modes)
    )
    for state in out:
        problem = address_problem(state)
        if problem is not None:
            raise EntryRefused("bad-term", problem)
    return out


def read_entry_meta(link: str, group: h5py.Group) -> ResultEntry:
    """Validate an entry without reading ``DATA``, or raise :class:`EntryRefused`."""
    name = _attr_text(group, "name")
    problem = name_problem(name) if name is not None else "absent"
    if problem is not None:
        raise EntryRefused("empty-name", f"{group.name}@name: {problem}")
    target = _attr_text(group, "target")
    if target not in TARGETS:
        raise EntryRefused("bad-param", f"@target {target!r} is not one of {TARGETS}")
    basis = _attr_text(group, "basis")
    if basis not in BASES:
        raise EntryRefused("bad-param", f"@basis {basis!r} is not one of {BASES}")
    per_state = _attr_flag(group, "per_state")
    if per_state is None:
        raise EntryRefused("bad-param", f"{group.name} has no @per_state")
    linear = _attr_flag(group, "linear") or 0
    states: tuple[StateAddress, ...] = ()
    if per_state:
        if "STATES" not in group or not isinstance(group["STATES"], h5py.Group):
            raise EntryRefused("bad-param", f"{group.name} is per state and has no STATES")
        states = _read_states(group["STATES"])
    s = len(states) if per_state else 1

    on = _ON[target]
    if "RESULTS" not in group or on not in group["RESULTS"] or "values" not in group["RESULTS"][on]:
        raise EntryRefused("bad-param", f"{group.name} has no RESULTS/{on}/values")
    values = group[f"RESULTS/{on}/values"]
    if not isinstance(values, h5py.Group):
        raise EntryRefused("bad-param", f"{values.name} is not a group")
    holders = [values[b] for b in values] if target == "elements" else [values]
    if not holders:
        raise EntryRefused("bad-param", f"{values.name} has no bucket")
    for holder in holders:
        if not isinstance(holder, h5py.Group):
            raise EntryRefused("bad-param", f"{holder.name} is not a bucket group")
        _check_holder(holder, target, s)
        if target == "elements":
            _check_placement(group, holder.name.rsplit("/", 1)[-1])
    return ResultEntry(name, target, basis, bool(per_state), bool(linear), link, states)


def _check_placement(group: h5py.Group, bucket: str) -> None:
    """A stage carries its own ``GP_PARAM`` (h5-schema.md, ``/results``)."""
    where = f"MODEL/ELEMENTS/{bucket}"
    if where not in group or not isinstance(group[where], h5py.Group):
        raise EntryRefused("bad-param", f"{group.name} has no {where}")
    placed = group[where]
    texts = {}
    for key in ("TOPOLOGY", "PARAM_DOMAIN", "FAMILY"):
        texts[key] = _attr_text(placed, key)
        if not texts[key]:
            raise EntryRefused("bad-param", f"{placed.name} has no @{key}")
    if "QUADRATURE" not in placed or not isinstance(placed["QUADRATURE"], h5py.Group):
        raise EntryRefused("bad-param", f"{placed.name} has no QUADRATURE")
    gp = _dataset(placed["QUADRATURE"], "GP_PARAM", "f8", 2)
    num_gp = np.asarray(placed.attrs["NUM_GP"]) if "NUM_GP" in placed.attrs else None
    if (num_gp is None or num_gp.size != 1 or num_gp.dtype.kind not in "iu"
            or int(num_gp.ravel()[0]) != gp.shape[0]):
        raise EntryRefused("bad-param", f"{placed.name}@NUM_GP does not match GP_PARAM {gp.shape}")
    order = np.asarray(placed.attrs["ORDER"]) if "ORDER" in placed.attrs else None
    if order is None or order.dtype.kind not in "iu":
        raise EntryRefused("bad-param", f"{placed.name}@ORDER is absent or not integers")
    _check_order(order, gp, placed.name)
    _check_frame(gp[...], texts["PARAM_DOMAIN"], texts["TOPOLOGY"], texts["FAMILY"], placed.name)


def _check_holder(holder: h5py.Group, target: str, s: int) -> None:
    data = _dataset(holder, "DATA", "f8", 3)
    if data.shape[0] != s:
        raise EntryRefused("bad-param", f"{data.name} has {data.shape[0]} rows on axis 0, expected {s}")
    n = data.shape[1]
    if target in ("nodes", "elements"):
        ids = _dataset(holder, "ID", "i4", 2)
        if ids.shape != (n, 1):
            raise EntryRefused("bad-param", f"{ids.name} has shape {ids.shape}, expected ({n}, 1)")
    elif target == "entities":
        if (_dataset(holder, "ENTITY_KIND", "str", 1).shape != (n,)
                or _dataset(holder, "ENTITY_ROW", "i4", 1).shape != (n,)):
            raise EntryRefused("bad-param", f"{holder.name}: entity columns must have {n} rows")
    elif n != 1:
        raise EntryRefused("bad-param", f"{data.name}: a domain value has n = 1, got {n}")
    if _dataset(holder, "STEP", "i4", 1).shape != (s,) or _dataset(holder, "TIME", "f8", 1).shape != (s,):
        raise EntryRefused("bad-param", f"{holder.name}: STEP and TIME must have {s} rows")
    if target == "elements":
        _check_column_map(holder, data.shape[2])
    else:
        text = _attr_text(holder, "COMPONENTS")
        if not text or len(text.split(",")) != data.shape[2]:
            raise EntryRefused("bad-param", f"{holder.name}@COMPONENTS {text!r} does not name "
                               f"{data.shape[2]} columns")


def _check_column_map(holder: h5py.Group, width: int) -> None:
    if "COLUMN_MAP" not in holder or not isinstance(holder["COLUMN_MAP"], h5py.Group):
        raise EntryRefused("bad-param", f"{holder.name} has no COLUMN_MAP")
    cmap = holder["COLUMN_MAP"]
    text = _attr_text(cmap, "COMP_NAMES")
    if not text:
        raise EntryRefused("bad-param", f"{cmap.name} has no @COMP_NAMES")
    lines = text.split("\n")
    mult = _dataset(cmap, "MULTIPLICITY", "i4", 2)[...].ravel()
    gauss = _dataset(cmap, "GAUSS_ID", "i4", 2)[...].ravel()
    if mult.size != len(lines) or gauss.size != len(lines):
        raise EntryRefused("bad-param", f"{cmap.name} rows disagree with its @COMP_NAMES lines")
    total = sum(len(line.split(",")) * int(m) for line, m in zip(lines, mult))
    if total != width:
        raise EntryRefused("bad-param", f"{cmap.name} covers {total} columns, DATA has {width}")


def read_entry(entry: ResultEntry, group: h5py.Group) -> UserResult:
    """Load a validated entry, ``DATA`` included."""
    values = group[f"RESULTS/{_ON[entry.target]}/values"]
    units_text = _attr_text(group, "units")
    common: dict[str, Any] = {
        "name": entry.name, "target": entry.target, "basis": entry.basis,
        "states": entry.states if entry.per_state else None, "linear": entry.linear,
        "units": tuple(units_text.split(",")) if units_text else None,
        "element_index_space": _attr_text(group, "element_index_space"),
    }
    if entry.target == "elements":
        return UserResult(**common, buckets=tuple(
            _read_bucket(values[b], group, b) for b in values
        ))
    components = tuple((_attr_text(values, "COMPONENTS") or "").split(","))
    data = values["DATA"][...]
    if entry.target == "nodes":
        return UserResult(**common, components=components, data=data,
                          ids=values["ID"][...].ravel())
    if entry.target == "entities":
        return UserResult(
            **common, components=components, data=data,
            entity_kind=tuple(decode_cell(k, "/results", values.name)
                              for k in values["ENTITY_KIND"][...]),
            entity_row=values["ENTITY_ROW"][...],
        )
    return UserResult(**common, components=components, data=data)


def _read_bucket(holder: h5py.Group, entry_group: h5py.Group, name: str) -> ElementBucket:
    where = f"MODEL/ELEMENTS/{name}"
    if where not in entry_group:
        raise EntryRefused("bad-param", f"{entry_group.name} has no {where}")
    placed = entry_group[where]
    cmap = holder["COLUMN_MAP"]
    return ElementBucket(
        name=name,
        ids=holder["ID"][...].ravel(),
        data=holder["DATA"][...],
        components=tuple(
            tuple(line.split(",")) for line in (_attr_text(cmap, "COMP_NAMES") or "").split("\n")
        ),
        gauss_ids=tuple(int(g) for g in cmap["GAUSS_ID"][...].ravel()),
        multiplicity=tuple(int(m) for m in cmap["MULTIPLICITY"][...].ravel()),
        topology=_attr_text(placed, "TOPOLOGY") or "",
        param_domain=_attr_text(placed, "PARAM_DOMAIN") or "",
        family=_attr_text(placed, "FAMILY") or "",
        order=tuple(int(o) for o in np.asarray(placed.attrs["ORDER"]).ravel()),
        gp_param=_dataset(placed["QUADRATURE"], "GP_PARAM", "f8", 2)[...],
    )
