"""The ``/entities`` zone: section cuts, storeys and point sets (ADR 0130 D8).

Three named tables, each a row per entity, with ragged columns held as
``<column>`` plus ``<kind>_offsets``. A user result on ``target="entities"``
names a row by ``ENTITY_KIND`` (:data:`KINDS`) and ``ENTITY_ROW``.

* A **cut** is a :class:`~apeGmsh.cuts.SectionCutDef` with a name and the point
  its moments are about (``SectionCutDef.to_entity``). Its ``element_ids`` are
  OpenSees tags, as the def's are: ``@element_index_space`` is ``"opensees"``.
* A **storey** is a level, a node set and a reference point (the centre of mass).
* A **point set** is coordinates, each with an optional host element (``-1``).

Every array is a copy the entity owns. The writer writes all three tables, empty
ones included, and refuses any bad row. The reader refuses the zone when a table
is missing or a column malformed, and refuses a bad row alone: it stays in its
table, so the rows after it keep their ``ENTITY_ROW``, and :meth:`Entities.row`
refuses it by name (:attr:`Entities.refused`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import h5py
import numpy as np

from ._descriptor import decode_cell
from ._errors import ZoneRefused
from ._names import name_problem, text_or_empty

if TYPE_CHECKING:
    from ...cuts import SectionCutDef

__all__ = [
    "ELEMENT_INDEX_SPACE",
    "KINDS",
    "CutEntity",
    "Entities",
    "EntityRefusal",
    "PointSet",
    "Storey",
]

#: ``ENTITY_KIND`` of an ``ON_ENTITIES`` result, and the table it names.
KINDS = {"cut": "cuts", "storey": "storeys", "point_set": "point_sets"}
#: The space every element id in the zone is in: cut elements, host elements.
ELEMENT_INDEX_SPACE = "opensees"
SIDES = ("positive", "negative")
_ZONE = "/entities"
_STR = h5py.string_dtype(encoding="utf-8")
_I4_MIN, _I4_MAX = -(2 ** 31), 2 ** 31 - 1
#: How far a stored normal may be from unit length.
_UNIT_TOL = 1e-9


def _vec3(v: Any, what: str) -> np.ndarray:
    a = np.array(v, dtype="f8").ravel()
    if a.shape != (3,):
        raise ValueError(f"{what} must be three numbers, got {a.tolist()}")
    return a


def _points(v: Any, what: str) -> np.ndarray:
    a = np.array(v, dtype="f8")
    if a.size == 0:
        a = a.reshape(0, 3)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"{what} must be (P, 3) coordinates, got shape {a.shape}")
    return a


def _ids(v: Any, what: str) -> np.ndarray:
    a = np.array(v)
    if a.size == 0:
        a = a.astype("i4")
    if a.ndim != 1 or a.dtype.kind not in "iu":
        raise ValueError(f"{what} must be a 1-D integer array, got {a.dtype} {a.shape}")
    if a.size and (int(a.min()) < _I4_MIN or int(a.max()) > _I4_MAX):
        raise ValueError(f"{what} does not fit i4 (never int64 in this file)")
    return a.astype("i4")


def _same(a: Any, b: Any, dtype: str) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.astype(dtype).tobytes() == b.astype(dtype).tobytes()


@dataclass(frozen=True, eq=False)
class CutEntity:
    """One row of ``/entities/cuts``.

    ``point`` and ``normal`` are the cut plane (``normal`` of unit length),
    ``side`` the kept side, ``element_ids`` the OpenSees tags it filters to,
    ``polygon`` its bounding polygon (``(0, 3)`` for none), and ``reference``
    the point a stored resultant's moments are about.
    """

    name: str
    point: np.ndarray
    normal: np.ndarray
    reference: np.ndarray
    element_ids: np.ndarray
    side: str = "positive"
    polygon: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "point", _vec3(self.point, f"cut {self.name!r} point"))
        object.__setattr__(self, "normal", _vec3(self.normal, f"cut {self.name!r} normal"))
        object.__setattr__(
            self, "reference", _vec3(self.reference, f"cut {self.name!r} reference"))
        object.__setattr__(
            self, "element_ids", _ids(self.element_ids, f"cut {self.name!r} element_ids"))
        poly = np.zeros((0, 3)) if self.polygon is None else self.polygon
        object.__setattr__(self, "polygon", _points(poly, f"cut {self.name!r} polygon"))

    def to_cut_def(self) -> "SectionCutDef":
        """The :class:`~apeGmsh.cuts.SectionCutDef` this row stores, labelled
        with the row's name; ``to_spec()`` of it feeds the cut kernel."""
        from ...cuts import SectionCutDef

        poly = self.polygon
        problem = _cut_problem(self)
        if problem is not None:
            raise ValueError(problem)
        p, n = self.point, self.normal
        stored = (float(n[0]), float(n[1]), float(n[2]))
        cut = SectionCutDef(
            plane_point=(float(p[0]), float(p[1]), float(p[2])),
            plane_normal=(float(n[0]), float(n[1]), float(n[2])),
            element_ids=tuple(int(e) for e in self.element_ids),
            side="positive" if self.side == "positive" else "negative",
            label=self.name,
            bounding_polygon=(
                None if poly is None or len(poly) == 0
                else tuple((float(v[0]), float(v[1]), float(v[2])) for v in poly)
            ),
        )
        # The def normalises again, and n / |n| of a unit n can move the last bit
        # ((1, 1, 0) / sqrt 2 does), which moves to_spec(). The stored normal is
        # unit to within _UNIT_TOL (_cut_problem), so it is set as stored.
        object.__setattr__(cut, "plane_normal", stored)
        return cut


@dataclass(frozen=True, eq=False)
class Storey:
    """One row of ``/entities/storeys``: a level, its nodes, its reference point."""

    name: str
    level: float
    reference: np.ndarray
    node_ids: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "level", float(self.level))
        object.__setattr__(
            self, "reference", _vec3(self.reference, f"storey {self.name!r} reference"))
        object.__setattr__(
            self, "node_ids", _ids(self.node_ids, f"storey {self.name!r} node_ids"))


@dataclass(frozen=True, eq=False)
class PointSet:
    """One row of ``/entities/point_sets``: points, each with a host element or ``-1``."""

    name: str
    xyz: np.ndarray
    host_element: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        xyz = _points(self.xyz, f"point set {self.name!r} xyz")
        object.__setattr__(self, "xyz", xyz)
        host = (np.full(len(xyz), -1, dtype="i4") if self.host_element is None
                else self.host_element)
        object.__setattr__(
            self, "host_element", _ids(host, f"point set {self.name!r} host_element"))


@dataclass(frozen=True)
class EntityRefusal:
    """A row of ``/entities`` that does not stand: ``kind`` is an ``ENTITY_KIND``,
    ``row`` its position, ``reason`` one of ``empty-name``, ``duplicate``, ``bad-param``."""

    kind: str
    row: int
    name: str
    reason: str
    detail: str


@dataclass(frozen=True, eq=False)
class Entities:
    """The ``/entities`` zone. A row's position in its table is its ``ENTITY_ROW``.

    A read keeps a refused row in its table, so positions hold; ``refused`` names
    each one, and :meth:`row` refuses it. A written ``Entities`` has none.
    """

    cuts: tuple[CutEntity, ...] = ()
    storeys: tuple[Storey, ...] = ()
    point_sets: tuple[PointSet, ...] = ()
    refused: tuple[EntityRefusal, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("cuts", "storeys", "point_sets", "refused"):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))

    def table(self, kind: str) -> tuple[Any, ...]:
        """The rows an ``ENTITY_KIND`` names. Raises ``KeyError`` on another kind."""
        if kind not in KINDS:
            raise KeyError(f"entity kind {kind!r} is not one of {sorted(KINDS)}")
        rows: tuple[Any, ...] = getattr(self, KINDS[kind])
        return rows

    def row(self, kind: str, index: int) -> Any:
        """The entity an ``(ENTITY_KIND, ENTITY_ROW)`` pair names, or ``KeyError``."""
        rows = self.table(kind)
        if not 0 <= int(index) < len(rows):
            raise KeyError(f"/entities/{KINDS[kind]} holds {len(rows)} rows, not row {index}")
        for ref in self.refused:
            if (ref.kind, ref.row) == (kind, int(index)):
                raise KeyError(f"/entities/{KINDS[kind]} row {index} ({ref.name!r}) is "
                               f"refused: {ref.reason} ({ref.detail})")
        return rows[int(index)]

    def same_bits(self, other: "Entities") -> bool:
        """Every row equal bit for bit, the strings byte-equal."""
        if (len(self.cuts), len(self.storeys), len(self.point_sets)) != (
                len(other.cuts), len(other.storeys), len(other.point_sets)):
            return False
        for a, b in zip(self.cuts, other.cuts):
            if (a.name, a.side) != (b.name, b.side) or not all(
                    _same(x, y, t) for x, y, t in (
                        (a.point, b.point, "f8"), (a.normal, b.normal, "f8"),
                        (a.reference, b.reference, "f8"), (a.polygon, b.polygon, "f8"),
                        (a.element_ids, b.element_ids, "i4"))):
                return False
        for s, t in zip(self.storeys, other.storeys):
            if s.name != t.name or not (
                    _same(np.float64(s.level), np.float64(t.level), "f8")
                    and _same(s.reference, t.reference, "f8")
                    and _same(s.node_ids, t.node_ids, "i4")):
                return False
        for p, q in zip(self.point_sets, other.point_sets):
            if p.name != q.name or not (
                    _same(p.xyz, q.xyz, "f8") and _same(p.host_element, q.host_element, "i4")):
                return False
        return True


# ---------------------------------------------------------------------------
# The rules, shared by the writer and the reader
# ---------------------------------------------------------------------------


def _finite(*arrays: Any) -> bool:
    return all(bool(np.all(np.isfinite(np.asarray(a, dtype="f8")))) for a in arrays)


def _cut_problem(c: CutEntity) -> Optional[str]:
    if c.side not in SIDES:
        return f"cut {c.name!r}: side {c.side!r} is not one of {SIDES}"
    if not _finite(c.point, c.normal, c.reference, c.polygon):
        return f"cut {c.name!r}: a coordinate is not finite"
    if abs(float(np.linalg.norm(c.normal)) - 1.0) > _UNIT_TOL:
        return f"cut {c.name!r}: normal {c.normal.tolist()} is not of unit length"
    if c.element_ids.size == 0:
        return f"cut {c.name!r} filters to no element"
    if c.polygon is not None and 0 < len(c.polygon) < 3:
        return f"cut {c.name!r}: a bounding polygon has at least three vertices"
    return None


def _storey_problem(st: Storey) -> Optional[str]:
    if not _finite(st.level, st.reference):
        return f"storey {st.name!r}: the level or reference is not finite"
    if st.node_ids.size == 0:
        return f"storey {st.name!r} has no node"
    return None


def _point_set_problem(p: PointSet) -> Optional[str]:
    if not _finite(p.xyz):
        return f"point set {p.name!r}: a coordinate is not finite"
    if len(p.xyz) == 0:
        return f"point set {p.name!r} has no point"
    if p.host_element is None or p.host_element.shape != (len(p.xyz),):
        return f"point set {p.name!r}: one host_element per point"
    if int(p.host_element.min()) < -1:
        return f"point set {p.name!r}: a host_element is below -1 (-1 = none)"
    return None


_ROW_RULES: tuple[tuple[str, Callable[[Any], Optional[str]]], ...] = (
    ("cut", _cut_problem), ("storey", _storey_problem), ("point_set", _point_set_problem))


def row_refusals(e: Entities) -> tuple[EntityRefusal, ...]:
    """Every row of ``e`` that does not stand, and why: the name rule of A2,
    names unique within a table (each such row refuses), and the kind's own rule."""
    out: list[EntityRefusal] = []
    for kind, rule in _ROW_RULES:
        rows = e.table(kind)
        counts: dict[str, int] = {}
        for r in rows:
            counts[r.name] = counts.get(r.name, 0) + 1
        for i, r in enumerate(rows):
            problem = name_problem(r.name)
            if problem is not None:
                out.append(EntityRefusal(kind, i, str(r.name), "empty-name",
                                         f"name {r.name!r}: {problem}"))
            elif counts[r.name] > 1:
                out.append(EntityRefusal(kind, i, r.name, "duplicate",
                                         f"{counts[r.name]} {KINDS[kind]} rows are named {r.name!r}"))
            else:
                problem = rule(r)
                if problem is not None:
                    out.append(EntityRefusal(kind, i, r.name, "bad-param", problem))
    return tuple(out)


def entities_problem(e: Entities) -> Optional[str]:
    """Why ``e`` cannot be written as ``/entities``, or ``None``: every row must stand."""
    bad = row_refusals(e)
    if not bad:
        return None
    first = bad[0]
    return f"{KINDS[first.kind]} row {first.row}: {first.reason}: {first.detail}"


def _offsets(lengths: Sequence[int]) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(lengths, dtype="i8")]).astype("i4")


def _concat(parts: Sequence[np.ndarray], dtype: str, width: Optional[int] = None) -> np.ndarray:
    if parts:
        return np.concatenate([np.asarray(p, dtype=dtype) for p in parts], axis=0)
    return np.zeros((0,) if width is None else (0, width), dtype=dtype)


def write_entities(out: h5py.File, e: Entities) -> None:
    """Write ``/entities`` into ``out``, which holds none. The caller checked ``e``."""
    zone = out.create_group("entities")
    zone.attrs["element_index_space"] = ELEMENT_INDEX_SPACE

    cuts = zone.create_group("cuts")
    cuts.create_dataset("name", data=[c.name for c in e.cuts], dtype=_STR)
    cuts.create_dataset("side", data=[c.side for c in e.cuts], dtype=_STR)
    for col in ("point", "normal", "reference"):
        cuts.create_dataset(col, data=_concat(
            [np.reshape(getattr(c, col), (1, 3)) for c in e.cuts], "f8", 3))
    cuts.create_dataset("element_offsets", data=_offsets([c.element_ids.size for c in e.cuts]))
    cuts.create_dataset("element_ids", data=_concat([c.element_ids for c in e.cuts], "i4"))
    polys = [np.zeros((0, 3)) if c.polygon is None else c.polygon for c in e.cuts]
    cuts.create_dataset("polygon_offsets", data=_offsets([len(p) for p in polys]))
    cuts.create_dataset("polygon", data=_concat(polys, "f8", 3))

    storeys = zone.create_group("storeys")
    storeys.create_dataset("name", data=[s.name for s in e.storeys], dtype=_STR)
    storeys.create_dataset("level", data=np.array([s.level for s in e.storeys], dtype="f8"))
    storeys.create_dataset("reference", data=_concat(
        [np.reshape(s.reference, (1, 3)) for s in e.storeys], "f8", 3))
    storeys.create_dataset("node_offsets", data=_offsets([s.node_ids.size for s in e.storeys]))
    storeys.create_dataset("node_ids", data=_concat([s.node_ids for s in e.storeys], "i4"))

    points = zone.create_group("point_sets")
    points.create_dataset("name", data=[p.name for p in e.point_sets], dtype=_STR)
    points.create_dataset("point_offsets", data=_offsets([len(p.xyz) for p in e.point_sets]))
    points.create_dataset("xyz", data=_concat([p.xyz for p in e.point_sets], "f8", 3))
    points.create_dataset("host_element", data=_concat(
        [p.host_element for p in e.point_sets if p.host_element is not None], "i4"))


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _col(group: h5py.Group, name: str, kind: str, shape: tuple[Optional[int], ...]) -> np.ndarray:
    where = f"{group.name}/{name}"
    if name not in group or not isinstance(group[name], h5py.Dataset):
        raise ZoneRefused(_ZONE, f"{group.name} has no {name} column")
    ds = group[name]
    good = h5py.check_string_dtype(ds.dtype) is not None if kind == "str" else (
        (ds.dtype.kind, ds.dtype.itemsize) == (kind[0], int(kind[1:])))
    fits = ds.ndim == len(shape) and all(
        want is None or got == want for got, want in zip(ds.shape, shape))
    if not good or not fits:
        raise ZoneRefused(_ZONE, f"{where} is {ds.dtype} {ds.shape}, expected {kind} {shape}")
    if kind == "str":
        return np.array([decode_cell(x, _ZONE, where) for x in ds[...]], dtype=object)
    return ds[...]


def _spans(group: h5py.Group, offsets: str, k: int, total: int) -> list[slice]:
    off = _col(group, offsets, "i4", (k + 1,))
    if int(off[0]) != 0 or int(off[-1]) != total or np.any(np.diff(off) < 0):
        raise ZoneRefused(
            _ZONE, f"{group.name}/{offsets} {off.tolist()} does not run from 0 to {total}")
    return [slice(int(a), int(b)) for a, b in zip(off[:-1], off[1:])]


def _table(zone: h5py.Group, name: str) -> h5py.Group:
    if name not in zone or not isinstance(zone[name], h5py.Group):
        raise ZoneRefused(_ZONE, f"it has no {name} table")
    group: h5py.Group = zone[name]
    return group


def read_entities(zone: h5py.Group) -> Entities:
    """``/entities`` as :class:`Entities`. A missing or malformed table refuses the zone."""
    if not isinstance(zone, h5py.Group):
        raise ZoneRefused(_ZONE, "it is not a group")
    space = zone.attrs["element_index_space"] if "element_index_space" in zone.attrs else None
    if isinstance(space, np.ndarray):
        space = space.ravel()[0] if space.size == 1 else None
    if space is None or text_or_empty(space) != ELEMENT_INDEX_SPACE:
        raise ZoneRefused(
            _ZONE, f"@element_index_space is {space!r}; this reader knows "
            f"{ELEMENT_INDEX_SPACE!r} only")

    g = _table(zone, "cuts")
    names = _col(g, "name", "str", (None,))
    k = len(names)
    side = _col(g, "side", "str", (k,))
    point, normal, reference = (_col(g, c, "f8", (k, 3)) for c in ("point", "normal", "reference"))
    eids = _col(g, "element_ids", "i4", (None,))
    poly = _col(g, "polygon", "f8", (None, 3))
    e_span = _spans(g, "element_offsets", k, len(eids))
    p_span = _spans(g, "polygon_offsets", k, len(poly))
    try:  # a column the dataclasses cannot take refuses the zone
        cuts = tuple(
            CutEntity(name=str(names[i]), side=str(side[i]), point=point[i], normal=normal[i],
                      reference=reference[i], element_ids=eids[e_span[i]], polygon=poly[p_span[i]])
            for i in range(k))

        g = _table(zone, "storeys")
        names = _col(g, "name", "str", (None,))
        k = len(names)
        level = _col(g, "level", "f8", (k,))
        ref = _col(g, "reference", "f8", (k, 3))
        nids = _col(g, "node_ids", "i4", (None,))
        n_span = _spans(g, "node_offsets", k, len(nids))
        storeys = tuple(
            Storey(name=str(names[i]), level=float(level[i]), reference=ref[i],
                   node_ids=nids[n_span[i]])
            for i in range(k))

        g = _table(zone, "point_sets")
        names = _col(g, "name", "str", (None,))
        k = len(names)
        xyz = _col(g, "xyz", "f8", (None, 3))
        host = _col(g, "host_element", "i4", (len(xyz),))
        x_span = _spans(g, "point_offsets", k, len(xyz))
        point_sets = tuple(
            PointSet(name=str(names[i]), xyz=xyz[x_span[i]], host_element=host[x_span[i]])
            for i in range(k))
    except ValueError as exc:
        raise ZoneRefused(_ZONE, str(exc)) from exc
    held = Entities(cuts=cuts, storeys=storeys, point_sets=point_sets)
    return Entities(cuts=cuts, storeys=storeys, point_sets=point_sets,
                    refused=row_refusals(held))


def binding_problem(entities: Entities, name: str, kinds: Sequence[str],
                    rows: Any) -> Optional[str]:
    """Why the ``ON_ENTITIES`` result ``name`` names a row ``entities`` does not
    hold, or a refused one; ``None`` when every row it names stands."""
    for kind, row in zip(kinds, np.asarray(rows).ravel()):
        if kind not in KINDS:
            return f"user result {name!r}: entity kind {kind!r} is not one of {sorted(KINDS)}"
        held = len(entities.table(kind))
        if not 0 <= int(row) < held:
            return (f"user result {name!r} names {kind} row {int(row)}, and "
                    f"/entities/{KINDS[kind]} holds {held}")
        for ref in entities.refused:
            if (ref.kind, ref.row) == (kind, int(row)):
                return (f"user result {name!r} names {kind} row {int(row)}, which is "
                        f"refused: {ref.reason}")
    return None
