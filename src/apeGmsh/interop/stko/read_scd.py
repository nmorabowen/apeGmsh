"""Read an STKO ``.scd`` document (HDF5) into an :class:`ScdModel`.

Layout (``FILE_INFO/VERSION`` 4.1.0, checked on the San Ramon Tier 1–4
documents):

- ``GEOMETRIES/GEOM_<id>``: ``NAME``, ``SHAPE_ID`` (1-based child of
  ``GEOMETRIES/COMPOUND_SHAPE``), and ``{ELEM,PHYS}_PROP_ASN`` /
  ``LOCAX_ASN`` with one ID per sub-shape under ``VER/EDG/FAC/SOL``.
- ``MESH``: ``NODE_IDS``, ``NODE_COORDINATES``, ``NODE_FLAGS``; a flat
  ``ELEMENTS`` array of records ``[id, type, rule, 0, n, 0, node_1..n]``;
  ``MESH_OF_GEOMETRIES/MOG_k`` (``GEOM_ID``) with ``VERTICES`` (node per
  vertex) and ``{EDGES,FACES,SOLIDS}/DOM_<i+1>/ELEMENTS``;
  ``MESH_OF_INTERACTIONS/MOI_k`` (``INTERACTION_ID``, ``ELEMENTS``);
  ``ELEMENT_ORIENTATION_{IDS,QUATERNIONS}``.
- ``SELECTION_SETS/SET_n/ITEMS/ITEM_<geom>``: ``WHOLE_GEOM`` plus 0-based
  ``VERTICES/EDGES/FACES/SOLIDS``.
- ``PHYSICAL_PROPERTIES`` / ``ELEMENT_PROPERTIES`` / ``CONDITIONS`` /
  ``DEFINITIONS`` / ``ANALYSIS_STEPS``: ``<X>_n`` with ``ID``, ``NAME``
  and an ``XOBJ`` (``XOBJ_META`` type, ``ATTRIBUTES/ATTR_n`` parameters).
  Conditions add ``ASN_GEOMETRY`` (flat ``[geom, nV, nE, nF, nS,
  indices…]`` blocks) or ``ASN_INTERACTION`` (interaction IDs).
- ``INTERACTIONS/INTER_n``: ``TYPE``, ``MASTERS`` / ``SLAVES`` rows
  ``[geom, kind code, index]``.
- ``LOCAL_AXES/LOCAX_n``: 12 values, origin then the x, y, z axes.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .model import (
    KINDS,
    Condition,
    Geometry,
    Interaction,
    LocalAxes,
    Mesh,
    MeshElement,
    ScdModel,
    SelectionSet,
    SubShapeRef,
    SubShapes,
    XObject,
)

#: ``*_ASN`` sub-group names, per kind.
_ASN_KEYS = {"vertices": "VER", "edges": "EDG", "faces": "FAC", "solids": "SOL"}

#: Scalar parameter encodings (HDF5 attributes of ``ATTR_n``).
_SCALARS = {"BOOL": bool, "INT": int, "REAL": float, "QNT_SCALAR": float, "INDEX": int}


def read_scd(path: str | Path) -> ScdModel:
    """Read an STKO ``.scd`` document.

    Reads everything but the OCC geometry itself, which
    :func:`~apeGmsh.interop.stko.write_brep` extracts.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        geometries = _read_geometries(f)
        mesh = _read_mesh(f)
        interactions = _read_interactions(f)
        return ScdModel(
            path=path,
            version=tuple(int(v) for v in f["FILE_INFO/VERSION"][()]),
            geometries=geometries,
            mesh=mesh,
            selection_sets=_read_selection_sets(f),
            physical_properties=_read_xobjects(f, "PHYSICAL_PROPERTIES"),
            element_properties=_read_xobjects(f, "ELEMENT_PROPERTIES"),
            conditions=_read_conditions(f),
            interactions=interactions,
            local_axes=_read_local_axes(f),
            definitions=_read_xobjects(f, "DEFINITIONS"),
            analysis_steps=_read_xobjects(f, "ANALYSIS_STEPS"),
        )


# ── helpers ───────────────────────────────────────────────────────────

def _str(value: Any) -> str:
    """An HDF5 string attribute / dataset (stored as length-1 ``|S``
    arrays) as ``str``."""
    if isinstance(value, np.ndarray):
        value = value.ravel()[0] if value.size else b""
    return value.decode() if isinstance(value, bytes) else str(value)


def _int(value: Any) -> int:
    return int(np.asarray(value).ravel()[0])


def _children(group: h5py.Group, prefix: str) -> list[tuple[int, h5py.Group]]:
    """``<prefix>_<n>`` children, sorted by ``n``."""
    out = []
    for key, child in group.items():
        m = re.fullmatch(rf"{prefix}_(\d+)", key)
        if m and isinstance(child, h5py.Group):
            out.append((int(m.group(1)), child))
    return sorted(out, key=lambda kv: kv[0])


def _subshapes(group: h5py.Group) -> SubShapes:
    return SubShapes(**{
        kind: tuple(int(i) for i in group[kind.upper()][()])
        for kind in KINDS if kind.upper() in group
    })


# ── geometry ──────────────────────────────────────────────────────────

def _read_geometries(f: h5py.File) -> dict[int, Geometry]:
    out: dict[int, Geometry] = {}
    for gid, g in _children(f["GEOMETRIES"], "GEOM"):
        def per_kind(asn: str) -> dict[str, np.ndarray]:
            if asn not in g:
                return {}
            return {
                kind: g[asn][key][()].astype(np.int64)
                for kind, key in _ASN_KEYS.items() if key in g[asn]
            }
        eprop = per_kind("ELEM_PROP_ASN")
        out[gid] = Geometry(
            id=gid,
            name=_str(g.attrs["NAME"]),
            shape_index=_int(g.attrs["SHAPE_ID"]),
            counts={kind: len(a) for kind, a in eprop.items()},
            element_property=eprop,
            physical_property=per_kind("PHYS_PROP_ASN"),
            local_axes=per_kind("LOCAX_ASN"),
        )
    return out


# ── mesh ──────────────────────────────────────────────────────────────

def _read_mesh(f: h5py.File) -> Mesh:
    m = f["MESH"]
    node_ids = m["NODE_IDS"][()].astype(np.int64)
    order = np.argsort(node_ids)

    flat = m["ELEMENTS"][()]
    elements: dict[int, MeshElement] = {}
    i = 0
    while i < len(flat):
        n = int(flat[i + 4])
        elements[int(flat[i])] = MeshElement(
            type=int(flat[i + 1]), rule=int(flat[i + 2]),
            nodes=tuple(int(v) for v in flat[i + 6:i + 6 + n]),
        )
        i += 6 + n

    domains: dict[tuple[int, str], dict[int, np.ndarray]] = {}
    vertex_nodes: dict[int, np.ndarray] = {}
    for _, mog in _children(m["MESH_OF_GEOMETRIES"], "MOG"):
        gid = _int(mog.attrs["GEOM_ID"])
        if "VERTICES" in mog:
            vertex_nodes[gid] = mog["VERTICES"][()].astype(np.int64)
        for kind in ("edges", "faces", "solids"):
            if kind.upper() not in mog:
                continue
            domains[(gid, kind)] = {
                n - 1: dom["ELEMENTS"][()].astype(np.int64)
                for n, dom in _children(mog[kind.upper()], "DOM")
                if "ELEMENTS" in dom
            }

    orientation: dict[int, tuple[float, float, float, float]] = {}
    if "ELEMENT_ORIENTATION_IDS" in m:
        ids = m["ELEMENT_ORIENTATION_IDS"][()]
        quats = m["ELEMENT_ORIENTATION_QUATERNIONS"][()]
        orientation = {
            int(e): (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
            for e, q in zip(ids, quats)
        }

    return Mesh(
        node_ids=node_ids[order],
        coordinates=m["NODE_COORDINATES"][()][order],
        node_flags=m["NODE_FLAGS"][()][order],
        elements=elements,
        domains=domains,
        vertex_nodes=vertex_nodes,
        orientation=orientation,
    )


# ── selection sets, conditions, interactions, axes ────────────────────

def _read_selection_sets(f: h5py.File) -> dict[str, SelectionSet]:
    out: dict[str, SelectionSet] = {}
    for sid, s in _children(f["SELECTION_SETS"], "SET"):
        name = _str(s.attrs["NAME"])
        if name in out:
            raise ValueError(f"{f.filename}: two selection sets named {name!r}")
        items: dict[int, SubShapes] = {}
        whole: set[int] = set()
        for gid, item in _children(s["ITEMS"], "ITEM"):
            items[gid] = _subshapes(item)
            if _int(item.attrs.get("WHOLE_GEOM", 0)):
                whole.add(gid)
        out[name] = SelectionSet(id=sid, name=name, items=items, whole=frozenset(whole))
    return out


def _read_conditions(f: h5py.File) -> dict[int, Condition]:
    xobjects = _read_xobjects(f, "CONDITIONS")
    out: dict[int, Condition] = {}
    for cid, c in _children(f["CONDITIONS"], "COND"):
        geometry: dict[int, SubShapes] = {}
        if "ASN_GEOMETRY" in c:
            flat = c["ASN_GEOMETRY"][()]
            i = 0
            while i < len(flat):
                gid = int(flat[i])
                counts = [int(n) for n in flat[i + 1:i + 5]]
                i += 5
                lists = {}
                for kind, n in zip(KINDS, counts):
                    lists[kind] = tuple(int(v) for v in flat[i:i + n])
                    i += n
                geometry[gid] = SubShapes(**lists)
        interactions = tuple(
            int(v) for v in c["ASN_INTERACTION"][()]
        ) if "ASN_INTERACTION" in c else ()
        out[cid] = Condition(xobject=xobjects[cid], geometry=geometry, interactions=interactions)
    return out


def _read_interactions(f: h5py.File) -> dict[int, Interaction]:
    elements: dict[int, tuple[int, ...]] = {}
    moi = f["MESH"].get("MESH_OF_INTERACTIONS")
    if moi is not None:
        for _, m in _children(moi, "MOI"):
            elements[_int(m.attrs["INTERACTION_ID"])] = tuple(int(e) for e in m["ELEMENTS"][()])

    def refs(rows: np.ndarray) -> tuple[SubShapeRef, ...]:
        return tuple(SubShapeRef(int(g), int(k), int(i)) for g, k, i in rows.reshape(-1, 3))

    out: dict[int, Interaction] = {}
    for iid, g in _children(f["INTERACTIONS"], "INTER"):
        out[iid] = Interaction(
            id=iid,
            name=_str(g.attrs["NAME"]),
            type=_str(g.attrs["TYPE"]),
            masters=refs(g["MASTERS"][()]) if "MASTERS" in g else (),
            slaves=refs(g["SLAVES"][()]) if "SLAVES" in g else (),
            elements=elements.get(iid, ()),
            element_property=_int(g.attrs.get("ELEM_PROP_ID", 0)),
            physical_property=_int(g.attrs.get("PHYS_PROP_ID", 0)),
        )
    return out


def _read_local_axes(f: h5py.File) -> dict[int, LocalAxes]:
    """``LOCAX_<n>`` entries. STKO stores each as a 12-value dataset with
    ``NAME`` / ``TYPE`` attributes (1B's ``rotatedWalls``); a group holding
    one such dataset is read too."""
    entries: list[tuple[int, h5py.Group | h5py.Dataset]] = []
    for key, child in f["LOCAL_AXES"].items():
        m = re.fullmatch(r"LOCAX_(\d+)", key)
        if m:
            entries.append((int(m.group(1)), child))
    out: dict[int, LocalAxes] = {}
    for lid, g in sorted(entries, key=lambda kv: kv[0]):
        if isinstance(g, h5py.Dataset):
            values = g[()]
        else:
            values = next(v[()] for v in g.values() if isinstance(v, h5py.Dataset))
        v = [float(x) for x in np.asarray(values).ravel()]
        if len(v) != 12:
            raise ValueError(f"{f.filename}: LOCAL_AXES/LOCAX_{lid} has {len(v)} values, expected 12")
        out[lid] = LocalAxes(
            id=lid, name=_str(g.attrs["NAME"]), type=_str(g.attrs.get("TYPE", b"")),
            origin=(v[0], v[1], v[2]), x=(v[3], v[4], v[5]),
            y=(v[6], v[7], v[8]), z=(v[9], v[10], v[11]),
        )
    return out


# ── XOBJ parameters ───────────────────────────────────────────────────

def _read_xobjects(f: h5py.File, group: str) -> dict[int, XObject]:
    prefix = {
        "PHYSICAL_PROPERTIES": "PHYS_PROP", "ELEMENT_PROPERTIES": "ELEM_PROP",
        "CONDITIONS": "COND", "DEFINITIONS": "DEF", "ANALYSIS_STEPS": "STEP",
    }[group]
    out: dict[int, XObject] = {}
    for oid, g in _children(f[group], prefix):
        x = g["XOBJ"]
        attributes: dict[str, Any] = {}
        references: set[str] = set()
        for _, a in _children(x["ATTRIBUTES"], "ATTR"):
            name = _str(a.attrs["ATTR_META"])
            if name in attributes:
                raise ValueError(
                    f"{f.filename}: {group}/{prefix}_{oid} has two parameters named {name!r}"
                )
            attributes[name], is_ref = _attribute_value(a)
            if is_ref:
                references.add(name)
        out[oid] = XObject(
            id=oid, name=_str(g.attrs["NAME"]), type=_str(x.attrs["XOBJ_META"]),
            attributes=attributes, references=frozenset(references),
        )
    return out


def _attribute_value(a: h5py.Group) -> tuple[Any, bool]:
    """One ``ATTR_n`` value and whether it references other objects."""
    for key, cast in _SCALARS.items():
        if key in a.attrs:
            return cast(np.asarray(a.attrs[key]).ravel()[0]), key == "INDEX"
    if "STRING" in a:
        return _str(a["STRING"][()]), False
    for key in ("QNT_VEC3", "QNT_VECN"):
        if key in a:
            return tuple(float(v) for v in np.asarray(a[key][()]).ravel()), False
    if "INDEX_VEC" in a:
        return tuple(int(v) for v in np.asarray(a["INDEX_VEC"][()]).ravel()), True
    if "CUSTOM_OBJECT" in a:
        return _nested(a["CUSTOM_OBJECT"]), False
    return None, False


def _nested(obj: h5py.Group | h5py.Dataset) -> Any:
    """A custom object (section outline, fiber section) as nested dicts
    of arrays, attributes under ``"@attrs"``."""
    if isinstance(obj, h5py.Dataset):
        return obj[()]
    out: dict[str, Any] = {k: _nested(v) for k, v in obj.items()}
    if obj.attrs:
        out["@attrs"] = {k: v for k, v in obj.attrs.items()}
    return out
