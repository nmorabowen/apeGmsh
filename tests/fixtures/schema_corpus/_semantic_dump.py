"""The semantic dump: the oracle of the schema corpus (ADR 0113 D8).

One function pair turns what a reader returned into plain JSON:
:func:`dump_fem` for a :class:`FEMData` and :func:`dump_model` for an
:class:`OpenSeesModel`.  The *same* code runs twice:

* in the frozen era, under ``PYTHONPATH=<era worktree>/src``, on the
  objects the era's **own** reader returned
  (``scripts/build_schema_corpus.py`` stores the result beside the file);
* today, in ``tests/opensees/h5/test_schema_corpus.py``, on the objects
  today's reader returns from the same bytes.

So the reader is the only thing that varies.  This module must stay
importable by every era from the zone floors on: it imports nothing from
apeGmsh and touches only the public surface that has been stable since
neutral 2.10.0 / opensees 2.12.0 (``FEMData.snapshot_id``, the node and
element composites, ``NamedGroupSet.get_all/get_name/node_ids/
element_ids``, ``MeshSelectionStore.get_all/get_name/get_nodes/
get_elements``, ``NodeComposite.ndf_for``, the record sets' ``len``, and
``OpenSeesModel``'s record accessors).  Editing it changes the oracle for
every committed era, so rebuild the corpus after any change.

:func:`dump_stamps` is the one exception to "reader objects only": it
reads the ``/meta`` ``ndm`` and ``ndf`` attributes raw with h5py, because
``FEMData.from_h5`` does not carry them.  Those are the writer's
declarations, equal in every era by construction; they sit in the dump so
each file says what its era stamped (a pre-2.34.0 frame stamps the mesh
dimension, which is the case the ``read_spatial_ndm`` shim exists for).
"""
from __future__ import annotations

import dataclasses
import enum
import math
from typing import Any

#: 1: the #1329 corpus. 2: adds ``mesh_selections``, the per-node ``ndf``
#: histogram and the raw ``/meta`` stamps (#1303, opensees floor 2.12).
DUMP_FORMAT = 2


def _norm(x: Any) -> Any:
    """Reduce a reader value to JSON: dataclasses by field, arrays to lists."""
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, enum.Enum):
        return _norm(x.value)
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float):
        return x if math.isfinite(x) else repr(x)
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        out = {"__type__": type(x).__name__}
        for f in dataclasses.fields(x):
            out[f.name] = _norm(getattr(x, f.name))
        return out
    # numpy scalars and arrays, without importing numpy here.
    if hasattr(x, "tolist") and hasattr(x, "dtype"):
        return _norm(x.tolist())
    if isinstance(x, dict) or (hasattr(x, "items") and hasattr(x, "keys")):
        return {str(k): _norm(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_norm(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted((_norm(v) for v in x), key=repr)
    raise TypeError(
        f"semantic dump: no JSON form for {type(x).__module__}.{type(x).__qualname__}"
    )


def _group_sizes(group_set: Any, *, side: str) -> dict[str, int]:
    """``"dim:name" -> size`` for one side of a physical-group or label set."""
    out: dict[str, int] = {}
    for dim, tag in group_set.get_all():
        name = group_set.get_name(dim, tag)
        if side == "node":
            n = len(group_set.node_ids(name, dim=dim))
        else:
            try:
                n = len(group_set.element_ids(name, dim=dim))
            except (KeyError, ValueError):
                # A node-only group has no element side; record that.
                n = -1
        out[f"{dim}:{name}"] = int(n)
    return out


def _element_counts(fem: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for grp in fem.elements:
        out[str(grp.element_type.name)] = int(len(grp.ids))
    return out


def _patterns(records: Any) -> list[str]:
    return sorted({str(getattr(r, "pattern", "")) for r in records})


def _mesh_selections(store: Any) -> dict[str, dict[str, int]]:
    """``"dim:name" -> {nodes, elements}`` of the saved mesh selections.

    ``elements`` is ``-1`` for a node-only set (no element side), the
    same convention as :func:`_group_sizes`.  ``None`` is the reader's
    "no ``/mesh_selections`` group" and dumps as an empty mapping.
    """
    out: dict[str, dict[str, int]] = {}
    if store is None:
        return out
    for dim, tag in store.get_all():
        name = store.get_name(dim, tag)
        n_nodes = int(len(store.get_nodes(dim, tag)["tags"]))
        try:
            n_elems = int(len(store.get_elements(dim, tag)["element_ids"]))
        except ValueError:
            n_elems = -1
        out[f"{dim}:{name}"] = {"nodes": n_nodes, "elements": n_elems}
    return out


def _node_ndf(nodes: Any) -> dict[str, Any]:
    """The per-node ``ndf`` stream as a histogram, ``{"6": 7}``, plus the
    count of nodes whose snapshot carries no ndf (``ndf_for`` raises
    ``LookupError`` for them: the sentinel 0 and the absent dataset alike).

    Every corpus generator leaves ndf to the bridge, so today each dump
    records ``declared == {}`` and every node undeclared: the field holds
    the stream's shape, not yet a value.  It becomes live the day a
    generator declares ``g.node_ndf`` / ``ops.ndf(...)``."""
    declared: dict[str, int] = {}
    undeclared = 0
    for nid in nodes.ids.tolist():
        try:
            value = int(nodes.ndf_for(int(nid)))
        except LookupError:
            undeclared += 1
            continue
        declared[str(value)] = declared.get(str(value), 0) + 1
    return {"declared": dict(sorted(declared.items())), "undeclared": undeclared}


def dump_fem(fem: Any) -> dict[str, Any]:
    """The neutral-zone semantic dump of a :class:`FEMData`."""
    nodes, elements = fem.nodes, fem.elements
    return {
        "dump_format": DUMP_FORMAT,
        "snapshot_id": str(fem.snapshot_id),
        "n_nodes": int(len(nodes.ids)),
        "element_counts": _element_counts(fem),
        "physical_groups": {
            "node_side": _group_sizes(nodes.physical, side="node"),
            "element_side": _group_sizes(elements.physical, side="element"),
        },
        "labels": {
            "node_side": _group_sizes(nodes.labels, side="node"),
            "element_side": _group_sizes(elements.labels, side="element"),
        },
        "loads": {
            "nodal": int(len(nodes.loads)),
            "nodal_patterns": _patterns(nodes.loads),
            "element": int(len(elements.loads)),
            "element_patterns": _patterns(elements.loads),
            "sp": int(len(nodes.sp)),
            "sp_patterns": _patterns(nodes.sp),
        },
        "constraints": {
            "node": int(len(nodes.constraints)),
            "element": int(len(elements.constraints)),
        },
        "masses": int(len(nodes.masses)),
        "mesh_selections": _mesh_selections(fem.mesh_selection),
        "ndf": _node_ndf(nodes),
    }


def dump_stamps(path: str) -> dict[str, int]:
    """The neutral writer's ``/meta`` ``ndm`` and ``ndf`` stamps, read raw.

    Both attributes have been written by every neutral writer from 2.10.0
    on (``FEMData.to_h5`` stamps the mesh dimension, or 0, and its ``ndf``
    argument; ``apeSees.h5`` stamps the ``ops.model`` pair from neutral
    2.34.0 on).  A missing attribute is an error, not a default.
    """
    import h5py

    with h5py.File(path, "r") as f:
        attrs = f["meta"].attrs
        return {"ndm": int(attrs["ndm"]), "ndf": int(attrs["ndf"])}


_MODEL_ACCESSORS = (
    "sections", "transforms", "beam_integration", "time_series",
    "patterns", "recorders", "elements", "fixes", "masses",
)


def dump_model(model: Any) -> dict[str, Any]:
    """The bridge-zone semantic dump of an :class:`OpenSeesModel`."""
    out: dict[str, Any] = {
        "dump_format": DUMP_FORMAT,
        "model_name": str(model.model_name),
        "ndm": int(model.ndm),
        "ndf": int(model.ndf),
        "snapshot_id": str(model.snapshot_id),
        "materials": _norm(dict(model.materials_by_family())),
    }
    for name in _MODEL_ACCESSORS:
        out[name] = _norm(tuple(getattr(model, name)()))
    out["analysis"] = _norm(dict(model.analysis()))
    out["fem"] = dump_fem(model.fem)
    return out


def dump_assembly(asm: Any, zone: Any) -> dict[str, Any]:
    """The ``/assembly`` dump (ADR 0117 D5): what ``Assembly.from_h5``
    re-lists, and every row of the zone as its reader returned it.

    Added with the assembly zone (AS3, #1540). It touches none of the
    dumps above, so no committed era's oracle changes.
    """
    return {
        "dump_format": DUMP_FORMAT,
        "name": str(asm.name),
        "instances": [
            {
                "label": str(i.label),
                "source": i.source.as_posix(),
                "translate": _norm(i.translate),
                "rotate": _norm(i.rotate),
            }
            for i in asm.instances
        ],
        "ties": [
            {
                "master": str(t.master), "slave": str(t.slave),
                "enforce": str(t.enforce), "method": str(t.method),
                "dofs": _norm(t.dofs), "tolerance": float(t.tolerance),
                "name": t.name,
            }
            for t in asm.ties
        ],
        "zone": _norm(zone),
    }
