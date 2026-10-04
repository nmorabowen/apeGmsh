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
neutral 2.10.0 / opensees 2.11.0 (``FEMData.snapshot_id``, the node and
element composites, ``NamedGroupSet.get_all/get_name/node_ids/
element_ids``, the record sets' ``len``, and ``OpenSeesModel``'s record
accessors).  Editing it changes the oracle for every committed era, so
rebuild the corpus after any change.
"""
from __future__ import annotations

import dataclasses
import enum
import math
from typing import Any

DUMP_FORMAT = 1


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
    }


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
