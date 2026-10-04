"""``NodalLoadRecord.source`` (#1338): the resolver stamps every nodal
record with the definition kind it was reduced from, and the vocabulary in
``NodalLoadSource`` partitions exactly the kinds that resolve to nodal
records.

The partition test is the completeness oracle: a new ``LoadDef`` kind that
produces nodal records must be classified as a body or a boundary load
here, or the bridge's double-count guard fails closed on it.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from apeGmsh._kernel.defs import loads as defs
from apeGmsh._kernel.records import NodalLoadSource
from apeGmsh._kernel.resolvers._load_resolver import LoadResolver

#: Definitions that resolve to ``SPRecord``, never to a nodal load.
_SP_KINDS = {"face_sp", "point_sp"}


def _def_kinds() -> set[str]:
    kinds = set()
    for name in defs.__all__:
        cls = getattr(defs, name)
        if cls is defs.LoadDef or not dataclasses.is_dataclass(cls):
            continue
        kinds.add(cls(target="x").kind)
    return kinds


def test_vocabulary_partitions_the_nodal_definition_kinds():
    assert NodalLoadSource.ALL == _def_kinds() - _SP_KINDS
    assert not (NodalLoadSource.BODY_KINDS & NodalLoadSource.BOUNDARY_KINDS)
    assert NodalLoadSource.ALL == (
        NodalLoadSource.BODY_KINDS | NodalLoadSource.BOUNDARY_KINDS
    )


def test_record_default_is_unknown():
    from apeGmsh._kernel.records._loads import NodalLoadRecord
    assert NodalLoadRecord(node_id=1, force_xyz=(0.0, 0.0, -1.0)).source is None


_TET4 = {1: (0, 0, 0), 2: (1, 0, 0), 3: (0, 1, 0), 4: (0, 0, 1)}


def _resolver(coords_by_tag):
    tags = np.array(sorted(coords_by_tag), dtype=int)
    coords = np.array([coords_by_tag[int(t)] for t in tags], dtype=float)
    return LoadResolver(tags, coords)


@pytest.mark.parametrize("defn, call", [
    (defs.PointLoadDef(target="x", force_xyz=(0.0, 0.0, -1.0)),
     lambda r, d: r.resolve_point(d, {1, 2})),
    (defs.LineLoadDef(target="x", q_xyz=(0.0, 0.0, -1.0)),
     lambda r, d: r.resolve_line_tributary(d, [(1, 2)])),
    (defs.LineLoadDef(target="x", q_xyz=(0.0, 0.0, -1.0),
                      reduction="consistent"),
     lambda r, d: r.resolve_line_consistent(d, [[1, 2]])),
    (defs.SurfaceLoadDef(target="x", magnitude=5.0, mode="pressure"),
     lambda r, d: r.resolve_surface_tributary(d, [[1, 2, 3]])),
    (defs.SurfaceLoadDef(target="x", magnitude=5.0, mode="traction",
                         direction=(0.0, 0.0, -1.0), reduction="consistent"),
     lambda r, d: r.resolve_surface_consistent(d, [[1, 2, 3]])),
    (defs.GravityLoadDef(target="x", density=1.0, g=(0.0, 0.0, -1.0)),
     lambda r, d: r.resolve_gravity_tributary(d, [np.array([1, 2, 3, 4])])),
    (defs.GravityLoadDef(target="x", density=1.0, g=(0.0, 0.0, -1.0),
                         reduction="consistent"),
     lambda r, d: r.resolve_gravity_consistent(d, [np.array([1, 2, 3, 4])])),
    (defs.BodyLoadDef(target="x", force_per_volume=(0.0, 0.0, -1.0)),
     lambda r, d: r.resolve_body_tributary(d, [np.array([1, 2, 3, 4])])),
    (defs.FaceLoadDef(target="x", force_xyz=(0.0, 0.0, -3.0)),
     lambda r, d: r.resolve_face_load(d, [1, 2, 3])),
], ids=lambda x: getattr(x, "kind", None) or "")
def test_resolver_stamps_source_with_the_definition_kind(defn, call):
    recs = call(_resolver(_TET4), defn)
    assert recs, "the path must produce records for the stamp to be tested"
    assert {r.source for r in recs} == {defn.kind}
    assert defn.kind in NodalLoadSource.ALL
