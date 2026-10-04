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


_LINE3 = {1: (0, 0, 0), 2: (2, 0, 0), 3: (1, 0, 0)}
_Q = np.array([0.0, 0.0, -1.0])


@pytest.mark.parametrize("call", [
    pytest.param(
        lambda r, d: r.resolve_line_per_edge_tributary(d, [(1, 2, _Q)]),
        id="per_edge_tributary"),
    pytest.param(
        lambda r, d: r.resolve_line_per_edge_consistent(d, [([1, 2, 3], _Q)]),
        id="per_edge_consistent"),
    pytest.param(
        lambda r, d: r.resolve_line_per_edge_consistent_varying(
            d, [([1, 2, 3], _Q, lambda xyz: 1.0)]),
        id="per_edge_consistent_varying"),
])
def test_per_edge_line_paths_stamp_line(call):
    """The three per-edge line reductions (the ``normal=True`` / callable
    magnitude routes) stamp ``line`` like the plain ones."""
    defn = defs.LineLoadDef(target="x", magnitude=1.0, reduction="consistent")
    recs = call(_resolver(_LINE3), defn)
    assert recs
    assert {r.source for r in recs} == {NodalLoadSource.LINE}


def test_chain_phase_router_stamps_point():
    """``route_def_to_fem`` builds point-load records itself, outside the
    resolver: it must stamp the same ``point`` source."""
    from apeGmsh._kernel.resolvers._chain_phase_router import route_def_to_fem
    from tests.test_chain_phase_fail_loud import _quad_face_fem

    fem = _quad_face_fem()
    defn = defs.PointLoadDef(target=[1, 2], force_xyz=(0.0, 0.0, -5.0),
                             pattern="live")
    new_fem = route_def_to_fem(fem, defn)
    assert new_fem is not None
    recs = new_fem.nodes.loads.by_pattern("live")
    assert {r.node_id for r in recs} == {1, 2}
    assert {r.source for r in recs} == {NodalLoadSource.POINT}
