"""P2-I invisibility proof — the v2 terminals == the legacy chains.

selection-unification-v2 **P2-I** (``docs/plans/selection-unification-v2.md``
§6 P2-I, §6.1 STOP-2; the analogue of P1-K's byte-unchanged proof).

P2-I repoints the five ``.select()`` host hooks to return the two v2
terminals (``EntitySelection`` / ``MeshSelection``) instead of the five
legacy chains.  The legacy chains stay **defined-and-importable but
unwired** (P3 deletes them).  This file is the *invisibility proof*: for
each of the four point host contexts (broker node, broker element,
results node, results element, live mesh) **and** the entity context, it
constructs

  * the NEW type via the repointed host hook, and
  * the LEGACY chain directly, from the *same* seed + the *same* engine,

and asserts they are behaviourally indistinguishable:

  * identical ``_items`` (the canonical atom identity);
  * identical set-algebra results (``| & - ^``);
  * identical spatial-filter results (``in_box`` / ``in_sphere`` /
    ``on_plane`` / ``nearest_to`` / ``where``);
  * identical ``.result()`` (broker / live) / ``.values()`` (results) /
    ``.ids`` / ``.coords`` outputs;
  * identical iteration *content* — modulo the **deliberate** HT8
    pair-presentation: ``MeshSelection.__iter__`` yields
    ``(id, payload)`` (the ratified design), so it is compared against
    the *legacy chain's ``.result()`` payload iteration*, which is the
    same pair view.

If any check fails the v2 terminal is NOT behaviour-faithful → the bug
is in the new type, not here.

No ``openseespy`` dependency (curated no-openseespy CI gate): pure
apeGmsh + gmsh + numpy + a tiny synthetic native HDF5 results file.
Fixture *patterns* are mirrored (not imported) from
``tests/test_selection_idiom.py`` / the per-domain focused tests.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apeGmsh import apeGmsh

# The legacy chains (still defined-and-importable through P2-I) + their
# engine adapters / singletons, and the two v2 terminals.
from apeGmsh.core._selection import (
    EntitySelection,
    GeometryChain,
    Selection,
)
from apeGmsh.mesh._node_chain import NodeChain
from apeGmsh.mesh._elem_chain import ElementChain
from apeGmsh.mesh._mesh_selection import MeshSelection
from apeGmsh.mesh._mesh_selection_chain import (
    MeshSelectionChain,
    engine_for as live_engine_for,
)
from apeGmsh.results import Results
from apeGmsh.results._result_chain import (
    ResultChain,
    engine_for as result_engine_for,
)
from apeGmsh.results.writers import NativeWriter


# =====================================================================
# Helpers
# =====================================================================

def _atoms(chain) -> tuple:
    """The chain's canonical atom tuple (``_items``) — the identity
    set-algebra / spatial verbs operate on (unaffected by the
    ``MeshSelection`` pair-view)."""
    return tuple(chain._items)


def _sorted_atoms(chain) -> list:
    return sorted(_atoms(chain), key=repr)


def _legacy_pairs(legacy_result):
    """Iterate a legacy ``.result()`` payload as the pair view.

    * ``NodeResult``           → ``(id, xyz)``
    * ``GroupResult``          → flattened ``(eid, conn_tuple)`` over
      its ``ElementGroup`` blocks
    * live-mesh ``dict``       → zipped ``(eid, conn_tuple)`` /
      ``(nid, xyz)``
    """
    if isinstance(legacy_result, dict):
        if "element_ids" in legacy_result:
            return [
                (int(e), tuple(int(n) for n in row))
                for e, row in zip(
                    legacy_result["element_ids"],
                    legacy_result["connectivity"],
                )
            ]
        return [
            (int(t), np.asarray(c, dtype=float))
            for t, c in zip(
                legacy_result["tags"], legacy_result["coords"]
            )
        ]
    # NodeResult: iterates (nid, xyz); GroupResult: iterates groups
    if hasattr(legacy_result, "_groups") or (
        hasattr(legacy_result, "__iter__")
        and not hasattr(legacy_result, "coords")
    ):
        # GroupResult — flatten its ElementGroup blocks
        out = []
        for grp in legacy_result:
            for eid, conn in grp:
                out.append((int(eid), tuple(int(n) for n in conn)))
        return out
    # NodeResult
    return [(int(nid), np.asarray(xyz, dtype=float))
            for nid, xyz in legacy_result]


def _assert_pairs_equal(new_pairs, legacy_pairs, ctx: str):
    assert len(new_pairs) == len(legacy_pairs), (
        f"{ctx}: pair-view length differs "
        f"({len(new_pairs)} vs {len(legacy_pairs)})"
    )
    for (ni, nv), (li, lv) in zip(new_pairs, legacy_pairs):
        assert int(ni) == int(li), (
            f"{ctx}: pair id differs ({ni} vs {li})"
        )
        np.testing.assert_array_equal(
            np.asarray(nv), np.asarray(lv),
            err_msg=f"{ctx}: pair payload differs for id {ni}",
        )


def _parity_point(new_chain, legacy_chain, *, box, sphere, plane,
                   near, ctx: str):
    """Full behavioural-equivalence battery for a point context."""
    # ── identical atoms ─────────────────────────────────────
    assert _atoms(new_chain) == _atoms(legacy_chain), (
        f"{ctx}: _items differ"
    )
    assert len(new_chain) == len(legacy_chain)

    # ── identical spatial-filter results ────────────────────
    lo, hi = box
    assert _atoms(new_chain.in_box(lo, hi)) == \
        _atoms(legacy_chain.in_box(lo, hi)), f"{ctx}: in_box differs"
    assert _atoms(new_chain.in_box(lo, hi, inclusive=True)) == \
        _atoms(legacy_chain.in_box(lo, hi, inclusive=True)), (
            f"{ctx}: in_box(inclusive=) differs"
        )
    c, r = sphere
    assert _atoms(new_chain.in_sphere(c, r)) == \
        _atoms(legacy_chain.in_sphere(c, r)), f"{ctx}: in_sphere differs"
    p, n, tol = plane
    assert _atoms(new_chain.on_plane(p, n, tol=tol)) == \
        _atoms(legacy_chain.on_plane(p, n, tol=tol)), (
            f"{ctx}: on_plane differs"
        )
    assert _atoms(new_chain.nearest_to(near, count=2)) == \
        _atoms(legacy_chain.nearest_to(near, count=2)), (
            f"{ctx}: nearest_to differs"
        )
    assert _atoms(new_chain.where(lambda xyz: True)) == \
        _atoms(legacy_chain.where(lambda xyz: True)), (
            f"{ctx}: where differs"
        )

    # ── identical .coords / .ids ────────────────────────────
    np.testing.assert_array_equal(
        np.asarray(new_chain.coords),
        np.asarray(legacy_chain._coords_of(legacy_chain._items)),
        err_msg=f"{ctx}: .coords differ",
    )
    assert new_chain.ids == [int(a) for a in legacy_chain._items], (
        f"{ctx}: .ids differ"
    )


def _parity_set_algebra(new_a, new_b, leg_a, leg_b, ctx: str):
    for op in ("union", "intersect", "difference",
               "symmetric_difference"):
        nv = _atoms(getattr(new_a, op)(new_b))
        lv = _atoms(getattr(leg_a, op)(leg_b))
        assert nv == lv, f"{ctx}: {op} differs ({nv} vs {lv})"
    # operators agree with named methods
    assert _atoms(new_a | new_b) == _atoms(leg_a | leg_b)
    assert _atoms(new_a & new_b) == _atoms(leg_a & leg_b)
    assert _atoms(new_a - new_b) == _atoms(leg_a - leg_b)
    assert _atoms(new_a ^ new_b) == _atoms(leg_a ^ leg_b)


# =====================================================================
# Fixtures (patterns mirrored from tests/test_selection_idiom.py)
# =====================================================================

@pytest.fixture(scope="module")
def cube_fem():
    g = apeGmsh(model_name="p2i_cube", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                                 label="box")
        g.physical.add_volume("box", name="Body")
        g.mesh.structured.set_transfinite_box("box", n=3)
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data(dim=3)
    finally:
        g.end()
    return fem


@pytest.fixture
def live():
    g = apeGmsh(model_name="p2i_live", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                                 label="box")
        g.physical.add_volume("box", name="Body")
        g.mesh.structured.set_transfinite_box("box", n=3)
        g.mesh.generation.generate(dim=3)
        yield g
    finally:
        g.end()


@pytest.fixture
def cube_geo():
    g = apeGmsh(model_name="p2i_geo", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                                 label="box")
        g.model.sync()
        faces = g.model.queries.boundary("box", dim=3, oriented=False)
        g.physical.add_surface([int(t) for _d, t in faces],
                               name="Faces")
        yield g
    finally:
        g.end()


def _make_results_with_fem(tmp_path: Path):
    """Pattern from tests/test_selection_idiom._make_results_with_fem."""
    path = tmp_path / "synthetic.h5"
    time = np.array([0.0, 1.0])
    node_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    ux = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                   [1.1, 1.2, 1.3, 1.4, 1.5]])
    elem_idx = np.array([10], dtype=np.int64)
    gforce = np.array(
        [[[10.0, 11.0, 12.0, 13.0]],
         [[20.0, 21.0, 22.0, 23.0]]],
        dtype=np.float64,
    )
    with NativeWriter(path) as w:
        w.open(source_type="domain_capture")
        sid = w.begin_stage(name="static", kind="static", time=time)
        w.write_nodes(sid, "partition_0", node_ids=node_ids,
                      components={"displacement_x": ux})
        w.write_nodal_forces_group(
            sid, "partition_0", "group_0",
            class_tag=1, frame="global",
            element_index=elem_idx,
            components={"globalForce": gforce},
        )
        w.end_stage()

    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [5.0, 5.0, 5.0],
    ], dtype=np.float64)
    type_info = SimpleNamespace(name="quad4")

    def _resolve(*, element_type=None):
        return (
            np.array([10], dtype=np.int64),
            np.array([[1, 2, 3, 4]], dtype=np.int64),
        )

    nodes_ns = SimpleNamespace(
        ids=node_ids,
        coords=coords,
        physical=SimpleNamespace(node_ids=lambda n: {
            "TopRow": np.array([3, 4], dtype=np.int64),
        }[n]),
        labels=SimpleNamespace(
            node_ids=lambda n: np.array([], dtype=np.int64),
        ),
    )
    elements_ns = SimpleNamespace(
        ids=np.array([10], dtype=np.int64),
        types=[type_info],
        resolve=_resolve,
        physical=SimpleNamespace(
            element_ids=lambda n: np.array([10], dtype=np.int64),
        ),
        labels=SimpleNamespace(
            element_ids=lambda n: np.array([], dtype=np.int64),
        ),
    )
    fem = SimpleNamespace(
        snapshot_id="testhash",
        nodes=nodes_ns,
        elements=elements_ns,
    )
    return Results.from_native(path, fem=fem)


# =====================================================================
# 1. Broker node — MeshSelection vs NodeChain
# =====================================================================

def test_parity_broker_node(cube_fem):
    fem = cube_fem
    new = fem.nodes.select(pg="Body")
    legacy = NodeChain(list(new._items), _engine=fem.nodes)
    assert isinstance(new, MeshSelection)
    assert isinstance(legacy, NodeChain)

    _parity_point(
        new, legacy,
        box=((0, 0, 0), (1, 1, 1)),
        sphere=((0.0, 0.0, 0.0), 0.6),
        plane=((0, 0, 0), (0, 0, 1), 1e-9),
        near=(0.0, 0.0, 0.0),
        ctx="broker-node",
    )

    base = [int(x) for x in fem.nodes.get(pg="Body").ids]
    na = fem.nodes.select(ids=base[:5])
    nb = fem.nodes.select(ids=base[3:8])
    la = NodeChain(list(na._items), _engine=fem.nodes)
    lb = NodeChain(list(nb._items), _engine=fem.nodes)
    _parity_set_algebra(na, nb, la, lb, "broker-node")

    # .result() parity (same NodeResult ids/coords) + pair-view content
    nr_new = new.result()
    nr_leg = legacy._materialize()
    assert type(nr_new) is type(nr_leg)
    np.testing.assert_array_equal(
        np.asarray(nr_new.ids, dtype=np.int64),
        np.asarray(nr_leg.ids, dtype=np.int64),
    )
    np.testing.assert_array_equal(nr_new.coords, nr_leg.coords)
    _assert_pairs_equal(
        list(new), _legacy_pairs(nr_leg), "broker-node iter"
    )


# =====================================================================
# 2. Broker element — MeshSelection vs ElementChain
# =====================================================================

def test_parity_broker_element(cube_fem):
    fem = cube_fem
    new = fem.elements.select(pg="Body")
    legacy = ElementChain(list(new._items), _engine=fem.elements)
    assert isinstance(new, MeshSelection)
    assert isinstance(legacy, ElementChain)

    _parity_point(
        new, legacy,
        box=((0.0, 0.0, 0.0), (0.75, 0.75, 0.75)),
        sphere=((0.0, 0.0, 0.0), 0.5),
        plane=((0, 0, 0.25), (0, 0, 1), 0.1),
        near=(0.0, 0.0, 0.0),
        ctx="broker-element",
    )

    eids = sorted(int(e) for e, _ in new)
    na = fem.elements.select(ids=eids[:5])
    nb = fem.elements.select(ids=eids[3:])
    la = ElementChain(list(na._items), _engine=fem.elements)
    lb = ElementChain(list(nb._items), _engine=fem.elements)
    _parity_set_algebra(na, nb, la, lb, "broker-element")

    gr_new = new.result()
    gr_leg = legacy._materialize()
    assert type(gr_new) is type(gr_leg)
    np.testing.assert_array_equal(
        np.asarray(gr_new.ids, dtype=np.int64),
        np.asarray(gr_leg.ids, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        gr_new.connectivity, gr_leg.connectivity
    )
    # .groups() parity (per-type ElementGroup blocks preserved)
    assert [g.type_name for g in new.groups()] == \
        [g.type_name for g in gr_leg]
    _assert_pairs_equal(
        list(new), _legacy_pairs(gr_leg), "broker-element iter"
    )


# =====================================================================
# 3a. Results node — MeshSelection vs ResultChain (node level)
# =====================================================================

def test_parity_results_node(tmp_path):
    r = _make_results_with_fem(tmp_path)
    new = r.nodes.select()
    eng = result_engine_for(r, r.nodes, "node")
    legacy = ResultChain(list(new._items), _engine=eng)
    assert isinstance(new, MeshSelection)
    assert isinstance(legacy, ResultChain)
    assert new._level == "node" == legacy._level

    _parity_point(
        new, legacy,
        box=((0.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
        sphere=((0.0, 0.0, 0.0), 1.5),
        plane=((0, 0, 0), (0, 0, 1), 1e-9),
        near=(0.0, 0.0, 0.0),
        ctx="results-node",
    )

    na = r.nodes.select(ids=[1, 2, 3])
    nb = r.nodes.select(ids=[2, 3, 4, 5])
    la = ResultChain(
        list(na._items),
        _engine=result_engine_for(r, r.nodes, "node"),
    )
    lb = ResultChain(
        list(nb._items),
        _engine=result_engine_for(r, r.nodes, "node"),
    )
    _parity_set_algebra(na, nb, la, lb, "results-node")

    # .values(...) == legacy ResultChain.get(...) — same slab
    s_new = (r.nodes.select(pg="TopRow")
             .values(component="displacement_x"))
    s_leg = ResultChain(
        list(r.nodes.select(pg="TopRow")._items),
        _engine=result_engine_for(r, r.nodes, "node"),
    ).get(component="displacement_x")
    assert type(s_new) is type(s_leg)
    np.testing.assert_array_equal(
        np.asarray(s_new.node_ids), np.asarray(s_leg.node_ids)
    )
    np.testing.assert_array_equal(s_new.values, s_leg.values)
    # bare .result() fails loud identically (directs to component)
    with pytest.raises(RuntimeError, match="needs .get.component"):
        new.result()


# =====================================================================
# 3b. Results element — MeshSelection vs ResultChain (element level)
# =====================================================================

def test_parity_results_element(tmp_path):
    r = _make_results_with_fem(tmp_path)
    new = r.elements.select()
    eng = result_engine_for(r, r.elements, "element")
    legacy = ResultChain(list(new._items), _engine=eng)
    assert isinstance(new, MeshSelection)
    assert isinstance(legacy, ResultChain)
    assert new._level == "element" == legacy._level

    assert _atoms(new) == _atoms(legacy)
    # the single quad centroid is (0.5, 0.5, 0.0)
    box = ((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0))
    assert _atoms(new.in_box(*box)) == _atoms(legacy.in_box(*box))
    assert _atoms(
        new.on_plane((0, 0, 0), (0, 0, 1), tol=1e-9)
    ) == _atoms(
        legacy.on_plane((0, 0, 0), (0, 0, 1), tol=1e-9)
    )
    np.testing.assert_array_equal(
        np.asarray(new.coords),
        np.asarray(legacy._coords_of(legacy._items)),
    )

    s_new = (r.elements.select(pg="Beams")
             .values(component="globalForce"))
    s_leg = ResultChain(
        list(r.elements.select(pg="Beams")._items),
        _engine=result_engine_for(r, r.elements, "element"),
    ).get(component="globalForce")
    assert type(s_new) is type(s_leg)
    np.testing.assert_array_equal(
        np.asarray(s_new.element_ids), np.asarray(s_leg.element_ids)
    )
    np.testing.assert_array_equal(s_new.values, s_leg.values)


# =====================================================================
# 4. Live mesh — MeshSelection vs MeshSelectionChain
# =====================================================================

def test_parity_live_mesh_node(live):
    ms = live.mesh_selection
    new = ms.select()
    eng = live_engine_for(ms, "node", 0)
    legacy = MeshSelectionChain(list(new._items), _engine=eng)
    assert isinstance(new, MeshSelection)
    assert isinstance(legacy, MeshSelectionChain)

    _parity_point(
        new, legacy,
        box=((0, 0, 0), (1, 1, 1)),
        sphere=((0.0, 0.0, 0.0), 0.6),
        plane=((0, 0, 0), (0, 0, 1), 1e-9),
        near=(0.0, 0.0, 0.0),
        ctx="live-node",
    )

    base, _ = ms._get_mesh_nodes()
    base = [int(n) for n in base]
    na = ms.select(ids=base[:5])
    nb = ms.select(ids=base[3:8])
    la = MeshSelectionChain(
        list(na._items), _engine=live_engine_for(ms, "node", 0)
    )
    lb = MeshSelectionChain(
        list(nb._items), _engine=live_engine_for(ms, "node", 0)
    )
    _parity_set_algebra(na, nb, la, lb, "live-node")

    # .result() dict-shape parity + pair-view content
    d_new = new.result()
    d_leg = legacy._materialize()
    assert set(d_new) == set(d_leg)
    np.testing.assert_array_equal(
        np.asarray(d_new["tags"], dtype=np.int64),
        np.asarray(d_leg["tags"], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.asarray(d_new["coords"], dtype=float),
        np.asarray(d_leg["coords"], dtype=float),
    )
    _assert_pairs_equal(
        list(new), _legacy_pairs(d_leg), "live-node iter"
    )

    # .save_as registers into the live mesh-selection store (the only
    # context where the mutable store is reachable) and round-trips.
    new.in_box((-1, -1, -1), (2, 2, 2)).save_as("p2i_saved")
    tag = ms.get_tag(0, "p2i_saved")
    assert tag is not None
    saved = set(int(t) for t in ms.get_nodes(0, tag)["tags"])
    expect = set(
        int(a) for a in ms.select().in_box((-1, -1, -1), (2, 2, 2))._items
    )
    assert saved == expect


def test_parity_live_mesh_element(live):
    ms = live.mesh_selection
    new = ms.select(level="element", dim=3)
    eng = live_engine_for(ms, "element", 3)
    legacy = MeshSelectionChain(list(new._items), _engine=eng)
    assert isinstance(new, MeshSelection)
    assert isinstance(legacy, MeshSelectionChain)
    assert new._level == "element" == legacy._level

    assert _atoms(new) == _atoms(legacy)
    box = ((-1, -1, -1), (2, 2, 2))
    assert _atoms(new.in_box(*box)) == _atoms(legacy.in_box(*box))
    assert _atoms(
        new.on_plane((0, 0, 0.25), (0, 0, 1), tol=0.1)
    ) == _atoms(
        legacy.on_plane((0, 0, 0.25), (0, 0, 1), tol=0.1)
    )
    np.testing.assert_array_equal(
        np.asarray(new.coords),
        np.asarray(legacy._coords_of(legacy._items)),
    )
    # element-level .result() dict parity + pair-view content
    d_leg = legacy._materialize()
    _assert_pairs_equal(
        list(new), _legacy_pairs(d_leg), "live-element iter"
    )


# =====================================================================
# 5. Entity context — EntitySelection vs GeometryChain
# =====================================================================

def test_parity_entity(cube_geo):
    g = cube_geo
    new = g.model.select("Faces")
    legacy = GeometryChain(
        list(new._items), _engine=g.model.queries
    )
    assert isinstance(new, EntitySelection)
    assert isinstance(legacy, GeometryChain)

    # identical atoms + entity-family spatial parity
    assert _atoms(new) == _atoms(legacy)
    enclosing = ((-0.1, -0.1, -0.1), (1.1, 1.1, 1.1))
    half = ((-1.0, -1.0, -1.0), (0.5, 2.0, 2.0))
    assert _atoms(new.in_box(*enclosing)) == \
        _atoms(legacy.in_box(*enclosing))
    assert _atoms(new.in_box(*half)) == _atoms(legacy.in_box(*half))
    assert _atoms(
        new.on_plane((0, 0, 0), (0, 0, 1), tol=1e-6)
    ) == _atoms(
        legacy.on_plane((0, 0, 0), (0, 0, 1), tol=1e-6)
    )
    assert _atoms(new.in_sphere((0.5, 0.5, 0.5), 5.0)) == \
        _atoms(legacy.in_sphere((0.5, 0.5, 0.5), 5.0))
    assert _atoms(new.nearest_to((0, 0, 0), count=3)) == \
        _atoms(legacy.nearest_to((0, 0, 0), count=3))
    np.testing.assert_array_equal(
        np.asarray(new._coords_of(new._items)),
        np.asarray(legacy._coords_of(legacy._items)),
    )

    # in_box(inclusive=) is loud on BOTH (entity family has no
    # half-open knob — R3)
    for bad in (dict(inclusive=True), dict(inclusive=False),
                dict(bogus=1)):
        with pytest.raises(TypeError):
            new.in_box(*enclosing, **bad)
        with pytest.raises(TypeError):
            legacy.in_box(*enclosing, **bad)

    # set-algebra parity
    a_new = new.on_plane((0, 0, 0), (0, 0, 1), tol=1e-6)
    b_new = new.on_plane((0, 0, 0), (1, 0, 0), tol=1e-6)
    a_leg = legacy.on_plane((0, 0, 0), (0, 0, 1), tol=1e-6)
    b_leg = legacy.on_plane((0, 0, 0), (1, 0, 0), tol=1e-6)
    _parity_set_algebra(a_new, b_new, a_leg, b_leg, "entity")

    # iteration content identical (entity family yields bare (dim,tag);
    # NOT pair-presented — only MeshSelection overrides __iter__)
    assert list(new) == list(legacy)

    # .result() → the byte-unchanged legacy Selection, equal content
    sel_new = new.result()
    sel_leg = legacy._materialize()
    assert isinstance(sel_new, Selection)
    assert isinstance(sel_leg, Selection)
    assert list(sel_new) == list(sel_leg)

    # .to_label / .to_physical are distinct registries (ADR 0015):
    # to_physical → raw PG (Tier-2); to_label → _label: PG (Tier-1).
    g.model.select("Faces").to_physical("ClashName")
    g.model.select("Faces").to_label("ClashName")
    pgs = {
        __import__("gmsh").model.getPhysicalName(d, t)
        for d, t in __import__("gmsh").model.getPhysicalGroups()
    }
    assert "ClashName" in pgs               # Tier-2 raw
    assert "_label:ClashName" in pgs        # Tier-1 prefixed
    # the two coexist — not merged (the ADR-locked invariant)
    assert "ClashName" != "_label:ClashName"

    # .to_dataframe() — NEW terminal, correct column set, no viz import
    df = g.model.select("Faces").to_dataframe()
    assert list(df.columns) == [
        "dim", "tag", "kind", "label", "x", "y", "z", "mass",
    ]
    assert len(df) == 6
