"""S3b (P3-R) — the FEMData ``.select()`` point-family terminal.

Phase S3b of the selection-unification work
(``docs/plans/selection-unification.md`` §5/§6), carried through
selection-unification-v2 P3-R (``docs/plans/selection-unification-v2.md``
§6.2 / §6.3).  ``fem.nodes.select(...)`` / ``fem.elements.select(...)``
return the unified point-family terminal ``MeshSelection`` (the legacy
``NodeChain`` / ``ElementChain`` are deleted), seeded by delegating
name resolution verbatim to the existing broker resolvers
(``NodeComposite._resolve_nodes`` / ``ElementComposite._resolve_elem_ids``
— the exact methods the now-removed ``.get()`` used; FP-4 swallow
asymmetry preserved *by reuse*).

What this locks:

* ``fem.nodes.select(...)`` / ``fem.elements.select(...)`` return
  ``MeshSelection``, seeded by delegating to the existing
  ``_resolve_nodes`` / ``_resolve_elem_ids`` (a spy proves the call —
  delegation, not a re-implementation).
* Node & element daisy-chaining:
  ``.select(pg=...).in_box(...).on_plane(...)`` composes, every verb
  returning ``MeshSelection``.
* Point-family ``in_box`` is half-open ``[lo, hi)`` by default and
  closed ``[lo, hi]`` with ``inclusive=True`` (R4) — for both engines.
* Set algebra ``| & -`` with insertion-order dedup; cross-engine and
  cross-type combination is loud.
* ``.result()`` returns ``NodeResult`` / ``GroupResult``; its id set
  (and ``.resolve()`` flat ids/conn) is frozen as an explicit literal
  against the deterministic fixture (P3-K proved ``select`` ≡ the
  removed ``get``; the parity-vs-``.get()`` half is gone with ``.get``).
* The element centroid path fails loud if connectivity references an
  unknown node (never the silent row-0 substitution the generic
  ``_mesh_filters.element_centroids`` does).

No ``openseespy`` dependency (curated no-openseespy CI gate): pure
apeGmsh + gmsh + numpy.  A deterministic structured unit cube (3x3x3
node lattice -> 27 nodes [ids 1..27], 8 hex8 cells [ids 1..8], all
coords in {0, 0.5, 1}) is the fixture, so every count is an exact
integer literal.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh._kernel.chain import SelectionChain
# selection-unification-v2 P3-R (§6.2 / §6.3): the legacy ``NodeChain``
# / ``ElementChain`` modules are **deleted** and the broker
# ``fem.nodes.get`` / ``fem.elements.get`` / ``get_ids`` /
# ``get_coords`` / ``resolve`` surface is **removed**; the unified
# point-family terminal ``fem.nodes.select(...)`` /
# ``fem.elements.select(...)`` → ``MeshSelection`` is the sole survivor
# and migration target.  The legacy-class structural-shape tests are
# deleted (now covered by ``tests/test_selection_idiom.py`` over the
# two v2 terminals); the behaviour-bearing tests below keep their full
# coverage on the ``.select(...)`` path, with the (now-removed)
# parity-vs-``.get()`` halves frozen as explicit literals against the
# deterministic 3x3x3 fixture (the proof-file freeze pattern — P3-K
# already proved ``select`` ≡ the removed ``get``).
from apeGmsh.mesh._mesh_selection import MeshSelection
from apeGmsh.mesh.FEMData import (
    FEMData, NodeComposite, ElementComposite, NodeResult,
)
from apeGmsh.mesh._element_types import GroupResult


# =====================================================================
# Fixture — structured unit cube, fully deterministic
# =====================================================================

@pytest.fixture(scope="module")
def cube_fem():
    """3x3x3 lattice: 27 nodes at every {0,0.5,1}^3, 8 hex8 cells.

    Volume label/PG ``Body``.  Transfinite ``n=3`` makes every count an
    exact integer (no "about N"): 8 cell centroids sit at the 8
    combinations of {0.25, 0.75}.
    """
    g = apeGmsh(model_name="s3b_cube", verbose=False)
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


def _ids(obj) -> list[int]:
    return sorted(int(x) for x in obj.ids)


# =====================================================================
# Class-shape invariant — the broker hooks return the v2 terminal
# (the legacy NodeChain/ElementChain structural-shape + __init_subclass__
# gate tests are deleted; covered by tests/test_selection_idiom.py)
# =====================================================================

def test_select_hooks_return_point_family_meshselection(cube_fem):
    fem = cube_fem
    n = fem.nodes.select(pg="Body")
    e = fem.elements.select(pg="Body")
    assert isinstance(n, MeshSelection) and isinstance(e, MeshSelection)
    assert issubclass(MeshSelection, SelectionChain)
    assert n.FAMILY == "point" and e.FAMILY == "point"


# =====================================================================
# .select() host hook — delegates to the existing resolver
# =====================================================================

def test_node_select_returns_nodechain_seeded_by_resolver(cube_fem):
    fem = cube_fem
    sel = fem.nodes.select(pg="Body")
    assert isinstance(sel, MeshSelection)      # P2-I: was NodeChain
    assert sel.FAMILY == "point"
    # 27-node lattice volume PG
    assert len(sel) == 27
    # no-arg seeds every domain node
    assert len(fem.nodes.select()) == len(fem.nodes)
    # explicit ids= path
    assert _ids(fem.nodes.select(ids=[1, 2, 3]).result()) == [1, 2, 3]


def test_element_select_returns_elementchain_seeded_by_resolver(cube_fem):
    fem = cube_fem
    sel = fem.elements.select(pg="Body")
    assert isinstance(sel, MeshSelection)      # P2-I: was ElementChain
    assert sel.FAMILY == "point"
    assert len(sel) == 8                       # 8 hex8 cells
    assert len(fem.elements.select()) == len(fem.elements)
    eid0 = int(fem.elements.ids[0])
    assert _ids(fem.elements.select(ids=[eid0]).result()) == [eid0]


def test_node_select_delegates_to_resolve_nodes(cube_fem, monkeypatch):
    # Prove the host hook calls the EXISTING broker node resolver
    # (NodeComposite._resolve_nodes — the one .get() uses) rather than
    # re-implementing tier logic. FP-4: the node path's KeyError-only
    # swallow asymmetry is preserved precisely because it is reused.
    fem = cube_fem
    seen = {}
    real = NodeComposite._resolve_nodes

    def _spy(self, target, *, pg, label, tag, dim=None):
        seen["called"] = True
        seen["pg"] = pg
        return real(self, target, pg=pg, label=label, tag=tag, dim=dim)

    monkeypatch.setattr(NodeComposite, "_resolve_nodes", _spy)
    out = fem.nodes.select(pg="Body").result()
    assert seen.get("called") is True
    assert seen.get("pg") == "Body"
    assert isinstance(out, NodeResult)


def test_element_select_delegates_to_resolve_elem_ids(cube_fem, monkeypatch):
    # Prove the host hook calls the EXISTING broker element resolver
    # (ElementComposite._resolve_elem_ids — the one .get() uses). FP-4:
    # the element path's (KeyError, ValueError) swallow is preserved
    # because the same method is reused (not re-implemented).
    fem = cube_fem
    seen = {}
    real = ElementComposite._resolve_elem_ids

    def _spy(self, target, *, pg, label, tag):
        seen["called"] = True
        seen["pg"] = pg
        return real(self, target, pg=pg, label=label, tag=tag)

    monkeypatch.setattr(ElementComposite, "_resolve_elem_ids", _spy)
    out = fem.elements.select(pg="Body").result()
    assert seen.get("called") is True
    assert seen.get("pg") == "Body"
    assert isinstance(out, GroupResult)


# =====================================================================
# Daisy-chaining + point-family spatial semantics
# =====================================================================

def test_node_chain_daisychains_each_verb_returns_nodechain(cube_fem):
    fem = cube_fem
    step1 = fem.nodes.select(pg="Body")
    step2 = step1.in_box((-1, -1, -1), (2, 2, 2))
    step3 = step2.on_plane((0, 0, 0), (0, 0, 1), tol=1e-9)
    for s in (step1, step2, step3):
        assert isinstance(s, MeshSelection)    # P2-I: was NodeChain
    # full fluent one-liner: z=0 plane of the lattice = 9 nodes
    chained = (fem.nodes.select(pg="Body")
                 .in_box((-1, -1, -1), (2, 2, 2))
                 .on_plane((0, 0, 0), (0, 0, 1), tol=1e-9))
    assert len(chained) == 9


def test_element_chain_daisychains_each_verb_returns_elementchain(cube_fem):
    fem = cube_fem
    step1 = fem.elements.select(pg="Body")
    step2 = step1.in_box((-1, -1, -1), (2, 2, 2))
    step3 = step2.on_plane((0, 0, 0.25), (0, 0, 1), tol=0.1)
    for s in (step1, step2, step3):
        assert isinstance(s, MeshSelection)    # P2-I: was ElementChain
    # 8 cells; centroids at z in {0.25, 0.75}; the z=0.25 plane keeps 4
    chained = (fem.elements.select(pg="Body")
                 .in_box((-1, -1, -1), (2, 2, 2))
                 .on_plane((0, 0, 0.25), (0, 0, 1), tol=0.1))
    assert len(chained) == 4


def test_point_family_in_box_half_open_default_and_inclusive(cube_fem):
    fem = cube_fem

    # Nodes: half-open [0,1)^3 drops the entire upper {x|y|z = 1} shell;
    # only the single interior-ish corner block of 8 lattice nodes with
    # every coord in {0, 0.5} survives. inclusive=True restores all 27.
    alln = fem.nodes.select(pg="Body")
    half = alln.in_box((0, 0, 0), (1, 1, 1))
    closed = alln.in_box((0, 0, 0), (1, 1, 1), inclusive=True)
    assert len(half) == 8
    assert len(closed) == 27

    # Elements: centroids are {0.25, 0.75}^3. Half-open box with upper
    # bound exactly 0.75 EXCLUDES centroids sitting on 0.75 -> only the
    # single (0.25,0.25,0.25) cell. inclusive=True keeps all 8.
    alle = fem.elements.select(pg="Body")
    he = alle.in_box((0.0, 0.0, 0.0), (0.75, 0.75, 0.75))
    ce = alle.in_box((0.0, 0.0, 0.0), (0.75, 0.75, 0.75), inclusive=True)
    assert len(he) == 1
    assert len(ce) == 8


def test_point_family_sphere_plane_nearest_where_centroid(cube_fem):
    fem = cube_fem
    alle = fem.elements.select(pg="Body")

    # in_sphere: tiny closed ball at one centroid catches exactly it.
    one = alle.in_sphere((0.25, 0.25, 0.25), 0.01)
    assert len(one) == 1

    # nearest_to: order by centroid distance; count caps the result.
    near1 = alle.nearest_to((0.25, 0.25, 0.25), count=1)
    assert len(near1) == 1
    assert set(tuple(near1)) == set(tuple(one))
    assert len(alle.nearest_to((0.5, 0.5, 0.5), count=3)) == 3

    # where: predicate on the centroid row -> cells with centroid x<0.5
    # is the 4 cells in the x=0.25 layer.
    w = alle.where(lambda xyz: xyz[0] < 0.5)
    assert len(w) == 4

    # Node-side plane: y=0 face of the lattice = 9 nodes.
    nface = fem.nodes.select(pg="Body").on_plane(
        (0, 0, 0), (0, 1, 0), tol=1e-9)
    assert len(nface) == 9

    # point-family input validation is loud (both chains).
    with pytest.raises(ValueError, match="radius must be non-negative"):
        alle.in_sphere((0, 0, 0), -1.0)
    with pytest.raises(ValueError, match="tolerance must be non-negative"):
        fem.nodes.select(pg="Body").on_plane((0, 0, 0), (0, 0, 1),
                                              tol=-1.0)
    with pytest.raises(ValueError, match="normal vector has zero length"):
        alle.on_plane((0, 0, 0), (0, 0, 0), tol=1e-6)


# =====================================================================
# Set algebra — insertion-order dedup; cross-* is loud
# =====================================================================

def test_set_algebra_union_intersect_difference_symmetric(cube_fem):
    fem = cube_fem
    base = [int(x) for x in fem.nodes.select(pg="Body").ids]
    a = fem.nodes.select(ids=base[:5])
    b = fem.nodes.select(ids=base[3:8])

    assert len(a | b) == 8                     # union (5 + 5 - 2 dup)
    assert len(a & b) == 2                     # intersection
    assert len(a - b) == 3                     # difference
    assert len(a ^ b) == 6                     # symmetric difference
    assert len(a | a) == 5                     # idempotent (one law)
    # named aliases match the operators — compare on the ATOMS
    # (``_items``); P2-I ``MeshSelection.__iter__`` yields
    # ``(id, payload)`` pairs (ndarray payload → ambiguous tuple ``==``),
    # but set-algebra is defined on ``_items`` and is unaffected.
    assert (a.union(b))._items == (a | b)._items
    assert (a.difference(b))._items == (a - b)._items
    # every set-algebra result is itself the chain type (chainable)
    for s in (a | b, a & b, a - b, a ^ b):
        assert isinstance(s, MeshSelection)    # P2-I: was NodeChain

    # element side too
    eids = [int(x) for x in fem.elements.select(pg="Body").ids]
    ea = fem.elements.select(ids=eids[:5])
    eb = fem.elements.select(ids=eids[3:])
    assert len(ea | eb) == 8
    assert isinstance(ea & eb, MeshSelection)  # P2-I: was ElementChain


def test_cross_type_and_cross_engine_set_algebra_is_loud(cube_fem):
    fem = cube_fem
    nc = fem.nodes.select(ids=[1])
    ec = fem.elements.select(ids=[int(fem.elements.ids[0])])
    # node-selection ⊕ element-selection is **loud** — the contract
    # under test (it must raise, never silently combine).
    # selection-unification-v2 P2-I: both hosts now return the single
    # ``MeshSelection`` terminal, so ``_compatible`` discriminates on
    # the ENGINE identity (node engine = NodeComposite vs element
    # engine = ElementComposite — different objects) rather than the
    # chain *type* arm.  Loudness is preserved; only which guard arm
    # (and message) fires changed by the unification, so the regex
    # accepts either the legacy "same chain type" or the
    # engine-identity message.
    _loud = r"same chain type|different engines"
    with pytest.raises(TypeError, match=_loud):
        nc | ec
    with pytest.raises(TypeError, match=_loud):
        nc & ec

    # cross-engine (two different FEMData node engines) — loud
    g2 = apeGmsh(model_name="s3b_other", verbose=False)
    g2.begin()
    try:
        g2.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                                  label="box")
        g2.physical.add_volume("box", name="Body")
        g2.mesh.sizing.set_global_size(0.5)
        g2.mesh.generation.generate(dim=3)
        fem2 = g2.mesh.queries.get_fem_data(dim=3)
    finally:
        g2.end()
    other = fem2.nodes.select(ids=[1])
    with pytest.raises(TypeError, match="different engines"):
        nc | other


# =====================================================================
# .result() -> the EXISTING NodeResult / GroupResult (PARITY)
# =====================================================================

def test_node_result_is_noderesult_frozen(cube_fem):
    fem = cube_fem
    sel = fem.nodes.select(pg="Body").result()
    # EXACT terminal type — what the broker node terminal yields.
    assert isinstance(sel, NodeResult)
    assert type(sel).__name__ == "NodeResult"
    # P3-R: ``.get`` is removed; freeze the v2 id set as an explicit
    # literal against the deterministic 3x3x3 lattice (P3-K proved
    # ``select`` ≡ the removed ``get``).
    assert _ids(sel) == list(range(1, 28))
    assert len(sel) == 27
    # coords carried correctly (object-dtype ids + (N,3) float64 coords)
    assert sel.coords.shape == (27, 3)
    assert sel.coords.dtype == np.float64
    assert sel.ids.dtype == object
    # same id set through the label-less target= path
    assert _ids(fem.nodes.select(target="Body").result()) == list(
        range(1, 28))


def test_element_result_is_groupresult_frozen(cube_fem):
    fem = cube_fem
    sel = fem.elements.select(pg="Body").result()
    assert isinstance(sel, GroupResult)
    assert type(sel).__name__ == "GroupResult"
    # P3-R: ``.get`` / ``.resolve`` are removed; freeze the v2 id set
    # + the ``.result().resolve()`` flat-shape literal against the
    # deterministic 8-hex8 fixture.
    assert _ids(sel) == list(range(1, 9))
    sel_ids, sel_conn = sel.resolve()
    assert sorted(int(x) for x in sel_ids) == list(range(1, 9))
    assert sel_ids.shape == (8,)
    assert sel_conn.shape == (8, 8)             # 8 hex8 cells, npe=8
    # same id set through the label-less target= path
    assert _ids(fem.elements.select(target="Body").result()) == list(
        range(1, 9))


def test_element_centroid_fails_loud_on_unknown_node(cube_fem):
    # The S3b correctness invariant: a centroid must never silently
    # substitute row 0 for a missing node id (the generic
    # _mesh_filters.element_centroids does that). We corrupt one
    # connectivity entry and assert the spatial path raises.
    fem = cube_fem
    grp = next(iter(fem.elements._groups.values()))
    saved = grp.connectivity.copy()
    # also clear any memoised centroid cache from earlier tests
    if hasattr(fem.elements, "_apegmsh_elem_centroid"):
        del fem.elements._apegmsh_elem_centroid
    try:
        bad = grp.connectivity.copy()
        bad[0, 0] = 10 ** 9          # node id that cannot exist
        grp.connectivity = bad
        with pytest.raises(KeyError, match="not in the FEM node set"):
            fem.elements.select(pg="Body").in_box((-9, -9, -9),
                                                  (9, 9, 9))
    finally:
        grp.connectivity = saved
        if hasattr(fem.elements, "_apegmsh_elem_centroid"):
            del fem.elements._apegmsh_elem_centroid


# =====================================================================
# selection-unification-v2 P3-R: the legacy ``.get()`` / ``.get_ids``
# / ``.get_coords`` / ``.resolve()`` broker surface is **removed**.
# The test that pinned its byte-behaviour
# (``test_legacy_get_and_resolve_unchanged_by_additive_select``) is
# deleted — there is no longer a ``.get()`` to pin, and the
# NodeResult/GroupResult shape + count + dtype coverage it carried is
# fully subsumed by the frozen ``test_node_result_is_noderesult_frozen``
# / ``test_element_result_is_groupresult_frozen`` above (the
# ``.select(...)`` path now yields those terminals directly).
# =====================================================================


# =====================================================================
# from_msh import-origin FEMData: centroids still work (in-memory wiring)
# =====================================================================

def test_select_works_on_import_origin_fem(tmp_path):
    """The ElementChain centroid path must NOT need a live Gmsh
    session — the sibling NodeComposite is wired in FEMData.__init__,
    which every construction path (incl. from_msh) funnels through.
    """
    import gmsh

    g = apeGmsh(model_name="s3b_imp", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                                 label="box")
        g.physical.add_volume("box", name="Body")
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(dim=3)
        msh = str(tmp_path / "m.msh")
        gmsh.write(msh)
    finally:
        g.end()

    fem = FEMData.from_msh(msh, dim=3)
    # P3-R: ``.get`` is removed; the invariant is that the v2
    # ``.select()`` terminal materialises on an import-origin FEMData
    # with NO live session — its no-arg id set is exactly the broker's
    # full id universe (``fem.nodes.ids`` / ``fem.elements.ids``).
    assert _ids(fem.nodes.select().result()) == sorted(
        int(x) for x in fem.nodes.ids)
    assert _ids(fem.elements.select().result()) == sorted(
        int(x) for x in fem.elements.ids)
    # centroid spatial verb works in-memory (no Gmsh)
    inside = fem.elements.select().in_box((-1, -1, -1), (2, 2, 2),
                                          inclusive=True)
    assert len(inside) == len(fem.elements)
