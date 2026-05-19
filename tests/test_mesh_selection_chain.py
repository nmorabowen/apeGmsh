"""S3d — MeshSelectionChain over the PRE-SNAPSHOT live mesh.

Sub-phase S3d of the selection-unification work
(``docs/plans/selection-unification.md`` §5/§6).  S3a–S3c landed the
``SelectionChain`` mixin, ``GeometryChain`` (entity family) and the
point-family ``NodeChain`` / ``ElementChain`` /
``ResultChain``.  S3d adds the daisy-chainable ``.select()`` to the
*mutable, session-bound* ``g.mesh_selection`` composite
(:class:`~apeGmsh.mesh.MeshSelectionSet.MeshSelectionSet`), which
reads the **live** ``gmsh.model.mesh`` — distinct from the immutable
``MeshSelectionStore`` snapshot on ``FEMData``.

What this locks:

* ``g.mesh_selection.select(...)`` returns a point-family
  ``MeshSelectionChain``, seeded from explicit ``ids=`` or the full
  live-mesh node / element universe — the same universe the eager
  ``add_nodes`` / ``add_elements`` start from.
* Daisy-chaining: ``select().in_box(...).on_plane(...)`` composes,
  every verb returning the same concrete chain type.
* **Parity** (the headline S3d invariant): a chained
  ``select().in_box(b).on_plane(p, n, tol=t)`` is *id-for-id* the same
  node / element set the eager
  ``add_nodes(in_box=b, on_plane=(...))`` /
  ``add_elements(in_box=b, on_plane=(...))`` produces.
* Point-family ``in_box`` is half-open ``[lo, hi)`` by default and
  closed ``[lo, hi]`` with ``inclusive=True`` (S2 parity), matching
  ``NodeChain`` / ``ElementChain`` and the eager API's ``inclusive=``.
* Set algebra ``| & - ^`` with insertion-order dedup; cross-level and
  cross-session combination is loud.
* ``.result()`` returns the **same shape** ``get_nodes`` /
  ``get_elements`` return today; ``.ids`` exposes the raw id list.
* The legacy eager API (``add`` / ``add_nodes`` / ``add_elements`` /
  ``filter_set`` / ``union`` / ``intersection`` / ``difference`` /
  ``_get_mesh_nodes`` / ``_snapshot``) is byte-behaviour unchanged by
  the additive ``.select()``.
* The element centroid path is fail-loud on a connectivity entry that
  references an unknown node id (never the silent row-0 substitution
  the generic ``_mesh_filters.element_centroids`` does).

No ``openseespy`` dependency (curated no-openseespy CI gate): pure
apeGmsh + gmsh + numpy.  A deterministic structured unit cube (3x3x3
node lattice -> 27 nodes, 8 hex8 cells, all coords in {0, 0.5, 1}) is
the fixture, so every boundary count is an exact integer.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh._kernel.chain import SelectionChain
# selection-unification-v2 P3-R (§6.2 / §6.3): the legacy
# ``MeshSelectionChain`` class is **deleted**; ``_LiveMeshEngine`` is
# **relocated** verbatim to ``apeGmsh.mesh._live_engine`` (plan §3 —
# pure typing-only leaf, no behaviour change).  The eager
# ``MeshSelectionSet.add_nodes`` / ``add_elements`` API is **removed**
# (SC-11 — the ``_mesh_filters`` silent-row-0 family); its
# resolution/spatial behaviour now lives on the v2 terminal
# ``g.mesh_selection.select(...)`` → ``MeshSelection``.  The retained
# eager surface (``add`` / ``filter_set`` / ``union`` /
# ``intersection`` / ``difference`` / ``get_tag`` / ``summary`` /
# ``_snapshot`` / ``from_physical``) is unchanged.  The legacy
# ``MeshSelectionChain`` structural-shape / __init_subclass__ tests
# are deleted (now covered by ``tests/test_selection_idiom.py``); the
# parity-vs-``add_nodes``/``add_elements`` halves are dropped and the
# v2 ``.select(...)`` behaviour is frozen as explicit literals against
# the deterministic 27-node / 8-hex8 live lattice.
from apeGmsh.mesh._live_engine import _LiveMeshEngine
from apeGmsh.mesh._mesh_selection import MeshSelection


# =====================================================================
# Fixture — structured unit cube, live session (NOT snapshotted)
# =====================================================================

@pytest.fixture
def live():
    """Yield a *live* session whose mesh is a 3x3x3 lattice.

    27 nodes at every {0,0.5,1}^3, 8 hex8 cells (centroids at the 8
    combinations of {0.25, 0.75}).  The session stays open (``g.end()``
    only in teardown) so ``g.mesh_selection`` reads the live
    ``gmsh.model.mesh`` — this is the pre-snapshot composite under
    test, *not* ``fem.mesh_selection``.
    """
    g = apeGmsh(model_name="s3d_cube", verbose=False)
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


def _sorted_ids(seq) -> list[int]:
    return sorted(int(x) for x in seq)


def _set_node_ids(ms, tag) -> list[int]:
    return _sorted_ids(ms.get_nodes(0, tag)["tags"])


# =====================================================================
# Class-shape invariant — the host hook returns the v2 terminal; the
# relocated _LiveMeshEngine still validates its level.
# (the legacy MeshSelectionChain structural-shape + __init_subclass__
# gate tests are deleted; covered by tests/test_selection_idiom.py)
# =====================================================================

def test_select_returns_point_family_meshselection(live):
    ms = live.mesh_selection
    sel = ms.select()
    assert isinstance(sel, MeshSelection)
    assert issubclass(MeshSelection, SelectionChain)
    assert sel.FAMILY == "point"


def test_relocated_engine_adapter_rejects_bad_level():
    # _LiveMeshEngine relocated to apeGmsh.mesh._live_engine (P3-R §3,
    # behaviour-verbatim); its level guard is unchanged.
    with pytest.raises(ValueError, match="level.*invalid"):
        _LiveMeshEngine(object(), "bogus", 0)


# =====================================================================
# .select() host hook — additive, seeds from the live universe
# =====================================================================

def test_node_select_seeds_full_live_node_universe(live):
    ms = live.mesh_selection
    sel = ms.select()                       # default level="node"
    assert isinstance(sel, MeshSelection)   # P2-I: was MeshSelectionChain
    assert sel.FAMILY == "point"
    all_ids, _ = ms._get_mesh_nodes()
    assert len(sel) == len(all_ids) == 27   # whole lattice
    # explicit ids= path
    assert ms.select(ids=[1, 2, 3]).ids == [1, 2, 3]


def test_element_select_seeds_full_live_element_universe(live):
    ms = live.mesh_selection
    sel = ms.select(level="element", dim=3)
    assert isinstance(sel, MeshSelection)   # P2-I: was MeshSelectionChain
    assert sel.FAMILY == "point"
    eids, _ = ms._get_mesh_elements(3)
    assert len(sel) == len(eids) == 8       # 8 hex8 cells
    e0 = int(eids[0])
    assert ms.select(level="element", dim=3, ids=[e0]).ids == [e0]


def test_select_rejects_bad_level(live):
    with pytest.raises(ValueError, match="must be 'node' or 'element'"):
        live.mesh_selection.select(level="bogus")


def test_select_reuses_existing_live_mesh_path(live, monkeypatch):
    """The chain must reach the live mesh ONLY through the existing
    ``MeshSelectionSet._get_mesh_nodes`` (the exact method
    ``add_nodes`` uses) — proving reuse, not a second fetch path."""
    ms = live.mesh_selection
    from apeGmsh.mesh.MeshSelectionSet import MeshSelectionSet
    seen = {"n": 0}
    real = MeshSelectionSet._get_mesh_nodes

    def _spy(self):
        seen["n"] += 1
        return real(self)

    monkeypatch.setattr(MeshSelectionSet, "_get_mesh_nodes", _spy)
    out = ms.select().in_box((-1, -1, -1), (2, 2, 2)).result()
    assert seen["n"] >= 1                    # delegated, not re-impl'd
    assert out["tags"].dtype == object
    assert out["coords"].shape[1] == 3


# =====================================================================
# Daisy-chaining — every verb returns MeshSelectionChain
# =====================================================================

def test_node_chain_daisychains_each_verb(live):
    ms = live.mesh_selection
    s1 = ms.select()
    s2 = s1.in_box((-1, -1, -1), (2, 2, 2))
    s3 = s2.on_plane((0, 0, 0), (0, 0, 1), tol=1e-9)
    for s in (s1, s2, s3):
        assert isinstance(s, MeshSelection)  # P2-I: was MeshSelectionChain
    chained = (ms.select()
                 .in_box((-1, -1, -1), (2, 2, 2))
                 .on_plane((0, 0, 0), (0, 0, 1), tol=1e-9))
    assert len(chained) == 9                 # z=0 lattice plane


def test_element_chain_daisychains_each_verb(live):
    ms = live.mesh_selection
    s1 = ms.select(level="element", dim=3)
    s2 = s1.in_box((-1, -1, -1), (2, 2, 2))
    s3 = s2.on_plane((0, 0, 0.25), (0, 0, 1), tol=0.1)
    for s in (s1, s2, s3):
        assert isinstance(s, MeshSelection)  # P2-I: was MeshSelectionChain
    # 8 cells; centroids z in {0.25, 0.75}; z=0.25 plane keeps 4
    chained = (ms.select(level="element", dim=3)
                 .in_box((-1, -1, -1), (2, 2, 2))
                 .on_plane((0, 0, 0.25), (0, 0, 1), tol=0.1))
    assert len(chained) == 4


# =====================================================================
# selection-unification-v2 P3-R: the eager ``add_nodes`` /
# ``add_elements`` API is **removed** (SC-11); its spatial/resolution
# behaviour is now the v2 ``g.mesh_selection.select(...)`` terminal.
# The parity-vs-``add_*`` halves are dropped; the ``.select(...)``
# result is frozen as an explicit literal against the deterministic
# 27-node / 8-hex8 live lattice (P3-K proved ``select`` ≡ the removed
# eager spatial path).
# =====================================================================

def test_node_select_in_box_on_plane_frozen(live):
    """``select().in_box(b).on_plane(p,n,tol)`` resolves the frozen
    node set (was: parity vs the removed ``add_nodes(in_box=,
    on_plane=)``)."""
    ms = live.mesh_selection

    chained = (ms.select()
                 .in_box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
                 .on_plane((0, 0, 0), (0, 0, 1), tol=1e-9))
    assert _sorted_ids(chained.ids) == [2, 12, 17, 25]
    # terminal carries the same ids too (and is the get_nodes shape)
    assert _sorted_ids(chained.result()["tags"]) == [2, 12, 17, 25]
    # in_box-only (half-open default): drops the upper {x|y|z=1} shell.
    cb = ms.select().in_box((0, 0, 0), (1, 1, 1))
    assert _sorted_ids(cb.ids) == [2, 9, 12, 17, 21, 23, 25, 27]


def test_element_select_in_box_frozen(live):
    """``select(level='element').in_box(b)`` resolves the frozen
    centroid-based element set (was: parity vs the removed
    ``add_elements(in_box=b)``).  The point-family ``in_box`` operates
    on element centroids — centroids are {0.25,0.75}^3, so the
    half-open ``[.,.75)`` box keeps exactly the 4 cells with centroid
    z==0.25; inclusive=True closes it to all 8.
    """
    ms = live.mesh_selection
    box = (-1.0, -1.0, -1.0, 2.0, 2.0, 0.75)

    chained = (ms.select(level="element", dim=3)
                 .in_box((box[0], box[1], box[2]),
                         (box[3], box[4], box[5])))
    assert _sorted_ids(chained.ids) == [1, 2, 3, 4]
    # terminal == get_elements shape, same element ids
    res = chained.result()
    assert _sorted_ids(res["element_ids"]) == [1, 2, 3, 4]
    assert res["element_ids"].dtype == object
    assert res["connectivity"].dtype == object
    # inclusive=True closes the box -> all 8 cells.
    cb = (ms.select(level="element", dim=3)
            .in_box((box[0], box[1], box[2]),
                    (box[3], box[4], box[5]), inclusive=True))
    assert _sorted_ids(cb.ids) == [1, 2, 3, 4, 5, 6, 7, 8]
    assert len(cb) == 8


def test_predicate_chain_where_frozen(live):
    """``select().where(pred)`` resolves the frozen node set (was:
    parity vs the removed ``add_nodes(predicate=...)``)."""
    ms = live.mesh_selection
    chained = ms.select().where(lambda xyz: xyz[0] < 0.5)
    # lattice x in {0, 0.5, 1}; x < 0.5 keeps only the x==0 face = 9
    assert _sorted_ids(chained.ids) == [1, 2, 3, 4, 9, 10, 11, 12, 21]
    assert len(chained) == 9


# =====================================================================
# Half-open default + inclusive=True closed (frozen — was S2 parity
# vs the removed eager add_nodes/add_elements inclusive=)
# =====================================================================

def test_in_box_half_open_default_and_inclusive_node(live):
    ms = live.mesh_selection
    alln = ms.select()
    half = alln.in_box((0, 0, 0), (1, 1, 1))
    closed = alln.in_box((0, 0, 0), (1, 1, 1), inclusive=True)
    # half-open [0,1)^3 drops the entire upper shell -> the 8 nodes
    # with every coord in {0, 0.5}; inclusive=True restores all 27.
    assert len(half) == 8
    assert len(closed) == 27
    # frozen literal (was: parity vs the removed eager add_nodes).
    assert _sorted_ids(half.ids) == [2, 9, 12, 17, 21, 23, 25, 27]
    assert _sorted_ids(closed.ids) == list(range(1, 28))


def test_in_box_half_open_default_and_inclusive_element(live):
    ms = live.mesh_selection
    alle = ms.select(level="element", dim=3)
    # centroids are {0.25,0.75}^3; upper bound exactly 0.75 excludes
    # centroids on 0.75 -> only the (0.25,0.25,0.25) cell. inclusive
    # restores all 8.
    he = alle.in_box((0.0, 0.0, 0.0), (0.75, 0.75, 0.75))
    ce = alle.in_box((0.0, 0.0, 0.0), (0.75, 0.75, 0.75),
                     inclusive=True)
    assert len(he) == 1
    assert len(ce) == 8
    # frozen literal (was: parity vs the removed eager add_elements).
    assert _sorted_ids(he.ids) == [1]
    assert _sorted_ids(ce.ids) == [1, 2, 3, 4, 5, 6, 7, 8]


def test_sphere_plane_nearest_where(live):
    ms = live.mesh_selection
    alle = ms.select(level="element", dim=3)

    one = alle.in_sphere((0.25, 0.25, 0.25), 0.01)
    assert len(one) == 1

    near1 = alle.nearest_to((0.25, 0.25, 0.25), count=1)
    assert len(near1) == 1
    assert set(near1) == set(one)
    assert len(alle.nearest_to((0.5, 0.5, 0.5), count=3)) == 3

    w = alle.where(lambda xyz: xyz[0] < 0.5)
    assert len(w) == 4                        # x=0.25 centroid layer

    nface = ms.select().on_plane((0, 0, 0), (0, 1, 0), tol=1e-9)
    assert len(nface) == 9                    # y=0 lattice face

    # point-family input validation is loud
    with pytest.raises(ValueError, match="radius must be non-negative"):
        alle.in_sphere((0, 0, 0), -1.0)
    with pytest.raises(ValueError, match="tolerance must be non-negative"):
        ms.select().on_plane((0, 0, 0), (0, 0, 1), tol=-1.0)
    with pytest.raises(ValueError, match="normal vector has zero length"):
        alle.on_plane((0, 0, 0), (0, 0, 0), tol=1e-6)


# =====================================================================
# Set algebra — insertion-order dedup; cross-* is loud
# =====================================================================

def test_set_algebra_node_level(live):
    ms = live.mesh_selection
    base, _ = ms._get_mesh_nodes()
    base = [int(n) for n in base]
    a = ms.select(ids=base[:5])
    b = ms.select(ids=base[3:8])
    assert len(a | b) == 8                     # 5 + 5 - 2 dup
    assert len(a & b) == 2
    assert len(a - b) == 3
    assert len(a ^ b) == 6
    assert len(a | a) == 5                     # idempotent (one law)
    # compare on the ATOMS (``_items``); P2-I
    # ``MeshSelection.__iter__`` yields ``(id, payload)`` pairs
    # (ndarray payload → ambiguous tuple ``==``), but set-algebra is
    # defined on ``_items`` and is unaffected by the pair-view.
    assert (a.union(b))._items == (a | b)._items
    assert (a.difference(b))._items == (a - b)._items
    for s in (a | b, a & b, a - b, a ^ b):
        assert isinstance(s, MeshSelection)  # P2-I: was MeshSelectionChain


def test_set_algebra_element_level_same_engine(live):
    ms = live.mesh_selection
    eids, _ = ms._get_mesh_elements(3)
    eids = [int(e) for e in eids]
    ea = ms.select(level="element", dim=3, ids=eids[:5])
    eb = ms.select(level="element", dim=3, ids=eids[3:])
    assert len(ea | eb) == 8
    assert isinstance(ea & eb, MeshSelection)  # P2-I: was MeshSelectionChain


def test_cross_level_and_cross_dim_set_algebra_is_loud(live):
    ms = live.mesh_selection
    n = ms.select(ids=[1])
    e = ms.select(level="element", dim=3,
                  ids=[int(ms._get_mesh_elements(3)[0][0])])
    # node vs element -> different engine adapter -> loud
    with pytest.raises(TypeError, match="different engines"):
        n | e
    # element dim=3 vs dim=2 -> different adapter -> loud
    e2 = ms.select(level="element", dim=2)
    with pytest.raises(TypeError, match="different engines"):
        e | e2


def test_cross_session_set_algebra_is_loud():
    """Two different sessions have different ``MeshSelectionSet``
    objects -> different engine adapters -> cross-session set-algebra
    is loud.

    gmsh's context is process-global (``gmsh.initialize`` /
    ``gmsh.finalize``), so two sessions cannot be live at once; this
    test runs them **sequentially**.  The base
    :meth:`SelectionChain._compatible` rejects on engine *identity*
    *before* any coordinate fetch, and ``select(ids=[...])`` seeds from
    the explicit list without touching the live mesh — so the loud
    rejection is provable without either session still being open.
    """
    g1 = apeGmsh(model_name="s3d_sess_a", verbose=False)
    g1.begin()
    try:
        g1.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                                  label="box")
        g1.physical.add_volume("box", name="Body")
        g1.mesh.sizing.set_global_size(0.5)
        g1.mesh.generation.generate(dim=3)
        a = g1.mesh_selection.select(ids=[1])     # no live fetch (ids=)
    finally:
        g1.end()

    g2 = apeGmsh(model_name="s3d_sess_b", verbose=False)
    g2.begin()
    try:
        g2.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                                  label="box")
        g2.physical.add_volume("box", name="Body")
        g2.mesh.sizing.set_global_size(0.5)
        g2.mesh.generation.generate(dim=3)
        other = g2.mesh_selection.select(ids=[1])  # different adapter
        with pytest.raises(TypeError, match="different engines"):
            a | other
    finally:
        g2.end()


# =====================================================================
# Fail-loud — element centroid never silently substitutes row 0
# =====================================================================

def test_element_centroid_fails_loud_on_unknown_node(live, monkeypatch):
    """A centroid must never silently substitute row 0 for a missing
    node id (the generic ``_mesh_filters.element_centroids`` does
    that).  We corrupt the live element connectivity and assert the
    spatial path raises."""
    ms = live.mesh_selection
    real = ms._get_mesh_elements

    def _bad(dim):
        eids, conn = real(dim)
        conn = conn.copy()
        conn[0, 0] = 10 ** 9                 # node id that cannot exist
        return eids, conn

    monkeypatch.setattr(ms, "_get_mesh_elements", _bad)
    with pytest.raises(KeyError, match="not in the live mesh node set"):
        (ms.select(level="element", dim=3)
           .in_box((-9, -9, -9), (9, 9, 9)))


# =====================================================================
# RETAINED eager API byte-behaviour unchanged (P3-R: the
# add_nodes/add_elements portions are dropped — that API is removed;
# the retained add/filter_set/union/intersection/difference/get_tag/
# summary/_snapshot surface is still exercised verbatim)
# =====================================================================

def test_retained_eager_api_unchanged_by_additive_select(live):
    ms = live.mesh_selection

    # explicit add(), filter_set(), union/intersection/difference,
    # get_tag, summary still behave (all RETAINED surfaces).  ``nt`` is
    # now an explicit add() set (was an add_nodes spatial seed —
    # add_nodes is removed) so union/intersection still have a 2nd set.
    at = ms.add(0, [1, 2, 3, 4], name="explicit")
    assert _set_node_ids(ms, at) == [1, 2, 3, 4]
    nt = ms.add(0, [3, 4, 5, 6], name="ln2")
    ft = ms.filter_set(0, at, in_box=(-1, -1, -1, 2, 2, 2),
                       inclusive=True, name="filt")
    assert _set_node_ids(ms, ft) == [1, 2, 3, 4]
    ut = ms.union(0, at, nt, name="u")
    assert set(_set_node_ids(ms, ut)) == {1, 2, 3, 4, 5, 6}
    it = ms.intersection(0, at, nt, name="i")
    assert set(_set_node_ids(ms, it)) == {3, 4}
    dt = ms.difference(0, at, at, name="d")
    assert _set_node_ids(ms, dt) == []
    assert ms.get_tag(0, "explicit") == at
    assert not ms.summary().empty

    # _snapshot still yields an immutable MeshSelectionStore mirror
    from apeGmsh.mesh.MeshSelectionSet import MeshSelectionStore
    snap = ms._snapshot()
    assert isinstance(snap, MeshSelectionStore)
    assert _sorted_ids(snap.get_nodes(0, at)["tags"]) == [1, 2, 3, 4]

    # the chain engine cache lives in its OWN private attr — it does
    # not leak into _sets (persistence explicitly NOT implemented)
    ms.select(ids=[1]).in_box((-9, -9, -9), (9, 9, 9), inclusive=True)
    assert all(isinstance(k, tuple) and len(k) == 2
               for k in ms._sets)             # only (dim, tag) keys
    assert "_apegmsh_mesh_selection_chain_engines" not in ms._sets


def test_select_does_not_register_a_set(live):
    """Persistence is out of S3d scope: ``.select()`` (and its full
    daisy-chain + terminal) must not add anything to ``_sets`` or
    allocate a tag."""
    ms = live.mesh_selection
    sets_before = dict(ms._sets)
    next_tag_before = dict(ms._next_tag)
    (ms.select()
       .in_box((0, 0, 0), (1, 1, 1))
       .on_plane((0, 0, 0), (0, 0, 1), tol=1e-9)
       .result())
    (ms.select(level="element", dim=3)
       .in_box((-9, -9, -9), (9, 9, 9))
       .result())
    assert ms._sets == sets_before            # nothing registered
    assert ms._next_tag == next_tag_before    # no tag allocated
