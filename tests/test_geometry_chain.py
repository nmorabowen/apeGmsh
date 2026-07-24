"""S3a (P3-R) — the entity-family ``g.model.select(...)`` terminal.

Phase S3a of the selection-unification work
(``docs/plans/selection-unification.md`` §5/§6, ratified decision R3),
carried through selection-unification-v2 P3-R
(``docs/plans/selection-unification-v2.md`` §6.2 / §6.3): the
fluent, daisy-chainable entity-family terminal ``EntitySelection``
(the legacy ``GeometryChain`` is deleted) is reached via the
``g.model.select(...)`` host hook and materialises the RETAINED legacy
``core/_selection.Selection`` terminal (R-v2-8).

What this locks:

* ``g.model.select(...)`` returns an ``EntitySelection`` and delegates
  name resolution to the contract-locked geometry resolver
  (``resolve_to_dimtags``) — label -> PG -> part tiering, not
  re-implemented here.
* The chain daisy-chains: ``.select(...).in_box(...).on_plane(...)``
  composes, every verb returning a new ``EntitySelection``.
* Entity-family spatial semantics:
    - ``in_box`` uses ``gmsh.model.getEntitiesInBoundingBox`` (BRep
      bbox-INTERSECT — Gmsh's own entity query), NOT coordinate
      half-open;
    - ``inclusive=`` (any keyword) raises ``TypeError`` — the
      half-open knob is inexpressible for the BRep query (R3, fail
      loud, never silently ignored);
    - ``on_plane`` reuses the legacy ``Plane`` / ``_bb_corners``
      8-corner "on" test;
    - ``in_sphere`` / ``nearest_to`` / ``where`` operate on the entity
      bounding-box centre.
* Set algebra ``| & - ^`` with insertion-order dedup; cross-type
  combination is loud.
* ``.result()`` returns a **legacy** ``core/_selection.Selection``
  instance (the class is RETAINED — R-v2-8), so ``.to_label()`` /
  ``.to_physical()`` / ``.tags()`` keep working through the unchanged
  terminal.
* The geometric-predicate engine is now reached through the v2
  ``g.model.select(...).crossing_plane(spec, mode=)`` verb (the legacy
  ``g.model.queries.select`` / ``g.model.selection.select_*`` entry
  points are removed by P3-R); the two RETAINED ``Selection`` classes
  (``core/_selection.Selection`` list-subclass vs ``viz`` frozen) stay
  irreconcilable and distinct (FP-2 / R-v2-8).

No ``openseespy`` dependency (curated no-openseespy CI gate): pure
apeGmsh + gmsh + numpy.  A tiny unit cube with a 6-face physical group
is the deterministic fixture.
"""
from __future__ import annotations

import pytest

from apeGmsh import apeGmsh
from apeGmsh._kernel.chain import SelectionChain
from apeGmsh.core._selection import EntitySelection, Selection
from apeGmsh.mesh._mesh_selection import MeshSelection
# selection-unification-v2 P3-R (§6.2 / §6.3): the legacy
# ``GeometryChain`` / ``NodeChain`` classes are **deleted** and the
# removed ``g.model.queries.select`` / ``g.model.selection.select_*``
# legacy entry points no longer exist.  ``g.model.select(...)`` returns
# the v2 terminal ``EntitySelection``.  The GeometryChain
# structural-shape / __init_subclass__ tests are deleted (now covered
# by ``tests/test_selection_idiom.py`` over the two v2 terminals); the
# behaviour-bearing tests below keep their full coverage on the
# ``g.model.select(...)`` path (the legacy ``queries.select``
# geometric-predicate test is rewritten onto the v2
# ``crossing_plane`` verb with frozen ``(dim, tag)`` literals — P2-G
# proved the equivalence, ``tests/test_p2g_parity.py``; the removed
# ``g.model.selection.select_*`` test is deleted, no v2 successor
# [SC-12 precedent / §6.3 disposition 3]).


# =====================================================================
# Fixture — unit cube; "box" volume label + "Faces" 6-face PG
# =====================================================================

@pytest.fixture
def cube(g):
    """A 1x1x1 box: label ``box`` (dim-3) + PG ``Faces`` (all 6 dim-2).

    Faces are tagged 1..6 by OCC; we resolve them by PG name through
    the chain so the test never hard-codes raw tags (apeGmsh is
    verbose-by-name).
    """
    g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, label="box")
    g.model.sync()
    faces = g.model.queries.boundary("box", dim=3, oriented=False)
    g.physical.add_surface([int(t) for _d, t in faces], name="Faces")
    return g


# =====================================================================
# Class-shape invariant — the host hook returns the v2 terminal
# (the legacy GeometryChain structural-shape + __init_subclass__ gate
# tests are deleted; covered by tests/test_selection_idiom.py)
# =====================================================================

def test_select_returns_entityselection_entity_family(cube):
    g = cube
    sel = g.model.select("Faces")
    assert isinstance(sel, EntitySelection)
    assert issubclass(EntitySelection, SelectionChain)
    assert sel.FAMILY == "entity"
    # legacy terminal is the RETAINED list subclass (R-v2-8)
    assert issubclass(Selection, list)
    assert EntitySelection is not Selection


# =====================================================================
# .select() host hook — delegates to the locked resolver
# =====================================================================

def test_select_returns_entityselection_seeded_by_tier_resolution(cube):
    g = cube
    # string -> label tier (Tier 1): "box" is a dim-3 volume label.
    vol = g.model.select("box")
    assert isinstance(vol, EntitySelection)    # P2-I: was GeometryChain
    assert vol.FAMILY == "entity"
    assert sorted(tuple(vol)) == [(3, 1)]

    # string -> PG tier (Tier 2): "Faces" is the 6-face physical group.
    faces = g.model.select("Faces")
    assert isinstance(faces, EntitySelection)  # P2-I: was GeometryChain
    assert len(faces) == 6
    assert {d for d, _ in faces} == {2}


def test_select_delegates_to_resolve_to_dimtags(monkeypatch):
    # Prove the host hook calls the EXISTING geometry resolver verbatim
    # (apeGmsh/core/_helpers.py resolve_to_dimtags) rather than
    # re-implementing tier logic. We spy on it and assert it is hit
    # with the forwarded args.
    import apeGmsh.core._helpers as _h

    seen = {}
    real = _h.resolve_to_dimtags

    def _spy(ref, *, default_dim, session):
        seen["ref"] = ref
        seen["default_dim"] = default_dim
        return real(ref, default_dim=default_dim, session=session)

    g = apeGmsh(model_name="gc_delegate", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                                 label="box")
        g.model.sync()
        monkeypatch.setattr(_h, "resolve_to_dimtags", _spy)
        # dim=3 matches the box volume (a mismatched dim now raises, so
        # use the matching dim to exercise the forwarding delegation).
        ch = g.model.select("box", dim=3)
        assert isinstance(ch, EntitySelection)  # P2-I: was GeometryChain
        assert seen["ref"] == "box"
        assert seen["default_dim"] == 3     # dim= forwarded as default_dim
    finally:
        g.end()


def test_select_accepts_dimtag_and_int_and_list(cube):
    g = cube
    # (dim, tag) passthrough
    a = g.model.select((3, 1))
    assert sorted(tuple(a)) == [(3, 1)]
    # bare int with dim= as resolve_to_dimtags default_dim
    b = g.model.select(1, dim=3)
    assert (3, 1) in tuple(b)
    # list of mixed refs resolved independently + concatenated
    c = g.model.select([(3, 1), "Faces"])
    assert (3, 1) in tuple(c)
    assert sum(1 for d, _ in c if d == 2) == 6


def test_select_requires_target_or_dim(cube):
    with pytest.raises(ValueError, match="pass a target.*or a dim"):
        cube.model.select()


# =====================================================================
# Daisy-chaining + entity-family spatial semantics
# =====================================================================

def test_chain_daisychains_and_each_verb_returns_geometrychain(cube):
    g = cube
    step1 = g.model.select("Faces")
    step2 = step1.in_box((-1, -1, -1), (2, 2, 2))
    step3 = step2.on_plane((0, 0, 0), (0, 0, 1), tol=1e-6)
    for s in (step1, step2, step3):
        assert isinstance(s, EntitySelection)  # P2-I: was GeometryChain
    # the full fluent one-liner composes to exactly the z=0 face
    chained = (g.model.select("Faces")
                 .in_box((-1, -1, -1), (2, 2, 2))
                 .on_plane((0, 0, 0), (0, 0, 1), tol=1e-6))
    assert len(chained) == 1


def test_in_box_uses_gmsh_brep_query_not_coordinate_halfopen(cube):
    g = cube
    faces = g.model.select("Faces")

    # Entity semantics, NOT the point-family coordinate box. The gmsh
    # BRep query is bbox-CONTAINMENT (the whole entity bbox must lie
    # inside the query box, expanded by Geometry.Tolerance ~1e-8) —
    # it is neither half-open nor bbox-intersect.

    # (a) Whole-box query enclosing every face -> all 6 contained.
    allf = faces.in_box((-0.1, -0.1, -0.1), (1.1, 1.1, 1.1))
    assert len(allf) == 6

    # (b) CONTAINMENT, not intersect: a box covering x in [-1, 0.5]
    # returns ONLY the x=0 face (bbox x-extent [0,0] fully inside).
    # The four side faces and the x=1 face all have x-extent reaching
    # 1.0, so they are NOT contained (an intersect query would keep 5).
    sub = faces.in_box((-1, -1, -1), (0.5, 2, 2))
    x0 = faces.on_plane((0, 0, 0), (1, 0, 0), tol=1e-6)   # x=0 face
    assert len(sub) == 1
    assert set(tuple(sub)) == set(tuple(x0))

    # (c) Strict-on-bound: a query whose UPPER z bound is EXACTLY 0.0
    # does NOT contain the z=0 face (its bbox z=0 sits on the raw
    # bound; the tolerance expansion is ~1e-8, far below 0). This is
    # the gmsh BRep behaviour, deliberately distinct from the
    # point-family half-open rule (which is about coordinates, not
    # entity bounding boxes).
    on_bound = faces.in_box((-0.1, -0.1, -0.1), (1.1, 1.1, 0.0))
    assert len(on_bound) == 0
    # Nudge the bound out by more than the gmsh tolerance and the z=0
    # face becomes contained -> proves it is the BRep query, closed
    # under containment, that drives membership.
    nudged = faces.in_box((-0.1, -0.1, -0.1), (1.1, 1.1, 1e-6))
    z0 = faces.on_plane((0, 0, 0), (0, 0, 1), tol=1e-6)
    assert set(tuple(nudged)) == set(tuple(z0)) and len(z0) == 1

    # The result still refines the chain (intersected with current
    # atoms) and stays the entity terminal type.
    assert isinstance(sub, EntitySelection)    # P2-I: was GeometryChain
    assert set(tuple(sub)).issubset(set(tuple(faces)))


def test_in_box_inclusive_keyword_raises_typeerror(cube):
    g = cube
    faces = g.model.select("Faces")
    # R3: the half-open / inclusive= knob is inexpressible for the BRep
    # query and MUST fail loud — both True and False, and any keyword.
    with pytest.raises(TypeError, match="inclusive"):
        faces.in_box((0, 0, 0), (1, 1, 1), inclusive=True)
    with pytest.raises(TypeError, match="inclusive"):
        faces.in_box((0, 0, 0), (1, 1, 1), inclusive=False)
    with pytest.raises(TypeError):
        faces.in_box((0, 0, 0), (1, 1, 1), something=1)
    # the positional form (no keyword) still works
    assert isinstance(
        faces.in_box((-1, -1, -1), (2, 2, 2)), EntitySelection
    )                                          # P2-I: was GeometryChain


def test_on_plane_in_sphere_nearest_where_entity_bbox_semantics(cube):
    g = cube
    faces = g.model.select("Faces")

    # on_plane: legacy 8-corner "on" test -> exactly the z=0 face.
    bottom = faces.on_plane((0, 0, 0), (0, 0, 1), tol=1e-6)
    assert len(bottom) == 1

    # in_sphere: bbox-centre within radius. Tiny ball at the bottom
    # face's bbox centre (0.5, 0.5, 0.0) catches only that face.
    sph = faces.in_sphere((0.5, 0.5, 0.0), 0.05)
    assert set(tuple(sph)) == set(tuple(bottom))

    # nearest_to: order by bbox-centre distance; count caps the result.
    near = faces.nearest_to((0.5, 0.5, 0.0), count=1)
    assert len(near) == 1
    assert set(tuple(near)) == set(tuple(bottom))
    assert len(faces.nearest_to((0.5, 0.5, 0.5), count=3)) == 3

    # where: predicate on the bbox-centre row -> faces with centre x<0.5
    # is exactly the x=0 face (centre (0,0.5,0.5)).
    w = faces.where(lambda xyz: xyz[0] < 0.5)
    x0 = faces.on_plane((0, 0, 0), (1, 0, 0), tol=1e-6)
    assert set(tuple(w)) == set(tuple(x0))
    assert len(w) == 1

    # entity-family input validation is loud.
    with pytest.raises(ValueError, match="radius must be non-negative"):
        faces.in_sphere((0, 0, 0), -1.0)
    with pytest.raises(ValueError, match="tolerance must be non-negative"):
        faces.on_plane((0, 0, 0), (0, 0, 1), tol=-1.0)
    with pytest.raises(ValueError, match="normal vector has zero length"):
        faces.on_plane((0, 0, 0), (0, 0, 0), tol=1e-6)


# =====================================================================
# Set algebra — insertion-order dedup; cross-type is loud
# =====================================================================

def test_set_algebra_union_intersect_difference_symmetric(cube):
    g = cube
    bottom = g.model.select("Faces").on_plane((0, 0, 0), (0, 0, 1),
                                               tol=1e-6)
    left = g.model.select("Faces").on_plane((0, 0, 0), (1, 0, 0),
                                             tol=1e-6)
    assert len(bottom) == 1 and len(left) == 1
    assert tuple(bottom) != tuple(left)        # distinct faces

    assert len(bottom | left) == 2             # union
    assert len(bottom & left) == 0             # disjoint
    assert len(bottom - left) == 1             # difference
    assert len(bottom ^ left) == 2             # symmetric difference
    # idempotent union (insertion-order dedup, the one law)
    assert len(bottom | bottom) == 1
    # named aliases match the operators
    assert tuple(bottom.union(left)) == tuple(bottom | left)
    assert tuple(bottom.difference(left)) == tuple(bottom - left)
    # every set-algebra result is itself the entity terminal (chainable)
    for s in (bottom | left, bottom & left, bottom - left, bottom ^ left):
        assert isinstance(s, EntitySelection)  # P2-I: was GeometryChain


def test_cross_type_set_algebra_is_loud(cube):
    gc = cube.model.select("Faces")
    nc = MeshSelection((), _engine=None)       # P3-R: was NodeChain
    with pytest.raises(TypeError, match="same chain type"):
        gc | nc
    with pytest.raises(TypeError, match="same chain type"):
        gc & nc


# =====================================================================
# .result() -> the LEGACY Selection terminal (unchanged behaviour)
# =====================================================================

def test_result_returns_legacy_selection_instance(cube):
    g = cube
    sel = g.model.select("Faces").on_plane((0, 0, 0), (0, 0, 1),
                                            tol=1e-6).result()
    # EXACT legacy type — not a subclass-of-list lookalike.
    assert isinstance(sel, Selection)
    assert type(sel).__name__ == "Selection"
    assert isinstance(sel, list)               # legacy is a list subclass
    assert list(sel) == [(2, 5)] or len(sel) == 1


def test_result_terminal_to_physical_and_tags_still_work(cube):
    g = cube
    sel = (g.model.select("Faces")
             .on_plane((0, 0, 0), (0, 0, 1), tol=1e-6)
             .result())
    # legacy .tags() method (NOT the viz .tags property) — proves it is
    # the core/_selection.Selection terminal, byte-unchanged.
    tags = sel.tags()
    assert all(isinstance(t, int) for t in tags) and len(tags) == 1
    # legacy .to_physical registers via the unchanged terminal path.
    sel.to_physical("Base")
    assert sorted(int(x) for x in g.physical.entities("Base", dim=2)) \
        == sorted(tags)
    # legacy chain-on-terminal (.select(on=)) still works off .result()
    again = sel.select(on={"z": 0})
    assert type(again).__name__ == "Selection"
    assert len(again) == 1


# =====================================================================
# selection-unification-v2 P3-R: the legacy entry points
# ``g.model.queries.select`` and ``g.model.selection.select_*`` are
# **removed**.
#
# * ``test_legacy_queries_select_unchanged`` is deleted — the
#   geometric-predicate engine is now reached via the v2
#   ``g.model.select(...).crossing_plane(spec, mode=)`` verb, whose
#   behaviour (every spec / mode / set-algebra on the RETAINED
#   ``Selection`` terminal) is exhaustively pinned with frozen
#   ``(dim, tag)`` literals in the dedicated ``tests/test_p2g_parity.py``
#   (P2-G proved ``crossing_plane`` ≡ the now-removed ``queries.select``
#   engine).  The legacy ``queries.select("box", dim=2, ...)``
#   boundary-extraction-then-predicate call is a removed surface with
#   no byte-equivalent v2 call (v2 seeds from explicitly-resolved
#   entities) — keeping a half-rewritten copy here would duplicate
#   ``test_p2g_parity.py`` worse.
# * ``test_legacy_selection_composite_select_star_unchanged`` is
#   deleted — ``g.model.selection`` (the ``SelectionComposite``) is
#   removed with **no v2 successor** for its ``select_volumes`` /
#   ``select_surfaces`` filter grammar (the SC-12 precedent / §6.3
#   M-NOTE-G7-cascade disposition 3; a P4-documented capability gap).
#
# The RETAINED two-``Selection``-classes distinctness invariant
# (behaviour-bearing — both classes survive P3-R, R-v2-8) is kept and
# rewritten off the deleted ``GeometryChain`` onto the v2
# ``EntitySelection`` terminal.
# =====================================================================

def test_two_retained_selection_classes_remain_distinct():
    # R-v2-8 / FP-2: the core/_selection.Selection (list subclass,
    # .tags() method) and the viz/Selection.Selection (frozen
    # __slots__, .tags property) are RETAINED through P3-R and stay
    # irreconcilable.  EntitySelection (the v2 entity terminal) is a
    # third, independent type — it is neither of them.
    from apeGmsh.core._selection import Selection as CoreSel
    from apeGmsh.viz.Selection import Selection as VizSel

    assert CoreSel is not VizSel
    assert issubclass(CoreSel, list)            # mutable list subclass
    assert not issubclass(VizSel, list)         # frozen __slots__
    assert callable(CoreSel.tags)               # .tags() METHOD
    assert isinstance(VizSel.tags, property)    # .tags PROPERTY
    assert EntitySelection is not CoreSel
    assert EntitySelection is not VizSel
