"""
Tests for the v2 ``g.model.select(...).crossing_plane(...)`` idiom and
the RETAINED ``Selection`` / ``Plane`` / ``Line`` / ``_parse_primitive``
surface.

selection-unification-v2 P3-R (``docs/plans/selection-unification-v2.md``
§5-C / §6.2): the legacy ``g.model.queries.select(...)`` /
``g.model.queries.line(...)`` entry points are **removed**.  The
geometric-predicate engine they fronted is the RETAINED
``EntitySelection.crossing_plane(spec, *, mode=)`` verb (P2-G proved
the equivalence — exhaustively pinned with frozen literals in
``tests/test_p2g_parity.py``); the ``EntitySelection`` → legacy
``Selection`` terminal (``.to_label`` / ``.to_physical`` / ``.tags()``
/ ``.select(on=)`` / repr / set-algebra) is covered by
``tests/test_geometry_chain.py``.  This file is **rewritten** (plan
§5-C) onto the v2 idiom: every ``queries.select(SEED, MODE=SPEC)``
becomes ``g.model.select(SEED).crossing_plane(SPEC, mode=MODE)``;
``queries.line(p1,p2)`` becomes the RETAINED ``Line.through(p1,p2)``;
``queries.plane(...)`` is RETAINED and unchanged.  The retained
``_parse_primitive`` / ``Plane`` / ``Line`` / ``Selection``-terminal /
``boundary_curves`` / ``partition_by`` classes are unchanged.

Covers:
  * Geometric primitives: dict (axis-aligned plane), 2-pt line, 3-pt plane.
  * The v2 ``crossing_plane`` predicate (on / crossing / not_on /
    not_crossing) over resolved dimtag seeds.
  * The chainable ``EntitySelection`` → legacy ``Selection`` terminal:
    ``.to_label()``, ``.to_physical()``, ``.tags()``, repr.
  * Mixed-dim selections, set-algebra, negation, error paths.
"""
from __future__ import annotations

import pytest

from apeGmsh.core._selection import (
    Selection, Plane, Line, _parse_primitive,
)


def _is_selection(obj) -> bool:
    """Robust isinstance check (legacy ``Selection`` terminal).

    ``test_library_contracts.py`` purges ``apeGmsh`` from ``sys.modules``
    and re-imports; checking by class name is robust to that.
    """
    return type(obj).__name__ == 'Selection'


def _tags(sel) -> list:
    """Sorted tag list from an ``EntitySelection`` (atoms are bare
    ``(dim, tag)``)."""
    return sorted(int(t) for _d, t in sel._items)


# =====================================================================
# Helpers
# =====================================================================

def _box_edges(g, vol):
    """All 12 unique edges of a hex volume, via the two-step boundary query."""
    faces = g.model.queries.boundary(vol, oriented=False)
    edges = list(dict.fromkeys(
        g.model.queries.boundary(faces, combined=False, oriented=False)
    ))
    return edges


# =====================================================================
# Primitive parser — RETAINED (_parse_primitive / Plane / Line)
# =====================================================================

class TestParsePrimitive:
    """``_parse_primitive`` infers Plane/Line from raw user input."""

    def test_dict_axis_aligned_plane(self):
        prim = _parse_primitive({'z': 0})
        assert isinstance(prim, Plane)
        assert tuple(prim.normal) == (0.0, 0.0, 1.0)

    def test_two_points_yield_line(self):
        prim = _parse_primitive([(0, 0, 0), (1, 0, 0)])
        assert isinstance(prim, Line)

    def test_three_points_yield_plane(self):
        prim = _parse_primitive([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
        assert isinstance(prim, Plane)

    def test_passthrough_plane(self):
        p = Plane.at(z=0)
        assert _parse_primitive(p) is p

    def test_invalid_count_raises(self):
        with pytest.raises(ValueError, match="Cannot infer"):
            _parse_primitive([(0, 0, 0)])

    def test_collinear_plane_raises(self):
        with pytest.raises(ValueError, match="collinear"):
            _parse_primitive([(0, 0, 0), (1, 0, 0), (2, 0, 0)])

    def test_coincident_line_raises(self):
        with pytest.raises(ValueError, match="coincident"):
            _parse_primitive([(0, 0, 0), (0, 0, 0)])


# =====================================================================
# crossing_plane — 2-D rectangle (v2 idiom; frozen literals)
# =====================================================================

class TestSelect2D:
    """A flat rectangle has 4 edges; predicates pick them out."""

    def test_on_axis_aligned_picks_one_edge(self, g):
        surf = g.model.geometry.add_rectangle(0, 0, 0, 5, 10)
        curves = g.model.queries.boundary(surf, oriented=False)

        bottom = g.model.select(curves).crossing_plane({'y': 0}, mode='on')
        top    = g.model.select(curves).crossing_plane({'y': 10}, mode='on')
        left   = g.model.select(curves).crossing_plane({'x': 0}, mode='on')
        right  = g.model.select(curves).crossing_plane({'x': 5}, mode='on')

        assert len(bottom) == len(top) == len(left) == len(right) == 1
        # the four edges are distinct
        all_tags = {_tags(bottom)[0], _tags(top)[0],
                    _tags(left)[0], _tags(right)[0]}
        assert len(all_tags) == 4

    def test_crossing_two_point_line(self, g):
        surf = g.model.geometry.add_rectangle(0, 0, 0, 5, 10)
        curves = g.model.queries.boundary(surf, oriented=False)

        # horizontal mid-line — the two vertical edges (left, right) cross it
        mid = g.model.select(curves).crossing_plane(
            [(0, 5, 0), (5, 5, 0)], mode='crossing')
        assert len(mid) == 2

    def test_no_match_returns_empty(self, g):
        surf = g.model.geometry.add_rectangle(0, 0, 0, 5, 10)
        curves = g.model.queries.boundary(surf, oriented=False)

        empty = g.model.select(curves).crossing_plane({'x': 99}, mode='on')
        assert len(empty) == 0
        assert _is_selection(empty.result())


# =====================================================================
# crossing_plane — 3-D box
# =====================================================================

class TestSelect3D:
    """A box has 6 faces; predicates pick them out by plane."""

    def test_on_plane_picks_each_face(self, g):
        vol = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(vol, oriented=False)

        assert len(g.model.select(faces).crossing_plane(
            {'z': 0}, mode='on')) == 1
        assert len(g.model.select(faces).crossing_plane(
            {'z': 10}, mode='on')) == 1
        assert len(g.model.select(faces).crossing_plane(
            {'x': 0}, mode='on')) == 1
        assert len(g.model.select(faces).crossing_plane(
            {'x': 5}, mode='on')) == 1

    def test_crossing_plane_picks_vertical_faces(self, g):
        vol = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(vol, oriented=False)

        # mid-height plane — 4 vertical faces straddle it
        side = g.model.select(faces).crossing_plane({'z': 5}, mode='crossing')
        assert len(side) == 4

    def test_crossing_3pt_plane(self, g):
        vol = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(vol, oriented=False)

        # equivalent to z=5 expressed via 3 points
        side = g.model.select(faces).crossing_plane(
            [(0, 0, 5), (1, 0, 5), (0, 1, 5)], mode='crossing')
        assert len(side) == 4

    def test_volume_crosses_plane(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        crossed = g.model.select([(3, v)]).crossing_plane(
            {'z': 5}, mode='crossing')
        assert len(crossed) == 1


# =====================================================================
# Chainable EntitySelection → legacy Selection terminal
# =====================================================================

class TestSelectionChain:

    def test_stacking_is_AND(self, g):
        """Stacked crossing_plane narrows progressively (AND logic)."""
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        edges = _box_edges(g, v)

        # corner edge at x=0, y=0 — exactly 1 edge survives both filters
        corner = (g.model.select(edges)
                    .crossing_plane({'x': 0}, mode='on')
                    .crossing_plane({'y': 0}, mode='on'))
        assert len(corner) == 1

    def test_chain_returns_selection_with_back_ref(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)
        sel = g.model.select(faces).crossing_plane({'z': 0}, mode='on')
        # the v2 terminal materialises the legacy Selection (back-ref'd
        # to the queries engine, R-v2-2).
        leg = sel.result()
        assert _is_selection(leg)
        assert leg._queries is g.model.queries
        # chain still works (legacy Selection.select(on=) unchanged)
        empty = leg.select(on={'x': 99})
        assert _is_selection(empty)
        assert empty._queries is g.model.queries

    def test_tags_drops_dim(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)
        sel = g.model.select(faces).crossing_plane({'z': 5}, mode='crossing')
        tags = sel.result().tags()
        assert all(isinstance(t, int) for t in tags)
        assert len(tags) == 4

    def test_repr_mentions_dim_count_and_chain(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)
        sel = g.model.select(faces).crossing_plane(
            {'z': 5}, mode='crossing').result()
        r = repr(sel)
        assert "4" in r and "surfaces" in r and ".select" in r

    def test_empty_repr(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)
        sel = g.model.select(faces).crossing_plane(
            {'z': 999}, mode='on').result()
        assert "empty" in repr(sel)


# =====================================================================
# to_label / to_physical (direct EntitySelection terminals)
# =====================================================================

class TestSelectionRegister:

    def test_to_label_creates_label_and_returns_self(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)

        sel = g.model.select(faces).crossing_plane(
            {'z': 0}, mode='on').to_label('base')
        assert isinstance(sel, type(g.model.select(faces)))
        # round-trip through the labels composite
        assert g.labels.entities('base', dim=2) == _tags(sel)

    def test_to_physical_creates_pg_and_returns_self(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)

        sel = g.model.select(faces).crossing_plane(
            {'z': 10}, mode='on').to_physical('Top')
        assert isinstance(sel, type(g.model.select(faces)))
        # downstream API can reach it by PG name
        assert g.physical.entities('Top')

    def test_chained_register_then_select_again(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        edges = _box_edges(g, v)

        # full fluent chain: filter → label → keep filtering
        result = (g.model.select(edges)
                    .crossing_plane({'x': 0}, mode='on')
                    .to_label('left_face_edges')
                    .crossing_plane({'y': 0}, mode='on'))
        assert len(result) == 1
        assert g.labels.entities('left_face_edges', dim=1)  # label exists

    def test_mixed_dim_groups_by_dimension(self, g):
        """A Selection holding entities from multiple dims labels each separately."""
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)
        edges = _box_edges(g, v)

        # mixed manual selection — face on z=0 plus an edge on x=0,y=0
        face_sel = g.model.select(faces).crossing_plane({'z': 0}, mode='on')
        edge_sel = (g.model.select(edges)
                     .crossing_plane({'x': 0}, mode='on')
                     .crossing_plane({'y': 0}, mode='on'))
        mixed = Selection(list(face_sel._items) + list(edge_sel._items),
                          _queries=g.model.queries)
        mixed.to_label('mixed')

        assert g.labels.entities('mixed', dim=2) == _tags(face_sel)
        assert g.labels.entities('mixed', dim=1) == _tags(edge_sel)


# =====================================================================
# boundary helpers — RETAINED (queries.boundary_curves / _points)
# =====================================================================

class TestBoundaryHelpers:
    """boundary_curves / boundary_points convenience methods."""

    def test_boundary_curves_box(self, g):
        g.model.geometry.add_box(0, 0, 0, 5, 5, 10, label='box')
        assert len(g.model.queries.boundary_curves('box')) == 12

    def test_boundary_curves_int_tag(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        assert len(g.model.queries.boundary_curves(v)) == 12

    def test_boundary_curves_surface(self, g):
        s = g.model.geometry.add_rectangle(0, 0, 0, 5, 10)
        assert len(g.model.queries.boundary_curves(s)) == 4

    def test_boundary_points_box(self, g):
        g.model.geometry.add_box(0, 0, 0, 5, 5, 10, label='box')
        assert len(g.model.queries.boundary_points('box')) == 8


# =====================================================================
# Label-seeded selection + crossing_plane.  P3-R: the legacy
# ``queries.select(label, dim=N, predicate)`` boundary-extraction is
# removed (the v2 ``g.model.select(label, dim=N)`` resolves the label
# entity, NOT its dim-N boundary).  The dim-N boundary is resolved
# explicitly (the byte-unchanged ``queries.boundary``) and the
# predicate applied via the v2 ``crossing_plane`` — same final set,
# the ``tests/test_p2g_parity.py`` seed pattern.
# =====================================================================

class TestSelectFromLabel:
    """Label-seeded selection + the v2 crossing_plane predicate."""

    def test_label_str_with_dim2(self, g):
        g.model.geometry.add_box(0, 0, 0, 5, 5, 10, label='box')
        g.model.sync()
        faces = g.model.queries.boundary('box', dim=3, oriented=False)
        # the bottom face of the box
        assert len(g.model.select(faces).crossing_plane(
            {'z': 0}, mode='on')) == 1

    def test_label_str_with_dim1(self, g):
        g.model.geometry.add_box(0, 0, 0, 5, 5, 10, label='box')
        g.model.sync()
        edges = g.model.queries.boundary_curves('box')
        # 4 edges on the left face
        assert len(g.model.select(edges).crossing_plane(
            {'x': 0}, mode='on')) == 4

    def test_label_str_with_dim0(self, g):
        g.model.geometry.add_box(0, 0, 0, 5, 5, 10, label='box')
        g.model.sync()
        pts = g.model.queries.boundary_points('box')
        # 4 corner points on the left face
        assert len(g.model.select(pts).crossing_plane(
            {'x': 0}, mode='on')) == 4

    def test_label_string_list(self, g):
        """g.model.select() resolves a *list* of label strings."""
        gm = g.model.geometry
        gm.add_point(0, 0, 0, label='a')
        gm.add_point(0, 0, 1, label='b')
        gm.add_point(1, 0, 0, label='c')
        gm.add_point(1, 0, 1, label='d')
        c1 = gm.add_line('a', 'b', label='col_left')
        c2 = gm.add_line('c', 'd', label='col_right')
        c3 = gm.add_line('a', 'c', label='arch')

        sel = g.model.select(['col_left', 'arch', 'col_right'])
        assert sorted(t for _, t in sel._items) == sorted([c1, c2, c3])

        # mixed list: bare tag + label strings resolve independently
        mixed = g.model.select([c1, 'arch', 'col_right'])
        assert sorted(t for _, t in mixed._items) == sorted([c1, c2, c3])

        # flows into to_physical -> all 3 entities (the original footgun)
        g.model.select(
            ['col_left', 'arch', 'col_right']).to_physical('frames')
        assert sorted(int(x) for x in g.physical.entities('frames', dim=1)) \
            == sorted([c1, c2, c3])


class TestMixedDimLabelNoWarning:
    """Mixed-dim Selection.to_label should not warn about dim collision."""

    def test_no_warning(self, g, recwarn):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)
        edges = g.model.queries.boundary_curves(v)
        face_sel = g.model.select(faces).crossing_plane({'z': 0}, mode='on')
        edge_sel = (g.model.select(edges)
                      .crossing_plane({'x': 0}, mode='on')
                      .crossing_plane({'y': 0}, mode='on'))
        Selection(list(face_sel._items) + list(edge_sel._items),
                  _queries=g.model.queries).to_label('mixed_label')

        # only label-collision warnings would be relevant here; ensure none
        relevant = [w for w in recwarn.list
                    if "already exists at dim" in str(w.message)]
        assert relevant == []


# =====================================================================
# Primitive factories — RETAINED queries.plane(...) + Line.through
# (the legacy queries.line(...) is removed; Line.through is its
# byte-identical replacement — queries.line only ever called it).
# =====================================================================

class TestPrimitiveFactories:
    """plane() factory on _Queries (RETAINED) + Line.through."""

    def test_plane_axis_aligned(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)
        z_mid = g.model.queries.plane(z=5)
        assert len(g.model.select(faces).crossing_plane(
            z_mid, mode='crossing')) == 4

    def test_plane_three_points(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)
        p = g.model.queries.plane((0, 0, 5), (1, 0, 5), (0, 1, 5))
        assert len(g.model.select(faces).crossing_plane(
            p, mode='crossing')) == 4

    def test_plane_normal_through(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)
        p = g.model.queries.plane(normal=(0, 0, 1), through=(0, 0, 5))
        assert len(g.model.select(faces).crossing_plane(
            p, mode='crossing')) == 4

    def test_line_factory(self, g):
        s = g.model.geometry.add_rectangle(0, 0, 0, 5, 10)
        curves = g.model.queries.boundary(s, oriented=False)
        cut = Line.through((0, 5, 0), (5, 5, 0))
        assert len(g.model.select(curves).crossing_plane(
            cut, mode='crossing')) == 2

    def test_plane_invalid_call_raises(self, g):
        with pytest.raises(ValueError):
            g.model.queries.plane()
        with pytest.raises(ValueError):
            g.model.queries.plane((0, 0, 0))   # 1 positional arg


class TestSetOperations:
    """EntitySelection supports |, &, - with set semantics."""

    def test_union_dedups(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)
        bottom = g.model.select(faces).crossing_plane({'z': 0}, mode='on')
        top    = g.model.select(faces).crossing_plane({'z': 10}, mode='on')
        union  = bottom | top
        assert len(union) == 2
        # idempotent
        assert len(bottom | bottom) == 1

    def test_intersection(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)
        side    = g.model.select(faces).crossing_plane(
            {'z': 5}, mode='crossing')
        on_x0   = g.model.select(faces).crossing_plane({'x': 0}, mode='on')
        common  = side & on_x0
        assert len(common) == 1   # the left face is both vertical and at x=0

    def test_difference(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)
        bottom = g.model.select(faces).crossing_plane({'z': 0}, mode='on')
        all_sel = g.model.select(faces)
        without_bottom = all_sel - bottom
        assert len(without_bottom) == 5

    def test_set_ops_preserve_back_ref(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)
        a = g.model.select(faces).crossing_plane({'z': 0}, mode='on')
        b = g.model.select(faces).crossing_plane({'z': 10}, mode='on')
        for sel in (a | b, a & b, a - b):
            # the materialised legacy terminal stays back-ref'd
            leg = sel.result()
            assert leg._queries is g.model.queries
            # chain still works
            leg.select(on={'x': 0})


class TestPartitionBy:
    """partition_by groups by dominant bounding-box axis (RETAINED
    legacy ``Selection.partition_by``)."""

    def test_curves_of_box_three_directions(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 7, 11)
        edges = Selection(g.model.queries.boundary_curves(v),
                          _queries=g.model.queries)
        groups = edges.partition_by()
        assert {'x', 'y', 'z'} == set(groups)
        # each direction has 4 parallel edges on a box
        assert len(groups['x']) == 4
        assert len(groups['y']) == 4
        assert len(groups['z']) == 4

    def test_partition_single_axis_returns_selection(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 7, 11)
        edges = Selection(g.model.queries.boundary_curves(v),
                          _queries=g.model.queries)
        x_edges = edges.partition_by('x')
        assert _is_selection(x_edges)
        assert len(x_edges) == 4

    def test_invalid_axis_raises(self, g):
        sel = Selection([], _queries=g.model.queries)
        with pytest.raises(ValueError, match="must be 'x', 'y', or 'z'"):
            sel.partition_by('w')

    def test_surfaces_partition_by_normal(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = Selection(g.model.queries.boundary(v, oriented=False),
                          _queries=g.model.queries)
        groups = faces.partition_by()
        # 2 faces with x-normal, 2 with y-normal, 2 with z-normal
        assert len(groups['x']) == 2
        assert len(groups['y']) == 2
        assert len(groups['z']) == 2


class TestNegation:
    """not_on / not_crossing negation modes of crossing_plane."""

    def test_not_on_excludes_match(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)
        # 6 faces, 1 is on z=0, so not_on z=0 → 5 faces
        result = g.model.select(faces).crossing_plane(
            {'z': 0}, mode='not_on')
        assert len(result) == 5

    def test_not_crossing_excludes_match(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)
        # 4 vertical faces cross z=5; not_crossing → 2 (top and bottom)
        result = g.model.select(faces).crossing_plane(
            {'z': 5}, mode='not_crossing')
        assert len(result) == 2

    def test_not_on_chained(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)
        # all faces except top and bottom (the four vertical faces)
        result = (g.model.select(faces)
                    .crossing_plane({'z': 0}, mode='not_on')
                    .crossing_plane({'z': 10}, mode='not_on'))
        assert len(result) == 4

    def test_negation_matches_complement(self, g):
        """not_X should give exactly the entities that X excludes."""
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)
        positive = set(g.model.select(faces).crossing_plane(
            {'z': 0}, mode='on')._items)
        negative = set(g.model.select(faces).crossing_plane(
            {'z': 0}, mode='not_on')._items)
        all_set  = {(int(d), int(t)) for d, t in faces}
        assert positive | negative == all_set
        assert positive & negative == set()


class TestSetTransfiniteBox:
    """g.mesh.structured.set_transfinite_box convenience builder."""

    def test_by_size_label(self, g):
        g.model.geometry.add_box(0, 0, 0, 5, 5, 10, label='box')
        g.mesh.structured.set_transfinite_box('box', size=0.5)
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data(dim=3)
        # 5/0.5+1=11 nodes per x/y, 10/0.5+1=21 per z → 11*11*21
        assert fem.info.n_nodes == 11 * 11 * 21

    def test_by_n_int_tag(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        g.mesh.structured.set_transfinite_box(v, n=11)
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data(dim=3)
        assert fem.info.n_nodes == 11 ** 3

    def test_size_and_n_mutually_exclusive(self, g):
        g.model.geometry.add_box(0, 0, 0, 5, 5, 10, label='box')
        with pytest.raises(ValueError, match="exactly one"):
            g.mesh.structured.set_transfinite_box('box', size=0.5, n=11)
        with pytest.raises(ValueError, match="exactly one"):
            g.mesh.structured.set_transfinite_box('box')

    def test_no_recombine_gives_tets(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 2, 2, 2)
        g.mesh.structured.set_transfinite_box(v, n=3, recombine=False)
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data(dim=3)
        # 3x3x3 = 27 nodes; tet count > hex count for same node count
        assert fem.info.n_nodes == 27


class TestErrors:

    def test_crossing_plane_rejects_invalid_spec_and_mode(self, g):
        """selection-unification-v2 P3-R: the legacy
        ``queries.select`` zero/multi-predicate validation is removed;
        the v2 ``crossing_plane`` validates spec + mode loudly (the
        ``tests/test_p2g_parity.py`` ``test_invalid_mode_is_loud``
        contract, mirrored here for the error-path coverage)."""
        surf = g.model.geometry.add_rectangle(0, 0, 0, 1, 1)
        curves = g.model.queries.boundary(surf, oriented=False)

        with pytest.raises(ValueError, match="mode="):
            g.model.select(curves).crossing_plane({'x': 0}, mode='sideways')
        with pytest.raises(ValueError, match="Cannot infer primitive"):
            g.model.select(curves).crossing_plane([(0, 0, 0)], mode='on')

    def test_plane_at_requires_one_kwarg(self):
        with pytest.raises(ValueError, match="exactly one"):
            Plane.at()
        with pytest.raises(ValueError, match="exactly one"):
            Plane.at(x=0, y=0)

    def test_plane_at_unknown_axis(self):
        with pytest.raises(ValueError, match="Unknown axis"):
            Plane.at(w=0)


# =====================================================================
# boundary() — entity-family topological traversal (one dim down).
# A *traversal* verb (not a filter): volume -> faces -> edges, kept
# inside the chain so it composes with the spatial verbs / terminals
# without dropping out to ``queries.boundary`` + raw dimtags.
# =====================================================================

class TestBoundaryTraversal:
    """``g.model.select(...).boundary()`` steps one dimension down."""

    def test_volume_boundary_is_six_faces(self, g):
        g.model.geometry.add_box(0, 0, 0, 5, 5, 10, label='box')
        faces = g.model.select('box', dim=3).boundary()
        assert len(faces) == 6
        assert all(d == 2 for d, _t in faces._items)

    def test_unsigned_tags_only(self, g):
        """oriented=False always — atoms must be positive (a signed tag
        would neither de-dup nor register as a PG)."""
        g.model.geometry.add_box(0, 0, 0, 5, 5, 10, label='box')
        faces = g.model.select('box', dim=3).boundary()
        assert all(t > 0 for _d, t in faces._items)

    def test_chains_with_spatial_verb_and_terminal(self, g):
        """The whole point: boundary then refine then name, no raw tags."""
        g.model.geometry.add_box(0, 0, 0, 5, 5, 10, label='box')
        (g.model.select('box', dim=3)
            .boundary()
            .on_plane((0, 0, 0), (0, 0, 1), tol=1e-6)   # the z=0 face
            .to_physical('Base'))
        base = g.physical.entities('Base', dim=2)
        assert len(base) == 1

    def test_double_boundary_gives_edges(self, g):
        # The 6 faces form a closed shell, so combined=True cancels every
        # shared edge (boundary of a closed manifold is empty) — use
        # combined=False on the second hop to get the 12 unique edges.
        g.model.geometry.add_box(0, 0, 0, 5, 5, 10, label='box')
        faces = g.model.select('box', dim=3).boundary()
        assert len(faces.boundary(combined=True)) == 0   # closed shell
        edges = faces.boundary(combined=False)
        assert len(edges) == 12
        assert all(d == 1 for d, _t in edges._items)

    def test_combined_drops_shared_interface(self, g):
        """Two stacked boxes sharing one face: combined=True returns the
        outer skin (10 faces), combined=False keeps the shared interface
        (11 unique faces)."""
        b1 = g.model.geometry.add_box(0, 0, 0, 5, 5, 5)
        b2 = g.model.geometry.add_box(0, 0, 5, 5, 5, 5)
        g.model.boolean.fragment([(3, b1)], [(3, b2)])
        vols = g.model.queries.entities_in_bounding_box(
            -1, -1, -1, 6, 6, 11, dim=3)
        sel = g.model.select(vols)
        outer = sel.boundary(combined=True)
        allf = sel.boundary(combined=False)
        assert len(outer) == 10
        assert len(allf) == 11

    def test_empty_selection_boundary_is_empty(self, g):
        g.model.geometry.add_box(0, 0, 0, 5, 5, 10, label='box')
        empty = (g.model.select('box', dim=3)
                  .in_box((100, 100, 100), (200, 200, 200)))
        assert len(empty.boundary()) == 0

    def test_point_family_has_no_boundary(self, g):
        """boundary is entity-only (not a REQUIRED_VERB) — the point
        family raises a plain AttributeError, not a fail-loud stub."""
        g.model.geometry.add_box(0, 0, 0, 5, 5, 10, label='box')
        g.physical.add_volume('box', name='Body')
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data(dim=3)
        with pytest.raises(AttributeError):
            fem.nodes.select(pg='Body').boundary()
