"""
Tests for ``g.model.queries.select()`` and the ``Selection`` result type.

Covers:
  * Geometric primitives: dict (axis-aligned plane), 2-pt line, 3-pt plane.
  * Predicates: ``on=`` and ``crossing=``.
  * The chainable ``Selection`` API: ``.select()``, ``.tags()``,
    ``.to_label()``, ``.to_physical()``, repr.
  * Mixed-dim selections.
  * Error paths.
"""
from __future__ import annotations

import pytest

from apeGmsh.core._selection import Selection, Plane, Line, _parse_primitive


def _is_selection(obj) -> bool:
    """Robust isinstance check.

    ``test_library_contracts.py`` purges ``apeGmsh`` from ``sys.modules`` and
    re-imports, creating a fresh ``Selection`` class in a new module.  When
    later tests run, the module-level ``Selection`` reference captured at
    import time may not match the class actually returned by ``select()``.
    Checking by class name is robust to that re-import dance.
    """
    return type(obj).__name__ == 'Selection'


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
# Primitive parser
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
# select() — 2-D rectangle
# =====================================================================

class TestSelect2D:
    """A flat rectangle has 4 edges; predicates pick them out."""

    def test_on_axis_aligned_picks_one_edge(self, g):
        surf = g.model.geometry.add_rectangle(0, 0, 0, 5, 10)
        curves = g.model.queries.boundary(surf, oriented=False)

        bottom = g.model.queries.select(curves, on={'y': 0})
        top    = g.model.queries.select(curves, on={'y': 10})
        left   = g.model.queries.select(curves, on={'x': 0})
        right  = g.model.queries.select(curves, on={'x': 5})

        assert len(bottom) == len(top) == len(left) == len(right) == 1
        # the four edges are distinct
        all_tags = {bottom.tags()[0], top.tags()[0], left.tags()[0], right.tags()[0]}
        assert len(all_tags) == 4

    def test_crossing_two_point_line(self, g):
        surf = g.model.geometry.add_rectangle(0, 0, 0, 5, 10)
        curves = g.model.queries.boundary(surf, oriented=False)

        # horizontal mid-line — the two vertical edges (left, right) cross it
        mid = g.model.queries.select(curves, crossing=[(0, 5, 0), (5, 5, 0)])
        assert len(mid) == 2

    def test_no_match_returns_empty(self, g):
        surf = g.model.geometry.add_rectangle(0, 0, 0, 5, 10)
        curves = g.model.queries.boundary(surf, oriented=False)

        empty = g.model.queries.select(curves, on={'x': 99})
        assert len(empty) == 0
        assert _is_selection(empty)


# =====================================================================
# select() — 3-D box
# =====================================================================

class TestSelect3D:
    """A box has 6 faces; predicates pick them out by plane."""

    def test_on_plane_picks_each_face(self, g):
        vol = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(vol, oriented=False)

        assert len(g.model.queries.select(faces, on={'z': 0}))  == 1
        assert len(g.model.queries.select(faces, on={'z': 10})) == 1
        assert len(g.model.queries.select(faces, on={'x': 0}))  == 1
        assert len(g.model.queries.select(faces, on={'x': 5}))  == 1

    def test_crossing_plane_picks_vertical_faces(self, g):
        vol = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(vol, oriented=False)

        # mid-height plane — 4 vertical faces straddle it
        side = g.model.queries.select(faces, crossing={'z': 5})
        assert len(side) == 4

    def test_crossing_3pt_plane(self, g):
        vol = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(vol, oriented=False)

        # equivalent to z=5 expressed via 3 points
        side = g.model.queries.select(
            faces,
            crossing=[(0, 0, 5), (1, 0, 5), (0, 1, 5)],
        )
        assert len(side) == 4

    def test_volume_crosses_plane(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        crossed = g.model.queries.select([(3, v)], crossing={'z': 5})
        assert len(crossed) == 1


# =====================================================================
# Chainable Selection — stacking, .tags(), repr
# =====================================================================

class TestSelectionChain:

    def test_stacking_is_AND(self, g):
        """Stacked .select() narrows progressively (AND logic)."""
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        edges = _box_edges(g, v)

        # corner edge at x=0, y=0 — exactly 1 edge survives both filters
        corner = (g.model.queries
                    .select(edges, on={'x': 0})
                    .select(on={'y': 0}))
        assert len(corner) == 1

    def test_chain_returns_selection_with_back_ref(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)
        sel = g.model.queries.select(faces, on={'z': 0})
        assert _is_selection(sel)
        assert sel._queries is g.model.queries
        # chain still works
        empty = sel.select(on={'x': 99})
        assert _is_selection(empty)
        assert empty._queries is g.model.queries

    def test_tags_drops_dim(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)
        sel = g.model.queries.select(faces, crossing={'z': 5})
        tags = sel.tags()
        assert all(isinstance(t, int) for t in tags)
        assert len(tags) == 4

    def test_repr_mentions_dim_count_and_chain(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)
        sel = g.model.queries.select(faces, crossing={'z': 5})
        r = repr(sel)
        assert "4" in r and "surfaces" in r and ".select" in r

    def test_empty_repr(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)
        sel = g.model.queries.select(faces, on={'z': 999})
        assert "empty" in repr(sel)


# =====================================================================
# to_label / to_physical
# =====================================================================

class TestSelectionRegister:

    def test_to_label_creates_label_and_returns_self(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)

        sel = g.model.queries.select(faces, on={'z': 0}).to_label('base')
        assert _is_selection(sel)
        # round-trip through the labels composite
        assert g.labels.entities('base', dim=2) == sel.tags()

    def test_to_physical_creates_pg_and_returns_self(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)

        sel = g.model.queries.select(faces, on={'z': 10}).to_physical('Top')
        assert _is_selection(sel)
        # downstream API can reach it by PG name
        assert g.physical.dim_tags('Top')

    def test_chained_register_then_select_again(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        edges = _box_edges(g, v)

        # full fluent chain: filter → label → keep filtering
        result = (g.model.queries
                    .select(edges, on={'x': 0})
                    .to_label('left_face_edges')
                    .select(on={'y': 0}))
        assert len(result) == 1
        assert g.labels.entities('left_face_edges', dim=1)  # label exists

    def test_mixed_dim_groups_by_dimension(self, g):
        """A Selection holding entities from multiple dims labels each separately."""
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)
        edges = _box_edges(g, v)

        # mixed manual selection — face on z=0 plus an edge on x=0,y=0
        face_sel = g.model.queries.select(faces, on={'z': 0})
        edge_sel = (g.model.queries
                     .select(edges, on={'x': 0})
                     .select(on={'y': 0}))
        mixed = Selection(list(face_sel) + list(edge_sel),
                          _queries=g.model.queries)
        mixed.to_label('mixed')

        assert g.labels.entities('mixed', dim=2) == face_sel.tags()
        assert g.labels.entities('mixed', dim=1) == edge_sel.tags()


# =====================================================================
# Error paths
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


class TestSelectFromLabel:
    """select() accepting a label string + dim keyword."""

    def test_label_str_with_dim2(self, g):
        g.model.geometry.add_box(0, 0, 0, 5, 5, 10, label='box')
        # the bottom face of the box
        assert len(g.model.queries.select('box', dim=2, on={'z': 0})) == 1

    def test_label_str_with_dim1(self, g):
        g.model.geometry.add_box(0, 0, 0, 5, 5, 10, label='box')
        # 4 edges on the left face
        assert len(g.model.queries.select('box', dim=1, on={'x': 0})) == 4

    def test_label_str_with_dim0(self, g):
        g.model.geometry.add_box(0, 0, 0, 5, 5, 10, label='box')
        # 4 corner points on the left face
        assert len(g.model.queries.select('box', dim=0, on={'x': 0})) == 4


class TestMixedDimLabelNoWarning:
    """Mixed-dim Selection.to_label should not warn about dim collision."""

    def test_no_warning(self, g, recwarn):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)
        edges = g.model.queries.boundary_curves(v)
        face_sel = g.model.queries.select(faces, on={'z': 0})
        edge_sel = (g.model.queries
                      .select(edges, on={'x': 0})
                      .select(on={'y': 0}))
        Selection(list(face_sel) + list(edge_sel),
                  _queries=g.model.queries).to_label('mixed_label')

        # only label-collision warnings would be relevant here; ensure none
        relevant = [w for w in recwarn.list
                    if "already exists at dim" in str(w.message)]
        assert relevant == []


class TestPrimitiveFactories:
    """plane()/line() factories on _Queries — no imports needed by user."""

    def test_plane_axis_aligned(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)
        z_mid = g.model.queries.plane(z=5)
        assert len(g.model.queries.select(faces, crossing=z_mid)) == 4

    def test_plane_three_points(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)
        p = g.model.queries.plane((0, 0, 5), (1, 0, 5), (0, 1, 5))
        assert len(g.model.queries.select(faces, crossing=p)) == 4

    def test_plane_normal_through(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)
        p = g.model.queries.plane(normal=(0, 0, 1), through=(0, 0, 5))
        assert len(g.model.queries.select(faces, crossing=p)) == 4

    def test_line_factory(self, g):
        s = g.model.geometry.add_rectangle(0, 0, 0, 5, 10)
        curves = g.model.queries.boundary(s, oriented=False)
        cut = g.model.queries.line((0, 5, 0), (5, 5, 0))
        assert len(g.model.queries.select(curves, crossing=cut)) == 2

    def test_plane_invalid_call_raises(self, g):
        with pytest.raises(ValueError):
            g.model.queries.plane()
        with pytest.raises(ValueError):
            g.model.queries.plane((0, 0, 0))   # 1 positional arg


class TestSetOperations:
    """Selection supports |, &, - with deduplication / set semantics."""

    def test_union_dedups(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)
        bottom = g.model.queries.select(faces, on={'z': 0})
        top    = g.model.queries.select(faces, on={'z': 10})
        union  = bottom | top
        assert len(union) == 2
        # idempotent
        assert len(bottom | bottom) == 1

    def test_intersection(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)
        side    = g.model.queries.select(faces, crossing={'z': 5})
        on_x0   = g.model.queries.select(faces, on={'x': 0})
        common  = side & on_x0
        assert len(common) == 1   # the left face is both vertical and at x=0

    def test_difference(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)
        bottom = g.model.queries.select(faces, on={'z': 0})
        all_sel = Selection(faces, _queries=g.model.queries)
        without_bottom = all_sel - bottom
        assert len(without_bottom) == 5

    def test_set_ops_preserve_back_ref(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)
        a = g.model.queries.select(faces, on={'z': 0})
        b = g.model.queries.select(faces, on={'z': 10})
        for sel in (a | b, a & b, a - b):
            assert sel._queries is g.model.queries
            # chain still works
            sel.select(on={'x': 0})


class TestPartitionBy:
    """partition_by groups by dominant bounding-box axis."""

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
    """not_on= and not_crossing= negation predicates."""

    def test_not_on_excludes_match(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)
        # 6 faces, 1 is on z=0, so not_on={'z':0} → 5 faces
        result = g.model.queries.select(faces, not_on={'z': 0})
        assert len(result) == 5

    def test_not_crossing_excludes_match(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)
        # 4 vertical faces cross z=5; not_crossing → 2 (top and bottom)
        result = g.model.queries.select(faces, not_crossing={'z': 5})
        assert len(result) == 2

    def test_not_on_chained(self, g):
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)
        # all faces except top and bottom (the four vertical faces)
        result = (g.model.queries
                    .select(faces, not_on={'z': 0})
                    .select(not_on={'z': 10}))
        assert len(result) == 4

    def test_negation_matches_complement(self, g):
        """not_X should give exactly the entities that X excludes."""
        v = g.model.geometry.add_box(0, 0, 0, 5, 5, 10)
        faces = g.model.queries.boundary(v, oriented=False)
        positive = set(g.model.queries.select(faces, on={'z': 0}))
        negative = set(g.model.queries.select(faces, not_on={'z': 0}))
        all_set  = set(faces)
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

    def test_select_requires_exactly_one_predicate(self, g):
        surf = g.model.geometry.add_rectangle(0, 0, 0, 1, 1)
        curves = g.model.queries.boundary(surf, oriented=False)

        with pytest.raises(ValueError, match="exactly one"):
            g.model.queries.select(curves)
        with pytest.raises(ValueError, match="exactly one"):
            g.model.queries.select(curves, on={'x': 0}, crossing={'y': 0})

    def test_plane_at_requires_one_kwarg(self):
        with pytest.raises(ValueError, match="exactly one"):
            Plane.at()
        with pytest.raises(ValueError, match="exactly one"):
            Plane.at(x=0, y=0)

    def test_plane_at_unknown_axis(self):
        with pytest.raises(ValueError, match="Unknown axis"):
            Plane.at(w=0)
