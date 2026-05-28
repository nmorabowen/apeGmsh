"""Phase 4 — FEMDataSource.boundary_faces_for (Compose v1.1-A.2 PR B / ADR 0041).

Tests the new element-side query on
:class:`apeGmsh._kernel.resolvers._source.FEMDataSource` for the
tied-contact / mortar code path.  Builds in-memory :class:`FEMData`
fixtures with mixed volume / surface element groups and exercises the
node-ownership filter that mirrors
:meth:`PartsRegistry.build_face_map`.

No gmsh, no openseespy, no file I/O.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh._kernel.record_sets import ComposeSet
from apeGmsh._kernel.resolvers._source import FEMDataSource
from apeGmsh.mesh._element_types import make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)
from apeGmsh._kernel.payloads import ElementGroup


# ---------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------


def _two_hex_with_quad4_interface_fem(
    *,
    interface_label: str = "contact_surf",
) -> FEMData:
    """Build a FEMData with two hex8 blocks sharing a quad4 interface.

    Layout (front view, y=0 face):

        z=1   5---6   13--14
               |   |   |   |
        z=0   1---2   9--10
              x=0 x=1 x=1 x=2   (the x=1 face is shared)

    Node ids 1..8 → hex A (corners of unit cube at origin).
    Node ids 9..12 → hex B's right-face corners (the left face IS the
    shared interface 2,3,7,6 → so hex B uses 2,9 / 3,10 / 6,13 / 7,14
    in Gmsh hex8 order).

    For simplicity: hex A has corners (0,0,0)..(1,1,1) ids 1..8.
    Hex B has corners (1,0,0)..(2,1,1) ids 2,9,10,3,6,13,14,7.

    One quad4 surface element (id 500) on the shared interface using
    nodes 2,3,7,6 in Gmsh quad4 order.

    A node-side PG names ``interface_label`` covering nodes 2,3,6,7.
    """
    # Hex A corners: ids 1..8 (Gmsh hex8 order)
    coords_a = np.array(
        [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],  # 1, 2
            [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],  # 3, 4
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],  # 5, 6
            [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],  # 7, 8
        ],
        dtype=np.float64,
    )
    # Hex B unique corners: ids 9, 10, 13, 14 (the other four come
    # from hex A — 2, 3, 6, 7).
    coords_b_unique = np.array(
        [
            [2.0, 0.0, 0.0],  # 9
            [2.0, 1.0, 0.0],  # 10
            [2.0, 0.0, 1.0],  # 13
            [2.0, 1.0, 1.0],  # 14
        ],
        dtype=np.float64,
    )
    # We need ids 1..10 + 13, 14, but a sparse id list works for the
    # broker.  Use consecutive ids 1..12 in stored order; map "13"→11,
    # "14"→12 for cleanliness.  Connectivity uses the stored ids.
    node_ids = np.arange(1, 13, dtype=np.int64)
    coords = np.vstack([
        coords_a,                  # 1..8
        coords_b_unique[:2],       # 9, 10
        coords_b_unique[2:],       # 11 (was 13), 12 (was 14)
    ])

    # Hex A: gmsh order is bottom CCW then top CCW.
    hex_a_conn = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)
    # Hex B: bottom CCW starting at the shared edge 2→9 / top 6→11.
    # Local order: 2, 9, 10, 3, 6, 11, 12, 7
    hex_b_conn = np.array([[2, 9, 10, 3, 6, 11, 12, 7]], dtype=np.int64)
    hex_conn = np.vstack([hex_a_conn, hex_b_conn])
    hex_info = make_type_info(
        code=5, gmsh_name="Hexahedron 8", dim=3, order=1, npe=8, count=2,
    )
    hex_group = ElementGroup(
        element_type=hex_info,
        ids=np.array([100, 101], dtype=np.int64),
        connectivity=hex_conn,
    )

    # Quad4 surface element on the shared face — gmsh quad4 order.
    quad_conn = np.array([[2, 3, 7, 6]], dtype=np.int64)
    quad_info = make_type_info(
        code=3, gmsh_name="Quadrangle 4", dim=2, order=1, npe=4, count=1,
    )
    quad_group = ElementGroup(
        element_type=quad_info,
        ids=np.array([500], dtype=np.int64),
        connectivity=quad_conn,
    )

    # Node-side PG naming the interface (4 nodes).
    node_pgs = {
        (0, 7): {
            "name": interface_label,
            "node_ids": np.array([2, 3, 6, 7], dtype=np.int64),
            "node_coords": coords[[1, 2, 5, 6]],
        },
    }
    elements = ElementComposite(
        groups={5: hex_group, 3: quad_group},
        physical=PhysicalGroupSet({}),
        labels=LabelSet({}),
    )
    nodes = NodeComposite(
        node_ids=node_ids,
        node_coords=coords,
        physical=PhysicalGroupSet(node_pgs),
        labels=LabelSet({}),
    )
    info = MeshInfo(
        n_nodes=node_ids.size, n_elems=3, bandwidth=1,
        types=[hex_info, quad_info],
    )
    return FEMData(
        nodes=nodes,
        elements=elements,
        info=info,
        composed_from=ComposeSet(()),
    )


def _tri3_surface_only_fem(
    *,
    label: str = "tri_surf",
) -> FEMData:
    """A simple FEMData with ONLY tri3 surface elements (no volume).

    Two tri3 faces on the unit square (z=0) — node ids 1..4, conn
    [[1,2,3], [1,3,4]].  A node-side PG names all four nodes.
    """
    coords = np.array(
        [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    node_ids = np.arange(1, 5, dtype=np.int64)
    tri_conn = np.array([[1, 2, 3], [1, 3, 4]], dtype=np.int64)
    tri_info = make_type_info(
        code=2, gmsh_name="Triangle 3", dim=2, order=1, npe=3, count=2,
    )
    tri_group = ElementGroup(
        element_type=tri_info,
        ids=np.array([200, 201], dtype=np.int64),
        connectivity=tri_conn,
    )
    node_pgs = {
        (0, 8): {
            "name": label,
            "node_ids": node_ids,
            "node_coords": coords,
        },
    }
    elements = ElementComposite(
        groups={2: tri_group},
        physical=PhysicalGroupSet({}),
        labels=LabelSet({}),
    )
    nodes = NodeComposite(
        node_ids=node_ids,
        node_coords=coords,
        physical=PhysicalGroupSet(node_pgs),
        labels=LabelSet({}),
    )
    info = MeshInfo(
        n_nodes=node_ids.size, n_elems=2, bandwidth=1, types=[tri_info],
    )
    return FEMData(
        nodes=nodes,
        elements=elements,
        info=info,
        composed_from=ComposeSet(()),
    )


def _hex_only_fem() -> FEMData:
    """FEMData with one hex8 (dim=3) only — no surface elements.

    Carries node-side PGs for master/slave to exercise the
    ADR 0041 §"Decision 5" "no dim=2 groups" branch.
    """
    coords = np.array(
        [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    node_ids = np.arange(1, 9, dtype=np.int64)
    hex_info = make_type_info(
        code=5, gmsh_name="Hexahedron 8", dim=3, order=1, npe=8, count=1,
    )
    hex_group = ElementGroup(
        element_type=hex_info,
        ids=np.array([100], dtype=np.int64),
        connectivity=np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64),
    )
    node_pgs = {
        (0, 1): {
            "name": "face_a",
            "node_ids": np.array([1, 2, 3, 4], dtype=np.int64),
            "node_coords": coords[:4],
        },
    }
    elements = ElementComposite(
        groups={5: hex_group},
        physical=PhysicalGroupSet({}),
        labels=LabelSet({}),
    )
    nodes = NodeComposite(
        node_ids=node_ids,
        node_coords=coords,
        physical=PhysicalGroupSet(node_pgs),
        labels=LabelSet({}),
    )
    info = MeshInfo(
        n_nodes=node_ids.size, n_elems=1, bandwidth=1, types=[hex_info],
    )
    return FEMData(
        nodes=nodes,
        elements=elements,
        info=info,
        composed_from=ComposeSet(()),
    )


# ---------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------


class TestBoundaryFacesQuad4:
    def test_resolves_quad4_interface(self) -> None:
        fem = _two_hex_with_quad4_interface_fem()
        src = FEMDataSource(fem)

        out = src.boundary_faces_for("contact_surf")
        assert out.shape == (1, 4)
        # The single quad row, sorted into a set for order-independence.
        row = tuple(sorted(int(x) for x in out[0]))
        assert row == (2, 3, 6, 7)


class TestBoundaryFacesTri3:
    def test_resolves_tri3_surface_mesh(self) -> None:
        fem = _tri3_surface_only_fem()
        src = FEMDataSource(fem)

        out = src.boundary_faces_for("tri_surf")
        assert out.shape == (2, 3)
        rows = {
            tuple(sorted(int(x) for x in row)) for row in out
        }
        assert rows == {(1, 2, 3), (1, 3, 4)}


# ---------------------------------------------------------------------
# Sub-selection by node ownership
# ---------------------------------------------------------------------


class TestNodeOwnershipFilter:
    def test_only_full_rows_survive(self) -> None:
        """A PG covering only some of a quad's corners produces no
        face rows (every row needs ALL nodes in the set)."""
        fem = _tri3_surface_only_fem()
        # Replace the PG with one covering only nodes 1, 2, 3 (which
        # exactly matches the first triangle but NOT the second).
        new_node_pgs = {
            (0, 8): {
                "name": "partial",
                "node_ids": np.array([1, 2, 3], dtype=np.int64),
                "node_coords": fem.nodes.coords[:3],
            },
        }
        new_nodes = NodeComposite(
            node_ids=fem.nodes.ids,
            node_coords=fem.nodes.coords,
            physical=PhysicalGroupSet(new_node_pgs),
            labels=LabelSet({}),
        )
        fem2 = FEMData(
            nodes=new_nodes,
            elements=fem.elements,
            info=fem.info,
            composed_from=ComposeSet(()),
        )
        src = FEMDataSource(fem2)
        out = src.boundary_faces_for("partial")
        # Only the first triangle [1,2,3] survives.
        assert out.shape == (1, 3)
        assert tuple(sorted(int(x) for x in out[0])) == (1, 2, 3)


# ---------------------------------------------------------------------
# No dim=2 ElementGroups → ValueError (ADR 0041 §"Decision 5")
# ---------------------------------------------------------------------


class TestNoSurfaceGroupsRaises:
    def test_hex_only_broker_raises_value_error(self) -> None:
        fem = _hex_only_fem()
        src = FEMDataSource(fem)
        with pytest.raises(ValueError) as ei:
            src.boundary_faces_for("face_a")
        msg = str(ei.value)
        assert "no dim=2 ElementGroups" in msg
        assert "re-extract" in msg
        assert "dim=None" in msg


# ---------------------------------------------------------------------
# Empty filter result — broker has dim=2 groups but no rows match
# ---------------------------------------------------------------------


class TestEmptyFilterResult:
    def test_returns_empty_when_target_nodes_dont_cover_any_face(
        self,
    ) -> None:
        """A target whose node set exists in the broker but doesn't
        cover any full face row returns an empty array (no raise),
        consistent with :meth:`PartsRegistry.build_face_map` for an
        empty instance."""
        fem = _two_hex_with_quad4_interface_fem()
        # Replace the PG with one naming a single node (id 9) that is
        # NOT part of the quad's 4 corners (2, 3, 6, 7).
        new_node_pgs = {
            (0, 7): {
                "name": "lone",
                "node_ids": np.array([9], dtype=np.int64),
                "node_coords": fem.nodes.coords[8:9],
            },
        }
        new_nodes = NodeComposite(
            node_ids=fem.nodes.ids,
            node_coords=fem.nodes.coords,
            physical=PhysicalGroupSet(new_node_pgs),
            labels=LabelSet({}),
        )
        fem2 = FEMData(
            nodes=new_nodes,
            elements=fem.elements,
            info=fem.info,
            composed_from=ComposeSet(()),
        )
        src = FEMDataSource(fem2)
        out = src.boundary_faces_for("lone")
        assert out.shape == (0, 0)
        assert out.dtype == np.int64


# ---------------------------------------------------------------------
# Unknown target → KeyError (propagated from nodes_for)
# ---------------------------------------------------------------------


class TestUnknownTarget:
    def test_unknown_target_raises_key_error(self) -> None:
        fem = _two_hex_with_quad4_interface_fem()
        src = FEMDataSource(fem)
        with pytest.raises(KeyError, match="resolves to neither"):
            src.boundary_faces_for("not_a_label")


# ---------------------------------------------------------------------
# Mixed npe (tri3 + quad4 both surviving) → ValueError
# ---------------------------------------------------------------------


class TestMixedNpeRaises:
    def test_tri3_plus_quad4_both_kept_raises(self) -> None:
        """Two dim=2 groups (tri3 + quad4) where the target's node set
        covers full rows in BOTH groups — must raise because the
        downstream resolver expects a single rectangular array."""
        # Start from tri3 fixture, then add a quad4 element on the
        # same node set.
        fem = _tri3_surface_only_fem()
        # Add quad4 connecting all 4 nodes (1,2,3,4) — this row is
        # fully owned by the existing "tri_surf" PG.
        quad_info = make_type_info(
            code=3, gmsh_name="Quadrangle 4", dim=2, order=1, npe=4,
            count=1,
        )
        quad_group = ElementGroup(
            element_type=quad_info,
            ids=np.array([300], dtype=np.int64),
            connectivity=np.array([[1, 2, 3, 4]], dtype=np.int64),
        )
        new_groups = dict(fem.elements._groups)
        new_groups[3] = quad_group
        new_elements = ElementComposite(
            groups=new_groups,
            physical=PhysicalGroupSet({}),
            labels=LabelSet({}),
        )
        fem2 = FEMData(
            nodes=fem.nodes,
            elements=new_elements,
            info=fem.info,
            composed_from=ComposeSet(()),
        )
        src = FEMDataSource(fem2)
        with pytest.raises(
            ValueError, match="mixed surface element types",
        ):
            src.boundary_faces_for("tri_surf")
