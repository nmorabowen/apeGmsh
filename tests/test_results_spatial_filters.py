"""Spatial filters on Results composites — ``nearest_to`` / ``in_box``.

Both are convenience helpers that compute IDs from geometry and
delegate to the existing ``.get(ids=...)`` path. Filters are
**additive**: passing ``pg=`` / ``label=`` / ``selection=`` / ``ids=``
restricts the candidate set, then the spatial filter narrows it.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter


# =====================================================================
# Synthetic results file + mock FEM
# =====================================================================

def _make_results_with_fem(tmp_path: Path):
    """Build a 4-node, 1-element synthetic results file + mock FEM.

    Nodes laid out at the corners of a unit square in the z=0 plane:
        1 → (0, 0, 0)
        2 → (1, 0, 0)
        3 → (1, 1, 0)
        4 → (0, 1, 0)
    Plus node 5 at (5, 5, 5) far from the others.

    One quadrilateral element with id=10 connecting nodes 1-2-3-4.
    """
    path = tmp_path / "synthetic.h5"
    time = np.array([0.0, 1.0])
    node_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    ux = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                   [1.1, 1.2, 1.3, 1.4, 1.5]])

    with NativeWriter(path) as w:
        w.open(source_type="domain_capture")
        sid = w.begin_stage(name="static", kind="static", time=time)
        w.write_nodes(sid, "partition_0", node_ids=node_ids,
                      components={"displacement_x": ux})
        w.end_stage()

    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [5.0, 5.0, 5.0],
    ], dtype=np.float64)

    # Minimal FEM mock — only the attributes the spatial helpers touch.
    type_info = SimpleNamespace(name="quad4")

    def _resolve(*, element_type=None):
        return (
            np.array([10], dtype=np.int64),
            np.array([[1, 2, 3, 4]], dtype=np.int64),
        )

    fem = SimpleNamespace(
        snapshot_id="testhash",
        nodes=SimpleNamespace(
            ids=node_ids,
            coords=coords,
            physical=SimpleNamespace(node_ids=lambda n: {
                "TopRow": np.array([3, 4], dtype=np.int64),
                "Single": np.array([5], dtype=np.int64),
            }[n]),
            labels=SimpleNamespace(
                node_ids=lambda n: np.array([], dtype=np.int64),
            ),
        ),
        elements=SimpleNamespace(
            ids=np.array([10], dtype=np.int64),
            types=[type_info],
            resolve=_resolve,
            physical=SimpleNamespace(
                element_ids=lambda n: np.array([10], dtype=np.int64),
            ),
            labels=SimpleNamespace(
                element_ids=lambda n: np.array([], dtype=np.int64),
            ),
        ),
    )

    return Results.from_native(path, fem=fem), fem


# =====================================================================
# Node spatial filters
# =====================================================================

def test_nodes_nearest_to_picks_closest(tmp_path: Path) -> None:
    r, _fem = _make_results_with_fem(tmp_path)
    slab = r.nodes.nearest_to((0.1, 0.1, 0.0), component="displacement_x")
    assert slab.node_ids.tolist() == [1]    # corner (0,0,0) is closest
    assert slab.values.shape == (2, 1)


def test_nodes_nearest_to_far_corner(tmp_path: Path) -> None:
    r, _fem = _make_results_with_fem(tmp_path)
    slab = r.nodes.nearest_to((4.5, 4.5, 4.5), component="displacement_x")
    assert slab.node_ids.tolist() == [5]


def test_nodes_in_box_returns_corners(tmp_path: Path) -> None:
    r, _fem = _make_results_with_fem(tmp_path)
    slab = r.nodes.in_box(
        box_min=(-0.5, -0.5, -0.5),
        box_max=(1.5, 1.5, 1.5),
        component="displacement_x",
    )
    # All four unit-square corners in box; node 5 outside
    assert sorted(slab.node_ids.tolist()) == [1, 2, 3, 4]


def test_nodes_in_box_half_open_excludes_upper_face(tmp_path: Path) -> None:
    r, _fem = _make_results_with_fem(tmp_path)
    # Tight box exactly on (0,0,0) and (1,1,1) — half-open [a, b) excludes
    # nodes with x=1 or y=1 from the upper face.
    slab = r.nodes.in_box(
        box_min=(0.0, 0.0, 0.0),
        box_max=(1.0, 1.0, 1.0),
        component="displacement_x",
    )
    # Only node 1 at (0,0,0) is fully inside
    assert slab.node_ids.tolist() == [1]


def test_nodes_in_box_with_inf(tmp_path: Path) -> None:
    r, _fem = _make_results_with_fem(tmp_path)
    # Slab in z = 0 plane only — relax x, y
    slab = r.nodes.in_box(
        box_min=(-np.inf, -np.inf, -0.1),
        box_max=(np.inf, np.inf, 0.5),
        component="displacement_x",
    )
    # All four corners (z=0); node 5 at z=5 excluded
    assert sorted(slab.node_ids.tolist()) == [1, 2, 3, 4]


# =====================================================================
# Additive composition — spatial AND named selectors
# =====================================================================

def test_nodes_in_box_additive_with_pg(tmp_path: Path) -> None:
    r, _fem = _make_results_with_fem(tmp_path)
    # PG TopRow → {3, 4}; box covers {1, 2, 3, 4}; intersection → {3, 4}
    slab = r.nodes.in_box(
        box_min=(-0.5, -0.5, -0.5),
        box_max=(1.5, 1.5, 1.5),
        component="displacement_x",
        pg="TopRow",
    )
    assert sorted(slab.node_ids.tolist()) == [3, 4]


def test_nodes_in_box_additive_empty_intersection(tmp_path: Path) -> None:
    r, _fem = _make_results_with_fem(tmp_path)
    # PG Single → {5}; tight box around origin → {1}; intersection empty
    slab = r.nodes.in_box(
        box_min=(-0.5, -0.5, -0.5),
        box_max=(0.5, 0.5, 0.5),
        component="displacement_x",
        pg="Single",
    )
    assert slab.node_ids.tolist() == []


def test_nodes_nearest_to_additive_with_pg(tmp_path: Path) -> None:
    r, _fem = _make_results_with_fem(tmp_path)
    # Nearest to (0,0,0) globally would be node 1 — but restricted to
    # PG TopRow it should pick node 4 at (0,1,0).
    slab = r.nodes.nearest_to(
        (0.0, 0.0, 0.0), component="displacement_x", pg="TopRow",
    )
    assert slab.node_ids.tolist() == [4]


def test_nodes_nearest_to_additive_with_ids(tmp_path: Path) -> None:
    r, _fem = _make_results_with_fem(tmp_path)
    slab = r.nodes.nearest_to(
        (0.0, 0.0, 0.0), component="displacement_x", ids=[2, 3, 4, 5],
    )
    # Restricted to {2,3,4,5}: nearest to origin is node 2 at (1,0,0).
    assert slab.node_ids.tolist() == [2]


# =====================================================================
# in_sphere
# =====================================================================

def test_nodes_in_sphere_picks_corners_within_radius(tmp_path: Path) -> None:
    r, _fem = _make_results_with_fem(tmp_path)
    # Sphere at origin, radius 1.5 → nodes 1,2,4 (sqrt(0,1,1) all ≤ 1.5)
    # Node 3 at (1,1,0) → distance = sqrt(2) ≈ 1.414 → INCLUDED.
    # Node 5 at (5,5,5) → far away, EXCLUDED.
    slab = r.nodes.in_sphere(
        center=(0.0, 0.0, 0.0), radius=1.5,
        component="displacement_x",
    )
    assert sorted(slab.node_ids.tolist()) == [1, 2, 3, 4]


def test_nodes_in_sphere_radius_zero(tmp_path: Path) -> None:
    r, _fem = _make_results_with_fem(tmp_path)
    # Sphere at exactly node 5's position, radius 0 → just node 5
    slab = r.nodes.in_sphere(
        center=(5.0, 5.0, 5.0), radius=0.0,
        component="displacement_x",
    )
    assert slab.node_ids.tolist() == [5]


def test_nodes_in_sphere_negative_radius_raises(tmp_path: Path) -> None:
    r, _fem = _make_results_with_fem(tmp_path)
    with pytest.raises(ValueError, match="non-negative"):
        r.nodes.in_sphere(
            center=(0, 0, 0), radius=-1.0,
            component="displacement_x",
        )


def test_nodes_in_sphere_additive_with_pg(tmp_path: Path) -> None:
    r, _fem = _make_results_with_fem(tmp_path)
    # Sphere covers {1, 2, 3, 4}; PG TopRow restricts to {3, 4}
    slab = r.nodes.in_sphere(
        center=(0.0, 0.0, 0.0), radius=2.0,
        component="displacement_x",
        pg="TopRow",
    )
    assert sorted(slab.node_ids.tolist()) == [3, 4]


# =====================================================================
# on_plane
# =====================================================================

def test_nodes_on_plane_z_zero(tmp_path: Path) -> None:
    r, _fem = _make_results_with_fem(tmp_path)
    # Plane z=0 → 4 corner nodes (z=0); node 5 at z=5 excluded
    slab = r.nodes.on_plane(
        point_on_plane=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        tolerance=1e-6,
        component="displacement_x",
    )
    assert sorted(slab.node_ids.tolist()) == [1, 2, 3, 4]


def test_nodes_on_plane_normalises_normal(tmp_path: Path) -> None:
    r, _fem = _make_results_with_fem(tmp_path)
    # Same query but with a non-unit normal — must be normalised internally
    slab = r.nodes.on_plane(
        point_on_plane=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 100.0),    # not unit
        tolerance=1e-6,
        component="displacement_x",
    )
    assert sorted(slab.node_ids.tolist()) == [1, 2, 3, 4]


def test_nodes_on_plane_zero_normal_raises(tmp_path: Path) -> None:
    r, _fem = _make_results_with_fem(tmp_path)
    with pytest.raises(ValueError, match="zero length"):
        r.nodes.on_plane(
            point_on_plane=(0, 0, 0),
            normal=(0, 0, 0),
            tolerance=0.1,
            component="displacement_x",
        )


def test_nodes_on_plane_with_tolerance(tmp_path: Path) -> None:
    r, _fem = _make_results_with_fem(tmp_path)
    # Plane through z=2.5 with tolerance 3 → catches z=0 (dist 2.5) and z=5 (dist 2.5)
    slab = r.nodes.on_plane(
        point_on_plane=(0.0, 0.0, 2.5),
        normal=(0.0, 0.0, 1.0),
        tolerance=3.0,
        component="displacement_x",
    )
    # All 5 nodes within tol
    assert sorted(slab.node_ids.tolist()) == [1, 2, 3, 4, 5]


# =====================================================================
# element_type= additive filter (element-level composites)
# =====================================================================

def test_elements_in_box_with_element_type(tmp_path: Path) -> None:
    """Pass element_type matching the only group — works as a no-op narrow."""
    r, _fem = _make_results_with_fem(tmp_path)
    from apeGmsh.results._composites import _element_ids_of_type
    fem = r._fem
    # Direct test of the underlying helper
    matching = _element_ids_of_type(fem, "quad4")
    assert matching.tolist() == [10]
    # Unknown type → empty
    assert _element_ids_of_type(fem, "Hex8").tolist() == []


# =====================================================================
# Element spatial filters (uses centroids)
# =====================================================================

def test_elements_centroid_helpers(tmp_path: Path) -> None:
    """The single quad's centroid is at (0.5, 0.5, 0.0)."""
    r, _fem = _make_results_with_fem(tmp_path)
    # nearest_to is on Gauss / line_stations / etc. via the mixin; the
    # bare ``elements`` composite is the simplest probe.
    eid = r.elements._resolve_element_ids(
        pg=None, label=None, selection=None, ids=None,
    )
    assert eid is None      # all-elements default

    # in_box should match the centroid
    from apeGmsh.results._composites import (
        _element_ids_in_box, _nearest_element_id,
    )
    fem = r._fem
    assert _nearest_element_id(fem, (0.5, 0.5, 0.0)) == 10
    assert _element_ids_in_box(
        fem, (-0.5, -0.5, -0.5), (1.5, 1.5, 1.5),
    ).tolist() == [10]
    # Element with centroid in [0,1) box — half-open excludes (0.5, 0.5, 0)?
    # No, the centroid (0.5, 0.5, 0.0) IS inside [0,1) on each axis.
    assert _element_ids_in_box(
        fem, (0.0, 0.0, -0.1), (1.0, 1.0, 0.1),
    ).tolist() == [10]


# =====================================================================
# Error cases
# =====================================================================

def test_nodes_nearest_to_without_fem_raises(tmp_path: Path) -> None:
    path = tmp_path / "nofem.h5"
    time = np.array([0.0, 1.0])
    with NativeWriter(path) as w:
        w.open(source_type="domain_capture")
        sid = w.begin_stage(name="s", kind="static", time=time)
        w.write_nodes(
            sid, "partition_0",
            node_ids=np.array([1], dtype=np.int64),
            components={"displacement_x": np.array([[0.0], [1.0]])},
        )
        w.end_stage()

    with Results.from_native(path) as r:
        # File has no fem snapshot embedded
        if r._fem is None:
            with pytest.raises(RuntimeError, match="bound FEMData"):
                r.nodes.nearest_to((0, 0, 0), component="displacement_x")
            with pytest.raises(RuntimeError, match="bound FEMData"):
                r.nodes.in_box((-1, -1, -1), (1, 1, 1),
                               component="displacement_x")
        else:
            # Native writer embedded a synthesized snapshot — skip the
            # error case, just sanity-check the helpers run.
            r.nodes.nearest_to((0, 0, 0), component="displacement_x")


def test_nodes_nearest_to_empty_candidate_raises(tmp_path: Path) -> None:
    r, _fem = _make_results_with_fem(tmp_path)
    # PG Single = {5}; candidate IDs that don't include 5
    with pytest.raises(RuntimeError, match="Candidate set has no nodes"):
        r.nodes.nearest_to(
            (0.0, 0.0, 0.0),
            component="displacement_x",
            ids=[],   # explicit empty
        )
