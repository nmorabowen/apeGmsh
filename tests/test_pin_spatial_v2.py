"""P0-C — spatial-kernel characterization pins (v2 plan, HT7).

These are **characterization tests** in the exact sense of the v1 S0b
battery (``tests/test_characterization_selection.py``): every assertion
pins the CURRENT observed behavior on the untouched worktree —
*including* the known silent-corruption path. They exist so that the
deliberate spatial-kernel unification in **P3** ("Unify the 6 spatial
copies -> `_kernel/spatial.py` as pinned reviewed flips") shows up as a
*reviewed pin-flip diff*, never as silent drift.

Plan: ``docs/plans/selection-unification-v2.md`` (§3 HT7 row, §6 P0-C).

HT7 (verbatim from the plan §3 table):
    spatial = **6 copies with real divergence** — axis-aligned
    ``nodes_on_plane(coords,axis,value,atol)`` vs ``(point,normal,tol)``;
    silent row-0 centroid bug. Source: ``mesh/_mesh_filters.py:23-42``,
    ``:137-162`` (``:159`` ``.get(nid,0)``).
    Disposition: Unify via **pinned reviewed flips** (P3).

DO NOT "fix" a surprising assertion here. The row-0 centroid is
*wrong on purpose* in pin 2 — that is the point: it is reality on HEAD
today, and P3 flips the pin in its own commit so the diff is the
decision record.

No ``openseespy`` dependency (curated no-openseespy CI gate): pure
apeGmsh + gmsh + numpy. The mesh pin reuses the same deterministic
unit-cube fixture idiom as the v1 battery (transfinite ``n=3`` -> a
3x3x3 node lattice at coords {0.0, 0.5, 1.0}, 8 hex cells).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.mesh import _mesh_filters as _flt


# =====================================================================
# Fixture — deterministic, tiny (same idiom as v1 S0b ``box_fem``)
# =====================================================================

@pytest.fixture(scope="module")
def plane_box_fem():
    """Structured unit cube: 3x3x3 node lattice, 8 hex8 cells.

    Transfinite ``n=3`` -> exactly 27 nodes at every combination of
    coords {0.0, 0.5, 1.0}. The ``ZPlane`` mesh-selection set is built
    BEFORE ``get_fem_data`` so it is snapshotted into
    ``fem.mesh_selection`` (identical pattern to v1 ``box_fem``).
    """
    g = apeGmsh(model_name="pin_v2_plane_box", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                                 label="box")
        g.physical.add_volume("box", name="Body")
        g.mesh.structured.set_transfinite_box("box", n=3)
        g.mesh.generation.generate(dim=3)
        # legacy axis-aligned API: on_plane=(axis, value, atol)
        g.mesh_selection.add_nodes(
            on_plane=("z", 0.0, 1e-3), name="ZPlane")
        fem = g.mesh.queries.get_fem_data(dim=3)
    finally:
        g.end()
    return fem


# =====================================================================
# Pin 1 — legacy on_plane API is AXIS-ALIGNED (axis,value,atol) (HT7)
# =====================================================================

def test_pin_meshselection_on_plane_is_axis_aligned(plane_box_fem):
    """Legacy ``g.mesh_selection.add_nodes(on_plane=...)`` is axis-aligned.

    Characterizes: ``src/apeGmsh/mesh/_mesh_filters.py:23-42``
    ``nodes_on_plane(coords, axis, value, atol)`` — an AXIS-ALIGNED
    kernel with signature ``(axis, value, atol)``, dispatched from
    ``src/apeGmsh/mesh/MeshSelectionSet.py:236-239`` (``on_plane[0]`` is
    an axis name, ``on_plane[1]`` the plane value, ``on_plane[2]`` the
    abs-tolerance). This is STRUCTURALLY DISTINCT from the chain's
    arbitrary ``(point, normal, tol)`` plane predicate.

    HT7: "axis-aligned ``nodes_on_plane(coords,axis,value,atol)`` vs
    ``(point,normal,tol)``" — plan §3 table.

    FLIPS IN: P3 (the plane-kernel unification into
    ``_kernel/spatial.py``; the legacy ``(axis,value,atol)`` form
    becoming the unified ``(point,normal,tol)`` form is a visible,
    reviewed decision, not silent drift).

    Pinned (observed on HEAD, 3x3x3 transfinite unit cube): the
    ``("z", 0.0, 1e-3)`` set is EXACTLY the lattice nodes with
    |z| <= 1e-3 — computed independently from the mesh coords and
    asserted equal to the observed selection.
    """
    fem = plane_box_fem
    all_ids = np.asarray(fem.nodes.ids)
    all_coords = np.asarray(fem.nodes.coords)

    # Sanity anchor (same idiom as v1 test_item2_total_lattice...):
    # transfinite n=3 -> 27 nodes, z planes at {0.0, 0.5, 1.0}.
    assert len(all_ids) == 27
    zs = sorted(set(np.round(all_coords[:, 2], 6).tolist()))
    assert zs == [0.0, 0.5, 1.0]

    observed = np.sort(np.asarray(fem.mesh_selection.node_ids("ZPlane")))

    # Expected set computed INDEPENDENTLY from the mesh coords using
    # the axis-aligned predicate the legacy path is pinned to use:
    # the z-column within atol of the plane value 0.0.
    expected_mask = np.abs(all_coords[:, 2] - 0.0) <= 1e-3
    expected_ids = np.sort(all_ids[expected_mask])

    # (a) exact id-set equality: legacy on_plane == axis-aligned |z|<=atol
    assert observed.tolist() == expected_ids.tolist()
    # The exact node ids on HEAD for this deterministic mesh (9 nodes,
    # the full z==0.0 lattice face). Pinned literally so a half-open /
    # generalized-plane regression in P3 is a visible diff, not a count.
    assert observed.tolist() == [2, 4, 6, 8, 12, 16, 17, 19, 25]
    assert len(observed) == 9

    # (b) the API shape itself is the axis-tuple ``(axis, value, atol)``,
    #     NOT ``(point, normal, tol)``. Re-issue the call form to lock
    #     that a 3-tuple whose [0] is an AXIS NAME (not a 3-vector point)
    #     is what selects this plane. A (point, normal, tol) caller would
    #     pass on_plane[0] as a length-3 sequence; the legacy kernel
    #     instead routes on_plane[0]="z" through _axis_index.
    g2 = apeGmsh(model_name="pin_v2_plane_axisform", verbose=False)
    g2.begin()
    try:
        g2.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                                  label="box")
        g2.physical.add_volume("box", name="Body")
        g2.mesh.structured.set_transfinite_box("box", n=3)
        g2.mesh.generation.generate(dim=3)
        # axis name as a STRING in slot 0 -> only the axis-aligned
        # signature accepts this; a (point,normal,tol) kernel could not.
        g2.mesh_selection.add_nodes(
            on_plane=("z", 0.0, 1e-3), name="ZPlane2")
        fem2 = g2.mesh.queries.get_fem_data(dim=3)
    finally:
        g2.end()
    again = np.sort(np.asarray(fem2.mesh_selection.node_ids("ZPlane2")))
    assert again.tolist() == [2, 4, 6, 8, 12, 16, 17, 19, 25]

    # Direct kernel call: the pinned signature is positionally
    # (coords, axis, value, atol) — axis is a NAME, value a scalar.
    direct = _flt.nodes_on_plane(all_coords, "z", 0.0, 1e-3)
    assert np.array_equal(direct, expected_mask)
    assert np.sort(all_ids[direct]).tolist() == [
        2, 4, 6, 8, 12, 16, 17, 19, 25
    ]


# =====================================================================
# Pin 2 — element_centroids SILENTLY maps a missing node id to ROW 0
# =====================================================================

def test_pin_meshfilters_element_centroids_silent_row0():
    """``element_centroids`` silently corrupts (no raise) on a missing id.

    Characterizes: ``src/apeGmsh/mesh/_mesh_filters.py:137-162``
    ``element_centroids(connectivity, node_id_to_idx, node_coords)``.
    Line ``:159`` is::

        idx = np.array([node_id_to_idx.get(int(nid), 0) for nid in col_ids])

    A node id ABSENT from ``node_id_to_idx`` falls back to ``0`` via
    ``dict.get(..., 0)`` -> it is silently read as ROW 0 of
    ``node_coords``, producing a CORRUPTED centroid instead of raising.

    HT7: "silent row-0 centroid bug" — plan §3 table, ``:159``
    ``.get(nid,0)``.

    FLIPS IN: P3 (spatial dedup into ``_kernel/spatial.py`` makes the
    centroid path fail-loud; this pin's silent-wrong value becomes a
    raised error — a reviewed same-commit pin-flip, not silent drift).

    Crafted-array characterization (testing the internal numeric helper
    directly is more robust than coaxing a real mesh into the corrupt
    state). Setup:
      - node-id -> row map: {10:0, 20:1, 30:2}
      - node_coords rows:    (0,0,0), (10,10,10), (20,20,20)
      - element 0: nodes (20, 30)            -> centroid (15,15,15)
      - element 1: nodes (777 MISSING, 30)
            777 -> .get(777, 0) -> ROW 0 -> (0,0,0)
            centroid = ((0,0,0) + (20,20,20)) / 2 = (10,10,10)  [WRONG]

    The element-1 centroid is the row-0-contaminated value, NOT a
    correct centroid for node 777. Pinned EXACTLY; no exception, no
    warning is raised on HEAD.
    """
    # row 0 deliberately (0,0,0) so the contamination is unmistakable
    # and DISTINCT from any legitimate centroid in this setup.
    node_coords = np.array(
        [
            [0.0, 0.0, 0.0],     # row 0  (id 10) — the silent fallback
            [10.0, 10.0, 10.0],  # row 1  (id 20)
            [20.0, 20.0, 20.0],  # row 2  (id 30)
        ],
        dtype=np.float64,
    )
    node_id_to_idx = {10: 0, 20: 1, 30: 2}
    # element 1 references id 777, which is ABSENT from the map.
    connectivity = np.array(
        [
            [20, 30],    # both present -> (15,15,15)
            [777, 30],   # 777 missing -> .get(777,0) -> row 0 -> WRONG
        ],
        dtype=np.int64,
    )

    raised = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            centroids = _flt.element_centroids(
                connectivity, node_id_to_idx, node_coords
            )
        except Exception as exc:  # pragma: no cover - pin asserts NO raise
            raised = exc

    # (1) NO exception on HEAD — the missing id is swallowed silently.
    assert raised is None, (
        f"HEAD must NOT raise on a missing node id (silent row-0 bug); "
        f"got {raised!r}"
    )
    # (2) NO warning either — fully silent.
    assert len(caught) == 0

    # (3) Exact corrupted output. Element 0 is correct; element 1 is
    #     the ROW-0-contaminated centroid (NOT node 777's real coord).
    expected = np.array(
        [
            [15.0, 15.0, 15.0],   # element 0: legitimate
            [10.0, 10.0, 10.0],   # element 1: CORRUPTED via row-0 fallback
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(centroids, expected)

    # (4) The corruption is real, not a coincidence: the contaminated
    #     row-1 centroid is the row-0-blended value and is DISTINCT
    #     from element 0's centroid. P3's fail-loud flip turns the
    #     (3)/(1) pins from "wrong+silent" into "raises".
    assert centroids[1].tolist() == [10.0, 10.0, 10.0]
    assert centroids[1].tolist() != centroids[0].tolist()
    # It equals exactly ((row0) + (id30 coord)) / 2 — the silent blend.
    assert np.array_equal(
        centroids[1],
        (node_coords[0] + node_coords[2]) / 2.0,
    )
