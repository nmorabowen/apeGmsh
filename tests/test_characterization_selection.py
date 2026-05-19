"""S0b — characterization battery for the selection / resolution unification.

These are **characterization tests**: every assertion pins the CURRENT
observed behavior on the untouched worktree — *including* the known
divergences and "bugs". They exist so that the deliberate behavior
changes in later phases (S1 resolver unification, S2 mesh-box
closed -> half-open, S3 chainable family, S5 import-origin fail-loud)
show up as a *reviewed pin-flip diff*, never as silent drift.

Plan: ``docs/plans/selection-unification.md`` (§6 S0b).

DO NOT "fix" a surprising assertion here. If a value looks wrong, that
is the point — it is reality on ``main`` today. The owning phase flips
the pin in its own commit and the diff is the decision record.

No ``openseespy`` dependency (curated no-openseespy CI gate): pure
apeGmsh + gmsh + numpy. One structured unit-box fem fixture is reused
across the spatial pins; it is deterministic (transfinite n=3 -> a
3x3x3 node lattice at coords {0.0, 0.5, 1.0}, 8 hex cells).

Each pin comments the exact source file:line it characterizes and the
S1/S2/S3/S5 risk it guards.
"""
from __future__ import annotations

import tempfile
import types

import gmsh
import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.mesh import _mesh_filters as _flt
from apeGmsh.results import _composites as _rc
from apeGmsh.results._composites import NodeResultsComposite
from apeGmsh.core.LoadsComposite import LoadsComposite
from apeGmsh.mesh.FEMData import FEMData, NodeComposite


# =====================================================================
# Fixtures — deterministic, tiny, reused
# =====================================================================

@pytest.fixture(scope="module")
def box_fem():
    """Structured unit cube: 3x3x3 node lattice, 8 hex8 cells.

    Transfinite ``n=3`` -> exactly 27 nodes at every combination of
    coords {0.0, 0.5, 1.0} and 8 hexahedra. Fully deterministic, so
    boundary-count pins are exact integers, not "about N".
    """
    g = apeGmsh(model_name="s0b_box", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                                 label="box")
        g.physical.add_volume("box", name="Body")
        g.mesh.structured.set_transfinite_box("box", n=3)
        g.mesh.generation.generate(dim=3)
        # mesh-selection sets used by item 2 (built BEFORE get_fem_data
        # so they are snapshotted into fem.mesh_selection).
        # selection-unification v2 P3-R: ``add_nodes`` / ``add_elements``
        # are removed (SC-11); the SAME half-open ``in_box`` set is
        # registered via the v2 ``select(...).in_box(...).save_as(name)``
        # (the half-open behaviour these item-2 pins characterise is
        # preserved on the v2 path — see ``test_mesh_selection_chain``).
        # ``filter_set`` is RETAINED and unchanged; it refines the
        # v2-built ``allbox`` set by tag exactly as before.
        g.mesh_selection.select().in_box(
            (0, 0, 0), (1, 1, 1)).save_as("allbox")
        n_tag = g.mesh_selection.get_tag(0, "allbox")
        g.mesh_selection.select(level="element", dim=3).in_box(
            (0, 0, 0), (1, 1, 1)).save_as("allelem")
        g.mesh_selection.select().in_box(
            (0, 0, 0), (0.5, 1, 1)).save_as("halfbox")
        g.mesh_selection.filter_set(
            0, n_tag, in_box=(0, 0, 0, 0.5, 1, 1), name="filtered")
        fem = g.mesh.queries.get_fem_data(dim=3)
    finally:
        g.end()
    return fem


@pytest.fixture(scope="module")
def two_pg_fem():
    """Unit cube with PGs ``A``/``B`` on two OCC box faces.

    Used by the multi-target ordering pin (item 5). The two faces of an
    OCC box do not share mesh nodes, so the union has no dedupe overlap
    here — ordering is still pinned by the per-name *block* order.
    """
    g = apeGmsh(model_name="s0b_twopg", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                                 label="box")
        faces = g.model.queries.boundary("box", dim=2)
        tags = [int(t) for _d, t in faces]
        g.physical.add_surface([tags[0]], name="A")
        g.physical.add_surface([tags[1]], name="B")
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data(dim=2)
    finally:
        g.end()
    return fem


@pytest.fixture(scope="module")
def shared_name_fem():
    """A points-only label and a same-named PG with elements.

    ``Shared`` is BOTH a dim-0 label (one embedded point, no elements)
    and a volume PG (821 tets). This is the exact scenario that
    exercises the node-vs-element resolver swallow asymmetry (item 3).
    """
    g = apeGmsh(model_name="s0b_shared", verbose=False)
    g.begin()
    try:
        vol = g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                                       label="box")
        pt = g.model.geometry.add_point(0.5, 0.5, 0.5)
        gmsh.model.mesh.embed(0, [int(pt)], 3, int(vol))
        g.physical.add_volume([int(vol)], name="Shared")     # PG
        g.labels.add(0, [int(pt)], "Shared")                 # label
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data(dim=3)
    finally:
        g.end()
    return fem


@pytest.fixture(scope="module")
def import_origin_fem():
    """An import-origin FEMData via ``FEMData.from_msh`` (item 7).

    ``_from_msh`` calls ``cls(nodes, elements, info)`` with no
    ``mesh_selection=`` kwarg, so ``fem.mesh_selection is None``
    (mesh/_fem_factory.py:516 + FEMData.__init__ default at
    mesh/FEMData.py:1142,1147).
    """
    g = apeGmsh(model_name="s0b_imp", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                                 label="box")
        g.physical.add_volume("box", name="Body")
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(dim=3)
        with tempfile.NamedTemporaryFile(
            suffix=".msh", delete=False
        ) as fh:
            msh_path = fh.name
        gmsh.write(msh_path)
    finally:
        g.end()
    return FEMData.from_msh(msh_path, dim=3)


def _selection_mixin(fem):
    """A real ``NodeResultsComposite`` over a stub ``_r`` (no reader).

    ``_SelectionMixin._resolve_node_ids`` only reads ``self._r._fem``
    (results/_composites.py:275, :295) — it never touches the reader
    for pg/label/selection resolution. So this exercises the *real*
    production resolver verbatim, with no results file / openseespy.
    """
    comp = NodeResultsComposite.__new__(NodeResultsComposite)
    comp._r = types.SimpleNamespace(_fem=fem)
    return comp


# =====================================================================
# Item 1 — BOX DIVERGENCE (the core S2 target)
# =====================================================================
#
# mesh box is CLOSED-CLOSED; results box is HALF-OPEN upper. They
# diverge on `main` today on a coord lying exactly on the upper face.
# S2 flips the mesh side to half-open (+ an `inclusive=` escape) and
# MUST update these two assertions in the same commit.

def test_item1_mesh_box_is_half_open_excludes_upper_face():
    # pin: src/apeGmsh/mesh/_mesh_filters.py nodes_in_box
    #   S2: reconciled closed->half-open (was `<= xmax/ymax/zmax`,
    #   now `< xmax/ymax/zmax`), matching results/_composites.py
    #   _node_ids_in_box. inclusive=True restores the old closed upper.
    coords = np.array(
        [
            [0.0, 0.0, 0.0],   # interior corner (on lower bound)
            [0.5, 0.5, 0.5],   # strictly interior
            [1.0, 1.0, 1.0],   # EXACTLY on the upper face (all axes)
            [1.0, 0.5, 0.5],   # x exactly on the upper bound
        ],
        dtype=np.float64,
    )
    bbox = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    mask = _flt.nodes_in_box(coords, bbox)
    # S2: reconciled closed->half-open (was [True, True, True, True]).
    # HALF-OPEN: the two on-upper-face points are now EXCLUDED.
    assert mask.tolist() == [True, True, False, False]
    # inclusive=True reproduces the OLD closed-closed result.
    mask_closed = _flt.nodes_in_box(coords, bbox, inclusive=True)
    assert mask_closed.tolist() == [True, True, True, True]


def test_item1_mesh_elements_in_box_inherits_half_open():
    # pin: src/apeGmsh/mesh/_mesh_filters.py elements_in_box
    #   elements_in_box delegates straight to nodes_in_box, so it
    #   inherits S2's closed->half-open flip (and the inclusive= escape).
    # S2: reconciled closed->half-open (was [True, True]).
    cent = np.array(
        [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]], dtype=np.float64
    )
    mask = _flt.elements_in_box(cent, (0.0, 0.0, 0.0, 1.0, 1.0, 1.0))
    # HALF-OPEN: the on-upper-face centroid is now EXCLUDED.
    assert mask.tolist() == [True, False]
    # inclusive=True reproduces the OLD closed result via the delegation.
    mask_closed = _flt.elements_in_box(
        cent, (0.0, 0.0, 0.0, 1.0, 1.0, 1.0), inclusive=True
    )
    assert mask_closed.tolist() == [True, True]


def test_item1_results_box_is_half_open_excludes_upper_face():
    # pin: src/apeGmsh/results/_composites.py:82 (_node_ids_in_box)
    #   mask = np.all((coords >= lo) & (coords < hi), axis=1)
    #   -> HALF-OPEN upper: a coord == hi is EXCLUDED.
    # why pinned: this is the *canonical* side (R4). S2 makes mesh
    #   match it; if results ever drifts to closed this pin fails.
    fem = types.SimpleNamespace()
    fem.nodes = types.SimpleNamespace(
        ids=np.array([10, 11, 12, 13], dtype=np.int64),
        coords=np.array(
            [
                [0.0, 0.0, 0.0],   # included
                [0.5, 0.5, 0.5],   # included
                [1.0, 1.0, 1.0],   # on upper face -> EXCLUDED
                [1.0, 0.5, 0.5],   # x on upper face -> EXCLUDED
            ],
            dtype=np.float64,
        ),
    )
    ids = _rc._node_ids_in_box(fem, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    # Only the two strictly-below-hi points survive.
    assert ids.tolist() == [10, 11]


def test_item1_results_element_box_is_half_open():
    # pin: src/apeGmsh/results/_composites.py:189 (_element_ids_in_box)
    #   mask = np.all((cent >= lo) & (cent < hi), axis=1)
    # why pinned: results element box is the half-open reference for
    #   S2; an on-face centroid must stay EXCLUDED.
    # Faithful stub: 3 nodes; element 100 is a 1-node "element" on
    # node id 1 (coord 0,0,0 -> centroid strictly inside); element
    # 200 is a 1-node "element" on node id 3 (coord exactly == hi ->
    # centroid on the upper face). _element_centroids maps
    # connectivity -> coords via searchsorted on sorted node ids.
    fem = types.SimpleNamespace()
    fem.nodes = types.SimpleNamespace(
        ids=np.array([1, 2, 3], dtype=np.int64),
        coords=np.array(
            [[0.0, 0.0, 0.0], [9.9, 9.9, 9.9], [0.5, 0.5, 0.5]],
            dtype=np.float64,
        ),
    )
    # P3-R / §6.3 M-STOP-3 + disposition 4: ``_element_ids_in_box``
    # funnels through ``_element_centroids``, which now iterates
    # ``fem.elements._groups.values()`` directly — one
    # ``ElementGroup``-shaped group carrying the SAME (ids, conn) the
    # legacy ``resolve`` returned.
    fem.elements = types.SimpleNamespace(
        types=[types.SimpleNamespace(name="P1")],
        resolve=lambda *, element_type: (
            np.array([100, 200], dtype=np.int64),
            np.array([[1], [3]], dtype=np.int64),
        ),
        _groups={0: types.SimpleNamespace(
            ids=np.array([100, 200], dtype=np.int64),
            connectivity=np.array([[1], [3]], dtype=np.int64),
            type_name="P1",
        )},
    )
    ids = _rc._element_ids_in_box(fem, (0.0, 0.0, 0.0), (0.5, 0.5, 0.5))
    # element 100 centroid (0,0,0) is strictly inside -> kept.
    # element 200 centroid (0.5,0.5,0.5) == hi -> EXCLUDED (half-open).
    assert ids.tolist() == [100]


def test_item1_mesh_and_results_box_reconciled_in_s2():
    # S2: reconciled closed->half-open. Pre-S2 this test asserted the
    # DIVERGENCE (mesh CLOSED includes the on-face point, results
    # HALF-OPEN excludes it -> mesh != results). After S2 the mesh box
    # matches the results box: identical box, identical on-face point,
    # SAME answer. Converted from divergence pin to a parity pin.
    # ref: _mesh_filters.py nodes_in_box (now half-open, was closed)
    #      == _composites.py:82 _node_ids_in_box (half-open reference).
    on_face = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
    bbox = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    mesh_in = bool(_flt.nodes_in_box(on_face, bbox)[0])
    fem = types.SimpleNamespace()
    fem.nodes = types.SimpleNamespace(
        ids=np.array([1], dtype=np.int64), coords=on_face,
    )
    results_in = _rc._node_ids_in_box(
        fem, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    ).size > 0
    assert mesh_in is False           # mesh: now HALF-OPEN (was True)
    assert results_in is False        # results: HALF-OPEN (unchanged)
    assert mesh_in == results_in      # PARITY (was the divergence)
    # The inclusive= escape restores the OLD divergent mesh answer.
    mesh_in_closed = bool(
        _flt.nodes_in_box(on_face, bbox, inclusive=True)[0]
    )
    assert mesh_in_closed is True     # pre-S2 closed behavior, on demand
    assert mesh_in_closed != results_in


# =====================================================================
# Item 2 — MESH-SELECTION BOUNDARY COUNTS (S2 flips these)
# =====================================================================
#
# A box whose upper face coincides with mesh nodes/centroids. The
# current (CLOSED) counts include the on-face nodes. S2 will reduce
# these and must update the numbers in the same commit.

def test_item2_total_lattice_is_deterministic(box_fem):
    # Sanity anchor for the boundary-count pins: transfinite n=3 ->
    # exactly 27 nodes at coords {0.0, 0.5, 1.0} and 8 hex cells.
    assert len(box_fem.nodes.ids) == 27
    xs = sorted(set(np.round(box_fem.nodes.coords[:, 0], 6).tolist()))
    assert xs == [0.0, 0.5, 1.0]
    # P3-R: ``fem.elements.resolve(element_type=)`` removed; the total
    # element count is the broker id universe (8 hex cells).
    assert len(box_fem.elements.ids) == 8


def test_item2_add_nodes_in_box_half_open_count(box_fem):
    # pin: src/apeGmsh/mesh/MeshSelectionSet.py add_nodes(in_box=)
    #   -> _flt.nodes_in_box. S2: reconciled closed->half-open (was 27).
    # The box (0,0,0)-(1,1,1) under half-open keeps only nodes with
    # x<1 & y<1 & z<1 -> each axis in {0.0, 0.5} -> 2*2*2 = 8.
    # The three upper-face planes (x==1, y==1, z==1) are now excluded.
    ids = box_fem.mesh_selection.node_ids("allbox")
    assert len(ids) == 8     # S2: was 27 (closed included all faces)
    # inclusive=True restores the OLD closed count (all 27 nodes).
    g = apeGmsh(model_name="s2_recover_allbox", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                                 label="box")
        g.physical.add_volume("box", name="Body")
        g.mesh.structured.set_transfinite_box("box", n=3)
        g.mesh.generation.generate(dim=3)
        g.mesh_selection.select().in_box(           # P3-R: was add_nodes
            (0, 0, 0), (1, 1, 1), inclusive=True).save_as("closed")
        fem2 = g.mesh.queries.get_fem_data(dim=3)
    finally:
        g.end()
    assert len(fem2.mesh_selection.node_ids("closed")) == 27


def test_item2_add_elements_in_box_half_open_count(box_fem):
    # pin: src/apeGmsh/mesh/MeshSelectionSet.py add_elements(in_box=)
    #   -> _flt.elements_in_box. S2: reconciled closed->half-open.
    #   The 8 hex centroids sit at {0.25, 0.75} per axis -> all
    #   strictly < 1, so the half-open box (0,0,0)-(1,1,1) still keeps
    #   all 8 (no centroid lies exactly on a face). Count is unchanged
    #   at 8; the value is pinned so an off-by-one half-open vs closed
    #   regression on an on-face centroid is still caught.
    ids = box_fem.mesh_selection.element_ids("allelem")
    assert len(ids) == 8     # S2: unchanged (no on-face centroid)
    # inclusive=True yields the same 8 here (no on-face centroid to
    # discriminate) -> proves the escape is wired without changing this
    # particular count.
    g = apeGmsh(model_name="s2_recover_allelem", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                                 label="box")
        g.physical.add_volume("box", name="Body")
        g.mesh.structured.set_transfinite_box("box", n=3)
        g.mesh.generation.generate(dim=3)
        g.mesh_selection.select(level="element", dim=3).in_box(
            (0, 0, 0), (1, 1, 1), inclusive=True   # P3-R: was add_elements
        ).save_as("closed")
        fem2 = g.mesh.queries.get_fem_data(dim=3)
    finally:
        g.end()
    assert len(fem2.mesh_selection.element_ids("closed")) == 8


def test_item2_add_nodes_half_box_excludes_split_plane(box_fem):
    # pin: src/apeGmsh/mesh/MeshSelectionSet.py add_nodes(in_box=)
    #   S2: reconciled closed->half-open (was 18).
    #   Box (0,0,0)-(0.5,1,1): under half-open the x==0.5 split plane
    #   is EXCLUDED (and the y==1, z==1 upper-face planes too), so only
    #   nodes with x<0.5 & y<1 & z<1 survive: x in {0.0}, y in
    #   {0.0,0.5}, z in {0.0,0.5} -> 1*2*2 = 4.
    #   (Pre-S2 closed kept the x==0.5 plane: 2*9 = 18.)
    ids = box_fem.mesh_selection.node_ids("halfbox")
    assert len(ids) == 4     # S2: was 18 (closed kept the x==0.5 plane)
    # inclusive=True restores the OLD closed count (18).
    g = apeGmsh(model_name="s2_recover_halfbox", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                                 label="box")
        g.physical.add_volume("box", name="Body")
        g.mesh.structured.set_transfinite_box("box", n=3)
        g.mesh.generation.generate(dim=3)
        g.mesh_selection.select().in_box(           # P3-R: was add_nodes
            (0, 0, 0), (0.5, 1, 1), inclusive=True).save_as("closed")
        fem2 = g.mesh.queries.get_fem_data(dim=3)
    finally:
        g.end()
    assert len(fem2.mesh_selection.node_ids("closed")) == 18


def test_item2_filter_set_in_box_half_open_count(box_fem):
    # pin: src/apeGmsh/mesh/MeshSelectionSet.py filter_set(in_box=)
    #   -> _flt.nodes_in_box. S2: reconciled closed->half-open (was 18).
    #   filter_set is the third caller of the box. Here the effect is
    #   CHAINED: `filtered` refines `allbox`, which is ITSELF now
    #   half-open (8 nodes: x,y,z in {0.0,0.5}). Refining that by
    #   (0,0,0)-(0.5,1,1) half-open keeps x<0.5 -> x in {0.0},
    #   y in {0.0,0.5}, z in {0.0,0.5} -> 1*2*2 = 4.
    #   (Pre-S2 both closed: allbox=27, refined to x==0.5 plane -> 18.)
    ids = box_fem.mesh_selection.node_ids("filtered")
    assert len(ids) == 4     # S2: was 18 (closed allbox + closed filter)
    # inclusive=True on BOTH the source build and the filter restores
    # the OLD closed count (allbox=27 -> filter keeps x<=0.5 -> 18).
    g = apeGmsh(model_name="s2_recover_filtered", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                                 label="box")
        g.physical.add_volume("box", name="Body")
        g.mesh.structured.set_transfinite_box("box", n=3)
        g.mesh.generation.generate(dim=3)
        # P3-R: ``add_nodes`` removed → v2 select(...).save_as; the
        # RETAINED ``filter_set`` then refines it by tag (unchanged).
        g.mesh_selection.select().in_box(
            (0, 0, 0), (1, 1, 1), inclusive=True).save_as("allbox_c")
        n_tag = g.mesh_selection.get_tag(0, "allbox_c")
        g.mesh_selection.filter_set(
            0, n_tag, in_box=(0, 0, 0, 0.5, 1, 1), name="filtered_c",
            inclusive=True)
        fem2 = g.mesh.queries.get_fem_data(dim=3)
    finally:
        g.end()
    assert len(fem2.mesh_selection.node_ids("filtered_c")) == 18


# =====================================================================
# Item 3 — FEMDATA RESOLVER SWALLOW ASYMMETRY (S1/S3 must preserve)
# =====================================================================
#
# Deliberate, documented: NODE path catches KeyError ONLY; ELEMENT
# path catches (KeyError, ValueError) and DOES fall through. S1
# unifies Loads+Masses only and must NOT touch FEMData; S3 reparenting
# must not regress this correctness invariant.

def test_item3_shared_name_setup_is_faithful(shared_name_fem):
    # Establishes the asymmetry-triggering scenario at source level:
    #  - label "Shared" is dim-0, points-only (1 node)
    #  - PG "Shared" is a volume with elements
    #  - labels.element_ids("Shared") raises ValueError
    # pin: src/apeGmsh/mesh/_group_set.py:247-249 (ValueError text)
    fem = shared_name_fem
    lab_nodes = np.asarray(fem.nodes.labels.node_ids("Shared"))
    pg_nodes = np.asarray(fem.nodes.physical.node_ids("Shared"))
    pg_elems = np.asarray(fem.elements.physical.element_ids("Shared"))
    assert len(lab_nodes) == 1                 # points-only label
    assert len(pg_nodes) > len(lab_nodes)      # PG is the big volume
    assert len(pg_elems) > 0                    # PG has elements
    with pytest.raises(ValueError, match="no element data"):
        fem.elements.labels.element_ids("Shared")


def test_item3_node_path_does_not_fall_through_to_same_named_pg(
    shared_name_fem,
):
    # pin: src/apeGmsh/mesh/FEMData.py:438-447 (_resolve_one_target)
    #   String path tries labels.node_ids first; it succeeds for the
    #   points-only label and returns it. The `except KeyError:` arms
    #   (:441, :446) are NARROW — only KeyError falls through, so a
    #   points-only label resolves to ITS points and never bleeds into
    #   the same-named PG. (Structurally a ValueError here would
    #   propagate, not silently shadow the PG — the documented intent.)
    # why pinned: S1 (Loads/Masses) must not touch this; S3 reparent
    #   must keep the node path's narrow KeyError-only catch.
    fem = shared_name_fem
    # P3-R: ``fem.nodes.get(target=)`` removed; ``.select(target=)``
    # reuses the SAME ``_resolve_nodes`` (P-NODE) so the narrow
    # KeyError-only swallow asymmetry under pin is preserved by reuse
    # (FP-4).  ``.result()`` is the NodeResult terminal.
    res = fem.nodes.select(target="Shared").result()
    lab_nodes = sorted(
        np.asarray(fem.nodes.labels.node_ids("Shared")).tolist()
    )
    pg_nodes = sorted(
        np.asarray(fem.nodes.physical.node_ids("Shared")).tolist()
    )
    got = sorted(np.asarray(res.ids).tolist())
    assert got == lab_nodes        # resolved the points-only LABEL
    assert got != pg_nodes         # did NOT fall through to the PG
    assert len(got) == 1


def test_item3_element_path_does_fall_through_swallowing_valueerror(
    shared_name_fem,
):
    # pin: src/apeGmsh/mesh/FEMData.py:828-835 (_resolve_one_elem_target)
    #   `except (KeyError, ValueError):` — labels.element_ids("Shared")
    #   raises ValueError (dim-0 group, no element data); it IS
    #   swallowed, so resolution falls through to the same-named PG,
    #   which HAS elements.
    # why pinned: this is the opposite arm of the asymmetry; S1 must
    #   not unify this away and S3 must not regress it to KeyError-only.
    fem = shared_name_fem
    # P3-R: ``fem.elements.get(target=)`` removed; ``.select(target=)``
    # reuses the SAME ``_resolve_elem_ids`` (P-ELEM-IDS) so the
    # (KeyError, ValueError) fall-through asymmetry under pin is
    # preserved by reuse.  ``.result()`` is the GroupResult terminal.
    res = fem.elements.select(target="Shared").result()
    pg_elems = sorted(
        np.asarray(fem.elements.physical.element_ids("Shared")).tolist()
    )
    got = sorted(np.asarray(res.ids).tolist())
    assert len(got) > 0
    assert got == pg_elems         # fell through to the PG's elements


# =====================================================================
# Item 4 — np.int64 DIMTAG accepted (S1/S3 must not regress)
# =====================================================================

def test_item4_is_dimtag_tuple_accepts_numpy_integer():
    # pin: src/apeGmsh/mesh/FEMData.py:399-406
    #   `@staticmethod NodeComposite._is_dimtag_tuple`:
    #   `isinstance(v, (int, np.integer))` — np.int64/np.int32 scalars
    #   ARE accepted as a DimTag pair, identically to Python int. It is
    #   the single predicate the node path (:421) and element path
    #   (:815, via NodeComposite._is_dimtag_tuple) both route on.
    # why pinned: this predicate is the EXACT S1/S3 regression risk —
    #   a refactor that narrows the isinstance to bare `int` would
    #   silently break every caller that iterates a numpy tag array.
    #   Pure pin (no Gmsh session needed): it is the routing decision.
    assert NodeComposite._is_dimtag_tuple((3, 1)) is True
    assert NodeComposite._is_dimtag_tuple((np.int64(3), np.int64(1))) is True
    assert NodeComposite._is_dimtag_tuple((np.int32(2), np.int32(7))) is True
    # negative controls: not a 2-tuple of integers
    assert NodeComposite._is_dimtag_tuple((3, 1, 0)) is False
    assert NodeComposite._is_dimtag_tuple("Body") is False
    assert NodeComposite._is_dimtag_tuple((3.0, 1.0)) is False


def test_item4_np_int64_dimtag_resolves_same_as_python_int():
    # pin: src/apeGmsh/mesh/FEMData.py:421-427 + :459-474
    #   A (np.int64, np.int64) target is routed through
    #   _nodes_on_dimtag exactly like a Python-int (d, t) and yields
    #   the SAME node set.
    # why pinned: end-to-end proof the numpy-scalar DimTag is not just
    #   accepted by the predicate but resolves correctly. NOTE: the
    #   DimTag path calls LIVE gmsh (_nodes_on_dimtag, :461-466), so
    #   this MUST run inside a still-open session — a detached
    #   (module-scoped) fem would raise RuntimeError("Gmsh session may
    #   have been closed"). Hence an inline session, queried before
    #   .end().
    g = apeGmsh(model_name="s0b_dimtag", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                                 label="box")
        g.physical.add_volume("box", name="Body")
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data(dim=3)
        # P3-R: ``.get(target=)`` removed → ``.select(target=).result()``
        # (same _resolve_nodes / _nodes_on_dimtag path, P-NODE).
        got_int = fem.nodes.select(target=(int(3), int(1))).result()
        got_np = fem.nodes.select(
            target=(np.int64(3), np.int64(1))).result()
    finally:
        g.end()
    assert len(got_int.ids) > 0
    assert len(got_np.ids) == len(got_int.ids)
    assert np.array_equal(
        np.sort(np.asarray(got_np.ids)),
        np.sort(np.asarray(got_int.ids)),
    )


# =====================================================================
# Item 5 — MULTI-TARGET UNION ORDERING (broker insertion vs results sorted)
# =====================================================================
#
# Broker dedupes preserving first-occurrence INSERTION order; results
# unions via np.unique -> SORTED. S1/S3 must not silently reorder the
# broker; the results sorted order is the other half of the divergence.

def test_item5_broker_multi_target_preserves_insertion_order(
    two_pg_fem,
):
    # pin: src/apeGmsh/mesh/FEMData.py:507-511 (_dedupe_node_parts)
    #   np.unique(..., return_index=True); unique_idx.sort()  ->
    #   concatenation order with first-occurrence kept (INSERTION).
    # why pinned: S1/S3 must not reorder broker multi-target output.
    fem = two_pg_fem
    # P3-R: ``fem.nodes.get(target=)`` removed; ``.select(target=)``
    # reuses the SAME multi-target dedupe (``_dedupe_node_parts``,
    # P-NODE) so the first-occurrence INSERTION order under pin is
    # preserved by reuse.  ``.result().ids`` is the NodeResult ids.
    a = np.asarray(fem.nodes.select(target="A").result().ids).tolist()
    b = np.asarray(fem.nodes.select(target="B").result().ids).tolist()
    ab = np.asarray(
        fem.nodes.select(target=["A", "B"]).result().ids).tolist()
    ba = np.asarray(
        fem.nodes.select(target=["B", "A"]).result().ids).tolist()
    # exact insertion-order dedupe of A-block then B-block
    assert ab == list(dict.fromkeys(a + b))
    assert ba == list(dict.fromkeys(b + a))
    # order is target-order dependent and NOT globally sorted
    assert ab != ba
    assert ab != sorted(ab)


def test_item5_results_multi_target_union_is_sorted(two_pg_fem):
    # pin: src/apeGmsh/results/_composites.py:306
    #   return np.unique(np.concatenate(all_ids))  -> SORTED, and
    #   therefore order-INDEPENDENT of the pg= list order.
    # why pinned: this is the contrast to the broker insertion order;
    #   S3 unification must keep both behaviors distinguishable (the
    #   plan does NOT collapse them).
    comp = _selection_mixin(two_pg_fem)
    ab = comp._resolve_node_ids(
        pg=["A", "B"], label=None, selection=None, ids=None
    ).tolist()
    ba = comp._resolve_node_ids(
        pg=["B", "A"], label=None, selection=None, ids=None
    ).tolist()
    assert ab == sorted(ab)        # SORTED
    assert ab == ba                # order-independent (unlike broker)


# =====================================================================
# Item 6 — VIZ physical= vs label= COLLISION
# =====================================================================
#
# selection-unification-v2 P3-R: ``test_item6_viz_physical_filter_is_
# pg_only_ignores_same_named_label`` is **RETIRED** (deleted).  It
# pinned ``g.model.selection.select_volumes(physical=/labels=)`` — the
# ``SelectionComposite._filter_by_identity`` filter grammar — which is
# **removed** with **no v2 successor** (``EntitySelection`` has only
# spatial verbs + set-ops + ``to_label``/``to_physical``/``to_dataframe``;
# the rich ``physical=``/``labels=`` filter grammar has no replacement).
# This is the SC-12 precedent / §6.3 M-NOTE-G7-cascade disposition 3 —
# a removed-surface capability with no v2 successor → a P4-documented
# capability gap, not papered over (sibling of the deleted
# ``tests/test_selection_filters.py``).


# =====================================================================
# Item 7 — selection= ON IMPORT-ORIGIN FEM (S5 makes these fail loud)
# =====================================================================
#
# import-origin fem has mesh_selection=None. Pin the CURRENT behavior
# at each reachable consumer:
#  - results _resolve_node_ids: raises RuntimeError (loud already).
#  - loads _target_nodes __ms__ arm: returns set() silently.
# S5 will make the silent-empty path fail loud; the RuntimeError pin
# guards that the already-loud path stays loud.

def test_item7_import_origin_fem_has_no_mesh_selection(
    import_origin_fem,
):
    # pin: src/apeGmsh/mesh/_fem_factory.py:516 (_from_msh returns
    #   cls(nodes, elements, info) — no mesh_selection=) +
    #   src/apeGmsh/mesh/FEMData.py:1142,1147 (default None).
    # why pinned: S5's fail-loud guard keys off exactly this None.
    assert import_origin_fem.mesh_selection is None


def test_item7_results_selection_on_import_origin_raises_runtimeerror(
    import_origin_fem,
):
    # pin: src/apeGmsh/results/_composites.py:294-301 (_resolve_node_ids)
    #   `if store is None: raise RuntimeError("selection= requires
    #   fem.mesh_selection to be present ...")`.
    # why pinned: this consumer is ALREADY loud. S5 must keep it loud;
    #   pin the RuntimeError so an S5 refactor can't downgrade it to a
    #   silent empty to "match" the loads path.
    comp = _selection_mixin(import_origin_fem)
    with pytest.raises(RuntimeError, match="mesh_selection to be present"):
        comp._resolve_node_ids(
            pg=None, label=None, selection="anything", ids=None
        )


def test_item7_loads_ms_sentinel_missing_set_raises():
    # S5: silent return set() -> fail-loud raise.
    #   FLIPPED PIN (was ..._returns_empty_silently, which asserted
    #   out == set()). src/apeGmsh/core/LoadsComposite.py _target_nodes
    #   __ms__ arm now: if info is None: raise KeyError(
    #   "...Refusing to silently bind this load to zero nodes
    #   (fail loud).")  -- was a silent `return set()`.
    #
    # LIMITATION (unchanged): LoadsComposite resolves against the live
    # SESSION's `_parent.mesh_selection`, NOT a FEMData. This drives
    # the literal __ms__ missing-set arm at unit level (the behavior
    # S5 converts to fail-loud), not an import-origin end-to-end path.
    class _MissingSets(dict):
        def get(self, _key, _default=None):   # always "set lost"
            return None

    ms = types.SimpleNamespace()
    # _resolve_target iterates ms._sets.items() to match the name to a
    # sentinel ("__ms__", dim, tag); .items() must yield the match.
    ms._sets = _MissingSets()
    ms._sets[(0, 99)] = {"name": "ghost", "node_ids": [1, 2, 3]}
    parent = types.SimpleNamespace(mesh_selection=ms, parts=None)

    lc = LoadsComposite.__new__(LoadsComposite)
    lc._parent = parent

    with pytest.raises(KeyError, match="Refusing to silently bind"):
        lc._target_nodes(
            "ghost", node_map=None, all_nodes=None, source="auto"
        )
