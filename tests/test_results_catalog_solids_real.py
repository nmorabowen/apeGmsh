"""Phase 11a catalog growth — parametrized real-openseespy validation.

For every continuum-solid class in the catalog, build the simplest
possible single-element model under uniform compression, run a static
analysis through real openseespy, capture stress via DomainCapture,
and check that the recovered components have the expected shape and
satisfy a basic physical sanity (negative σ_zz under z-compression,
zero shear stresses, σ_xx == σ_yy by symmetry).

Cross-validates that each catalogued ``(class_name, int_rule, "stress")``
entry agrees with what OpenSees actually emits.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

openseespy = pytest.importorskip(
    "openseespy.opensees", reason="openseespy required",
)
ops = openseespy

from apeGmsh.solvers._element_response import IntRule, lookup


# =====================================================================
# Per-class minimal model factories
# =====================================================================
#
# Each factory returns ``(node_ids, coords, element_id, class_name)``
# and creates the relevant element in the active ops domain. The
# caller fixes the bottom face, applies a vertical load on the top
# face, and runs a static analysis.

def _build_unit_cube_8node(class_name: str) -> tuple[
    np.ndarray, np.ndarray, int, str,
]:
    """8-node hex unit cube (Brick / BbarBrick / SSPbrick)."""
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=np.float64)
    node_ids = np.arange(1, 9, dtype=np.int64)
    for nid, xyz in zip(node_ids, coords):
        ops.node(int(nid), float(xyz[0]), float(xyz[1]), float(xyz[2]))
    # Pin the bottom face (z = 0).
    for nid in (1, 2, 3, 4):
        ops.fix(int(nid), 1, 1, 1)
    ops.element(class_name, 1, *(int(n) for n in node_ids), 1)
    return node_ids, coords, 1, class_name


def _build_tet_4node() -> tuple[np.ndarray, np.ndarray, int, str]:
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 1.0],
    ], dtype=np.float64)
    node_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    for nid, xyz in zip(node_ids, coords):
        ops.node(int(nid), float(xyz[0]), float(xyz[1]), float(xyz[2]))
    for nid in (1, 2, 3):
        ops.fix(int(nid), 1, 1, 1)
    ops.element("FourNodeTetrahedron", 1, 1, 2, 3, 4, 1)
    return node_ids, coords, 1, "FourNodeTetrahedron"


def _build_unit_cube_20node() -> tuple[np.ndarray, np.ndarray, int, str]:
    """20-node serendipity hex (Twenty_Node_Brick).

    Node order: 8 corners (same as 8-node hex), then 12 edge midpoints
    (bottom-face edges 1-2, 2-3, 3-4, 4-1; vertical edges 1-5, 2-6,
    3-7, 4-8; top-face edges 5-6, 6-7, 7-8, 8-5).
    """
    corners = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=np.float64)
    # Edge midpoint helper.
    def mid(a: int, b: int) -> np.ndarray:
        return 0.5 * (corners[a] + corners[b])
    edge_pairs = [
        (0, 1), (1, 2), (2, 3), (3, 0),   # bottom face
        (0, 4), (1, 5), (2, 6), (3, 7),   # vertical edges
        (4, 5), (5, 6), (6, 7), (7, 4),   # top face
    ]
    midpts = np.array([mid(a, b) for (a, b) in edge_pairs])
    coords = np.vstack([corners, midpts])
    node_ids = np.arange(1, 21, dtype=np.int64)
    for nid, xyz in zip(node_ids, coords):
        ops.node(int(nid), float(xyz[0]), float(xyz[1]), float(xyz[2]))
    # Pin all bottom-face nodes (z = 0): corners 1-4, plus the 4
    # bottom-face edge midpoints (nodes 9-12) and the 4 vertical-edge
    # midpoint that sits at z=0 — only the bottom-face ones (9, 10, 11, 12).
    bottom_face_nids = [1, 2, 3, 4, 9, 10, 11, 12]
    for nid in bottom_face_nids:
        ops.fix(int(nid), 1, 1, 1)
    ops.element(
        "20NodeBrick", 1, *(int(n) for n in node_ids), 1,
    )
    return node_ids, coords, 1, "Twenty_Node_Brick"


def _build_tet_10node() -> tuple[np.ndarray, np.ndarray, int, str]:
    # Corners + edge midpoints of the same 4-node tet.
    corners = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.5, 0.5, 1.0],
    ], dtype=np.float64)
    # Edge midpoints (1-2, 2-3, 3-1, 1-4, 2-4, 3-4).
    midpts = np.array([
        0.5 * (corners[0] + corners[1]),
        0.5 * (corners[1] + corners[2]),
        0.5 * (corners[2] + corners[0]),
        0.5 * (corners[0] + corners[3]),
        0.5 * (corners[1] + corners[3]),
        0.5 * (corners[2] + corners[3]),
    ])
    coords = np.vstack([corners, midpts])
    node_ids = np.arange(1, 11, dtype=np.int64)
    for nid, xyz in zip(node_ids, coords):
        ops.node(int(nid), float(xyz[0]), float(xyz[1]), float(xyz[2]))
    # Pin all base-face nodes (corners 1, 2, 3 plus their midpoints).
    for nid in (1, 2, 3, 5, 6, 7):
        ops.fix(int(nid), 1, 1, 1)
    ops.element("TenNodeTetrahedron", 1, *(int(n) for n in node_ids), 1)
    return node_ids, coords, 1, "TenNodeTetrahedron"


# =====================================================================
# Mock fem (only needs nodes + a hash for DomainCapture)
# =====================================================================

class _MinimalFem:
    def __init__(self, node_ids: np.ndarray, coords: np.ndarray) -> None:
        self.nodes = SimpleNamespace(ids=node_ids, coords=coords)
        self.elements = []

    @property
    def snapshot_id(self) -> str:
        from apeGmsh.mesh._femdata_hash import compute_snapshot_id
        return compute_snapshot_id(self)

    def to_native_h5(self, group) -> None:
        group.attrs["snapshot_id"] = self.snapshot_id
        group.attrs["ndm"] = 3
        group.attrs["ndf"] = 3
        group.attrs["model_name"] = ""
        group.attrs["units"] = ""
        n = group.create_group("nodes")
        n.create_dataset("ids", data=self.nodes.ids)
        n.create_dataset("coords", data=self.nodes.coords)
        group.create_group("elements")


def _vertical_load_on_top_nodes(top_nids: list[int], total_load: float) -> None:
    """Distribute a vertical load equally across the listed top-face nodes."""
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    per_node = total_load / len(top_nids)
    for nid in top_nids:
        ops.load(int(nid), 0.0, 0.0, per_node)


def _solve_static_step() -> None:
    ops.system("ProfileSPD")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    rc = ops.analyze(1)
    assert rc == 0, f"analyze() returned {rc}"


# =====================================================================
# Per-class fixtures
# =====================================================================
#
# Each parametrize entry provides:
#   (class_name, int_rule, n_gp_expected, model_factory, top_nids,
#    expected_szz_sign)

_CASES = [
    pytest.param(
        "FourNodeTetrahedron", IntRule.Tet_GL_1, 1,
        _build_tet_4node, [4], -1.0,
        id="FourNodeTetrahedron",
    ),
    # TenNodeTetrahedron skipped: upstream OpenSees bug —
    # ``TenNodeTetrahedron::getResponse`` (line ~1845 of
    # ``SRC/element/tetrahedron/TenNodeTetrahedron.cpp``) declares
    # ``static Vector stresses(6)`` but writes 24 floats (4 GPs × 6),
    # causing heap corruption and returning only 6 values to
    # ``ops.eleResponse(eid, "stresses")``. MPCO works around this
    # through material-level probing, so the catalog entry is correct
    # for the MPCO path; DomainCapture and the .out transcoder need
    # the upstream fix (``static Vector stresses(24)``) before they
    # can record TenNodeTet stress reliably.
    pytest.param(
        "Brick", IntRule.Hex_GL_2, 8,
        lambda: _build_unit_cube_8node("stdBrick"), [5, 6, 7, 8], -1.0,
        id="Brick",
    ),
    # BbarBrick skipped: in this OpenSees build, BbarBrick's
    # ``ops.eleResponse(eid, "stresses")`` returns 48 zeros even
    # though the analysis converges and node displacements are correct
    # (resisting forces and ``"strains"`` come back populated). The
    # element-level "stresses" path is broken on BbarBrick; MPCO works
    # because it probes materials directly. The catalog entry is
    # correct for the MPCO path; DomainCapture and the .out transcoder
    # need the upstream behavior fixed before they can record
    # BbarBrick stress.
    pytest.param(
        "SSPbrick", IntRule.Hex_GL_1, 1,
        lambda: _build_unit_cube_8node("SSPbrick"), [5, 6, 7, 8], -1.0,
        id="SSPbrick",
    ),
    # Twenty_Node_Brick skipped: the OpenSees 20-node serendipity hex
    # uses a node-ordering convention that (a) is not documented in
    # the source and (b) triggers a non-positive Jacobian + ``exit(-1)``
    # in ``Twenty_Node_Brick::Jacobian3d`` (line ~1995) when the
    # ordering doesn't match — which terminates the openseespy
    # process and hangs pytest. The catalog entry is structural fact
    # (27 GPs at the corner-edge-face-centroid positions are absolute
    # coordinates, independent of the user's node ordering); MPCO
    # reads it correctly because it follows the user's
    # ``connectedExternalNodes`` order. End-to-end validation in this
    # test would require figuring out the exact node ordering
    # convention; that's a side quest, not a catalog correctness bug.
]


@pytest.mark.parametrize(
    "cpp_class,int_rule,n_gp_expected,build_model,top_nids,szz_sign", _CASES,
)
def test_real_capture_for_class(
    tmp_path: Path,
    cpp_class: str,
    int_rule: int,
    n_gp_expected: int,
    build_model,
    top_nids: list[int],
    szz_sign: float,
) -> None:
    """Static compression: capture stress and verify the catalog matches reality.

    ``cpp_class`` is the OpenSees ``getClassType()`` name (also the
    catalog key and ``ops.eleType`` return). The Tcl element name used
    to construct the element may differ (``stdBrick`` → ``Brick``,
    ``bbarBrick`` → ``BbarBrick``) — that's the build_model factory's
    job to pass.
    """
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 3)

    # Material must exist before any element references it.
    ops.nDMaterial("ElasticIsotropic", 1, 200e9, 0.3)

    node_ids, coords, eid, factory_class_name = build_model()
    fem = _MinimalFem(node_ids=node_ids, coords=coords)

    _vertical_load_on_top_nodes(top_nids, total_load=-1.0e6)

    # Sanity: ops.eleType returns the C++ class name (catalog key).
    assert ops.eleType(eid) == cpp_class, (
        f"build_model registered element under Tcl name "
        f"{factory_class_name!r}; expected ops.eleType={cpp_class!r}, "
        f"got {ops.eleType(eid)!r}."
    )

    # Catalog flat size must match what the element actually emits.
    layout = lookup(cpp_class, int_rule, "stress")
    assert layout.n_gauss_points == n_gp_expected

    _solve_static_step()
    flat = np.asarray(ops.eleResponse(eid, "stresses"), dtype=np.float64)
    assert flat.size == layout.flat_size_per_element, (
        f"{cpp_class}: ops.eleResponse returned {flat.size} values; "
        f"catalog expects {layout.flat_size_per_element}."
    )

    # Now drive DomainCapture against the same (already-loaded) state.
    from apeGmsh.results.capture._domain import DomainCapture
    from apeGmsh.solvers._recorder_specs import (
        ResolvedRecorderRecord,
        ResolvedRecorderSpec,
    )

    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(ResolvedRecorderRecord(
            category="gauss", name="rec",
            components=tuple(layout.component_layout),
            dt=None, n_steps=None,
            element_ids=np.array([eid]),
        ),),
    )

    capture_path = tmp_path / "cap.h5"
    with DomainCapture(spec, capture_path, fem, ndm=3, ndf=3) as cap:
        cap.begin_stage("static_load", kind="static")
        cap.step(t=ops.getTime())
        cap.end_stage()

    from apeGmsh.results import Results
    with Results.from_native(capture_path, fem=fem) as r:
        s = r.stage(r.stages[0].id)
        sxx = s.elements.gauss.get(component="stress_xx")
        syy = s.elements.gauss.get(component="stress_yy")
        szz = s.elements.gauss.get(component="stress_zz")
        sxy = s.elements.gauss.get(component="stress_xy")

        # Shape sanity: 1 element × n_gp_expected GPs.
        assert sxx.values.shape == (1, n_gp_expected)
        assert sxx.element_index.tolist() == [eid] * n_gp_expected

        # Physical sanity per GP: vertical compression dominates.
        # σ_zz < 0 at every GP; σ_xx ≈ σ_yy by symmetry; shears small.
        for g in range(n_gp_expected):
            assert szz_sign * szz.values[0, g] > 0, (
                f"{cpp_class}: σ_zz at GP {g} = {szz.values[0, g]} "
                f"has wrong sign for vertical compression."
            )
            np.testing.assert_allclose(
                sxx.values[0, g], syy.values[0, g], rtol=1e-6,
                err_msg=f"{cpp_class}: σ_xx ≠ σ_yy at GP {g}",
            )
            assert abs(sxy.values[0, g]) < 1e-3, (
                f"{cpp_class}: σ_xy at GP {g} = {sxy.values[0, g]} "
                f"unexpectedly large under symmetric vertical load."
            )
