"""Phase 11a catalog growth — real-openseespy validation for 2-D solids.

Mirrors ``test_results_catalog_solids_real.py`` for plane elements:
build a unit-square (or unit-triangle) model under uniform vertical
compression, run a static analysis, capture σ_xx / σ_yy / σ_xy via
DomainCapture, and check that the recovered shape and physics match
the catalog.

Plane stress with E=200 GPa, ν=0.3, traction t = -F/L on top:
under a free lateral boundary the only nonzero stress is σ_yy.
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
# Per-class minimal model factories (ndm=2, ndf=2)
# =====================================================================

def _build_unit_square_quad() -> tuple[
    np.ndarray, np.ndarray, int, str,
]:
    """4-node quad unit square. Tcl: ``element quad ... thick type matTag``."""
    coords = np.array([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
    ], dtype=np.float64)
    node_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    for nid, xy in zip(node_ids, coords):
        ops.node(int(nid), float(xy[0]), float(xy[1]))
    # Pin bottom edge (y=0): nodes 1 and 2.
    for nid in (1, 2):
        ops.fix(int(nid), 1, 1)
    ops.element("quad", 1, 1, 2, 3, 4, 1.0, "PlaneStress", 1)
    return node_ids, coords, 1, "FourNodeQuad"


def _build_unit_triangle_tri31() -> tuple[
    np.ndarray, np.ndarray, int, str,
]:
    """3-node triangle. Tcl: ``element tri31 ... thick type matTag``."""
    coords = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.5, 1.0],
    ], dtype=np.float64)
    node_ids = np.array([1, 2, 3], dtype=np.int64)
    for nid, xy in zip(node_ids, coords):
        ops.node(int(nid), float(xy[0]), float(xy[1]))
    for nid in (1, 2):
        ops.fix(int(nid), 1, 1)
    ops.element("tri31", 1, 1, 2, 3, 1.0, "PlaneStress", 1)
    return node_ids, coords, 1, "Tri31"


def _build_unit_square_8node_quad() -> tuple[
    np.ndarray, np.ndarray, int, str,
]:
    """8-node serendipity quad (EightNodeQuad).

    Node order: 4 corners CCW (1-4), then 4 edge midpoints CCW
    starting at the bottom edge (5: between 1-2, 6: between 2-3,
    7: between 3-4, 8: between 4-1).
    """
    corners = np.array([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
    ], dtype=np.float64)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    midpts = np.array([0.5 * (corners[a] + corners[b]) for a, b in edges])
    coords = np.vstack([corners, midpts])
    node_ids = np.arange(1, 9, dtype=np.int64)
    for nid, xy in zip(node_ids, coords):
        ops.node(int(nid), float(xy[0]), float(xy[1]))
    # Pin the bottom edge: corners 1, 2, plus the bottom-edge
    # midpoint at node 5.
    for nid in (1, 2, 5):
        ops.fix(int(nid), 1, 1)
    # Tcl element name for EightNodeQuad is ``quad8n`` (see
    # OpenSeesElementCommands.cpp:674).
    ops.element(
        "quad8n", 1, 1, 2, 3, 4, 5, 6, 7, 8,
        1.0, "PlaneStress", 1,
    )
    return node_ids, coords, 1, "EightNodeQuad"


def _build_unit_square_sspquad() -> tuple[
    np.ndarray, np.ndarray, int, str,
]:
    """4-node SSPquad. Tcl: ``element SSPquad ... matTag type thickness``."""
    coords = np.array([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
    ], dtype=np.float64)
    node_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    for nid, xy in zip(node_ids, coords):
        ops.node(int(nid), float(xy[0]), float(xy[1]))
    for nid in (1, 2):
        ops.fix(int(nid), 1, 1)
    ops.element("SSPquad", 1, 1, 2, 3, 4, 1, "PlaneStress", 1.0)
    return node_ids, coords, 1, "SSPquad"


class _MinimalFem:
    def __init__(self, node_ids: np.ndarray, coords: np.ndarray) -> None:
        # Pad 2D coords to 3D so compute_snapshot_id stays uniform.
        coords3d = np.zeros((coords.shape[0], 3), dtype=np.float64)
        coords3d[:, :coords.shape[1]] = coords
        self.nodes = SimpleNamespace(ids=node_ids, coords=coords3d)
        self.elements = []

    @property
    def snapshot_id(self) -> str:
        from apeGmsh.mesh._femdata_hash import compute_snapshot_id
        return compute_snapshot_id(self)

    def to_native_h5(self, group) -> None:
        group.attrs["snapshot_id"] = self.snapshot_id
        group.attrs["ndm"] = 2
        group.attrs["ndf"] = 2
        group.attrs["model_name"] = ""
        group.attrs["units"] = ""
        n = group.create_group("nodes")
        n.create_dataset("ids", data=self.nodes.ids)
        n.create_dataset("coords", data=self.nodes.coords)
        group.create_group("elements")


# =====================================================================
# Parametrized test cases
# =====================================================================

_CASES = [
    pytest.param(
        "FourNodeQuad", IntRule.Quad_GL_2, 4,
        _build_unit_square_quad, [3, 4],
        id="FourNodeQuad",
    ),
    pytest.param(
        "Tri31", IntRule.Triangle_GL_1, 1,
        _build_unit_triangle_tri31, [3],
        id="Tri31",
    ),
    pytest.param(
        "SSPquad", IntRule.Quad_GL_1, 1,
        _build_unit_square_sspquad, [3, 4],
        id="SSPquad",
    ),
    # Top-edge nodes for the 8-node quad: 2 corners (3, 4) + the
    # top-edge midpoint (node 7).
    pytest.param(
        "EightNodeQuad", IntRule.Quad_GL_3, 9,
        _build_unit_square_8node_quad, [3, 4, 7],
        id="EightNodeQuad",
    ),
]


@pytest.mark.parametrize(
    "cpp_class,int_rule,n_gp_expected,build_model,top_nids", _CASES,
)
def test_real_2d_capture_for_class(
    tmp_path: Path,
    cpp_class: str,
    int_rule: int,
    n_gp_expected: int,
    build_model,
    top_nids: list[int],
) -> None:
    """Vertical compression on a plane element; capture via DomainCapture."""
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    ops.nDMaterial("ElasticIsotropic", 1, 200e9, 0.3)

    node_ids, coords, eid, factory_class_name = build_model()
    fem = _MinimalFem(node_ids=node_ids, coords=coords)

    # Vertical (-y) compressive load distributed over top-edge nodes.
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    per_node = -1.0e6 / len(top_nids)
    for nid in top_nids:
        ops.load(int(nid), 0.0, per_node)

    assert ops.eleType(eid) == cpp_class, (
        f"factory {factory_class_name!r} produced ops.eleType="
        f"{ops.eleType(eid)!r}, expected {cpp_class!r}."
    )

    layout = lookup(cpp_class, int_rule, "stress")
    assert layout.n_gauss_points == n_gp_expected
    assert layout.n_components_per_gp == 3   # plane → 3 components

    ops.system("ProfileSPD"); ops.numberer("RCM"); ops.constraints("Plain")
    ops.algorithm("Linear"); ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    rc = ops.analyze(1)
    assert rc == 0

    flat = np.asarray(ops.eleResponse(eid, "stresses"), dtype=np.float64)
    assert flat.size == layout.flat_size_per_element, (
        f"{cpp_class}: ops.eleResponse returned {flat.size} values; "
        f"catalog expects {layout.flat_size_per_element}."
    )

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
    with DomainCapture(spec, capture_path, fem, ndm=2, ndf=2) as cap:
        cap.begin_stage("static_load", kind="static")
        cap.step(t=ops.getTime())
        cap.end_stage()

    from apeGmsh.results import Results
    with Results.from_native(capture_path, fem=fem) as r:
        s = r.stage(r.stages[0].id)
        sxx = s.elements.gauss.get(component="stress_xx")
        syy = s.elements.gauss.get(component="stress_yy")
        sxy = s.elements.gauss.get(component="stress_xy")

        assert sxx.values.shape == (1, n_gp_expected)
        assert sxx.element_index.tolist() == [eid] * n_gp_expected
        assert sxx.natural_coords.shape == (n_gp_expected, 2)

        # Vertical compression: σ_yy must be the most-negative
        # component. The base BCs introduce Poisson-coupled σ_xx, so
        # σ_xx isn't zero — but σ_yy must dominate. This is the real
        # catalog-correctness test: it confirms the component layout
        # is (xx, yy, xy), not flipped.
        for g in range(n_gp_expected):
            assert syy.values[0, g] < 0, (
                f"{cpp_class}: σ_yy at GP {g} = {syy.values[0, g]} "
                f"has wrong sign for vertical compression."
            )
            assert syy.values[0, g] < sxx.values[0, g], (
                f"{cpp_class}: σ_yy ({syy.values[0, g]}) is not more "
                f"negative than σ_xx ({sxx.values[0, g]}) at GP {g} — "
                f"likely a component-order swap in the catalog."
            )
            # σ_xy is not strictly zero (off-center GPs in quad), but
            # must be smaller in magnitude than σ_yy.
            assert abs(sxy.values[0, g]) < abs(syy.values[0, g]), (
                f"{cpp_class}: |σ_xy| ({sxy.values[0, g]}) ≥ |σ_yy| "
                f"({syy.values[0, g]}) at GP {g} — likely a "
                f"component-order swap in the catalog."
            )
