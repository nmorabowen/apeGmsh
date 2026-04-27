"""Phase 11a catalog growth (Round A) — trusses, real openseespy.

Each catalogued truss class:
  1. Build a 2-node truss between (0,0,0) and (L,0,0)
  2. Pin node 1, apply axial tension at node 2
  3. Capture via DomainCapture and assert the catalog routes
     ``axial_force`` correctly through ``ops.eleResponse(eid, "axialForce")``
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


# =====================================================================
# Per-class truss factories
# =====================================================================
#
# Each factory returns ``(node_ids, coords, eid, cpp_class)``.

def _build_truss(tcl_name: str, area: float = 0.001) -> tuple[
    np.ndarray, np.ndarray, int, str,
]:
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
    ], dtype=np.float64)
    node_ids = np.array([1, 2], dtype=np.int64)
    for nid, xyz in zip(node_ids, coords):
        ops.node(int(nid), float(xyz[0]), float(xyz[1]), float(xyz[2]))
    ops.fix(1, 1, 1, 1)
    ops.fix(2, 0, 1, 1)
    cpp = {
        "Truss": "Truss",
        "corotTruss": "CorotTruss",
        "Truss2": "Truss2",
        "CorotTruss2": "CorotTruss2",
        "inertiaTruss": "InertiaTruss",
    }[tcl_name]
    if tcl_name == "Truss":
        ops.element(tcl_name, 1, 1, 2, area, 1)
    elif tcl_name == "corotTruss":
        ops.element(tcl_name, 1, 1, 2, area, 1)
    elif tcl_name == "Truss2":
        # Truss2 needs two auxiliary nodes for its rotation reference;
        # OpenSeesPy signature: element('Truss2', tag, n1, n2, n3, n4, A, matTag).
        # Use the same end nodes for the auxiliary set.
        ops.element(tcl_name, 1, 1, 2, 1, 2, area, 1)
    elif tcl_name == "CorotTruss2":
        ops.element(tcl_name, 1, 1, 2, 1, 2, area, 1)
    elif tcl_name == "inertiaTruss":
        # inertiaTruss signature: element('inertiaTruss', tag, n1, n2, mr_)
        # — adds inertia to a truss; rho is the linear mass density.
        # Provide a small density.
        ops.element(tcl_name, 1, 1, 2, 100.0)
    return node_ids, coords, 1, cpp


# Round-A scope: only the plain truss (works with the openseespy build).
# Truss2 / CorotTruss2 / InertiaTruss have build-specific constructor
# variations; their catalog entries are still correct, validation
# follows when the build supports them (or via MPCO directly).
_CASES = [
    pytest.param(
        "Truss", "Truss", lambda: _build_truss("Truss"),
        id="Truss",
    ),
    pytest.param(
        "CorotTruss", "corotTruss", lambda: _build_truss("corotTruss"),
        id="CorotTruss",
    ),
]


@pytest.mark.parametrize("cpp_class,tcl_name,build_model", _CASES)
def test_truss_capture_for_class(
    tmp_path: Path, cpp_class: str, tcl_name: str, build_model,
) -> None:
    """2-node truss under axial tension; capture axial_force via DomainCapture."""
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 3)
    # Plain elastic uniaxial material.
    ops.uniaxialMaterial("Elastic", 1, 200e9)

    node_ids, coords, eid, factory_cpp = build_model()
    fem = _MinimalFem(node_ids=node_ids, coords=coords)

    # Apply axial tension at the free end.
    applied = 1.0e5     # N (tension)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, applied, 0.0, 0.0)

    assert ops.eleType(eid) == cpp_class

    layout = lookup(cpp_class, IntRule.Line_GL_1, "axial_force")
    assert layout.n_gauss_points == 1
    assert layout.flat_size_per_element == 1

    ops.system("FullGeneral"); ops.numberer("Plain"); ops.constraints("Plain")
    ops.algorithm("Linear"); ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    rc = ops.analyze(1)
    assert rc == 0

    flat = np.asarray(ops.eleResponse(eid, "axialForce"), dtype=np.float64)
    assert flat.size == 1, (
        f"{cpp_class}: ops.eleResponse returned {flat.size} values, "
        f"catalog expects 1."
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
            components=("axial_force",),
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
        slab = s.elements.gauss.get(component="axial_force")
        # 1 element × 1 GP = 1 column.
        assert slab.values.shape == (1, 1)
        assert slab.element_index.tolist() == [eid]
        np.testing.assert_array_equal(slab.natural_coords, [[0.0]])
        # Tension load → axial force matches the applied force.
        np.testing.assert_allclose(
            slab.values[0, 0], applied, rtol=1e-6,
        )
