"""Phase 11b Step 3b — real openseespy nodal-forces capture test.

Builds a tiny ElasticBeam3d cantilever directly via openseespy, runs
a static analysis, captures per-element-node forces in BOTH global
and local frames through ``DomainCapture._NodalForcesCapturer``, and
reads back. Verifies physical correctness (cantilever moment Mz at
fixed end = -Fy*L) and frame-distinguished routing.

Skipped if openseespy isn't importable.
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
        group.attrs["ndf"] = 6
        group.attrs["model_name"] = ""
        group.attrs["units"] = ""
        n = group.create_group("nodes")
        n.create_dataset("ids", data=self.nodes.ids)
        n.create_dataset("coords", data=self.nodes.coords)
        group.create_group("elements")


def test_elastic_beam_3d_global_and_local(tmp_path: Path) -> None:
    """ElasticBeam3d cantilever, capture both globalForce and localForce.

    With beam along global x and combined transverse + axial tip load,
    Mz at the fixed end equals -Fy*L (cantilever moment).
    """
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    L = 5.0
    Fx, Fy = 1.0e3, 5.0e2

    coords = np.array(
        [[0.0, 0.0, 0.0], [L, 0.0, 0.0]], dtype=np.float64,
    )
    node_ids = np.array([1, 2], dtype=np.int64)
    for nid, xyz in zip(node_ids, coords):
        ops.node(int(nid), float(xyz[0]), float(xyz[1]), float(xyz[2]))
    ops.fix(1, 1, 1, 1, 1, 1, 1)

    ops.geomTransf("Linear", 1, 0.0, 1.0, 0.0)
    ops.element(
        "elasticBeamColumn", 1, 1, 2,
        0.01, 200e9, 80e9, 1.0e-4, 1.0e-4, 1.5e-4, 1,
    )

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, Fx, Fy, 0.0, 0.0, 0.0, 0.0)

    ops.system("ProfileSPD")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")

    fem = _MinimalFem(node_ids=node_ids, coords=coords)

    from apeGmsh.results import Results
    from apeGmsh.results.capture._domain import DomainCapture
    from apeGmsh.solvers._recorder_specs import (
        ResolvedRecorderRecord,
        ResolvedRecorderSpec,
    )

    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(
            ResolvedRecorderRecord(
                category="elements", name="beam_global",
                components=(
                    "nodal_resisting_force_x",
                    "nodal_resisting_force_y",
                    "nodal_resisting_moment_z",
                ),
                dt=None, n_steps=None,
                element_ids=np.array([1], dtype=np.int64),
            ),
            ResolvedRecorderRecord(
                category="elements", name="beam_local",
                components=(
                    "nodal_resisting_force_local_x",
                    "nodal_resisting_force_local_y",
                    "nodal_resisting_moment_local_z",
                ),
                dt=None, n_steps=None,
                element_ids=np.array([1], dtype=np.int64),
            ),
        ),
    )

    path = tmp_path / "cap.h5"
    with DomainCapture(spec, path, fem, ndm=3, ndf=6, ops=ops) as cap:
        cap.begin_stage("static", kind="static")
        ops.analyze(1)
        cap.step(t=ops.getTime())
        cap.end_stage()
        # Both capturers should have run with no skips.
        assert len(cap._nodal_force_capturers) == 2
        for nfc in cap._nodal_force_capturers:
            assert nfc.skipped_elements == [], (
                f"unexpected skipped elements: {nfc.skipped_elements}"
            )

    with Results.from_native(path) as r:
        s = r.stage(r.stages[0].id)
        comps = set(s.elements.available_components())
        # Global frame components present.
        for n in ("nodal_resisting_force_x", "nodal_resisting_force_y",
                  "nodal_resisting_moment_z"):
            assert n in comps
        # Local frame components present.
        for n in ("nodal_resisting_force_local_x",
                  "nodal_resisting_force_local_y",
                  "nodal_resisting_moment_local_z"):
            assert n in comps

        # ── Global Fx at end nodes should equal applied axial load ──
        slab_fx = s.elements.get(component="nodal_resisting_force_x")
        assert slab_fx.values.shape == (1, 1, 2)
        np.testing.assert_allclose(
            np.abs(slab_fx.values[0, 0]), [Fx, Fx], rtol=1e-9, atol=1e-6,
        )

        # ── Mz at node 1 (fixed) = -Fy*L; at node 2 (free) = 0 ──────
        slab_mz = s.elements.get(component="nodal_resisting_moment_z")
        node1_mz = slab_mz.values[0, 0, 0]
        node2_mz = slab_mz.values[0, 0, 1]
        np.testing.assert_allclose(abs(node1_mz), Fy * L, rtol=1e-9, atol=1e-6)
        assert abs(node2_mz) < 1e-6

        # ── Local axial = Fx (beam along global x) ──────────────────
        slab_n = s.elements.get(component="nodal_resisting_force_local_x")
        np.testing.assert_allclose(
            np.abs(slab_n.values[0, 0]), [Fx, Fx], rtol=1e-9, atol=1e-6,
        )
