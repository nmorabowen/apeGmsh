"""Phase 7 — real openseespy integration test for DomainCapture.

Builds a tiny standalone OpenSees model (4 nodes, 1 tet) without any
apeGmsh involvement, runs a static analysis, captures via
``DomainCapture`` against a hand-built FEMData. The point is to prove
the DomainCapture flow works against the real ``ops`` API, not to
exercise apeGmsh's solver bridge (mocked tests already cover the
capture logic itself).

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
    """Tiny synthetic FEMData backed by a real snapshot_id."""
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
        nodes_grp = group.create_group("nodes")
        nodes_grp.create_dataset("ids", data=self.nodes.ids)
        nodes_grp.create_dataset("coords", data=self.nodes.coords)
        group.create_group("elements")


def test_single_tet_static_capture(tmp_path: Path) -> None:
    """Static analysis on one tet, capture and read back."""
    # Always start from a clean slate
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 3)

    # Four nodes — three on the base, one apex.
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 1.0],
    ], dtype=np.float64)
    node_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    for nid, xyz in zip(node_ids, coords):
        ops.node(int(nid), float(xyz[0]), float(xyz[1]), float(xyz[2]))

    # Pin the base
    for nid in (1, 2, 3):
        ops.fix(nid, 1, 1, 1)

    # One tet
    ops.nDMaterial("ElasticIsotropic", 1, 200e9, 0.3)
    ops.element("FourNodeTetrahedron", 1, 1, 2, 3, 4, 1)

    # Vertical load at the apex
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(4, 0.0, 0.0, -1.0e6)

    ops.system("ProfileSPD")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")

    # ── Build a synthetic FEMData matching the OpenSees domain ──────
    fem = _MinimalFem(node_ids=node_ids, coords=coords)

    # ── Declare a recorder spec by hand and capture ─────────────────
    from apeGmsh.results.capture._domain import DomainCapture
    from apeGmsh.solvers._recorder_specs import (
        ResolvedRecorderRecord,
        ResolvedRecorderSpec,
    )

    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(
            ResolvedRecorderRecord(
                category="nodes", name="apex",
                components=("displacement_x", "displacement_y", "displacement_z"),
                dt=None, n_steps=None,
                node_ids=np.array([4]),
            ),
            ResolvedRecorderRecord(
                category="nodes", name="base_rxn",
                components=("reaction_force_z",),
                dt=None, n_steps=None,
                node_ids=np.array([1, 2, 3]),
            ),
        ),
    )

    capture_path = tmp_path / "tet.h5"
    with DomainCapture(spec, capture_path, fem, ndm=3, ndf=3) as cap:
        cap.begin_stage("static_load", kind="static")
        ok = ops.analyze(1)
        assert ok == 0, f"analyze() returned {ok}"
        cap.step(t=ops.getTime())
        cap.end_stage()

    # ── Read back via Results ────────────────────────────────────────
    from apeGmsh.results import Results
    with Results.from_native(capture_path, fem=fem) as r:
        stages = r.stages
        assert len(stages) == 1
        assert stages[0].name == "static_load"

        apex_disp_z = r.nodes.get(
            ids=[4], component="displacement_z",
        )
        assert apex_disp_z.values.shape == (1, 1)
        # Downward 1 MN load → apex moves down (negative z).
        assert apex_disp_z.values[0, 0] < 0

        base_rxn_z = r.nodes.get(
            ids=[1, 2, 3], component="reaction_force_z",
        )
        # Equilibrium: sum of base reactions Fz = +1e6 (upward).
        assert base_rxn_z.values.sum() == pytest.approx(1.0e6, rel=1e-3)


# Note: a real modal-capture-against-openseespy test would need a model
# with several free DOFs (ARPACK requires NCV ≤ N, default NCV ≈ 2·NEV+1).
# The mocked modal test in test_results_domain_capture.py already covers
# the DomainCapture modal flow logic; full openseespy modal validation
# is a separate integration test for a future phase.
