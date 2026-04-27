"""Phase 11b Step 3b — nodal-forces DomainCapture, mocked ops.

Exercises ``_NodalForcesCapturer`` for closed-form line elements
without spinning up OpenSees. A fake ``ops`` module returns
deterministic ``eleType`` and ``eleResponse(eid, "globalForce" |
"localForce")`` values so the catalog → unflatten_nodal →
write_nodal_forces_group flow can be inspected on disk.

Real-openseespy coverage lives in
``test_results_domain_capture_nodal_forces_real.py``.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.capture._domain import DomainCapture, _NodalForcesCapturer
from apeGmsh.solvers._element_response import (
    ELE_TAG_ElasticBeam2d,
    ELE_TAG_ElasticBeam3d,
    ELE_TAG_ElasticTimoshenkoBeam3d,
)
from apeGmsh.solvers._recorder_specs import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)


# =====================================================================
# Fake ops with element-response support
# =====================================================================

class _FakeOpsBeams:
    def __init__(self) -> None:
        self.ele_class: dict[int, str] = {}
        # (eid, keyword) → ndarray
        self.ele_response: dict[tuple[int, str], np.ndarray] = {}

    def eleType(self, eid: int) -> str:
        return self.ele_class[int(eid)]

    def eleResponse(self, eid: int, *args) -> list[float]:
        keyword = args[0] if len(args) == 1 else ".".join(str(a) for a in args)
        return list(self.ele_response[(int(eid), keyword)])

    def reactions(self) -> None: ...


class _MockFem:
    def __init__(self, node_ids) -> None:
        ids = np.asarray(node_ids, dtype=np.int64)
        self.nodes = SimpleNamespace(
            ids=ids, coords=np.zeros((ids.size, 3), dtype=np.float64),
        )
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
        n.create_dataset("ids", data=np.asarray(self.nodes.ids, dtype=np.int64))
        n.create_dataset(
            "coords", data=np.asarray(self.nodes.coords, dtype=np.float64),
        )
        group.create_group("elements")


def _spec_with(*records, snapshot_id) -> ResolvedRecorderSpec:
    return ResolvedRecorderSpec(
        fem_snapshot_id=snapshot_id,
        records=tuple(records),
    )


# =====================================================================
# Frame routing
# =====================================================================

class TestFrameRouting:
    def test_global_components_route_to_globalForce(self) -> None:
        rec = ResolvedRecorderRecord(
            category="elements", name="r",
            components=("nodal_resisting_force_x", "nodal_resisting_moment_z"),
            dt=None, n_steps=None,
            element_ids=np.array([1]),
        )
        cap = _NodalForcesCapturer(rec)
        assert cap._ops_keyword == "globalForce"
        assert cap._catalog_token == "global_force"

    def test_local_components_route_to_localForce(self) -> None:
        rec = ResolvedRecorderRecord(
            category="elements", name="r",
            components=("nodal_resisting_force_local_x",
                        "nodal_resisting_moment_local_z"),
            dt=None, n_steps=None,
            element_ids=np.array([1]),
        )
        cap = _NodalForcesCapturer(rec)
        assert cap._ops_keyword == "localForce"
        assert cap._catalog_token == "local_force"

    def test_mixed_frames_raises(self) -> None:
        rec = ResolvedRecorderRecord(
            category="elements", name="r",
            components=("nodal_resisting_force_x",
                        "nodal_resisting_force_local_y"),
            dt=None, n_steps=None,
            element_ids=np.array([1]),
        )
        with pytest.raises(ValueError, match="mixes global and local"):
            _NodalForcesCapturer(rec)

    def test_no_recognised_components_raises(self) -> None:
        rec = ResolvedRecorderRecord(
            category="elements", name="r",
            components=("displacement_x",),    # nodal kinematic, not nodal_forces
            dt=None, n_steps=None,
            element_ids=np.array([1]),
        )
        with pytest.raises(ValueError, match="no recognised nodal-forces"):
            _NodalForcesCapturer(rec)


# =====================================================================
# Single-class capture
# =====================================================================

class TestElasticBeam3dGlobalCapture:
    def test_two_step_capture_writes_correct_shape(
        self, tmp_path: Path,
    ) -> None:
        fem = _MockFem([1, 2, 3])
        spec = _spec_with(
            ResolvedRecorderRecord(
                category="elements", name="beam_global",
                components=(
                    "nodal_resisting_force_x", "nodal_resisting_force_y",
                    "nodal_resisting_force_z", "nodal_resisting_moment_x",
                    "nodal_resisting_moment_y", "nodal_resisting_moment_z",
                ),
                dt=None, n_steps=None,
                element_ids=np.array([7, 11]),
            ),
            snapshot_id=fem.snapshot_id,
        )

        ops = _FakeOpsBeams()
        ops.ele_class[7] = "ElasticBeam3d"
        ops.ele_class[11] = "ElasticBeam3d"

        path = tmp_path / "cap.h5"
        with DomainCapture(spec, path, fem, ndm=3, ndf=6, ops=ops) as cap:
            cap.begin_stage("static", kind="static")
            for t in (0.0, 1.0):
                offset = int(t) * 1000
                # Per element: 2 nodes × 6 comps = 12 doubles, node-major.
                # Encoding: (eid * 100) + (n * 10) + k + offset.
                for eid in (7, 11):
                    flat = np.zeros(12, dtype=np.float64)
                    for n in range(2):
                        for k in range(6):
                            flat[n * 6 + k] = eid * 100 + n * 10 + k + offset
                    ops.ele_response[(eid, "globalForce")] = flat
                cap.step(t=t)
            cap.end_stage()

        with Results.from_native(path) as r:
            s = r.stage(r.stages[0].id)
            slab_fx = s.elements.get(component="nodal_resisting_force_x")
            # 2 elements × 2 nodes × 2 steps.
            assert slab_fx.values.shape == (2, 2, 2)
            # k=0: t=0,e=7,n=0 → 700; t=0,e=7,n=1 → 710; t=0,e=11,n=0 → 1100;
            # t=0,e=11,n=1 → 1110; t=1,e=7,n=0 → 1700; etc.
            np.testing.assert_array_equal(
                slab_fx.values[0],
                [[700.0, 710.0], [1100.0, 1110.0]],
            )
            np.testing.assert_array_equal(
                slab_fx.values[1],
                [[1700.0, 1710.0], [2100.0, 2110.0]],
            )

            # Mz = k=5: t=0,e=7,n=0 → 705; t=0,e=11,n=1 → 1115.
            slab_mz = s.elements.get(component="nodal_resisting_moment_z")
            assert slab_mz.values[0, 0, 0] == 705.0
            assert slab_mz.values[0, 1, 1] == 1115.0


class TestElasticBeam3dLocalCapture:
    def test_local_frame_uses_localForce_keyword(self, tmp_path: Path) -> None:
        fem = _MockFem([1, 2])
        spec = _spec_with(
            ResolvedRecorderRecord(
                category="elements", name="beam_local",
                components=(
                    "nodal_resisting_force_local_x",
                    "nodal_resisting_force_local_y",
                ),
                dt=None, n_steps=None,
                element_ids=np.array([1]),
            ),
            snapshot_id=fem.snapshot_id,
        )
        ops = _FakeOpsBeams()
        ops.ele_class[1] = "ElasticBeam3d"
        # Provide localForce response (12 doubles).
        ops.ele_response[(1, "localForce")] = np.arange(12, dtype=np.float64) + 100

        path = tmp_path / "cap.h5"
        with DomainCapture(spec, path, fem, ops=ops) as cap:
            cap.begin_stage("g", kind="static")
            cap.step(t=0.0)
            cap.end_stage()

        with Results.from_native(path) as r:
            s = r.stage(r.stages[0].id)
            slab_n = s.elements.get(
                component="nodal_resisting_force_local_x",
            )
            # k=0 (N): n=0 → 100; n=1 → 106.
            np.testing.assert_array_equal(slab_n.values[0, 0], [100.0, 106.0])

            slab_vy = s.elements.get(
                component="nodal_resisting_force_local_y",
            )
            # k=1 (Vy): n=0 → 101; n=1 → 107.
            np.testing.assert_array_equal(slab_vy.values[0, 0], [101.0, 107.0])


class TestElasticBeam2dGlobalCapture:
    def test_2d_capture_3_components_per_node(self, tmp_path: Path) -> None:
        fem = _MockFem([1, 2])
        spec = _spec_with(
            ResolvedRecorderRecord(
                category="elements", name="beam2d",
                components=(
                    "nodal_resisting_force_x",
                    "nodal_resisting_force_y",
                    "nodal_resisting_moment_z",
                ),
                dt=None, n_steps=None,
                element_ids=np.array([1]),
            ),
            snapshot_id=fem.snapshot_id,
        )
        ops = _FakeOpsBeams()
        ops.ele_class[1] = "ElasticBeam2d"
        # 2 nodes × 3 comps = 6 doubles.
        ops.ele_response[(1, "globalForce")] = np.array(
            [10, 20, 30, 40, 50, 60], dtype=np.float64,
        )

        path = tmp_path / "cap.h5"
        with DomainCapture(spec, path, fem, ops=ops) as cap:
            cap.begin_stage("g", kind="static")
            cap.step(t=0.0)
            cap.end_stage()

        with Results.from_native(path) as r:
            s = r.stage(r.stages[0].id)
            slab_fx = s.elements.get(component="nodal_resisting_force_x")
            np.testing.assert_array_equal(slab_fx.values[0, 0], [10.0, 40.0])
            slab_mz = s.elements.get(component="nodal_resisting_moment_z")
            np.testing.assert_array_equal(slab_mz.values[0, 0], [30.0, 60.0])


# =====================================================================
# Multi-class records — one group per class
# =====================================================================

class TestMixedClassCapture:
    def test_eb3d_and_etb3d_in_one_record(self, tmp_path: Path) -> None:
        """Same record covers ElasticBeam3d + ElasticTimoshenkoBeam3d."""
        fem = _MockFem([1, 2, 3, 4])
        spec = _spec_with(
            ResolvedRecorderRecord(
                category="elements", name="all_beams",
                components=("nodal_resisting_force_x",),
                dt=None, n_steps=None,
                element_ids=np.array([10, 20]),
            ),
            snapshot_id=fem.snapshot_id,
        )
        ops = _FakeOpsBeams()
        ops.ele_class[10] = "ElasticBeam3d"
        ops.ele_class[20] = "ElasticTimoshenkoBeam3d"
        ops.ele_response[(10, "globalForce")] = np.arange(12, dtype=np.float64)
        ops.ele_response[(20, "globalForce")] = (
            np.arange(12, dtype=np.float64) + 100
        )

        path = tmp_path / "cap.h5"
        with DomainCapture(spec, path, fem, ops=ops) as cap:
            cap.begin_stage("g", kind="static")
            cap.step(t=0.0)
            cap.end_stage()

        with Results.from_native(path) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.get(component="nodal_resisting_force_x")
        # 2 elements × 2 nodes (one group per class internally, but the
        # public read merges across).
        assert slab.values.shape == (1, 2, 2)
        # element_ids should include both 10 and 20.
        assert sorted(slab.element_ids.tolist()) == [10, 20]


# =====================================================================
# Skip / error behaviours
# =====================================================================

class TestSkipBehaviours:
    def test_uncatalogued_class_skipped(self, tmp_path: Path) -> None:
        fem = _MockFem([1, 2])
        spec = _spec_with(
            ResolvedRecorderRecord(
                category="elements", name="r",
                components=("nodal_resisting_force_x",),
                dt=None, n_steps=None,
                element_ids=np.array([5]),
            ),
            snapshot_id=fem.snapshot_id,
        )
        ops = _FakeOpsBeams()
        ops.ele_class[5] = "MysteryBeam3d"

        path = tmp_path / "cap.h5"
        with DomainCapture(spec, path, fem, ops=ops) as cap:
            cap.begin_stage("g", kind="static")
            cap.step(t=0.0)
            cap.end_stage()
            (nfc,) = cap._nodal_force_capturers
            assert any(
                "NODAL_FORCE_CATALOG" in reason
                for _, reason in nfc.skipped_elements
            )

    def test_wrong_size_response_raises(self, tmp_path: Path) -> None:
        fem = _MockFem([1, 2])
        spec = _spec_with(
            ResolvedRecorderRecord(
                category="elements", name="r",
                components=("nodal_resisting_force_x",),
                dt=None, n_steps=None,
                element_ids=np.array([1]),
            ),
            snapshot_id=fem.snapshot_id,
        )
        ops = _FakeOpsBeams()
        ops.ele_class[1] = "ElasticBeam3d"
        # ElasticBeam3d expects 12 values; provide 11.
        ops.ele_response[(1, "globalForce")] = np.zeros(11, dtype=np.float64)

        path = tmp_path / "cap.h5"
        with DomainCapture(spec, path, fem, ops=ops) as cap:
            cap.begin_stage("g", kind="static")
            with pytest.raises(ValueError, match="returned 11 values"):
                cap.step(t=0.0)


# =====================================================================
# Mixed: nodal_forces + line_stations + gauss in one capture
# =====================================================================

class TestMixedCategoriesAllPhases:
    def test_nodal_forces_alongside_gauss(self, tmp_path: Path) -> None:
        """A capture spec with gauss + elements records works end-to-end."""
        fem = _MockFem([1, 2])
        ops = _FakeOpsBeams()
        ops.ele_class[1] = "ElasticBeam3d"
        ops.ele_response[(1, "globalForce")] = np.arange(12, dtype=np.float64)
        ops.ele_class[10] = "FourNodeTetrahedron"
        ops.ele_response[(10, "stresses")] = np.arange(6, dtype=np.float64)

        spec = _spec_with(
            ResolvedRecorderRecord(
                category="elements", name="beam",
                components=("nodal_resisting_force_x",),
                dt=None, n_steps=None,
                element_ids=np.array([1]),
            ),
            ResolvedRecorderRecord(
                category="gauss", name="solid",
                components=("stress_xx",),
                dt=None, n_steps=None,
                element_ids=np.array([10]),
            ),
            snapshot_id=fem.snapshot_id,
        )

        path = tmp_path / "mixed.h5"
        with DomainCapture(spec, path, fem, ops=ops) as cap:
            cap.begin_stage("g", kind="static")
            cap.step(t=0.0)
            cap.end_stage()

        with Results.from_native(path) as r:
            s = r.stage(r.stages[0].id)
            es = s.elements.get(component="nodal_resisting_force_x")
            assert es.values.shape == (1, 1, 2)
            # k=0: n=0 → 0, n=1 → 6.
            np.testing.assert_array_equal(es.values[0, 0], [0.0, 6.0])
            gs = s.elements.gauss.get(component="stress_xx")
            assert gs.values.shape == (1, 1)
