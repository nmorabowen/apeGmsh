"""Phase 11a Step C — element-level (gauss) DomainCapture, mocked ops.

Exercises the in-process capture path for continuum stress/strain
without spinning up OpenSees. Uses a fake ``ops`` module that returns
deterministic ``eleType`` / ``eleResponse`` values so the catalog
lookup → unflatten → write_gauss_group flow can be inspected on disk.

Real-openseespy coverage lives in
``test_results_domain_capture_gauss_real.py``.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.capture._domain import DomainCapture
from apeGmsh.solvers._element_response import (
    IntRule,
    flatten,
    lookup,
)
from apeGmsh.solvers._recorder_specs import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)


# =====================================================================
# Fake ops with element-response support
# =====================================================================

class _FakeOpsElements:
    """Stand-in for openseespy that knows about elements.

    ``ele_class`` maps eid → C++ class name (what ``ops.eleType``
    returns). ``ele_response`` maps (eid, keyword) → flat ndarray
    (what ``ops.eleResponse`` returns).
    """

    def __init__(self) -> None:
        self.ele_class: dict[int, str] = {}
        self.ele_response: dict[tuple[int, str], np.ndarray] = {}

    def eleType(self, eid: int) -> str:
        return self.ele_class[int(eid)]

    def eleResponse(self, eid: int, *args) -> list[float]:
        # OpenSees joins multi-token responses; for our use we always
        # pass a single keyword so args is len 1.
        keyword = args[0] if len(args) == 1 else " ".join(args)
        return list(self.ele_response[(int(eid), keyword)])

    # Stubs in case any node-side code path executes.
    def reactions(self) -> None: ...


# =====================================================================
# Mock fem (mirrors the existing capture-test pattern)
# =====================================================================

class _MockFem:
    def __init__(self, node_ids) -> None:
        ids = np.asarray(node_ids, dtype=np.int64)
        coords = np.zeros((ids.size, 3), dtype=np.float64)
        self.nodes = SimpleNamespace(ids=ids, coords=coords)
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


def _make_spec(*records, snapshot_id):
    return ResolvedRecorderSpec(
        fem_snapshot_id=snapshot_id,
        records=tuple(records),
    )


# =====================================================================
# Single-class, single-GP capture
# =====================================================================

class TestFourNodeTetCapture:
    def test_two_step_capture_writes_correct_shape(self, tmp_path: Path) -> None:
        layout = lookup("FourNodeTetrahedron", IntRule.Tet_GL_1, "stress")
        fem = _MockFem([1, 2, 3, 4])
        spec = _make_spec(
            ResolvedRecorderRecord(
                category="gauss", name="solid_stress",
                components=tuple(layout.component_layout),
                dt=None, n_steps=None,
                element_ids=np.array([10, 20]),
            ),
            snapshot_id=fem.snapshot_id,
        )

        # Per-element step values.
        ops = _FakeOpsElements()
        ops.ele_class[10] = "FourNodeTetrahedron"
        ops.ele_class[20] = "FourNodeTetrahedron"
        # eid=10, t=0: [1, 2, 3, 4, 5, 6]; t=1: [11, 12, ...]
        # eid=20, t=0: [101, 102, ...]; t=1: [111, 112, ...]
        # We push values via per-step priming below.

        path = tmp_path / "cap.h5"
        with DomainCapture(spec, path, fem, ndm=3, ndf=3, ops=ops) as cap:
            cap.begin_stage("static", kind="static")
            for t in (0.0, 1.0):
                offset = int(t) * 10
                ops.ele_response[(10, "stresses")] = np.array(
                    [1 + offset, 2 + offset, 3 + offset,
                     4 + offset, 5 + offset, 6 + offset],
                    dtype=np.float64,
                )
                ops.ele_response[(20, "stresses")] = np.array(
                    [101 + offset, 102 + offset, 103 + offset,
                     104 + offset, 105 + offset, 106 + offset],
                    dtype=np.float64,
                )
                cap.step(t=t)
            cap.end_stage()

        # Read back through the public API.
        with Results.from_native(path) as r:
            s = r.stage(r.stages[0].id)

            sxx = s.elements.gauss.get(component="stress_xx")
            # 2 elements × 1 GP = 2 cols; 2 steps.
            assert sxx.values.shape == (2, 2)
            np.testing.assert_array_equal(
                sxx.values, [[1.0, 101.0], [11.0, 111.0]],
            )
            np.testing.assert_array_equal(sxx.element_index, [10, 20])

            # Stress_xz should be the 6th component → 6, 16 / 106, 116.
            sxz = s.elements.gauss.get(component="stress_xz")
            np.testing.assert_array_equal(
                sxz.values, [[6.0, 106.0], [16.0, 116.0]],
            )

    def test_natural_coords_match_catalog(self, tmp_path: Path) -> None:
        fem = _MockFem([1])
        spec = _make_spec(
            ResolvedRecorderRecord(
                category="gauss", name="r",
                components=("stress_xx",),
                dt=None, n_steps=None,
                element_ids=np.array([7]),
            ),
            snapshot_id=fem.snapshot_id,
        )
        ops = _FakeOpsElements()
        ops.ele_class[7] = "FourNodeTetrahedron"
        ops.ele_response[(7, "stresses")] = np.zeros(6)

        path = tmp_path / "cap.h5"
        with DomainCapture(spec, path, fem, ops=ops) as cap:
            cap.begin_stage("g", kind="static")
            cap.step(t=0.0)
            cap.end_stage()

        with Results.from_native(path) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.gauss.get(component="stress_xx")
            np.testing.assert_allclose(
                slab.natural_coords, [[0.25, 0.25, 0.25]],
            )


# =====================================================================
# Multi-class capture — two element types in one record
# =====================================================================

class TestMixedClassCapture:
    def test_tet1_and_tet2_in_one_record(self, tmp_path: Path) -> None:
        """One record over a mixed-class PG: each class gets its own group."""
        fem = _MockFem([1, 2, 3, 4])
        spec = _make_spec(
            ResolvedRecorderRecord(
                category="gauss", name="all_solid",
                components=("stress_xx", "stress_yy", "stress_zz",
                            "stress_xy", "stress_yz", "stress_xz"),
                dt=None, n_steps=None,
                element_ids=np.array([100, 200]),
            ),
            snapshot_id=fem.snapshot_id,
        )

        ops = _FakeOpsElements()
        ops.ele_class[100] = "FourNodeTetrahedron"
        ops.ele_class[200] = "TenNodeTetrahedron"
        # FourNodeTet: 6 values (1 GP × 6 comp).
        ops.ele_response[(100, "stresses")] = np.arange(6, dtype=np.float64)
        # TenNodeTet: 24 values (4 GPs × 6 comp), GP-slowest.
        ops.ele_response[(200, "stresses")] = np.arange(24, dtype=np.float64) + 100

        path = tmp_path / "cap.h5"
        with DomainCapture(spec, path, fem, ops=ops) as cap:
            cap.begin_stage("g", kind="static")
            cap.step(t=0.0)
            cap.end_stage()

        with Results.from_native(path) as r:
            s = r.stage(r.stages[0].id)
            sxx = s.elements.gauss.get(component="stress_xx")
            # 1 GP for tet1 (eid 100) + 4 GPs for tet2 (eid 200) = 5 cols.
            assert sxx.values.shape == (1, 5)
            # tet1 occupies col 0; stress_xx at GP 0 = 0.
            assert sxx.values[0, 0] == 0.0
            # tet2 occupies cols 1..4; stress_xx values are flat[0,6,12,18] + 100.
            np.testing.assert_array_equal(
                sxx.values[0, 1:], [100.0, 106.0, 112.0, 118.0],
            )
            # element_index repeats the right tag per GP.
            np.testing.assert_array_equal(
                sxx.element_index, [100, 200, 200, 200, 200],
            )


# =====================================================================
# Catalog filtering — uncatalogued elements are skipped, not crashed
# =====================================================================

class TestCatalogFiltering:
    def test_uncatalogued_class_is_skipped(self, tmp_path: Path) -> None:
        fem = _MockFem([1])
        spec = _make_spec(
            ResolvedRecorderRecord(
                category="gauss", name="mixed",
                components=("stress_xx",),
                dt=None, n_steps=None,
                element_ids=np.array([1, 2]),
            ),
            snapshot_id=fem.snapshot_id,
        )
        ops = _FakeOpsElements()
        ops.ele_class[1] = "FourNodeTetrahedron"
        ops.ele_class[2] = "NotInCatalog"
        ops.ele_response[(1, "stresses")] = np.array(
            [10, 20, 30, 40, 50, 60], dtype=np.float64,
        )
        # No ele_response set for element 2 — must never be queried.

        path = tmp_path / "cap.h5"
        with DomainCapture(spec, path, fem, ops=ops) as cap:
            cap.begin_stage("g", kind="static")
            cap.step(t=0.0)
            cap.end_stage()

        with Results.from_native(path) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.gauss.get(component="stress_xx")
            # Only element 1 captured; element 2 silently dropped.
            assert slab.element_index.tolist() == [1]
            assert slab.values[0, 0] == 10.0


# =====================================================================
# Error paths
# =====================================================================

class TestErrorPaths:
    def test_mixed_stress_and_strain_raises(self, tmp_path: Path) -> None:
        fem = _MockFem([1])
        spec = _make_spec(
            ResolvedRecorderRecord(
                category="gauss", name="mixed",
                components=("stress_xx", "strain_xx"),
                dt=None, n_steps=None,
                element_ids=np.array([1]),
            ),
            snapshot_id=fem.snapshot_id,
        )
        path = tmp_path / "cap.h5"
        # _GaussCapturer construction raises in __enter__.
        with pytest.raises(ValueError, match="mixes work-conjugate families"):
            with DomainCapture(spec, path, fem, ops=_FakeOpsElements()):
                pass

    def test_response_size_mismatch_raises(self, tmp_path: Path) -> None:
        fem = _MockFem([1])
        spec = _make_spec(
            ResolvedRecorderRecord(
                category="gauss", name="r",
                components=("stress_xx",),
                dt=None, n_steps=None,
                element_ids=np.array([1]),
            ),
            snapshot_id=fem.snapshot_id,
        )
        ops = _FakeOpsElements()
        ops.ele_class[1] = "FourNodeTetrahedron"
        # Returns 5 values; catalog expects 6.
        ops.ele_response[(1, "stresses")] = np.zeros(5)

        path = tmp_path / "cap.h5"
        with DomainCapture(spec, path, fem, ops=ops) as cap:
            cap.begin_stage("g", kind="static")
            with pytest.raises(ValueError, match="returned 5 values"):
                cap.step(t=0.0)
