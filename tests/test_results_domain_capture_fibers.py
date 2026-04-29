"""Phase 11e — beam fiber-section DomainCapture, mocked ops.

Exercises the in-process capture path for ``category="fibers"``
records (FiberSection2d / FiberSection3d on beam-columns) without
spinning up OpenSees. A fake ops module returns deterministic
``eleType``, ``integrationPoints``, and ``section.<sec>.fiberData2``
values so the discover-on-first-step → write_fibers_group flow can
be inspected on disk.

The capture relies on ``ops.eleResponse(eid, "section", str(sec),
"fiberData2")`` returning ``[y, z, area, material_tag, stress,
strain] × n_fibers`` in one flat call. Geometry (the first 4
entries per fiber) is captured on the first step; stress and strain
are captured on every step.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.capture._domain import DomainCapture
from apeGmsh.solvers._recorder_specs import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)


# =====================================================================
# Fake ops with fiber-section support
# =====================================================================

class _FakeOpsFibers:
    """Stand-in for openseespy that knows about beam-column fiber sections.

    ``ele_class`` maps eid → C++ class name returned by ``ops.eleType``.
    ``integration_points`` maps eid → physical xi*L for the integration
    rule (length = number of sections). ``fiber_data`` maps
    (eid, section_index_1based) → ndarray of shape ``(n_fibers, 6)``
    with columns ``[y, z, area, material_tag, stress, strain]``.

    ``ops.eleResponse`` joins multi-token calls with the same routing
    used in the line-station fake.
    """

    def __init__(self) -> None:
        self.ele_class: dict[int, str] = {}
        self.integration_points: dict[int, np.ndarray] = {}
        self.fiber_data: dict[tuple[int, int], np.ndarray] = {}

    def eleType(self, eid: int) -> str:
        return self.ele_class[int(eid)]

    def eleResponse(self, eid: int, *args) -> list[float]:
        if len(args) == 1 and args[0] == "integrationPoints":
            return list(self.integration_points[int(eid)].astype(float))
        if (
            len(args) == 3
            and args[0] == "section"
            and args[2] == "fiberData2"
        ):
            sec = int(args[1])
            arr = self.fiber_data[(int(eid), sec)]
            return list(arr.flatten().astype(float))
        raise KeyError(f"unhandled eleResponse args: {args}")

    def reactions(self) -> None: ...


# =====================================================================
# Mock fem
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


def _spec_with(*records, snapshot_id) -> ResolvedRecorderSpec:
    return ResolvedRecorderSpec(
        fem_snapshot_id=snapshot_id,
        records=tuple(records),
    )


def _fiber_block(geom_then_state) -> np.ndarray:
    """Helper: stack ``[y, z, area, mat, stress, strain]`` rows."""
    return np.asarray(geom_then_state, dtype=np.float64)


# =====================================================================
# Single-element, single-section, two-fiber capture
# =====================================================================

class TestSingleBeamSingleSection:
    def test_two_step_capture_writes_fiber_slab(self, tmp_path: Path) -> None:
        """One ForceBeamColumn3d, 2 sections, 3 fibers each, 2 time steps."""
        fem = _MockFem([1, 2])
        spec = _spec_with(
            ResolvedRecorderRecord(
                category="fibers", name="rebar",
                components=("fiber_stress", "fiber_strain"),
                dt=None, n_steps=None,
                element_ids=np.array([10]),
            ),
            snapshot_id=fem.snapshot_id,
        )

        ops = _FakeOpsFibers()
        ops.ele_class[10] = "ForceBeamColumn3d"
        # Two sections at xi*L = [0.25, 0.75] (any positive values work
        # — capturer only uses the count).
        ops.integration_points[10] = np.array([0.25, 0.75], dtype=np.float64)

        # Three fibers per section. Geometry: identical across steps.
        # Stress/strain: change with step.
        # Section 1, fibers 0, 1, 2 — geometry frozen
        sec1_geom = [
            (0.10, 0.10, 0.01, 5),    # fiber 0: matTag=5
            (-0.10, 0.10, 0.02, 6),   # fiber 1: matTag=6
            (0.0, -0.10, 0.03, 5),    # fiber 2: matTag=5
        ]
        sec2_geom = [
            (0.20, 0.20, 0.04, 7),
            (-0.20, 0.20, 0.05, 8),
        ]

        path = tmp_path / "cap.h5"
        with DomainCapture(spec, path, fem, ndm=3, ndf=6, ops=ops) as cap:
            cap.begin_stage("static", kind="static")
            for t_idx, t in enumerate((0.0, 1.0)):
                # Stress = 100*step + 10*sec + fiberIdx; strain = stress/1000
                sec1_state = []
                for f, (y, z, a, m) in enumerate(sec1_geom):
                    sigma = 100.0 * t_idx + 10.0 * 1 + f
                    eps = sigma / 1000.0
                    sec1_state.append([y, z, a, m, sigma, eps])
                sec2_state = []
                for f, (y, z, a, m) in enumerate(sec2_geom):
                    sigma = 100.0 * t_idx + 10.0 * 2 + f
                    eps = sigma / 1000.0
                    sec2_state.append([y, z, a, m, sigma, eps])
                ops.fiber_data[(10, 1)] = _fiber_block(sec1_state)
                ops.fiber_data[(10, 2)] = _fiber_block(sec2_state)
                cap.step(t=t)
            cap.end_stage()

        # Read back
        with Results.from_native(path, fem=fem) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.fibers.get(component="fiber_stress")
            # 2 sections × (3 + 2) fibers wait — sec1 has 3, sec2 has 2,
            # total = 5 fibers across this single element
            assert slab.values.shape == (2, 5)
            # element_index repeats element_id for each fiber
            np.testing.assert_array_equal(
                slab.element_index, [10, 10, 10, 10, 10],
            )
            # gp_index: 0,0,0 (sec 1) then 1,1 (sec 2)
            np.testing.assert_array_equal(
                slab.gp_index, [0, 0, 0, 1, 1],
            )
            # Geometry is captured on first step and constant
            np.testing.assert_allclose(
                slab.y, [0.10, -0.10, 0.0, 0.20, -0.20],
            )
            np.testing.assert_allclose(
                slab.area, [0.01, 0.02, 0.03, 0.04, 0.05],
            )
            np.testing.assert_array_equal(
                slab.material_tag, [5, 6, 5, 7, 8],
            )
            # Stress at t=0: sec1 fibers (10, 11, 12), sec2 fibers (20, 21)
            # Stress at t=1: sec1 fibers (110, 111, 112), sec2 (120, 121)
            np.testing.assert_allclose(
                slab.values,
                [[10.0, 11.0, 12.0, 20.0, 21.0],
                 [110.0, 111.0, 112.0, 120.0, 121.0]],
            )

            # Strain comes through too
            strain_slab = s.elements.fibers.get(component="fiber_strain")
            np.testing.assert_allclose(
                strain_slab.values, slab.values / 1000.0,
            )


# =====================================================================
# Stress-only record — strain is read but discarded
# =====================================================================

class TestStressOnlyFilter:
    def test_strain_not_written_when_not_requested(
        self, tmp_path: Path,
    ) -> None:
        fem = _MockFem([1])
        spec = _spec_with(
            ResolvedRecorderRecord(
                category="fibers", name="stress_only",
                components=("fiber_stress",),       # no strain
                dt=None, n_steps=None,
                element_ids=np.array([10]),
            ),
            snapshot_id=fem.snapshot_id,
        )
        ops = _FakeOpsFibers()
        ops.ele_class[10] = "ForceBeamColumn3d"
        ops.integration_points[10] = np.array([0.5])
        ops.fiber_data[(10, 1)] = _fiber_block([
            [0.0, 0.0, 0.01, 1, 42.0, 0.001],
        ])

        path = tmp_path / "cap.h5"
        with DomainCapture(spec, path, fem, ops=ops) as cap:
            cap.begin_stage("g", kind="static")
            cap.step(t=0.0)
            cap.end_stage()

        with Results.from_native(path, fem=fem) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.fibers.get(component="fiber_stress")
            np.testing.assert_allclose(slab.values, [[42.0]])
            # Strain wasn't recorded — reader yields an empty slab
            # (no group has the component dataset).
            strain = s.elements.fibers.get(component="fiber_strain")
            assert strain.values.shape[1] == 0


# =====================================================================
# Multi-class record — beams of two different classes share one record
# =====================================================================

class TestMultiClassFiberCapture:
    def test_two_classes_split_into_separate_groups(
        self, tmp_path: Path,
    ) -> None:
        fem = _MockFem([1])
        spec = _spec_with(
            ResolvedRecorderRecord(
                category="fibers", name="all_beams",
                components=("fiber_stress",),
                dt=None, n_steps=None,
                element_ids=np.array([10, 20]),
            ),
            snapshot_id=fem.snapshot_id,
        )
        ops = _FakeOpsFibers()
        ops.ele_class[10] = "ForceBeamColumn3d"
        ops.ele_class[20] = "DispBeamColumn3d"
        ops.integration_points[10] = np.array([0.5])
        ops.integration_points[20] = np.array([0.5])
        ops.fiber_data[(10, 1)] = _fiber_block([
            [0.0, 0.0, 0.01, 1, 1.0, 0.0001],
            [0.1, 0.0, 0.02, 1, 2.0, 0.0002],
        ])
        ops.fiber_data[(20, 1)] = _fiber_block([
            [0.0, 0.0, 0.05, 2, 100.0, 0.01],
        ])

        path = tmp_path / "cap.h5"
        with DomainCapture(spec, path, fem, ops=ops) as cap:
            cap.begin_stage("g", kind="static")
            cap.step(t=0.0)
            cap.end_stage()

        with Results.from_native(path, fem=fem) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.fibers.get(component="fiber_stress")
            # 2 fibers (eid=10) + 1 fiber (eid=20) = 3 fibers total
            assert slab.values.shape == (1, 3)
            # h5 group iteration order is alphabetical, not insertion;
            # check the (eid → value) mapping rather than absolute order.
            pairs = sorted(zip(
                slab.element_index.tolist(),
                slab.values[0].tolist(),
            ))
            assert pairs == [(10, 1.0), (10, 2.0), (20, 100.0)]


# =====================================================================
# Skipped: uncatalogued class
# =====================================================================

class TestSkippedUncatalogued:
    def test_uncatalogued_class_is_skipped_with_reason(
        self, tmp_path: Path,
    ) -> None:
        fem = _MockFem([1])
        spec = _spec_with(
            ResolvedRecorderRecord(
                category="fibers", name="r",
                components=("fiber_stress",),
                dt=None, n_steps=None,
                element_ids=np.array([10, 99]),
            ),
            snapshot_id=fem.snapshot_id,
        )
        ops = _FakeOpsFibers()
        ops.ele_class[10] = "ForceBeamColumn3d"
        ops.ele_class[99] = "NotABeam"          # not in FIBER_CATALOG
        ops.integration_points[10] = np.array([0.5])
        ops.fiber_data[(10, 1)] = _fiber_block([
            [0.0, 0.0, 0.01, 1, 1.0, 0.0001],
        ])

        path = tmp_path / "cap.h5"
        with DomainCapture(spec, path, fem, ops=ops) as cap:
            cap.begin_stage("g", kind="static")
            cap.step(t=0.0)
            cap.end_stage()

            # The capturer tracks skipped elements with the reason
            assert len(cap._fiber_capturers) == 1
            skipped = cap._fiber_capturers[0].skipped_elements
            assert len(skipped) == 1
            eid, reason = skipped[0]
            assert eid == 99
            assert "NotABeam" in reason
            assert "FIBER_CATALOG" in reason


# =====================================================================
# Validation — bad components and bad fiberData2 sizes
# =====================================================================

class TestValidation:
    def test_record_with_no_recognised_components_raises(
        self, tmp_path: Path,
    ) -> None:
        fem = _MockFem([1])
        spec = _spec_with(
            ResolvedRecorderRecord(
                category="fibers", name="bad",
                components=("displacement_x",),       # not a fiber comp
                dt=None, n_steps=None,
                element_ids=np.array([10]),
            ),
            snapshot_id=fem.snapshot_id,
        )
        ops = _FakeOpsFibers()
        with pytest.raises(ValueError, match="recognised components"):
            with DomainCapture(spec, tmp_path / "x.h5", fem, ops=ops):
                pass

    def test_fiberData2_wrong_size_raises_at_step(
        self, tmp_path: Path,
    ) -> None:
        fem = _MockFem([1])
        spec = _spec_with(
            ResolvedRecorderRecord(
                category="fibers", name="r",
                components=("fiber_stress",),
                dt=None, n_steps=None,
                element_ids=np.array([10]),
            ),
            snapshot_id=fem.snapshot_id,
        )
        ops = _FakeOpsFibers()
        ops.ele_class[10] = "ForceBeamColumn3d"
        ops.integration_points[10] = np.array([0.5])
        # Length 5 — not a multiple of 6.
        ops.fiber_data[(10, 1)] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with pytest.raises(RuntimeError, match="multiple of 6"):
            with DomainCapture(spec, tmp_path / "x.h5", fem, ops=ops) as cap:
                cap.begin_stage("g", kind="static")
                cap.step(t=0.0)
