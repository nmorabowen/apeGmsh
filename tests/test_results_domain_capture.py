"""Phase 7 — DomainCapture in-process recording (mocked openseespy).

These tests use a fake ``ops`` module so the full flow can be
exercised without running an actual OpenSees analysis. They check
the per-step buffering, multi-stage handling, and end-to-end
integration with NativeReader.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apeGmsh.results.capture._domain import DomainCapture
from apeGmsh.solvers._recorder_specs import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)


# =====================================================================
# Fake ops module (deterministic returns)
# =====================================================================

class _FakeOps:
    """Minimal openseespy stand-in for capture tests.

    Stores configurable return tables for each ops.* call.
    """

    def __init__(self) -> None:
        self.disp_table: dict[tuple[int, int], float] = {}
        self.vel_table: dict[tuple[int, int], float] = {}
        self.accel_table: dict[tuple[int, int], float] = {}
        self.reaction_table: dict[tuple[int, int], float] = {}
        self.unbalance_table: dict[tuple[int, int], float] = {}
        self.pressure_table: dict[int, float] = {}
        # Mode shapes: {(nid, mode, dof): value}
        self.eigen_vector_table: dict[tuple[int, int, int], float] = {}
        self.eigenvalues: list[float] = []
        self.reactions_called: int = 0

    # Per-step accessors
    def nodeDisp(self, nid: int, dof: int) -> float:
        return self.disp_table.get((nid, dof), 0.0)

    def nodeVel(self, nid: int, dof: int) -> float:
        return self.vel_table.get((nid, dof), 0.0)

    def nodeAccel(self, nid: int, dof: int) -> float:
        return self.accel_table.get((nid, dof), 0.0)

    def nodeReaction(self, nid: int, dof: int) -> float:
        return self.reaction_table.get((nid, dof), 0.0)

    def nodeUnbalance(self, nid: int, dof: int) -> float:
        return self.unbalance_table.get((nid, dof), 0.0)

    def nodePressure(self, nid: int) -> float:
        return self.pressure_table.get(nid, 0.0)

    def reactions(self) -> None:
        self.reactions_called += 1

    # Eigen
    def eigen(self, n_modes: int) -> list[float]:
        return list(self.eigenvalues[:n_modes])

    def nodeEigenvector(self, nid: int, mode: int, dof: int) -> float:
        return self.eigen_vector_table.get((nid, mode, dof), 0.0)


# =====================================================================
# FEMData fixture (synthetic mock — no gmsh needed for capture flow)
# =====================================================================

class _MockFem:
    """Tiny mock FEMData with the surface area capture needs.

    Uses the real ``compute_snapshot_id`` so the bind-contract round
    trip works (writer embeds the hash, reader recomputes from the
    reconstructed data, the two must match).
    """
    def __init__(self, node_ids, salt: int = 0) -> None:
        ids = np.asarray(node_ids, dtype=np.int64)
        # Add a tiny per-salt offset so two MockFem instances with the
        # same node_ids produce different hashes when needed.
        coords = np.zeros((ids.size, 3), dtype=np.float64) + float(salt) * 1e-9
        self.nodes = SimpleNamespace(ids=ids, coords=coords)
        # Empty elements iterator with the shape compute_snapshot_id expects.
        self.elements = []

    @property
    def snapshot_id(self) -> str:
        from apeGmsh.mesh._femdata_hash import compute_snapshot_id
        return compute_snapshot_id(self)

    def to_native_h5(self, group) -> None:
        # Minimal embedded snapshot — node IDs + coords + snapshot_id attr.
        group.attrs["snapshot_id"] = self.snapshot_id
        group.attrs["ndm"] = 3
        group.attrs["ndf"] = 6
        group.attrs["model_name"] = ""
        group.attrs["units"] = ""
        nodes_grp = group.create_group("nodes")
        nodes_grp.create_dataset(
            "ids", data=np.asarray(self.nodes.ids, dtype=np.int64),
        )
        nodes_grp.create_dataset(
            "coords", data=np.asarray(self.nodes.coords, dtype=np.float64),
        )
        group.create_group("elements")


def _make_spec(*records, snapshot_id):
    return ResolvedRecorderSpec(
        fem_snapshot_id=snapshot_id,
        records=tuple(records),
    )


# =====================================================================
# Basic node capture
# =====================================================================

def test_node_capture_displacement(tmp_path: Path) -> None:
    fem = _MockFem([1, 2, 3])
    spec = _make_spec(
        ResolvedRecorderRecord(
            category="nodes", name="all_disp",
            components=("displacement_x", "displacement_y", "displacement_z"),
            dt=None, n_steps=None,
            node_ids=np.array([1, 2, 3]),
        ),
        snapshot_id=fem.snapshot_id,
    )
    fake = _FakeOps()
    # Step 0: disp_x = 0.1*nid for each node
    for nid in (1, 2, 3):
        fake.disp_table[(nid, 1)] = 0.1 * nid
        fake.disp_table[(nid, 2)] = 0.2 * nid
        fake.disp_table[(nid, 3)] = 0.0

    path = tmp_path / "run.h5"
    with DomainCapture(spec, path, fem, ndm=3, ndf=6, ops=fake) as cap:
        cap.begin_stage("static", kind="static")
        cap.step(t=0.5)
        cap.end_stage()

    from apeGmsh.results import Results
    with Results.from_native(path, fem=fem) as r:
        slab = r.nodes.get(component="displacement_x")
        np.testing.assert_allclose(slab.values, [[0.1, 0.2, 0.3]])
        np.testing.assert_allclose(slab.time, [0.5])
        slab_y = r.nodes.get(component="displacement_y")
        np.testing.assert_allclose(slab_y.values, [[0.2, 0.4, 0.6]])


def test_multi_step_capture(tmp_path: Path) -> None:
    fem = _MockFem([1, 2])
    spec = _make_spec(
        ResolvedRecorderRecord(
            category="nodes", name="dyn",
            components=("displacement_x",),
            dt=None, n_steps=None,
            node_ids=np.array([1, 2]),
        ),
        snapshot_id=fem.snapshot_id,
    )
    fake = _FakeOps()

    path = tmp_path / "run.h5"
    with DomainCapture(spec, path, fem, ops=fake) as cap:
        cap.begin_stage("dyn")
        for k, t in enumerate([0.0, 0.1, 0.2, 0.3]):
            # disp varies with step: simple linear ramp
            fake.disp_table[(1, 1)] = float(k) * 0.01
            fake.disp_table[(2, 1)] = float(k) * 0.02
            cap.step(t=t)
        cap.end_stage()

    from apeGmsh.results import Results
    with Results.from_native(path, fem=fem) as r:
        slab = r.nodes.get(component="displacement_x")
        np.testing.assert_allclose(slab.values, [
            [0.0, 0.0],
            [0.01, 0.02],
            [0.02, 0.04],
            [0.03, 0.06],
        ])
        np.testing.assert_allclose(slab.time, [0.0, 0.1, 0.2, 0.3])


# =====================================================================
# Reaction capture triggers ops.reactions()
# =====================================================================

def test_reaction_triggers_reactions_call(tmp_path: Path) -> None:
    fem = _MockFem([10])
    spec = _make_spec(
        ResolvedRecorderRecord(
            category="nodes", name="r",
            components=("reaction_force_x", "reaction_force_y", "reaction_force_z"),
            dt=None, n_steps=None,
            node_ids=np.array([10]),
        ),
        snapshot_id=fem.snapshot_id,
    )
    fake = _FakeOps()
    fake.reaction_table[(10, 1)] = -100.0
    fake.reaction_table[(10, 2)] = 0.0
    fake.reaction_table[(10, 3)] = 50.0

    path = tmp_path / "run.h5"
    with DomainCapture(spec, path, fem, ops=fake) as cap:
        cap.begin_stage("g", kind="static")
        cap.step(t=0.0)
        cap.step(t=1.0)
        cap.end_stage()

    # ops.reactions() should be called once per step
    assert fake.reactions_called == 2

    from apeGmsh.results import Results
    with Results.from_native(path, fem=fem) as r:
        slab = r.nodes.get(component="reaction_force_x")
        np.testing.assert_allclose(slab.values, [[-100.0], [-100.0]])


def test_no_reactions_call_without_reaction_components(tmp_path: Path) -> None:
    fem = _MockFem([1])
    spec = _make_spec(
        ResolvedRecorderRecord(
            category="nodes", name="r",
            components=("displacement_x",),
            dt=None, n_steps=None,
            node_ids=np.array([1]),
        ),
        snapshot_id=fem.snapshot_id,
    )
    fake = _FakeOps()
    path = tmp_path / "run.h5"
    with DomainCapture(spec, path, fem, ops=fake) as cap:
        cap.begin_stage("g")
        cap.step(t=0.0)
        cap.end_stage()
    assert fake.reactions_called == 0


# =====================================================================
# Multi-stage capture
# =====================================================================

def test_two_stage_capture(tmp_path: Path) -> None:
    fem = _MockFem([1, 2])
    spec = _make_spec(
        ResolvedRecorderRecord(
            category="nodes", name="r",
            components=("displacement_x",),
            dt=None, n_steps=None,
            node_ids=np.array([1, 2]),
        ),
        snapshot_id=fem.snapshot_id,
    )
    fake = _FakeOps()

    path = tmp_path / "run.h5"
    with DomainCapture(spec, path, fem, ops=fake) as cap:
        # Gravity
        cap.begin_stage("gravity", kind="static")
        fake.disp_table[(1, 1)] = 0.001
        fake.disp_table[(2, 1)] = 0.002
        cap.step(t=1.0)
        cap.end_stage()
        # Dynamic
        cap.begin_stage("dynamic", kind="transient")
        for k, t in enumerate([0.0, 0.1]):
            fake.disp_table[(1, 1)] = float(k) * 10.0
            fake.disp_table[(2, 1)] = float(k) * 20.0
            cap.step(t=t)
        cap.end_stage()

    from apeGmsh.results import Results
    with Results.from_native(path, fem=fem) as r:
        stages = r.stages
        names = sorted(s.name for s in stages)
        assert names == ["dynamic", "gravity"]

        grav = r.stage("gravity")
        assert grav.kind == "static"
        np.testing.assert_allclose(
            grav.nodes.get(component="displacement_x").values,
            [[0.001, 0.002]],
        )

        dyn = r.stage("dynamic")
        assert dyn.kind == "transient"
        assert dyn.n_steps == 2


# =====================================================================
# Modal capture
# =====================================================================

def test_modal_capture(tmp_path: Path) -> None:
    fem = _MockFem([1, 2, 3])
    spec = _make_spec(
        ResolvedRecorderRecord(
            category="modal", name="modes",
            components=(),
            dt=None, n_steps=None,
            n_modes=2,
        ),
        snapshot_id=fem.snapshot_id,
    )
    fake = _FakeOps()
    fake.eigenvalues = [100.0, 400.0]
    # Mode 1: simple shape
    for nid in (1, 2, 3):
        fake.eigen_vector_table[(nid, 1, 1)] = 0.1 * nid
        fake.eigen_vector_table[(nid, 1, 2)] = 0.0
        fake.eigen_vector_table[(nid, 1, 3)] = 0.0
        fake.eigen_vector_table[(nid, 2, 1)] = 0.0
        fake.eigen_vector_table[(nid, 2, 2)] = 0.2 * nid
        fake.eigen_vector_table[(nid, 2, 3)] = 0.0

    path = tmp_path / "run.h5"
    with DomainCapture(spec, path, fem, ndm=3, ndf=3, ops=fake) as cap:
        cap.capture_modes()

    from apeGmsh.results import Results
    with Results.from_native(path, fem=fem) as r:
        modes = sorted(r.modes, key=lambda m: m.mode_index)
        assert len(modes) == 2
        assert modes[0].mode_index == 1
        assert modes[0].eigenvalue == pytest.approx(100.0)
        assert modes[1].eigenvalue == pytest.approx(400.0)

        shape1_x = modes[0].nodes.get(component="displacement_x").values
        np.testing.assert_allclose(shape1_x, [[0.1, 0.2, 0.3]])
        shape2_y = modes[1].nodes.get(component="displacement_y").values
        np.testing.assert_allclose(shape2_y, [[0.2, 0.4, 0.6]])


def test_modal_capture_with_rotational_dofs(tmp_path: Path) -> None:
    """ndf=6 captures rotation_x/y/z too."""
    fem = _MockFem([1])
    spec = _make_spec(
        ResolvedRecorderRecord(
            category="modal", name="m",
            components=(), dt=None, n_steps=None, n_modes=1,
        ),
        snapshot_id=fem.snapshot_id,
    )
    fake = _FakeOps()
    fake.eigenvalues = [50.0]
    for dof in range(1, 7):
        fake.eigen_vector_table[(1, 1, dof)] = float(dof) * 0.1

    path = tmp_path / "run.h5"
    with DomainCapture(spec, path, fem, ndm=3, ndf=6, ops=fake) as cap:
        cap.capture_modes()

    from apeGmsh.results import Results
    with Results.from_native(path, fem=fem) as r:
        m = r.modes[0]
        assert "displacement_x" in m.nodes.available_components()
        assert "rotation_x" in m.nodes.available_components()
        np.testing.assert_allclose(
            m.nodes.get(component="rotation_z").values, [[0.6]]
        )


# =====================================================================
# Snapshot mismatch detection
# =====================================================================

def test_snapshot_mismatch_raises(tmp_path: Path) -> None:
    """spec resolved against different fem → DomainCapture refuses."""
    fem_orig = _MockFem([1, 2], salt=0)
    fem_other = _MockFem([1, 2], salt=1)   # different coords → different hash
    assert fem_orig.snapshot_id != fem_other.snapshot_id  # sanity

    spec = _make_spec(
        ResolvedRecorderRecord(
            category="nodes", name="r",
            components=("displacement_x",),
            dt=None, n_steps=None,
            node_ids=np.array([1, 2]),
        ),
        snapshot_id=fem_orig.snapshot_id,
    )
    fake = _FakeOps()
    path = tmp_path / "run.h5"

    with pytest.raises(RuntimeError, match="snapshot_id mismatch"):
        with DomainCapture(spec, path, fem_other, ops=fake):
            pass


# =====================================================================
# Element-level records still without catalog wiring: NotImplementedError
# =====================================================================
#
# Phase 11a wired ``gauss`` records (continuum stress/strain) — see
# ``test_results_domain_capture_gauss.py``. Phase 11b wired
# ``line_stations`` and per-element-node ``elements``. Phase 11e wired
# beam ``fibers`` — see ``test_results_domain_capture_fibers.py``.
# Phase 11f wired layered shells (``layers``) — see
# ``test_results_domain_capture_layers.py``. The capturer requires
# ``layer_section_metadata`` on the resolved record; absence raises.

def test_layer_records_without_metadata_raise(tmp_path: Path) -> None:
    """A ``layers`` record resolved without an OpenSees back-reference
    has no ``layer_section_metadata`` and raises a clear error at
    DomainCapture open time."""
    fem = _MockFem([1])
    spec = _make_spec(
        ResolvedRecorderRecord(
            category="layers", name="r",
            components=("fiber_stress",),
            dt=None, n_steps=None,
            element_ids=np.array([10, 20]),
            # layer_section_metadata=None (default) — the failure mode.
        ),
        snapshot_id=fem.snapshot_id,
    )
    fake = _FakeOps()

    path = tmp_path / "run.h5"
    with pytest.raises(ValueError, match="layer_section_metadata"):
        with DomainCapture(spec, path, fem, ops=fake):
            pass


# =====================================================================
# spec.capture(...) factory
# =====================================================================

def test_spec_capture_factory(tmp_path: Path) -> None:
    """ResolvedRecorderSpec.capture(...) returns a working DomainCapture."""
    fem = _MockFem([1])
    spec = _make_spec(
        ResolvedRecorderRecord(
            category="nodes", name="r",
            components=("displacement_x",),
            dt=None, n_steps=None,
            node_ids=np.array([1]),
        ),
        snapshot_id=fem.snapshot_id,
    )
    fake = _FakeOps()
    fake.disp_table[(1, 1)] = 0.123

    path = tmp_path / "run.h5"
    with spec.capture(path, fem, ops=fake) as cap:
        cap.begin_stage("g", kind="static")
        cap.step(t=0.0)
        cap.end_stage()

    from apeGmsh.results import Results
    with Results.from_native(path, fem=fem) as r:
        np.testing.assert_allclose(
            r.nodes.get(component="displacement_x").values, [[0.123]],
        )
