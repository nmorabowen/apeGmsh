"""
Unit tests for MP constraint emission — Phase 7b, ADR 0022.

This file complements ``tests/test_constraint_emission.py`` (kernel
iterator routing contract — PR-B/PR-C/PR-D coverage from the
constraints deep-review).  Where that file pins the broker's
iterator semantics, this file pins the bridge's emit path:

* Per-emitter shape — TclEmitter, PyEmitter, LiveOpsEmitter,
  H5Emitter, RecordingEmitter — each implementing the five new
  Protocol methods.
* The :func:`emit_mp_constraints` fan-out: phantom-node pre-step,
  rigid_link / equal_dof / rigid_diaphragm / kinematic_coupling
  dispatch, surface-coupling routing through ASDEmbeddedNodeElement.
* INV-1 (runnable deck) lives in ``tests/opensees/integration/
  test_runnable_deck.py``; this file covers INV-2 (name round-trip)
  / INV-3 (phantom ordering) / INV-4 (every emitter implements every
  method) at the unit level.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pytest

from apeGmsh._kernel.records._constraints import (
    InterpolationRecord,
    NodeGroupRecord,
    NodePairRecord,
    NodeToSurfaceRecord,
    SurfaceCouplingRecord,
)
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build import emit_mp_constraints
from apeGmsh.opensees._internal.tag_allocator import TagAllocator
from apeGmsh.opensees.emitter.h5 import H5Emitter
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.section.fiber import FiberPoint

from tests.fixtures.schema import OPENSEES_CURRENT
from tests.opensees.fixtures.fem_stub import make_two_column_frame


# ---------------------------------------------------------------------------
# Section 1 — Protocol shape on each concrete emitter (INV-4)
# ---------------------------------------------------------------------------


class TestProtocolShape:
    """INV-4: every concrete emitter implements every method."""

    @pytest.mark.parametrize(
        "method_name",
        (
            "equalDOF",
            "rigidLink",
            "rigidDiaphragm",
            "embeddedNode",
            "mp_constraint_comment",
        ),
    )
    def test_tcl_has_method(self, method_name: str) -> None:
        e = TclEmitter()
        assert callable(getattr(e, method_name, None))

    @pytest.mark.parametrize(
        "method_name",
        (
            "equalDOF",
            "rigidLink",
            "rigidDiaphragm",
            "embeddedNode",
            "mp_constraint_comment",
        ),
    )
    def test_py_has_method(self, method_name: str) -> None:
        e = PyEmitter()
        assert callable(getattr(e, method_name, None))

    @pytest.mark.parametrize(
        "method_name",
        (
            "equalDOF",
            "rigidLink",
            "rigidDiaphragm",
            "embeddedNode",
            "mp_constraint_comment",
        ),
    )
    def test_h5_has_method(self, method_name: str) -> None:
        e = H5Emitter()
        assert callable(getattr(e, method_name, None))

    @pytest.mark.parametrize(
        "method_name",
        (
            "equalDOF",
            "rigidLink",
            "rigidDiaphragm",
            "embeddedNode",
            "mp_constraint_comment",
        ),
    )
    def test_recording_has_method(self, method_name: str) -> None:
        e = RecordingEmitter()
        assert callable(getattr(e, method_name, None))

    def test_node_accepts_ndf_kwarg_on_tcl(self) -> None:
        e = TclEmitter()
        e.node(99, 1.0, 2.0, 3.0, ndf=6)
        assert any("-ndf 6" in line for line in e.lines())

    def test_node_accepts_ndf_kwarg_on_py(self) -> None:
        e = PyEmitter()
        e.node(99, 1.0, 2.0, 3.0, ndf=6)
        joined = "\n".join(e.lines())
        assert "-ndf" in joined and "99" in joined

    def test_node_accepts_ndf_kwarg_on_recording(self) -> None:
        e = RecordingEmitter()
        e.node(99, 1.0, 2.0, 3.0, ndf=6)
        assert e.calls[0] == ("node", (99, 1.0, 2.0, 3.0), {"ndf": 6})

    def test_node_without_ndf_records_no_kwargs(self) -> None:
        """Regular nodes (ndf=None) record an empty kwargs dict so the
        existing 6038-test baseline shape is unchanged."""
        e = RecordingEmitter()
        e.node(1, 0.0, 0.0, 0.0)
        assert e.calls[0] == ("node", (1, 0.0, 0.0, 0.0), {})


# ---------------------------------------------------------------------------
# Section 2 — Per-kind / per-emitter line shapes
# ---------------------------------------------------------------------------


class TestEqualDOF:
    def test_emits_tcl_line(self) -> None:
        e = TclEmitter()
        e.equalDOF(1, 2, 1, 2, 3)
        assert "equalDOF 1 2 1 2 3" in e.lines()

    def test_emits_py_line(self) -> None:
        e = PyEmitter()
        e.equalDOF(1, 2, 1, 2, 3)
        assert "ops.equalDOF(1, 2, 1, 2, 3)" in e.lines()

    def test_recorded(self) -> None:
        e = RecordingEmitter()
        e.equalDOF(1, 2, 1, 2, 3)
        assert e.calls == [("equalDOF", (1, 2, 1, 2, 3), {})]

    def test_emits_to_h5_dataset(self, tmp_path: Path) -> None:
        e = H5Emitter()
        e.model(ndm=3, ndf=6)
        e.equalDOF(1, 2, 1, 2, 3)
        e.equalDOF(3, 4, 1, 2, 3, 4, 5, 6)
        out = tmp_path / "x.h5"
        e.write(str(out))
        with h5py.File(out, "r") as f:
            ds = f["opensees/constraints/equalDOF"][:]
        assert len(ds) == 2
        assert int(ds[0]["master"]) == 1
        assert int(ds[0]["slave"]) == 2
        assert list(int(d) for d in ds[1]["dofs"]) == [1, 2, 3, 4, 5, 6]


class TestRigidLink:
    def test_emits_beam_tcl(self) -> None:
        e = TclEmitter()
        e.rigidLink("beam", 1, 2)
        assert "rigidLink beam 1 2" in e.lines()

    def test_emits_bar_tcl(self) -> None:
        e = TclEmitter()
        e.rigidLink("bar", 1, 2)
        assert "rigidLink bar 1 2" in e.lines()

    def test_emits_beam_py(self) -> None:
        e = PyEmitter()
        e.rigidLink("beam", 1, 2)
        assert "ops.rigidLink('beam', 1, 2)" in e.lines()

    def test_recorded(self) -> None:
        e = RecordingEmitter()
        e.rigidLink("beam", 1, 2)
        assert e.calls == [("rigidLink", ("beam", 1, 2), {})]

    def test_emits_to_h5_dataset(self, tmp_path: Path) -> None:
        e = H5Emitter()
        e.model(ndm=3, ndf=6)
        e.rigidLink("beam", 1, 2)
        e.rigidLink("bar", 3, 4)
        out = tmp_path / "x.h5"
        e.write(str(out))
        with h5py.File(out, "r") as f:
            ds = f["opensees/constraints/rigidLink"][:]
        assert len(ds) == 2
        # h5py returns bytes for variable-length strings.
        kinds = [
            (k.decode("utf-8") if isinstance(k, bytes) else k)
            for k in ds["kind"]
        ]
        assert kinds == ["beam", "bar"]


class TestRigidDiaphragm:
    def test_emits_tcl_line(self) -> None:
        e = TclEmitter()
        e.rigidDiaphragm(3, 1, 2, 3, 4)
        assert "rigidDiaphragm 3 1 2 3 4" in e.lines()

    def test_emits_py_line(self) -> None:
        e = PyEmitter()
        e.rigidDiaphragm(3, 1, 2, 3, 4)
        assert "ops.rigidDiaphragm(3, 1, 2, 3, 4)" in e.lines()

    def test_recorded(self) -> None:
        e = RecordingEmitter()
        e.rigidDiaphragm(3, 1, 2, 3)
        assert e.calls == [("rigidDiaphragm", (3, 1, 2, 3), {})]

    def test_emits_to_h5_dataset(self, tmp_path: Path) -> None:
        e = H5Emitter()
        e.model(ndm=3, ndf=6)
        e.rigidDiaphragm(3, 100, 1, 2, 3, 4)
        out = tmp_path / "x.h5"
        e.write(str(out))
        with h5py.File(out, "r") as f:
            ds = f["opensees/constraints/rigidDiaphragm"][:]
        assert len(ds) == 1
        assert int(ds[0]["perp_dir"]) == 3
        assert int(ds[0]["master"]) == 100
        n = int(ds[0]["n_slaves"])
        assert n == 4
        assert list(int(s) for s in ds[0]["slaves"][:n]) == [1, 2, 3, 4]


class TestEmbeddedNode:
    def test_emits_tcl_line(self) -> None:
        # ADR 0035: -K $K is always emitted, with the C++ default
        # (1.0e18) when the user leaves ``stiffness`` untouched.
        e = TclEmitter()
        e.embeddedNode(1000, 5, 10, 1, 2, 3, 4)
        assert (
            "element ASDEmbeddedNodeElement 1000 5 10 1 2 3 4 -K 1e+18"
            in e.lines()
        )

    def test_emits_py_line(self) -> None:
        e = PyEmitter()
        e.embeddedNode(1000, 5, 10, 1, 2, 3, 4)
        assert (
            "ops.element('ASDEmbeddedNodeElement', 1000, 5, 10, 1, 2, 3, 4, '-K', 1e+18)"
            in e.lines()
        )

    def test_recorded(self) -> None:
        e = RecordingEmitter()
        e.embeddedNode(1000, 5, 10, 1, 2, 3, 4)
        assert e.calls == [
            (
                "embeddedNode",
                (1000, 5, 10, 1, 2, 3, 4),
                {
                    "stiffness": 1.0e18,
                    "stiffness_p": None,
                    "rotational": False,
                    "pressure": False,
                },
            ),
        ]

    def test_emits_to_h5_dataset(self, tmp_path: Path) -> None:
        e = H5Emitter()
        e.model(ndm=3, ndf=6)
        e.embeddedNode(1000, 5, 10, 1, 2)
        out = tmp_path / "x.h5"
        e.write(str(out))
        with h5py.File(out, "r") as f:
            ds = f["opensees/constraints/embeddedNode"][:]
        assert len(ds) == 1
        assert int(ds[0]["ele_tag"]) == 1000
        assert int(ds[0]["cnode"]) == 5
        # ADR 0035 schema 2.12.0 — defaults persist as typed columns.
        assert float(ds[0]["stiffness"]) == 1.0e18
        assert int(ds[0]["has_stiffness_p"]) == 0
        assert int(ds[0]["rotational"]) == 0
        assert int(ds[0]["pressure"]) == 0

    # -- ADR 0035: optional flag exposure ---------------------------------

    def test_tcl_emits_stiffness_override(self) -> None:
        e = TclEmitter()
        e.embeddedNode(1000, 5, 10, 1, 2, 3, stiffness=1.0e8)
        assert (
            "element ASDEmbeddedNodeElement 1000 5 10 1 2 3 -K 100000000.0"
            in e.lines()
        )

    def test_tcl_emits_rotational_flag(self) -> None:
        e = TclEmitter()
        e.embeddedNode(1000, 5, 10, 1, 2, 3, rotational=True)
        assert (
            "element ASDEmbeddedNodeElement 1000 5 10 1 2 3 -rot -K 1e+18"
            in e.lines()
        )

    def test_tcl_emits_pressure_with_kp(self) -> None:
        e = TclEmitter()
        e.embeddedNode(
            1000, 5, 10, 1, 2, 3,
            stiffness=1.0e8, stiffness_p=1.0e6, pressure=True,
        )
        line = (
            "element ASDEmbeddedNodeElement 1000 5 10 1 2 3 "
            "-p -K 100000000.0 -KP 1000000.0"
        )
        assert line in e.lines()

    def test_py_emits_flags(self) -> None:
        e = PyEmitter()
        e.embeddedNode(
            1000, 5, 10, 1, 2, 3, rotational=True, stiffness=1.0e8,
        )
        assert (
            "ops.element('ASDEmbeddedNodeElement', 1000, 5, 10, 1, 2, 3, "
            "'-rot', '-K', 100000000.0)"
            in e.lines()
        )

    def test_recording_captures_kwargs(self) -> None:
        e = RecordingEmitter()
        e.embeddedNode(
            1000, 5, 10, 1, 2, 3,
            stiffness=2.0e7, stiffness_p=3.0e7, pressure=True,
        )
        assert e.calls == [
            (
                "embeddedNode",
                (1000, 5, 10, 1, 2, 3),
                {
                    "stiffness": 2.0e7,
                    "stiffness_p": 3.0e7,
                    "rotational": False,
                    "pressure": True,
                },
            ),
        ]

    def test_h5_round_trips_flag_columns(self, tmp_path: Path) -> None:
        e = H5Emitter()
        e.model(ndm=3, ndf=6)
        e.embeddedNode(
            1000, 5, 10, 1, 2,
            stiffness=1.0e8, stiffness_p=2.0e8,
            rotational=False, pressure=True,
        )
        out = tmp_path / "x.h5"
        e.write(str(out))
        with h5py.File(out, "r") as f:
            ds = f["opensees/constraints/embeddedNode"][:]
        assert float(ds[0]["stiffness"]) == 1.0e8
        assert float(ds[0]["stiffness_p"]) == 2.0e8
        assert int(ds[0]["has_stiffness_p"]) == 1
        assert int(ds[0]["rotational"]) == 0
        assert int(ds[0]["pressure"]) == 1


# ---------------------------------------------------------------------------
# Section 3 — mp_constraint_comment INV-2 — name round-trip
# ---------------------------------------------------------------------------


class TestMpConstraintComment:
    def test_tcl_comment_round_trips(self) -> None:
        e = TclEmitter()
        e.mp_constraint_comment("floor_1")
        e.rigidDiaphragm(3, 1, 2, 3)
        lines = e.lines()
        # The comment immediately precedes the rigidDiaphragm line.
        idx = lines.index("# floor_1")
        assert lines[idx + 1] == "rigidDiaphragm 3 1 2 3"

    def test_py_comment_round_trips(self) -> None:
        e = PyEmitter()
        e.mp_constraint_comment("floor_1")
        e.rigidDiaphragm(3, 1, 2, 3)
        lines = e.lines()
        idx = lines.index("# floor_1")
        assert "ops.rigidDiaphragm(3, 1, 2, 3)" in lines[idx + 1]

    def test_recording_records_comment(self) -> None:
        e = RecordingEmitter()
        e.mp_constraint_comment("floor_1")
        assert e.calls == [("mp_constraint_comment", ("floor_1",), {})]

    def test_h5_attaches_name_to_next_record(
        self, tmp_path: Path,
    ) -> None:
        e = H5Emitter()
        e.model(ndm=3, ndf=6)
        e.mp_constraint_comment("floor_1")
        e.rigidDiaphragm(3, 100, 1, 2, 3)
        out = tmp_path / "x.h5"
        e.write(str(out))
        with h5py.File(out, "r") as f:
            ds = f["opensees/constraints/rigidDiaphragm"][:]
        name = ds[0]["name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        assert name == "floor_1"

    def test_h5_consumes_pending_name_once(
        self, tmp_path: Path,
    ) -> None:
        """The pending name is consumed by the first MP-constraint call;
        the second call gets an empty name."""
        e = H5Emitter()
        e.model(ndm=3, ndf=6)
        e.mp_constraint_comment("first")
        e.equalDOF(1, 2, 1, 2, 3)
        e.equalDOF(3, 4, 1, 2, 3)  # no preceding mp_constraint_comment
        out = tmp_path / "x.h5"
        e.write(str(out))
        with h5py.File(out, "r") as f:
            ds = f["opensees/constraints/equalDOF"][:]
        names = [
            (n.decode("utf-8") if isinstance(n, bytes) else n)
            for n in ds["name"]
        ]
        assert names[0] == "first"
        assert names[1] == ""

    def test_live_comment_is_noop(self) -> None:
        """LiveOpsEmitter's mp_constraint_comment is a no-op (live can't
        carry comments).  Just confirm the signature accepts the call
        without raising."""
        from apeGmsh.opensees.emitter.live import LiveOpsEmitter
        try:
            e = LiveOpsEmitter()
        except ImportError:
            pytest.skip("openseespy not available")
        e.mp_constraint_comment("anything")


# ---------------------------------------------------------------------------
# Section 4 — emit_mp_constraints fan-out (the build helper)
# ---------------------------------------------------------------------------


class TestEmitMpConstraintsFanout:
    """Drive ``emit_mp_constraints(emitter, fem)`` against a hand-built
    fixture and assert the call sequence is correct."""

    def test_no_constraints_no_calls(self) -> None:
        fem = make_two_column_frame()
        rec = RecordingEmitter()
        emit_mp_constraints(rec, cast(Any, fem), TagAllocator())
        assert rec.calls == []

    def test_equal_dof_record_dispatches(self) -> None:
        fem = make_two_column_frame()
        fem.add_node_constraints([
            NodePairRecord(
                kind=ConstraintKind.EQUAL_DOF,
                master_node=2, slave_node=4,
                dofs=[1, 2, 3],
            ),
        ])
        rec = RecordingEmitter()
        emit_mp_constraints(rec, cast(Any, fem), TagAllocator())
        names = [c[0] for c in rec.calls]
        assert names == ["equalDOF"]
        assert rec.calls[0] == ("equalDOF", (2, 4, 1, 2, 3), {})

    def test_rigid_beam_pair_record_dispatches(self) -> None:
        fem = make_two_column_frame()
        fem.add_node_constraints([
            NodePairRecord(
                kind=ConstraintKind.RIGID_BEAM,
                master_node=1, slave_node=3,
            ),
        ])
        rec = RecordingEmitter()
        emit_mp_constraints(rec, cast(Any, fem), TagAllocator())
        assert rec.calls[0] == ("rigidLink", ("beam", 1, 3), {})

    def test_rigid_rod_pair_record_dispatches_to_bar(self) -> None:
        fem = make_two_column_frame()
        fem.add_node_constraints([
            NodePairRecord(
                kind=ConstraintKind.RIGID_ROD,
                master_node=1, slave_node=3,
            ),
        ])
        rec = RecordingEmitter()
        emit_mp_constraints(rec, cast(Any, fem), TagAllocator())
        assert rec.calls[0] == ("rigidLink", ("bar", 1, 3), {})

    def test_rigid_diaphragm_group_record_dispatches(self) -> None:
        fem = make_two_column_frame()
        fem.add_node_constraints([
            NodeGroupRecord(
                kind=ConstraintKind.RIGID_DIAPHRAGM,
                master_node=1, slave_nodes=[2, 3, 4],
                dofs=[1, 2, 6],
                plane_normal=np.array([0.0, 0.0, 1.0]),
            ),
        ])
        rec = RecordingEmitter()
        emit_mp_constraints(rec, cast(Any, fem), TagAllocator())
        # perp_dirn=3 derived from plane_normal=(0, 0, 1).
        assert rec.calls[0] == ("rigidDiaphragm", (3, 1, 2, 3, 4), {})

    def test_rigid_body_group_record_dispatches_to_rigid_link(self) -> None:
        fem = make_two_column_frame()
        fem.add_node_constraints([
            NodeGroupRecord(
                kind=ConstraintKind.RIGID_BODY,
                master_node=1, slave_nodes=[2, 3, 4],
                dofs=[1, 2, 3, 4, 5, 6],
            ),
        ])
        rec = RecordingEmitter()
        emit_mp_constraints(rec, cast(Any, fem), TagAllocator())
        # One rigidLink per (master, slave) pair.
        names = [c[0] for c in rec.calls]
        assert names == ["rigidLink", "rigidLink", "rigidLink"]
        assert rec.calls[0][1][:3] == ("beam", 1, 2)
        assert rec.calls[1][1][:3] == ("beam", 1, 3)
        assert rec.calls[2][1][:3] == ("beam", 1, 4)

    def test_kinematic_coupling_dispatches_to_equal_dof(self) -> None:
        fem = make_two_column_frame()
        fem.add_node_constraints([
            NodeGroupRecord(
                kind=ConstraintKind.KINEMATIC_COUPLING,
                master_node=1, slave_nodes=[2, 3, 4],
                dofs=[1, 2],
            ),
        ])
        rec = RecordingEmitter()
        emit_mp_constraints(rec, cast(Any, fem), TagAllocator())
        # equalDOF preserves the per-DOF selectivity.
        names = [c[0] for c in rec.calls]
        assert names == ["equalDOF", "equalDOF", "equalDOF"]
        assert rec.calls[0] == ("equalDOF", (1, 2, 1, 2), {})
        assert rec.calls[1] == ("equalDOF", (1, 3, 1, 2), {})
        assert rec.calls[2] == ("equalDOF", (1, 4, 1, 2), {})

    def test_node_to_surface_dispatches_phantoms_links_and_equal_dofs(
        self,
    ) -> None:
        fem = make_two_column_frame()
        n2s = NodeToSurfaceRecord(
            kind=ConstraintKind.NODE_TO_SURFACE,
            master_node=100,
            slave_nodes=[2, 4],
            phantom_nodes=[200, 201],
            phantom_coords=np.array([
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
            ]),
            rigid_link_records=[
                NodePairRecord(
                    kind=ConstraintKind.RIGID_BEAM,
                    master_node=100, slave_node=200,
                ),
                NodePairRecord(
                    kind=ConstraintKind.RIGID_BEAM,
                    master_node=100, slave_node=201,
                ),
            ],
            equal_dof_records=[
                NodePairRecord(
                    kind=ConstraintKind.EQUAL_DOF,
                    master_node=200, slave_node=2,
                    dofs=[1, 2, 3],
                ),
                NodePairRecord(
                    kind=ConstraintKind.EQUAL_DOF,
                    master_node=201, slave_node=4,
                    dofs=[1, 2, 3],
                ),
            ],
            dofs=[1, 2, 3],
        )
        fem.add_node_constraints([n2s])
        rec = RecordingEmitter()
        emit_mp_constraints(rec, cast(Any, fem), TagAllocator())
        names = [c[0] for c in rec.calls]
        # Expected order: 2 phantom nodes, 2 rigid links, 2 equal_dofs.
        assert names == [
            "node", "node",
            "rigidLink", "rigidLink",
            "equalDOF", "equalDOF",
        ]
        # Phantom nodes carry ndf=6.
        assert rec.calls[0] == ("node", (200, 0.0, 0.0, 1.0), {"ndf": 6})
        assert rec.calls[1] == ("node", (201, 1.0, 0.0, 1.0), {"ndf": 6})

    def test_surface_coupling_tie_dispatches_to_embedded_node(self) -> None:
        # 3 master nodes = tri3 host (minimum supported by C++
        # ASDEmbeddedNodeElement; 2 masters would crash OpenSees at
        # runtime, caught by _check_embedded_rnode_count at emit time).
        fem = make_two_column_frame()
        interp = InterpolationRecord(
            kind=ConstraintKind.TIE,
            slave_node=5,
            master_nodes=[1, 2, 3],
            weights=np.array([1 / 3, 1 / 3, 1 / 3]),
            dofs=[1, 2, 3],
        )
        fem.add_surface_constraints([interp])
        rec = RecordingEmitter()
        emit_mp_constraints(rec, cast(Any, fem), TagAllocator())
        names = [c[0] for c in rec.calls]
        assert names == ["embeddedNode"]
        # ASDEmbeddedNodeElement signature: $tag $Cnode $Rnode1 $Rnode2 ...
        # The Emitter Protocol forwards as (ele_tag, Cnode, *Rnodes).
        # Cnode is the embedded (slave) node; Rnodes are the host element corners.
        assert rec.calls[0][1][1] == 5            # Cnode = slave_node
        assert rec.calls[0][1][2:] == (1, 2, 3)   # Rnodes = master_nodes

    def test_surface_coupling_tied_contact_expands_slave_records(
        self,
    ) -> None:
        # Each slave_record uses 3 masters (tri3 host); see comment on
        # the sibling tie test for the C++ Rnode-count constraint.
        fem = make_two_column_frame()
        coupling = SurfaceCouplingRecord(
            kind=ConstraintKind.TIED_CONTACT,
            slave_nodes=[5, 6],
            master_nodes=[1, 2, 3, 4],
            slave_records=[
                InterpolationRecord(
                    kind=ConstraintKind.TIED_CONTACT,
                    slave_node=5,
                    master_nodes=[1, 2, 3],
                    weights=np.array([1 / 3, 1 / 3, 1 / 3]),
                    dofs=[1, 2, 3],
                ),
                InterpolationRecord(
                    kind=ConstraintKind.TIED_CONTACT,
                    slave_node=6,
                    master_nodes=[2, 3, 4],
                    weights=np.array([1 / 3, 1 / 3, 1 / 3]),
                    dofs=[1, 2, 3],
                ),
            ],
            dofs=[1, 2, 3],
        )
        fem.add_surface_constraints([coupling])
        rec = RecordingEmitter()
        emit_mp_constraints(rec, cast(Any, fem), TagAllocator())
        names = [c[0] for c in rec.calls]
        assert names == ["embeddedNode", "embeddedNode"]


# ---------------------------------------------------------------------------
# Section 5 — Required gates (phantom ordering, dedup, name round-trip)
# ---------------------------------------------------------------------------


class TestRequiredGates:
    """Phase 7b spec's required-pass gates."""

    def test_phantoms_emitted_before_references(self) -> None:
        """Record what RecordingEmitter sees and assert
        ``node(phantom_tag, ...)`` precedes ``rigidLink(phantom_tag, ...)``
        in the call sequence (INV-3)."""
        fem = make_two_column_frame()
        n2s = NodeToSurfaceRecord(
            kind=ConstraintKind.NODE_TO_SURFACE,
            master_node=100,
            slave_nodes=[2],
            phantom_nodes=[200],
            phantom_coords=np.array([[0.0, 0.0, 1.0]]),
            rigid_link_records=[
                NodePairRecord(
                    kind=ConstraintKind.RIGID_BEAM,
                    master_node=100, slave_node=200,
                ),
            ],
            equal_dof_records=[
                NodePairRecord(
                    kind=ConstraintKind.EQUAL_DOF,
                    master_node=200, slave_node=2,
                    dofs=[1, 2, 3],
                ),
            ],
        )
        fem.add_node_constraints([n2s])
        rec = RecordingEmitter()
        emit_mp_constraints(rec, cast(Any, fem), TagAllocator())

        node_idx: int | None = None
        ref_idx: int | None = None
        for i, (name, args, _kwargs) in enumerate(rec.calls):
            if name == "node" and len(args) >= 1 and args[0] == 200:
                node_idx = i
                break
        for i, (name, args, _kwargs) in enumerate(rec.calls):
            if name == "rigidLink" and 200 in args:
                ref_idx = i
                break
            if name == "equalDOF" and 200 in args:
                ref_idx = i
                break
        assert node_idx is not None, "phantom node was never emitted"
        assert ref_idx is not None, "phantom node was never referenced"
        assert node_idx < ref_idx, (
            f"phantom node 200 emitted at {node_idx} after "
            f"constraint reference at {ref_idx}"
        )

    def test_phantom_tags_not_double_emitted(self) -> None:
        """Two NodeToSurfaceRecords sharing a phantom tag (defensive
        scenario; the resolver does not collide in practice) — each
        phantom tag should still emit as a node only once."""
        fem = make_two_column_frame()
        n2s_a = NodeToSurfaceRecord(
            kind=ConstraintKind.NODE_TO_SURFACE,
            master_node=100,
            slave_nodes=[2],
            phantom_nodes=[200],
            phantom_coords=np.array([[0.0, 0.0, 1.0]]),
            rigid_link_records=[
                NodePairRecord(
                    kind=ConstraintKind.RIGID_BEAM,
                    master_node=100, slave_node=200,
                ),
            ],
            equal_dof_records=[
                NodePairRecord(
                    kind=ConstraintKind.EQUAL_DOF,
                    master_node=200, slave_node=2,
                    dofs=[1, 2, 3],
                ),
            ],
        )
        n2s_b = NodeToSurfaceRecord(
            kind=ConstraintKind.NODE_TO_SURFACE,
            master_node=101,
            slave_nodes=[4],
            phantom_nodes=[200],  # SAME tag as above (defensive test)
            phantom_coords=np.array([[1.0, 0.0, 1.0]]),
            rigid_link_records=[
                NodePairRecord(
                    kind=ConstraintKind.RIGID_BEAM,
                    master_node=101, slave_node=200,
                ),
            ],
            equal_dof_records=[
                NodePairRecord(
                    kind=ConstraintKind.EQUAL_DOF,
                    master_node=200, slave_node=4,
                    dofs=[1, 2, 3],
                ),
            ],
        )
        fem.add_node_constraints([n2s_a, n2s_b])
        rec = RecordingEmitter()
        emit_mp_constraints(rec, cast(Any, fem), TagAllocator())
        node_emits = [
            c for c in rec.calls
            if c[0] == "node" and len(c[1]) >= 1 and c[1][0] == 200
        ]
        assert len(node_emits) == 1, (
            f"phantom tag 200 emitted as node {len(node_emits)} times "
            "(expected exactly 1)"
        )

    def test_constraint_name_round_trips_into_tcl(self) -> None:
        r"""``g.constraints.rigid_diaphragm(name='floor_1', ...)`` produces
        a ``# floor_1\nrigidDiaphragm 3 ...`` pair in the Tcl output."""
        fem = make_two_column_frame()
        fem.add_node_constraints([
            NodeGroupRecord(
                kind=ConstraintKind.RIGID_DIAPHRAGM,
                master_node=1, slave_nodes=[2, 3, 4],
                dofs=[1, 2, 6],
                plane_normal=np.array([0.0, 0.0, 1.0]),
                name="floor_1",
            ),
        ])
        e = TclEmitter()
        emit_mp_constraints(e, cast(Any, fem), TagAllocator())
        lines = e.lines()
        idx = lines.index("# floor_1")
        assert lines[idx + 1].startswith("rigidDiaphragm")

    def test_pass_runs_between_elements_and_patterns(self) -> None:
        """INV-5: the MP-constraint pass runs between element emission
        and pattern emission inside ``BuiltModel.emit``.  Drive a
        minimal apeSees model and inspect the call ordering."""
        fem = make_two_column_frame()
        fem.add_node_constraints([
            NodePairRecord(
                kind=ConstraintKind.EQUAL_DOF,
                master_node=2, slave_node=4,
                dofs=[1, 2, 3, 4, 5, 6],
                name="rigid_floor",
            ),
        ])
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)

        steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
        sec = ops.section.Fiber(
            fibers=(FiberPoint(material=steel, y=0.0, z=0.0, area=0.01),),
        )
        transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
        integ = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
        ops.element.forceBeamColumn(
            pg="Cols", transf=transf, integration=integ,
        )
        ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
        ts = ops.timeSeries.Linear()
        with ops.pattern.Plain(series=ts) as p:
            p.load(node=2, forces=(100e3, 0.0, 0.0))

        bm = ops.build()
        rec = RecordingEmitter()
        bm.emit(rec)

        names = [c[0] for c in rec.calls]
        last_element = max(
            (i for i, n in enumerate(names) if n == "element"),
            default=-1,
        )
        first_pattern = min(
            (i for i, n in enumerate(names) if n == "pattern_open"),
            default=len(names),
        )
        first_eqdof = min(
            (i for i, n in enumerate(names) if n == "equalDOF"),
            default=-1,
        )
        assert last_element != -1, "no element emitted"
        assert first_pattern < len(names), "no pattern emitted"
        assert first_eqdof != -1, "no equalDOF emitted"
        assert last_element < first_eqdof < first_pattern


# ---------------------------------------------------------------------------
# Section 6 — H5 schema integration
# ---------------------------------------------------------------------------


class TestH5SchemaIntegration:
    def test_phantom_node_tags_written_under_constraints(
        self, tmp_path: Path,
    ) -> None:
        """``/opensees/constraints/phantom_node_tags`` records the tags
        that the bridge declared as phantoms via the stateless
        ``set_phantom_node_tags`` predicate.

        Per S2 (ADR 0033) the per-node ``ndf=K`` kwarg is no longer the
        phantom-vs-real discriminator (real broker nodes legally carry
        ``ndf=K`` for shell-on-solid mixed-ndf models).  The bridge's
        :func:`emit_mp_constraints` helper pre-loads the complete
        phantom-tag set on the emitter ONCE, before any node emission;
        the H5 emitter consults it per call.
        """
        from apeGmsh.opensees._internal.tag_resolution import (
            set_phantom_node_tags,
        )

        e = H5Emitter()
        e.model(ndm=3, ndf=6)
        # Pre-load the phantom-tag predicate once — order-independent.
        set_phantom_node_tags(e, {200, 201})
        e.node(1, 0.0, 0.0, 0.0)                       # regular broker node
        e.node(200, 0.0, 0.0, 1.0, ndf=6)              # phantom — by predicate
        e.node(201, 1.0, 0.0, 1.0, ndf=6)              # phantom — by predicate
        out = tmp_path / "x.h5"
        e.write(str(out))
        with h5py.File(out, "r") as f:
            tags = f["opensees/constraints/phantom_node_tags"][:]
        assert sorted(int(t) for t in tags) == [200, 201]

    def test_constraints_group_absent_when_no_constraints(
        self, tmp_path: Path,
    ) -> None:
        """No MP constraints emitted → ``/opensees/constraints/`` is
        not created (additive-only — old readers see no new groups)."""
        e = H5Emitter()
        e.model(ndm=3, ndf=6)
        out = tmp_path / "x.h5"
        e.write(str(out))
        with h5py.File(out, "r") as f:
            assert "opensees/constraints" not in f

    def test_schema_version_matches_opensees_current(self, tmp_path: Path) -> None:
        # Phase 7b bumped to 2.7.0 originally; the embeddedNode rename
        # bumped to 2.8.0; ADR 0024 (region() Protocol widening) bumped
        # to 2.9.0; ADR 0027 INV-5 amendment + partition emission
        # carried it to 2.10.0; the 0-based rank-gate fix bumped to
        # 2.11.0 (partition_NN groups changed naming convention); ADR
        # 0035 (ASDEmbeddedNodeElement option exposure) bumped to 2.12.0.
        from apeGmsh.opensees.emitter.h5 import SCHEMA_VERSION
        assert SCHEMA_VERSION == OPENSEES_CURRENT
        e = H5Emitter()
        out = tmp_path / "x.h5"
        e.write(str(out))
        with h5py.File(out, "r") as f:
            assert f["meta"].attrs["schema_version"] == OPENSEES_CURRENT
            assert f["meta"].attrs["opensees_schema_version"] == OPENSEES_CURRENT

    def test_reader_window_accepts_2_14_and_2_15(self, tmp_path: Path) -> None:
        """The 2-version window for OpenSees zone is now 2.14.x — 2.15.x
        (ADR 0053 D3b: the /opensees/dampings damping-object store)."""
        from apeGmsh.opensees._internal.schema_version import (
            OPENSEES,
            SchemaVersion,
            SchemaVersionError,
            reader_version,
            validate_zone_version,
        )
        reader = reader_version(OPENSEES)
        assert reader == SchemaVersion(2, 15, 0)
        validate_zone_version(SchemaVersion(2, 14, 0), reader, zone=OPENSEES)
        validate_zone_version(SchemaVersion(2, 15, 0), reader, zone=OPENSEES)
        # 2.13.x is now outside the window.
        with pytest.raises(SchemaVersionError):
            validate_zone_version(
                SchemaVersion(2, 13, 0), reader, zone=OPENSEES,
            )


# ---------------------------------------------------------------------------
# Section 7 — Phase 8 Transformation auto-emit (the fold-in)
# ---------------------------------------------------------------------------


def _build_minimal_apesees_model(fem) -> "Any":
    """Build a minimal :class:`apeSees` deck (no constraint handler).

    Used by the auto-emit tests below; mirrors
    :meth:`test_pass_runs_between_elements_and_patterns` minus the
    explicit constraint-handler declaration.
    """
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)

    steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
    sec = ops.section.Fiber(
        fibers=(FiberPoint(material=steel, y=0.0, z=0.0, area=0.01),),
    )
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
    ops.element.forceBeamColumn(
        pg="Cols", transf=transf, integration=integ,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(node=2, forces=(100e3, 0.0, 0.0))
    return ops


class TestTransformationAutoEmit:
    """Phase 8 fold-in — auto-emit ``constraints("Transformation")``
    when MP constraints are present and the user did not declare a
    handler.

    Addresses the Phase 7b footgun: default OpenSees handler
    ``Plain`` silently ignores MP constraints.
    """

    def test_auto_emit_transformation_when_mp_constraints_present(
        self,
    ) -> None:
        """MP constraints present + no explicit constraints() declaration
        → bridge emits 'Transformation' constraint handler + UserWarning."""
        import warnings

        fem = make_two_column_frame()
        fem.add_node_constraints([
            NodePairRecord(
                kind=ConstraintKind.EQUAL_DOF,
                master_node=2, slave_node=4,
                dofs=[1, 2, 3, 4, 5, 6],
            ),
        ])
        ops = _build_minimal_apesees_model(fem)
        bm = ops.build()
        rec = RecordingEmitter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bm.emit(rec)
        # Auto-emit fires.
        constraints_calls = [
            c for c in rec.calls if c[0] == "constraints"
        ]
        assert len(constraints_calls) == 1, (
            f"expected exactly one constraints() call (the auto-emit); "
            f"got {constraints_calls}"
        )
        assert constraints_calls[0] == ("constraints", ("Transformation",), {})
        # UserWarning fires (Phase 8 fold-in contract).
        user_warnings = [
            w for w in caught if issubclass(w.category, UserWarning)
        ]
        assert any(
            "auto-emit" in str(w.message).lower()
            and "transformation" in str(w.message).lower()
            for w in user_warnings
        ), f"expected an auto-emit UserWarning; got {[str(w.message) for w in user_warnings]}"

    def test_no_auto_emit_when_no_mp_constraints(self) -> None:
        """Model with no MP constraints → no warning, no auto-emit."""
        import warnings

        fem = make_two_column_frame()  # no constraints added
        ops = _build_minimal_apesees_model(fem)
        bm = ops.build()
        rec = RecordingEmitter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bm.emit(rec)
        constraints_calls = [
            c for c in rec.calls if c[0] == "constraints"
        ]
        assert constraints_calls == [], (
            f"expected no constraints() calls; got {constraints_calls}"
        )
        user_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "constraint" in str(w.message).lower()
        ]
        assert user_warnings == [], (
            f"expected no constraint-related UserWarning; got "
            f"{[str(w.message) for w in user_warnings]}"
        )

    def test_explicit_plain_with_mp_warns_but_respects(self) -> None:
        """User explicitly declares Plain + MP constraints present:
        UserWarning fires, but Plain is still emitted (the user's
        choice is respected)."""
        import warnings

        fem = make_two_column_frame()
        fem.add_node_constraints([
            NodePairRecord(
                kind=ConstraintKind.EQUAL_DOF,
                master_node=2, slave_node=4,
                dofs=[1, 2, 3, 4, 5, 6],
            ),
        ])
        ops = _build_minimal_apesees_model(fem)
        # Explicit Plain — Phase 8 fold-in respects but warns.
        ops.constraints.Plain()
        bm = ops.build()
        rec = RecordingEmitter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bm.emit(rec)
        # Plain still emitted (user's choice respected), no Transformation.
        constraints_calls = [
            c for c in rec.calls if c[0] == "constraints"
        ]
        assert len(constraints_calls) == 1
        assert constraints_calls[0] == ("constraints", ("Plain",), {})
        # UserWarning fires with the "silently ignored" wording.
        user_warnings = [
            w for w in caught if issubclass(w.category, UserWarning)
        ]
        assert any(
            "plain handler" in str(w.message).lower()
            and "silently ignored" in str(w.message).lower()
            for w in user_warnings
        ), f"expected the 'Plain silently ignored' UserWarning; got {[str(w.message) for w in user_warnings]}"
