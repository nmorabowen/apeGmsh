"""Unit tests for the ``RecordingEmitter`` test fixture.

These verify that every Protocol method is callable and that calls
are captured in the documented ``(name, args, kwargs)`` shape.
Concrete emitters (Tcl, py, live, h5) get their own test files in
later phases.
"""
from __future__ import annotations

from typing import Any

import pytest

from apeGmsh.opensees.emitter.recording import RecordingEmitter


def test_recording_emitter_constructs_with_no_args() -> None:
    emitter = RecordingEmitter()
    assert emitter.calls == []


# ---------------------------------------------------------------------------
# Per-method recording — one parametrize per group, each verifying the
# (name, args, kwargs) shape.
# ---------------------------------------------------------------------------

def test_model_records_kwargs() -> None:
    e = RecordingEmitter()
    e.model(ndm=3, ndf=6)
    assert e.calls == [("model", (), {"ndm": 3, "ndf": 6})]


def test_node_records_tag_and_coords() -> None:
    e = RecordingEmitter()
    e.node(7, 1.0, 2.0, 3.0)
    assert e.calls == [("node", (7, 1.0, 2.0, 3.0), {})]


def test_fix_records_tag_and_dofs() -> None:
    e = RecordingEmitter()
    e.fix(1, 1, 1, 0, 0, 0, 0)
    assert e.calls == [("fix", (1, 1, 1, 0, 0, 0, 0), {})]


def test_uniaxialMaterial_records_type_tag_params() -> None:
    e = RecordingEmitter()
    e.uniaxialMaterial("Steel02", 42, 420e6, 200e9, 0.01)
    assert e.calls == [
        ("uniaxialMaterial", ("Steel02", 42, 420e6, 200e9, 0.01), {})
    ]


def test_element_records_type_tag_args() -> None:
    e = RecordingEmitter()
    e.element("forceBeamColumn", 5, 1, 2, 3, 4)
    assert e.calls == [
        ("element", ("forceBeamColumn", 5, 1, 2, 3, 4), {})
    ]


def test_pattern_open_close_pair() -> None:
    e = RecordingEmitter()
    e.pattern_open("Plain", 1, 1)
    e.load(7, 100.0, 0.0, 0.0)
    e.pattern_close()
    assert [c[0] for c in e.calls] == [
        "pattern_open", "load", "pattern_close",
    ]
    assert e.calls[0] == ("pattern_open", ("Plain", 1, 1), {})
    assert e.calls[1] == ("load", (7, 100.0, 0.0, 0.0), {})
    assert e.calls[2] == ("pattern_close", (), {})


def test_section_open_close_pair_with_patch_and_fiber() -> None:
    e = RecordingEmitter()
    e.section_open("Fiber", 3)
    e.patch("rect", 1, 10, 10, -1.0, -1.0, 1.0, 1.0)
    e.fiber(0.0, 0.0, 0.001, 1)
    e.section_close()
    names = [c[0] for c in e.calls]
    assert names == ["section_open", "patch", "fiber", "section_close"]


def test_analyze_returns_int_and_records() -> None:
    e = RecordingEmitter()
    rc = e.analyze(steps=20, dt=0.01)
    assert isinstance(rc, int)
    assert rc == 0
    assert e.calls == [
        ("analyze", (), {"steps": 20, "dt": 0.01})
    ]


def test_analyze_dt_default_none() -> None:
    e = RecordingEmitter()
    e.analyze(steps=5)
    assert e.calls == [("analyze", (), {"steps": 5, "dt": None})]


# ---------------------------------------------------------------------------
# Coverage spot-check: every Protocol method should at least be defined
# on the emitter. We don't enumerate the full Protocol surface here —
# that's the contract layer's job — but we sanity-check a representative
# set so the fixture isn't silently missing a method.
# ---------------------------------------------------------------------------

REPRESENTATIVE_METHODS = (
    "model", "node", "fix", "mass",
    "uniaxialMaterial", "nDMaterial", "section", "geomTransf",
    "section_open", "section_close", "patch", "fiber", "layer",
    "element", "timeSeries",
    "pattern_open", "pattern_close", "load", "eleLoad", "sp",
    "recorder",
    "constraints", "numberer", "system", "test",
    "algorithm", "integrator", "analysis", "analyze",
    # Eigen — one-shot, returns values from live emitter.
    "eigen",
    # MP constraint methods (ADR 0022, Phase 7b)
    "equalDOF", "rigidLink", "rigidDiaphragm",
    "embeddedNode", "mp_constraint_comment",
)


@pytest.mark.parametrize("method_name", REPRESENTATIVE_METHODS)
def test_recording_emitter_has_method(method_name: str) -> None:
    e = RecordingEmitter()
    method: Any = getattr(e, method_name, None)
    assert callable(method), f"RecordingEmitter missing method: {method_name}"


# ---------------------------------------------------------------------------
# MP constraint Protocol shape — Phase 7b, ADR 0022
# ---------------------------------------------------------------------------


def test_equalDOF_records_master_slave_dofs() -> None:
    e = RecordingEmitter()
    e.equalDOF(1, 2, 1, 2, 3)
    assert e.calls == [("equalDOF", (1, 2, 1, 2, 3), {})]


def test_rigidLink_records_kind_master_slave() -> None:
    e = RecordingEmitter()
    e.rigidLink("beam", 1, 2)
    assert e.calls == [("rigidLink", ("beam", 1, 2), {})]


def test_rigidDiaphragm_records_perp_master_slaves() -> None:
    e = RecordingEmitter()
    e.rigidDiaphragm(3, 100, 1, 2, 3)
    assert e.calls == [("rigidDiaphragm", (3, 100, 1, 2, 3), {})]


def test_embeddedNode_records_ele_tag_cnode_args() -> None:
    e = RecordingEmitter()
    e.embeddedNode(1000, 5, 10, 20, 30)
    assert e.calls == [("embeddedNode", (1000, 5, 10, 20, 30), {})]


def test_mp_constraint_comment_records_name() -> None:
    e = RecordingEmitter()
    e.mp_constraint_comment("floor_1")
    assert e.calls == [("mp_constraint_comment", ("floor_1",), {})]


def test_eigen_records_num_modes_and_solver_default() -> None:
    e = RecordingEmitter()
    rc = e.eigen(5)
    assert rc == []
    assert e.calls == [
        ("eigen", (), {"num_modes": 5, "solver": "-genBandArpack"}),
    ]


def test_eigen_records_custom_solver() -> None:
    e = RecordingEmitter()
    e.eigen(3, solver="-fullGenLapack")
    assert e.calls == [
        ("eigen", (), {"num_modes": 3, "solver": "-fullGenLapack"}),
    ]


def test_node_accepts_ndf_kwarg_for_phantom() -> None:
    """Per-node ``ndf=6`` override is the Phase 7b phantom-node idiom."""
    e = RecordingEmitter()
    e.node(99, 1.0, 2.0, 3.0, ndf=6)
    assert e.calls == [("node", (99, 1.0, 2.0, 3.0), {"ndf": 6})]
