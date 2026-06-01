"""Unit tests for ADR 0052 slice 1 — ``s.support`` HOLD constraints.

Covers:

1. Emitter Protocol contract — ``sp_hold(node, dof)``.
2. TclEmitter / PyEmitter / RecordingEmitter emit shapes.
3. End-to-end flat emit (tcl + py): a stage ``s.support`` emits a
   dedicated ``Plain`` pattern bound to a shared ``Constant`` series,
   with ``sp ... [nodeDisp ...] -const`` per held DOF, inside the stage
   block, exactly once (not double-emitted globally).
4. ``s.fix`` stays absolute (unchanged) — both verbs can coexist.
5. V2 validator — a ``fix`` and a ``support`` on the same (node, DOF)
   collide.
"""
from __future__ import annotations

import inspect
from typing import get_type_hints

import pytest

from apeGmsh.opensees.apesees import apeSees, BridgeError
from apeGmsh.opensees.emitter.base import Emitter
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


# ---------------------------------------------------------------------------
# 1. Protocol contract
# ---------------------------------------------------------------------------


def test_emitter_protocol_has_sp_hold() -> None:
    assert hasattr(Emitter, "sp_hold")
    hints = get_type_hints(Emitter.sp_hold)
    assert hints.get("node") is int
    assert hints.get("dof") is int
    assert hints.get("return") is type(None)
    sig = inspect.signature(Emitter.sp_hold)
    assert [p.name for p in sig.parameters.values()] == ["self", "node", "dof"]


# ---------------------------------------------------------------------------
# 2. Emitter shapes
# ---------------------------------------------------------------------------


def test_tcl_sp_hold_shape() -> None:
    e = TclEmitter()
    e.sp_hold(7, 2)
    assert e.lines()[-1] == "sp 7 2 [nodeDisp 7 2] -const"


def test_tcl_sp_hold_indents_inside_block_pattern() -> None:
    e = TclEmitter()
    e.timeSeries("Constant", 1)
    e.pattern_open("Plain", 5, 1)
    e.sp_hold(7, 2)
    e.pattern_close()
    body = [ln for ln in e.lines() if "nodeDisp" in ln]
    assert body == ["    sp 7 2 [nodeDisp 7 2] -const"]


def test_py_sp_hold_shape() -> None:
    e = PyEmitter()
    e.sp_hold(7, 2)
    assert e.lines()[-1] == "ops.sp(7, 2, ops.nodeDisp(7, 2), '-const')"


def test_recording_sp_hold_records_call() -> None:
    e = RecordingEmitter()
    e.sp_hold(7, 2)
    assert e.calls[-1] == ("sp_hold", (7, 2), {})


# ---------------------------------------------------------------------------
# 3 + 4. End-to-end flat emit
# ---------------------------------------------------------------------------


def _make_single_quad_fem() -> FEMStub:
    return FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4],
            coords=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            ],
            node_pgs={"Left": [1, 4], "Bottom": [1, 2]},
        ),
        elements=_ElementsStub(
            elem_pgs={
                "Rock": _ElementGroupView(
                    ids=(1,), connectivity=((1, 2, 3, 4),),
                ),
            },
        ),
    )


def _full_chain(ops: apeSees) -> dict[str, object]:
    return {
        "test":        ops.test.NormDispIncr(tol=1e-4, max_iter=50),
        "algorithm":   ops.algorithm.Newton(),
        "integrator":  ops.integrator.LoadControl(dlam=0.1),
        "constraints": ops.constraints.Plain(),
        "numberer":    ops.numberer.RCM(),
        "system":      ops.system.UmfPack(),
        "analysis":    ops.analysis.Static(),
    }


def _two_stage_with_support() -> apeSees:
    """Stage 1 = insitu (no support); stage 2 = relax, holds node 3."""
    fem = _make_single_quad_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    ops.fix(pg="Left", dofs=(1, 0))
    ops.fix(pg="Bottom", dofs=(0, 1))

    with ops.stage(name="insitu") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=10, dt=0.1)

    with ops.stage(name="relax") as s:
        # Hold the (unconstrained) top-right node at its drifted position.
        s.support(nodes=[3], dofs=(1, 1))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=5, dt=0.1)

    return ops


def test_tcl_support_emits_constant_pattern_and_sp_hold(tmp_path) -> None:
    ops = _two_stage_with_support()
    deck = tmp_path / "deck.tcl"
    ops.tcl(str(deck))
    lines = deck.read_text().splitlines()

    # A Constant series exists (shared HOLD series).
    assert any(ln.startswith("timeSeries Constant ") for ln in lines)
    # The two held DOFs each emit a HOLD sp inside a Plain pattern.
    hold = [ln for ln in lines if "nodeDisp 3" in ln]
    assert hold == [
        "    sp 3 1 [nodeDisp 3 1] -const",
        "    sp 3 2 [nodeDisp 3 2] -const",
    ]
    # The HOLD lines sit in the 'relax' stage block (after its banner).
    relax_idx = next(
        i for i, ln in enumerate(lines) if ln == "# === Stage: relax ==="
    )
    first_hold_idx = next(i for i, ln in enumerate(lines) if "nodeDisp 3" in ln)
    assert first_hold_idx > relax_idx


def test_tcl_support_hold_pattern_not_double_emitted(tmp_path) -> None:
    """The dedicated HOLD pattern is claimed → emitted exactly once
    (inside the stage), never in the global post-element pattern pass."""
    ops = _two_stage_with_support()
    deck = tmp_path / "deck.tcl"
    ops.tcl(str(deck))
    text = deck.read_text()
    # Exactly two HOLD sp lines, total, across the whole deck.
    assert text.count("-const") == 2


def test_py_support_emits_runtime_nodedisp_capture(tmp_path) -> None:
    ops = _two_stage_with_support()
    deck = tmp_path / "deck.py"
    ops.py(str(deck))
    text = deck.read_text()
    assert "ops.sp(3, 1, ops.nodeDisp(3, 1), '-const')" in text
    assert "ops.sp(3, 2, ops.nodeDisp(3, 2), '-const')" in text


def test_support_and_fix_coexist_in_same_stage(tmp_path) -> None:
    """s.fix (ANCHOR) and s.support (HOLD) on DIFFERENT dofs of the same
    node are both legal and both emit."""
    fem = _make_single_quad_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)

    with ops.stage(name="mix") as s:
        s.fix(nodes=[3], dofs=(1, 0))       # ANCHOR dof 1
        s.support(nodes=[3], dofs=(0, 1))   # HOLD dof 2
        s.analysis(**_full_chain(ops))
        s.run(n_increments=5)

    deck = tmp_path / "deck.tcl"
    ops.tcl(str(deck))
    text = deck.read_text()
    assert "fix 3 1 0" in text
    assert "sp 3 2 [nodeDisp 3 2] -const" in text


# ---------------------------------------------------------------------------
# 5. V2 validator — fix/support collision on the same (node, DOF)
# ---------------------------------------------------------------------------


def test_fix_support_same_node_dof_collides(tmp_path) -> None:
    fem = _make_single_quad_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)

    with ops.stage(name="bad") as s:
        s.fix(nodes=[3], dofs=(1, 0))
        s.support(nodes=[3], dofs=(1, 0))   # same (node 3, DOF 1)
        s.analysis(**_full_chain(ops))
        s.run(n_increments=5)

    with pytest.raises(BridgeError, match="node 3 DOF 1"):
        ops.tcl(str(tmp_path / "deck.tcl"))
