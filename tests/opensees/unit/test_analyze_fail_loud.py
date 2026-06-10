"""Fail-loud analyze loops — the deck aborts on the first failed increment.

A batched ``analyze N`` short-circuits internally on the first failed
increment and returns -3; nothing in the old decks checked the return
code, so the deck ran on with the stage silently partial (or not applied
at all) — observed on the Cerro Lindo excavated tunnel twin, where a
40-increment gravity stage did nothing and the run "completed" with
exit 0.

Every emitted analyze is now a per-increment loop whose rc check aborts
the deck (``raise SystemExit`` / Tcl ``error``) with a banner naming the
loop (stage name via ``label=``), the increment, and the pseudo-time.

Covers:

* generated-code behavior: the py loop really aborts at the failing
  increment and never runs the rest (exec'd against an ops stub),
* stage-level deck shape for BOTH the hook-bearing (initial_stress)
  and hook-free stage forms, py + tcl, with the stage name in the
  banner,
* the live path: the orchestrator raises BridgeError when the live
  emitter reports a failed stage analyze.
"""
from __future__ import annotations

import pytest

from apeGmsh.opensees.apesees import apeSees, BridgeError
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


# =====================================================================
# Generated-code behavior (py): exec the emitted loop against a stub
# =====================================================================

class _OpsStub:
    def __init__(self, fail_at: int) -> None:
        self.calls = 0
        self._fail_at = fail_at

    def analyze(self, *_args: object) -> int:
        self.calls += 1
        return -3 if self.calls == self._fail_at else 0

    def getTime(self) -> float:
        return 0.5


def _emitted_loop(e: PyEmitter) -> str:
    skip = ("#", "import ", "ops.wipe(")
    return "\n".join(
        ln for ln in e.lines() if not any(ln.startswith(s) for s in skip)
    )


def test_py_loop_aborts_at_failing_increment() -> None:
    e = PyEmitter()
    e.analyze(steps=4, dt=0.5, label="Gravity")
    stub = _OpsStub(fail_at=3)
    with pytest.raises(SystemExit) as ei:
        exec(compile(_emitted_loop(e), "<deck>", "exec"), {"ops": stub})
    msg = str(ei.value)
    assert "increment 3/4" in msg
    assert "of stage 'Gravity'" in msg
    assert "0.5" in msg                    # pseudo-time from ops.getTime()
    assert stub.calls == 3                 # aborted; increment 4 never ran


def test_py_loop_runs_all_increments_when_converging() -> None:
    e = PyEmitter()
    e.analyze(steps=4, dt=0.5)
    stub = _OpsStub(fail_at=99)
    exec(compile(_emitted_loop(e), "<deck>", "exec"), {"ops": stub})
    assert stub.calls == 4


# =====================================================================
# Stage-level deck shape (hook-bearing + hook-free stages)
# =====================================================================

def _make_single_quad_fem() -> FEMStub:
    return FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4],
            coords=[
                (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
            ],
            node_pgs={"Left": [1, 4]},
        ),
        elements=_ElementsStub(
            elem_pgs={
                "Rock": _ElementGroupView(
                    ids=(1,), connectivity=((1, 2, 3, 4),),
                ),
            },
        ),
    )


def _two_stage_ops() -> apeSees:
    """Stage 'insitu' carries initial_stress (hooks); stage 'relax'
    has no hooks."""
    fem = _make_single_quad_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    ops.fix(pg="Left", dofs=(1, 1))

    def chain():
        return {
            "test":        ops.test.NormDispIncr(tol=1e-4, max_iter=50),
            "algorithm":   ops.algorithm.Newton(),
            "integrator":  ops.integrator.LoadControl(dlam=0.1),
            "constraints": ops.constraints.Plain(),
            "numberer":    ops.numberer.RCM(),
            "system":      ops.system.UmfPack(),
            "analysis":    ops.analysis.Static(),
        }

    with ops.stage(name="insitu") as s:
        s.initial_stress(
            name="ramp", pg="Rock",
            sigma_xx=-100.0, sigma_yy=-100.0, sigma_zz=-100.0,
            ramp_steps=5,
        )
        s.analysis(**chain())
        s.run(n_increments=5, dt=0.2)

    with ops.stage(name="relax") as s:
        s.analysis(**chain())
        s.run(n_increments=3, dt=0.5)

    return ops


def test_py_deck_both_stage_forms_carry_rc_check(tmp_path) -> None:
    ops = _two_stage_ops()
    deck = tmp_path / "deck.py"
    ops.py(str(deck))
    text = deck.read_text()

    # hook-bearing stage: dispatcher calls AND the rc check, named.
    assert "_apesees_call_before_step()" in text
    assert "of stage 'insitu'" in text
    # hook-free stage: rc check, named — and NEVER a batched analyze.
    assert "of stage 'relax'" in text
    assert "ops.analyze(5" not in text
    assert "ops.analyze(3" not in text
    assert text.count("raise SystemExit(") == 2


def test_tcl_deck_both_stage_forms_carry_rc_check(tmp_path) -> None:
    ops = _two_stage_ops()
    deck = tmp_path / "deck.tcl"
    ops.tcl(str(deck))
    text = deck.read_text()

    assert "_apesees_call_before_step" in text
    assert "of stage 'insitu'" in text
    assert "of stage 'relax'" in text
    assert "analyze 5" not in text
    assert "analyze 3" not in text
    assert text.count('error "apeGmsh: analyze FAILED') == 2


# =====================================================================
# Live path: orchestrator raises on a failed stage analyze
# =====================================================================

def test_live_stage_failure_raises_bridge_error(monkeypatch) -> None:
    from apeGmsh.opensees.emitter.recording import RecordingEmitter

    ops = _two_stage_ops()
    bm = ops.build()

    emitter = RecordingEmitter()
    rc_by_label = {"insitu": 0, "relax": -3}

    def failing_analyze(*, steps, dt=None, label=None, strategy=None):
        emitter.calls.append(("analyze", (), {"steps": steps, "dt": dt,
                                              "label": label}))
        return rc_by_label.get(label, 0)

    monkeypatch.setattr(emitter, "analyze", failing_analyze)
    with pytest.raises(BridgeError, match="stage 'relax'.*FAILED"):
        bm.emit(emitter)
