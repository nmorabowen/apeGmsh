"""Unit tests for Phase SSI-2.A staged analysis.

Covers:

1. Emitter Protocol contract — ``stage_open(name)`` + ``stage_close()``.
2. RecordingEmitter captures both events.
3. TclEmitter / PyEmitter emit shape — comment header on open;
   loadConst + wipeAnalysis + (conditional) hook clear on close.
4. ``apeSees.stage(name)`` context manager — builds StageRecord,
   moves initial_stress records out of the bridge's global pool,
   validates required fields on __exit__.
5. Bridge emit orchestration — staged builds skip the global pre-
   element emit of analysis-chain primitives and emit them per
   stage instead; per-stage analyze loop emits between stage_open
   and stage_close.
6. Live emitter rejects staged builds with NotImplementedError.
"""
from __future__ import annotations

import inspect
from typing import get_type_hints

import pytest

from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter.base import Emitter
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
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


def test_emitter_protocol_has_stage_open_close() -> None:
    assert hasattr(Emitter, "stage_open")
    assert hasattr(Emitter, "stage_close")
    open_hints = get_type_hints(Emitter.stage_open)
    assert open_hints.get("name") is str
    assert open_hints.get("return") is type(None)
    sig_open = inspect.signature(Emitter.stage_open)
    assert [p.name for p in sig_open.parameters.values()] == ["self", "name"]
    sig_close = inspect.signature(Emitter.stage_close)
    assert [p.name for p in sig_close.parameters.values()] == ["self"]


# ---------------------------------------------------------------------------
# 2. RecordingEmitter
# ---------------------------------------------------------------------------


def test_recording_emitter_captures_stage_events() -> None:
    e = RecordingEmitter()
    e.stage_open("insitu")
    e.stage_close()
    e.stage_open("relax")
    e.stage_close()
    assert e.calls == [
        ("stage_open", ("insitu",), {}),
        ("stage_close", (), {}),
        ("stage_open", ("relax",), {}),
        ("stage_close", (), {}),
    ]


# ---------------------------------------------------------------------------
# 3. TclEmitter + PyEmitter shape
# ---------------------------------------------------------------------------


def test_tcl_stage_open_emits_comment_header() -> None:
    e = TclEmitter()
    e.stage_open("insitu")
    assert e.lines()[-1] == "# === Stage: insitu ==="


def test_tcl_stage_close_emits_loadconst_wipeanalysis() -> None:
    e = TclEmitter()
    e.stage_close()
    out = e.lines()[-2:]
    assert out == ["loadConst -time 0.0", "wipeAnalysis"]


def test_tcl_stage_close_clears_hook_lists_when_hooks_registered() -> None:
    e = TclEmitter()
    e.step_hook_ramp(
        "ramp",
        targets=((1, -100.0),),
        n_steps_to_full=10.0,
        phase="before",
    )
    n_before = len(e.lines())
    e.stage_close()
    text = "\n".join(e.lines()[n_before:])
    assert "loadConst -time 0.0" in text
    assert "wipeAnalysis" in text
    assert "set _apesees_before_step_hooks {}" in text
    assert "set _apesees_after_step_hooks {}" in text
    # The flag is now reset — analyze() emits unwrapped until the
    # next stage's step_hook_ramp.
    n_after_close = len(e.lines())
    e.analyze(steps=5)
    assert e.lines()[n_after_close] == "analyze 5"


def test_py_stage_open_close_shape() -> None:
    e = PyEmitter()
    e.stage_open("insitu")
    assert e.lines()[-1] == "# === Stage: insitu ==="
    e.stage_close()
    text = "\n".join(e.lines())
    assert "ops.loadConst('-time', 0.0)" in text
    assert "ops.wipeAnalysis()" in text


def test_py_stage_close_clears_hook_lists() -> None:
    e = PyEmitter()
    e.step_hook_ramp(
        "ramp",
        targets=((1, -100.0),),
        n_steps_to_full=10.0,
        phase="before",
    )
    n_before = len(e.lines())
    e.stage_close()
    text = "\n".join(e.lines()[n_before:])
    assert "_apesees_before_step_hooks.clear()" in text
    assert "_apesees_after_step_hooks.clear()" in text


def test_tcl_stage_close_skips_hook_clear_when_no_hooks() -> None:
    e = TclEmitter()
    n_before = len(e.lines())
    e.stage_close()
    text = "\n".join(e.lines()[n_before:])
    # Only loadConst + wipeAnalysis; no hook clear lines.
    assert "loadConst -time 0.0" in text
    assert "wipeAnalysis" in text
    assert "_apesees_before_step_hooks" not in text


# ---------------------------------------------------------------------------
# 4. _StageBuilder context manager + bridge integration
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
    """Build a complete analysis chain (for stage.analysis(**chain))."""
    return {
        "test":        ops.test.NormDispIncr(tol=1e-4, max_iter=50),
        "algorithm":   ops.algorithm.Newton(),
        "integrator":  ops.integrator.LoadControl(dlam=0.1),
        "constraints": ops.constraints.Plain(),
        "numberer":    ops.numberer.RCM(),
        "system":      ops.system.UmfPack(),
        "analysis":    ops.analysis.Static(),
    }


def test_stage_builder_records_complete_stage() -> None:
    fem = _make_single_quad_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)

    with ops.stage(name="insitu") as s:
        s.add(ops.initial_stress(
            name="rock_in", pg="Rock",
            sigma_xx=-100.0, sigma_yy=-100.0, sigma_zz=-100.0,
            ramp_steps=10,
        ))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=10, dt=0.1)

    # One stage record, with the initial_stress moved out of the
    # bridge's global pool.
    assert len(ops._stage_records) == 1
    assert ops._initial_stress_records == []
    assert ops._stage_records[0].name == "insitu"
    assert len(ops._stage_records[0].initial_stress_records) == 1
    assert ops._stage_records[0].n_increments == 10
    assert ops._stage_records[0].dt == 0.1


def test_stage_builder_missing_analysis_raises() -> None:
    fem = _make_single_quad_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    with pytest.raises(ValueError, match="missing s.analysis"):
        with ops.stage(name="bad") as s:
            s.run(n_increments=10, dt=0.1)


def test_stage_builder_missing_run_raises() -> None:
    fem = _make_single_quad_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    with pytest.raises(ValueError, match="missing s.run"):
        with ops.stage(name="bad") as s:
            s.analysis(**_full_chain(ops))


def test_stage_builder_drops_on_exception() -> None:
    """A user exception inside ``with`` discards the in-progress stage."""
    fem = _make_single_quad_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    with pytest.raises(RuntimeError, match="user error"):
        with ops.stage(name="will-fail") as s:
            s.analysis(**_full_chain(ops))
            raise RuntimeError("user error")
    # No StageRecord appended.
    assert ops._stage_records == []


def test_stage_builder_double_analysis_raises() -> None:
    fem = _make_single_quad_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    with pytest.raises(ValueError, match="already called"):
        with ops.stage(name="bad") as s:
            s.analysis(**_full_chain(ops))
            s.analysis(**_full_chain(ops))
            s.run(n_increments=10)


def test_stage_builder_add_unknown_record_type_raises() -> None:
    fem = _make_single_quad_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    with pytest.raises(TypeError, match="unsupported record type"):
        with ops.stage(name="bad") as s:
            s.add("not-a-record")  # type: ignore[arg-type]
            s.analysis(**_full_chain(ops))
            s.run(n_increments=10)


def test_stage_builder_double_add_raises() -> None:
    """Adding the same InitialStressRecord twice (across stages) is
    caught — the second add can't find it in the bridge's pool."""
    fem = _make_single_quad_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    rec = ops.initial_stress(
        name="rock", pg="Rock",
        sigma_xx=-100.0, sigma_yy=-100.0, sigma_zz=-100.0,
        ramp_steps=10,
    )
    with ops.stage(name="A") as s:
        s.add(rec)
        s.analysis(**_full_chain(ops))
        s.run(n_increments=10)
    # rec moved into stage A's pool — bridge global is empty.
    with pytest.raises(ValueError, match="not in the bridge's global pool"):
        with ops.stage(name="B") as s:
            s.add(rec)
            s.analysis(**_full_chain(ops))
            s.run(n_increments=10)


def test_stage_builder_invalid_n_increments_raises() -> None:
    fem = _make_single_quad_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    with pytest.raises(ValueError, match="n_increments must be >= 1"):
        with ops.stage(name="bad") as s:
            s.analysis(**_full_chain(ops))
            s.run(n_increments=0)


def test_empty_stage_name_raises() -> None:
    fem = _make_single_quad_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    with pytest.raises(ValueError, match="name= must be non-empty"):
        ops.stage(name="")


# ---------------------------------------------------------------------------
# 5. Bridge emit orchestration
# ---------------------------------------------------------------------------


def _build_two_stage_ops(tmp_path) -> apeSees:
    """Build a 2-stage 1-quad model — used by the next several tests."""
    fem = _make_single_quad_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    ops.fix(pg="Left", dofs=(1, 0))
    ops.fix(pg="Bottom", dofs=(0, 1))

    with ops.stage(name="insitu") as s:
        s.add(ops.initial_stress(
            name="rock_in", pg="Rock",
            sigma_xx=-100.0, sigma_yy=-100.0, sigma_zz=-100.0,
            ramp_steps=10,
        ))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=10, dt=0.1)

    with ops.stage(name="relax") as s:
        s.add(ops.initial_stress(
            name="rock_rel", pg="Rock",
            sigma_xx=-100.0, sigma_yy=-100.0, sigma_zz=-100.0,
            ramp_steps=5, lambda_install=0.5,
        ))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=5, dt=0.1)

    return ops


def test_staged_emit_skips_global_chain_emit(tmp_path) -> None:
    """Analysis-chain primitives are NOT emitted globally before the
    first stage_open — they're emitted per-stage instead."""
    ops = _build_two_stage_ops(tmp_path)
    deck_path = tmp_path / "deck.tcl"
    ops.tcl(str(deck_path))
    text = deck_path.read_text()
    lines = text.splitlines()

    # Find the first stage banner line.
    first_stage_idx = next(
        i for i, ln in enumerate(lines) if ln.startswith("# === Stage:")
    )
    # Before the first stage, NO test/algorithm/integrator/etc. should
    # appear (other than the model topology + materials + fixes).
    head = lines[:first_stage_idx]
    for chain_kw in (
        "test ", "algorithm ", "integrator ", "constraints ",
        "numberer ", "system ", "analysis Static",
    ):
        assert not any(ln.startswith(chain_kw) for ln in head), (
            f"chain keyword {chain_kw!r} leaked into pre-stage emit"
        )


def test_staged_emit_contains_two_stage_blocks(tmp_path) -> None:
    ops = _build_two_stage_ops(tmp_path)
    deck_path = tmp_path / "deck.tcl"
    ops.tcl(str(deck_path))
    text = deck_path.read_text()
    assert text.count("# === Stage: insitu ===") == 1
    assert text.count("# === Stage: relax ===") == 1
    # Two stage-close blocks → two ``wipeAnalysis`` lines.
    assert text.count("wipeAnalysis") == 2
    assert text.count("loadConst -time 0.0") == 2


def test_staged_emit_each_stage_has_its_own_analyze_loop(tmp_path) -> None:
    ops = _build_two_stage_ops(tmp_path)
    deck_path = tmp_path / "deck.tcl"
    ops.tcl(str(deck_path))
    text = deck_path.read_text()
    # Two for-loops (one per stage).  Each wraps analyze 1 with hook
    # dispatcher calls (each stage registered a step_hook_ramp).
    assert text.count("for {set _apesees_i 0}") == 2
    assert "{$_apesees_i < 10}" in text  # stage 1: 10 steps
    assert "{$_apesees_i < 5}" in text   # stage 2: 5 steps


def test_staged_emit_each_stage_has_own_chain_lines(tmp_path) -> None:
    """Each stage emits its own test/algorithm/.../analysis Static."""
    ops = _build_two_stage_ops(tmp_path)
    deck_path = tmp_path / "deck.tcl"
    ops.tcl(str(deck_path))
    text = deck_path.read_text()
    # 2 stages × 7 chain commands = 14 chain lines minimum.  The
    # chain components share underlying primitives across stages
    # (each ``ops.test.NormDispIncr`` etc. registered once in
    # ``_full_chain`` per stage), so the lines come out twice.
    assert text.count("constraints Plain") == 2
    assert text.count("numberer RCM") == 2
    assert text.count("system UmfPack") == 2
    assert text.count("algorithm Newton") == 2
    assert text.count("analysis Static") == 2


def test_staged_emit_resets_hook_state_between_stages(tmp_path) -> None:
    """The hook-list clear lines appear inside both stage_close blocks
    because each stage registered a step_hook_ramp."""
    ops = _build_two_stage_ops(tmp_path)
    deck_path = tmp_path / "deck.tcl"
    ops.tcl(str(deck_path))
    text = deck_path.read_text()
    # The dispatcher boilerplate emits ``set ..._step_hooks {}`` once
    # at first-registration; each of the two stage_close blocks emits
    # the same line again to clear the dispatcher list between
    # stages.  Total = 1 (init) + 2 (per-stage clears) = 3.
    assert text.count("set _apesees_before_step_hooks {}") == 3
    assert text.count("set _apesees_after_step_hooks {}") == 3


# ---------------------------------------------------------------------------
# 6. Live emitter rejection
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_live_stage_open_raises() -> None:
    e = LiveOpsEmitter(wipe=True)
    with pytest.raises(NotImplementedError, match="does not support staged"):
        e.stage_open("insitu")


@pytest.mark.live
def test_live_stage_close_raises() -> None:
    e = LiveOpsEmitter(wipe=True)
    with pytest.raises(NotImplementedError, match="does not support staged"):
        e.stage_close()


def test_apesees_analyze_rejects_staged_models() -> None:
    """``ops.analyze(...)`` (live entry point) refuses staged models
    upfront with a clear error message — before any LiveOpsEmitter
    actually runs."""
    fem = _make_single_quad_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)

    with ops.stage(name="insitu") as s:
        s.add(ops.initial_stress(
            name="rock", pg="Rock",
            sigma_xx=-100.0, sigma_yy=-100.0, sigma_zz=-100.0,
            ramp_steps=10,
        ))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=10, dt=0.1)

    with pytest.raises(NotImplementedError, match="live execution does not support staged"):
        ops.analyze(steps=10, dt=0.1)
