"""TIMs A8 — ``s.profile(deep=, memory=, per_step=)`` per-stage profiler bracket.

Covers:

- ``_StageBuilder.profile`` populates ``StageRecord.profile`` (a
  :class:`ProfileRecord`); a second call raises; a stage that never
  calls it keeps ``profile is None`` (old fixtures unaffected).
- Flat emit (``RecordingEmitter``): ``profiler`` calls bracket ONLY the
  profiled stage's ``analyze`` call — a sibling stage in the same model
  is unbracketed.
- Tcl / py deck text: locks the literal ``profiler start [flags]`` /
  ``profiler report <stage name>.h5`` lines and their flag variants
  (mirrors ``ops.profiler.start``'s three flags exactly).
- ``Pardiso(stats=True)`` still emits ``-stats`` on its ``system Pardiso``
  line inside a stage, with and without ``s.profile`` on that stage.
- H5 archival of ``s.profile`` refuses loudly (mirrors the
  ``phantom_node_tags`` refusal in ``H5Emitter.set_stage_records``).
"""
from __future__ import annotations

import pytest

from apeGmsh.opensees.apesees import _StageBuilder, apeSees
from apeGmsh.opensees._internal.build import ProfileRecord
from apeGmsh.opensees.emitter.h5 import H5Emitter
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
# Fixtures
# ---------------------------------------------------------------------------


def _make_two_pg_fem() -> FEMStub:
    """Two-PG quad pair: rock (1) + cimbra (2), 6 nodes."""
    return FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4, 5, 6],
            coords=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (2.0, 0.0, 0.0),
                (2.0, 1.0, 0.0),
            ],
            node_pgs={
                "Left":       [1, 4],
                "CimbraOnly": [5, 6],
            },
        ),
        elements=_ElementsStub(
            elem_pgs={
                "rock":   _ElementGroupView(
                    ids=(1,), connectivity=((1, 2, 3, 4),),
                ),
                "cimbra": _ElementGroupView(
                    ids=(2,), connectivity=((2, 5, 6, 3),),
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


def _two_stage_ops() -> apeSees:
    fem = _make_two_pg_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="cimbra", thickness=1.0, material=mat)
    return ops


def _bucket_flat(rec: RecordingEmitter) -> dict[int, list[tuple[str, tuple, dict]]]:
    """Bucket recorded calls by stage_idx (-1 = pre-stage)."""
    buckets: dict[int, list[tuple[str, tuple, dict]]] = {}
    stage_idx = -1
    for name, args, kwargs in rec.calls:
        if name == "stage_open":
            stage_idx += 1
            continue
        if name == "stage_close":
            continue
        buckets.setdefault(stage_idx, []).append((name, args, kwargs))
    return buckets


# ===========================================================================
# __slots__ + builder behavior
# ===========================================================================


def test_stage_builder_slots_include_profile_field() -> None:
    assert "_profile" in _StageBuilder.__slots__


def test_s_profile_populates_stage_record_with_flags() -> None:
    ops = _two_stage_ops()
    with ops.stage(name="dyn") as s:
        s.profile(deep=True, memory=True, per_step=True)
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    rec = ops._stage_records[0]
    assert rec.profile == ProfileRecord(deep=True, memory=True, per_step=True)


def test_s_profile_defaults_all_false() -> None:
    ops = _two_stage_ops()
    with ops.stage(name="dyn") as s:
        s.profile()
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    rec = ops._stage_records[0]
    assert rec.profile == ProfileRecord(deep=False, memory=False, per_step=False)


def test_stage_without_profile_call_keeps_none_default() -> None:
    """A stage that never calls ``s.profile`` keeps ``profile is None``
    — old fixtures / stages are unaffected."""
    ops = _two_stage_ops()
    with ops.stage(name="bare") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    assert ops._stage_records[0].profile is None


def test_s_profile_called_twice_raises() -> None:
    ops = _two_stage_ops()
    with ops.stage(name="dyn") as s:
        s.profile(deep=True)
        with pytest.raises(ValueError, match="already called"):
            s.profile()
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)


# ===========================================================================
# Flat emit — bracket scoped to the profiled stage only
# ===========================================================================


def test_flat_profiler_brackets_only_the_profiled_stage() -> None:
    """Two stages; only the second calls ``s.profile``.  The recording
    emitter must show ``profiler`` calls in stage 1's bucket only."""
    ops = _two_stage_ops()
    with ops.stage(name="quiet") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="profiled") as s:
        s.profile(deep=True)
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    buckets = _bucket_flat(rec)
    stage0_profiler = [c for c in buckets.get(0, []) if c[0] == "profiler"]
    stage1_profiler = [c for c in buckets.get(1, []) if c[0] == "profiler"]
    assert stage0_profiler == [], "unprofiled stage must carry no profiler calls"
    assert len(stage1_profiler) == 3
    assert stage1_profiler[0] == ("profiler", ("start", "-deep"), {})
    assert stage1_profiler[1] == ("profiler", ("stop",), {})
    assert stage1_profiler[2] == ("profiler", ("report", "profiled.h5"), {})


def test_flat_profiler_start_brackets_immediately_around_analyze() -> None:
    """``profiler start`` sits immediately before ``analyze``; ``profiler
    report`` sits immediately after — no other calls in between."""
    ops = _two_stage_ops()
    with ops.stage(name="dyn") as s:
        s.profile()
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    names = [c[0] for c in rec.calls]
    start_i = names.index("profiler")
    analyze_i = names.index("analyze")
    report_i = len(names) - 1 - names[::-1].index("profiler")
    assert start_i + 1 == analyze_i
    assert analyze_i + 2 == report_i
    assert rec.calls[start_i] == ("profiler", ("start",), {})
    assert rec.calls[analyze_i + 1] == ("profiler", ("stop",), {})
    assert rec.calls[report_i] == ("profiler", ("report", "dyn.h5"), {})


# ===========================================================================
# Tcl / py deck text — literal lines + flag variants
# ===========================================================================


def test_tcl_emit_profiler_bracket_lines() -> None:
    ops = _two_stage_ops()
    with ops.stage(name="quiet") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="profiled") as s:
        s.profile(deep=True, memory=True, per_step=True)
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    emitter = TclEmitter()
    ops.build().emit(emitter)
    lines = emitter.lines()
    text = "\n".join(lines)
    assert "profiler start -deep -memory -perStep" in text
    assert "profiler stop" in text
    assert "profiler report profiled.h5" in text
    # The unprofiled stage's block carries no ``profiler`` line at all.
    quiet_i = lines.index("# === Stage: quiet ===")
    profiled_i = lines.index("# === Stage: profiled ===")
    quiet_block = lines[quiet_i:profiled_i]
    assert not any(ln.startswith("profiler") for ln in quiet_block)


def test_py_emit_profiler_bracket_lines() -> None:
    ops = _two_stage_ops()
    with ops.stage(name="profiled") as s:
        s.profile(deep=True)
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    emitter = PyEmitter()
    ops.build().emit(emitter)
    text = "\n".join(emitter.lines())
    assert "ops.profiler('start', '-deep')" in text
    assert "ops.profiler('stop')" in text
    assert "ops.profiler('report', 'profiled.h5')" in text


@pytest.mark.parametrize(
    "kwargs,expected_flags",
    [
        ({}, ""),
        ({"deep": True}, " -deep"),
        ({"memory": True}, " -memory"),
        ({"per_step": True}, " -perStep"),
        (
            {"deep": True, "memory": True, "per_step": True},
            " -deep -memory -perStep",
        ),
    ],
)
def test_tcl_start_flags_match_each_kwarg(
    kwargs: dict[str, bool], expected_flags: str,
) -> None:
    """Each of the three flags changes the emitted ``profiler start``
    line the same way ``ops.profiler.start``'s flags do."""
    ops = _two_stage_ops()
    with ops.stage(name="dyn") as s:
        s.profile(**kwargs)
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    emitter = TclEmitter()
    ops.build().emit(emitter)
    text = "\n".join(emitter.lines())
    assert f"profiler start{expected_flags}" in text


# ===========================================================================
# Pardiso(stats=True) — lock the -stats flag inside a stage
# (was Mumps: ADR 0106 D5 refuses an explicit Mumps on a serial deck)
# ===========================================================================


def _pardiso_chain(ops: apeSees) -> dict[str, object]:
    chain = _full_chain(ops)
    chain["system"] = ops.system.Pardiso(stats=True)
    return chain


def test_pardiso_stats_emits_inside_stage_without_profile() -> None:
    ops = _two_stage_ops()
    with ops.stage(name="dyn") as s:
        s.analysis(**_pardiso_chain(ops))
        s.run(n_increments=1)
    emitter = TclEmitter()
    ops.build().emit(emitter)
    text = "\n".join(emitter.lines())
    assert "-stats" in text
    assert "system Pardiso" in text


def test_pardiso_stats_emits_inside_stage_with_profile() -> None:
    """``s.profile`` must not interfere with the stage's own
    ``system Pardiso ... -stats`` line."""
    ops = _two_stage_ops()
    with ops.stage(name="dyn") as s:
        s.profile(deep=True)
        s.analysis(**_pardiso_chain(ops))
        s.run(n_increments=1)
    emitter = TclEmitter()
    ops.build().emit(emitter)
    lines = emitter.lines()
    text = "\n".join(lines)
    assert "-stats" in text
    pardiso_line = next(ln for ln in lines if ln.startswith("system Pardiso"))
    assert "-stats" in pardiso_line
    # And the profiler bracket is still there, distinct from the chain line.
    assert "profiler start -deep" in text
    assert "profiler report dyn.h5" in text


# ===========================================================================
# H5 archival — refuse loudly (mirrors the phantom_node_tags refusal)
# ===========================================================================


def test_h5_set_stage_records_refuses_profile() -> None:
    ops = _two_stage_ops()
    with ops.stage(name="dyn") as s:
        s.profile(deep=True)
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    bm = ops.build()
    emitter = H5Emitter(model_name="m", snapshot_id="")
    bm.emit(emitter)
    with pytest.raises(NotImplementedError, match="s.profile"):
        emitter.set_stage_records(bm.stage_records)


def test_h5_set_stage_records_ok_without_profile() -> None:
    """A stage that never calls ``s.profile`` archives fine — the new
    refusal is scoped to ``profile is not None`` only."""
    ops = _two_stage_ops()
    with ops.stage(name="dyn") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    bm = ops.build()
    emitter = H5Emitter(model_name="m", snapshot_id="")
    bm.emit(emitter)
    emitter.set_stage_records(bm.stage_records)  # no raise
