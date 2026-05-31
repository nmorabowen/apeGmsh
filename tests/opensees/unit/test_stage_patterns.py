"""Unit tests for ADR 0051 (BL-3) — stage-scoped load patterns.

``_StageBuilder.pattern(series=)`` returns a stage-owned ``Plain``
(context manager) that emits inside that stage's block — after the
stage's analysis chain, before its ``analyze`` loop — and is frozen by
the stage's ``stage_close`` ``loadConst``.  Claimed patterns are
excluded from the global post-element pattern pass (mirroring the
recorder / constraint claim machinery).

All assertions are deck-level: RecordingEmitter call sequences plus one
Tcl deck-text ordering check.  No GPU / no viewer.
"""
from __future__ import annotations

from typing import Any, cast

import pytest

from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter.recording import RecordingEmitter

from tests.opensees.fixtures.fem_stub import (
    make_two_column_frame,
    make_two_column_frame_partitioned,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _full_chain(ops: apeSees) -> dict[str, object]:
    """A complete analysis chain for ``stage.analysis(**chain)``."""
    return {
        "test":        ops.test.NormDispIncr(tol=1e-4, max_iter=50),
        "algorithm":   ops.algorithm.Newton(),
        "integrator":  ops.integrator.LoadControl(dlam=0.1),
        "constraints": ops.constraints.Plain(),
        "numberer":    ops.numberer.RCM(),
        "system":      ops.system.UmfPack(),
        "analysis":    ops.analysis.Static(),
    }


def _frame_ops(fem: Any) -> apeSees:
    """A 2-D (ndf=3) two-column frame ready for a staged build."""
    ops = apeSees(cast("object", fem), default_orientation=None)
    ops.model(ndm=2, ndf=3)
    t = ops.geomTransf.Linear()
    ops.element.elasticBeamColumn(pg="Cols", transf=t, A=0.01, E=200e9, Iz=1e-4)
    ops.fix(pg="Base", dofs=(1, 1, 1))
    return ops


def _names(rec: RecordingEmitter) -> list[str]:
    return [c[0] for c in rec.calls]


def _owning_rank(calls: list, idx: int) -> int | None:
    """Runtime rank of the ``partition_open`` block enclosing call ``idx``."""
    rank: int | None = None
    for j in range(idx):
        if calls[j][0] == "partition_open":
            rank = calls[j][1][0]
        elif calls[j][0] == "partition_close":
            rank = None
    return rank


# ---------------------------------------------------------------------------
# 1. Builder records & claims the stage pattern
# ---------------------------------------------------------------------------


def test_stage_pattern_records_and_claims() -> None:
    ops = _frame_ops(make_two_column_frame())
    with ops.stage(name="push") as s:
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            p.load(node=2, forces=(50.0, 0.0, 0.0))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=5, dt=0.1)

    assert len(ops._stage_records) == 1
    s_rec = ops._stage_records[0]
    assert len(s_rec.pattern_specs) == 1
    pat = s_rec.pattern_specs[0]
    # Claimed (so the global pass skips it) AND registered (so it has a tag).
    assert id(pat) in ops._stage_claimed_pattern_ids
    assert pat in ops._primitives
    assert ops.tag_for(pat) is not None


def test_pattern_method_returns_context_manager() -> None:
    ops = _frame_ops(make_two_column_frame())
    with ops.stage(name="push") as s:
        pat = s.pattern(series=ops.timeSeries.Linear())
        # It is a context manager (Plain.__enter__/__exit__).
        with pat as p:
            assert p is pat
            p.load(node=2, forces=(1.0, 0.0, 0.0))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    assert ops._stage_records[0].pattern_specs == (pat,)


# ---------------------------------------------------------------------------
# 2. The pattern emits inside the stage block (flat path)
# ---------------------------------------------------------------------------


def test_stage_pattern_load_lands_inside_stage_block() -> None:
    ops = _frame_ops(make_two_column_frame())
    with ops.stage(name="push") as s:
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            p.load(node=2, forces=(50.0, 0.0, 0.0))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=5, dt=0.1)

    rec = RecordingEmitter()
    ops.build().emit(rec)
    names = _names(rec)

    # Exactly one pattern, and it sits inside the only stage block.
    assert names.count("pattern_open") == 1
    i_so = names.index("stage_open")
    i_sc = names.index("stage_close")
    i_po = names.index("pattern_open")
    i_pc = names.index("pattern_close")
    assert i_so < i_po < i_pc < i_sc
    # No pattern leaked into the global pre-stage emit.
    assert i_po > i_so

    # The load lands between pattern_open and pattern_close, verbatim
    # (node= forces are emitted as-given; no broker ndf mapping).
    load_calls = [(j, c) for j, c in enumerate(rec.calls) if c[0] == "load"]
    assert len(load_calls) == 1
    j, c = load_calls[0]
    assert i_po < j < i_pc
    assert c[1] == (2, 50.0, 0.0, 0.0)


def test_stage_pattern_emits_after_chain_before_analyze() -> None:
    ops = _frame_ops(make_two_column_frame())
    with ops.stage(name="push") as s:
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            p.load(node=2, forces=(50.0, 0.0, 0.0))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=5, dt=0.1)

    rec = RecordingEmitter()
    ops.build().emit(rec)
    names = _names(rec)
    i_po = names.index("pattern_open")
    i_analyze = names.index("analyze")
    # The stage's analysis chain emits before the pattern; the pattern
    # emits before analyze (ADR 0051 §6).
    i_analysis_directive = max(
        j for j, c in enumerate(rec.calls) if c[0] == "analysis"
    )
    assert i_analysis_directive < i_po < i_analyze


def test_stage_pattern_tcl_load_before_loadconst(tmp_path) -> None:
    ops = _frame_ops(make_two_column_frame())
    with ops.stage(name="push") as s:
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            p.load(node=2, forces=(50.0, 0.0, 0.0))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=5, dt=0.1)

    deck = tmp_path / "deck.tcl"
    ops.tcl(str(deck))
    lines = deck.read_text().splitlines()

    i_stage = next(
        i for i, ln in enumerate(lines) if ln.strip().startswith("# === Stage:")
    )
    i_pattern = next(
        i for i, ln in enumerate(lines) if ln.strip().startswith("pattern Plain")
    )
    # The load command is indented inside the Plain pattern block.
    i_load = next(
        i for i, ln in enumerate(lines) if ln.strip().startswith("load ")
    )
    i_lc = next(
        i for i, ln in enumerate(lines) if ln.strip().startswith("loadConst")
    )
    # pattern + load are inside the stage block and frozen by the
    # stage's loadConst.
    assert i_stage < i_pattern < i_load < i_lc


# ---------------------------------------------------------------------------
# 3. Two stages carry independent patterns
# ---------------------------------------------------------------------------


def test_two_stages_have_independent_patterns() -> None:
    ops = _frame_ops(make_two_column_frame())
    with ops.stage(name="push") as s:
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            p.load(node=2, forces=(50.0, 0.0, 0.0))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=5, dt=0.1)
    with ops.stage(name="pull") as s:
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            p.load(node=4, forces=(0.0, -30.0, 0.0))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=3, dt=0.1)

    rec = RecordingEmitter()
    ops.build().emit(rec)
    names = _names(rec)

    assert names.count("pattern_open") == 2
    assert names.count("stage_open") == 2

    stage_open_idx = [j for j, c in enumerate(rec.calls) if c[0] == "stage_open"]
    stage_close_idx = [
        j for j, c in enumerate(rec.calls) if c[0] == "stage_close"
    ]
    # Stage 1 block holds the node-2 load; stage 2 holds the node-4 load.
    s1 = (stage_open_idx[0], stage_close_idx[0])
    s2 = (stage_open_idx[1], stage_close_idx[1])

    def _loads_in(block) -> list[tuple]:
        lo, hi = block
        return [c[1] for j, c in enumerate(rec.calls)
                if c[0] == "load" and lo < j < hi]

    assert _loads_in(s1) == [(2, 50.0, 0.0, 0.0)]
    assert _loads_in(s2) == [(4, 0.0, -30.0, 0.0)]


# ---------------------------------------------------------------------------
# 4. from_model inside a stage pattern
# ---------------------------------------------------------------------------


def test_stage_pattern_from_model_imports_case() -> None:
    from apeGmsh._kernel.record_sets import NodalLoadSet
    from apeGmsh._kernel.records._loads import NodalLoadRecord

    fem = make_two_column_frame()
    fem.nodes.loads = NodalLoadSet([  # type: ignore[attr-defined]
        NodalLoadRecord(node_id=2, force_xyz=(10.0, 20.0, 0.0),
                        pattern="live"),
        NodalLoadRecord(node_id=4, force_xyz=(30.0, 40.0, 0.0),
                        pattern="live"),
    ])
    ops = _frame_ops(fem)
    with ops.stage(name="push") as s:
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            p.from_model("live")
        s.analysis(**_full_chain(ops))
        s.run(n_increments=5, dt=0.1)

    rec = RecordingEmitter()
    ops.build().emit(rec)
    names = _names(rec)

    i_po = names.index("pattern_open")
    i_pc = names.index("pattern_close")
    i_so = names.index("stage_open")
    i_sc = names.index("stage_close")
    assert i_so < i_po < i_pc < i_sc

    # ndf=3 -> (node, fx, fy, mz); moment absent -> mz = 0.0.  Both
    # land inside the stage pattern block.
    block_loads = [c[1] for j, c in enumerate(rec.calls)
                   if c[0] == "load" and i_po < j < i_pc]
    assert (2, 10.0, 20.0, 0.0) in block_loads
    assert (4, 30.0, 40.0, 0.0) in block_loads


def test_stage_pattern_from_model_imports_prescribed_sp_only() -> None:
    from apeGmsh._kernel.record_sets import SPSet
    from apeGmsh._kernel.records._loads import SPRecord

    fem = make_two_column_frame()
    fem.nodes.sp = SPSet([  # type: ignore[attr-defined]
        SPRecord(node_id=2, dof=1, value=0.01,
                 is_homogeneous=False, pattern="settle"),
        SPRecord(node_id=4, dof=2, value=0.0,
                 is_homogeneous=True, pattern="settle"),  # fix — skipped
    ])
    ops = _frame_ops(fem)
    with ops.stage(name="push") as s:
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            p.from_model("settle")
        s.analysis(**_full_chain(ops))
        s.run(n_increments=5, dt=0.1)

    rec = RecordingEmitter()
    ops.build().emit(rec)
    names = _names(rec)
    i_po = names.index("pattern_open")
    i_pc = names.index("pattern_close")

    sp_block = [c[1] for j, c in enumerate(rec.calls)
                if c[0] == "sp" and i_po < j < i_pc]
    assert (2, 1, 0.01) in sp_block           # prescribed -> imported
    assert all(c[0] != 4 for c in sp_block)   # homogeneous fix -> skipped


def test_stage_pattern_mix_from_model_and_explicit_load() -> None:
    from apeGmsh._kernel.record_sets import NodalLoadSet
    from apeGmsh._kernel.records._loads import NodalLoadRecord

    fem = make_two_column_frame()
    fem.nodes.loads = NodalLoadSet([  # type: ignore[attr-defined]
        NodalLoadRecord(node_id=2, force_xyz=(10.0, 20.0, 0.0),
                        pattern="live"),
    ])
    ops = _frame_ops(fem)
    with ops.stage(name="push") as s:
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            p.load(node=4, forces=(99.0, 0.0, 0.0))
            p.from_model("live")
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)

    rec = RecordingEmitter()
    ops.build().emit(rec)
    names = _names(rec)
    i_po = names.index("pattern_open")
    i_pc = names.index("pattern_close")
    block_loads = [c[1] for j, c in enumerate(rec.calls)
                   if c[0] == "load" and i_po < j < i_pc]
    assert (4, 99.0, 0.0, 0.0) in block_loads   # explicit
    assert (2, 10.0, 20.0, 0.0) in block_loads  # from_model


# ---------------------------------------------------------------------------
# 5. A global pattern (non-staged) still emits globally — no claim
# ---------------------------------------------------------------------------


def test_global_pattern_unaffected_by_bl3() -> None:
    """A vanilla ``ops.pattern.Plain`` (no stages) still emits in the
    global post-element pass — the BL-3 claim machinery is empty."""
    fem = make_two_column_frame()
    ops = _frame_ops(fem)
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(node=2, forces=(50.0, 0.0, 0.0))

    rec = RecordingEmitter()
    ops.build().emit(rec)
    names = _names(rec)
    assert names.count("pattern_open") == 1
    assert "stage_open" not in names
    assert ops.build()._claimed_pattern_ids() == set()


# ---------------------------------------------------------------------------
# 6. Partitioned — stage pattern routes per rank
# ---------------------------------------------------------------------------


def test_stage_pattern_partitioned_routes_per_rank() -> None:
    fem = make_two_column_frame_partitioned()  # rank0: 1,2 / rank1: 3,4
    ops = _frame_ops(fem)
    with ops.stage(name="push") as s:
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            p.load(node=2, forces=(50.0, 0.0, 0.0))   # rank 0
            p.load(node=4, forces=(0.0, -30.0, 0.0))  # rank 1
        s.analysis(**_full_chain(ops))
        s.run(n_increments=2, dt=0.1)

    rec = RecordingEmitter()
    ops.build().emit(rec)
    names = _names(rec)

    # Inside the stage block.
    i_so = names.index("stage_open")
    i_sc = names.index("stage_close")

    load2 = [j for j, c in enumerate(rec.calls)
             if c[0] == "load" and c[1] == (2, 50.0, 0.0, 0.0)]
    load4 = [j for j, c in enumerate(rec.calls)
             if c[0] == "load" and c[1] == (4, 0.0, -30.0, 0.0)]
    assert len(load2) == 1 and len(load4) == 1
    assert i_so < load2[0] < i_sc
    assert i_so < load4[0] < i_sc

    # node-2 load is emitted inside rank-0's partition block; node-4
    # inside rank-1's.  No cross-rank leakage.
    assert _owning_rank(rec.calls, load2[0]) == 0
    assert _owning_rank(rec.calls, load4[0]) == 1


def test_stage_pattern_partitioned_skips_empty_rank_bracket() -> None:
    """A rank that owns no pattern node must NOT open an empty
    ``partition_open`` bracket (an empty ``if getPID()==K:`` body is a
    Python SyntaxError on the Py emitter)."""
    fem = make_two_column_frame_partitioned()  # rank0: 1,2 / rank1: 3,4
    ops = _frame_ops(fem)
    with ops.stage(name="push") as s:
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            p.load(node=2, forces=(50.0, 0.0, 0.0))  # rank 0 only
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)

    rec = RecordingEmitter()
    ops.build().emit(rec)

    # The stage-pattern per-rank pass opens exactly one partition block
    # (rank 0); rank 1 owns no pattern node so its bracket is skipped.
    i_so = [j for j, c in enumerate(rec.calls) if c[0] == "stage_open"][0]
    i_sc = [j for j, c in enumerate(rec.calls) if c[0] == "stage_close"][0]
    # Collect partition_open ranks that wrap a pattern_open within the stage.
    pat_open_idx = [j for j, c in enumerate(rec.calls)
                    if c[0] == "pattern_open" and i_so < j < i_sc]
    assert len(pat_open_idx) == 1
    assert _owning_rank(rec.calls, pat_open_idx[0]) == 0
