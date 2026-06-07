"""ADR 0052 — stage-bound `s.support` HOLD supports under MP.

Extends the 4-quad / 2-partition fixture from
`test_emit_partitioned_stage_bound_bcs.py` to exercise the per-rank HOLD
fan-out (the partitioned slice that lifted the prior fail-loud guard in
`_emit_stages_partitioned`). Locked invariants:

1. A `s.support` on a node owned by rank K lands inside
   `partition_open(K)` for that stage, wrapped in a `pattern_open`/
   `pattern_close` block (the stage's dedicated `Plain` HOLD pattern,
   emitted locally per owning rank — same convention as
   `_emit_one_pattern_partitioned`).
2. A rank that owns none of the held nodes emits no `sp_hold`.
3. HOLD supports never leak into the global (rank=None) scope.
4. The global `domain_change` fires once for a support-only stage
   (unified gate covers HOLD supports).
5. The held `sp_hold` lines fan out per owning rank: a held node on
   rank 1 emits only on rank 1, never on rank 0.
"""
from __future__ import annotations

from typing import cast

from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter.recording import RecordingEmitter

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


# ---------------------------------------------------------------------------
# Fixture — same shape as test_emit_partitioned_stage_bound_bcs.py
# ---------------------------------------------------------------------------


def _make_4quad_2pg_2part_fem() -> FEMStub:
    fem = FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            coords=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (2.0, 0.0, 0.0),
                (2.0, 1.0, 0.0),
                (5.0, 0.0, 0.0),
                (6.0, 0.0, 0.0),
                (6.0, 1.0, 0.0),
                (5.0, 1.0, 0.0),
                (7.0, 0.0, 0.0),
                (7.0, 1.0, 0.0),
            ],
            node_pgs={
                "rock_base": [1],
                "cimbra_base": [7],
                "cimbra_top": [9, 12],
            },
        ),
        elements=_ElementsStub(
            elem_pgs={
                "rock":   _ElementGroupView(
                    ids=(1, 2),
                    connectivity=((1, 2, 3, 4), (2, 5, 6, 3)),
                ),
                "cimbra": _ElementGroupView(
                    ids=(3, 4),
                    connectivity=((7, 8, 9, 10), (8, 11, 12, 9)),
                ),
            },
        ),
    )
    fem.set_partitions([
        (0, [1, 2, 3, 4, 5, 6], [1, 2]),
        (1, [7, 8, 9, 10, 11, 12], [3, 4]),
    ])
    return fem


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


def _setup_with_stage_bound_support(fem: FEMStub) -> apeSees:
    """Stage 1 = rock only.  Stage 2 = install cimbra + s.support HOLD on
    cimbra_top (nodes 9, 12 — both rank-1 owned), dof 2 only."""
    ops = apeSees(cast("object", fem), default_orientation=None)  # type: ignore[arg-type]
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="cimbra", thickness=1.0, material=mat)
    ops.fix(pg="rock_base", dofs=(1, 1))

    with ops.stage(name="rock_only") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)

    with ops.stage(name="install_cimbra") as s:
        s.activate(pgs=["cimbra"])
        s.support(pg="cimbra_top", dofs=(0, 1))  # nodes 9, 12 — rank 1
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    return ops


def _bucket_calls_by_scope(
    rec: RecordingEmitter,
) -> dict[tuple[int, "int | None"], list[tuple[str, tuple, dict]]]:
    """Bucket recorded calls by `(stage_idx, rank)` — same helper as
    test_emit_partitioned_stage_bound_bcs.py."""
    buckets: dict[
        tuple[int, "int | None"], list[tuple[str, tuple, dict]]
    ] = {}
    stage_idx = -1
    rank: "int | None" = None
    for name, args, kwargs in rec.calls:
        if name == "stage_open":
            stage_idx += 1
            continue
        if name == "stage_close":
            continue
        if name == "partition_open":
            rank = int(args[0])
            continue
        if name == "partition_close":
            rank = None
            continue
        buckets.setdefault((stage_idx, rank), []).append(
            (name, args, kwargs),
        )
    return buckets


# ---------------------------------------------------------------------------
# 1. HOLD lands inside the owning rank's bracket, wrapped in a pattern
# ---------------------------------------------------------------------------


def test_stage2_support_inside_rank1_partition_open() -> None:
    fem = _make_4quad_2pg_2part_fem()
    ops = _setup_with_stage_bound_support(fem)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    buckets = _bucket_calls_by_scope(rec)

    stage1_rank1 = buckets.get((1, 1), [])
    hold_calls = [c for c in stage1_rank1 if c[0] == "sp_hold"]
    # Two held nodes (9, 12), dof 2 each → one sp_hold per node.
    assert len(hold_calls) == 2
    held = sorted((int(c[1][0]), int(c[1][1])) for c in hold_calls)
    assert held == [(9, 2), (12, 2)]


def test_stage2_support_wrapped_in_pattern_on_rank1() -> None:
    """The HOLD sp lines on rank 1 are bracketed by a pattern_open /
    pattern_close pair (the stage's dedicated Plain HOLD pattern)."""
    fem = _make_4quad_2pg_2part_fem()
    ops = _setup_with_stage_bound_support(fem)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    buckets = _bucket_calls_by_scope(rec)

    names = [c[0] for c in buckets.get((1, 1), [])]
    # Exactly one pattern_open / pattern_close pair for the HOLD block.
    assert names.count("pattern_open") == 1
    assert names.count("pattern_close") == 1
    po = names.index("pattern_open")
    pc = names.index("pattern_close")
    inner = names[po + 1:pc]
    assert inner == ["sp_hold", "sp_hold"], (
        f"pattern body should be exactly the two HOLD sp lines; got {inner}"
    )


# ---------------------------------------------------------------------------
# 2. Rank 0 owns none of the held nodes → no HOLD on rank 0
# ---------------------------------------------------------------------------


def test_stage2_rank0_block_has_no_support() -> None:
    fem = _make_4quad_2pg_2part_fem()
    ops = _setup_with_stage_bound_support(fem)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    buckets = _bucket_calls_by_scope(rec)

    stage1_rank0 = buckets.get((1, 0), [])
    assert not [c for c in stage1_rank0 if c[0] == "sp_hold"], (
        "rank 0 owns none of the held nodes (9, 12) — it must emit no "
        f"sp_hold; got {[c for c in stage1_rank0 if c[0] == 'sp_hold']}"
    )


# ---------------------------------------------------------------------------
# 3. HOLD supports never leak into the global (rank=None) scope
# ---------------------------------------------------------------------------


def test_support_never_in_global_scope() -> None:
    fem = _make_4quad_2pg_2part_fem()
    ops = _setup_with_stage_bound_support(fem)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    buckets = _bucket_calls_by_scope(rec)

    for (stage_idx, rank), calls in buckets.items():
        if rank is None:
            stray = [c for c in calls if c[0] == "sp_hold"]
            assert stray == [], (
                f"unexpected sp_hold in global scope "
                f"(stage={stage_idx}, rank=None): {stray}"
            )


# ---------------------------------------------------------------------------
# 4. Unified gate — a support-only stage still drives the per-rank loop +
#    a single global domain_change
# ---------------------------------------------------------------------------


def test_support_only_stage_drives_per_rank_loop_and_domain_change() -> None:
    """A stage whose only content is `s.support` (no activation, no
    other BCs) must still open the owning rank's bracket and emit one
    global domain_change (unified gate covers HOLD supports)."""
    fem = _make_4quad_2pg_2part_fem()
    ops = apeSees(cast("object", fem), default_orientation=None)  # type: ignore[arg-type]
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="cimbra", thickness=1.0, material=mat)
    ops.fix(pg="rock_base", dofs=(1, 1))

    # Single stage: no activation, only a HOLD support on the globally-
    # emitted node 3 (rank 0 owned, not otherwise constrained).
    with ops.stage(name="support_only") as s:
        s.support(nodes=[3], dofs=(0, 1))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)

    rec = RecordingEmitter()
    ops.build().emit(rec)
    buckets = _bucket_calls_by_scope(rec)

    # Rank 0 owns node 3 → its stage block carries exactly one sp_hold.
    stage0_rank0 = buckets.get((0, 0), [])
    hold_calls = [c for c in stage0_rank0 if c[0] == "sp_hold"]
    assert len(hold_calls) == 1
    assert int(hold_calls[0][1][0]) == 3

    # Rank 1 owns no held node here — its bracket should be skipped.
    assert (0, 1) not in buckets, (
        f"rank 1 stage 0 bracket should be SKIPPED (no content); "
        f"got entries: {buckets.get((0, 1), [])}"
    )

    # Global domain_change must still fire exactly once.
    stage0_global = buckets.get((0, None), [])
    dc_calls = [c for c in stage0_global if c[0] == "domain_change"]
    assert len(dc_calls) == 1, (
        f"support-only stage should emit one global domain_change; "
        f"got {dc_calls}"
    )
