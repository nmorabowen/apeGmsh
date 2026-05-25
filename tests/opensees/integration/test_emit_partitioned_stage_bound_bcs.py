"""Phase SSI-2.D PR-B — stage-bound `s.fix` / `s.mass` under MP.

Extends the 4-quad fixture from `test_emit_partitioned_staged.py` to
exercise the per-rank BC fan-out introduced by PR-B. Locked invariants:

1. Stage-bound fix on a stage-bound node lands inside `partition_open(K)`
   for the rank that owns the node (rank 1 for cimbra in this fixture);
   rank 0's stage block contains no such fix.
2. The empty-bracket skip (Phase SSI-2.D, Red #12) fires correctly:
   when a stage has BCs only on rank 1, rank 0 emits NO
   `partition_open(0)` block in that stage.
3. The global `domain_change` after the per-rank loop fires once per
   stage that adds BCs, regardless of which ranks participated.
4. The unified gate (Red #21) lifts: a stage with BCs but no
   activation still drives the per-rank loop + global `domain_change`.
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
# Fixture — same shape as test_emit_partitioned_staged.py
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


def _setup_with_stage_bound_bcs(fem: FEMStub) -> apeSees:
    """Stage 1 = rock only.  Stage 2 = install cimbra + s.fix on
    stage-bound cimbra_base (node 7) + s.mass on stage-bound cimbra_top
    (nodes 9, 12).  All stage-bound BC targets are on rank 1."""
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
        s.fix(pg="cimbra_base", dofs=(1, 1))    # node 7 — stage-bound, rank 1
        s.mass(pg="cimbra_top", values=(100.0, 100.0))  # nodes 9, 12 — rank 1
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    return ops


def _bucket_calls_by_scope(
    rec: RecordingEmitter,
) -> dict[tuple[int, "int | None"], list[tuple[str, tuple, dict]]]:
    """Bucket recorded calls by `(stage_idx, rank)` — same shape as
    the helper in `test_emit_partitioned_staged.py`."""
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
# 1. Stage-bound fix on stage-bound node lands inside owning rank's block
# ---------------------------------------------------------------------------


def test_stage2_fix_inside_rank1_partition_open() -> None:
    """Stage 2's `s.fix(pg='cimbra_base')` resolves to node 7, which
    rank 1 owns; the `fix 7 ...` line must land inside
    `partition_open(1)` in stage 2."""
    fem = _make_4quad_2pg_2part_fem()
    ops = _setup_with_stage_bound_bcs(fem)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    buckets = _bucket_calls_by_scope(rec)

    stage1_rank1 = buckets.get((1, 1), [])
    fix_calls = [c for c in stage1_rank1 if c[0] == "fix"]
    assert len(fix_calls) == 1, (
        f"stage 2 rank 1 should emit 1 fix call (node 7); got {fix_calls}"
    )
    assert int(fix_calls[0][1][0]) == 7


def test_stage2_mass_inside_rank1_partition_open() -> None:
    fem = _make_4quad_2pg_2part_fem()
    ops = _setup_with_stage_bound_bcs(fem)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    buckets = _bucket_calls_by_scope(rec)

    stage1_rank1 = buckets.get((1, 1), [])
    mass_calls = [c for c in stage1_rank1 if c[0] == "mass"]
    # Two nodes (9, 12) — one mass call each.
    assert len(mass_calls) == 2
    mass_node_ids = sorted(int(c[1][0]) for c in mass_calls)
    assert mass_node_ids == [9, 12]


def test_stage2_rank0_block_has_no_stage_bound_bcs() -> None:
    """All stage-bound BCs target rank-1-owned nodes; rank 0's stage 2
    block must contain zero fix / mass calls."""
    fem = _make_4quad_2pg_2part_fem()
    ops = _setup_with_stage_bound_bcs(fem)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    buckets = _bucket_calls_by_scope(rec)

    stage1_rank0 = buckets.get((1, 0), [])
    assert not [c for c in stage1_rank0 if c[0] in ("fix", "mass")], (
        f"stage 2 rank 0 should emit no fix/mass (all targets on rank 1); "
        f"got {[c for c in stage1_rank0 if c[0] in ('fix', 'mass')]}"
    )


# ---------------------------------------------------------------------------
# 2. Stage-bound BCs do not leak into the global (rank=None) scope
# ---------------------------------------------------------------------------


def test_stage_bound_fix_mass_never_in_global_scope() -> None:
    fem = _make_4quad_2pg_2part_fem()
    ops = _setup_with_stage_bound_bcs(fem)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    buckets = _bucket_calls_by_scope(rec)

    for (stage_idx, rank), calls in buckets.items():
        if rank is None:
            stray = [c for c in calls if c[0] in ("fix", "mass")]
            assert stray == [], (
                f"unexpected fix/mass in global scope "
                f"(stage={stage_idx}, rank=None): {stray}"
            )


# ---------------------------------------------------------------------------
# 3. Global domain_change still fires once per stage with BCs
# ---------------------------------------------------------------------------


def test_stage2_global_domain_change_fires_with_bcs() -> None:
    fem = _make_4quad_2pg_2part_fem()
    ops = _setup_with_stage_bound_bcs(fem)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    buckets = _bucket_calls_by_scope(rec)

    stage1_global = buckets.get((1, None), [])
    dc_calls = [c for c in stage1_global if c[0] == "domain_change"]
    assert len(dc_calls) == 1, (
        f"stage 2 should emit exactly one global domain_change "
        f"(unified gate covers topology + BCs); got {dc_calls}"
    )


# ---------------------------------------------------------------------------
# 4. Unified gate — BC-only stage (no activation) still drives per-rank loop
# ---------------------------------------------------------------------------


def test_bc_only_stage_drives_per_rank_loop_and_domain_change() -> None:
    """A stage with stage-bound BCs but no `s.activate(...)` must
    still trigger the per-rank loop on owning ranks and emit a
    global `domain_change` (Red #21 unified gate)."""
    fem = _make_4quad_2pg_2part_fem()
    ops = apeSees(cast("object", fem), default_orientation=None)  # type: ignore[arg-type]
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="cimbra", thickness=1.0, material=mat)

    # Single stage: no activation, only a BC on the globally-emitted
    # node 1 (rank 0 owned).
    with ops.stage(name="bc_only") as s:
        s.fix(nodes=[1], dofs=(1, 1))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)

    rec = RecordingEmitter()
    ops.build().emit(rec)
    buckets = _bucket_calls_by_scope(rec)

    # Rank 0 owns node 1 → its stage block should carry exactly one fix.
    stage0_rank0 = buckets.get((0, 0), [])
    fix_calls = [c for c in stage0_rank0 if c[0] == "fix"]
    assert len(fix_calls) == 1
    assert int(fix_calls[0][1][0]) == 1

    # Rank 1 owns no BC targets here — its bracket should NOT exist
    # in stage 0 (empty-bracket skip — Red #12 Py-emitter safety).
    assert (0, 1) not in buckets, (
        f"rank 1 stage 0 bracket should be SKIPPED (no content); "
        f"got entries: {buckets.get((0, 1), [])}"
    )

    # Global domain_change must still fire.
    stage0_global = buckets.get((0, None), [])
    dc_calls = [c for c in stage0_global if c[0] == "domain_change"]
    assert len(dc_calls) == 1, (
        f"BC-only stage should emit one global domain_change; "
        f"got {dc_calls}"
    )
