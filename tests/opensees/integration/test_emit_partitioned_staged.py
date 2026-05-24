"""Phase SSI-2.C: combining MP partitions with staged builds.

Geometry: four quads in two disjoint pairs (no shared interface so
partitions are clean).  Rock pair is global; cimbra pair activates in
stage 2 and carries an ``initial_stress`` record so the per-rank
``addToParameter`` fan-out is exercised end-to-end.

  Rock pair (left, partition 0 — global):
    quad 1: nodes 1, 2, 3, 4
    quad 2: nodes 2, 5, 6, 3
  Cimbra pair (right, partition 1 — activated in stage 2):
    quad 3: nodes 7, 8, 9, 10
    quad 4: nodes 8, 11, 12, 9

Verified invariants:

1. The build no longer raises ``NotImplementedError`` (the prior gate
   in :meth:`BuiltModel.emit` was lifted).
2. Pre-stage per-rank blocks carry only globally-owned topology
   (rock on rank 0; rank 1 empty since cimbra is stage-bound).
3. Stage 2's per-rank topology pass carries cimbra on rank 1
   (rank 0 stays empty in stage 2).
4. Stage 2's ``domainChange`` is emitted GLOBALLY (rank ``None``) and
   appears AFTER the activated cimbra elements.
5. ``addToParameter`` calls land INSIDE ``partition_open(1)`` (the
   rank owning cimbra), never in the global scope.
"""
from __future__ import annotations

from typing import cast

from apeGmsh.opensees.apesees import apeSees
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


def _make_4quad_2pg_2part_fem() -> FEMStub:
    """4 quads in 2 disjoint pairs, partitioned 0 | 1.

    Rock (rank 0): nodes 1-6, elements 1-2.
    Cimbra (rank 1): nodes 7-12, elements 3-4.
    """
    fem = FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            coords=[
                (0.0, 0.0, 0.0),  # 1
                (1.0, 0.0, 0.0),  # 2
                (1.0, 1.0, 0.0),  # 3
                (0.0, 1.0, 0.0),  # 4
                (2.0, 0.0, 0.0),  # 5
                (2.0, 1.0, 0.0),  # 6
                (5.0, 0.0, 0.0),  # 7
                (6.0, 0.0, 0.0),  # 8
                (6.0, 1.0, 0.0),  # 9
                (5.0, 1.0, 0.0),  # 10
                (7.0, 0.0, 0.0),  # 11
                (7.0, 1.0, 0.0),  # 12
            ],
            node_pgs={"rock_base": [1], "cimbra_base": [7]},
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
    """A complete analysis chain — Plain handler / RCM numberer /
    UmfPack matches the existing flat-staged test fixture.  In a real
    MP run users would substitute Transformation / ParallelPlain /
    Mumps; for emit-shape tests we only need a syntactically complete
    chain.  Phase SSI-2.C skips the partitioned auto-emit (each stage
    carries its own chain), so no MP-incompatibility warning fires
    here.
    """
    return {
        "test":        ops.test.NormDispIncr(tol=1e-4, max_iter=50),
        "algorithm":   ops.algorithm.Newton(),
        "integrator":  ops.integrator.LoadControl(dlam=0.1),
        "constraints": ops.constraints.Plain(),
        "numberer":    ops.numberer.RCM(),
        "system":      ops.system.UmfPack(),
        "analysis":    ops.analysis.Static(),
    }


def _setup_partitioned_staged_ops(fem: FEMStub) -> apeSees:
    """Build a 2-stage partitioned apeSees model with rock global +
    cimbra activated in stage 2 + initial_stress on cimbra."""
    ops = apeSees(cast("object", fem), default_orientation=None)  # type: ignore[arg-type]
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="cimbra", thickness=1.0, material=mat)
    ops.fix(pg="rock_base", dofs=(1, 1))

    with ops.stage(name="rock_only") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=2)

    rec = ops.initial_stress(
        name="cimbra_in_situ",
        pg="cimbra",
        sigma_xx=-1000.0, sigma_yy=-1000.0, sigma_zz=0.0,
        ramp_steps=3,
    )
    with ops.stage(name="install_cimbra") as s:
        s.add(rec)
        s.activate(pgs=["cimbra"])
        s.analysis(**_full_chain(ops))
        s.run(n_increments=3)

    return ops


def _bucket_calls_by_scope(
    rec: RecordingEmitter,
) -> dict[tuple[int, "int | None"], list[tuple[str, tuple, dict]]]:
    """Bucket recorded calls by ``(stage_idx, rank)``.

    ``stage_idx == -1`` is the pre-stage global zone (before any
    ``stage_open``).  ``rank is None`` is the cross-partition global
    zone (outside any ``partition_open`` block).  ``stage_open``,
    ``stage_close``, ``partition_open``, ``partition_close`` are
    structural markers and are not stored in any bucket.
    """
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
# 1. Gate lifted — the build no longer raises.
# ---------------------------------------------------------------------------


def test_partitioned_staged_build_does_not_raise() -> None:
    """Phase SSI-2.C lifts the prior gate; combining stages with
    partitions must build without error."""
    fem = _make_4quad_2pg_2part_fem()
    ops = _setup_partitioned_staged_ops(fem)
    bm = ops.build()
    # Should not raise NotImplementedError (the prior gate) or anything else.
    bm.emit(RecordingEmitter())


# ---------------------------------------------------------------------------
# 2. Per-rank pre-stage blocks carry the right node / element subset.
# ---------------------------------------------------------------------------


def test_rank0_global_block_has_rock_nodes_and_elements() -> None:
    """Rock is global → rank 0's pre-stage block emits all rock nodes
    + the two rock element calls.  Cimbra is stage-bound and does
    NOT emit here."""
    fem = _make_4quad_2pg_2part_fem()
    ops = _setup_partitioned_staged_ops(fem)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    buckets = _bucket_calls_by_scope(rec)

    rank0_pre = buckets.get((-1, 0), [])
    node_ids = sorted(int(a[0]) for n, a, _k in rank0_pre if n == "node")
    assert node_ids == [1, 2, 3, 4, 5, 6], (
        f"rank 0 pre-stage nodes mismatch: {node_ids}"
    )
    element_tags = [int(a[1]) for n, a, _k in rank0_pre if n == "element"]
    assert len(element_tags) == 2, (
        f"rank 0 pre-stage should have 2 rock element calls; got {element_tags}"
    )


def test_rank1_global_block_has_no_topology() -> None:
    """Cimbra is stage-bound → rank 1's pre-stage block has no
    nodes / elements."""
    fem = _make_4quad_2pg_2part_fem()
    ops = _setup_partitioned_staged_ops(fem)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    buckets = _bucket_calls_by_scope(rec)

    rank1_pre = buckets.get((-1, 1), [])
    node_calls = [c for c in rank1_pre if c[0] == "node"]
    elem_calls = [c for c in rank1_pre if c[0] == "element"]
    assert node_calls == [], (
        f"rank 1 pre-stage should emit no nodes; got {node_calls}"
    )
    assert elem_calls == [], (
        f"rank 1 pre-stage should emit no elements; got {elem_calls}"
    )


# ---------------------------------------------------------------------------
# 3. Stage 2 per-rank blocks carry the activated cimbra topology.
# ---------------------------------------------------------------------------


def test_stage2_rank1_block_has_cimbra_topology() -> None:
    """Stage 2 (index 1) activates cimbra → rank 1's stage block
    emits the 6 cimbra nodes + 2 cimbra element calls."""
    fem = _make_4quad_2pg_2part_fem()
    ops = _setup_partitioned_staged_ops(fem)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    buckets = _bucket_calls_by_scope(rec)

    stage1_rank1 = buckets.get((1, 1), [])
    node_ids = sorted(int(a[0]) for n, a, _k in stage1_rank1 if n == "node")
    assert node_ids == [7, 8, 9, 10, 11, 12]
    elem_tags = [int(a[1]) for n, a, _k in stage1_rank1 if n == "element"]
    assert len(elem_tags) == 2, (
        f"stage 2 rank 1 should emit 2 cimbra elements; got {elem_tags}"
    )


def test_stage2_rank0_block_has_no_cimbra_topology() -> None:
    """All cimbra elements are owned by rank 1; rank 0's stage 2
    topology pass emits nothing."""
    fem = _make_4quad_2pg_2part_fem()
    ops = _setup_partitioned_staged_ops(fem)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    buckets = _bucket_calls_by_scope(rec)

    stage1_rank0 = buckets.get((1, 0), [])
    node_calls = [c for c in stage1_rank0 if c[0] == "node"]
    elem_calls = [c for c in stage1_rank0 if c[0] == "element"]
    assert node_calls == [], (
        f"stage 2 rank 0 should emit no nodes; got {node_calls}"
    )
    assert elem_calls == [], (
        f"stage 2 rank 0 should emit no elements; got {elem_calls}"
    )


# ---------------------------------------------------------------------------
# 4. domainChange — global, after the activated cimbra elements.
# ---------------------------------------------------------------------------


def test_stage2_domain_change_is_global() -> None:
    """``domainChange`` lives in the global scope of stage 2 (rank
    ``None``), not inside a ``partition_open`` block — OpenSeesMP
    executes the line on every rank's local domain."""
    fem = _make_4quad_2pg_2part_fem()
    ops = _setup_partitioned_staged_ops(fem)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    buckets = _bucket_calls_by_scope(rec)

    stage1_global = buckets.get((1, None), [])
    domain_calls = [c for c in stage1_global if c[0] == "domain_change"]
    assert len(domain_calls) == 1, (
        f"stage 2 should emit exactly one global domainChange; "
        f"got {domain_calls}"
    )


def test_stage2_domain_change_after_cimbra_elements() -> None:
    """Verify the deck-text ordering: stage 2's ``domainChange`` line
    appears AFTER the last cimbra element line."""
    fem = _make_4quad_2pg_2part_fem()
    ops = _setup_partitioned_staged_ops(fem)
    emitter = TclEmitter()
    ops.build().emit(emitter)
    lines = emitter.lines()

    stage2_idx = next(
        i for i, ln in enumerate(lines)
        if ln.startswith("# === Stage: install_cimbra")
    )
    # Element lines inside ``partition_open`` are indented 4 spaces;
    # use ``.lstrip`` so the assertion matches regardless of indent.
    elem_indices = [
        i for i, ln in enumerate(lines)
        if i > stage2_idx and ln.lstrip().startswith("element quad ")
    ]
    assert elem_indices, "no cimbra element lines found inside stage 2"
    last_elem_idx = max(elem_indices)
    domain_idx = next(
        i for i, ln in enumerate(lines)
        if i > stage2_idx and ln == "domainChange"
    )
    assert domain_idx > last_elem_idx, (
        f"domainChange (line {domain_idx}) must follow the last cimbra "
        f"element (line {last_elem_idx})"
    )


# ---------------------------------------------------------------------------
# 5. addToParameter — inside partition_open(1) only, never global.
# ---------------------------------------------------------------------------


def test_addToParameter_inside_rank1_partition_open() -> None:
    """Stage 2's ``initial_stress`` covers cimbra (rank 1 only).  The
    per-rank ``addToParameter`` pass emits 2 elements × 3 components
    = 6 calls inside ``partition_open(1)`` and zero inside
    ``partition_open(0)``."""
    fem = _make_4quad_2pg_2part_fem()
    ops = _setup_partitioned_staged_ops(fem)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    buckets = _bucket_calls_by_scope(rec)

    stage1_rank1 = buckets.get((1, 1), [])
    addp_rank1 = [c for c in stage1_rank1 if c[0] == "addToParameter"]
    assert len(addp_rank1) == 6, (
        f"stage 2 rank 1 should emit 6 addToParameter calls "
        f"(2 elems x 3 components); got {len(addp_rank1)}"
    )

    stage1_rank0 = buckets.get((1, 0), [])
    addp_rank0 = [c for c in stage1_rank0 if c[0] == "addToParameter"]
    assert addp_rank0 == [], (
        f"stage 2 rank 0 should emit no addToParameter calls "
        f"(cimbra owned by rank 1); got {addp_rank0}"
    )


def test_no_addToParameter_in_global_scope() -> None:
    """``addToParameter`` calls always live inside a partition_open
    block — never in the global (rank=None) zone."""
    fem = _make_4quad_2pg_2part_fem()
    ops = _setup_partitioned_staged_ops(fem)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    buckets = _bucket_calls_by_scope(rec)

    for (stage_idx, rank), calls in buckets.items():
        if rank is None:
            stray = [c for c in calls if c[0] == "addToParameter"]
            assert stray == [], (
                f"unexpected addToParameter in global scope "
                f"(stage={stage_idx}, rank=None): {stray}"
            )
