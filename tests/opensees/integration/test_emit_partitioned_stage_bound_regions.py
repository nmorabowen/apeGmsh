"""Phase SSI-2.D PR-C — stage-bound `s.region` under MP.

Locks the per-stage region tag cache:

1. All ranks contributing to a stage-bound region emit the SAME tag
   (one tag per (stage, region_name), shared across rank fan-out).
2. Two stages with regions named identically would have been refused
   by V3 (PR-A); names within a single stage merge into one region.
3. Empty-rank-intersection skip (INV-4) — a rank with no owned region
   members emits no `region` line for that name.
4. Stage-bound region tags are disjoint from global region tags
   (TagAllocator's monotonic counter — no collision).
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
                "rock_base":  [1],
                "cimbra_top": [9, 12],
                # Cross-rank region: members on BOTH ranks (rock node 2,
                # cimbra node 8).  Used in the cross-rank tag-cache test.
                "mixed":      [2, 8],
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


def _bucket_calls_by_scope(
    rec: RecordingEmitter,
) -> dict[tuple[int, "int | None"], list[tuple[str, tuple, dict]]]:
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


# ===========================================================================
# 1. Cross-rank region — both contributing ranks emit the SAME tag
# ===========================================================================


def test_cross_rank_stage_region_shares_tag_across_ranks() -> None:
    """A stage-bound region whose members span both ranks must emit
    one `region` line on each rank, both lines carrying the SAME tag
    (per-stage region_tag_cache works under MP).
    """
    fem = _make_4quad_2pg_2part_fem()
    ops = apeSees(cast("object", fem), default_orientation=None)  # type: ignore[arg-type]
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="cimbra", thickness=1.0, material=mat)
    ops.fix(pg="rock_base", dofs=(1, 1))

    with ops.stage(name="probe") as s:
        s.activate(pgs=["cimbra"])
        s.region(name="mixed_r", pg="mixed")
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)

    rec = RecordingEmitter()
    ops.build().emit(rec)
    buckets = _bucket_calls_by_scope(rec)

    rank0_region_tags = [
        int(c[1][0]) for c in buckets.get((0, 0), []) if c[0] == "region"
    ]
    rank1_region_tags = [
        int(c[1][0]) for c in buckets.get((0, 1), []) if c[0] == "region"
    ]
    assert len(rank0_region_tags) == 1, (
        f"rank 0 should emit one region line; got {rank0_region_tags}"
    )
    assert len(rank1_region_tags) == 1, (
        f"rank 1 should emit one region line; got {rank1_region_tags}"
    )
    assert rank0_region_tags[0] == rank1_region_tags[0], (
        f"per-stage region_tag_cache broken: rank 0 tag "
        f"{rank0_region_tags[0]} != rank 1 tag {rank1_region_tags[0]}"
    )


# ===========================================================================
# 2. Single-rank region — only the owning rank emits the region line
# ===========================================================================


def test_single_rank_stage_region_emits_only_on_owning_rank() -> None:
    """A stage region whose members are all on rank 1 emits a
    `region` line only inside `partition_open(1)`; rank 0 has no
    region line (INV-4 empty-intersection skip)."""
    fem = _make_4quad_2pg_2part_fem()
    ops = apeSees(cast("object", fem), default_orientation=None)  # type: ignore[arg-type]
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="cimbra", thickness=1.0, material=mat)
    ops.fix(pg="rock_base", dofs=(1, 1))

    with ops.stage(name="probe") as s:
        s.activate(pgs=["cimbra"])
        s.region(name="cimbra_top_r", pg="cimbra_top")  # nodes 9, 12
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)

    rec = RecordingEmitter()
    ops.build().emit(rec)
    buckets = _bucket_calls_by_scope(rec)

    rank0_regions = [
        c for c in buckets.get((0, 0), []) if c[0] == "region"
    ]
    rank1_regions = [
        c for c in buckets.get((0, 1), []) if c[0] == "region"
    ]
    assert rank0_regions == [], (
        f"rank 0 owns no cimbra_top members; should emit zero region "
        f"lines, got {rank0_regions}"
    )
    assert len(rank1_regions) == 1, (
        f"rank 1 should emit one region line; got {rank1_regions}"
    )


# ===========================================================================
# 3. Per-stage tag cache scoping — two stages, distinct tags
# ===========================================================================


def test_per_stage_region_tag_cache_disjoint_across_stages() -> None:
    """Two stages each declare regions; the per-stage tag cache must
    NOT leak across stages — every region gets its own tag.  V3
    refuses same NAME across scopes, so each stage's region name
    must be unique."""
    fem = _make_4quad_2pg_2part_fem()
    ops = apeSees(cast("object", fem), default_orientation=None)  # type: ignore[arg-type]
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="cimbra", thickness=1.0, material=mat)
    ops.fix(pg="rock_base", dofs=(1, 1))

    with ops.stage(name="A") as s:
        s.region(name="region_A", pg="rock_base")
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="B") as s:
        s.activate(pgs=["cimbra"])
        s.region(name="region_B", pg="cimbra_top")
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)

    rec = RecordingEmitter()
    ops.build().emit(rec)
    buckets = _bucket_calls_by_scope(rec)

    # Stage A has region_A on rank 0 only (rock_base = node 1).
    stageA_tags = [
        int(c[1][0]) for (sidx, _rank), calls in buckets.items()
        if sidx == 0 for c in calls if c[0] == "region"
    ]
    # Stage B has region_B on rank 1 only (cimbra_top = nodes 9, 12).
    stageB_tags = [
        int(c[1][0]) for (sidx, _rank), calls in buckets.items()
        if sidx == 1 for c in calls if c[0] == "region"
    ]
    assert len(stageA_tags) == 1 and len(stageB_tags) == 1
    assert stageA_tags[0] != stageB_tags[0], (
        f"per-stage tag cache leaked: stage A tag={stageA_tags[0]} "
        f"== stage B tag={stageB_tags[0]}"
    )


# ===========================================================================
# 4. Global + stage regions — disjoint tags via the monotonic allocator
# ===========================================================================


def test_global_and_stage_region_tags_disjoint() -> None:
    """A global region + a stage-bound region must get disjoint tags
    even though their members are completely independent — V3 refuses
    same NAME across scopes; tag disjointness comes from the
    monotonic TagAllocator."""
    fem = _make_4quad_2pg_2part_fem()
    ops = apeSees(cast("object", fem), default_orientation=None)  # type: ignore[arg-type]
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="cimbra", thickness=1.0, material=mat)
    ops.fix(pg="rock_base", dofs=(1, 1))
    ops.region(name="global_r", pg="rock_base")  # rank 0 only

    with ops.stage(name="A") as s:
        s.activate(pgs=["cimbra"])
        s.region(name="stage_r", pg="cimbra_top")  # rank 1 only
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)

    rec = RecordingEmitter()
    ops.build().emit(rec)
    all_region_tags: set[int] = set()
    for name, args, _kwargs in rec.calls:
        if name == "region":
            all_region_tags.add(int(args[0]))
    # Two regions emitted in this fixture — both must have distinct
    # tags, no collision.
    assert len(all_region_tags) == 2, (
        f"expected two distinct region tags (global + stage-bound); "
        f"got {sorted(all_region_tags)}"
    )
