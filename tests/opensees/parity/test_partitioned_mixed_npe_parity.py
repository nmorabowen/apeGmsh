"""ADR 0100 P3 — 3-rank, mixed-npe, staged partitioned parity fixture.

Every partitioned parity fixture before this one is 2-rank and
single-npe (2-4 elements), so the P3 surgery targets — the columnar
node-index lookup (D3), the columnar rank-membership sets (D2), the
lazy per-rank element plan (D4), and the columnar reverse tag map
(R8) — were only ever exercised at np=2 with one element width.  This
fixture pins the behaviours that surgery must preserve, on a shape it
cannot fake:

* **3 ranks with shared boundary nodes** — membership tests and the
  idempotent node replication (a shared face emits its ``node`` lines
  on every owning rank) exercise D2/D3 beyond the 2-rank diagonal.
* **Mixed npe across element types** — an 8-node ``stdBrick`` spec and
  2-node ``Truss`` specs in one plan, so D4's per-rank buckets carry
  different row widths side by side.
* **Staged deck with stage-bound activation** — D4's staged bucket
  consumer and D2's ``rank_owned ∩ stage_owned`` intersection run
  after ``partition_close`` (the reason drop-at-close is a dead
  variant, per the ADR).
* **``s.remove_element`` on a partitioned stage** — the reverse tag
  map's ONLY consumer routes the removal to the owning rank's stage
  bracket.  No other partitioned test covers it.

Geometry (ndm=3, ndf=3): a two-brick stack (ranks 0-1, shared face
5-8 / 9-12) with a truss lattice hanging off the top (rank 2), plus a
stage-activated truss pair on rank 1.
"""
from __future__ import annotations

from pathlib import Path
from typing import cast

from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter.recording import RecordingEmitter

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)
from tests.opensees.integration.test_emit_partitioned_staged import (
    _bucket_calls_by_scope,
    _full_chain,
)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def _make_3rank_mixed_npe_fem() -> FEMStub:
    """Two stacked bricks + trusses across 3 ranks with shared faces.

    * Brick 1 (eid 1, nodes 1-8)      -> rank 0
    * Brick 2 (eid 2, nodes 5-12)     -> rank 1 (face 5-8 shared with 0)
    * Trusses (eids 3-5, "bars")      -> rank 2 (nodes 9, 10 shared
      with rank 1; 13, 14 exclusive)
    * Trusses (eids 6-7, "late")      -> rank 1, stage-activated
      (nodes 15, 16 exclusive to the stage; 10, 12 global)
    """
    fem = FEMStub(
        nodes=_NodesStub(
            ids=list(range(1, 17)),
            coords=[
                (0.0, 0.0, 0.0),  # 1
                (1.0, 0.0, 0.0),  # 2
                (1.0, 1.0, 0.0),  # 3
                (0.0, 1.0, 0.0),  # 4
                (0.0, 0.0, 1.0),  # 5
                (1.0, 0.0, 1.0),  # 6
                (1.0, 1.0, 1.0),  # 7
                (0.0, 1.0, 1.0),  # 8
                (0.0, 0.0, 2.0),  # 9
                (1.0, 0.0, 2.0),  # 10
                (1.0, 1.0, 2.0),  # 11
                (0.0, 1.0, 2.0),  # 12
                (0.0, 0.0, 3.0),  # 13
                (1.0, 0.0, 3.0),  # 14
                (2.0, 0.0, 2.0),  # 15
                (2.0, 1.0, 2.0),  # 16
            ],
            node_pgs={"base": [1, 2, 3, 4]},
        ),
        elements=_ElementsStub(
            elem_pgs={
                "solids": _ElementGroupView(
                    ids=(1, 2),
                    connectivity=(
                        (1, 2, 3, 4, 5, 6, 7, 8),
                        (5, 6, 7, 8, 9, 10, 11, 12),
                    ),
                ),
                "bars": _ElementGroupView(
                    ids=(3, 4, 5),
                    connectivity=((9, 13), (10, 14), (13, 14)),
                ),
                "late": _ElementGroupView(
                    ids=(6, 7),
                    connectivity=((10, 15), (12, 16)),
                ),
            },
        ),
    )
    fem.set_partitions([
        (0, [1, 2, 3, 4, 5, 6, 7, 8], [1]),
        (1, [5, 6, 7, 8, 9, 10, 11, 12, 15, 16], [2, 6, 7]),
        (2, [9, 10, 13, 14], [3, 4, 5]),
    ])
    return fem


def _setup_ops(fem: FEMStub) -> apeSees:
    """2-stage model: everything global except pg "late" (stage 2),
    which also removes truss eid 5 (rank 2)."""
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=3)
    nd = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ux = ops.uniaxialMaterial.ElasticMaterial(E=2e11)
    ops.element.stdBrick(pg="solids", material=nd)
    ops.element.Truss(pg="bars", A=1e-4, material=ux)
    ops.element.Truss(pg="late", A=1e-4, material=ux)
    ops.fix(pg="base", dofs=(1, 1, 1))

    with ops.stage(name="base_state") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=2)

    with ops.stage(name="swap_bars") as s:
        s.activate(pgs=["late"])
        s.remove_element(elements=[5])
        s.analysis(**_full_chain(ops))
        s.run(n_increments=2)

    return ops


def _factory() -> apeSees:
    return _setup_ops(_make_3rank_mixed_npe_fem())


def _record() -> RecordingEmitter:
    rec = RecordingEmitter()
    _factory().build().emit(rec)
    return rec


def _node_tags(bucket: "list[tuple[str, tuple, dict]]") -> set[int]:
    return {int(c[1][0]) for c in bucket if c[0] == "node"}


def _element_calls(
    bucket: "list[tuple[str, tuple, dict]]",
) -> "list[tuple[str, tuple]]":
    return [(c[1][0], c[1][1:]) for c in bucket if c[0] == "element"]


# ---------------------------------------------------------------------------
# 1. Per-rank routing of the mixed-npe plan (D3 / D4).
# ---------------------------------------------------------------------------


def test_global_pass_routes_mixed_npe_elements_to_owner_ranks() -> None:
    buckets = _bucket_calls_by_scope(_record())

    r0 = _element_calls(buckets[(-1, 0)])
    assert len(r0) == 1 and r0[0][0] == "stdBrick"
    assert r0[0][1][1:9] == (1, 2, 3, 4, 5, 6, 7, 8)

    r1 = _element_calls(buckets[(-1, 1)])
    assert len(r1) == 1 and r1[0][0] == "stdBrick"
    assert r1[0][1][1:9] == (5, 6, 7, 8, 9, 10, 11, 12)

    r2 = _element_calls(buckets[(-1, 2)])
    assert [kind for kind, _ in r2] == ["Truss", "Truss", "Truss"]
    assert [args[1:3] for _, args in r2] == [(9, 13), (10, 14), (13, 14)]


def test_shared_boundary_nodes_replicate_on_every_owning_rank() -> None:
    buckets = _bucket_calls_by_scope(_record())
    n0 = _node_tags(buckets[(-1, 0)])
    n1 = _node_tags(buckets[(-1, 1)])
    n2 = _node_tags(buckets[(-1, 2)])

    assert n0 == {1, 2, 3, 4, 5, 6, 7, 8}
    # 15/16 are stage-owned: absent from rank 1's GLOBAL block.
    assert n1 == {5, 6, 7, 8, 9, 10, 11, 12}
    assert n2 == {9, 10, 13, 14}
    # Shared faces replicate (idempotent node lines).
    assert {5, 6, 7, 8} <= n0 & n1
    assert {9, 10} <= n1 & n2


def test_fixes_land_on_owner_rank_only() -> None:
    buckets = _bucket_calls_by_scope(_record())
    assert [c[1][0] for c in buckets[(-1, 0)] if c[0] == "fix"] == [1, 2, 3, 4]
    for rank in (1, 2):
        assert not [c for c in buckets[(-1, rank)] if c[0] == "fix"]


# ---------------------------------------------------------------------------
# 2. Stage 2: activation topology + remove_element routing (D2 / R8).
# ---------------------------------------------------------------------------


def test_stage2_activates_late_bars_on_rank1_only() -> None:
    buckets = _bucket_calls_by_scope(_record())

    s1 = buckets[(1, 1)]
    assert _node_tags(s1) == {15, 16}
    late = _element_calls(s1)
    assert [kind for kind, _ in late] == ["Truss", "Truss"]
    assert [args[1:3] for _, args in late] == [(10, 15), (12, 16)]

    # Rank 0 has no stage-2 content at all: its bracket never opens.
    assert (1, 0) not in buckets


def test_stage2_remove_element_routes_to_owning_rank_bracket() -> None:
    """R8's sole consumer: fem eid 5 -> ops tag via the reverse map ->
    ``remove_element`` inside rank 2's stage bracket, nowhere else."""
    buckets = _bucket_calls_by_scope(_record())

    # Self-consistent tag oracle: the tag the global pass gave truss
    # eid 5 is the tag whose element call carries nodes (13, 14).
    tag_e5 = next(
        args[0] for kind, args in _element_calls(buckets[(-1, 2)])
        if args[1:3] == (13, 14)
    )

    removes = [
        (scope, c[1][0])
        for scope, calls in buckets.items()
        for c in calls
        if c[0] == "remove_element"
    ]
    assert removes == [((1, 2), tag_e5)]


# ---------------------------------------------------------------------------
# 3. Stream-vs-list byte identity on this fixture (the P3 gate shape).
# ---------------------------------------------------------------------------


def test_stream_monolithic_byte_identical_on_3rank_mixed_npe(
    tmp_path: Path,
) -> None:
    list_path = tmp_path / "list.tcl"
    stream_path = tmp_path / "stream.tcl"
    _factory().tcl(str(list_path))
    _factory().tcl(str(stream_path), stream=True)
    assert stream_path.read_bytes() == list_path.read_bytes()


def test_stream_per_rank_byte_identical_on_3rank_mixed_npe(
    tmp_path: Path,
) -> None:
    list_dir = tmp_path / "list"
    stream_dir = tmp_path / "stream"
    list_dir.mkdir()
    stream_dir.mkdir()
    _factory().tcl(str(list_dir / "main.tcl"), per_rank=True)
    _factory().tcl(str(stream_dir / "main.tcl"), per_rank=True, stream=True)

    assert (stream_dir / "main.tcl").read_bytes() == (
        (list_dir / "main.tcl").read_bytes()
    )
    list_frags = sorted(
        p.name for p in (list_dir / "ranks").glob("rank*.tcl")
    )
    stream_frags = sorted(
        p.name for p in (stream_dir / "ranks").glob("rank*.tcl")
    )
    assert list_frags, "oracle wrote no fragments — fixture degraded"
    assert stream_frags == list_frags
    for name in list_frags:
        assert (
            (stream_dir / "ranks" / name).read_bytes()
            == (list_dir / "ranks" / name).read_bytes()
        ), f"fragment {name} differs between stream and list mode"
