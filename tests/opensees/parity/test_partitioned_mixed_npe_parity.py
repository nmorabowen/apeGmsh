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
# 2b. A rank with ZERO primary-owned nodes (M7 guard, refuter lens 1).
# ---------------------------------------------------------------------------


def _make_zero_primary_rank_fem() -> FEMStub:
    """Rank 1 owns ONLY shared nodes ({2, 3} ⊂ rank 0's {1, 2, 3}).

    ``primary_owner_map`` assigns each node to its LOWEST owning rank
    (documented tie-break), so every rank-1 node is primary on rank 0
    and ``rank_primary_nodes[1]`` must exist as an EMPTY set — a
    dropped seed is a KeyError in the per-rank pattern pass.  No prior
    fixture had such a rank.
    """
    fem = FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3],
            coords=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
            node_pgs={"anchor": [1], "tip": [3]},
        ),
        elements=_ElementsStub(
            elem_pgs={
                "bars": _ElementGroupView(
                    ids=(1, 2),
                    connectivity=((1, 2), (2, 3)),
                ),
            },
        ),
    )
    fem.set_partitions([
        (0, [1, 2, 3], [1]),
        (1, [2, 3], [2]),
    ])
    return fem


def test_zero_primary_rank_emits_loads_once_and_does_not_crash() -> None:
    fem = _make_zero_primary_rank_fem()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=3)
    ux = ops.uniaxialMaterial.ElasticMaterial(E=2e11)
    ops.element.Truss(pg="bars", A=1e-4, material=ux)
    ops.fix(pg="anchor", dofs=(1, 1, 1))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(node=2, forces=(1.0, 0.0, 0.0))
        p.load(node=3, forces=(0.0, 0.0, -1.0))

    rec = RecordingEmitter()
    ops.build().emit(rec)
    buckets = _bucket_calls_by_scope(rec)

    loads = [
        (scope, c[1][0])
        for scope, calls in buckets.items()
        for c in calls
        if c[0] == "load"
    ]
    # Additive invariant: each load emits exactly ONCE, on the primary
    # rank (rank 0 — the lowest owner); rank 1's bracket carries none.
    assert sorted(loads) == [((-1, 0), 2), ((-1, 0), 3)]



# ---------------------------------------------------------------------------
# 2b. Stage 2: s.update_material_stage replicates on EVERY rank (SSI-2.E).
# ---------------------------------------------------------------------------


def _material_stage_ops() -> apeSees:
    """Same 3-rank mesh, but the bricks carry a SANISAND material that
    a second stage flips to elastoplastic."""
    ops = apeSees(cast("object", _make_3rank_mixed_npe_fem()))
    ops.model(ndm=3, ndf=3)
    sand = ops.nDMaterial.ManzariDafalias(
        name="sand", G0=125.0, nu=0.05, e_init=0.8, Mc=1.25, c=0.712,
        lambda_c=0.019, e0=0.934, ksi=0.7, P_atm=101.3, m=0.01, h0=7.05,
        Ch=0.968, nb=1.1, A0=0.704, nd=3.5, z_max=4.0, cz=600.0, rho=1.6,
    )
    ux = ops.uniaxialMaterial.ElasticMaterial(E=2e11)
    ops.element.stdBrick(pg="solids", material=sand)
    ops.element.Truss(pg="bars", A=1e-4, material=ux)
    ops.fix(pg="base", dofs=(1, 1, 1))

    with ops.stage(name="gravity") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="push") as s:
        s.update_material_stage(materials=[sand], stage=1)
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    return ops


def test_stage_material_flip_replicates_in_every_rank_bracket() -> None:
    """``ManzariDafalias::mElastFlag`` is a per-process static, and each
    rank is its own process — so the flip must land inside EVERY rank's
    stage bracket, not just the owner's (unlike ``remove_element``)."""
    ops = _material_stage_ops()
    rec = RecordingEmitter()
    ops.build().emit(rec)

    tag = ops.tag_for(ops._names["sand"])
    rank = None
    per_rank: "dict[int, list[tuple[int, int]]]" = {}
    seen_push = False
    for name, args, _kw in rec.calls:
        if name == "stage_open":
            seen_push = args[0] == "push"
        elif name == "partition_open":
            rank = int(args[0])
        elif name == "partition_close":
            rank = None
        elif name == "update_material_stage" and seen_push:
            assert rank is not None, "flip emitted outside a rank bracket"
            per_rank.setdefault(rank, []).append((args[0], args[1]))

    assert per_rank == {0: [(tag, 1)], 1: [(tag, 1)], 2: [(tag, 1)]}

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


# ---------------------------------------------------------------------------
# 4. Committed golden deck — full-text ordering pin (refuter lens 1).
# ---------------------------------------------------------------------------
#
# ``deck_lines`` in the bench gate is a COUNT and the routing tests above
# assert SETS — neither proves line ORDER at deck level.  This golden
# pins every byte of the emitted deck for a deterministic stub whose
# node ids are UNSORTED and non-contiguous (the real partitioned-broker
# shape: ``fem.nodes.ids`` is entity-grouped), so the D3 argsort
# permutation is non-trivial — an id-sorted stub would let a dropped
# permutation pass.  A stub (not a live Gmsh mesh) keeps the digest out
# of gmsh/METIS version churn; the golden is byte-exact on every
# platform (``.gitattributes`` marks it ``-text``).
#
# Re-baselining: a DELIBERATE deck-shape change regenerates the golden
# (run ``_emit_unsorted_deck`` and overwrite the file) and commits it in
# the same PR, so the diff shows the deck change being accepted — same
# policy as ``emit_gate_baseline.json``.

GOLDEN_DECK = Path(__file__).with_name("partitioned_mixed_npe.golden.tcl")

# Storage-order node ids (index i holds the id of the original fixture's
# node i+1) — unsorted, non-contiguous, gap-heavy.
_UNSORTED_IDS = (
    104, 7, 9001, 55, 2, 300, 18, 73,
    210, 9, 64, 5000, 31, 88, 402, 11,
)


def _remap(*orig: int) -> tuple[int, ...]:
    return tuple(_UNSORTED_IDS[o - 1] for o in orig)


def _make_unsorted_ids_fem() -> FEMStub:
    """The 3-rank mixed-npe fixture with unsorted broker node ids."""
    fem = FEMStub(
        nodes=_NodesStub(
            ids=list(_UNSORTED_IDS),
            coords=[
                (0.0, 0.0, 0.0),  # 104
                (1.0, 0.0, 0.0),  # 7
                (1.0, 1.0, 0.0),  # 9001
                (0.0, 1.0, 0.0),  # 55
                (0.0, 0.0, 1.0),  # 2
                (1.0, 0.0, 1.0),  # 300
                (1.0, 1.0, 1.0),  # 18
                (0.0, 1.0, 1.0),  # 73
                (0.0, 0.0, 2.0),  # 210
                (1.0, 0.0, 2.0),  # 9
                (1.0, 1.0, 2.0),  # 64
                (0.0, 1.0, 2.0),  # 5000
                (0.0, 0.0, 3.0),  # 31
                (1.0, 0.0, 3.0),  # 88
                (2.0, 0.0, 2.0),  # 402
                (2.0, 1.0, 2.0),  # 11
            ],
            node_pgs={"base": list(_remap(1, 2, 3, 4))},
        ),
        elements=_ElementsStub(
            elem_pgs={
                "solids": _ElementGroupView(
                    ids=(1, 2),
                    connectivity=(
                        _remap(1, 2, 3, 4, 5, 6, 7, 8),
                        _remap(5, 6, 7, 8, 9, 10, 11, 12),
                    ),
                ),
                "bars": _ElementGroupView(
                    ids=(3, 4, 5),
                    connectivity=(
                        _remap(9, 13), _remap(10, 14), _remap(13, 14),
                    ),
                ),
                "late": _ElementGroupView(
                    ids=(6, 7),
                    connectivity=(_remap(10, 15), _remap(12, 16)),
                ),
            },
        ),
    )
    fem.set_partitions([
        (0, list(_remap(1, 2, 3, 4, 5, 6, 7, 8)), [1]),
        (1, list(_remap(5, 6, 7, 8, 9, 10, 11, 12, 15, 16)), [2, 6, 7]),
        (2, list(_remap(9, 10, 13, 14)), [3, 4, 5]),
    ])
    return fem


def _emit_unsorted_deck(out_path: Path) -> None:
    _setup_ops(_make_unsorted_ids_fem()).tcl(str(out_path))


def test_unsorted_ids_deck_matches_committed_golden(tmp_path: Path) -> None:
    deck = tmp_path / "deck.tcl"
    _emit_unsorted_deck(deck)
    assert GOLDEN_DECK.exists(), (
        f"missing committed golden {GOLDEN_DECK.name}; regenerate via "
        "_emit_unsorted_deck (see the re-baselining note above)"
    )
    # EOL-normalised: ``open(..., 'w')`` translates to the platform EOL
    # (CRLF on Windows), which is the OS's byte, not the emitter's
    # ordering.  Everything else — every line, in order — is exact.
    got = deck.read_bytes().replace(b"\r\n", b"\n")
    want = GOLDEN_DECK.read_bytes().replace(b"\r\n", b"\n")
    if got != want:
        got_lines = got.decode("utf-8").splitlines()
        want_lines = want.decode("utf-8").splitlines()
        first = next(
            (i for i, (g, w) in enumerate(zip(got_lines, want_lines))
             if g != w),
            min(len(got_lines), len(want_lines)),
        )
        raise AssertionError(
            f"emitted deck diverges from {GOLDEN_DECK.name} at line "
            f"{first + 1}: emitted {got_lines[first] if first < len(got_lines) else '<EOF>'!r} "
            f"vs golden {want_lines[first] if first < len(want_lines) else '<EOF>'!r} "
            f"({len(got_lines)} vs {len(want_lines)} lines). If the deck "
            "change is deliberate, regenerate the golden in this PR."
        )
