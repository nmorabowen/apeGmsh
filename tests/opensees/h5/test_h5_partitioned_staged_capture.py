"""Partitioned staged H5 capture (ADR 0055 Phase 5 / P5.1, schema 2.19.0).

The partitioned staged emit brackets each stage per rank and
REPLICATES stage-bound emission across owning ranks (ADR 0027
INV-1/INV-4 inside ``_emit_stages_partitioned``): a stage fix / HOLD
on a cross-rank shared node emits in both rank brackets; the stage's
patterns (including the ADR 0052 HOLD pattern) re-open the same tag
once per owning rank with rank-filtered lines; stage regions emit the
same tag once per contributing rank with the rank-intersection of
members; stage MP constraints replicate with foreign ghost-node
declarations preceding them.

P5.1 makes the H5 capture rank-agnostic — the locked contract:

1. ``ops.h5`` on a partitioned staged build WRITES (the last staged
   guard is lifted).
2. The stage zone carries the flat logical program ONCE: stage bucket
   content (owned topology, fixes, HOLD lines, pattern lines, region
   member unions, chain) is CONTENT-equal to the same model captured
   unpartitioned.  Capture order is rank-major — content equality,
   not byte equality, is the contract (the unpartitioned build's emit
   order differs).
3. ``/opensees/partitions`` carries exactly ONE ``partition_NN``
   group per rank (stage re-brackets resume the rank's accumulator),
   now including stage-owned topology; the ``partition_ids`` column
   stamps stage-owned elements with their real rank.
4. ``from_h5 → to_h5`` of a partitioned staged archive is
   ``model_hash``-stable; two fresh builds of the same model hash
   identically.
5. Foreign ghost-node declarations (stage MP, INV-2) never enter the
   stage bucket's ``owned_node_ids`` (unit level, via the
   ``set_stage_owned_node_tags`` side-channel).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from apeGmsh.mesh.FEMData import FEMData
from apeGmsh.opensees import OpenSeesModel
from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees._internal.tag_resolution import (
    set_stage_owned_node_tags,
)
from apeGmsh.opensees.emitter.h5 import H5Emitter

from tests.opensees.h5.test_h5_partitions_roundtrip import (
    build_partitioned_two_quad_fem,
)
from tests.opensees.h5.test_h5_stages_reader import build_two_quad_fem
from tests.opensees.h5.test_h5_stages_writer import _chain


_MP_AUTO_EMIT_FILTERS = (
    "ignore:MP constraints are present in the model:UserWarning",
    "ignore:len.fem.partitions. > 1 with no user-declared numberer:UserWarning",
    "ignore:len.fem.partitions. > 1 with no user-declared system:UserWarning",
)
pytestmark = [pytest.mark.filterwarnings(f) for f in _MP_AUTO_EMIT_FILTERS]


#: Stage-owned node 5 lives on BOTH ranks — the cross-rank stage
#: boundary the dedupe paths need.  (The default split has no shared
#: stage node: stage topology {5, 6} would be rank-1-only.)
_OVERLAPPING_SPLIT = {
    1: {"node_ids": [1, 2, 3, 4, 5], "element_ids": [1]},
    2: {"node_ids": [3, 4, 5, 6], "element_ids": [2]},
}


def _staged_bridge(fem: FEMData) -> apeSees:
    """Two-stage program exercising every P5.1 merge path.

    Stage ``construction``: activates ``Fill`` (element 2, stage
    nodes 5+6 — node 5 cross-rank), stage fix + HOLD + region on
    ``FillTop`` (per-rank replication / fragment paths).  Stage
    ``loading``: a stage pattern loading ``Fill`` (nodes 3-6 — spans
    both ranks, so the pattern re-opens per rank).
    """
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="Fill", thickness=1.0, material=mat)
    ops.fix(pg="Base", dofs=(1, 1))

    with ops.stage(name="construction") as s:
        s.activate(pgs=["Fill"])
        s.fix(pg="FillTop", dofs=(1, 0))
        s.support(pg="FillTop", dofs=(0, 1))
        s.region(name="fill_top_r", pg="FillTop")
        s.analysis(**_chain(ops))
        s.run(n_increments=5)
    with ops.stage(name="loading") as s:
        ts = ops.timeSeries.Linear()
        with s.pattern(series=ts) as p:
            p.load(pg="Fill", forces=(10.0, 0.0))
        s.analysis(**_chain(ops))
        s.run(n_increments=3, dt=0.01)
    return ops


def _partitioned_bridge() -> apeSees:
    return _staged_bridge(
        build_partitioned_two_quad_fem(partitions=_OVERLAPPING_SPLIT),
    )


def _flat_bridge() -> apeSees:
    return _staged_bridge(build_two_quad_fem())


def _model_hash_of(path: Path) -> str:
    with h5py.File(str(path), "r") as f:
        return str(f["meta"]["lineage"].attrs["model_hash"])


# ---------------------------------------------------------------------------
# 1+3. Write succeeds; partitions zone shape
# ---------------------------------------------------------------------------


def test_partitioned_staged_writes_with_one_block_per_rank(
    tmp_path: Path,
) -> None:
    out = tmp_path / "ps.h5"
    _partitioned_bridge().h5(str(out))

    with h5py.File(str(out), "r") as f:
        assert "/opensees/stages" in f
        parts = f["/opensees/partitions"]
        assert int(parts.attrs["n_partitions"]) == 2, (
            "stage re-brackets must RESUME the rank accumulator — one "
            f"partition group per rank, got {list(parts.keys())!r}"
        )
        by_rank = {
            int(parts[k].attrs["rank"]): parts[k] for k in parts
        }
        assert sorted(by_rank) == [0, 1]
        # Map emitted OpenSees tags back to FEM eids — the partition
        # blocks store ops tags, the fixture reasons in FEM ids.
        meta = f["/opensees/element_meta"]
        fem_of_tag = {
            int(i): int(e)
            for g in meta
            for i, e in zip(meta[g]["ids"][:], meta[g]["fem_eids"][:])
        }
        # Stage-owned element 2 (Fill) accumulates under its owning
        # rank's block (rank 1), alongside the global element 1.
        assert [
            fem_of_tag[int(t)] for t in by_rank[1]["element_ids"][:]
        ] == [2]
        assert [
            fem_of_tag[int(t)] for t in by_rank[0]["element_ids"][:]
        ] == [1]
        # Stage-owned node 6 lands in rank 1's node list; the shared
        # stage node 5 appears in BOTH (deduped within each block).
        assert 6 in set(by_rank[1]["node_ids"][:])
        assert 5 in set(by_rank[0]["node_ids"][:])
        assert 5 in set(by_rank[1]["node_ids"][:])
        # partition_ids column: BOTH elements carry real ranks (the
        # stage-owned element stamps its rank too).
        ranks = {
            int(e): int(r)
            for g in meta
            for e, r in zip(
                meta[g]["fem_eids"][:], meta[g]["partition_ids"][:],
            )
        }
        assert ranks == {1: 0, 2: 1}, (
            f"stage-owned element must stamp its rank, got {ranks!r}"
        )


# ---------------------------------------------------------------------------
# 2. Stage-zone content equality vs the unpartitioned build
# ---------------------------------------------------------------------------


def test_stage_zone_content_equals_unpartitioned(tmp_path: Path) -> None:
    p_part = tmp_path / "part.h5"
    p_flat = tmp_path / "flat.h5"
    _partitioned_bridge().h5(str(p_part))
    _flat_bridge().h5(str(p_flat))

    st_p = OpenSeesModel.from_h5(str(p_part)).stages()
    st_f = OpenSeesModel.from_h5(str(p_flat)).stages()
    assert [s.name for s in st_p] == [s.name for s in st_f]

    for sp, sf in zip(st_p, st_f):
        # Owned topology — content equality (capture order is
        # rank-major on the partitioned side).
        assert set(sp.owned_node_ids) == set(sf.owned_node_ids), sp.name
        assert set(sp.owned_element_ids) == set(sf.owned_element_ids)
        # Stage-bound fixes: the shared-node fix captured once.
        assert sorted((f.tag, f.dofs) for f in sp.fixes) == sorted(
            (f.tag, f.dofs) for f in sf.fixes
        )
        # Regions: per-rank fragments merged — same count, same
        # member union.
        assert len(sp.regions) == len(sf.regions)
        for rp, rf in zip(sp.regions, sf.regions):
            assert rp.args[0] == rf.args[0] == "-node"
            assert set(rp.args[1:]) == set(rf.args[1:])
        # Patterns (incl. the HOLD pattern): same count; line content
        # equal as sets.
        assert len(sp.patterns) == len(sf.patterns), sp.name
        for pp, pf in zip(sp.patterns, sf.patterns):
            assert pp.type_token == pf.type_token
            assert sorted(
                (ld.target, ld.forces) for ld in pp.loads
            ) == sorted((ld.target, ld.forces) for ld in pf.loads)
            assert sorted(pp.sp_holds) == sorted(pf.sp_holds)
        # Analyze + chain: identical modulo the INV-5 runtime-fallback
        # keys the partitioned build may add.
        assert sp.analyze_steps == sf.analyze_steps
        assert sp.analyze_dt == sf.analyze_dt
        chain_p = {
            k: v for k, v in sp.chain_attrs.items()
            if not k.endswith("_runtime_fallback")
        }
        chain_f = {
            k: v for k, v in sf.chain_attrs.items()
            if not k.endswith("_runtime_fallback")
        }
        assert chain_p == chain_f


# ---------------------------------------------------------------------------
# 4. Round-trip + determinism hashes
# ---------------------------------------------------------------------------


def test_partitioned_staged_roundtrip_hash_stable(tmp_path: Path) -> None:
    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"
    _partitioned_bridge().h5(str(first))

    om = OpenSeesModel.from_h5(str(first))
    om.to_h5(str(second))

    assert _model_hash_of(second) == _model_hash_of(first)


def test_two_fresh_partitioned_staged_builds_hash_identically(
    tmp_path: Path,
) -> None:
    a = tmp_path / "a.h5"
    b = tmp_path / "b.h5"
    _partitioned_bridge().h5(str(a))
    _partitioned_bridge().h5(str(b))
    assert _model_hash_of(a) == _model_hash_of(b)


# ---------------------------------------------------------------------------
# 5. Unit level — foreign-decl filter + stage MP dedupe
# ---------------------------------------------------------------------------


def test_foreign_node_decl_never_enters_owned_node_ids() -> None:
    """Inside a stage's rank bracket, a node declaration outside the
    bridge-surfaced stage-owned set (a foreign ghost decl, ADR 0027
    INV-2) must not enter ``owned_node_ids`` — even when it PRECEDES
    the owning rank's declaration."""
    em = H5Emitter(model_name="u", snapshot_id="")
    em.stage_open("s1")
    set_stage_owned_node_tags(em, {10, 11})
    em.partition_open(0)
    em.node(99, 0.0, 0.0, 0.0)   # foreign ghost decl
    em.node(10, 1.0, 0.0, 0.0)   # stage-owned
    em.partition_close()
    em.partition_open(1)
    em.node(10, 1.0, 0.0, 0.0)   # shared stage node, second rank
    em.node(11, 2.0, 0.0, 0.0)   # stage-owned, rank 1 only
    em.partition_close()
    set_stage_owned_node_tags(em, None)
    em.stage_close()

    blk = em._stage_blocks[0]
    assert blk.owned_node_ids == [10, 11], blk.owned_node_ids


def test_stage_equal_dof_dedupes_with_single_seq_stamp() -> None:
    """A stage-claimed equalDOF replicated on both owning ranks
    captures ONE record and ONE emit_index stamp."""
    em = H5Emitter(model_name="u", snapshot_id="")
    em.stage_open("s1")
    em.partition_open(0)
    em.equalDOF(2, 5, 1, 2)
    em.partition_close()
    em.partition_open(1)
    em.equalDOF(2, 5, 1, 2)
    em.partition_close()
    em.stage_close()

    blk = em._stage_blocks[0]
    assert len(blk.equal_dofs) == 1
    assert len(blk.equal_dof_seq) == 1


def test_flat_staged_capture_untouched() -> None:
    """No partition brackets → the Phase-2 flat capture is unchanged
    (duplicates and all — the dedupe is partition-scoped)."""
    em = H5Emitter(model_name="u", snapshot_id="")
    em.stage_open("s1")
    em.fix(7, 1, 1)
    em.fix(7, 1, 1)
    em.stage_close()
    assert len(em._stage_blocks[0].fixes) == 2
