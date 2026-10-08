"""ADR 0117 D6 (AS4-b, #1530): an assembly's rank layout without a host.

Oracles, each naming the right answer independently of the code under test:

* #1530 — two unranked instances of the 2x2x2 block give a serial deck: no
  ``getPID`` block, and every element (2 x 8 bricks) and node (2 x 27) on
  its own line, byte for byte the ``tcl(flat=True)`` deck.
* dense ranks — with ``partition_rank=`` the deck has exactly one block per
  declared rank, ``0 .. n-1``, and rank ``k`` holds the 8 bricks of the
  instance declared on ``k``: their FEM ids sit in that instance's window
  ``i * 1_000_000`` (ADR 0038), whatever the declaration order.
* reference nodes — they sit on rank 0 (below every window).
* seams — a bad, clashing or mixed ``partition_rank`` raises before
  anything is recorded; a rank with no instance raises at ``bridge()``; a
  ranked instance of an assembly archive is refused, and so is any
  instance of a *ranked* archive, ranked outer assembly or not;
  ``partition_rank`` round-trips through ``/assembly/instances`` (``-1``
  for none) and a tampered mixed column is refused on read.
* merge engine: a host-less chain auto-ranks from 0; a host that owns
  only an element still claims rank 0; overriding a foreign partition
  assignment warns, on the host-ful and the host-less path, and a
  consistent one does not.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from tests.assembly.test_two_instances_one_tie import (
    GRANULE,
    H,
    block_fem,
    declare_block,
    write_instance,
)

#: Hex8 bricks and nodes of one 2x2x2 block.
BRICKS = 8
NODES = 27
PID_BLOCK = re.compile(r"^if \{\[getPID\] == (\d+)\} \{$")


@pytest.fixture(scope="module")
def files(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("as4b")
    return {
        "block": write_instance(d / "block.h5", block_fem(d), declare_block),
        "dir": d,
    }


def _stack(files, ranks=(None, None), *, node: bool = False):
    from apeGmsh.assembly import Assembly

    asm = Assembly("stack")
    asm.instance("pier_1", files["block"], partition_rank=ranks[0])
    asm.instance("pier_2", files["block"], translate=(0.0, 0.0, H),
                 partition_rank=ranks[1])
    if node:
        asm.node("ref", (5.0, 5.0, 3 * H))
    return asm


def _deck(ops, path: Path, *, flat: bool = False) -> list[str]:
    ops.tcl(str(path), flat=flat)
    return path.read_text(encoding="utf-8").splitlines()


def _blocks(deck: list[str]) -> dict[int, list[str]]:
    """``{rank: stripped lines}`` of each ``if {[getPID] == k} {...}`` block."""
    out: dict[int, list[str]] = {}
    cur: "int | None" = None
    for ln in deck:
        m = PID_BLOCK.match(ln)
        if m:
            assert cur is None, "nested getPID block"
            cur = int(m.group(1))
            assert cur not in out, f"rank {cur} opened twice"
            out[cur] = []
        elif ln == "}" and cur is not None:
            cur = None
        elif cur is not None:
            out[cur].append(ln.strip())
    return out


def _brick_ids(lines: list[str]) -> list[int]:
    return [int(ln.split()[2]) for ln in lines if ln.startswith("element stdBrick")]


# ---------------------------------------------------------------------------
# #1530 — an unranked assembly is serial and complete
# ---------------------------------------------------------------------------

def test_an_unranked_assembly_writes_a_serial_complete_deck(files, tmp_path):
    ops = _stack(files).bridge(ndm=3, ndf=3)
    assert ops.fem.nodes.partitions == []
    deck = _deck(ops, tmp_path / "default.tcl")
    assert not any("getPID" in ln for ln in deck)
    assert len(_brick_ids(deck)) == 2 * BRICKS
    assert sum(ln.startswith("node ") for ln in deck) == 2 * NODES
    assert deck == _deck(ops, tmp_path / "flat.tcl", flat=True)


def test_reference_nodes_keep_an_unranked_assembly_serial(files, tmp_path):
    ops = _stack(files, node=True).bridge(ndm=3, ndf=3)
    assert ops.fem.nodes.partitions == []
    deck = _deck(ops, tmp_path / "d.tcl")
    assert not any("getPID" in ln for ln in deck)
    assert "node 1 5.0 5.0 30.0" in deck


# ---------------------------------------------------------------------------
# Dense ranks, one instance each
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ranks", [(0, 1), (1, 0)])
def test_a_ranked_assembly_writes_one_block_per_rank(files, tmp_path, ranks):
    ops = _stack(files, ranks, node=True).bridge(ndm=3, ndf=3)
    assert ops.fem.nodes.partitions == [0, 1]
    blocks = _blocks(_deck(ops, tmp_path / "r.tcl"))
    assert sorted(blocks) == [0, 1]
    for k, rank in enumerate(ranks, start=1):  # instance k: window k * GRANULE
        ids = _brick_ids(blocks[rank])
        assert len(ids) == BRICKS
        assert all(k * GRANULE <= i < (k + 1) * GRANULE for i in ids), ids
    # The reference node (FEM id 1, below every window) is owned by rank 0.
    assert "node 1 5.0 5.0 30.0" in blocks[0]
    assert "node 1 5.0 5.0 30.0" not in blocks[1]


def test_a_rank_with_no_instance_raises_at_bridge(files):
    from apeGmsh.assembly import AssemblyError

    asm = _stack(files, (0, 2))
    with pytest.raises(AssemblyError, match=r"rank\(s\) \[1\] with no instance"):
        asm.bridge(ndm=3, ndf=3)


@pytest.mark.parametrize("rank", [True, -1, 1.0, "0"])
def test_a_bad_partition_rank_raises_before_recording(files, rank):
    from apeGmsh.assembly import Assembly, AssemblyError

    asm = Assembly("x")
    with pytest.raises(AssemblyError, match="partition_rank must be an int"):
        asm.instance("p", files["block"], partition_rank=rank)
    assert asm.instances == ()
    asm.instance("p", files["block"], partition_rank=0)  # nothing was recorded
    assert asm.instances[0].partition_rank == 0


@pytest.mark.parametrize("first, second, match", [
    (0, None, "give every instance a rank"),
    (None, 0, "give every instance a rank"),
    (0, 0, "rank 0 already holds instance 'p'"),
])
def test_mixed_or_shared_ranks_raise_before_recording(files, first, second, match):
    from apeGmsh.assembly import Assembly, AssemblyError

    asm = Assembly("x").instance("p", files["block"], partition_rank=first)
    with pytest.raises(AssemblyError, match=match):
        asm.instance("q", files["block"], partition_rank=second)
    assert [i.label for i in asm.instances] == ["p"]
    asm.instance("q", files["block"], partition_rank=1 if first == 0 else None)


def test_a_ranked_instance_of_an_assembly_archive_is_refused(files, tmp_path):
    from apeGmsh.assembly import Assembly, AssemblyError

    inner = Assembly("inner").instance("A", files["block"])
    inner.bridge(ndm=3, ndf=3)
    inner.h5(tmp_path / "inner.h5")
    outer = (Assembly("outer")
             .instance("X", tmp_path / "inner.h5", partition_rank=0)
             .instance("Y", tmp_path / "inner.h5", translate=(0.0, 0.0, H),
                       partition_rank=1))
    with pytest.raises(AssemblyError, match="itself an assembly archive"):
        outer.bridge(ndm=3, ndf=3)
    unranked = (Assembly("outer").instance("X", tmp_path / "inner.h5")
                .instance("Y", tmp_path / "inner.h5", translate=(0.0, 0.0, H)))
    assert unranked.bridge(ndm=3, ndf=3).fem.nodes.partitions == []


@pytest.fixture(scope="module")
def ranked_archive(files) -> Path:
    """An assembly archive whose instances ``A`` and ``B`` carry ranks 0, 1."""
    from apeGmsh.assembly import Assembly

    inner = (Assembly("inner")
             .instance("A", files["block"], partition_rank=0)
             .instance("B", files["block"], translate=(0.0, 0.0, H),
                       partition_rank=1))
    inner.bridge(ndm=3, ndf=3)
    out = files["dir"] / "ranked_inner.h5"
    inner.h5(out)
    return out


@pytest.mark.parametrize("ranks", [(None,), (None, None), (0,), (0, 1)])
def test_an_instance_of_a_ranked_archive_is_refused(
        files, ranked_archive, ranks):
    """Review F1: its ``/composed_from`` ranks would be hints the assembly
    never declared: an empty rank 2 for one unranked instance, a raw rank
    collision for two. Refused in ``instance()``, recording nothing."""
    from apeGmsh.assembly import Assembly, AssemblyError

    asm = Assembly("o")
    for k, rank in enumerate(ranks[:-1]):
        asm.instance(f"P{k}", files["block"], partition_rank=rank,
                     translate=(0.0, 0.0, 3 * H * (k + 1)))
    with pytest.raises(AssemblyError,
                       match=r"modules \['A', 'B'\] that carry a partition_rank"):
        asm.instance("X", ranked_archive, partition_rank=ranks[-1])
    assert [i.label for i in asm.instances] == [
        f"P{k}" for k in range(len(ranks) - 1)]


# ---------------------------------------------------------------------------
# The merge engine's host rank (``_rebuild_partitions_from_modules``)
# ---------------------------------------------------------------------------

def _record(label: str, rank: "int | None"):
    from apeGmsh._kernel.records._compose import ComposeRecord

    return ComposeRecord(
        label=label, source_path=f"{label}.h5", source_fem_hash=label,
        source_neutral_schema_version="2.9.0", translate=(0.0, 0.0, 0.0),
        partition_rank=rank, composed_at="2026-10-07T00:00:00Z")


def _chain(ranks, node_labels, elem_labels):
    """Nodes 1-3 and line elements 10, 11 labelled by module (``""``: host)."""
    from apeGmsh._kernel.record_sets import ComposeSet
    from tests.test_phase_3b_2d import _make_fem

    return _make_fem(
        composed_from=ComposeSet(tuple(_record(k, r) for k, r in ranks.items())),
        node_module_labels=list(node_labels),
        elem_module_labels=list(elem_labels))


def _members(fem) -> dict[int, tuple[list[int], list[int]]]:
    return {r: (sorted(fem.nodes._partitions[r]["node_ids"].tolist()),
                sorted(fem.elements._partitions[r]["element_ids"].tolist()))
            for r in fem.nodes.partitions}


def test_a_host_less_chain_auto_ranks_from_zero():
    """Review F3: hints ``(None, 1)`` on an empty host put ``A`` on rank 0."""
    from apeGmsh.mesh._compose import _rebuild_partitions_from_modules

    fem = _rebuild_partitions_from_modules(
        _chain({"A": None, "B": 1}, ["A", "B", "B"], ["A", "B"]))
    assert _members(fem) == {0: ([1], [10]), 1: ([2, 3], [11])}


def test_a_host_owning_only_an_element_claims_rank_0():
    """Review F3: no host node, one host element: the host keeps rank 0
    and the unhinted module goes to rank 1 (not a serial FEM)."""
    from apeGmsh.mesh._compose import _rebuild_partitions_from_modules

    fem = _rebuild_partitions_from_modules(
        _chain({"A": None}, ["A", "A", "A"], ["", "A"]))
    assert _members(fem) == {0: ([], [10]), 1: ([1, 2, 3], [11])}


@pytest.mark.parametrize("host_node", ["", "A"])   # host-ful, host-less
def test_overriding_a_foreign_partition_assignment_warns(host_node):
    """Review F2: a METIS-like rank 7 is not a rank the modules would get,
    so replacing it warns (ADR 0038 Layer 3), with or without a host."""
    from apeGmsh.mesh._compose import _rebuild_partitions_from_modules

    fem = _chain({"A": None}, ["A", "A", host_node], ["A", "A"])
    fem.nodes._partitions = {7: {"node_ids": np.array([1, 2, 3]),
                                 "element_ids": np.array([], dtype=np.int64)}}
    with pytest.warns(UserWarning, match="overriding the existing partition"):
        _rebuild_partitions_from_modules(fem)


def test_a_consistent_partition_assignment_is_replaced_silently():
    """The rank set compose itself writes ({0, 1}) is not an override."""
    from apeGmsh.mesh._compose import _rebuild_partitions_from_modules

    fem = _chain({"A": None}, ["A", "A", ""], ["A", "A"])
    fem.nodes._partitions = {r: {"node_ids": np.array([r + 1]),
                                 "element_ids": np.array([], dtype=np.int64)}
                             for r in (0, 1)}
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _rebuild_partitions_from_modules(fem)


# ---------------------------------------------------------------------------
# Reload: /assembly/instances partition_rank
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ranks, column", [((1, 0), [1, 0]),
                                           ((None, None), [-1, -1])])
def test_partition_rank_round_trips_through_the_archive(
        files, tmp_path, ranks, column):
    from apeGmsh.assembly import Assembly

    asm = _stack(files, ranks)
    asm.bridge(ndm=3, ndf=3)
    out = tmp_path / "asm.h5"
    asm.h5(out)
    with h5py.File(str(out), "r") as f:
        assert list(f["assembly/instances/partition_rank"][()]) == column
    back = Assembly.from_h5(out)
    assert [i.partition_rank for i in back.instances] == list(ranks)
    assert back.instances == asm.instances


def test_a_mixed_rank_column_is_refused_on_read(files, tmp_path):
    from apeGmsh.assembly import Assembly, AssemblyError

    asm = _stack(files, (0, 1))
    asm.bridge(ndm=3, ndf=3)
    out = tmp_path / "asm.h5"
    asm.h5(out)
    with h5py.File(str(out), "r+") as f:
        f["assembly/instances/partition_rank"][1] = -1
    with pytest.raises(AssemblyError, match="give every instance a rank"):
        Assembly.from_h5(out)
