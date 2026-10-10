"""ADR 0120 D2 — ``FEMData.repartition``: one graph, whatever module it came from.

Oracles, each independent of the code under test:

* a partition is a set of elements: every element of the snapshot is in
  exactly one rank, and a rank holds exactly the nodes of its elements
  (so a node is in several ranks iff its elements are);
* RCB on a uniform box of equal weights gives ranks whose weights differ
  by at most one element per cut level from ``total / n``;
* two calls on one snapshot give the same partition, and the input keeps
  its own;
* ``n_parts=1`` is the unpartitioned snapshot; ids run ``1 .. n``;
* the partition survives ``model.h5`` (``/partitions``) and drives
  ``select(partition=k)``;
* a composed model — one rank per module out of the merge engine — is
  re-cut across the modules, so a rank can hold elements of both.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.mesh import FEMData


@pytest.fixture(scope="module")
def box_fem():
    """A 6 x 4 x 2 structured hex8 box (48 equal bricks) plus its faces."""
    with apeGmsh(model_name="rp_box", verbose=False) as g:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 6.0, 4.0, 2.0, label="v")
        g.physical.add_volume("v", name="Vol")
        g.mesh.recipe.structured(size=1.0, fallback="strict")
        return g.mesh.queries.get_fem_data(dim=3)


def _elements(fem):
    return {int(e): tuple(int(n) for n in row)
            for grp in fem.elements for e, row in zip(grp.ids, grp.connectivity)}


@pytest.mark.parametrize("n", [2, 3, 4, 7])
def test_every_element_once_and_each_rank_holds_its_element_nodes(box_fem, n):
    fem = box_fem.repartition(n)
    parts = list(fem.partitions)
    assert [p.id for p in parts] == list(range(1, n + 1))
    conn = _elements(fem)
    seen: list[int] = []
    for p in parts:
        eids = [int(e) for e in p.element_ids]
        seen += eids
        expect = sorted({nd for e in eids for nd in conn[e]})
        assert [int(x) for x in p.node_ids] == expect
    assert sorted(seen) == sorted(conn)


@pytest.mark.parametrize("n", [2, 4, 8])
def test_equal_weights_balance_to_one_element(box_fem, n):
    fem = box_fem.repartition(n)
    sizes = [p.n_elements for p in fem.partitions]
    total = sum(sizes)
    assert max(sizes) - min(sizes) <= int(np.ceil(np.log2(n)))
    assert abs(max(sizes) - total / n) <= int(np.ceil(np.log2(n)))


def test_deterministic_and_the_input_keeps_its_partition(box_fem):
    a = box_fem.repartition(4)
    b = box_fem.repartition(4)
    assert [p.element_ids.tolist() for p in a.partitions] == [
        p.element_ids.tolist() for p in b.partitions]
    assert len(box_fem.partitions) == 0
    assert box_fem.nodes._partitions in ({}, None)
    assert a.snapshot_id == box_fem.snapshot_id   # partitions are not content


def test_one_part_is_unpartitioned(box_fem):
    four = box_fem.repartition(4)
    one = four.repartition(1)
    assert len(one.partitions) == 0
    assert len(four.partitions) == 4


def test_weights_mapping_and_callable(box_fem):
    heavy_left = box_fem.repartition(
        2, weights=lambda name, ids, cent: np.where(cent[:, 0] < 3.0, 3.0, 1.0))
    # three times the weight on x < 3: the left part takes fewer bricks
    sizes = sorted(p.n_elements for p in heavy_left.partitions)
    assert sizes[0] < sizes[1]
    same = box_fem.repartition(2, weights={"hex8": 1.0})
    assert [p.n_elements for p in same.partitions] == [24, 24]


@pytest.mark.parametrize("kw, exc, match", [
    ({"n_parts": 0}, ValueError, ">= 1"),
    ({"n_parts": 49}, ValueError, "cannot fill"),
    ({"n_parts": 2, "weights": {"tet4": 1.0}}, ValueError, "no entry"),
    ({"n_parts": 2, "weights": {"hex8": -1.0}}, ValueError, ">= 0"),
    ({"n_parts": 2, "weights": {"hex8": 0.0}}, ValueError, "zero"),
    ({"n_parts": 2.0}, TypeError, "int"),
])
def test_refusals(box_fem, kw, exc, match):
    n = kw.pop("n_parts")
    with pytest.raises(exc, match=match):
        box_fem.repartition(n, **kw)


def test_survives_model_h5_and_drives_select(box_fem, tmp_path: Path):
    fem = box_fem.repartition(3)
    path = tmp_path / "rp.h5"
    fem.to_h5(str(path))
    back = FEMData.from_h5(str(path))
    assert [(p.id, p.element_ids.tolist(), p.node_ids.tolist())
            for p in back.partitions] == [
        (p.id, p.element_ids.tolist(), p.node_ids.tolist())
        for p in fem.partitions]
    sel = back.elements.select(partition=2)
    assert sorted(int(e) for e in sel.ids) == fem.partitions[2].element_ids.tolist()


def test_a_composed_model_is_cut_across_its_modules(tmp_path: Path):
    """Two stacked blocks merged by ``Assembly`` with one rank each come
    back from ``repartition(2)`` cut across the stack (z is not the long
    axis of the pair), so each rank holds bricks of both instances."""
    from apeGmsh.assembly import Assembly

    with apeGmsh(model_name="blk", verbose=False,
                 save_to=str(tmp_path / "blk.h5")) as g:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 6.0, 2.0, 1.0, label="v")
        g.physical.add_volume("v", name="Vol")
        g.mesh.recipe.structured(size=1.0, fallback="strict")
        g.mesh.queries.get_fem_data(dim=3)
    asm = Assembly("stack")
    asm.instance("a", tmp_path / "blk.h5", partition_rank=0)
    asm.instance("b", tmp_path / "blk.h5", translate=(0.0, 0.0, 1.0),
                 partition_rank=1)
    ranked = asm.fem()
    assert len(ranked.partitions) == 2
    per_module = [set(int(e) for e in p.element_ids) for p in ranked.partitions]
    cut = ranked.repartition(2)
    for p in cut.partitions:
        mine = set(int(e) for e in p.element_ids)
        assert mine & per_module[0] and mine & per_module[1]
