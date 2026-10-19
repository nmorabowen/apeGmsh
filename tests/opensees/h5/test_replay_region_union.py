"""Replay unions same-tag ``-node … -ele …`` region rows of a partitioned
archive (program slice K R1b-S3a, #2254; unblocks #2195).

A partitioned emit writes a recorder's filter region once per rank that
owns a member, each row the rank's intersection (``-node ids``,
``-ele ids`` or ``-node ids -ele ids``). The flat bridge deck declares the
tag once, ``-node <all> -ele <all>``. Replay
(``OpenSeesModel.from_h5(...).build(..., flat=True)``) used to union only
plain ``-node`` rows and raised ``ValueError`` on the ``-ele`` form, so no
partitioned archive with an element-scoped recorder region replayed.

Oracles, each naming the right answer:

* **The flat bridge deck.** On the real-FEMData twin of the golden corpus's
  ``two_column_frame_partitioned`` / ``staged_partitioned`` / ``recording``
  cell (S3's ``_broad_partitioned``, copied from
  ``k-2195-broad-partitioned-twin`` @ b0e2fea9), the archive holds the two
  rank rows ``(-node 2 -ele 2)`` and ``(-node 4 -ele 3)``, and the replayed
  flat deck's ``region`` lines equal ``ops.tcl(flat=True)``'s.
* **Closed-form unions** of the three member-list shapes, including a rank
  that owns no element (``-node`` only) or no node (``-ele`` only).
* **Every other shape refuses**: a ``-rayleigh`` / ``-damp`` tail,
  ``-eleRange``, an empty flag list, a repeated or reordered flag.

The H5 writer's ``_merge_node_region_args`` is untouched; the golden corpus
and the writer tests pin that.
"""
from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)
from apeGmsh.opensees import OpenSeesModel, OpenSeesTarget, apeSees
from apeGmsh.opensees._internal.lineage import compute_model_hash
from apeGmsh.opensees._internal.typed_records import RegionRecord
from apeGmsh.opensees.emitter import h5_reader
from apeGmsh.opensees.opensees_model import _merge_region_rows

# ---------------------------------------------------------------------------
# The twin (copied from k-2195-broad-partitioned-twin @ b0e2fea9,
# tests/opensees/h5/test_h5_decl_params.py; S3 owns that file)
# ---------------------------------------------------------------------------

#: The golden corpus's frame split, as real Gmsh partitions (1-based ids;
#: the runtime rank is the index, not ``id - 1``): column A on rank 0,
#: column B on rank 1, nothing shared.
_FRAME_PARTITIONS: dict[int, dict[str, list[int]]] = {
    1: {"node_ids": [1, 2], "element_ids": [1]},
    2: {"node_ids": [3, 4], "element_ids": [2]},
}


def _two_column_frame_fem(
    partitions: "Mapping[int, Mapping[str, list[int]]]" = _FRAME_PARTITIONS,
) -> FEMData:
    """``fem_stub.make_two_column_frame_partitioned`` as a real
    :class:`FEMData`: columns A (nodes 1-2, element 1) and B (nodes 3-4,
    element 2) at x = 0 and 1, rising along z, in the PG ``"Cols"``, with
    the node sets ``"Base"`` (1, 3) and ``"Top"`` (2, 4), split by
    ``partitions``."""
    node_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    node_coords = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
        dtype=np.float64)
    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=2)
    lines = ElementGroup(
        element_type=line_info, ids=np.array([1, 2], dtype=np.int64),
        connectivity=np.array([[1, 2], [3, 4]], dtype=np.int64))

    def ids(values: "list[int]") -> np.ndarray:
        return np.array(values, dtype=np.int64)

    def pg(name: str, nodes: "list[int]", elements: "list[int]") -> dict[str, Any]:
        return {"name": name, "node_ids": ids(nodes),
                "node_coords": node_coords[[n - 1 for n in nodes]],
                "element_ids": ids(elements)}

    groups = PhysicalGroupSet({
        (1, 100): pg("Cols", [1, 2, 3, 4], [1, 2]),
        (0, 101): pg("Base", [1, 3], []),
        (0, 102): pg("Top", [2, 4], []),
    })
    parts = {int(pid): {k: ids(list(v)) for k, v in p.items()}
             for pid, p in partitions.items()}
    nodes = NodeComposite(
        node_ids=node_ids, node_coords=node_coords, physical=groups,
        labels=LabelSet({}), partitions=parts)
    elements = ElementComposite(
        groups={1: lines}, physical=groups, labels=LabelSet({}),
        partitions=parts)
    info = MeshInfo(n_nodes=4, n_elems=2, bandwidth=1, types=[line_info])
    return FEMData(nodes=nodes, elements=elements, info=info,
                   model_name="two_column_frame_partitioned")


def _broad_partitioned() -> apeSees:
    """The golden corpus's staged, partitioned, recording frame on a real
    :class:`FEMData`: an elastic column pair fixed at ``"Base"``, the node,
    element and MPCO recorders, and two stages each with its own pattern
    and the parallel static chain."""
    ops = apeSees(_two_column_frame_fem(), opensees=OpenSeesTarget(mode="stock"))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4)
    ops.fix(pg="Base", dofs=(1,) * 6)
    ops.recorder.Node(
        file="out/node_disp.out", response="disp", pg="Top", dofs=(1, 2))
    ops.recorder.Element(
        file="out/ele_force.out", response=("globalForce",), pg="Cols")
    ops.recorder.MPCO(
        file="out/run.mpco", nodal_responses=("displacement",),
        elem_responses=("force",), nodes_pg="Top", elements_pg="Cols")
    for name, force, n in (("gravity", 10.0, 2), ("push", -5.0, 1)):
        with ops.stage(name=name) as s:
            series = ops.timeSeries.Linear()
            with s.pattern(series=series) as p:
                p.load(pg="Top", forces=(force, 0.0, 0.0, 0.0, 0.0, 0.0))
            s.analysis(
                test=ops.test.NormDispIncr(tol=1e-6, max_iter=10),
                algorithm=ops.algorithm.Newton(),
                integrator=ops.integrator.LoadControl(dlam=0.5),
                constraints=ops.constraints.Plain(),
                numberer=ops.numberer.ParallelRCM(),
                system=ops.system.Mumps(),
                analysis=ops.analysis.Static())
            s.run(n_increments=n)
    return ops


def _region_lines(deck: str) -> list[str]:
    return [ln.strip() for ln in deck.splitlines()
            if ln.strip().startswith("region ")]


# ---------------------------------------------------------------------------
# Oracle 1: the flat bridge deck, on the S3 twin
# ---------------------------------------------------------------------------


def test_partitioned_twin_replays_the_flat_bridge_region(tmp_path: Path) -> None:
    ops = _broad_partitioned()
    deck = tmp_path / "bridge.tcl"
    archive = tmp_path / "model.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the auto-emit / ledger advisories
        ops.tcl(str(deck), progress=False, flat=True)
        ops.h5(str(archive))

    # The reproducer's shape: one row per rank, each '-node … -ele …'.
    with h5_reader.open(str(archive)) as m:
        rows = [(r.tag, tuple(r.args)) for r in m.regions()]
    assert rows == [
        (1, ("-node", 2, "-ele", 2)),
        (1, ("-node", 4, "-ele", 3)),
    ]

    bridge = _region_lines(deck.read_text(encoding="utf-8"))
    assert bridge == ["region 1 -node 2 4 -ele 2 3"]

    replay = OpenSeesModel.from_h5(str(archive)).build("tcl", flat=True)
    assert isinstance(replay, str)
    assert _region_lines(replay) == bridge


def test_partitioned_twin_h5_rewrite_echoes_the_rank_rows(tmp_path: Path) -> None:
    """The union is deck-side only: an H5 rewrite of the twin stores the
    per-rank rows verbatim and keeps ``model_hash`` (a fixed-point guard,
    not a regression test: this archive's rewrite never reached the
    union)."""
    ops = _broad_partitioned()
    src = tmp_path / "model.h5"
    out = tmp_path / "rewrite.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops.h5(str(src))
        OpenSeesModel.from_h5(str(src)).build("h5", out=str(out))

    def rows(path: Path) -> list[tuple[int, tuple[Any, ...]]]:
        with h5_reader.open(str(path)) as m:
            return [(r.tag, tuple(r.args)) for r in m.regions()]

    def model_hash(path: Path) -> str:
        with h5py.File(str(path), "r") as f:
            return compute_model_hash("", f["opensees"])

    assert rows(out) == rows(src) == [
        (1, ("-node", 2, "-ele", 2)),
        (1, ("-node", 4, "-ele", 3)),
    ]
    assert model_hash(out) == model_hash(src)


# ---------------------------------------------------------------------------
# Oracle 2: closed-form unions, and the refusals
# ---------------------------------------------------------------------------


def _rows(*arg_tails: tuple[Any, ...], tag: int = 7) -> tuple[RegionRecord, ...]:
    return tuple(RegionRecord(tag=tag, args=a) for a in arg_tails)


@pytest.mark.parametrize(("tails", "want"), [
    # Both ranks own nodes and elements (the twin's shape).
    ([("-node", 2, "-ele", 2), ("-node", 4, "-ele", 3)],
     ("-node", 2, 4, "-ele", 2, 3)),
    # A shared interface node appears once, first occurrence kept.
    ([("-node", 5, 2, "-ele", 9), ("-node", 2, 4, "-ele", 3, 9)],
     ("-node", 5, 2, 4, "-ele", 9, 3)),
    # A rank owning no element of the region writes '-node' only.
    ([("-node", 1, "-ele", 1), ("-node", 3)],
     ("-node", 1, 3, "-ele", 1)),
    ([("-node", 3), ("-node", 1, "-ele", 1)],
     ("-node", 3, 1, "-ele", 1)),
    # A rank owning no node of the region writes '-ele' only.
    ([("-ele", 4), ("-node", 1, "-ele", 1)],
     ("-node", 1, "-ele", 4, 1)),
    ([("-node", 1), ("-ele", 2)],
     ("-node", 1, "-ele", 2)),
    ([("-ele", 1), ("-ele", 2, 1)],
     ("-ele", 1, 2)),
    # The plain '-node' union keeps its old answer.
    ([("-node", 1, 2), ("-node", 2, 3)],
     ("-node", 1, 2, 3)),
    # Three ranks fold left to right.
    ([("-node", 1, "-ele", 1), ("-node", 2), ("-ele", 3)],
     ("-node", 1, 2, "-ele", 1, 3)),
])
def test_member_list_rows_union(
    tails: list[tuple[Any, ...]], want: tuple[Any, ...],
) -> None:
    (merged,) = _merge_region_rows(_rows(*tails))
    assert merged.tag == 7
    assert tuple(merged.args) == want


def test_distinct_tags_and_a_lone_row_pass_through() -> None:
    odd = ("-ele", 1, 2, "-rayleigh", 0.1, 0.0, 0.0, 0.0)
    rows = (
        RegionRecord(tag=1, args=("-node", 1, "-ele", 1)),
        RegionRecord(tag=2, args=odd),
        RegionRecord(tag=1, args=("-node", 3, "-ele", 2)),
    )
    out = _merge_region_rows(rows)
    assert [(r.tag, tuple(r.args)) for r in out] == [
        (1, ("-node", 1, 3, "-ele", 1, 2)),
        (2, odd),
    ]


@pytest.mark.parametrize(("first", "second"), [
    # A flag tail the union cannot parse.
    (("-node", 1, "-ele", 1), ("-ele", 2, "-rayleigh", 0.1, 0.0, 0.0, 0.0)),
    (("-node", 1), ("-node", 2, "-rayleigh", 0.1, 0.0, 0.0, 0.0)),
    (("-ele", 1, "-damp", 3), ("-ele", 2, "-damp", 3)),
    # '-ele' against '-eleRange'.
    (("-ele", 1, 2), ("-eleRange", 3, 4)),
    (("-node", 1, "-ele", 1), ("-node", 2, "-eleRange", 3, 4)),
    # An empty flag list: the bridge never writes one.
    (("-node", 1, "-ele"), ("-node", 2, "-ele", 2)),
    (("-node", "-ele", 1), ("-node", 2, "-ele", 2)),
    (("-node",), ("-node", 2)),
    # A flag repeated or out of the bridge's '-node' then '-ele' order.
    (("-ele", 1, "-node", 1), ("-node", 2, "-ele", 2)),
    (("-node", 1, "-node", 2), ("-node", 3)),
    # A non-integer member, or ids with no flag.
    (("-node", 1.5), ("-node", 2)),
    ((1, 2), ("-node", 2)),
    ((), ("-node", 2)),
])
def test_every_other_shape_refuses(
    first: tuple[Any, ...], second: tuple[Any, ...],
) -> None:
    for a, b in ((first, second), (second, first)):
        with pytest.raises(ValueError, match=r"two rows for region 7"):
            _merge_region_rows(_rows(a, b))
