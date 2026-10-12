"""End-to-end pin of the order partitioned emit declares each rank's nodes.

The owned-node pass of partitioned emit writes a rank's ``node`` lines
in ascending node id, whatever order the broker stores the nodes in and
whatever order the partition record lists them in (#1670). This model
feeds it both orders shuffled, with gapped ids, across two ranks that
share node 15, and asserts the per-rank blocks of the Tcl deck and the
per-rank ``ops.node`` calls of the Python deck against text written here.
The goldens cannot see this: their partition node lists arrive sorted.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import cast

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.material.uniaxial import ElasticMaterial

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)

#: Broker storage order, shuffled on purpose.
_BROKER_IDS = [40, 7, 23, 3, 61, 15]
_X = {3: 0.0, 7: 1.0, 15: 2.0, 23: 3.0, 40: 4.0, 61: 5.0}
_CHAIN = [3, 7, 15, 23, 40, 61]


def _bridge() -> apeSees:
    nodes = _NodesStub(
        ids=_BROKER_IDS,
        coords=[(_X[n], 0.0, 0.0) for n in _BROKER_IDS],
        node_pgs={"All": _BROKER_IDS},
    )
    elements = _ElementsStub(elem_pgs={"Bars": _ElementGroupView(
        ids=(101, 102, 103, 104, 105),
        connectivity=tuple(zip(_CHAIN, _CHAIN[1:])),
    )})
    fem = FEMStub(nodes=nodes, elements=elements)
    # Partition node lists unsorted too; node 15 sits on both ranks.
    fem.set_partitions([
        (0, [15, 3, 7], [101, 102]),
        (1, [61, 23, 40, 15], [103, 104, 105]),
    ])
    ops = apeSees(cast("object", fem), _artifacts=False)
    ops.model(ndm=3, ndf=3)
    mat = ElasticMaterial(E=1.0)
    ops.register(mat)
    ops.element.Truss(pg="Bars", A=1.0, material=mat)
    return ops


_TCL_RANK0 = """\
if {[getPID] == 0} {
    node 3 0.0 0.0 0.0
    node 7 1.0 0.0 0.0
    node 15 2.0 0.0 0.0
    element Truss 2 3 7 1.0 1
    element Truss 3 7 15 1.0 1
}
"""

_TCL_RANK1 = """\
if {[getPID] == 1} {
    node 15 2.0 0.0 0.0
    node 23 3.0 0.0 0.0
    node 40 4.0 0.0 0.0
    node 61 5.0 0.0 0.0
    element Truss 4 15 23 1.0 1
    element Truss 5 23 40 1.0 1
    element Truss 6 40 61 1.0 1
}
"""


def test_tcl_rank_blocks_declare_nodes_in_ascending_id(tmp_path: Path) -> None:
    path = tmp_path / "deck.tcl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _bridge().tcl(str(path))
    text = path.read_text()
    assert _TCL_RANK0 in text
    assert _TCL_RANK1 in text
    assert text.index(_TCL_RANK0) < text.index(_TCL_RANK1)


def test_py_rank_blocks_declare_nodes_in_ascending_id(tmp_path: Path) -> None:
    path = tmp_path / "deck.py"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _bridge().py(str(path))
    by_rank: dict[int, list[int]] = {}
    rank = None
    for line in path.read_text().splitlines():
        m = re.match(r"\s*if .*getPID\(\)\s*==\s*(\d+)", line)
        if m:
            rank = int(m.group(1))
            continue
        m = re.match(r"\s*ops\.node\((\d+),", line)
        if m:
            assert rank is not None, line
            by_rank.setdefault(rank, []).append(int(m.group(1)))
    assert by_rank == {0: [3, 7, 15], 1: [15, 23, 40, 61]}
