"""Displacement case names must survive the h5 neutral zone (schema 2.26.1).

Before the fix ``_write_sp_loads`` ignored ``SPRecord.pattern`` and
flattened every record into ``/loads/sp/default``; after any
``to_h5``/``from_h5`` round-trip a ``g.displacements.case("X")`` came
back as ``pattern='default'`` and ``p.from_model("X")`` imported
nothing.  These tests lock the per-pattern writer layout, the reader's
decode, the legacy-file fallback, and the compose survival of a module's
displacement-only case.

Runs entirely off the FEMData broker — no live gmsh, no openseespy.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from apeGmsh._core import apeGmsh
from apeGmsh._kernel.payloads import ElementGroup
from apeGmsh._kernel.record_sets import ComposeSet
from apeGmsh._kernel.records._loads import NodalLoadRecord, SPRecord
from apeGmsh.mesh._element_types import make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)


def _fem_with_cases() -> FEMData:
    """Four nodes on a line; SP records under three cases + a load case."""
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    node_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=1,
    )
    line_group = ElementGroup(
        element_type=line_info,
        ids=np.array([100], dtype=np.int64),
        connectivity=np.array([[1, 2]], dtype=np.int64),
    )
    nodes = NodeComposite(
        node_ids=node_ids, node_coords=coords,
        physical=PhysicalGroupSet({
            (0, 1): {"name": "tips",
                     "node_ids": np.array([4], dtype=np.int64),
                     "node_coords": coords[3:4]},
        }),
        labels=LabelSet({}),
        loads=[
            NodalLoadRecord(node_id=2, force_xyz=(0.0, 0.0, -10.0),
                            pattern="live"),
        ],
        sp=[
            SPRecord(node_id=1, dof=1, value=0.0, is_homogeneous=True,
                     pattern="default"),
            SPRecord(node_id=4, dof=3, value=-15.0, is_homogeneous=False,
                     pattern="push_gap", name="top_push_gap"),
            SPRecord(node_id=3, dof=3, value=-1.0, is_homogeneous=False,
                     pattern="settle"),
            SPRecord(node_id=3, dof=1, value=0.0, is_homogeneous=True,
                     pattern="push_gap"),   # case-scoped hold mask
        ],
    )
    elements = ElementComposite(
        groups={1: line_group},
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
    )
    info = MeshInfo(n_nodes=4, n_elems=1, bandwidth=1, types=[line_info])
    return FEMData(nodes=nodes, elements=elements, info=info,
                   composed_from=ComposeSet(()))


def test_sp_case_names_round_trip(tmp_path: Path) -> None:
    """The direct regression: case names survive to_h5 -> from_h5."""
    fem = _fem_with_cases()
    out = tmp_path / "m.h5"
    fem.to_h5(str(out))
    back = FEMData.from_h5(str(out))

    assert sorted(back.nodes.sp.patterns()) == sorted(
        ["default", "push_gap", "settle"])
    push = back.nodes.sp.by_pattern("push_gap")
    assert len(push) == 2
    pre = next(r for r in push if not r.is_homogeneous)
    assert pre.node_id == 4 and pre.dof == 3 and pre.value == -15.0
    assert pre.name == "top_push_gap"
    # The load case was never broken — still round-trips.
    assert back.nodes.loads.patterns() == ["live"]


def test_writer_emits_one_dataset_per_case(tmp_path: Path) -> None:
    """Writer-shape guard: /loads/sp/{case} exists per case, mirroring
    /loads/nodal/{pattern}."""
    fem = _fem_with_cases()
    out = tmp_path / "m.h5"
    fem.to_h5(str(out))
    with h5py.File(str(out), "r") as f:
        assert set(f["loads/sp"].keys()) == {"default", "push_gap", "settle"}
        assert len(f["loads/sp/push_gap"]) == 2
        assert len(f["loads/sp/settle"]) == 1


def test_legacy_flattened_file_still_decodes(tmp_path: Path) -> None:
    """A pre-2.26.1 file carries every SP record under /loads/sp/default.

    The pre-fix writer stored no pattern column, so a legacy file is
    byte-equivalent to writing the same records with pattern='default'
    — which is exactly what this constructs.  The reader must decode
    all records with pattern='default' (single dataset, no loss)."""
    fem = _fem_with_cases()
    flattened = [
        SPRecord(node_id=r.node_id, dof=r.dof, value=r.value,
                 is_homogeneous=r.is_homogeneous, name=r.name,
                 pattern="default")
        for r in fem.nodes.sp
    ]
    fem.nodes.sp = type(fem.nodes.sp)(flattened)  # type: ignore[attr-defined]
    out = tmp_path / "m.h5"
    fem.to_h5(str(out))
    with h5py.File(str(out), "r") as f:
        assert set(f["loads/sp"].keys()) == {"default"}   # legacy layout
    back = FEMData.from_h5(str(out))
    sps = list(back.nodes.sp)
    assert len(sps) == 4
    assert {r.pattern for r in sps} == {"default"}


def test_composed_module_displacement_case_survives(tmp_path: Path) -> None:
    """SP-side twin of test_compose_pattern_field_not_namespaced: a
    composed module's displacement case keeps its (un-namespaced) name
    on the host broker, so from_model('push_gap') can match it."""
    host = tmp_path / "host.h5"
    mod = tmp_path / "mod.h5"
    _fem_with_cases().to_h5(str(host))
    _fem_with_cases().to_h5(str(mod))

    g = apeGmsh.from_h5(host)
    g.compose(str(mod), label="M")
    sp = g._fem.nodes.sp
    # Host + module records both present, case names intact and
    # NOT namespaced ('push_gap', not 'M.push_gap').
    assert sorted(set(sp.patterns())) == sorted(
        ["default", "push_gap", "settle"])
    assert len(sp.by_pattern("push_gap")) == 4   # 2 host + 2 module
