"""Partitions survive ``from_h5 → to_h5`` (ADR 0055 Phase 5 / P5.0b).

Before P5.0b the re-write path was partition-blind: ``OpenSeesModel``
never loaded ``/opensees/partitions`` and ``_populate_emitter_h5``
re-drove the element pool with no rank brackets, so a
``from_h5 → to_h5`` of a partitioned archive silently DROPPED

* the ``/opensees/partitions/partition_NN`` groups (rank membership +
  ``boundary_node_ids`` — the viewer's partition metadata, ADR 0027
  schema 2.10.0), and
* the real ranks in every ``/opensees/element_meta/*/partition_ids``
  column (re-written as all ``-1`` sentinels),

drifting ``model_hash`` (partitions and element_meta both fold in —
``MODEL_HASH_EXCLUDED_CHILDREN`` excludes only cuts/sweeps/regions/
names).

These tests pin the restore: ``OpenSeesModel`` carries the partition
records read-side and ``to_h5`` echoes them back through
``H5Emitter.restore_partition_blocks`` (the ``restore_stage_blocks``
pattern), keeping the round trip byte-stable.

Fixtures use a REAL partitioned two-quad :class:`FEMData` (not a
``FEMStub``) because ``OpenSeesModel.from_h5`` reloads the neutral
zone via ``FEMData.from_h5``, which rejects stub-only files.
"""
from __future__ import annotations

from pathlib import Path
from typing import cast

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
from apeGmsh.opensees import OpenSeesModel
from apeGmsh.opensees.apesees import apeSees


# The partitioned build declares no MP-friendly chain — the ADR 0027
# INV-5 auto-emit warnings are contracted behavior locked elsewhere.
_MP_AUTO_EMIT_FILTERS = (
    "ignore:len.fem.partitions. > 1 with no user-declared numberer:UserWarning",
    "ignore:len.fem.partitions. > 1 with no user-declared system:UserWarning",
)
pytestmark = [pytest.mark.filterwarnings(f) for f in _MP_AUTO_EMIT_FILTERS]


def build_partitioned_two_quad_fem() -> FEMData:
    """Two stacked quads split across two partitions.

    Partition 1 owns nodes 1-4 + element 1 (``Rock``); partition 2
    owns nodes 3-6 + element 2 (``Fill``).  Nodes 3 and 4 are shared
    — the cross-rank boundary.  Partition ids are 1-based (the Gmsh
    producer convention; ``runtime_rank_from_partition_record`` maps
    them to 0-based runtime ranks).
    """
    node_ids = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    node_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    quad_info = make_type_info(
        code=3, gmsh_name="Quadrangle 4", dim=2, order=1, npe=4, count=2,
    )
    quad_group = ElementGroup(
        element_type=quad_info,
        ids=np.array([1, 2], dtype=np.int64),
        connectivity=np.array(
            [[1, 2, 3, 4], [4, 3, 5, 6]], dtype=np.int64,
        ),
    )

    def _sel(ids: "list[int]") -> np.ndarray:
        return np.array(ids, dtype=np.int64)

    def _coords(ids: "list[int]") -> np.ndarray:
        return node_coords[[i - 1 for i in ids]]

    pg = {
        (2, 201): {
            "name": "Rock",
            "node_ids": _sel([1, 2, 3, 4]),
            "node_coords": _coords([1, 2, 3, 4]),
            "element_ids": _sel([1]),
        },
        (2, 202): {
            "name": "Fill",
            "node_ids": _sel([3, 4, 5, 6]),
            "node_coords": _coords([3, 4, 5, 6]),
            "element_ids": _sel([2]),
        },
        (0, 203): {
            "name": "Base",
            "node_ids": _sel([1, 2]),
            "node_coords": _coords([1, 2]),
            "element_ids": np.array([], dtype=np.int64),
        },
    }
    partitions = {
        1: {
            "node_ids": _sel([1, 2, 3, 4]),
            "element_ids": _sel([1]),
        },
        2: {
            "node_ids": _sel([3, 4, 5, 6]),
            "element_ids": _sel([2]),
        },
    }
    nodes = NodeComposite(
        node_ids=node_ids, node_coords=node_coords,
        physical=PhysicalGroupSet(pg), labels=LabelSet({}),
        partitions=partitions,
    )
    elements = ElementComposite(
        groups={3: quad_group},
        physical=PhysicalGroupSet(pg), labels=LabelSet({}),
        partitions=partitions,
    )
    info = MeshInfo(
        n_nodes=6, n_elems=2, bandwidth=3, types=[quad_info],
    )
    return FEMData(nodes=nodes, elements=elements, info=info)


def _build_bridge() -> apeSees:
    ops = apeSees(build_partitioned_two_quad_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="Fill", thickness=1.0, material=mat)
    ops.fix(pg="Base", dofs=(1, 1))
    return ops


def _model_hash_of(path: Path) -> str:
    with h5py.File(str(path), "r") as f:
        return str(f["meta"]["lineage"].attrs["model_hash"])


def _partitions_snapshot(path: Path) -> "dict[str, dict[str, object]]":
    """``{partition_NN: {rank, element_ids, node_ids, boundary_node_ids}}``."""
    out: "dict[str, dict[str, object]]" = {}
    with h5py.File(str(path), "r") as f:
        if "/opensees/partitions" not in f:
            return out
        parts = f["/opensees/partitions"]
        for name in parts:
            g = parts[name]
            out[name] = {
                "rank": int(g.attrs["rank"]),
                "element_ids": [int(v) for v in g["element_ids"][:]],
                "node_ids": [int(v) for v in g["node_ids"][:]],
                "boundary_node_ids": [
                    int(v) for v in g["boundary_node_ids"][:]
                ],
            }
    return out


def _partition_ids_columns(path: Path) -> "dict[str, list[int]]":
    """``{element_meta type group: partition_ids column}``."""
    out: "dict[str, list[int]]" = {}
    with h5py.File(str(path), "r") as f:
        meta = f["/opensees/element_meta"]
        for name in meta:
            out[name] = [int(v) for v in meta[name]["partition_ids"][:]]
    return out


def test_write_carries_partitions_sanity(tmp_path: Path) -> None:
    """Sanity (passes pre-P5.0b): the FIRST write of a partitioned
    build carries ``/opensees/partitions`` with one group per rank and
    real ranks in the ``partition_ids`` columns."""
    first = tmp_path / "first.h5"
    _build_bridge().h5(str(first))

    snap = _partitions_snapshot(first)
    assert len(snap) == 2, f"expected 2 partition groups, got {snap.keys()}"
    ranks = sorted(v["rank"] for v in snap.values())
    assert ranks == [0, 1]
    cols = _partition_ids_columns(first)
    all_ranks = [r for col in cols.values() for r in col]
    assert sorted(all_ranks) == [0, 1], (
        f"first write must stamp real ranks, got {cols!r}"
    )


def test_roundtrip_preserves_partitions_zone(tmp_path: Path) -> None:
    """``from_h5 → to_h5`` must reproduce ``/opensees/partitions``
    group-for-group (rank, element_ids, node_ids, boundary_node_ids)."""
    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"
    _build_bridge().h5(str(first))

    om = OpenSeesModel.from_h5(str(first))
    om.to_h5(str(second))

    snap_first = _partitions_snapshot(first)
    snap_second = _partitions_snapshot(second)
    assert snap_second, (
        "/opensees/partitions dropped on from_h5 -> to_h5 "
        "(re-write path is partition-blind)"
    )
    assert snap_second == snap_first


def test_roundtrip_preserves_partition_ids_columns(tmp_path: Path) -> None:
    """``from_h5 → to_h5`` must reproduce the per-type
    ``element_meta/*/partition_ids`` columns (not all ``-1``)."""
    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"
    _build_bridge().h5(str(first))

    om = OpenSeesModel.from_h5(str(first))
    om.to_h5(str(second))

    assert _partition_ids_columns(second) == _partition_ids_columns(first)


def test_roundtrip_model_hash_stable(tmp_path: Path) -> None:
    """``from_h5 → to_h5`` of a partitioned archive is
    ``model_hash``-stable (partitions + element_meta fold in)."""
    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"
    _build_bridge().h5(str(first))

    om = OpenSeesModel.from_h5(str(first))
    om.to_h5(str(second))

    assert _model_hash_of(second) == _model_hash_of(first)
