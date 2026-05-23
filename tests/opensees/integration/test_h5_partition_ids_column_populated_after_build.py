"""Integration test for H5 emitter partition columns under partitioning.

Per ADR 0027 (P4) the H5 emitter writes:

* ``/opensees/partitions/partition_NN/`` per-rank groups carrying
  ``element_ids`` / ``node_ids`` int64 datasets and metadata attrs.
* A parallel ``partition_ids`` int64 column on every
  ``/opensees/element_meta/{type_token}/`` group, recording the rank
  each element was emitted under.

This test drives the bridge end-to-end with a partitioned FEM stub
and verifies both the new groups + column appear, with values matching
the partition the bridge fanned each element out under.
"""
from __future__ import annotations

import os
import tempfile
from typing import cast

import h5py
import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.h5 import H5Emitter

from tests.opensees.fixtures.fem_stub import (
    make_two_column_frame_partitioned,
)


def test_h5_writes_partitions_group_and_partition_ids_column() -> None:
    """A 2-rank build with two elements populates
    ``/opensees/partitions/`` and the per-element ``partition_ids``
    column.
    """
    fem = make_two_column_frame_partitioned()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )

    bm = ops.build()
    emitter = H5Emitter(model_name="ptest", snapshot_id="")
    bm.emit(emitter)

    # Write to a tempfile and re-open with h5py to inspect.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ptest.h5")
        emitter.write(path)
        with h5py.File(path, "r") as f:
            # Partition groups present.
            assert "/opensees/partitions" in f, (
                "expected /opensees/partitions/ under partitioned build"
            )
            parts_grp = f["/opensees/partitions"]
            assert parts_grp.attrs["n_partitions"] == 2, (
                f"n_partitions attr must be 2; got "
                f"{parts_grp.attrs.get('n_partitions')!r}"
            )
            assert "partition_00" in parts_grp
            assert "partition_01" in parts_grp

            # Per-element partition_ids column present and matches
            # rank ownership: element 1 → rank 0; element 2 → rank 1.
            elem_meta = f["/opensees/element_meta"]
            # The element type token under elasticBeamColumn is the
            # canonical OpenSees name. We don't hard-code it; just
            # find the single group under element_meta.
            ele_type_groups = list(elem_meta.keys())
            assert len(ele_type_groups) == 1, (
                f"expected one element-type group under element_meta; "
                f"got {ele_type_groups!r}"
            )
            type_grp = elem_meta[ele_type_groups[0]]
            assert "partition_ids" in type_grp, (
                "expected partition_ids column on per-type element_meta group"
            )
            partition_ids = np.asarray(type_grp["partition_ids"])
            # Two elements, one per rank.
            assert partition_ids.shape == (2,), (
                f"partition_ids must be 1-D length 2; got {partition_ids.shape}"
            )
            # The two ranks must be 0 and 1 (order matches emission
            # order — rank 0 emitted first per ascending partition id).
            assert sorted(partition_ids.tolist()) == [0, 1], (
                f"partition_ids must cover ranks 0 and 1; got "
                f"{partition_ids.tolist()}"
            )
