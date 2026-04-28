"""Mocked MPCO multi-partition reads.

Builds two synthetic partition files (``foo.part-0.mpco`` /
``foo.part-1.mpco``) in a tmpdir, drives ``MPCOMultiPartitionReader``
end-to-end without OpenSees, and verifies node-merge / element-concat
semantics.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.readers._mpco_multi import (
    MPCOMultiPartitionReader, discover_partition_files,
)
from apeGmsh.results.readers._protocol import ResultLevel
from apeGmsh.solvers._element_response import (
    ELE_TAG_FourNodeTetrahedron, IntRule,
)


# =====================================================================
# Synthetic partition file builder
# =====================================================================

def _create_partition(
    path: Path, *,
    node_ids: np.ndarray,
    coords: np.ndarray,
    element_ids: np.ndarray,
    element_connectivity: np.ndarray,
    n_steps: int,
    dt: float,
    disp_value_fn,
    stress_value_fn,
) -> None:
    """Write one minimal MPCO partition file.

    A 4-node tet bucket carries the elements; nodes carry
    DISPLACEMENT under ON_NODES; bricks carry stress under
    ON_ELEMENTS/stresses.
    """
    f = h5py.File(path, "w")
    info = f.create_group("INFO")
    info.create_dataset("SPATIAL_DIM", data=3)
    info.create_dataset("SOLVER_NAME", data=np.bytes_(b"OpenSees"))
    info.create_dataset("SOLVER_VERSION", data=np.array([3, 7, 2]))

    stage = f.create_group("MODEL_STAGE[1]")
    stage.attrs["STEP"] = 0
    stage.attrs["TIME"] = 0.0

    model = stage.create_group("MODEL")
    nodes = model.create_group("NODES")
    nodes.create_dataset("ID", data=node_ids.reshape(-1, 1).astype(np.int32))
    nodes.create_dataset("COORDINATES", data=coords.astype(np.float64))

    elements = model.create_group("ELEMENTS")
    bucket = np.column_stack([
        element_ids.astype(np.int32),
        element_connectivity.astype(np.int32),
    ])
    elements.create_dataset(
        f"{ELE_TAG_FourNodeTetrahedron}-FourNodeTetrahedron"
        f"[{IntRule.Tet_GL_1}:0]",
        data=bucket,
    )

    results_grp = stage.create_group("RESULTS")
    on_nodes = results_grp.create_group("ON_NODES")
    disp = on_nodes.create_group("DISPLACEMENT")
    disp.attrs["DISPLAY_NAME"] = np.bytes_(b"Displacement")
    disp.attrs["COMPONENTS"] = np.array([np.bytes_(b"Ux,Uy,Uz")])
    disp.create_dataset("ID", data=node_ids.reshape(-1, 1).astype(np.int32))
    disp_data = disp.create_group("DATA")
    for k in range(n_steps):
        ds = disp_data.create_dataset(
            f"STEP_{k}",
            data=np.array(
                [disp_value_fn(k, int(nid)) for nid in node_ids],
                dtype=np.float64,
            ),
        )
        ds.attrs["STEP"] = k
        ds.attrs["TIME"] = (k + 1) * dt

    on_elements = results_grp.create_group("ON_ELEMENTS")
    stresses = on_elements.create_group("stresses")
    bucket_key = (
        f"{ELE_TAG_FourNodeTetrahedron}-FourNodeTetrahedron"
        f"[{IntRule.Tet_GL_1}:0:0]"
    )
    bg = stresses.create_group(bucket_key)
    bg.attrs["NUM_COLUMNS"] = np.array([6], dtype=np.int32)
    meta = bg.create_group("META")
    meta.create_dataset(
        "MULTIPLICITY", data=np.array([[1]], dtype=np.int32),
    )
    meta.create_dataset(
        "GAUSS_IDS", data=np.array([[0]], dtype=np.int32),
    )
    meta.create_dataset(
        "NUM_COMPONENTS", data=np.array([[6]], dtype=np.int32),
    )
    meta.create_dataset(
        "COMPONENTS",
        data=np.array([b"0.1.sxx,syy,szz,sxy,syz,sxz"]),
    )
    bg.create_dataset(
        "ID", data=element_ids.reshape(-1, 1).astype(np.int32),
    )
    data = bg.create_group("DATA")
    for k in range(n_steps):
        ds = data.create_dataset(
            f"STEP_{k}",
            data=np.array(
                [stress_value_fn(k, int(eid)) for eid in element_ids],
                dtype=np.float64,
            ),
        )
        ds.attrs["STEP"] = k
        ds.attrs["TIME"] = (k + 1) * dt

    f.close()


@pytest.fixture
def two_partitions(tmp_path: Path):
    """Two partitions:
    - part-0: nodes [1, 2, 3, 4] (1 elem); first three are local, node 4 is shared.
    - part-1: nodes [4, 5, 6, 7] (1 elem); node 4 is the boundary (shared with part-0).
    """
    part0 = tmp_path / "Recorder.part-0.mpco"
    part1 = tmp_path / "Recorder.part-1.mpco"

    _create_partition(
        part0,
        node_ids=np.array([1, 2, 3, 4], dtype=np.int64),
        coords=np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ], dtype=np.float64),
        element_ids=np.array([10], dtype=np.int64),
        element_connectivity=np.array([[1, 2, 3, 4]], dtype=np.int64),
        n_steps=3, dt=0.1,
        # Encode (k, nid) so the slab can be checked.
        disp_value_fn=lambda k, nid: [k * 100 + nid, 0.0, 0.0],
        stress_value_fn=lambda k, eid: [
            k + eid + ax * 0.001 for ax in range(6)
        ],
    )
    _create_partition(
        part1,
        node_ids=np.array([4, 5, 6, 7], dtype=np.int64),
        coords=np.array([
            # Node 4: same coordinate as in part0 (boundary copy).
            [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
        ], dtype=np.float64),
        element_ids=np.array([20], dtype=np.int64),
        element_connectivity=np.array([[4, 5, 6, 7]], dtype=np.int64),
        n_steps=3, dt=0.1,
        disp_value_fn=lambda k, nid: [k * 100 + nid, 0.0, 0.0],
        stress_value_fn=lambda k, eid: [
            k + eid + ax * 0.001 for ax in range(6)
        ],
    )
    return part0, part1


# =====================================================================
# Partition discovery
# =====================================================================

class TestDiscoverPartitionFiles:
    def test_finds_both_siblings(self, two_partitions) -> None:
        part0, part1 = two_partitions
        found = discover_partition_files(part0)
        assert found == [part0, part1]

    def test_passing_part1_finds_both(self, two_partitions) -> None:
        part0, part1 = two_partitions
        found = discover_partition_files(part1)
        assert found == [part0, part1]   # Sorted by index.

    def test_non_partition_filename_returns_self(self, tmp_path: Path) -> None:
        # File without the .part-N pattern is returned as-is.
        plain = tmp_path / "single.mpco"
        plain.touch()
        found = discover_partition_files(plain)
        assert found == [plain]

    def test_missing_index_raises(self, tmp_path: Path) -> None:
        # Skip part-1, only ship part-0 and part-2 → non-contiguous.
        (tmp_path / "Run.part-0.mpco").touch()
        (tmp_path / "Run.part-2.mpco").touch()
        with pytest.raises(ValueError, match="not contiguous"):
            discover_partition_files(tmp_path / "Run.part-0.mpco")


# =====================================================================
# Construction-time validation
# =====================================================================

class TestConstructionValidation:
    def test_opens_two_partitions(self, two_partitions) -> None:
        part0, part1 = two_partitions
        reader = MPCOMultiPartitionReader([part0, part1])
        sid = reader.stages()[0].id
        assert reader.partitions(sid) == ["partition_0", "partition_1"]
        reader.close()

    def test_mismatched_time_raises(self, tmp_path: Path) -> None:
        part0 = tmp_path / "Bad.part-0.mpco"
        part1 = tmp_path / "Bad.part-1.mpco"
        _create_partition(
            part0,
            node_ids=np.array([1], dtype=np.int64),
            coords=np.zeros((1, 3)),
            element_ids=np.array([10], dtype=np.int64),
            element_connectivity=np.array([[1]], dtype=np.int64),
            n_steps=2, dt=0.1,
            disp_value_fn=lambda k, n: [0, 0, 0],
            stress_value_fn=lambda k, e: [0] * 6,
        )
        # Different step count.
        _create_partition(
            part1,
            node_ids=np.array([2], dtype=np.int64),
            coords=np.zeros((1, 3)),
            element_ids=np.array([20], dtype=np.int64),
            element_connectivity=np.array([[2]], dtype=np.int64),
            n_steps=5, dt=0.1,
            disp_value_fn=lambda k, n: [0, 0, 0],
            stress_value_fn=lambda k, e: [0] * 6,
        )
        with pytest.raises(ValueError, match="stage signatures"):
            MPCOMultiPartitionReader([part0, part1])


# =====================================================================
# Node merge — boundary deduplication
# =====================================================================

class TestNodeMerge:
    def test_unique_node_ids_after_merge(self, two_partitions) -> None:
        part0, _ = two_partitions
        r = Results.from_mpco(str(part0))
        slab = r.nodes.get(component="displacement_x")
        # Union: {1,2,3,4} ∪ {4,5,6,7} = {1..7} (7 unique).
        assert slab.node_ids.tolist() == [1, 2, 3, 4, 5, 6, 7]
        r._reader.close()

    def test_first_partition_wins_for_boundary(
        self, two_partitions,
    ) -> None:
        """Node 4 lives in both partitions with identical disp values.
        After merge, only one column for node 4 remains, and its
        value matches what BOTH partitions wrote (here both wrote
        ``k * 100 + 4`` so the merge is unambiguous)."""
        part0, _ = two_partitions
        r = Results.from_mpco(str(part0))
        slab = r.nodes.get(component="displacement_x")
        col = list(slab.node_ids).index(4)
        # k=0 → 4, k=1 → 104, k=2 → 204
        np.testing.assert_array_almost_equal(
            slab.values[:, col], [4.0, 104.0, 204.0],
        )
        r._reader.close()

    def test_per_partition_only_node_picks_partition_value(
        self, two_partitions,
    ) -> None:
        # Node 1 is part-0-only; node 5 is part-1-only.
        part0, _ = two_partitions
        r = Results.from_mpco(str(part0))
        slab = r.nodes.get(component="displacement_x")
        for nid in (1, 5):
            col = list(slab.node_ids).index(nid)
            # k * 100 + nid encoding.
            np.testing.assert_array_almost_equal(
                slab.values[:, col],
                [float(nid), 100.0 + nid, 200.0 + nid],
            )
        r._reader.close()


# =====================================================================
# Element concatenation
# =====================================================================

class TestElementConcat:
    def test_gauss_concatenates_across_partitions(
        self, two_partitions,
    ) -> None:
        part0, _ = two_partitions
        r = Results.from_mpco(str(part0))
        slab = r.elements.gauss.get(component="stress_xx", time=0)
        # 1 element × 1 GP from each partition = 2 columns.
        assert slab.values.shape == (1, 2)
        # Both elements present.
        np.testing.assert_array_equal(slab.element_index, [10, 20])
        r._reader.close()

    def test_available_components_union(self, two_partitions) -> None:
        part0, _ = two_partitions
        r = Results.from_mpco(str(part0))
        sid = r._reader.stages()[0].id
        comps = r._reader.available_components(sid, ResultLevel.GAUSS)
        # 6 stress components from both partitions, identical layout.
        assert "stress_xx" in comps
        assert "stress_yy" in comps
        r._reader.close()


# =====================================================================
# FEM merging
# =====================================================================

class TestFemMerge:
    def test_fem_has_union_of_nodes(self, two_partitions) -> None:
        part0, _ = two_partitions
        r = Results.from_mpco(str(part0))
        fem = r.fem
        assert fem is not None
        assert fem.info.n_nodes == 7  # {1..7}
        np.testing.assert_array_equal(
            np.sort(fem.nodes.ids), np.arange(1, 8),
        )
        r._reader.close()

    def test_fem_concatenates_elements(self, two_partitions) -> None:
        part0, _ = two_partitions
        r = Results.from_mpco(str(part0))
        fem = r.fem
        # Both partitions' 1 tet each → 2 elements after merge.
        assert fem.info.n_elems == 2
        r._reader.close()

    def test_boundary_node_has_one_coord_row(
        self, two_partitions,
    ) -> None:
        # Node 4 is in both files; merged FEM should have only one row.
        part0, _ = two_partitions
        r = Results.from_mpco(str(part0))
        fem = r.fem
        ids = np.asarray(fem.nodes.ids, dtype=np.int64)
        assert int(np.sum(ids == 4)) == 1
        r._reader.close()


# =====================================================================
# Opt-out (merge_partitions=False)
# =====================================================================

class TestOptOut:
    def test_merge_partitions_false_keeps_single_file(
        self, two_partitions,
    ) -> None:
        part0, _ = two_partitions
        r = Results.from_mpco(str(part0), merge_partitions=False)
        # Only part-0's nodes / elements visible.
        slab = r.nodes.get(component="displacement_x")
        assert sorted(slab.node_ids.tolist()) == [1, 2, 3, 4]
        gslab = r.elements.gauss.get(component="stress_xx", time=0)
        assert gslab.values.shape == (1, 1)
        np.testing.assert_array_equal(gslab.element_index, [10])
        r._reader.close()


# =====================================================================
# Explicit list path
# =====================================================================

class TestExplicitList:
    def test_passing_list_uses_multi_reader(self, two_partitions) -> None:
        part0, part1 = two_partitions
        r = Results.from_mpco([part0, part1])
        # Class-name match — pytest can pick up duplicate package
        # paths when site-packages and PYTHONPATH both shadow each
        # other. See test_results_mpco_multi_real.py for the same
        # rationale.
        assert type(r._reader).__name__ == "MPCOMultiPartitionReader"
        r._reader.close()

    def test_passing_single_item_list_uses_single_reader(
        self, two_partitions,
    ) -> None:
        part0, _ = two_partitions
        r = Results.from_mpco([part0])
        assert type(r._reader).__name__ == "MPCOReader"
        r._reader.close()
