"""Real STKO multi-partition MPCO read.

Auto-discovers the two partition files of the
``solid_partition_example`` fixture and validates that
``Results.from_mpco`` merges them transparently.

Skipped when the fixture is not on disk. Override the location via
``APEGMSH_STKO_EXAMPLES``.
"""
from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.readers._mpco_multi import (
    MPCOMultiPartitionReader, discover_partition_files,
)
from apeGmsh.results.readers._protocol import ResultLevel


_DEFAULT_EXAMPLES = Path(
    r"C:\Users\nmora\Github\STKO_to_python\stko_results_examples"
)


def _examples_dir() -> Path:
    override = os.environ.get("APEGMSH_STKO_EXAMPLES")
    return Path(override) if override else _DEFAULT_EXAMPLES


def _solid_partition_part0() -> Path:
    return (
        _examples_dir() / "solid_partition_example" / "Recorder.part-0.mpco"
    )


pytestmark = pytest.mark.skipif(
    not _solid_partition_part0().is_file(),
    reason=(
        "Real STKO MPCO fixture not on disk. Set APEGMSH_STKO_EXAMPLES "
        "to the directory containing solid_partition_example/."
    ),
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def part0_path() -> Path:
    return _solid_partition_part0()


@pytest.fixture
def merged_results(part0_path: Path):
    r = Results.from_mpco(str(part0_path))   # auto-detect both partitions
    yield r
    r._reader.close()


# Per-partition counts taken from inspection (see Phase 11c notes):
#   part-0: 854 nodes, 220 Bricks, 402 DispBeamColumn3d
#   part-1: 1001 nodes, 427 Bricks, 310 DispBeamColumn3d
# Boundary node overlap: 854 + 1001 − 1720 = 135 shared nodes.
P0_NODES, P1_NODES, MERGED_UNIQUE_NODES = 854, 1001, 1720
P0_BRICKS, P1_BRICKS = 220, 427
P0_BEAMS,  P1_BEAMS  = 402, 310


# =====================================================================
# Auto-discovery
# =====================================================================

class TestAutoDiscovery:
    def test_finds_both_partition_files(self, part0_path: Path) -> None:
        found = discover_partition_files(part0_path)
        assert len(found) == 2
        assert found[0].name == "Recorder.part-0.mpco"
        assert found[1].name == "Recorder.part-1.mpco"

    def test_from_mpco_returns_multi_reader(self, merged_results) -> None:
        # Match by class name rather than ``isinstance`` — pytest sessions
        # that have the package both installed (site-packages) and on
        # PYTHONPATH (worktree) can end up with two distinct class
        # objects for the same logical type.
        assert type(merged_results._reader).__name__ == "MPCOMultiPartitionReader"

    def test_two_partitions_listed(self, merged_results) -> None:
        sid = merged_results._reader.stages()[0].id
        assert merged_results._reader.partitions(sid) == [
            "partition_0", "partition_1",
        ]


# =====================================================================
# Stage / time consistency
# =====================================================================

class TestStageConsistency:
    def test_one_stage(self, merged_results) -> None:
        stages = merged_results._reader.stages()
        assert len(stages) == 1
        assert stages[0].name == "MODEL_STAGE[1]"

    def test_time_vector_unchanged(self, merged_results) -> None:
        sid = merged_results._reader.stages()[0].id
        time = merged_results._reader.time_vector(sid)
        assert time.size == 1667


# =====================================================================
# Node merge
# =====================================================================

class TestNodeMerge:
    def test_total_node_count_after_dedup(self, merged_results) -> None:
        slab = merged_results.nodes.get(component="displacement_x")
        # 1720 unique node IDs after boundary-node deduplication.
        assert slab.node_ids.size == MERGED_UNIQUE_NODES
        assert np.unique(slab.node_ids).size == MERGED_UNIQUE_NODES

    def test_boundary_nodes_count_matches_overlap(
        self, merged_results, part0_path: Path,
    ) -> None:
        # Per-partition node IDs from raw files.
        with h5py.File(part0_path, "r") as f:
            ids0 = f[
                "MODEL_STAGE[1]/MODEL/NODES/ID"
            ][...].flatten().astype(np.int64)
        part1 = part0_path.parent / "Recorder.part-1.mpco"
        with h5py.File(part1, "r") as f:
            ids1 = f[
                "MODEL_STAGE[1]/MODEL/NODES/ID"
            ][...].flatten().astype(np.int64)
        union_size = int(np.union1d(ids0, ids1).size)
        slab = merged_results.nodes.get(component="displacement_x")
        assert slab.node_ids.size == union_size

    def test_no_nan_in_merged_displacement(self, merged_results) -> None:
        # Every node was written to by at least one partition, so no NaN.
        slab = merged_results.nodes.get(component="displacement_x")
        assert not np.isnan(slab.values).any()


# =====================================================================
# Element concatenation
# =====================================================================

class TestElementConcat:
    def test_gauss_brick_count(self, merged_results) -> None:
        slab = merged_results.elements.gauss.get(
            component="stress_xx", time=1500,
        )
        # 220 + 427 = 647 Bricks × 8 GPs (Hex_GL_2) = 5176 columns.
        assert np.unique(slab.element_index).size == P0_BRICKS + P1_BRICKS
        assert slab.values.shape[1] == (P0_BRICKS + P1_BRICKS) * 8

    def test_line_stations_beam_count(self, merged_results) -> None:
        slab = merged_results.elements.line_stations.get(
            component="axial_force", time=1500,
        )
        # 402 + 310 = 712 beams × 2 IPs = 1424 stations.
        assert np.unique(slab.element_index).size == P0_BEAMS + P1_BEAMS
        assert slab.values.shape[1] == (P0_BEAMS + P1_BEAMS) * 2

    def test_fiber_count(self, merged_results) -> None:
        slab = merged_results.elements.fibers.get(
            component="fiber_stress", time=1500,
        )
        # 712 beams × 2 IPs × 6 fibers = 8544 fiber rows.
        assert slab.values.shape[1] == (P0_BEAMS + P1_BEAMS) * 2 * 6


# =====================================================================
# FEM merge
# =====================================================================

class TestFemMerge:
    def test_fem_node_count(self, merged_results) -> None:
        fem = merged_results.fem
        assert fem is not None
        assert fem.info.n_nodes == MERGED_UNIQUE_NODES

    def test_fem_element_count(self, merged_results) -> None:
        fem = merged_results.fem
        assert fem.info.n_elems == (
            P0_BRICKS + P1_BRICKS + P0_BEAMS + P1_BEAMS
        )


# =====================================================================
# Cross-partition value parity — element 8 stress matches single-file read
# =====================================================================

class TestCrossPartitionParity:
    def test_element_in_part0_matches_single_file_read(
        self, part0_path: Path,
    ) -> None:
        """Element 8 lives in part-0. Reading via single-file-only
        (``merge_partitions=False``) and via the merged reader must
        produce the same fiber-stress values for that element."""
        single = Results.from_mpco(
            str(part0_path), merge_partitions=False,
        )
        merged = Results.from_mpco(str(part0_path))   # auto-merge

        s_slab = single.elements.fibers.get(
            component="fiber_stress", time=1500, ids=[8],
        )
        m_slab = merged.elements.fibers.get(
            component="fiber_stress", time=1500, ids=[8],
        )
        np.testing.assert_array_almost_equal(
            s_slab.values, m_slab.values,
        )
        single._reader.close()
        merged._reader.close()
