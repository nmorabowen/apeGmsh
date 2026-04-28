"""Real STKO multi-partition layered-shell MPCO read.

Validates the layers topology end-to-end against a 4-partition
ASDShellQ4 / ASDShellT3 model with LayeredShellFiberSection. The
fixture lives at ``Test_NLShell/Results.part-{0..3}.mpco`` in the
``STKO_to_python`` companion repo and exercises:

- Auto-detected 4-partition merge
- Multi-stage discovery (the analysis has staged construction with
  three ``MODEL_STAGE[…]`` groups)
- Layered shells written under ``section.fiber.<X>`` (the
  *unswapped* keyword — MPCO's
  ``utils::shell::isShellElementTag`` doesn't fire for ASDShellQ4
  here, so the layers reader walks both ``material.fiber.<X>`` and
  ``section.fiber.<X>`` aliases).
- META-driven multi-component layout (each layer carries 5 stress
  components, surfaced as ``fiber_stress_0`` … ``fiber_stress_4``).
"""
from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.readers._protocol import ResultLevel


_DEFAULT_EXAMPLES = Path(
    r"C:\Users\nmora\Github\STKO_to_python\stko_results_examples"
)


def _examples_dir() -> Path:
    override = os.environ.get("APEGMSH_STKO_EXAMPLES")
    return Path(override) if override else _DEFAULT_EXAMPLES


def _nl_shell_part0() -> Path:
    return (
        _examples_dir() / "Test_NLShell" / "Results.part-0.mpco"
    )


pytestmark = pytest.mark.skipif(
    not _nl_shell_part0().is_file(),
    reason=(
        "Real NL shell MPCO fixture not on disk. Set "
        "APEGMSH_STKO_EXAMPLES to the directory containing "
        "Test_NLShell/."
    ),
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def part0_path() -> Path:
    return _nl_shell_part0()


@pytest.fixture
def merged_results(part0_path: Path):
    r = Results.from_mpco(str(part0_path))   # auto-detect all 4 partitions
    yield r
    r._reader.close()


@pytest.fixture
def last_stage(merged_results):
    return merged_results._reader.stages()[-1]


# =====================================================================
# Auto-discovery + multi-partition + multi-stage
# =====================================================================

class TestStructure:
    def test_four_partitions_detected(self, merged_results) -> None:
        sid = merged_results._reader.stages()[0].id
        assert merged_results._reader.partitions(sid) == [
            "partition_0", "partition_1", "partition_2", "partition_3",
        ]

    def test_three_stages_listed(self, merged_results) -> None:
        stages = merged_results._reader.stages()
        names = [s.name for s in stages]
        assert names == ["MODEL_STAGE[1]", "MODEL_STAGE[2]", "MODEL_STAGE[3]"]


# =====================================================================
# Layer topology — META-driven 5-component canonicals
# =====================================================================

class TestLayerComponents:
    def test_available_layer_canonicals(
        self, merged_results, last_stage,
    ) -> None:
        """ASDShellQ4 + LayeredShellFiberSection writes 5 stress
        components per (surface_GP, layer). The reader surfaces all
        5 via ``fiber_stress_<i>`` canonicals."""
        comps = merged_results._reader.available_components(
            last_stage.id, ResultLevel.LAYERS,
        )
        # Both stress and strain, 5 indices each = 10 total.
        for i in range(5):
            assert f"fiber_stress_{i}" in comps
            assert f"fiber_strain_{i}" in comps

    def test_no_bare_fiber_stress_for_multi_component(
        self, merged_results, last_stage,
    ) -> None:
        """In a multi-component bucket the bare ``fiber_stress``
        canonical is NOT one of the available components — the
        index-suffixed names replace it."""
        comps = merged_results._reader.available_components(
            last_stage.id, ResultLevel.LAYERS,
        )
        assert "fiber_stress" not in comps


# =====================================================================
# Slab shape + value parity vs raw bucket
# =====================================================================

class TestSlabShape:
    def test_layer_slab_packed_correctly(
        self, merged_results, last_stage,
    ) -> None:
        slab = merged_results.elements.layers.get(
            component="fiber_stress_0",
            time=last_stage.n_steps - 1,
            stage=last_stage.id,
        )
        # n_unique_elements × n_sgp(=4 for ASDShellQ4) × n_layers(=7)
        # — number of elements depends on the model; just sanity-
        # check the modulus.
        n_unique = int(np.unique(slab.element_index).size)
        # Each shell contributes 4 sgp × 7 layers = 28 rows.
        assert slab.values.shape[1] == n_unique * 28

    def test_per_element_layer_stride(
        self, merged_results, last_stage,
    ) -> None:
        """Within one element, gp_index walks 0..3 and layer_index
        walks 0..6 within each sgp. Element-by-element packing."""
        slab = merged_results.elements.layers.get(
            component="fiber_stress_0",
            time=last_stage.n_steps - 1,
            stage=last_stage.id,
        )
        first_eid = int(slab.element_index[0])
        mask = slab.element_index == first_eid
        # 28 rows per element.
        assert int(mask.sum()) == 28
        # gp_index pattern: [0]*7, [1]*7, [2]*7, [3]*7.
        np.testing.assert_array_equal(
            slab.gp_index[mask], np.repeat([0, 1, 2, 3], 7),
        )
        # layer_index pattern: [0..6] tiled 4 times.
        np.testing.assert_array_equal(
            slab.layer_index[mask], np.tile(np.arange(7), 4),
        )

    def test_slab_value_matches_raw_bucket(
        self, merged_results, last_stage, part0_path: Path,
    ) -> None:
        """Pick element from part-0's bucket; spot-check that the
        first 5 stress_0 values match raw bucket columns 0, 5, 10,
        15, 20 (sgp 0, layers 0..4)."""
        slab = merged_results.elements.layers.get(
            component="fiber_stress_0",
            time=last_stage.n_steps - 1,
            stage=last_stage.id,
        )
        with h5py.File(part0_path, "r") as f:
            data_path = (
                f"{last_stage.name}/RESULTS/ON_ELEMENTS/"
                f"section.fiber.stress/203-ASDShellQ4[201:0:0]"
            )
            data_grp = f[f"{data_path}/DATA"]
            step_keys = sorted(
                data_grp.keys(),
                key=lambda s: int(s.split("_", 1)[1]),
            )
            raw_row = data_grp[step_keys[-1]][0]
            first_eid = int(f[f"{data_path}/ID"][...].flatten()[0])
        # Expected stress_0 columns are at stride 5 starting from 0.
        expected = raw_row[0::5][:7]  # First sgp's 7 layers.
        mask = slab.element_index == first_eid
        np.testing.assert_array_almost_equal(
            slab.values[0, mask][:7], expected,
        )


# =====================================================================
# Component independence — different stress components are distinct
# =====================================================================

class TestComponentIndependence:
    def test_stress_components_differ(
        self, merged_results, last_stage,
    ) -> None:
        slabs = [
            merged_results.elements.layers.get(
                component=f"fiber_stress_{i}",
                time=last_stage.n_steps - 1,
                stage=last_stage.id,
            )
            for i in range(5)
        ]
        # Each component should have at least some non-trivial values
        # (the model has converged with damage; not every column is
        # non-zero, but at least one stress component should be).
        any_nonzero = any(np.any(s.values != 0.0) for s in slabs)
        assert any_nonzero
        # And the columns shouldn't all be identical to component 0.
        for i in range(1, 5):
            if np.any(slabs[i].values != 0.0) and np.any(slabs[0].values != 0.0):
                if not np.allclose(slabs[i].values, slabs[0].values):
                    return
        pytest.fail("All stress components are identical — META decoding suspect.")


# =====================================================================
# Multi-partition layer merge
# =====================================================================

class TestPartitionMerge:
    def test_layer_slab_spans_all_partitions(
        self, merged_results, last_stage, part0_path: Path,
    ) -> None:
        """Sum of per-partition unique element counts must equal the
        merged slab's unique element count (elements are disjoint
        across partitions in OpenSees parallel)."""
        merged_slab = merged_results.elements.layers.get(
            component="fiber_stress_0",
            time=last_stage.n_steps - 1,
            stage=last_stage.id,
        )
        merged_count = int(np.unique(merged_slab.element_index).size)

        per_partition_total = 0
        for i in range(4):
            single = Results.from_mpco(
                str(part0_path.parent / f"Results.part-{i}.mpco"),
                merge_partitions=False,
            )
            single_slab = single.elements.layers.get(
                component="fiber_stress_0",
                time=last_stage.n_steps - 1,
                stage=last_stage.id,
            )
            per_partition_total += int(
                np.unique(single_slab.element_index).size,
            )
            single._reader.close()

        assert merged_count == per_partition_total
