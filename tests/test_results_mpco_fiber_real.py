"""Phase 11c Stage 4 — real MPCO fiber-section reads.

Drives ``Results.from_mpco(...)`` against a real STKO/OpenSees output
file produced by a live nonlinear solid+beam analysis, in this case
the partitioned ``solid_partition_example`` fixture from the
``STKO_to_python`` companion repo.

Skipped when the fixture file is not on disk so the test stays
portable. The path is computed relative to a fixed base; override it
by setting the ``APEGMSH_STKO_EXAMPLES`` environment variable to the
``stko_results_examples`` directory.

The fixture holds a hybrid model:

- 56-Brick continuum elements (``stresses`` / ``material.stress``
  buckets — Phase 11a continuum, not exercised here).
- 64-DispBeamColumn3d displacement-based beam-columns with three
  distinct fiber sections (SECTION_1, SECTION_2, SECTION_3 — all
  6-fiber rectangular cross-sections of varying width). 402
  beam-columns total, 2 IPs each, 1667 recorded time steps.

Validates:

1. The reader discovers ``fiber_stress`` and ``fiber_strain`` at the
   ``fibers`` topology and not at ``layers``.
2. Slab shapes match ``402 elements × 2 IPs × 6 fibers = 4824`` rows.
3. Per-element section lookup picks the right ``(y, z, area)`` for
   each of the three sections.
4. Slab values at a chosen time-step match the raw bucket
   ``DATA/STEP_<k>`` row exactly (no off-by-one in the
   IP × fiber packing).
5. Element / GP filters narrow the slab as expected.
"""
from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.readers._protocol import ResultLevel


# =====================================================================
# Fixture path resolution
# =====================================================================

_DEFAULT_EXAMPLES = Path(
    r"C:\Users\nmora\Github\STKO_to_python\stko_results_examples"
)


def _examples_dir() -> Path:
    override = os.environ.get("APEGMSH_STKO_EXAMPLES")
    return Path(override) if override else _DEFAULT_EXAMPLES


def _solid_partition_part0() -> Path:
    return (
        _examples_dir()
        / "solid_partition_example"
        / "Recorder.part-0.mpco"
    )


def _has_fixture() -> bool:
    return _solid_partition_part0().is_file()


pytestmark = pytest.mark.skipif(
    not _has_fixture(),
    reason=(
        "Real STKO MPCO fixture not on disk. Set APEGMSH_STKO_EXAMPLES "
        "to the directory containing solid_partition_example/."
    ),
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def mpco_path() -> Path:
    return _solid_partition_part0()


@pytest.fixture
def results(mpco_path: Path):
    # This test file deliberately exercises single-partition behaviour
    # against ``Recorder.part-0.mpco`` only. The multi-partition merge
    # path has its own dedicated test file
    # (``test_results_mpco_multi_real.py``) and the auto-detect path
    # would silently double the element / node counts here.
    r = Results.from_mpco(str(mpco_path), merge_partitions=False)
    yield r
    r._reader.close()


# =====================================================================
# Stage / discovery
# =====================================================================

class TestStageDiscovery:
    def test_one_stage(self, results) -> None:
        stages = results._reader.stages()
        assert len(stages) == 1
        assert stages[0].name == "MODEL_STAGE[1]"

    def test_time_vector_size(self, results) -> None:
        sid = results._reader.stages()[0].id
        time = results._reader.time_vector(sid)
        assert time.size == 1667
        # First / last bracketed
        assert time[0] >= 0.0
        assert time[-1] > time[0]

    def test_available_fiber_components(self, results) -> None:
        sid = results._reader.stages()[0].id
        comps = results._reader.available_components(sid, ResultLevel.FIBERS)
        assert sorted(comps) == ["fiber_strain", "fiber_stress"]

    def test_no_layers_in_solid_model(self, results) -> None:
        sid = results._reader.stages()[0].id
        comps = results._reader.available_components(sid, ResultLevel.LAYERS)
        assert comps == []  # No shell layers in this model.


# =====================================================================
# Slab shape + packing
# =====================================================================

class TestFiberSlabShape:
    def test_full_slab_shape_first_step(self, results) -> None:
        slab = results.elements.fibers.get(
            component="fiber_stress", time=0,
        )
        # 402 DispBeamColumn3d × 2 IPs × 6 fibers.
        assert slab.values.shape == (1, 4824)
        assert slab.element_index.size == 4824
        assert slab.gp_index.size == 4824
        assert slab.y.size == 4824
        assert slab.material_tag.size == 4824

    def test_first_step_is_all_zero(self, results) -> None:
        # Static initial step should have zero stresses.
        slab = results.elements.fibers.get(
            component="fiber_stress", time=0,
        )
        assert np.count_nonzero(slab.values) == 0

    def test_late_step_has_nonzero_stress(self, results) -> None:
        slab = results.elements.fibers.get(
            component="fiber_stress", time=1500,
        )
        assert np.count_nonzero(slab.values) > 4000  # Mostly nonzero.
        # Realistic stress range — concrete/steel mixed fiber section.
        assert slab.values.min() < -10.0
        assert slab.values.max() > 10.0

    def test_unique_element_count_matches_bucket(self, results) -> None:
        slab = results.elements.fibers.get(
            component="fiber_stress", time=0,
        )
        assert np.unique(slab.element_index).size == 402

    def test_gp_index_pattern(self, results) -> None:
        # Each element gets gp_index = [0]*6 + [1]*6 (n_IP=2, n_fibers=6)
        # tiled for all 402 elements → first 12 are [0,0,0,0,0,0,1,1,1,1,1,1].
        slab = results.elements.fibers.get(
            component="fiber_stress", time=0,
        )
        np.testing.assert_array_equal(
            slab.gp_index[:12], [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        )


# =====================================================================
# Per-element section lookup (the multi-section bucket case)
# =====================================================================

class TestPerElementSection:
    def test_three_distinct_y_profiles(self, results) -> None:
        """The bucket spans 3 sections with widths 3.175, 4.775, 6.35.
        After per-element lookup the slab should expose all three
        magnitudes, not just one."""
        slab = results.elements.fibers.get(
            component="fiber_stress", time=0,
        )
        ys = np.unique(np.abs(np.round(slab.y, 6)))
        # Each section has 4 fibers at +/- y_outer plus 2 at y~0 → 3 distinct
        # outer-edge y magnitudes, plus near-zero inner. Allow a small slop:
        outer = ys[ys > 1.0]
        assert sorted(round(v, 3) for v in outer) == [3.175, 4.775, 6.35]

    def test_element_in_section1_has_section1_geometry(
        self, results, mpco_path: Path,
    ) -> None:
        # SECTION_1's first assigned element gets SECTION_1's y/z/area.
        with h5py.File(mpco_path, "r") as f:
            sa = f["MODEL_STAGE[1]/MODEL/SECTION_ASSIGNMENTS"]
            sec1_assign = sa["SECTION_1[UnkownClassType]/ASSIGNMENT"][...]
            sec1_fdata = sa["SECTION_1[UnkownClassType]/FIBER_DATA"][...]
        eid = int(sec1_assign[0, 0])

        slab = results.elements.fibers.get(
            component="fiber_stress", time=0,
        )
        mask = slab.element_index == eid
        # First IP, 6 fibers — matches SECTION_1 column 0 (y).
        eid_y = slab.y[mask][:6]
        np.testing.assert_array_almost_equal(eid_y, sec1_fdata[:, 0])
        eid_z = slab.z[mask][:6]
        np.testing.assert_array_almost_equal(eid_z, sec1_fdata[:, 1])
        eid_area = slab.area[mask][:6]
        np.testing.assert_array_almost_equal(eid_area, sec1_fdata[:, 2])

    def test_three_sections_used_correctly(
        self, results, mpco_path: Path,
    ) -> None:
        """One representative element per section: each one's y / z / area
        must come from its own section, not from a bucket-wide
        stand-in."""
        with h5py.File(mpco_path, "r") as f:
            sa = f["MODEL_STAGE[1]/MODEL/SECTION_ASSIGNMENTS"]
            cases = []
            for sec_name in [
                "SECTION_1[UnkownClassType]",
                "SECTION_2[UnkownClassType]",
                "SECTION_3[UnkownClassType]",
            ]:
                eid = int(sa[f"{sec_name}/ASSIGNMENT"][0, 0])
                fdata = sa[f"{sec_name}/FIBER_DATA"][...]
                cases.append((eid, fdata))

        slab = results.elements.fibers.get(
            component="fiber_stress", time=0,
        )
        for eid, fdata in cases:
            mask = slab.element_index == eid
            np.testing.assert_array_almost_equal(
                slab.y[mask][:6], fdata[:, 0],
                err_msg=f"y mismatch for element {eid}",
            )

    def test_material_tag_is_uniform(self, results) -> None:
        # All fibers in this model use material 1.
        slab = results.elements.fibers.get(
            component="fiber_stress", time=0,
        )
        assert np.unique(slab.material_tag).tolist() == [1]


# =====================================================================
# Value packing — slab matches raw bucket
# =====================================================================

class TestSlabValuesMatchBucket:
    def test_slab_values_match_raw_bucket_at_time_index(
        self, results, mpco_path: Path,
    ) -> None:
        """For element 8 (first in bucket) and t_idx=1500, slab cols
        0..11 must equal the bucket's 12 columns at the corresponding
        sparse step exactly."""
        slab = results.elements.fibers.get(
            component="fiber_stress", time=1500,
        )

        with h5py.File(mpco_path, "r") as f:
            data_grp = f[
                "MODEL_STAGE[1]/RESULTS/ON_ELEMENTS/section.fiber.stress/"
                "64-DispBeamColumn3d[1000:1:0]/DATA"
            ]
            step_keys = sorted(
                data_grp.keys(),
                key=lambda s: int(s.split("_", 1)[1]),
            )
            raw_row = data_grp[step_keys[1500]][0]  # First element row.

        np.testing.assert_array_almost_equal(slab.values[0, :12], raw_row)

    def test_strain_reads_too(self, results) -> None:
        slab = results.elements.fibers.get(
            component="fiber_strain", time=1500,
        )
        assert slab.values.shape == (1, 4824)
        # Strains in nonlinear range — small but nonzero.
        assert np.count_nonzero(slab.values) > 4000


# =====================================================================
# Filtering
# =====================================================================

class TestFilters:
    def test_element_id_filter_narrows_slab(self, results, mpco_path: Path) -> None:
        # Pick three elements known to live in different sections.
        with h5py.File(mpco_path, "r") as f:
            sa = f["MODEL_STAGE[1]/MODEL/SECTION_ASSIGNMENTS"]
            eids = [
                int(sa["SECTION_1[UnkownClassType]/ASSIGNMENT"][0, 0]),
                int(sa["SECTION_2[UnkownClassType]/ASSIGNMENT"][0, 0]),
                int(sa["SECTION_3[UnkownClassType]/ASSIGNMENT"][0, 0]),
            ]
        slab = results.elements.fibers.get(
            component="fiber_stress", time=0, ids=eids,
        )
        # 3 elements × 2 IPs × 6 fibers = 36 cols.
        assert slab.values.shape == (1, 36)
        np.testing.assert_array_equal(
            np.unique(slab.element_index), np.sort(np.array(eids)),
        )

    def test_gp_filter_picks_one_ip(self, results) -> None:
        slab = results.elements.fibers.get(
            component="fiber_stress", time=0, gp_indices=[1],
        )
        # 402 elements × 1 IP × 6 fibers = 2412 cols.
        assert slab.values.shape == (1, 2412)
        np.testing.assert_array_equal(
            np.unique(slab.gp_index), np.array([1]),
        )

    def test_combined_filter(self, results, mpco_path: Path) -> None:
        with h5py.File(mpco_path, "r") as f:
            sa = f["MODEL_STAGE[1]/MODEL/SECTION_ASSIGNMENTS"]
            eid_sec2 = int(
                sa["SECTION_2[UnkownClassType]/ASSIGNMENT"][0, 0],
            )
        slab = results.elements.fibers.get(
            component="fiber_stress",
            time=1500, ids=[eid_sec2], gp_indices=[0],
        )
        # 1 elem × 1 GP × 6 fibers = 6 cols.
        assert slab.values.shape == (1, 6)
        # Verify it matches the raw bucket: element eid_sec2, IP 0, 6 fibers.
        with h5py.File(mpco_path, "r") as f:
            ids_arr = f[
                "MODEL_STAGE[1]/RESULTS/ON_ELEMENTS/section.fiber.stress/"
                "64-DispBeamColumn3d[1000:1:0]/ID"
            ][...].flatten().astype(np.int64)
            row_idx = int(np.where(ids_arr == eid_sec2)[0][0])
            data_grp = f[
                "MODEL_STAGE[1]/RESULTS/ON_ELEMENTS/section.fiber.stress/"
                "64-DispBeamColumn3d[1000:1:0]/DATA"
            ]
            step_keys = sorted(
                data_grp.keys(),
                key=lambda s: int(s.split("_", 1)[1]),
            )
            raw_row = data_grp[step_keys[1500]][row_idx]
        # First 6 columns of bucket row = IP 0, 6 fibers.
        np.testing.assert_array_almost_equal(slab.values[0], raw_row[:6])


# =====================================================================
# Cross-topology smoke
# =====================================================================

class TestCrossTopologySmoke:
    def test_line_stations_also_works(self, results) -> None:
        """Phase 11b coverage check — same file exposes section.force
        on the same DispBeamColumn3d elements."""
        slab = results.elements.line_stations.get(
            component="axial_force", time=1500,
        )
        # 402 elements × 2 IPs = 804 stations.
        assert slab.values.shape[1] == 804

    def test_material_stress_alias_resolves(self, results) -> None:
        """The Brick continuum results live under ``material.stress`` /
        ``material.strain`` (modern MPCO keyword). Phase 11c added an
        alias so the gauss reader still finds them."""
        sid = results._reader.stages()[0].id
        comps = results._reader.available_components(sid, ResultLevel.GAUSS)
        # All 12 (stress + strain × 6 indices each).
        assert "stress_xx" in comps
        assert "strain_xx" in comps
        # Read should work and return non-trivial values at a late step.
        slab = results.elements.gauss.get(
            component="stress_xx", time=1500,
        )
        assert slab.values.size > 0
        assert np.count_nonzero(slab.values) > 0

    def test_material_state_two_component_damage(self, results) -> None:
        """ASDConcrete in this file emits ``d+,d-`` per GP. The
        material-state path resolves these via META and surfaces them
        as ``damage_tension`` / ``damage_compression`` canonicals."""
        sid = results._reader.stages()[0].id
        comps = results._reader.available_components(sid, ResultLevel.GAUSS)
        assert "damage_tension" in comps
        assert "damage_compression" in comps
        assert "equivalent_plastic_strain_tension" in comps
        assert "equivalent_plastic_strain_compression" in comps

        slab_t = results.elements.gauss.get(
            component="damage_tension", time=1500,
        )
        # 33 Bricks × 8 GPs (only some bricks track damage; others
        # are linear-elastic). Just check we read non-trivial values.
        assert slab_t.values.size > 0
        assert slab_t.values.min() >= 0.0
        # Damage saturates near 1.0 in late nonlinear steps.
        assert slab_t.values.max() > 0.9

    def test_damage_tension_compression_distinct(self, results) -> None:
        """Tension and compression damage are separate canonicals; the
        slabs are not identical (they probe different META columns)."""
        slab_t = results.elements.gauss.get(
            component="damage_tension", time=1500,
        )
        slab_c = results.elements.gauss.get(
            component="damage_compression", time=1500,
        )
        assert slab_t.values.shape == slab_c.values.shape
        # The two columns should differ in a damage-plasticity model.
        assert not np.allclose(slab_t.values, slab_c.values)
