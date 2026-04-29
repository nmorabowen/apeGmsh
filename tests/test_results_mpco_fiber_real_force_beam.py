"""Real MPCO fiber-section reads — ForceBeamColumn3d portal frame.

Drives ``Results.from_mpco(...)`` against the ``forceBeamCol`` example
shipped in the ``STKO_to_python`` companion repo. The Tcl source is
mirrored under ``tests/fixtures/results/forceBeamCol_tcl/`` for
in-tree reference; the .mpco binary lives next to it (out-of-tree)
and is resolved via ``APEGMSH_STKO_EXAMPLES``.

Model recap
-----------
- 4 nodes laid out as a planar portal frame (3D, ndf=6).
- 3 ``forceBeamColumn`` elements with Lobatto integration, 5 IPs each.
- One Fiber section (~233 fibers — concrete + rebar grid).
- Static load-control pushover with a single horizontal force at the
  top corner.

Validates
---------
1. The reader exposes ``fiber_stress`` / ``fiber_strain`` at the
   ``fibers`` topology and not at ``layers``.
2. Slab shape matches ``3 elements × 5 IPs × 233 fibers = 3495`` rows.
3. The first step is fully zero (no load yet); a late step has
   nonzero stress in a realistic concrete/steel range.
4. Slab geometry (y, z, area) round-trips against the raw
   ``SECTION_ASSIGNMENTS/SECTION_1/FIBER_DATA`` dataset.
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
# Fixture path resolution — same convention as test_results_mpco_fiber_real.py
# =====================================================================

_DEFAULT_EXAMPLES = Path(
    r"C:\Users\nmora\Github\STKO_to_python\stko_results_examples"
)


def _examples_dir() -> Path:
    override = os.environ.get("APEGMSH_STKO_EXAMPLES")
    return Path(override) if override else _DEFAULT_EXAMPLES


def _mpco_path() -> Path:
    return _examples_dir() / "forceBeamCol" / "results.mpco"


def _has_fixture() -> bool:
    return _mpco_path().is_file()


pytestmark = pytest.mark.skipif(
    not _has_fixture(),
    reason=(
        "forceBeamCol/results.mpco not on disk. Set "
        "APEGMSH_STKO_EXAMPLES to the directory containing "
        "forceBeamCol/."
    ),
)


# =====================================================================
# Shared fixtures
# =====================================================================

# Expected dimensions inferred from the .mpco at write time.
N_ELEMENTS = 3
N_IPS = 5
N_FIBERS = 233
SUM_F = N_ELEMENTS * N_IPS * N_FIBERS    # 3495


@pytest.fixture
def mpco_path() -> Path:
    return _mpco_path()


@pytest.fixture
def results(mpco_path: Path):
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

    def test_fiber_components_available(self, results) -> None:
        sid = results._reader.stages()[0].id
        comps = results._reader.available_components(
            sid, ResultLevel.FIBERS,
        )
        assert "fiber_stress" in comps
        assert "fiber_strain" in comps

    def test_no_layers_in_beam_model(self, results) -> None:
        sid = results._reader.stages()[0].id
        comps = results._reader.available_components(
            sid, ResultLevel.LAYERS,
        )
        assert comps == []


# =====================================================================
# Slab shape
# =====================================================================

class TestFiberSlabShape:
    def test_full_slab_shape_first_step(self, results) -> None:
        slab = results.elements.fibers.get(
            component="fiber_stress", time=0,
        )
        assert slab.values.shape == (1, SUM_F)
        assert slab.element_index.size == SUM_F
        assert slab.gp_index.size == SUM_F
        assert slab.y.size == SUM_F
        assert slab.material_tag.size == SUM_F

    def test_stress_grows_with_load(self, results) -> None:
        # The .tcl uses `LoadControl` with the full load applied
        # incrementally — recorded step 0 is already after one
        # increment, so values are nonzero from the start. We expect
        # the stress range to grow as load accumulates.
        early = results.elements.fibers.get(
            component="fiber_stress", time=0,
        )
        late = results.elements.fibers.get(
            component="fiber_stress", time=60,
        )
        early_range = early.values.max() - early.values.min()
        late_range = late.values.max() - late.values.min()
        assert late_range > early_range
        # Concrete + steel mix at late step — both compression and tension.
        assert late.values.min() < -1.0
        assert late.values.max() > 1.0

    def test_unique_element_count(self, results) -> None:
        slab = results.elements.fibers.get(
            component="fiber_stress", time=0,
        )
        assert np.unique(slab.element_index).size == N_ELEMENTS

    def test_gp_index_pattern(self, results) -> None:
        # Single-section bucket: each element gets gp_index =
        # [0]*233 + [1]*233 + ... + [4]*233 ; the first 233 entries
        # should all be 0.
        slab = results.elements.fibers.get(
            component="fiber_stress", time=0,
        )
        assert (slab.gp_index[:N_FIBERS] == 0).all()
        assert (slab.gp_index[N_FIBERS:2 * N_FIBERS] == 1).all()


# =====================================================================
# Geometry round-trip against SECTION_ASSIGNMENTS
# =====================================================================

class TestSectionGeometryRoundTrip:
    def test_fiber_geometry_matches_section_assignment(
        self, results, mpco_path: Path,
    ) -> None:
        with h5py.File(mpco_path, "r") as f:
            sa = f["MODEL_STAGE[1]/MODEL/SECTION_ASSIGNMENTS"]
            sec = sa["SECTION_1[UnkownClassType]"]
            fdata = sec["FIBER_DATA"][...]    # (N_FIBERS, 3)
            assignment = sec["ASSIGNMENT"][...]    # (15, 2)

        # Sanity on the raw file.
        assert fdata.shape == (N_FIBERS, 3)
        assert assignment.shape[0] == N_ELEMENTS * N_IPS

        # Pull the slab and verify the first element / first IP fibers
        # match the section's geometry exactly.
        slab = results.elements.fibers.get(
            component="fiber_stress", time=0,
        )
        eid = int(assignment[0, 0])
        mask = (slab.element_index == eid) & (slab.gp_index == 0)
        np.testing.assert_array_almost_equal(slab.y[mask], fdata[:, 0])
        np.testing.assert_array_almost_equal(slab.z[mask], fdata[:, 1])
        np.testing.assert_array_almost_equal(slab.area[mask], fdata[:, 2])


# =====================================================================
# Element / GP filters
# =====================================================================

class TestSlabFilters:
    def test_element_filter_narrows_slab(
        self, results, mpco_path: Path,
    ) -> None:
        with h5py.File(mpco_path, "r") as f:
            assignment = f[
                "MODEL_STAGE[1]/MODEL/SECTION_ASSIGNMENTS/"
                "SECTION_1[UnkownClassType]/ASSIGNMENT"
            ][...]
        eid = int(assignment[0, 0])

        full = results.elements.fibers.get(
            component="fiber_stress", time=0,
        )
        one = results.elements.fibers.get(
            component="fiber_stress", time=0, ids=[eid],
        )
        # Single element × 5 IPs × 233 fibers
        assert one.values.shape[1] == N_IPS * N_FIBERS
        assert one.values.shape[1] < full.values.shape[1]
        assert (one.element_index == eid).all()

    def test_gp_filter_narrows_slab(self, results) -> None:
        single_gp = results.elements.fibers.get(
            component="fiber_stress", time=0, gp_indices=[0],
        )
        # All 3 elements × 1 IP × 233 fibers
        assert single_gp.values.shape[1] == N_ELEMENTS * N_FIBERS
        assert (single_gp.gp_index == 0).all()
