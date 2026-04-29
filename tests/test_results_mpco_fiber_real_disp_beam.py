"""Real MPCO fiber-section reads — DispBeamColumn3d frame.

Drives ``Results.from_mpco(...)`` against the ``dispBeamCol`` example
shipped in the ``STKO_to_python`` companion repo. The Tcl source is
mirrored under ``tests/fixtures/results/dispBeamCol_tcl/`` for
in-tree reference; the .mpco binary lives next to it (out-of-tree)
and is resolved via ``APEGMSH_STKO_EXAMPLES``.

Model recap
-----------
- Multi-bay frame, 3D, ndf=6.
- 48 ``dispBeamColumn`` elements, each with Lobatto integration and
  5 IPs (one section per IP, all using the same Fiber section).
- 233-fiber concrete + rebar cross-section, identical to the
  ``forceBeamCol`` example.
- 200 saved time steps under load-control pushover.

Validates
---------
1. ``fiber_stress`` / ``fiber_strain`` are exposed at the ``fibers``
   topology (not ``layers``).
2. Slab shape: ``48 × 5 × 233 = 55920`` fiber rows.
3. Stress range grows monotonically with load.
4. Per-element / per-GP filtering narrows the slab as expected.
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


def _mpco_path() -> Path:
    return _examples_dir() / "dispBeamCol" / "results.mpco"


def _has_fixture() -> bool:
    return _mpco_path().is_file()


pytestmark = pytest.mark.skipif(
    not _has_fixture(),
    reason=(
        "dispBeamCol/results.mpco not on disk. Set "
        "APEGMSH_STKO_EXAMPLES to the directory containing "
        "dispBeamCol/."
    ),
)


# Expected dimensions inferred from the .mpco at write time.
N_ELEMENTS = 48
N_IPS = 5
N_FIBERS = 233
SUM_F = N_ELEMENTS * N_IPS * N_FIBERS    # 55920


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
        assert len(results._reader.stages()) == 1

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

    def test_unique_element_count(self, results) -> None:
        slab = results.elements.fibers.get(
            component="fiber_stress", time=0,
        )
        assert np.unique(slab.element_index).size == N_ELEMENTS

    def test_stress_grows_with_load(self, results) -> None:
        early = results.elements.fibers.get(
            component="fiber_stress", time=0,
        )
        late = results.elements.fibers.get(
            component="fiber_stress", time=180,
        )
        early_range = early.values.max() - early.values.min()
        late_range = late.values.max() - late.values.min()
        assert late_range > early_range
        assert late.values.min() < -1.0
        assert late.values.max() > 1.0


# =====================================================================
# Geometry round-trip
# =====================================================================

class TestSectionGeometryRoundTrip:
    def test_geometry_matches_section_assignment(
        self, results, mpco_path: Path,
    ) -> None:
        with h5py.File(mpco_path, "r") as f:
            sa = f["MODEL_STAGE[1]/MODEL/SECTION_ASSIGNMENTS"]
            sec = sa["SECTION_1[UnkownClassType]"]
            fdata = sec["FIBER_DATA"][...]
            assignment = sec["ASSIGNMENT"][...]

        assert fdata.shape == (N_FIBERS, 3)
        assert assignment.shape[0] == N_ELEMENTS * N_IPS

        slab = results.elements.fibers.get(
            component="fiber_stress", time=0,
        )
        eid = int(assignment[0, 0])
        mask = (slab.element_index == eid) & (slab.gp_index == 0)
        np.testing.assert_array_almost_equal(slab.y[mask], fdata[:, 0])


# =====================================================================
# Element / GP filters
# =====================================================================

class TestSlabFilters:
    def test_element_filter(self, results, mpco_path: Path) -> None:
        with h5py.File(mpco_path, "r") as f:
            assignment = f[
                "MODEL_STAGE[1]/MODEL/SECTION_ASSIGNMENTS/"
                "SECTION_1[UnkownClassType]/ASSIGNMENT"
            ][...]
        eid = int(assignment[0, 0])
        one = results.elements.fibers.get(
            component="fiber_stress", time=0, ids=[eid],
        )
        assert one.values.shape[1] == N_IPS * N_FIBERS
        assert (one.element_index == eid).all()

    def test_gp_filter(self, results) -> None:
        single_gp = results.elements.fibers.get(
            component="fiber_stress", time=0, gp_indices=[0],
        )
        assert single_gp.values.shape[1] == N_ELEMENTS * N_FIBERS
        assert (single_gp.gp_index == 0).all()
