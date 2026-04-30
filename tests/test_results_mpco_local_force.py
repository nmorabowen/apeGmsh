"""Coverage for the MPCO ``localForce`` line-station path.

ElasticBeam2d/3d are stiffness-formulated and write 12 / 6 end-force
components in the element local frame to ``RESULTS/ON_ELEMENTS/
localForce/<bracket_key>``. The previous reader only walked
``section.force`` (force-based beams with section integration points),
so ``available_components()`` returned ``[]`` and ``read_line_stations``
silently produced empty slabs. These tests pin both the unit-level
parser/discovery and the end-to-end ``Results`` API on the real
``elasticFrame.mpco`` fixture.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.readers._mpco_local_force_io import (
    _LocalForceBucket,
    available_components_in_local_force,
    discover_local_force_buckets,
    parse_local_force_layout,
    read_local_force_bucket_slab,
)


_FIXTURE = Path("tests/fixtures/results/elasticFrame.mpco")
_BRACKET = "5-ElasticBeam3d[1:0:0]"


# =====================================================================
# Layout parser
# =====================================================================

def test_parse_layout_3d_beam():
    raw = b"0.N_1,Vy_1,Vz_1,T_1,My_1,Mz_1,N_2,Vy_2,Vz_2,T_2,My_2,Mz_2"
    n_stations, layout = parse_local_force_layout(raw, bracket_key="x")
    assert n_stations == 2
    assert layout == {
        "axial_force":      [0, 6],
        "shear_y":          [1, 7],
        "shear_z":          [2, 8],
        "torsion":          [3, 9],
        "bending_moment_y": [4, 10],
        "bending_moment_z": [5, 11],
    }


def test_parse_layout_2d_beam():
    """2-D ElasticBeam emits N, Vy, Mz at each end (6 components total)."""
    raw = b"0.N_1,Vy_1,Mz_1,N_2,Vy_2,Mz_2"
    n_stations, layout = parse_local_force_layout(raw, bracket_key="x")
    assert n_stations == 2
    assert layout == {
        "axial_force":      [0, 3],
        "shear_y":          [1, 4],
        "bending_moment_z": [2, 5],
    }


def test_parse_layout_unknown_tokens_silently_skipped():
    """Foreign tokens (e.g. future MPCO additions) shouldn't break parsing."""
    raw = b"0.N_1,FooBar_1,N_2,FooBar_2"
    n_stations, layout = parse_local_force_layout(raw, bracket_key="x")
    assert n_stations == 2
    assert "axial_force" in layout
    assert layout["axial_force"] == [0, 2]


def test_parse_layout_empty_components_raises():
    with pytest.raises(ValueError, match="empty component list"):
        parse_local_force_layout(b"0.", bracket_key="x")


def test_parse_layout_no_recognized_tokens_raises():
    with pytest.raises(ValueError, match="no recognized"):
        parse_local_force_layout(b"0.Foo_1,Bar_2", bracket_key="x")


def test_parse_layout_partial_station_raises():
    """If a canonical has station 1 but not station 2 we must complain.

    A truly inconsistent recorder shouldn't be silently flattened.
    """
    raw = b"0.N_1,Mz_1,N_2"   # missing Mz_2
    with pytest.raises(ValueError, match="missing station indices"):
        parse_local_force_layout(raw, bracket_key="x")


# =====================================================================
# Discovery
# =====================================================================

@pytest.fixture
def fixture_h5():
    if not _FIXTURE.exists():
        pytest.skip(f"Missing fixture: {_FIXTURE}")
    with h5py.File(_FIXTURE, "r") as f:
        yield f


def test_discovery_finds_elastic_beam_bucket(fixture_h5):
    on_elements = fixture_h5["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
    buckets = discover_local_force_buckets(on_elements)
    assert len(buckets) == 1
    b = buckets[0]
    assert isinstance(b, _LocalForceBucket)
    assert b.bracket_key == _BRACKET
    assert b.n_stations == 2
    assert "axial_force" in b.layout
    assert "bending_moment_z" in b.layout


def test_discovery_returns_empty_when_localforce_absent():
    """A stage without ON_ELEMENTS/localForce/ should not raise."""
    class _Fake:
        def __contains__(self, k): return False
    buckets = discover_local_force_buckets(_Fake())
    assert buckets == []


def test_available_components_aggregate(fixture_h5):
    on_elements = fixture_h5["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
    comps = available_components_in_local_force(on_elements)
    assert comps == {
        "axial_force", "shear_y", "shear_z", "torsion",
        "bending_moment_y", "bending_moment_z",
    }


# =====================================================================
# Slab read — direct (one bucket)
# =====================================================================

def test_read_axial_force_returns_two_stations_per_element(fixture_h5):
    on_elements = fixture_h5["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
    bucket = discover_local_force_buckets(on_elements)[0]
    bucket_grp = on_elements[f"localForce/{_BRACKET}"]
    t_idx = np.array([0], dtype=np.int64)
    result = read_local_force_bucket_slab(
        bucket_grp, bucket, "axial_force",
        t_idx=t_idx, element_ids=None,
    )
    assert result is not None
    values, element_index, station_xi = result
    # 11 elements × 2 stations
    assert values.shape == (1, 22)
    assert element_index.shape == (22,)
    np.testing.assert_array_equal(
        element_index,
        np.repeat(np.arange(1, 12, dtype=np.int64), 2),
    )
    np.testing.assert_allclose(
        station_xi, np.tile([-1.0, 1.0], 11),
    )


def test_read_filters_to_requested_element_ids(fixture_h5):
    on_elements = fixture_h5["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
    bucket = discover_local_force_buckets(on_elements)[0]
    bucket_grp = on_elements[f"localForce/{_BRACKET}"]
    t_idx = np.array([0, 1, 2], dtype=np.int64)

    result = read_local_force_bucket_slab(
        bucket_grp, bucket, "axial_force",
        t_idx=t_idx, element_ids=np.array([3, 7]),
    )
    assert result is not None
    values, element_index, _ = result
    # 2 elements × 2 stations × 3 steps
    assert values.shape == (3, 4)
    np.testing.assert_array_equal(element_index, [3, 3, 7, 7])


def test_read_unknown_canonical_returns_none(fixture_h5):
    on_elements = fixture_h5["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
    bucket = discover_local_force_buckets(on_elements)[0]
    bucket_grp = on_elements[f"localForce/{_BRACKET}"]
    t_idx = np.array([0], dtype=np.int64)
    out = read_local_force_bucket_slab(
        bucket_grp, bucket, "stress_xx",
        t_idx=t_idx, element_ids=None,
    )
    assert out is None


def test_read_value_matches_raw_h5(fixture_h5):
    """Verify we pull the right column from the right row."""
    on_elements = fixture_h5["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
    bucket = discover_local_force_buckets(on_elements)[0]
    bucket_grp = on_elements[f"localForce/{_BRACKET}"]
    raw_step0 = np.asarray(bucket_grp["DATA/STEP_0"][...], dtype=np.float64)
    raw_ids = np.asarray(bucket_grp["ID"][...]).flatten().astype(np.int64)

    # For element 1 at step 0:
    # COMPONENTS = N_1,Vy_1,Vz_1,T_1,My_1,Mz_1,N_2,Vy_2,Vz_2,T_2,My_2,Mz_2
    #              0   1    2    3   4    5    6   7    8    9   10   11
    # bending_moment_z: cols [5, 11]
    expected_e1_s1 = raw_step0[0, 5]   # row 0 = element id 1, col 5
    expected_e1_s2 = raw_step0[0, 11]

    t_idx = np.array([0], dtype=np.int64)
    values, element_index, _ = read_local_force_bucket_slab(
        bucket_grp, bucket, "bending_moment_z",
        t_idx=t_idx, element_ids=None,
    )
    # element_index repeats per station; columns for element 1 are 0 and 1.
    assert int(element_index[0]) == int(raw_ids[0])
    np.testing.assert_allclose(values[0, 0], expected_e1_s1)
    np.testing.assert_allclose(values[0, 1], expected_e1_s2)


# =====================================================================
# End-to-end via Results API
# =====================================================================

def test_results_available_components_includes_local_force_canonicals():
    if not _FIXTURE.exists():
        pytest.skip(f"Missing fixture: {_FIXTURE}")
    r = Results.from_mpco(_FIXTURE)
    s = r.stage(r.stages[0].name)
    comps = set(s.elements.line_stations.available_components())
    assert {
        "axial_force", "bending_moment_y", "bending_moment_z",
        "shear_y", "shear_z", "torsion",
    }.issubset(comps)


def test_results_read_line_stations_returns_two_station_slab():
    if not _FIXTURE.exists():
        pytest.skip(f"Missing fixture: {_FIXTURE}")
    r = Results.from_mpco(_FIXTURE)
    s = r.stage(r.stages[0].name)
    slab = s.elements.line_stations.get(component="axial_force")
    # 10 steps × (11 elements × 2 stations)
    assert slab.values.shape == (10, 22)
    np.testing.assert_array_equal(
        np.asarray(slab.element_index, dtype=np.int64),
        np.repeat(np.arange(1, 12, dtype=np.int64), 2),
    )
    np.testing.assert_allclose(
        np.asarray(slab.station_natural_coord),
        np.tile([-1.0, 1.0], 11),
    )


def test_results_axial_equilibrium_on_elastic_beam():
    """Statics check: each element's two end axials sum to zero."""
    if not _FIXTURE.exists():
        pytest.skip(f"Missing fixture: {_FIXTURE}")
    r = Results.from_mpco(_FIXTURE)
    s = r.stage(r.stages[0].name)
    slab = s.elements.line_stations.get(component="axial_force")
    # Reshape (T, E*2) -> (T, E, 2) and check N_1 + N_2 == 0
    values = np.asarray(slab.values).reshape(slab.values.shape[0], -1, 2)
    np.testing.assert_allclose(values.sum(axis=2), 0.0, atol=1e-6)


# Dedup precedence (section.force trumps localForce for the same
# element) is exercised at scale by the existing real-world tests:
#   - tests/test_results_mpco_fiber_real.py::TestCrossTopologySmoke
#   - tests/test_results_mpco_multi_real.py::TestElementConcat
# Those fixtures contain DispBeamColumn3d elements that write BOTH
# buckets; they assert the station count equals (n_elements * n_IPs)
# rather than (n_elements * (n_IPs + 2)). If the dedup logic in
# MPCOReader.read_line_stations regresses, those tests fail loudly.
