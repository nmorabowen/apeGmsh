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
    NoLineStationTokensError,
    _LocalForceBucket,
    available_components_in_local_force,
    discover_local_force_buckets,
    parse_local_force_layout,
    read_local_force_bucket_slab,
)


from tests.conftest import _stub_model_h5_path

_FIXTURE = Path("tests/fixtures/results/elasticFrame.mpco")
_BRACKET = "5-ElasticBeam3d[1:0:0]"


def _w(raw: bytes) -> int:
    """The DATA width that agrees with ``raw``'s META token count."""
    return len([t for t in raw.decode().rsplit(".", 1)[1].split(",") if t])


# =====================================================================
# Layout parser
# =====================================================================

def test_parse_layout_3d_beam():
    raw = b"0.N_1,Vy_1,Vz_1,T_1,My_1,Mz_1,N_2,Vy_2,Vz_2,T_2,My_2,Mz_2"
    n_stations, layout = parse_local_force_layout(raw, data_width=_w(raw), bracket_key="x")
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
    """A 6-column bucket spelled with the 3-D names ``Vy``/``Mz`` reads too.

    No class in the fork tree writes this; the real 2-D spellings are in
    :func:`test_parse_layout_2d_spellings_of_every_family`.
    """
    raw = b"0.N_1,Vy_1,Mz_1,N_2,Vy_2,Mz_2"
    n_stations, layout = parse_local_force_layout(raw, data_width=_w(raw), bracket_key="x")
    assert n_stations == 2
    assert layout == {
        "axial_force":      [0, 3],
        "shear_y":          [1, 4],
        "bending_moment_z": [2, 5],
    }


def test_parse_layout_unknown_tokens_silently_skipped():
    """Foreign tokens (e.g. future MPCO additions) shouldn't break parsing."""
    raw = b"0.N_1,FooBar_1,N_2,FooBar_2"
    n_stations, layout = parse_local_force_layout(raw, data_width=_w(raw), bracket_key="x")
    assert n_stations == 2
    assert "axial_force" in layout
    assert layout["axial_force"] == [0, 2]


def test_parse_layout_empty_components_raises():
    with pytest.raises(ValueError, match="empty component list"):
        parse_local_force_layout(b"0.", data_width=0, bracket_key="x")


def test_parse_layout_no_recognized_tokens_raises():
    with pytest.raises(ValueError, match="no recognized"):
        parse_local_force_layout(b"0.Foo_1,Bar_2", data_width=2, bracket_key="x")


def test_parse_layout_partial_station_raises():
    """If a canonical has station 1 but not station 2 we must complain.

    A truly inconsistent recorder shouldn't be silently flattened.
    """
    raw = b"0.N_1,Mz_1,N_2"   # missing Mz_2
    with pytest.raises(ValueError, match="missing station indices"):
        parse_local_force_layout(raw, data_width=_w(raw), bracket_key="x")


# =====================================================================
# 2-D spellings (#2010)
# =====================================================================
#
# The META strings and bracket keys below are what the MPCO recorder wrote
# on fork build 79e0623670 for each 2-D beam family that records
# localForce (probe in PR #2010; the class list is every ``setResponse``
# "localForce" of the fork's 2-D beam classes). Before #2010 the 2-D
# ``V``/``M`` and the underscore-free ``N1`` were dropped: ``N_1`` buckets
# read ``axial_force`` only, ``N1`` buckets read nothing.

_LAYOUT_2D = {
    "axial_force":      [0, 3],
    "shear_y":          [1, 4],
    "bending_moment_z": [2, 5],
}

_SPELLINGS_2D = [
    # (bracket_key, META/COMPONENTS)
    ("3-ElasticBeam2d[1:0:0]", b"0.N_1,V_1,M_1,N_2,V_2,M_2"),
    ("145-ElasticTimoshenkoBeam2d[1:0:0]", b"0.N_1,V_1,M_1,N_2,V_2,M_2"),
    ("73-ForceBeamColumn2d[1000:1:0]", b"0.N_1,V_1,M_1,N_2,V_2,M_2"),
    ("62-DispBeamColumn2d[1000:1:0]", b"0.N1,V1,M1,N2,V2,M2"),
    ("33013-LadrunoDispBeamColumn2d[0:0:0]", b"0.N1,V1,M1,N2,V2,M2"),
]


@pytest.mark.parametrize("bracket, raw", _SPELLINGS_2D,
                         ids=[b for b, _ in _SPELLINGS_2D])
def test_parse_layout_2d_spellings_of_every_family(bracket, raw):
    n_stations, layout = parse_local_force_layout(
        raw, data_width=6, bracket_key=bracket,
    )
    assert n_stations == 2
    assert layout == _LAYOUT_2D


def test_parse_layout_2d_warping_extras_are_ignored():
    """ElasticForceBeamColumnWarping2d adds ``Q``; N, V, M still read."""
    raw = b"0.N_1,V_1,M_1,Q_1,N_2,V_2,M_2,Q_2"
    _, layout = parse_local_force_layout(
        raw, data_width=8, bracket_key="1-ElasticForceBeamColumnWarping2d[1:0:0]",
    )
    assert layout == {
        "axial_force": [0, 4], "shear_y": [1, 5], "bending_moment_z": [2, 6],
    }


def test_parse_layout_meta_data_width_mismatch_refuses():
    """ComponentElement3d's shape: six 2-D names over a 12-wide vector."""
    with pytest.raises(ValueError, match=r"names 6 columns .* are 12 wide"):
        parse_local_force_layout(
            b"0.N_1,V_1,M_1,N_2,V_2,M_2", data_width=12,
            bracket_key="9-ComponentElement3d[0:0:0]",
        )


@pytest.mark.parametrize("bracket", [
    "9-ComponentElement3d[0:0:0]",
    "78-ForceBeamColumnCBDI3d[0:0:0]",
])
def test_parse_layout_2d_names_on_a_3d_class_refuse(bracket):
    """MPCO sizes DATA by META, so the real file agrees on width 6.

    The fork build writes ``ForceBeamColumnCBDI3d`` this way: META
    ``N_1,V_1,M_1,N_2,V_2,M_2``, six columns, rows zero-filled, while live
    ``eleResponse`` returns 12 values. Only the class gate refuses it.
    """
    with pytest.raises(ValueError, match="3-D element class"):
        parse_local_force_layout(
            b"0.N_1,V_1,M_1,N_2,V_2,M_2", data_width=6, bracket_key=bracket,
        )


def test_parse_layout_mixed_2d_3d_spelling_refuses():
    with pytest.raises(ValueError, match="ambiguous"):
        parse_local_force_layout(
            b"0.N_1,V_1,Mz_1,N_2,V_2,Mz_2", data_width=6, bracket_key="x",
        )


def test_parse_layout_duplicate_station_refuses():
    with pytest.raises(ValueError, match="twice"):
        parse_local_force_layout(
            b"0.N_1,N1,N_2", data_width=3, bracket_key="x",
        )


def test_parse_layout_no_station_tokens_is_its_own_error():
    """A truss's ``C1..C4`` is not a malformed beam bucket."""
    with pytest.raises(NoLineStationTokensError):
        parse_local_force_layout(
            b"0.C1,C2,C3,C4", data_width=4, bracket_key="12-Truss[1:0:0]",
        )


# ── synthetic ON_ELEMENTS groups ─────────────────────────────────────

def _on_elements(tmp_path, buckets):
    """Write ``{bracket: (META bytes | None, [step arrays], ids)}``."""
    f = h5py.File(tmp_path / "lf.h5", "w")
    lf = f.create_group("ON_ELEMENTS/localForce")
    for bracket, (meta, steps, ids) in buckets.items():
        g = lf.create_group(bracket)
        if meta is not None:
            g.create_dataset("META/COMPONENTS", data=np.array([meta]))
        g.create_dataset("ID", data=np.asarray(ids, dtype=np.int64)[:, None])
        data = g.create_group("DATA")
        for i, arr in enumerate(steps):
            data.create_dataset(f"STEP_{i}", data=np.asarray(arr, dtype=float))
    return f


#: ``eleResponse(e, "localForce")`` of the #2010 repro girder.
_GIRDER = [5788.86, -1482.87, -4049.05, -5788.86, 1482.87, 1577.60]


@pytest.mark.parametrize("bracket, raw", _SPELLINGS_2D,
                         ids=[b for b, _ in _SPELLINGS_2D])
def test_read_2d_bucket_values(tmp_path, bracket, raw):
    """V and M read as end i as recorded, end j negated, like N."""
    rows = np.array([_GIRDER, [2.0 * v for v in _GIRDER]])
    with _on_elements(tmp_path, {bracket: (raw, [rows], [8, 9])}) as f:
        on_el = f["ON_ELEMENTS"]
        assert available_components_in_local_force(on_el) == set(_LAYOUT_2D)
        [bucket] = discover_local_force_buckets(on_el)
        for comp, (ci, cj) in _LAYOUT_2D.items():
            values, eidx, xi = read_local_force_bucket_slab(
                on_el[f"localForce/{bracket}"], bucket, comp,
                t_idx=np.array([0]), element_ids=None,
            )
            np.testing.assert_array_equal(eidx, [8, 8, 9, 9])
            np.testing.assert_array_equal(xi, [-1.0, 1.0, -1.0, 1.0])
            np.testing.assert_allclose(
                values[0],
                [rows[0, ci], -rows[0, cj], rows[1, ci], -rows[1, cj]],
                rtol=1e-12,
            )


def test_discovery_refuses_width_mismatch(tmp_path):
    bracket = "9-ComponentElement3d[0:0:0]"
    raw = b"0.N_1,V_1,M_1,N_2,V_2,M_2"
    with _on_elements(tmp_path, {bracket: (raw, [np.zeros((1, 12))], [1])}) as f:
        with pytest.raises(ValueError, match=r"ComponentElement3d.*6 columns.*12 wide"):
            discover_local_force_buckets(f["ON_ELEMENTS"])
        with pytest.raises(ValueError, match="12 wide"):
            available_components_in_local_force(f["ON_ELEMENTS"])


def test_discovery_surfaces_a_malformed_bucket(tmp_path):
    """Before #2010 a parse failure dropped the bucket with no word.

    The 3-D bearings and MixedBeamColumn3d spell ``N_ 1``/``Tz_1``/``Py_2``
    upstream, so a canonical misses a station.
    """
    raw = b"0.N_ 1,Vy_1,Vz_1,T_1,My_1,Tz_1,N_2,Py_2,Pz_2,T_2,My_2,Mz_2"
    with _on_elements(tmp_path, {
        "3-MixedBeamColumn3d[1:0:0]": (raw, [np.zeros((1, 12))], [1]),
    }) as f:
        with pytest.raises(ValueError, match="missing station indices"):
            discover_local_force_buckets(f["ON_ELEMENTS"])


def test_discovery_refuses_a_bucket_without_meta(tmp_path):
    with _on_elements(tmp_path, {
        "3-ElasticBeam2d[1:0:0]": (None, [np.zeros((1, 6))], [1]),
    }) as f:
        with pytest.raises(ValueError, match="no META/COMPONENTS"):
            discover_local_force_buckets(f["ON_ELEMENTS"])


def test_discovery_passes_over_a_truss_force_vector(tmp_path):
    """MPCO names an untagged response ``C1..Cn``: a Truss beside a beam."""
    with _on_elements(tmp_path, {
        "12-Truss[1:0:0]": (b"0.C1,C2,C3,C4", [np.ones((2, 4))], [1, 2]),
        "3-ElasticBeam2d[1:0:0]": (
            b"0.N_1,V_1,M_1,N_2,V_2,M_2", [np.ones((1, 6))], [3],
        ),
    }) as f:
        buckets = discover_local_force_buckets(f["ON_ELEMENTS"])
    assert [b.bracket_key for b in buckets] == ["3-ElasticBeam2d[1:0:0]"]


def test_discovery_passes_over_a_bucket_with_no_step(tmp_path):
    with _on_elements(tmp_path, {
        "3-ElasticBeam2d[1:0:0]": (b"0.N_1,V_1,M_1,N_2,V_2,M_2", [], [3]),
    }) as f:
        assert discover_local_force_buckets(f["ON_ELEMENTS"]) == []


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
# Slab read â€” direct (one bucket)
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
    # 11 elements Ã— 2 stations
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
    # 2 elements Ã— 2 stations Ã— 3 steps
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
    """Verify we pull the right column from the right row.

    Station 1 is the raw ``localForce`` value (force from joint on
    element). Station 2 is **negated** so the slab represents internal
    section forces in textbook convention (continuous across element
    boundaries). See the docstring on ``read_local_force_bucket_slab``.
    """
    on_elements = fixture_h5["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
    bucket = discover_local_force_buckets(on_elements)[0]
    bucket_grp = on_elements[f"localForce/{_BRACKET}"]
    raw_step0 = np.asarray(bucket_grp["DATA/STEP_0"][...], dtype=np.float64)
    raw_ids = np.asarray(bucket_grp["ID"][...]).flatten().astype(np.int64)

    # For element 1 at step 0:
    # COMPONENTS = N_1,Vy_1,Vz_1,T_1,My_1,Mz_1,N_2,Vy_2,Vz_2,T_2,My_2,Mz_2
    #              0   1    2    3   4    5    6   7    8    9   10   11
    # bending_moment_z: cols [5, 11]
    expected_e1_s1 = raw_step0[0, 5]    # station 1: as-is
    expected_e1_s2 = -raw_step0[0, 11]  # station 2: sign flipped

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
    r = Results.from_mpco(_FIXTURE, model_h5=_stub_model_h5_path())
    s = r.stage(r.stages[0].name)
    comps = set(s.elements.line_stations.available_components())
    assert {
        "axial_force", "bending_moment_y", "bending_moment_z",
        "shear_y", "shear_z", "torsion",
    }.issubset(comps)


def test_results_read_line_stations_returns_two_station_slab():
    if not _FIXTURE.exists():
        pytest.skip(f"Missing fixture: {_FIXTURE}")
    r = Results.from_mpco(_FIXTURE, model_h5=_stub_model_h5_path())
    s = r.stage(r.stages[0].name)
    slab = s.elements.line_stations.get(component="axial_force")
    # 10 steps Ã— (11 elements Ã— 2 stations)
    assert slab.values.shape == (10, 22)
    np.testing.assert_array_equal(
        np.asarray(slab.element_index, dtype=np.int64),
        np.repeat(np.arange(1, 12, dtype=np.int64), 2),
    )
    np.testing.assert_allclose(
        np.asarray(slab.station_natural_coord),
        np.tile([-1.0, 1.0], 11),
    )


def test_results_axial_force_constant_along_element():
    """After the station-2 sign flip the slab is in internal-force
    convention: axial force is constant along an element with no axial
    distributed load â€” i.e., values at xi=-1 equal values at xi=+1.
    """
    if not _FIXTURE.exists():
        pytest.skip(f"Missing fixture: {_FIXTURE}")
    r = Results.from_mpco(_FIXTURE, model_h5=_stub_model_h5_path())
    s = r.stage(r.stages[0].name)
    slab = s.elements.line_stations.get(component="axial_force")
    # Reshape (T, E*2) -> (T, E, 2). Internal axial is constant per
    # element so the two stations must agree.
    values = np.asarray(slab.values).reshape(slab.values.shape[0], -1, 2)
    np.testing.assert_allclose(values[:, :, 0], values[:, :, 1], atol=1e-6)


def test_results_bending_moment_slope_matches_shear():
    """``dMy/dx == +Vz`` with the sign-flip convention: per element,
    the slope of the linearly-varying bending moment over the element
    length equals the (constant) shear value at station 1.

    This is the cleanest verification that the sign convention is
    consistent across components â€” moments and shears agree on the
    same internal-force convention.
    """
    if not _FIXTURE.exists():
        pytest.skip(f"Missing fixture: {_FIXTURE}")
    r = Results.from_mpco(_FIXTURE, model_h5=_stub_model_h5_path())
    s = r.stage(r.stages[0].name)

    my = s.elements.line_stations.get(component="bending_moment_y")
    vz = s.elements.line_stations.get(component="shear_z")

    # Element length per element from the fem connectivity.
    fem = r.fem
    coords = np.asarray(fem.nodes.coords)
    id_to_idx = {int(n): i for i, n in enumerate(fem.nodes.ids)}
    eid_to_length: dict[int, float] = {}
    for group in fem.elements:
        for eid, conn in zip(group.ids, group.connectivity):
            i, j = id_to_idx[int(conn[0])], id_to_idx[int(conn[1])]
            eid_to_length[int(eid)] = float(np.linalg.norm(
                coords[j] - coords[i]
            ))

    # Reshape (T, E*2) -> (T, E, 2) and pair element_ids with the slabs.
    n_steps = my.values.shape[0]
    my_v = np.asarray(my.values).reshape(n_steps, -1, 2)
    vz_v = np.asarray(vz.values).reshape(n_steps, -1, 2)
    eids = np.asarray(my.element_index, dtype=np.int64).reshape(-1, 2)[:, 0]

    # Test on the last step where amplitudes are largest.
    step = n_steps - 1
    for i, eid in enumerate(eids):
        L = eid_to_length[int(eid)]
        slope = (my_v[step, i, 1] - my_v[step, i, 0]) / L
        # Shear is constant for a beam under end loads only â€” both
        # stations agree (after the sign flip). Pick station 1.
        expected_slope = vz_v[step, i, 0]
        np.testing.assert_allclose(slope, expected_slope, atol=1e-3)


# Dedup precedence (section.force trumps localForce for the same
# element) is exercised at scale by the existing real-world tests:
#   - tests/test_results_mpco_fiber_real.py::TestCrossTopologySmoke
#   - tests/test_results_mpco_multi_real.py::TestElementConcat
# Those fixtures contain DispBeamColumn3d elements that write BOTH
# buckets; they assert the station count equals (n_elements * n_IPs)
# rather than (n_elements * (n_IPs + 2)). If the dedup logic in
# MPCOReader.read_line_stations regresses, those tests fail loudly.
