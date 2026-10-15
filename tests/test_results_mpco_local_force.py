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
    n_stations, layout = parse_local_force_layout(raw, data_width=_w(raw), spatial_dim=3, bracket_key="x")
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
    n_stations, layout = parse_local_force_layout(raw, data_width=_w(raw), spatial_dim=3, bracket_key="x")
    assert n_stations == 2
    assert layout == {
        "axial_force":      [0, 3],
        "shear_y":          [1, 4],
        "bending_moment_z": [2, 5],
    }


def test_parse_layout_unknown_tokens_silently_skipped():
    """Foreign tokens (e.g. future MPCO additions) shouldn't break parsing."""
    raw = b"0.N_1,FooBar_1,N_2,FooBar_2"
    n_stations, layout = parse_local_force_layout(raw, data_width=_w(raw), spatial_dim=3, bracket_key="x")
    assert n_stations == 2
    assert "axial_force" in layout
    assert layout["axial_force"] == [0, 2]


def test_parse_layout_empty_components_raises():
    with pytest.raises(ValueError, match="empty component list"):
        parse_local_force_layout(b"0.", data_width=0, spatial_dim=3, bracket_key="x")


def test_parse_layout_no_recognized_tokens_raises():
    with pytest.raises(ValueError, match="no recognized"):
        parse_local_force_layout(b"0.Foo_1,Bar_2", data_width=2, spatial_dim=3, bracket_key="x")


def test_parse_layout_partial_station_raises():
    """If a canonical has station 1 but not station 2 we must complain.

    A truly inconsistent recorder shouldn't be silently flattened.
    """
    raw = b"0.N_1,Mz_1,N_2"   # missing Mz_2
    with pytest.raises(ValueError, match="missing station indices"):
        parse_local_force_layout(raw, data_width=_w(raw), spatial_dim=3, bracket_key="x")


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
        raw, data_width=6, spatial_dim=2, bracket_key=bracket,
    )
    assert n_stations == 2
    assert layout == _LAYOUT_2D


def test_parse_layout_2d_warping_extras_are_ignored():
    """ElasticForceBeamColumnWarping2d adds ``Q``; N, V, M still read."""
    raw = b"0.N_1,V_1,M_1,Q_1,N_2,V_2,M_2,Q_2"
    _, layout = parse_local_force_layout(
        raw, data_width=8, spatial_dim=2,
        bracket_key="1-ElasticForceBeamColumnWarping2d[1:0:0]",
    )
    assert layout == {
        "axial_force": [0, 4], "shear_y": [1, 5], "bending_moment_z": [2, 6],
    }


def test_parse_layout_meta_data_width_mismatch_refuses():
    """ComponentElement3d's shape: six 2-D names over a 12-wide vector."""
    with pytest.raises(ValueError, match=r"names 6 columns .* are 12 wide"):
        parse_local_force_layout(
            b"0.N_1,V_1,M_1,N_2,V_2,M_2", data_width=12, spatial_dim=3,
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
    ``eleResponse`` returns 12 values. The width gate cannot see it; the
    class gate refuses it even in a file that claims to be 2-D.
    """
    with pytest.raises(ValueError, match="3-D element class"):
        parse_local_force_layout(
            b"0.N_1,V_1,M_1,N_2,V_2,M_2", data_width=6, spatial_dim=2,
            bracket_key=bracket,
        )


@pytest.mark.parametrize("spatial_dim", [3, None])
def test_parse_layout_2d_names_need_a_2d_file(spatial_dim):
    """A 2-D class name is not enough: ``/INFO/SPATIAL_DIM`` must be 2."""
    with pytest.raises(ValueError, match="SPATIAL_DIM"):
        parse_local_force_layout(
            b"0.N_1,V_1,M_1,N_2,V_2,M_2", data_width=6,
            spatial_dim=spatial_dim, bracket_key="3-ElasticBeam2d[1:0:0]",
        )


def test_parse_layout_2d_names_under_an_unparsed_key_refuse():
    with pytest.raises(ValueError, match="does not read"):
        parse_local_force_layout(
            b"0.N_1,V_1,M_1,N_2,V_2,M_2", data_width=6, spatial_dim=2,
            bracket_key="ElasticBeam2d",
        )


def test_parse_layout_mixed_2d_3d_spelling_refuses():
    with pytest.raises(ValueError, match="ambiguous"):
        parse_local_force_layout(
            b"0.N_1,V_1,Mz_1,N_2,V_2,Mz_2", data_width=6, spatial_dim=2,
            bracket_key="3-ElasticBeam2d[1:0:0]",
        )


def test_parse_layout_duplicate_station_refuses():
    with pytest.raises(ValueError, match="twice"):
        parse_local_force_layout(
            b"0.N_1,N1,N_2", data_width=3, spatial_dim=3, bracket_key="x",
        )


def test_parse_layout_no_station_tokens_is_its_own_error():
    """A truss's ``C1..C4`` is not a malformed beam bucket."""
    with pytest.raises(NoLineStationTokensError):
        parse_local_force_layout(
            b"0.C1,C2,C3,C4", data_width=4, spatial_dim=2,
            bracket_key="12-Truss[1:0:0]",
        )


# ── synthetic ON_ELEMENTS groups ─────────────────────────────────────

def _on_elements(tmp_path, buckets, spatial_dim=2):
    """Write ``{bracket: (META bytes | None, [step arrays], ids)}``."""
    f = h5py.File(tmp_path / "lf.h5", "w")
    if spatial_dim is not None:
        f.create_dataset("INFO/SPATIAL_DIM",
                         data=np.array([spatial_dim], dtype=np.int32))
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
    """Force-beam ``section force`` signs (#2009): N and M are
    ``[-end_i, +end_j]``, V stays ``[end_i, -end_j]``."""
    sign = {"axial_force": -1.0, "shear_y": 1.0, "bending_moment_z": -1.0}
    rows = np.array([_GIRDER, [2.0 * v for v in _GIRDER]])
    with _on_elements(tmp_path, {bracket: (raw, [rows], [8, 9])}) as f:
        on_el = f["ON_ELEMENTS"]
        assert available_components_in_local_force(on_el) == set(_LAYOUT_2D)
        [bucket] = discover_local_force_buckets(on_el)
        assert bucket.refusal is None
        for comp, (ci, cj) in _LAYOUT_2D.items():
            values, eidx, xi = read_local_force_bucket_slab(
                on_el[f"localForce/{bracket}"], bucket, comp,
                t_idx=np.array([0]), element_ids=None,
            )
            np.testing.assert_array_equal(eidx, [8, 8, 9, 9])
            np.testing.assert_array_equal(xi, [-1.0, 1.0, -1.0, 1.0])
            np.testing.assert_allclose(
                values[0],
                [sign[comp] * rows[0, ci], -sign[comp] * rows[0, cj],
                 sign[comp] * rows[1, ci], -sign[comp] * rows[1, cj]],
                rtol=1e-12,
            )


_BEAM_3D = "5-ElasticBeam3d[1:0:0]"
_RAW_3D = b"0.N_1,Vy_1,Vz_1,T_1,My_1,Mz_1,N_2,Vy_2,Vz_2,T_2,My_2,Mz_2"
_ROW_3D = np.arange(1.0, 13.0)

#: refused bucket -> (META, step rows, spatial_dim, reason)
_REFUSED = {
    "width": ("9-ComponentElement3d[0:0:0]", b"0.N_1,V_1,M_1,N_2,V_2,M_2",
              np.zeros((1, 12)), 3, r"6 columns.*12 wide"),
    "class": ("78-ForceBeamColumnCBDI3d[0:0:0]", b"0.N_1,V_1,M_1,N_2,V_2,M_2",
              np.zeros((1, 6)), 3, "SPATIAL_DIM"),
    "station": ("3-MixedBeamColumn3d[1:0:0]",
                b"0.N_ 1,Vy_1,Vz_1,T_1,My_1,Tz_1,N_2,Py_2,Pz_2,T_2,My_2,Mz_2",
                np.zeros((1, 12)), 3, "missing station indices"),
    "meta": ("7-ElasticBeam3d[1:0:0]", None, np.zeros((1, 12)), 3,
             "no META/COMPONENTS"),
}


@pytest.mark.parametrize("case", sorted(_REFUSED))
def test_a_refused_bucket_blocks_only_its_own_elements(tmp_path, case):
    """#2020 review F1: one bad bucket must not hide the file's other beams.

    The ElasticBeam3d bucket (element 1) reads normally beside it, and is
    the only one listed; a read naming the bad bucket's element 2, or
    every element, raises the bucket's reason.
    """
    bracket, raw, rows, dim, reason = _REFUSED[case]
    with _on_elements(tmp_path, {
        _BEAM_3D: (_RAW_3D, [_ROW_3D[None, :]], [1]),
        bracket: (raw, [rows], [2]),
    }, spatial_dim=dim) as f:
        on_el = f["ON_ELEMENTS"]
        assert available_components_in_local_force(on_el) == {
            "axial_force", "shear_y", "shear_z", "torsion",
            "bending_moment_y", "bending_moment_z",
        }
        by_key = {b.bracket_key: b for b in discover_local_force_buckets(on_el)}
        good, bad = by_key[_BEAM_3D], by_key[bracket]
        assert good.refusal is None
        assert bad.refusal is not None and bracket in bad.refusal
        assert bad.layout == {}
        t0 = np.array([0])

        def read(bucket, key, ids):
            return read_local_force_bucket_slab(
                on_el[f"localForce/{key}"], bucket, "shear_y",
                t_idx=t0, element_ids=ids,
            )

        values, eidx, _ = read(good, _BEAM_3D, np.array([1]))
        np.testing.assert_array_equal(eidx, [1, 1])
        np.testing.assert_allclose(values[0], [_ROW_3D[1], -_ROW_3D[7]])
        assert read(bad, bracket, np.array([1])) is None
        for ids in (np.array([2]), np.array([1, 2]), None):
            with pytest.raises(ValueError, match=reason):
                read(bad, bracket, ids)


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

    The slab is in the force-beam ``section force`` convention (#2009):
    for a moment, station 1 is ``-end_i`` and station 2 is ``+end_j``.
    ``bending_moment_y`` is read because the fixture's ``Mz`` is all
    zero, which no sign can tell apart.
    """
    on_elements = fixture_h5["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
    bucket = discover_local_force_buckets(on_elements)[0]
    bucket_grp = on_elements[f"localForce/{_BRACKET}"]
    raw_step0 = np.asarray(bucket_grp["DATA/STEP_0"][...], dtype=np.float64)
    raw_ids = np.asarray(bucket_grp["ID"][...]).flatten().astype(np.int64)

    # For element 1 at step 0:
    # COMPONENTS = N_1,Vy_1,Vz_1,T_1,My_1,Mz_1,N_2,Vy_2,Vz_2,T_2,My_2,Mz_2
    #              0   1    2    3   4    5    6   7    8    9   10   11
    # bending_moment_y: cols [4, 10]
    assert raw_step0[0, 4] != 0.0 and raw_step0[0, 10] != 0.0
    expected_e1_s1 = -raw_step0[0, 4]   # station 1: -end_i
    expected_e1_s2 = raw_step0[0, 10]   # station 2: +end_j

    t_idx = np.array([0], dtype=np.int64)
    values, element_index, _ = read_local_force_bucket_slab(
        bucket_grp, bucket, "bending_moment_y",
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
