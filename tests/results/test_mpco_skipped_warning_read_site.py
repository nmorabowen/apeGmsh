"""A read names only the skipped MPCO buckets that hold an element it
asked for (#2165 part 2, slice #2293).

MPCO discovery used to emit ``MPCOBucketSkippedWarning`` itself, once per
skipped bucket, whatever the read asked for: a read narrowed with
``element_ids`` was told about buckets whose elements it never wanted.
Discovery now returns the skipped buckets with their reasons, and the read
names, once each, the ones holding a requested element. An unfiltered read
names every one, and a bucket whose ``ID`` cannot be read is always named.
A bucket refused by its decoder at read time follows the same rule.

The oracle is the file itself: each skipped bucket holds one known element
id, so the buckets a read must name are exactly the ones whose id it asks
for. The reader is called directly, so ``element_ids`` reach it untouched
(no tag map), and every warning must point at this file's line.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.opensees._response_catalog import (
    ELE_TAG_FourNodeTetrahedron,
    IntRule,
)
from apeGmsh.results import Results
from apeGmsh.results.readers._mpco import MPCOBucketSkippedWarning, MPCOReader
from tests.conftest import _stub_model_h5_path
from tests.results.test_mpco_custom_rule_discovery import (
    _add_damage,
    _add_tet_stress,
    _skeleton,
    _tet,
)
from tests.test_results_mpco_element_mock import (
    _add_stress_bucket,
    _create_mpco_skeleton,
)
from tests.test_results_mpco_spring_mock import _add_bucket, _make_spring_mpco

#: Gauss / material-state fixture: element 1 reads, each other id sits in
#: one skipped bucket.
_READ = _tet(IntRule.Tet_GL_1, 0)
_CUSTOM = _tet(IntRule.Custom, 3)          # nonzero custom-rule index
_UNCATALOGUED = _tet(IntRule.Tet_GL_2, 0)  # no RESPONSE_CATALOG row
_GAUSS_SKIPS = {2: _CUSTOM, 3: _UNCATALOGUED}


def _skips(record: "list[warnings.WarningMessage]") -> list[warnings.WarningMessage]:
    return [w for w in record if issubclass(w.category, MPCOBucketSkippedWarning)]


def _named_keys(skips: "list[warnings.WarningMessage]", keys) -> list[str]:
    """The keys of ``keys`` named by ``skips``, one entry per naming."""
    return sorted(
        key for w in skips for key in keys if f"/{key}:" in str(w.message)
    )


def _record(call):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = call()
    return result, _skips(record)


def _assert_at_caller(skips: "list[warnings.WarningMessage]") -> None:
    assert {Path(w.filename) for w in skips} == {Path(__file__)}, [
        (w.filename, w.lineno) for w in skips
    ]


# =====================================================================
# Gauss (continuum stress)
# =====================================================================

@pytest.fixture
def stress_file(tmp_path: Path) -> Path:
    path = tmp_path / "tets.mpco"
    f, stage = _skeleton(path)
    try:
        _add_tet_stress(f[stage], _READ, 1)
        for eid, key in _GAUSS_SKIPS.items():
            _add_tet_stress(f[stage], key, eid)
    finally:
        f.close()
    return path


def _read_gauss(path: Path, component: str, element_ids):
    with MPCOReader(path) as r:
        sid = r.stages()[0].id
        return _record(lambda: r.read_gauss(
            sid, component, element_ids=element_ids,
        ))


def test_gauss_read_excluding_every_skipped_bucket_is_silent(
    stress_file: Path,
) -> None:
    slab, skips = _read_gauss(stress_file, "stress_xx", np.array([1]))
    assert skips == [], [str(w.message) for w in skips]
    np.testing.assert_array_equal(slab.values, np.full((2, 1), 1.0))


@pytest.mark.parametrize("eid", sorted(_GAUSS_SKIPS))
def test_gauss_read_names_only_the_bucket_holding_a_requested_element(
    stress_file: Path, eid: int,
) -> None:
    slab, skips = _read_gauss(stress_file, "stress_xx", np.array([1, eid]))
    assert _named_keys(skips, _GAUSS_SKIPS.values()) == [_GAUSS_SKIPS[eid]]
    assert len(skips) == 1
    assert "gauss discovery of 'stress_xx' in tets.mpco left 1 bucket(s)" \
        in str(skips[0].message)
    _assert_at_caller(skips)
    np.testing.assert_array_equal(slab.element_index, [1])


def test_gauss_read_unfiltered_names_every_skipped_bucket_once(
    stress_file: Path,
) -> None:
    _, skips = _read_gauss(stress_file, "stress_xx", None)
    assert _named_keys(skips, _GAUSS_SKIPS.values()) == sorted(
        _GAUSS_SKIPS.values()
    )
    assert len(skips) == len(_GAUSS_SKIPS)
    _assert_at_caller(skips)


def test_a_skipped_bucket_with_no_readable_ids_is_named(
    tmp_path: Path,
) -> None:
    """No ``ID`` to rule it out: a filtered read still names it."""
    no_ids = _tet(IntRule.Custom, 5)
    path = tmp_path / "no_ids.mpco"
    f, stage = _skeleton(path)
    try:
        _add_tet_stress(f[stage], _READ, 1)
        f[stage]["RESULTS/ON_ELEMENTS/stresses"].create_group(no_ids)
    finally:
        f.close()
    _, skips = _read_gauss(path, "stress_xx", np.array([1]))
    assert _named_keys(skips, [no_ids]) == [no_ids]
    assert len(skips) == 1


# =====================================================================
# Gauss material state (META-driven path)
# =====================================================================

@pytest.fixture
def damage_file(tmp_path: Path) -> Path:
    path = tmp_path / "damage.mpco"
    f, stage = _skeleton(path)
    try:
        _add_damage(f[stage], _READ, eid=1, n_gp=1)
        _add_damage(f[stage], _CUSTOM, eid=2, n_gp=1)
    finally:
        f.close()
    return path


def test_material_state_read_excluding_the_skipped_bucket_is_silent(
    damage_file: Path,
) -> None:
    slab, skips = _read_gauss(damage_file, "damage_tension", np.array([1]))
    assert skips == [], [str(w.message) for w in skips]
    np.testing.assert_array_equal(slab.values, np.full((2, 1), 1.0))


@pytest.mark.parametrize("element_ids", [np.array([2]), None],
                         ids=["filtered", "unfiltered"])
def test_material_state_read_names_the_skipped_bucket_at_the_caller(
    damage_file: Path, element_ids,
) -> None:
    _, skips = _read_gauss(damage_file, "damage_tension", element_ids)
    assert _named_keys(skips, [_CUSTOM]) == [_CUSTOM]
    assert len(skips) == 1
    assert "gauss material-state discovery of 'damage_tension'" \
        in str(skips[0].message)
    _assert_at_caller(skips)


# =====================================================================
# Every family: an unparseable key holding element 99
# =====================================================================

_BAD = "not-a-bucket"

#: family -> (ON_ELEMENTS group, reader method, component).
_FAMILIES = {
    "gauss": ("stresses", "read_gauss", "stress_xx"),
    "material-state": ("material.damage", "read_gauss", "damage"),
    "spring": ("basicForce", "read_springs", "spring_force_0"),
    "fiber": ("section.fiber.stress", "read_fibers", "fiber_stress"),
    "line-station": ("section.force", "read_line_stations", "axial_force"),
}


@pytest.fixture
def bad_keys_file(tmp_path: Path) -> Path:
    path = tmp_path / "bad_keys.mpco"
    f, stage_name = _create_mpco_skeleton(
        path, node_ids=np.array([1, 2]), node_coords=np.zeros((2, 3)),
        n_steps=1, dt=1.0,
    )
    try:
        stage = f[stage_name]
        stage["MODEL"].create_group("ELEMENTS")
        stage["MODEL"].create_group("SECTION_ASSIGNMENTS")
        on_el = stage["RESULTS"].require_group("ON_ELEMENTS")
        for group, _method, _component in _FAMILIES.values():
            bucket = on_el.require_group(group).create_group(_BAD)
            bucket.create_dataset("ID", data=np.array([[99]], dtype=np.int32))
    finally:
        f.close()
    return path


@pytest.mark.parametrize("family", sorted(_FAMILIES))
@pytest.mark.parametrize(("element_ids", "named"), [
    (np.array([1]), False),
    (np.array([1, 99]), True),
    (None, True),
], ids=["excluded", "included", "unfiltered"])
def test_every_family_names_an_unparseable_key_only_when_asked(
    bad_keys_file: Path, family: str, element_ids, named: bool,
) -> None:
    group, method, component = _FAMILIES[family]
    with MPCOReader(bad_keys_file) as r:
        sid = r.stages()[0].id
        _, skips = _record(lambda: getattr(r, method)(
            sid, component, element_ids=element_ids,
        ))
    if not named:
        assert skips == [], [str(w.message) for w in skips]
        return
    assert len(skips) == 1, [str(w.message) for w in skips]
    assert f"/{group}/{_BAD}: its bracket key does not parse" \
        in str(skips[0].message)
    _assert_at_caller(skips)


# =====================================================================
# Springs: a discovery skip and a decoder refusal, one element each
# =====================================================================

_SPRING_CUSTOM = "19-ZeroLength[1:2:0]"   # discovery skip, element 3
_SPRING_NO_META = "19-ZeroLength[1:0:1]"  # decoder refusal, element 4


@pytest.fixture
def spring_file(tmp_path: Path) -> Path:
    path = _make_spring_mpco(
        tmp_path / "springs.mpco", element_ids=[1, 2], n_springs=1,
        force_values=np.full((2, 2, 1), 3.0),
    )
    _add_bucket(path, "basicForce", _SPRING_CUSTOM,
                element_ids=[3], values=np.full((2, 1, 1), 9.0))
    _add_bucket(path, "basicForce", _SPRING_NO_META,
                element_ids=[4], values=np.full((2, 1, 1), 7.0))
    with h5py.File(path, "a") as f:
        del f[f"MODEL_STAGE[1]/RESULTS/ON_ELEMENTS/basicForce/"
              f"{_SPRING_NO_META}/META"]
    return path


@pytest.mark.parametrize(("element_ids", "expected"), [
    (np.array([1]), []),
    (np.array([1, 3]), [_SPRING_CUSTOM]),
    (np.array([1, 4]), [_SPRING_NO_META]),
    (None, sorted([_SPRING_CUSTOM, _SPRING_NO_META])),
], ids=["excluded", "discovery-skip", "decoder-refusal", "unfiltered"])
def test_spring_read_names_only_skipped_buckets_it_asked_for(
    spring_file: Path, element_ids, expected: "list[str]",
) -> None:
    with MPCOReader(spring_file) as r:
        sid = r.stages()[0].id
        slab, skips = _record(lambda: r.read_springs(
            sid, "spring_force_0", element_ids=element_ids,
        ))
    keys = [_SPRING_CUSTOM, _SPRING_NO_META]
    assert _named_keys(skips, keys) == expected
    assert len(skips) == len(expected)
    if expected:
        _assert_at_caller(skips)
    assert set(slab.element_index.tolist()) == (
        {1} if element_ids is not None else {1, 2}
    )


# =====================================================================
# Decoder refusals at read time: the same filter
# =====================================================================

_HDR1 = _READ[:-2] + "1]"   # a second header of the readable class


@pytest.fixture
def refused_meta_file(tmp_path: Path) -> Path:
    """Element 1 reads. Element 5 sits in a header-1 stress bucket whose
    META labels refuse the catalog's names, and element 6 in a header-1
    damage bucket whose ``NUM_COLUMNS`` contradicts its META."""
    path = tmp_path / "refused.mpco"
    f, stage = _skeleton(path)
    try:
        _add_tet_stress(f[stage], _READ, 1)
        _add_stress_bucket(
            f[stage], bracket_key=_HDR1,
            class_tag=ELE_TAG_FourNodeTetrahedron,
            int_rule=IntRule.Tet_GL_1,
            element_ids=np.array([5], dtype=np.int32),
            flat_data=np.full((2, 1, 6), 5.0), n_gauss_points=1,
            component_labels=(
                "sigma_xx", "sigma_yy", "sigma_zz",
                "sigma_xy", "sigma_yz", "bogus",
            ),
        )
        _add_damage(f[stage], _READ, eid=1, n_gp=1)
        _add_damage(f[stage], _HDR1, eid=6, n_gp=1)
        f[stage][f"RESULTS/ON_ELEMENTS/material.damage/{_HDR1}"].attrs[
            "NUM_COLUMNS"
        ] = np.array([99], dtype=np.int32)
    finally:
        f.close()
    return path


@pytest.mark.parametrize(("component", "group", "eid", "what"), [
    ("stress_xx", "stresses", 5, "gauss read of 'stress_xx'"),
    ("damage_tension", "material.damage", 6,
     "gauss material-state read of 'damage_tension'"),
], ids=["gauss-meta-labels", "material-state-meta"])
@pytest.mark.parametrize("asked", [False, True], ids=["excluded", "included"])
def test_a_decoder_refusal_is_named_only_when_asked(
    refused_meta_file: Path, component: str, group: str, eid: int,
    what: str, asked: bool,
) -> None:
    element_ids = np.array([1, eid] if asked else [1])
    slab, skips = _read_gauss(refused_meta_file, component, element_ids)
    np.testing.assert_array_equal(slab.element_index, [1])
    if not asked:
        assert skips == [], [str(w.message) for w in skips]
        return
    assert len(skips) == 1, [str(w.message) for w in skips]
    msg = str(skips[0].message)
    assert f"/{group}/{_HDR1}:" in msg
    assert what in msg


# =====================================================================
# Through the composite: a listing that gates a filtered read is quiet
# =====================================================================

@pytest.fixture
def composite_stress_file(tmp_path: Path) -> Path:
    path = tmp_path / "composite.mpco"
    f, stage = _skeleton(path)
    try:
        _add_tet_stress(f[stage], _READ, 1)
        _add_tet_stress(f[stage], _CUSTOM, 2)
    finally:
        f.close()
    return path


_COMPOSITE_CALLS = {
    "derived": lambda g, ids: g.get(component="von_mises_stress", ids=ids),
    "census": lambda g, ids: g.tension_census(ids=ids),
}


@pytest.mark.parametrize("call", sorted(_COMPOSITE_CALLS))
@pytest.mark.parametrize(("ids", "named"), [
    ([1], False), ([1, 2], True), (None, True),
], ids=["excluded", "included", "unfiltered"])
def test_a_composite_listing_before_a_filtered_read_is_quiet(
    composite_stress_file: Path, call: str, ids, named: bool,
) -> None:
    """The derived read and the census list the stored components before
    they read. Under a filter only the read names a skipped bucket, and
    only one holding a requested element; unfiltered, it is named."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # the stub model's mismatch notice
        r = Results.from_mpco(
            str(composite_stress_file), model_h5=_stub_model_h5_path(),
        )
    try:
        _, skips = _record(
            lambda: _COMPOSITE_CALLS[call](r.elements.gauss, ids),
        )
    finally:
        r._reader.close()
    if not named:
        assert skips == [], [str(w.message) for w in skips]
        return
    assert _named_keys(skips, [_CUSTOM]), [str(w.message) for w in skips]


def test_a_shell_derived_listing_before_a_filtered_read_is_quiet(
    composite_stress_file: Path,
) -> None:
    """The shell-derived path lists too. On solid stress only it refuses
    (no shell resultants), and under a filter that excludes the skipped
    bucket it names no bucket on the way."""
    from apeGmsh._vocabulary import DERIVED_SHELL_SCALARS

    component = sorted(DERIVED_SHELL_SCALARS)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # the stub model's mismatch notice
        r = Results.from_mpco(
            str(composite_stress_file), model_h5=_stub_model_h5_path(),
        )
    try:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match="missing shell resultants"):
                r.elements.gauss.get(component=component, ids=[1])
    finally:
        r._reader.close()
    assert _skips(record) == [], [str(w.message) for w in _skips(record)]
