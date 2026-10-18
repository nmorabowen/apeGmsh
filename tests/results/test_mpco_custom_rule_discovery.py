"""MPCO Gauss and material-state discovery names custom-rule buckets (#2115).

``discover_gauss_buckets`` and ``discover_material_state_buckets`` used to
``continue`` on any ``custom_rule_idx != 0``, so such a bucket vanished
from listings and reads without a word.

MPCO opens a new custom-rule index for each distinct set of Gauss-point
positions (``GP_X``), so the catalog's fixed coordinates cannot be trusted
for any of them: two indices on one class mean that at least one of them
is not the catalog's rule. Until a reader verifies ``GP_X`` against a real
recorder file, every bucket with a nonzero custom-rule index is left
unread and named in one ``MPCOBucketSkippedWarning`` (path, class, custom
index, file), whatever its bracket rule, and the other buckets still read.

The three pinned reproducers are the #2159 review's: ``[1000:1:0]`` beside
``[1000:2:0]``, ``[0:1:0]`` (the rule-0 class fallback) and ``[101:5:0]``
(a standard rule's catalog row), each of which an earlier draft read with
the catalog's Gauss points and no warning.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.opensees._response_catalog import (
    ELE_TAG_FourNodeTetrahedron,
    ELE_TAG_SixNodeTri,
    IntRule,
    flatten,
    lookup,
)
from apeGmsh.results import Results
from apeGmsh.results.readers._mpco import MPCOBucketSkippedWarning
from apeGmsh.results.readers._protocol import ResultLevel
from tests.conftest import _stub_model_h5_path
from tests.test_results_mpco_element_mock import (
    _add_stress_bucket,
    _create_mpco_skeleton,
)

_TRI_LAYOUT = lookup("SixNodeTri", IntRule.Triangle_GL_2, "stress")


def _tri(rule: int, cust: int) -> str:
    return f"{ELE_TAG_SixNodeTri}-SixNodeTri[{rule}:{cust}:0]"


def _tet(rule: int, cust: int) -> str:
    return f"{ELE_TAG_FourNodeTetrahedron}-FourNodeTetrahedron[{rule}:{cust}:0]"


#: The #2159 review's reproducers: each tuple is one file's SixNodeTri
#: buckets. Every bucket here must be named, and none may read.
_REPRODUCERS = {
    "two-custom-indices": (_tri(IntRule.Custom, 1), _tri(IntRule.Custom, 2)),
    "rule-0-nonzero-index": (_tri(IntRule.NoIntegrationRule, 1),),
    "standard-rule-nonzero-index": (_tri(IntRule.Triangle_GL_2, 5),),
}


def _open(path: Path) -> Results:
    return Results.from_mpco(str(path), model_h5=_stub_model_h5_path())


def _skips(record: "list[warnings.WarningMessage]") -> list[str]:
    return [
        str(w.message) for w in record
        if issubclass(w.category, MPCOBucketSkippedWarning)
    ]


def _skeleton(path: Path) -> "tuple[h5py.File, str]":
    return _create_mpco_skeleton(
        path, node_ids=np.arange(1, 7, dtype=np.int32),
        node_coords=np.zeros((6, 3)), n_steps=2, dt=0.5,
    )


def _add_tri_stress(stage: "h5py.Group", key: str, eid: int) -> None:
    """One well-formed 3-GP SixNodeTri stress bucket under ``key``."""
    per_comp = {
        name: np.full((2, 1, 3), float(eid + k))
        for k, name in enumerate(_TRI_LAYOUT.component_layout)
    }
    _add_stress_bucket(
        stage, bracket_key=key, class_tag=ELE_TAG_SixNodeTri,
        int_rule=IntRule.Triangle_GL_2,
        element_ids=np.array([eid], dtype=np.int32),
        flat_data=flatten(per_comp, _TRI_LAYOUT), n_gauss_points=3,
        n_components=3,
        component_labels=("sigma_xx", "sigma_yy", "sigma_xy"),
    )


def _add_tet_stress(stage: "h5py.Group", key: str, eid: int) -> None:
    """One single-GP FourNodeTetrahedron bucket under ``key``."""
    _add_stress_bucket(
        stage, bracket_key=key, class_tag=ELE_TAG_FourNodeTetrahedron,
        int_rule=IntRule.Tet_GL_1,
        element_ids=np.array([eid], dtype=np.int32),
        flat_data=np.full((2, 1, 6), float(eid)), n_gauss_points=1,
    )


def _add_damage(
    stage: "h5py.Group", key: str, *, eid: int, n_gp: int,
) -> None:
    """A well-formed ``d+,d-`` material.damage bucket with ``n_gp`` GPs."""
    bg = stage["RESULTS"].require_group("ON_ELEMENTS").require_group(
        "material.damage",
    ).create_group(key)
    bg.attrs["NUM_COLUMNS"] = np.array([n_gp * 2], dtype=np.int32)
    meta = bg.create_group("META")
    meta.create_dataset("MULTIPLICITY", data=np.ones((n_gp, 1), np.int32))
    meta.create_dataset(
        "GAUSS_IDS", data=np.arange(n_gp, dtype=np.int32).reshape(-1, 1),
    )
    meta.create_dataset("NUM_COMPONENTS", data=np.full((n_gp, 1), 2, np.int32))
    meta.create_dataset("COMPONENTS", data=np.array([
        ";".join("0.1.4.d+,d-" for _ in range(n_gp)).encode("ascii"),
    ]))
    bg.create_dataset("ID", data=np.array([[eid]], dtype=np.int32))
    data = bg.create_group("DATA")
    for t in range(2):
        ds = data.create_dataset(
            f"STEP_{t}", data=np.full((1, n_gp * 2), float(eid)),
        )
        ds.attrs["STEP"] = t
        ds.attrs["TIME"] = (t + 1) * 0.5


def _assert_named(msgs: list[str], keys: "tuple[str, ...]", *,
                  group: str, cls: str, file: str, what: str) -> None:
    """One warning per key, each naming path, class, index and file."""
    assert len(msgs) == len(keys), msgs
    for key in keys:
        (msg,) = [m for m in msgs if f"{group}/{key}:" in m]
        cust = key.split(":")[-2]
        assert f"class {cls!r}" in msg
        assert f"custom integration rule {cust}" in msg
        assert f" in {file} " in msg
        assert what in msg


# =====================================================================
# Gauss (continuum stress): discover_gauss_buckets
# =====================================================================

class TestGaussCustomRule:
    @pytest.mark.parametrize("case", sorted(_REPRODUCERS))
    def test_reproducer_is_named_and_unread(
        self, tmp_path: Path, case: str,
    ) -> None:
        keys = _REPRODUCERS[case]
        path = tmp_path / "tri.mpco"
        f, stage = _skeleton(path)
        try:
            for i, key in enumerate(keys):
                _add_tri_stress(f[stage], key, 7 + i)
        finally:
            f.close()
        r = _open(path)
        try:
            with pytest.warns(MPCOBucketSkippedWarning) as rec:
                slab = r.elements.gauss.get(component="stress_xx")
        finally:
            r._reader.close()
        assert slab.values.shape == (2, 0)
        _assert_named(
            _skips(rec), keys, group="stresses", cls="SixNodeTri",
            file="tri.mpco", what="gauss discovery of 'stress_xx'",
        )

    def test_standard_bucket_beside_custom_ones_still_reads(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "tets.mpco"
        keys = (_tet(IntRule.Custom, 3), _tet(IntRule.Custom, 4))
        f, stage = _skeleton(path)
        try:
            _add_tet_stress(f[stage], _tet(IntRule.Tet_GL_1, 0), 1)
            for i, key in enumerate(keys):
                _add_tet_stress(f[stage], key, 2 + i)
        finally:
            f.close()
        r = _open(path)
        try:
            with pytest.warns(MPCOBucketSkippedWarning) as rec:
                slab = r.elements.gauss.get(component="stress_xx")
        finally:
            r._reader.close()
        np.testing.assert_array_equal(slab.values, np.full((2, 1), 1.0))
        _assert_named(
            _skips(rec), keys, group="stresses", cls="FourNodeTetrahedron",
            file="tets.mpco", what="gauss discovery of 'stress_xx'",
        )

    def test_listing_names_a_custom_rule_bucket(self, tmp_path: Path) -> None:
        key = _tri(IntRule.Custom, 1)
        path = tmp_path / "tri.mpco"
        f, stage = _skeleton(path)
        try:
            _add_tri_stress(f[stage], key, 7)
        finally:
            f.close()
        r = _open(path)
        try:
            with pytest.warns(MPCOBucketSkippedWarning) as rec:
                comps = r._reader.available_components(
                    r._reader.stages()[0].id, ResultLevel.GAUSS,
                )
        finally:
            r._reader.close()
        assert "stress_xx" not in comps
        assert any(f"stresses/{key}:" in m for m in _skips(rec))


# =====================================================================
# Material state: discover_material_state_buckets
# =====================================================================

class TestMaterialStateCustomRule:
    @pytest.mark.parametrize("case", sorted(_REPRODUCERS))
    def test_reproducer_is_named_and_unread(
        self, tmp_path: Path, case: str,
    ) -> None:
        keys = _REPRODUCERS[case]
        path = tmp_path / "tri_damage.mpco"
        f, stage = _skeleton(path)
        try:
            for i, key in enumerate(keys):
                _add_damage(f[stage], key, eid=7 + i, n_gp=3)
        finally:
            f.close()
        r = _open(path)
        try:
            with pytest.warns(MPCOBucketSkippedWarning) as rec:
                slab = r.elements.gauss.get(component="damage_tension")
        finally:
            r._reader.close()
        assert slab.values.shape == (2, 0)
        _assert_named(
            _skips(rec), keys, group="material.damage", cls="SixNodeTri",
            file="tri_damage.mpco",
            what="gauss material-state discovery of 'damage_tension'",
        )

    def test_standard_bucket_beside_a_custom_one_still_reads(
        self, tmp_path: Path,
    ) -> None:
        key = _tet(IntRule.Custom, 1)
        path = tmp_path / "tet_damage.mpco"
        f, stage = _skeleton(path)
        try:
            _add_damage(f[stage], _tet(IntRule.Tet_GL_1, 0), eid=1, n_gp=1)
            _add_damage(f[stage], key, eid=2, n_gp=1)
        finally:
            f.close()
        r = _open(path)
        try:
            with pytest.warns(MPCOBucketSkippedWarning) as rec:
                slab = r.elements.gauss.get(component="damage_tension")
        finally:
            r._reader.close()
        np.testing.assert_array_equal(slab.values, np.full((2, 1), 1.0))
        _assert_named(
            _skips(rec), (key,), group="material.damage",
            cls="FourNodeTetrahedron", file="tet_damage.mpco",
            what="gauss material-state discovery of 'damage_tension'",
        )
