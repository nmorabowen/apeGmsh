"""MPCO Gauss and material-state discovery reads custom-rule buckets (#2115).

MPCO numbers custom integration rules from 1 across the whole file
(``209-SixNodeTri[1000:2:0]``), so ``custom_rule_idx`` names no layout of
its own. ``discover_gauss_buckets`` and ``discover_material_state_buckets``
used to ``continue`` on any ``custom_rule_idx != 0``: the catalog's
``IntRule.Custom`` rows (SixNodeTri, BezierTri6, BezierTet10) were never
reached and every such bucket vanished in silence.

Now a custom-rule bucket is read when RESPONSE_CATALOG has a row for its
class at the bucket's rule, with META checked against that row, and is
named in one ``MPCOBucketSkippedWarning`` per bucket when it has none.

Oracles: the values are the ones the test writes, decoded by a closed
form per (component, step, element, GP); the Gauss points are the
catalog's Triangle_GL_2 coordinates for SixNodeTri.
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

_TRI_KEY = f"{ELE_TAG_SixNodeTri}-SixNodeTri[{IntRule.Custom}:2:0]"
_TET_CUSTOM_KEY = (
    f"{ELE_TAG_FourNodeTetrahedron}-FourNodeTetrahedron[{IntRule.Custom}:3:0]"
)
_TET_CUSTOM_KEY_2 = (
    f"{ELE_TAG_FourNodeTetrahedron}-FourNodeTetrahedron[{IntRule.Custom}:4:0]"
)
_TET_STD_KEY = (
    f"{ELE_TAG_FourNodeTetrahedron}-FourNodeTetrahedron[{IntRule.Tet_GL_1}:0:0]"
)
_TRI_LAYOUT = lookup("SixNodeTri", IntRule.Custom, "stress")


def _open(path: Path) -> Results:
    return Results.from_mpco(str(path), model_h5=_stub_model_h5_path())


def _skips(record: "list[warnings.WarningMessage]") -> list[str]:
    return [
        str(w.message) for w in record
        if issubclass(w.category, MPCOBucketSkippedWarning)
    ]


def _encoded(T: int, E: int, G: int, k: int) -> np.ndarray:
    """``(T, E, G)`` values unique per (component k, step, element, GP)."""
    t, e, g = np.meshgrid(
        np.arange(T), np.arange(E), np.arange(G), indexing="ij",
    )
    return k * 1000.0 + t * 100.0 + e * 10.0 + g


def _skeleton(path: Path) -> "tuple[h5py.File, str]":
    return _create_mpco_skeleton(
        path, node_ids=np.arange(1, 7, dtype=np.int32),
        node_coords=np.zeros((6, 3)), n_steps=2, dt=0.5,
    )


def _add_tri_stress(stage: "h5py.Group", *, n_gauss_points: int = 3) -> None:
    """Two SixNodeTri elements on a custom rule, ``n_gauss_points`` blocks."""
    T, E = 2, 2
    per_comp = {
        name: _encoded(T, E, n_gauss_points, k)
        for k, name in enumerate(_TRI_LAYOUT.component_layout)
    }
    layout = _TRI_LAYOUT
    if n_gauss_points != layout.n_gauss_points:
        flat = np.stack(
            [per_comp[n] for n in layout.component_layout], axis=-1,
        ).reshape(T, E, -1)
    else:
        flat = flatten(per_comp, layout)
    _add_stress_bucket(
        stage, bracket_key=_TRI_KEY, class_tag=ELE_TAG_SixNodeTri,
        int_rule=IntRule.Custom,
        element_ids=np.array([7, 8], dtype=np.int32), flat_data=flat,
        n_gauss_points=n_gauss_points, n_components=3,
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
    stage: "h5py.Group", key: str, *, eids: np.ndarray, n_gp: int,
    num_columns: "int | None" = None,
) -> None:
    """A ``d+,d-`` material.damage bucket with ``n_gp`` META blocks."""
    T, E = 2, eids.size
    bg = stage["RESULTS"].require_group("ON_ELEMENTS").require_group(
        "material.damage",
    ).create_group(key)
    bg.attrs["NUM_COLUMNS"] = np.array(
        [num_columns if num_columns is not None else n_gp * 2],
        dtype=np.int32,
    )
    meta = bg.create_group("META")
    meta.create_dataset("MULTIPLICITY", data=np.ones((n_gp, 1), np.int32))
    meta.create_dataset(
        "GAUSS_IDS", data=np.arange(n_gp, dtype=np.int32).reshape(-1, 1),
    )
    meta.create_dataset("NUM_COMPONENTS", data=np.full((n_gp, 1), 2, np.int32))
    meta.create_dataset("COMPONENTS", data=np.array([
        ";".join("0.1.4.d+,d-" for _ in range(n_gp)).encode("ascii"),
    ]))
    bg.create_dataset("ID", data=eids.reshape(-1, 1).astype(np.int32))
    # GP-slowest, component-fastest: col(gp, k) = gp * 2 + k.
    flat = np.stack(
        [_encoded(T, E, n_gp, 0), _encoded(T, E, n_gp, 1)], axis=-1,
    ).reshape(T, E, n_gp * 2)
    data = bg.create_group("DATA")
    for t in range(T):
        ds = data.create_dataset(f"STEP_{t}", data=flat[t])
        ds.attrs["STEP"] = t
        ds.attrs["TIME"] = (t + 1) * 0.5


# =====================================================================
# Gauss (continuum stress): discover_gauss_buckets
# =====================================================================

class TestGaussCustomRule:
    def test_catalogued_custom_rule_bucket_reads(self, tmp_path: Path) -> None:
        """SixNodeTri ``[1000:2:0]`` reads through the catalog's Custom row."""
        path = tmp_path / "tri.mpco"
        f, stage = _skeleton(path)
        try:
            _add_tri_stress(f[stage])
        finally:
            f.close()
        r = _open(path)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", MPCOBucketSkippedWarning)
                comps = r._reader.available_components(
                    r._reader.stages()[0].id, ResultLevel.GAUSS,
                )
                slab = r.elements.gauss.get(component="stress_xy")
        finally:
            r._reader.close()
        assert "stress_xy" in comps
        # 2 elements x 3 GPs, element-major / GP-fastest.
        assert slab.values.shape == (2, 6)
        k = _TRI_LAYOUT.component_layout.index("stress_xy")
        np.testing.assert_array_equal(
            slab.values, _encoded(2, 2, 3, k).reshape(2, 6),
        )
        np.testing.assert_array_equal(
            slab.natural_coords,
            np.tile(_TRI_LAYOUT.natural_coords, (2, 1)),
        )

    def test_custom_rule_bucket_with_wrong_gp_count_refuses(
        self, tmp_path: Path,
    ) -> None:
        """META guards the layout: 4 GP blocks against a 3-GP row raise."""
        path = tmp_path / "tri4.mpco"
        f, stage = _skeleton(path)
        try:
            _add_tri_stress(f[stage], n_gauss_points=4)
        finally:
            f.close()
        r = _open(path)
        try:
            with pytest.raises(ValueError, match=r"4 blocks but catalog expects 3"):
                r.elements.gauss.get(component="stress_xx")
        finally:
            r._reader.close()

    def test_uncatalogued_custom_rule_buckets_warn_one_each(
        self, tmp_path: Path,
    ) -> None:
        """No (FourNodeTetrahedron, Custom) row: each bucket is named once,
        and the standard-rule bucket beside them still reads."""
        path = tmp_path / "tets.mpco"
        f, stage = _skeleton(path)
        try:
            _add_tet_stress(f[stage], _TET_STD_KEY, 1)
            _add_tet_stress(f[stage], _TET_CUSTOM_KEY, 2)
            _add_tet_stress(f[stage], _TET_CUSTOM_KEY_2, 3)
        finally:
            f.close()
        r = _open(path)
        try:
            with pytest.warns(MPCOBucketSkippedWarning) as rec:
                slab = r.elements.gauss.get(component="stress_xx")
        finally:
            r._reader.close()
        np.testing.assert_array_equal(slab.values, np.full((2, 1), 1.0))
        msgs = _skips(rec)
        assert len(msgs) == 2
        for key, msg in zip((_TET_CUSTOM_KEY, _TET_CUSTOM_KEY_2), sorted(msgs)):
            assert f"stresses/{key}" in msg
            assert "'FourNodeTetrahedron'" in msg
            assert " in tets.mpco " in msg
            assert "'stress_xx'" in msg

    def test_listing_names_an_uncatalogued_custom_rule_bucket(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "tet.mpco"
        f, stage = _skeleton(path)
        try:
            _add_tet_stress(f[stage], _TET_CUSTOM_KEY, 2)
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
        assert any(f"stresses/{_TET_CUSTOM_KEY}" in m for m in _skips(rec))


# =====================================================================
# Material state: discover_material_state_buckets
# =====================================================================

_TRI_DAMAGE_KEY = f"{ELE_TAG_SixNodeTri}-SixNodeTri[{IntRule.Custom}:1:0]"
_TET_DAMAGE_KEY = (
    f"{ELE_TAG_FourNodeTetrahedron}-FourNodeTetrahedron[{IntRule.Custom}:1:0]"
)


class TestMaterialStateCustomRule:
    def test_catalogued_custom_rule_bucket_reads(self, tmp_path: Path) -> None:
        path = tmp_path / "tri_damage.mpco"
        f, stage = _skeleton(path)
        try:
            _add_damage(
                f[stage], _TRI_DAMAGE_KEY,
                eids=np.array([7, 8], dtype=np.int64), n_gp=3,
            )
        finally:
            f.close()
        r = _open(path)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", MPCOBucketSkippedWarning)
                comps = r._reader.available_components(
                    r._reader.stages()[0].id, ResultLevel.GAUSS,
                )
                slab = r.elements.gauss.get(component="damage_compression")
        finally:
            r._reader.close()
        assert {"damage_tension", "damage_compression"} <= set(comps)
        assert slab.values.shape == (2, 6)
        np.testing.assert_array_equal(
            slab.values, _encoded(2, 2, 3, 1).reshape(2, 6),
        )
        np.testing.assert_array_equal(
            slab.natural_coords,
            np.tile(_TRI_LAYOUT.natural_coords, (2, 1)),
        )

    def test_custom_rule_bucket_with_wrong_gp_count_is_named(
        self, tmp_path: Path,
    ) -> None:
        """META guards the layout: 4 GPs of d+,d- against the 3-GP row."""
        path = tmp_path / "tri_damage4.mpco"
        f, stage = _skeleton(path)
        try:
            _add_damage(
                f[stage], _TRI_DAMAGE_KEY,
                eids=np.array([7], dtype=np.int64), n_gp=4,
            )
        finally:
            f.close()
        r = _open(path)
        try:
            with pytest.warns(MPCOBucketSkippedWarning) as rec:
                slab = r.elements.gauss.get(component="damage_tension")
        finally:
            r._reader.close()
        assert slab.values.shape[1] == 0
        (msg,) = _skips(rec)
        assert f"material.damage/{_TRI_DAMAGE_KEY}" in msg
        assert "NUM_COLUMNS=8" in msg

    def test_uncatalogued_custom_rule_bucket_is_named(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "tet_damage.mpco"
        f, stage = _skeleton(path)
        try:
            _add_damage(
                f[stage], _TET_DAMAGE_KEY,
                eids=np.array([2], dtype=np.int64), n_gp=1,
            )
        finally:
            f.close()
        r = _open(path)
        try:
            with pytest.warns(MPCOBucketSkippedWarning) as rec:
                slab = r.elements.gauss.get(component="damage_tension")
        finally:
            r._reader.close()
        assert slab.values.shape[1] == 0
        (msg,) = _skips(rec)
        assert f"material.damage/{_TET_DAMAGE_KEY}" in msg
        assert "'FourNodeTetrahedron'" in msg
        assert "custom integration rule 1" in msg
        assert " in tet_damage.mpco " in msg
