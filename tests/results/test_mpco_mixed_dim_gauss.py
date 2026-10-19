"""#2165 remainder part 1: ``_mpco.py`` reads that join buckets refuse
what they cannot join.

1. **Mixed natural-coordinate widths.** A :class:`GaussSlab` has one
   ``natural_coords`` array, ``(sum_GP, dim)``, so one read cannot carry a
   2-D class (``FourNodeQuad``, ``(xi, eta)``) beside a 3-D class
   (``FourNodeTetrahedron``, ``(xi, eta, zeta)``). Before the fix the join
   crashed in ``np.concatenate``; now it raises
   :class:`MPCOGaussDimensionMixed` naming both classes. A read narrowed
   to one class's elements still answers.
2. **An element id in two buckets.** MPCO writes each element to one
   bucket per result, so an id in two buckets is a corrupt or merged
   file, and the join would return that element twice. The Gauss,
   material-state, line-station and fiber joins raise
   :class:`MPCOElementInTwoBuckets` naming the id, both bucket paths and
   the file (the nodal-force counterpart is #2173).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.opensees._response_catalog import (
    ELE_TAG_DispBeamColumn3d,
    ELE_TAG_ForceBeamColumn3d,
    ELE_TAG_FourNodeQuad,
    ELE_TAG_FourNodeTetrahedron,
    IntRule,
)
from apeGmsh.results import Results
from apeGmsh.results.readers._mpco import (
    MPCOElementInTwoBuckets,
    MPCOGaussDimensionMixed,
    MPCOReader,
)
from apeGmsh.results.readers._tag_translation import ElementTagTranslator
from tests.conftest import _stub_model_h5_path
from tests.test_results_mpco_element_mock import (
    _add_stress_bucket,
    _create_mpco_skeleton,
)
from tests.test_results_mpco_fiber_mock import (
    _add_connectivity,
    _add_fiber_bucket,
    _add_fiber_section,
    _build_unique_flat,
)
from tests.test_results_mpco_fiber_mock import (
    _create_skeleton as _create_fiber_skeleton,
)
from tests.test_results_mpco_line_mock import (
    _MPCO_TO_CANONICAL,
    _add_line_station_bucket,
    _build_flat,
)
from tests.test_results_mpco_layer_mock import (  # noqa: F401 (fixture)
    simple_layer_mpco,
)
from tests.test_results_mpco_line_mock import (
    _create_skeleton as _create_line_skeleton,
)
from tests.test_results_mpco_spring_mock import _add_bucket as _add_spring_bucket
from tests.test_results_mpco_spring_mock import _make_spring_mpco

_QUAD = f"{ELE_TAG_FourNodeQuad}-FourNodeQuad[{IntRule.Quad_GL_2}:0:0]"
_TET = f"{ELE_TAG_FourNodeTetrahedron}-FourNodeTetrahedron[{IntRule.Tet_GL_1}:0:0]"
_TET_H1 = f"{ELE_TAG_FourNodeTetrahedron}-FourNodeTetrahedron[{IntRule.Tet_GL_1}:0:1]"


def _gauss(path: Path, component: str, **kw):
    with Results.from_mpco(path, model_h5=_stub_model_h5_path()) as r:
        return r.stage(r.stages[0].id).elements.gauss.get(
            component=component, **kw,
        )


# =====================================================================
# Stress (catalog path)
# =====================================================================

def _stress_file(path: Path, buckets: list[tuple[str, list[int]]]) -> Path:
    """One step; a quad bucket holds 4 GP x 3 comps, a tet 1 GP x 6.

    ``stress_xx`` of element ``e`` at every GP is ``e``.
    """
    f, stage = _create_mpco_skeleton(
        path, node_ids=np.arange(1, 5), node_coords=np.eye(4, 3),
        n_steps=1, dt=1.0,
    )
    try:
        for key, ids in buckets:
            eids = np.asarray(ids)
            if "FourNodeQuad" in key:
                n_gp, labels = 4, ("sigma_xx", "sigma_yy", "sigma_xy")
            else:
                n_gp, labels = 1, (
                    "sigma_xx", "sigma_yy", "sigma_zz",
                    "sigma_xy", "sigma_yz", "sigma_xz",
                )
            flat = np.zeros((1, eids.size, n_gp * len(labels)))
            for g in range(n_gp):
                flat[0, :, g * len(labels)] = eids
            _add_stress_bucket(
                f[stage], bracket_key=key, class_tag=0, int_rule=0,
                element_ids=eids, flat_data=flat, n_gauss_points=n_gp,
                n_components=len(labels), component_labels=labels,
            )
    finally:
        f.close()
    return path


class TestMixedDimGauss:
    def test_quad_beside_tet_refuses_naming_both(self, tmp_path: Path) -> None:
        path = _stress_file(
            tmp_path / "mixed.mpco", [(_QUAD, [5]), (_TET, [10])],
        )
        with pytest.raises(MPCOGaussDimensionMixed) as info:
            _gauss(path, "stress_xx")
        msg = str(info.value)
        assert "FourNodeQuad" in msg and "FourNodeTetrahedron" in msg
        assert "mixed.mpco" in msg

    @pytest.mark.parametrize("ids, dim, n", [([5], 2, 4), ([10], 3, 1)])
    def test_one_class_read_answers(
        self, tmp_path: Path, ids: list[int], dim: int, n: int,
    ) -> None:
        """Narrowed to one class, the read is whole and unpadded."""
        path = _stress_file(
            tmp_path / "mixed.mpco", [(_QUAD, [5]), (_TET, [10])],
        )
        slab = _gauss(path, "stress_xx", ids=ids)
        assert slab.natural_coords.shape == (n, dim)
        np.testing.assert_array_equal(slab.element_index, ids * n)
        np.testing.assert_array_equal(slab.values, [[float(ids[0])] * n])


class TestDuplicateIdGauss:
    def test_header_0_and_1_share_an_element(self, tmp_path: Path) -> None:
        path = _stress_file(
            tmp_path / "dup.mpco", [(_TET, [7, 11]), (_TET_H1, [11, 33])],
        )
        with pytest.raises(MPCOElementInTwoBuckets) as info:
            _gauss(path, "stress_xx")
        msg = str(info.value)
        assert "element 11 " in msg and "dup.mpco" in msg
        assert _TET in msg and _TET_H1 in msg

    def test_disjoint_headers_read_unchanged(self, tmp_path: Path) -> None:
        path = _stress_file(
            tmp_path / "ok.mpco", [(_TET, [7, 11]), (_TET_H1, [33])],
        )
        slab = _gauss(path, "stress_xx")
        np.testing.assert_array_equal(slab.element_index, [7, 11, 33])
        np.testing.assert_array_equal(slab.values, [[7.0, 11.0, 33.0]])

    def test_a_read_that_skips_the_duplicate_answers(
        self, tmp_path: Path,
    ) -> None:
        """The refusal is about what the read joins, not the whole file."""
        path = _stress_file(
            tmp_path / "dup.mpco", [(_TET, [7, 11]), (_TET_H1, [11, 33])],
        )
        slab = _gauss(path, "stress_xx", ids=[7, 33])
        np.testing.assert_array_equal(slab.element_index, [7, 33])


# =====================================================================
# Material state (META path)
# =====================================================================

def _damage_file(path: Path, buckets: list[tuple[str, list[int]]]) -> Path:
    """``material.damage`` buckets, one ``d`` per GP, value = element id."""
    f, stage = _create_mpco_skeleton(
        path, node_ids=np.arange(1, 5), node_coords=np.eye(4, 3),
        n_steps=1, dt=1.0,
    )
    try:
        token = f[stage]["RESULTS"].require_group("ON_ELEMENTS").require_group(
            "material.damage",
        )
        for key, ids in buckets:
            n_gp = 4 if "FourNodeQuad" in key else 1
            eids = np.asarray(ids)
            bg = token.create_group(key)
            bg.attrs["NUM_COLUMNS"] = np.array([n_gp], dtype=np.int32)
            meta = bg.create_group("META")
            meta.create_dataset("MULTIPLICITY", data=np.ones((n_gp, 1), np.int32))
            meta.create_dataset(
                "GAUSS_IDS", data=np.arange(n_gp, dtype=np.int32)[:, None],
            )
            meta.create_dataset(
                "NUM_COMPONENTS", data=np.ones((n_gp, 1), np.int32),
            )
            meta.create_dataset("COMPONENTS", data=np.array([
                ";".join("0.1.4.d" for _ in range(n_gp)).encode("ascii"),
            ]))
            bg.create_dataset("ID", data=eids.astype(np.int32)[:, None])
            ds = bg.create_group("DATA").create_dataset(
                "STEP_0", data=np.repeat(eids[:, None], n_gp, 1).astype(float),
            )
            ds.attrs["STEP"] = 0
            ds.attrs["TIME"] = 1.0
    finally:
        f.close()
    return path


class TestMaterialState:
    def test_quad_beside_tet_refuses_naming_both(self, tmp_path: Path) -> None:
        path = _damage_file(
            tmp_path / "mixed_d.mpco", [(_QUAD, [5]), (_TET, [10])],
        )
        with pytest.raises(MPCOGaussDimensionMixed) as info:
            _gauss(path, "damage")
        msg = str(info.value)
        assert "FourNodeQuad" in msg and "FourNodeTetrahedron" in msg

    def test_one_class_read_answers(self, tmp_path: Path) -> None:
        path = _damage_file(
            tmp_path / "mixed_d.mpco", [(_QUAD, [5]), (_TET, [10])],
        )
        slab = _gauss(path, "damage", ids=[5])
        assert slab.natural_coords.shape == (4, 2)
        np.testing.assert_array_equal(slab.values, [[5.0] * 4])

    def test_duplicate_id_refuses(self, tmp_path: Path) -> None:
        path = _damage_file(
            tmp_path / "dup_d.mpco", [(_TET, [7, 11]), (_TET_H1, [11, 33])],
        )
        with pytest.raises(MPCOElementInTwoBuckets, match="element 11 "):
            _gauss(path, "damage")


# =====================================================================
# Line stations
# =====================================================================

_SECTION = ("P", "Mz", "My", "T", "Vy", "Vz")


def _line_stations(path: Path, **kw):
    with Results.from_mpco(path, model_h5=_stub_model_h5_path()) as r:
        return r.stage(r.stages[0].id).elements.line_stations.get(
            component="axial_force", **kw,
        )


def _section_force_file(path: Path, ids_1: list[int], ids_2: list[int]) -> Path:
    f, stage = _create_line_skeleton(
        path, node_ids=np.arange(1, 5), coords=np.zeros((4, 3)),
        n_steps=1, dt=1.0,
    )
    try:
        layout = tuple(_MPCO_TO_CANONICAL[n] for n in _SECTION)
        for cust, ids, gp_x in (
            (1, ids_1, np.linspace(-1.0, 1.0, 5)),
            (2, ids_2, np.array([-0.7745966, 0.0, 0.7745966])),
        ):
            if not ids:
                continue
            eids = np.asarray(ids, dtype=np.int32)
            flat, _ = _build_flat(
                component_layout=layout, n_ip=gp_x.size,
                element_ids=eids, n_steps=1,
            )
            _add_line_station_bucket(
                f[stage], class_tag=ELE_TAG_ForceBeamColumn3d,
                class_name="ForceBeamColumn3d", custom_rule_idx=cust,
                header_idx=0, element_ids=eids, gp_x=gp_x,
                section_names=_SECTION, flat_data=flat,
            )
    finally:
        f.close()
    return path


_LF_RAW = b"0.N_1,Vy_1,Vz_1,T_1,My_1,Mz_1,N_2,Vy_2,Vz_2,T_2,My_2,Mz_2"


def _local_force_file(path: Path, buckets: dict[str, list[int]]) -> Path:
    f, stage = _create_line_skeleton(
        path, node_ids=np.arange(1, 5), coords=np.zeros((4, 3)),
        n_steps=1, dt=1.0,
    )
    try:
        lf = f[stage]["RESULTS/ON_ELEMENTS"].create_group("localForce")
        for key, ids in buckets.items():
            g = lf.create_group(key)
            g.create_dataset("META/COMPONENTS", data=np.array([_LF_RAW]))
            g.create_dataset("ID", data=np.asarray(ids, np.int64)[:, None])
            ds = g.create_group("DATA").create_dataset(
                "STEP_0", data=np.tile(np.arange(1.0, 13.0), (len(ids), 1)),
            )
            ds.attrs["STEP"] = 0
            ds.attrs["TIME"] = 1.0
    finally:
        f.close()
    return path


class TestDuplicateIdLineStations:
    def test_section_force_two_custom_rules(self, tmp_path: Path) -> None:
        path = _section_force_file(tmp_path / "dup_sf.mpco", [1, 2], [2, 3])
        with pytest.raises(MPCOElementInTwoBuckets) as info:
            _line_stations(path)
        msg = str(info.value)
        assert "element 2 " in msg and "dup_sf.mpco" in msg
        assert "[1000:1:0]" in msg and "[1000:2:0]" in msg

    def test_section_force_disjoint_unchanged(self, tmp_path: Path) -> None:
        path = _section_force_file(tmp_path / "ok_sf.mpco", [1, 2], [3])
        slab = _line_stations(path)
        assert slab.values.shape == (1, 13)

    def test_local_force_two_headers(self, tmp_path: Path) -> None:
        """Before the fix the second bucket's copy of element 2 was
        dropped silently, as if a higher rank had covered it."""
        path = _local_force_file(tmp_path / "dup_lf.mpco", {
            "5-ElasticBeam3d[1:0:0]": [1, 2],
            "5-ElasticBeam3d[1:0:1]": [2, 3],
        })
        with pytest.raises(MPCOElementInTwoBuckets) as info:
            _line_stations(path)
        msg = str(info.value)
        assert "element 2 " in msg
        assert "[1:0:0]" in msg and "[1:0:1]" in msg

    def test_local_force_disjoint_unchanged(self, tmp_path: Path) -> None:
        path = _local_force_file(tmp_path / "ok_lf.mpco", {
            "5-ElasticBeam3d[1:0:0]": [1, 2],
            "5-ElasticBeam3d[1:0:1]": [3],
        })
        slab = _line_stations(path)
        np.testing.assert_array_equal(slab.element_index, [1, 1, 2, 2, 3, 3])


# =====================================================================
# Fibers
# =====================================================================

def _fiber_file(path: Path, ids_force: list[int], ids_disp: list[int]) -> Path:
    """A ForceBeamColumn3d bucket and a DispBeamColumn3d bucket, 3 IP x 4
    fibers each, under one fiber section."""
    f, stage = _create_fiber_skeleton(
        path, node_ids=np.arange(1, 4), coords=np.zeros((3, 3)),
        n_steps=1, dt=1.0,
    )
    try:
        gp_x = np.array([-0.7745966, 0.0, 0.7745966])
        pairs = []
        for tag, name, ids in (
            (ELE_TAG_ForceBeamColumn3d, "ForceBeamColumn3d", ids_force),
            (ELE_TAG_DispBeamColumn3d, "DispBeamColumn3d", ids_disp),
        ):
            eids = np.asarray(ids, dtype=np.int64)
            _add_connectivity(
                f[stage], class_tag=tag, class_name=name, custom_rule_idx=0,
                element_ids=eids, gp_x=gp_x,
            )
            _add_fiber_bucket(
                f[stage], class_tag=tag, class_name=name, custom_rule_idx=0,
                element_ids=eids, n_ip=3, n_fibers=4,
                flat_data=_build_unique_flat(T=1, E=eids.size, n_ip=3, n_fibers=4),
            )
            pairs += [(int(e), gp) for e in eids for gp in range(3)]
        _add_fiber_section(
            f[stage], section_tag=99, section_class="FiberSection3d",
            element_gp_pairs=sorted(set(pairs)),
            fiber_y=np.array([-0.1, 0.1, 0.1, -0.1]),
            fiber_z=np.array([-0.05, -0.05, 0.05, 0.05]),
            fiber_area=np.full(4, 0.01), fiber_material=np.full(4, 42),
        )
    finally:
        f.close()
    return path


def _fibers(path: Path):
    with Results.from_mpco(path, model_h5=_stub_model_h5_path()) as r:
        return r.stage(r.stages[0].id).elements.fibers.get(
            component="fiber_stress",
        )


class TestDuplicateIdFibers:
    def test_two_classes_share_an_element(self, tmp_path: Path) -> None:
        path = _fiber_file(tmp_path / "dup_fib.mpco", [10, 11], [11, 12])
        with pytest.raises(MPCOElementInTwoBuckets) as info:
            _fibers(path)
        msg = str(info.value)
        assert "element 11 " in msg and "dup_fib.mpco" in msg
        assert "ForceBeamColumn3d" in msg and "DispBeamColumn3d" in msg

    def test_disjoint_unchanged(self, tmp_path: Path) -> None:
        path = _fiber_file(tmp_path / "ok_fib.mpco", [10, 11], [12])
        slab = _fibers(path)
        assert sorted(set(slab.element_index.tolist())) == [10, 11, 12]
        assert slab.values.shape == (1, 3 * 3 * 4)


# =====================================================================
# Ranked line-station sources: a higher rank's element is not a repeat
# =====================================================================

def test_section_force_covers_an_element_two_local_force_buckets_hold(
    tmp_path: Path,
) -> None:
    """section.force [1, 2] beside localForce buckets [2, 9] and [2, 8]:
    element 2 is section.force's, so the localForce buckets return 9 and
    8 only, and the read answers rather than calling 2 a repeat."""
    path = _section_force_file(tmp_path / "ranked.mpco", [1, 2], [])
    with h5py.File(path, "a") as f:
        lf = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"].create_group("localForce")
        for key, ids in (
            ("5-ElasticBeam3d[1:0:0]", [2, 9]),
            ("5-ElasticBeam3d[1:0:1]", [2, 8]),
        ):
            g = lf.create_group(key)
            g.create_dataset("META/COMPONENTS", data=np.array([_LF_RAW]))
            g.create_dataset("ID", data=np.asarray(ids, np.int64)[:, None])
            ds = g.create_group("DATA").create_dataset(
                "STEP_0", data=np.tile(np.arange(1.0, 13.0), (2, 1)),
            )
            ds.attrs["STEP"] = 0
            ds.attrs["TIME"] = 1.0
    slab = _line_stations(path)
    # 2 section.force elements x 5 IPs, then 9 and 8 at their two ends.
    np.testing.assert_array_equal(
        slab.element_index, [1] * 5 + [2] * 5 + [9, 9, 8, 8],
    )


# =====================================================================
# An id twice inside one bucket; the fem id when a tag map is attached
# =====================================================================

def test_gauss_id_twice_in_one_bucket_refuses(tmp_path: Path) -> None:
    path = _stress_file(tmp_path / "twice.mpco", [(_TET, [7, 7, 11])])
    with pytest.raises(MPCOElementInTwoBuckets) as info:
        _gauss(path, "stress_xx")
    msg = str(info.value)
    assert "element 7 " in msg and "twice in one bucket" in msg
    assert _TET in msg


def test_error_names_the_fem_id_under_a_tag_map(tmp_path: Path) -> None:
    path = _stress_file(
        tmp_path / "dup.mpco", [(_TET, [7, 11]), (_TET_H1, [11, 33])],
    )
    reader = MPCOReader(path)
    try:
        reader.attach_tag_map(ElementTagTranslator(
            {107: 7, 111: 11, 133: 33}, {7: 107, 11: 111, 33: 133},
        ))
        with pytest.raises(MPCOElementInTwoBuckets) as info:
            reader.read_gauss(reader.stages()[0].id, "stress_xx")
    finally:
        reader.close()
    assert "element 11 (OpenSees tag; fem id 111)" in str(info.value)


# =====================================================================
# Springs and layers
# =====================================================================

def test_springs_two_headers_share_an_element(tmp_path: Path) -> None:
    """The reviewer's repro: [1, 2] at ``[1:0:0]``, [2, 3] at ``[1:0:1]``
    read back as [1, 2, 2, 3] with no word."""
    path = _make_spring_mpco(
        tmp_path / "dup_spr.mpco", element_ids=[1, 2], n_springs=1,
        force_values=np.full((2, 2, 1), 3.0),
    )
    _add_spring_bucket(path, "basicForce", "19-ZeroLength[1:0:1]",
                       element_ids=[2, 3], values=np.full((2, 2, 1), 9.0))
    with Results.from_mpco(path, model_h5=_stub_model_h5_path()) as r:
        with pytest.raises(MPCOElementInTwoBuckets) as info:
            r.elements.springs.get(component="spring_force")
    msg = str(info.value)
    assert "element 2 " in msg and "dup_spr.mpco" in msg
    assert "[1:0:0]" in msg and "[1:0:1]" in msg


def test_springs_disjoint_headers_unchanged(tmp_path: Path) -> None:
    path = _make_spring_mpco(
        tmp_path / "ok_spr.mpco", element_ids=[1, 2], n_springs=1,
        force_values=np.full((2, 2, 1), 3.0),
    )
    _add_spring_bucket(path, "basicForce", "19-ZeroLength[1:0:1]",
                       element_ids=[3], values=np.full((2, 1, 1), 9.0))
    with Results.from_mpco(path, model_h5=_stub_model_h5_path()) as r:
        slab = r.elements.springs.get(component="spring_force")
    np.testing.assert_array_equal(slab.element_index, [1, 2, 3])


def test_layers_two_headers_share_an_element(
    simple_layer_mpco,  # noqa: F811 (the imported fixture)
) -> None:
    """A copy of the ASDShellQ4 bucket at header 1 holds 20 and 21 again."""
    path = simple_layer_mpco["path"]
    key = simple_layer_mpco["bucket_key"]
    copy = key[:-2] + "1]"
    with h5py.File(path, "a") as f:
        token = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS/material.fiber.stress"]
        token.copy(key, copy)
    with Results.from_mpco(path, model_h5=_stub_model_h5_path()) as r:
        with pytest.raises(MPCOElementInTwoBuckets) as info:
            r.elements.layers.get(component="fiber_stress")
    msg = str(info.value)
    assert "element 20 " in msg
    assert key in msg and copy in msg
