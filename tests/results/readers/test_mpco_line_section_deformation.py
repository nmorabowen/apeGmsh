"""MPCO line-station section deformations (#2053).

A beam's ``-E section.deformation`` bucket names its META components with
the ``ResponseType`` tags of ``SectionForceDeformation::setResponse``
(fork 79e062367, ``SRC/material/section/SectionForceDeformation.cpp``
L651-672): ``eps`` (P), ``kappaZ`` (Mz), ``kappaY`` (My), ``theta`` (T),
``gammaY`` (Vy), ``gammaZ`` (Vz), in the section's own ``getType()`` order.
No section subclass overrides them, so the section kinds differ only in
which codes appear: a 2-D fiber or elastic section ``eps,kappaZ``; a 3-D
fiber section with torsion ``eps,kappaZ,kappaY,theta``; a shear-deformable
section (an aggregator with Vy/Vz, a Timoshenko elastic section) adds
``gammaY`` / ``gammaZ``. Before #2053 the reader knew only the force names
and raised on ``eps``; with no ``section.deformation`` group, a deformation
read came back empty and silent.

Oracle (synthetic): every value is the closed form ``k*1000 + t*100 + e*10
+ g`` of (component k in META order, step t, element e, station g), packed
GP-slowest as the recorder writes it, and written into the file by hand,
not through the catalog's ``flatten``. The canonical for each META name is
written out here, not read from the reader's table.

Oracle (live, fork lane): ``eleResponse(tag, "section", i, "deformation")``
column by column, and ``integrationPoints`` for the station coordinates.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.opensees._response_catalog import (
    CUSTOM_RULE_CATALOG,
    ELE_TAG_ForceBeamColumn2d,
    ELE_TAG_ForceBeamColumn3d,
    IntRule,
)
from apeGmsh.results import Results
from apeGmsh.results.readers import _mpco_line_io as _mline
from apeGmsh.results.readers._mpco import MPCOReader

from tests.conftest import _stub_model_h5_path

#: META name -> line-station canonical, written out (the oracle does not
#: read the reader's table).
DEFORMATION_CANONICAL = {
    "eps": "axial_strain",
    "kappaZ": "curvature_z",
    "kappaY": "curvature_y",
    "theta": "torsional_strain",
    "gammaY": "shear_strain_y",
    "gammaZ": "shear_strain_z",
}
FORCE_NAMES = ("P", "Mz", "My", "T", "Vy", "Vz")

#: section kind -> its META names, in getType() order.
SECTION_KINDS: dict[str, tuple[str, ...]] = {
    "2d_fiber": ("eps", "kappaZ"),
    "2d_shear": ("eps", "kappaZ", "gammaY"),
    "3d_fiber_torsion": ("eps", "kappaZ", "kappaY", "theta"),
    "3d_aggregated_shear": (
        "eps", "kappaZ", "kappaY", "theta", "gammaY", "gammaZ"),
}

GP_X_3 = np.array([-0.7745966692414834, 0.0, 0.7745966692414834])
GP_X_5 = np.array([-1.0, -0.6546536707079771, 0.0, 0.6546536707079771, 1.0])


# =====================================================================
# Synthetic MPCO writer
# =====================================================================

def _skeleton(path: Path, n_steps: int) -> h5py.File:
    f = h5py.File(path, "w")
    info = f.create_group("INFO")
    info.create_dataset("SPATIAL_DIM", data=3)
    stage = f.create_group("MODEL_STAGE[1]")
    stage.attrs["STEP"] = 0
    stage.attrs["TIME"] = 0.0
    model = stage.create_group("MODEL")
    nodes = model.create_group("NODES")
    nodes.create_dataset("ID", data=np.array([[1], [2]], dtype=np.int32))
    nodes.create_dataset("COORDINATES", data=np.zeros((2, 3)))
    model.create_group("ELEMENTS")
    on_nodes = stage.create_group("RESULTS/ON_NODES")
    disp = on_nodes.create_group("DISPLACEMENT")
    disp.create_dataset("ID", data=np.array([[1], [2]], dtype=np.int32))
    for k in range(n_steps):
        ds = disp.create_dataset(f"DATA/STEP_{k}", data=np.zeros((2, 3)))
        ds.attrs["STEP"] = k
        ds.attrs["TIME"] = float(k + 1)
    stage.create_group("RESULTS/ON_ELEMENTS")
    return f


def _closed_form(n_steps: int, n_ele: int, n_ip: int, n_comp: int) -> np.ndarray:
    """``(T, E, n_ip * n_comp)``, GP-slowest, value ``k*1000+t*100+e*10+g``."""
    out = np.empty((n_steps, n_ele, n_ip * n_comp))
    for t in range(n_steps):
        for e in range(n_ele):
            for g in range(n_ip):
                for k in range(n_comp):
                    out[t, e, g * n_comp + k] = k * 1000 + t * 100 + e * 10 + g
    return out


def _add_bucket(
    f: h5py.File, *, group: str, class_tag: int, class_name: str,
    names: tuple[str, ...], gp_x: np.ndarray, element_ids: list[int],
    n_steps: int, custom_idx: int = 1, connectivity: bool = True,
    header_idx: int = 0,
) -> None:
    stage = f["MODEL_STAGE[1]"]
    n_ip, n_comp = gp_x.size, len(names)
    if connectivity:
        conn = stage["MODEL/ELEMENTS"].create_dataset(
            f"{class_tag}-{class_name}[{IntRule.Custom}:{custom_idx}]",
            data=np.zeros((len(element_ids), 3), dtype=np.int32))
        conn.attrs["GP_X"] = gp_x
    grp = stage["RESULTS/ON_ELEMENTS"].require_group(group)
    b = grp.create_group(
        f"{class_tag}-{class_name}[{IntRule.Custom}:{custom_idx}:{header_idx}]")
    b.attrs["NUM_COLUMNS"] = np.array([n_ip * n_comp], dtype=np.int32)
    meta = b.create_group("META")
    meta.create_dataset("MULTIPLICITY", data=np.ones((n_ip, 1), np.int32))
    meta.create_dataset(
        "GAUSS_IDS", data=np.arange(n_ip, dtype=np.int32).reshape(-1, 1))
    meta.create_dataset(
        "NUM_COMPONENTS", data=np.full((n_ip, 1), n_comp, np.int32))
    comps = ";".join(f"0.1.2.{','.join(names)}" for _ in range(n_ip))
    meta.create_dataset("COMPONENTS", data=np.array([comps.encode("ascii")]))
    b.create_dataset(
        "ID", data=np.asarray(element_ids, np.int32).reshape(-1, 1))
    flat = _closed_form(n_steps, len(element_ids), n_ip, n_comp)
    for t in range(n_steps):
        ds = b.create_dataset(f"DATA/STEP_{t}", data=flat[t])
        ds.attrs["STEP"] = t
        ds.attrs["TIME"] = float(t + 1)


def _expected(k: int, n_steps: int, n_ele: int, n_ip: int) -> np.ndarray:
    return np.array([
        [k * 1000 + t * 100 + e * 10 + g
         for e in range(n_ele) for g in range(n_ip)]
        for t in range(n_steps)
    ], dtype=np.float64)


def _read(path: Path, component: str):
    with MPCOReader(path) as r:
        (stage,) = r.stages()
        return r.read_line_stations(stage.id, component)


# =====================================================================
# Every section kind decodes every canonical it carries
# =====================================================================

@pytest.mark.parametrize("kind", sorted(SECTION_KINDS))
def test_section_kind_reads_every_deformation_canonical(
    tmp_path: Path, kind: str,
) -> None:
    names = SECTION_KINDS[kind]
    is_3d = "kappaY" in names
    tag, cls = ((ELE_TAG_ForceBeamColumn3d, "ForceBeamColumn3d") if is_3d
                else (ELE_TAG_ForceBeamColumn2d, "ForceBeamColumn2d"))
    path = tmp_path / f"{kind}.mpco"
    with _skeleton(path, n_steps=2) as f:
        _add_bucket(f, group="section.deformation", class_tag=tag,
                    class_name=cls, names=names, gp_x=GP_X_5,
                    element_ids=[7, 9], n_steps=2)
    for k, name in enumerate(names):
        slab = _read(path, DEFORMATION_CANONICAL[name])
        np.testing.assert_array_equal(
            slab.values, _expected(k, 2, 2, 5), err_msg=name)
        assert slab.element_index.tolist() == [7] * 5 + [9] * 5
        np.testing.assert_allclose(
            slab.station_natural_coord, np.tile(GP_X_5, 2), rtol=0, atol=0)
        np.testing.assert_array_equal(slab.time, [1.0, 2.0])


def test_component_absent_from_section_reads_empty(tmp_path: Path) -> None:
    """A 2-D section has no kappaY: the bucket exists, the component does not."""
    path = tmp_path / "2d.mpco"
    with _skeleton(path, n_steps=1) as f:
        _add_bucket(f, group="section.deformation",
                    class_tag=ELE_TAG_ForceBeamColumn2d,
                    class_name="ForceBeamColumn2d",
                    names=SECTION_KINDS["2d_fiber"], gp_x=GP_X_3,
                    element_ids=[1], n_steps=1)
    assert _read(path, "curvature_y").values.shape == (1, 0)


# =====================================================================
# Every beam family the catalog routes section.deformation for
# =====================================================================

DEFORMATION_FAMILIES = sorted(
    (cname, layout.class_tag)
    for (cname, token), layout in CUSTOM_RULE_CATALOG.items()
    if token == "section_deformation"
)


def test_deformation_families_are_not_vacuous() -> None:
    names = {c for c, _ in DEFORMATION_FAMILIES}
    assert {"ForceBeamColumn2d", "ForceBeamColumn3d", "DispBeamColumn2d",
            "DispBeamColumn3d"} <= names
    assert len(names) >= 10


@pytest.mark.parametrize(("cname", "ctag"), DEFORMATION_FAMILIES)
def test_beam_family_reads_section_deformation(
    tmp_path: Path, cname: str, ctag: int,
) -> None:
    names = (SECTION_KINDS["3d_fiber_torsion"] if cname.endswith("3d")
             else SECTION_KINDS["2d_fiber"])
    path = tmp_path / f"{cname}.mpco"
    with _skeleton(path, n_steps=1) as f:
        _add_bucket(f, group="section.deformation", class_tag=ctag,
                    class_name=cname, names=names, gp_x=GP_X_3,
                    element_ids=[4], n_steps=1)
    for k, name in enumerate(names):
        slab = _read(path, DEFORMATION_CANONICAL[name])
        np.testing.assert_array_equal(slab.values, _expected(k, 1, 1, 3))
        assert slab.element_index.tolist() == [4, 4, 4]


# =====================================================================
# Force and deformation side by side never mix
# =====================================================================

def test_force_and_deformation_buckets_read_their_own_group(
    tmp_path: Path,
) -> None:
    path = tmp_path / "both.mpco"
    with _skeleton(path, n_steps=1) as f:
        _add_bucket(f, group="section.force",
                    class_tag=ELE_TAG_ForceBeamColumn3d,
                    class_name="ForceBeamColumn3d",
                    names=("P", "Mz", "My", "T"), gp_x=GP_X_3,
                    element_ids=[1], n_steps=1)
        _add_bucket(f, group="section.deformation",
                    class_tag=ELE_TAG_ForceBeamColumn3d,
                    class_name="ForceBeamColumn3d",
                    names=("theta", "eps", "kappaY", "kappaZ"),  # reordered
                    gp_x=GP_X_3, element_ids=[1], n_steps=1,
                    connectivity=False)
    np.testing.assert_array_equal(
        _read(path, "bending_moment_z").values, _expected(1, 1, 1, 3))
    # In the deformation bucket kappaZ is column 3 and theta column 0.
    np.testing.assert_array_equal(
        _read(path, "curvature_z").values, _expected(3, 1, 1, 3))
    np.testing.assert_array_equal(
        _read(path, "torsional_strain").values, _expected(0, 1, 1, 3))


# =====================================================================
# Every bucket is read: header splits and uncatalogued classes (F1)
# =====================================================================
#
# The fork (79e062367) splits one class's elements by section component
# list: two 3-D force beams, one with a bare elastic section and one with
# shear, wrote ``74-ForceBeamColumn3d[1000:1:0]`` (``eps,kappaZ,kappaY,
# theta``) and ``[1000:1:1]`` (``eps,kappaZ,gammaY,kappaY,gammaZ,theta``),
# both sharing connectivity ``[1000:1]``. A 2-D ``mixedBeamColumn`` wrote
# ``30766-MixedBeamColumn2d[1000:1:0]`` with ``GP_X``; that class is not in
# CUSTOM_RULE_CATALOG. Before the fix the reader dropped ``:1`` and the
# mixed beam with no warning.

MIXED_2D = (30766, "MixedBeamColumn2d")   # tag as the live file wrote it
AGG_3D_LIVE_ORDER = ("eps", "kappaZ", "gammaY", "kappaY", "gammaZ", "theta")


def _header_split_file(path: Path, group: str, names0, names1) -> None:
    with _skeleton(path, n_steps=2) as f:
        _add_bucket(f, group=group, class_tag=ELE_TAG_ForceBeamColumn3d,
                    class_name="ForceBeamColumn3d", names=names0,
                    gp_x=GP_X_3, element_ids=[1], n_steps=2)
        # Header 1 shares the [1000:1] connectivity dataset.
        _add_bucket(f, group=group, class_tag=ELE_TAG_ForceBeamColumn3d,
                    class_name="ForceBeamColumn3d", names=names1,
                    gp_x=GP_X_3, element_ids=[2], n_steps=2,
                    header_idx=1, connectivity=False)


def test_header_split_deformation_reads_both_headers(tmp_path: Path) -> None:
    path = tmp_path / "split.mpco"
    _header_split_file(path, "section.deformation",
                       SECTION_KINDS["3d_fiber_torsion"], AGG_3D_LIVE_ORDER)
    # axial_strain is column 0 in both headers: element 1 then element 2.
    slab = _read(path, "axial_strain")
    assert slab.element_index.tolist() == [1, 1, 1, 2, 2, 2]
    np.testing.assert_array_equal(
        slab.values, np.hstack([_expected(0, 2, 1, 3)] * 2))
    # shear_strain_y exists only in header 1, at column 2.
    slab = _read(path, "shear_strain_y")
    assert slab.element_index.tolist() == [2, 2, 2]
    np.testing.assert_array_equal(slab.values, _expected(2, 2, 1, 3))
    # theta: column 3 in header 0, column 5 in header 1.
    np.testing.assert_array_equal(
        _read(path, "torsional_strain").values,
        np.hstack([_expected(3, 2, 1, 3), _expected(5, 2, 1, 3)]))


def test_header_split_force_reads_both_headers(tmp_path: Path) -> None:
    path = tmp_path / "split_force.mpco"
    _header_split_file(path, "section.force", ("P", "Mz", "My", "T"),
                       ("P", "Mz", "Vy", "My", "Vz", "T"))
    slab = _read(path, "axial_force")
    assert slab.element_index.tolist() == [1, 1, 1, 2, 2, 2]
    slab = _read(path, "shear_y")
    assert slab.element_index.tolist() == [2, 2, 2]
    np.testing.assert_array_equal(slab.values, _expected(2, 2, 1, 3))


@pytest.mark.parametrize(("group", "names", "canonical", "k"), [
    ("section.deformation", ("eps", "kappaZ"), "curvature_z", 1),
    ("section.force", ("P", "Mz"), "bending_moment_z", 1),
])
def test_uncatalogued_custom_rule_class_reads(
    tmp_path: Path, group: str, names, canonical: str, k: int,
) -> None:
    tag, cls = MIXED_2D
    from apeGmsh.opensees._response_catalog import is_custom_rule_catalogued
    assert not is_custom_rule_catalogued(cls, "section_deformation")
    path = tmp_path / "mixed.mpco"
    with _skeleton(path, n_steps=1) as f:
        _add_bucket(f, group=group, class_tag=tag, class_name=cls,
                    names=names, gp_x=GP_X_3, element_ids=[3], n_steps=1)
    slab = _read(path, canonical)
    assert slab.element_index.tolist() == [3, 3, 3]
    np.testing.assert_array_equal(slab.values, _expected(k, 1, 1, 3))
    np.testing.assert_array_equal(slab.station_natural_coord, GP_X_3)


def test_rule_zero_beam_without_gp_x_raises_naming_class(
    tmp_path: Path,
) -> None:
    """The fork's own disp beam writes ``[0:0:0]`` and no GP_X: loud."""
    from apeGmsh.opensees._response_catalog import (
        ELE_TAG_LadrunoDispBeamColumn2d,
    )
    path = tmp_path / "rule0.mpco"
    with _skeleton(path, n_steps=1) as f:
        _add_bucket(f, group="section.deformation",
                    class_tag=ELE_TAG_LadrunoDispBeamColumn2d,
                    class_name="LadrunoDispBeamColumn2d",
                    names=("eps", "kappaZ"), gp_x=GP_X_3,
                    element_ids=[1], n_steps=1, connectivity=False)
        stage = f["MODEL_STAGE[1]"]
        stage["MODEL/ELEMENTS"].create_dataset(
            f"{ELE_TAG_LadrunoDispBeamColumn2d}-LadrunoDispBeamColumn2d[0:0]",
            data=np.zeros((1, 3), dtype=np.int32))
        grp = stage["RESULTS/ON_ELEMENTS/section.deformation"]
        (key,) = list(grp)
        grp.move(key, key.replace("[1000:1:0]", "[0:0:0]"))
    with pytest.raises(ValueError, match=r"LadrunoDispBeamColumn2d.*GP_X|GP_X.*LadrunoDispBeamColumn2d"):
        _read(path, "curvature_z")


def test_non_beam_standard_rule_bucket_is_not_a_line_station(
    tmp_path: Path,
) -> None:
    """An uncatalogued class on a standard rule (a zero-length section)."""
    path = tmp_path / "zl.h5"
    with h5py.File(path, "w") as f:
        f.create_group("ON_ELEMENTS/section.deformation").create_group(
            "19-ZeroLengthSection[1:0:0]")
    with h5py.File(path, "r") as f:
        _, buckets = _mline.discover_line_station_buckets(
            f["ON_ELEMENTS"], canonical_component="curvature_z")
    assert buckets == []


# =====================================================================
# Never empty in silence
# =====================================================================

@pytest.mark.parametrize("canonical", sorted(DEFORMATION_CANONICAL.values()))
def test_deformation_from_force_only_file_refuses(
    tmp_path: Path, canonical: str,
) -> None:
    """Only ``-E section.force`` recorded: the deformation read names the group."""
    path = tmp_path / "force_only.mpco"
    with _skeleton(path, n_steps=2) as f:
        _add_bucket(f, group="section.force",
                    class_tag=ELE_TAG_ForceBeamColumn3d,
                    class_name="ForceBeamColumn3d", names=FORCE_NAMES,
                    gp_x=GP_X_5, element_ids=[1, 2], n_steps=2)
    with Results.from_mpco(path, model_h5=_stub_model_h5_path()) as r:
        s = r.stage(r.stages[0].id)
        with pytest.raises(LookupError, match=r"section\.deformation"):
            s.elements.line_stations.get(component=canonical)
        # The force read of the same file is untouched.
        assert s.elements.line_stations.get(
            component="axial_force").values.shape == (2, 10)


def test_refusal_is_lookup_error_not_value_error(tmp_path: Path) -> None:
    """``_mpco.py`` skips undecodable buckets on ValueError (#2047)."""
    path = tmp_path / "force_only.mpco"
    with _skeleton(path, n_steps=1):
        pass
    with pytest.raises(LookupError) as info:
        _read(path, "curvature_z")
    assert not isinstance(info.value, ValueError)


def test_force_canonical_without_section_force_still_reads_empty(
    tmp_path: Path,
) -> None:
    """``localForce`` may supply force stations, so no refusal there."""
    path = tmp_path / "nothing.mpco"
    with _skeleton(path, n_steps=1):
        pass
    assert _read(path, "axial_force").values.shape == (1, 0)


def test_empty_deformation_group_reads_empty(tmp_path: Path) -> None:
    """Recorded, but this partition has no beams: MPCO still writes the group."""
    path = tmp_path / "no_beams.mpco"
    with _skeleton(path, n_steps=1) as f:
        f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"].create_group(
            "section.deformation")
    assert _read(path, "curvature_z").values.shape == (1, 0)


# =====================================================================
# A bucket must use its own group's vocabulary
# =====================================================================

def test_force_names_under_deformation_group_raise(tmp_path: Path) -> None:
    path = tmp_path / "wrong.mpco"
    with _skeleton(path, n_steps=1) as f:
        _add_bucket(f, group="section.deformation",
                    class_tag=ELE_TAG_ForceBeamColumn2d,
                    class_name="ForceBeamColumn2d", names=("P", "Mz"),
                    gp_x=GP_X_3, element_ids=[1], n_steps=1)
    with pytest.raises(ValueError, match=r"other section response kind"):
        _read(path, "curvature_z")


def test_deformation_names_under_force_group_raise(tmp_path: Path) -> None:
    path = tmp_path / "wrong.mpco"
    with _skeleton(path, n_steps=1) as f:
        _add_bucket(f, group="section.force",
                    class_tag=ELE_TAG_ForceBeamColumn2d,
                    class_name="ForceBeamColumn2d", names=("eps", "kappaZ"),
                    gp_x=GP_X_3, element_ids=[1], n_steps=1)
    with pytest.raises(ValueError, match=r"other section response kind"):
        _read(path, "bending_moment_z")


def test_unknown_deformation_name_raises(tmp_path: Path) -> None:
    """A warping code tags ``Unknown``: loud, never dropped."""
    path = tmp_path / "warp.mpco"
    with _skeleton(path, n_steps=1) as f:
        _add_bucket(f, group="section.deformation",
                    class_tag=ELE_TAG_ForceBeamColumn2d,
                    class_name="ForceBeamColumn2d",
                    names=("eps", "kappaZ", "Unknown"),
                    gp_x=GP_X_3, element_ids=[1], n_steps=1)
    with pytest.raises(ValueError, match=r"unknown section component"):
        _read(path, "curvature_z")


def test_unknown_kind_raises() -> None:
    with pytest.raises(ValueError, match=r"kind"):
        _mline._names_for_kind("stiffness")


# =====================================================================
# Live: ForceBeamColumn section deformation vs eleResponse (fork lane)
# =====================================================================

def _live_case(tmp: Path, ndm: int) -> tuple[Path, dict[int, list[np.ndarray]],
                                             dict[int, np.ndarray]]:
    from apeGmsh.opensees.emitter.live import get_ops
    ops = get_ops()
    ops.wipe()
    L, n_ip = 3.0, 4
    if ndm == 2:
        ops.model("basic", "-ndm", 2, "-ndf", 3)
        ops.node(1, 0.0, 0.0)
        ops.node(2, L, 0.0)
        ops.node(3, 2 * L, 0.0)
        ops.fix(1, 1, 1, 1)
        # A shear-deformable 2-D section: fiber-free elastic + Vy.
        ops.section("Elastic", 1, 2.0e11, 0.02, 3.0e-4, 8.0e10, 0.8)
        ops.geomTransf("Linear", 1)
    else:
        ops.model("basic", "-ndm", 3, "-ndf", 6)
        ops.node(1, 0.0, 0.0, 0.0)
        ops.node(2, L, 0.0, 0.0)
        ops.node(3, 2 * L, 0.0, 0.0)
        ops.fix(1, 1, 1, 1, 1, 1, 1)
        # A 3-D section with torsion and both shears.
        ops.section("Elastic", 1, 2.0e11, 0.02, 3.0e-4, 2.0e-4, 8.0e10,
                    1.0e-4, 0.8, 0.8)
        ops.geomTransf("Linear", 1, 0.0, 0.0, 1.0)
    ops.beamIntegration("Lobatto", 1, 1, n_ip)
    ops.element("forceBeamColumn", 1, 1, 2, 1, 1)
    ops.element("forceBeamColumn", 2, 2, 3, 1, 1)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    if ndm == 2:
        ops.load(3, 10e3, -30e3, 4e3)
    else:
        ops.load(3, 10e3, -30e3, 20e3, 3e3, -2e3, 4e3)
    mpco = tmp / f"fbc{ndm}d.mpco"
    ops.recorder("mpco", str(mpco), "-N", "displacement",
                 "-E", "section.deformation")
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1e-12, 10)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 0.5)
    ops.analysis("Static")
    live: dict[int, list[np.ndarray]] = {1: [], 2: []}
    for _ in range(2):
        assert ops.analyze(1) == 0
        for e in live:
            live[e].append(np.stack([
                np.asarray(ops.eleResponse(e, "section", i, "deformation"))
                for i in range(1, n_ip + 1)
            ]))                                   # (n_ip, n_comp)
    xi = {e: 2.0 * np.asarray(ops.eleResponse(e, "integrationPoints")) / L - 1.0
          for e in live}
    ops.remove("recorders")
    ops.wipe()
    return mpco, live, xi


#: The META names each live section must write (a shear-deformable 2-D
#: section; a 3-D section with torsion and both shears). The column order
#: is the section's own getType() order, which eleResponse shares.
LIVE_NAMES = {
    2: {"eps", "kappaZ", "gammaY"},
    3: {"eps", "kappaZ", "kappaY", "theta", "gammaY", "gammaZ"},
}


@pytest.mark.ladruno_fork
@pytest.mark.parametrize("ndm", [2, 3])
def test_live_force_beam_section_deformation_matches_ele_response(
    tmp_path: Path, ndm: int,
) -> None:
    mpco, live, xi = _live_case(tmp_path, ndm)
    with h5py.File(mpco, "r") as f:
        (bkey,) = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS/section.deformation"]
        meta = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS/section.deformation"][
            bkey]["META/COMPONENTS"][...][0].decode()
    names = meta.split(";")[0].rsplit(".", 1)[1].split(",")
    assert set(names) == LIVE_NAMES[ndm], meta
    for k, name in enumerate(names):
        slab = _read(mpco, DEFORMATION_CANONICAL[name])
        assert slab.values.shape == (2, 8), name
        assert slab.element_index.tolist() == [1] * 4 + [2] * 4
        np.testing.assert_allclose(
            slab.station_natural_coord, np.concatenate([xi[1], xi[2]]),
            rtol=0, atol=1e-12)
        for t in range(2):
            want = np.concatenate([live[1][t][:, k], live[2][t][:, k]])
            np.testing.assert_allclose(
                slab.values[t], want, rtol=1e-12, atol=1e-18, err_msg=name)
        assert np.all(np.abs(slab.values[-1]) > 0.0), name   # not vacuous


def _live_split(tmp: Path, case: str) -> tuple[Path, dict[int, np.ndarray]]:
    """F1 reproducers: two section kinds on one class; a mixed beam."""
    from apeGmsh.opensees.emitter.live import get_ops
    ops = get_ops()
    ops.wipe()
    if case == "3d_two_kinds":
        ops.model("basic", "-ndm", 3, "-ndf", 6)
        for i in range(3):
            ops.node(i + 1, 3.0 * i, 0.0, 0.0)
        ops.fix(1, 1, 1, 1, 1, 1, 1)
        ops.section("Elastic", 1, 2e11, 0.02, 3e-4, 2e-4, 8e10, 1e-4)
        ops.section("Elastic", 2, 2e11, 0.02, 3e-4, 2e-4, 8e10, 1e-4,
                    0.8, 0.8)
        ops.geomTransf("Linear", 1, 0.0, 0.0, 1.0)
        ops.beamIntegration("Lobatto", 1, 1, 3)
        ops.beamIntegration("Lobatto", 2, 2, 3)
        ops.element("forceBeamColumn", 1, 1, 2, 1, 1)
        ops.element("forceBeamColumn", 2, 2, 3, 1, 2)
        load = (1e3, -3e3, 2e3, 3e2, -2e2, 4e2)
    else:                                            # "2d_mixed"
        ops.model("basic", "-ndm", 2, "-ndf", 3)
        for i in range(3):
            ops.node(i + 1, 3.0 * i, 0.0)
        ops.fix(1, 1, 1, 1)
        ops.section("Elastic", 1, 2e11, 0.02, 3e-4)
        ops.geomTransf("Linear", 1)
        ops.beamIntegration("Lobatto", 1, 1, 3)
        ops.element("mixedBeamColumn", 1, 1, 2, 1, 1)
        ops.element("mixedBeamColumn", 2, 2, 3, 1, 1)
        load = (1e3, -3e3, 4e2)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(3, *load)
    mpco = tmp / f"{case}.mpco"
    ops.recorder("mpco", str(mpco), "-N", "displacement",
                 "-E", "section.force", "-E", "section.deformation")
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1e-12, 10)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    assert ops.analyze(1) == 0
    live = {e: np.stack([
        np.asarray(ops.eleResponse(e, "section", i, "deformation"))
        for i in (1, 2, 3)]) for e in (1, 2)}
    ops.remove("recorders")
    ops.wipe()
    return mpco, live


@pytest.mark.ladruno_fork
@pytest.mark.parametrize("case", ["3d_two_kinds", "2d_mixed"])
def test_live_every_bucket_matches_ele_response(
    tmp_path: Path, case: str,
) -> None:
    """F1: header ``:1`` and an uncatalogued class read, not dropped."""
    mpco, live = _live_split(tmp_path, case)
    names_of: dict[int, list[str]] = {}
    with h5py.File(mpco, "r") as f:
        grp = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS/section.deformation"]
        keys = sorted(grp)
        for key in keys:
            meta = grp[key]["META/COMPONENTS"][...][0].decode()
            names = meta.split(";")[0].rsplit(".", 1)[1].split(",")
            for e in grp[key]["ID"][...].ravel():
                names_of[int(e)] = names
    assert sorted(names_of) == [1, 2]
    if case == "3d_two_kinds":
        assert [k[-3:] for k in keys] == [":0]", ":1]"], keys
    else:
        assert keys[0].startswith("30766-MixedBeamColumn2d["), keys
    for name in sorted({n for v in names_of.values() for n in v}):
        slab = _read(mpco, DEFORMATION_CANONICAL[name])
        holders = [e for e in (1, 2) if name in names_of[e]]
        assert slab.element_index.tolist() == [
            e for e in holders for _ in range(3)], name
        want = np.concatenate(
            [live[e][:, names_of[e].index(name)] for e in holders])
        np.testing.assert_allclose(
            slab.values[0], want, rtol=1e-12, atol=1e-18, err_msg=name)
