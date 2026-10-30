"""MPCO buckets with a header index above 0 are read (#2033, #2096, #2093 item 2).

MPCO keys a bucket ``<tag>-<Class>[<rule>:<cust>:<hdr>]`` and opens a new
header ``hdr`` whenever one class's elements write a different META: two
materials with different response labels on one brick class (ASDConcrete3D
``d+,d-`` beside LadrunoConcrete3D ``C1,C2``), a plane-stress and a
plane-strain material on one quad class (3 and 4 stress columns per Gauss
point), fiber sections of different fiber counts on one beam class, layered
sections of different layer counts on one shell class. Each header bucket
carries its own META, so its layout is resolved from that META.

The material-state, Gauss, fiber and layer discoveries used to keep header 0
only, so the elements of every other header read as absent, with no warning.
Each case here writes a header-0 and a header-1 bucket of one class with
different META and checks that both are read, at the right components, with
values from a closed form of (header, time, element, Gauss point, column).

The nodal-force level names a header-1 bucket in
:class:`NodalForceBucketSkippedWarning` at read time; that case pins the
named warning beside the header-0 read.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.opensees._response_catalog import (
    ELE_TAG_ASDShellQ4,
    ELE_TAG_ElasticBeam3d,
    ELE_TAG_ForceBeamColumn3d,
    ELE_TAG_FourNodeQuad,
    ELE_TAG_FourNodeTetrahedron,
    ELE_TAG_LadrunoBrick,
    IntRule,
    lookup,
)
from apeGmsh.results import Results
from apeGmsh.results.readers._mpco import MPCOBucketSkippedWarning, MPCOReader
from apeGmsh.results.readers._mpco_layer_io import LayerBucketUnreadWarning
from apeGmsh.results.readers._mpco_nodal_io import (
    NodalForceBucketSkippedWarning,
)
from apeGmsh.results.readers._protocol import ResultLevel
from tests.conftest import _stub_model_h5_path

N_STEPS = 2


def _value(hdr: int, t: int, e: int, gp: int, c: int) -> float:
    """Closed form of one stored column: unique per (header, t, e, gp, c)."""
    return 100000.0 * (hdr + 1) + 1000.0 * t + 100.0 * e + 10.0 * gp + c


def _flat(hdr: int, n_elem: int, n_gp: int, n_comp: int) -> np.ndarray:
    """``(T, E, n_gp * n_comp)``, Gauss-point slowest, column fastest."""
    out = np.empty((N_STEPS, n_elem, n_gp * n_comp))
    for t in range(N_STEPS):
        for e in range(n_elem):
            for gp in range(n_gp):
                for c in range(n_comp):
                    out[t, e, gp * n_comp + c] = _value(hdr, t, e, gp, c)
    return out


def _file(path: Path, *, n_nodes: int = 4) -> h5py.File:
    """A one-stage MPCO skeleton with ``N_STEPS`` displacement steps."""
    f = h5py.File(path, "w")
    info = f.create_group("INFO")
    info.create_dataset("SPATIAL_DIM", data=3)
    stage = f.create_group("MODEL_STAGE[1]")
    stage.attrs["STEP"] = 0
    stage.attrs["TIME"] = 0.0
    node_ids = np.arange(1, n_nodes + 1, dtype=np.int32).reshape(-1, 1)
    nodes = stage.create_group("MODEL/NODES")
    nodes.create_dataset("ID", data=node_ids)
    nodes.create_dataset("COORDINATES", data=np.zeros((n_nodes, 3)))
    stage.create_group("MODEL/ELEMENTS")
    stage.create_group("MODEL/SECTION_ASSIGNMENTS")
    disp = stage.create_group("RESULTS/ON_NODES/DISPLACEMENT")
    disp.create_dataset("ID", data=node_ids)
    data = disp.create_group("DATA")
    for k in range(N_STEPS):
        ds = data.create_dataset(f"STEP_{k}", data=np.zeros((n_nodes, 3)))
        ds.attrs["STEP"] = k
        ds.attrs["TIME"] = 0.5 * (k + 1)
    stage.create_group("RESULTS/ON_ELEMENTS")
    return f


def _bucket(
    f: h5py.File, group: str, key: str, *, flat: np.ndarray, ids: list[int],
    multiplicity: list[int], gauss_ids: list[int], n_comp: list[int],
    components: str | None,
) -> None:
    b = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"].require_group(group).create_group(key)
    b.attrs["NUM_COLUMNS"] = np.array([flat.shape[2]], dtype=np.int32)
    meta = b.create_group("META")
    meta.create_dataset(
        "MULTIPLICITY", data=np.array(multiplicity, np.int32).reshape(-1, 1))
    meta.create_dataset(
        "GAUSS_IDS", data=np.array(gauss_ids, np.int32).reshape(-1, 1))
    meta.create_dataset(
        "NUM_COMPONENTS", data=np.array(n_comp, np.int32).reshape(-1, 1))
    if components is not None:
        meta.create_dataset("COMPONENTS", data=np.array([components.encode()]))
    b.create_dataset("ID", data=np.array(ids, dtype=np.int32).reshape(-1, 1))
    data = b.create_group("DATA")
    for k in range(flat.shape[0]):
        ds = data.create_dataset(f"STEP_{k}", data=flat[k])
        ds.attrs["STEP"] = k
        ds.attrs["TIME"] = 0.5 * (k + 1)


def _gauss_bucket(
    f: h5py.File, group: str, key: str, *, hdr: int, n_gp: int, labels: str,
    ids: list[int],
) -> None:
    """One META block per Gauss point, each listing ``labels``."""
    n_comp = len(labels.split(","))
    _bucket(f, group, key, flat=_flat(hdr, len(ids), n_gp, n_comp), ids=ids,
            multiplicity=[1] * n_gp, gauss_ids=list(range(n_gp)),
            n_comp=[n_comp] * n_gp,
            components=";".join(f"0.1.4.{labels}" for _ in range(n_gp)))


def _expect_gauss(slab, *, parts: list[tuple[int, list[int]]], n_gp: int,
                  col: int) -> None:
    """``parts`` is ``[(hdr, ids), ...]`` in bucket order; column ``col``."""
    want_index = [i for _, ids in parts for i in ids for _ in range(n_gp)]
    assert np.asarray(slab.element_index).tolist() == want_index
    want = np.array([
        [_value(hdr, t, e, gp, col)
         for hdr, ids in parts for e in range(len(ids)) for gp in range(n_gp)]
        for t in range(N_STEPS)
    ])
    np.testing.assert_array_equal(np.asarray(slab.values), want)


def _reader(path: Path) -> MPCOReader:
    return MPCOReader(path)


# ---------------------------------------------------------------------------
# Material state
# ---------------------------------------------------------------------------


def test_material_state_reads_the_ladruno_brick_header_1_pair(
    tmp_path: Path,
) -> None:
    """#2033: ASDConcrete3D ``d+,d-`` at header 0 and LadrunoConcrete3D's
    unnamed ``C1,C2`` at ``33002-LadrunoBrick[0:0:1]`` both read as
    tension then compression (the catalog's ``DAMAGE_TC`` names ``C1,C2``)."""
    path = tmp_path / "mixed_damage.mpco"
    key0 = f"{ELE_TAG_LadrunoBrick}-LadrunoBrick[0:0:0]"
    key1 = f"{ELE_TAG_LadrunoBrick}-LadrunoBrick[0:0:1]"
    with _file(path) as f:
        _gauss_bucket(f, "material.damage", key0, hdr=0, n_gp=8,
                      labels="d+,d-", ids=[7, 9])
        _gauss_bucket(f, "material.damage", key1, hdr=1, n_gp=8,
                      labels="C1,C2", ids=[8])
    with warnings.catch_warnings():
        warnings.simplefilter("error", MPCOBucketSkippedWarning)
        with _reader(path) as r:
            (stage,) = r.stages()
            comps = r.available_components(stage.id, ResultLevel.GAUSS)
            assert {"damage_tension", "damage_compression"} <= set(comps)
            for col, comp in enumerate(("damage_tension", "damage_compression")):
                slab = r.read_gauss(stage.id, comp)
                assert np.asarray(slab.values).shape == (N_STEPS, 3 * 8)
                _expect_gauss(slab, parts=[(0, [7, 9]), (1, [8])], n_gp=8,
                              col=col)


def test_material_state_header_1_alone_lists_its_components(
    tmp_path: Path,
) -> None:
    """The #2033 report: a lone ``[0:0:1]`` bucket listed ``[]`` and read
    shape ``(2, 0)``."""
    path = tmp_path / "lone_damage.mpco"
    key1 = f"{ELE_TAG_LadrunoBrick}-LadrunoBrick[0:0:1]"
    with _file(path) as f:
        _gauss_bucket(f, "material.damage", key1, hdr=1, n_gp=8,
                      labels="C1,C2", ids=[8])
    with _reader(path) as r:
        (stage,) = r.stages()
        assert {"damage_tension", "damage_compression"} <= set(
            r.available_components(stage.id, ResultLevel.GAUSS))
        slab = r.read_gauss(stage.id, "damage_compression")
        _expect_gauss(slab, parts=[(1, [8])], n_gp=8, col=1)


def test_damage_bucket_copied_to_header_1_reads_its_elements(
    tmp_path: Path,
) -> None:
    """#2096's repro: a well-formed damage bucket copied to
    ``[300:0:1]`` with IDs 12 and 13; ``gauss.get("damage_tension")``
    returns elements 10, 11, 12, 13."""
    path = tmp_path / "damage_copy.mpco"
    rule = int(IntRule.Tet_GL_1)
    key0 = f"{ELE_TAG_FourNodeTetrahedron}-FourNodeTetrahedron[{rule}:0:0]"
    key1 = f"{ELE_TAG_FourNodeTetrahedron}-FourNodeTetrahedron[{rule}:0:1]"
    assert key1.endswith("[300:0:1]")
    with _file(path) as f:
        _gauss_bucket(f, "material.damage", key0, hdr=0, n_gp=1,
                      labels="d+,d-", ids=[10, 11])
        grp = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS/material.damage"]
        grp.copy(key0, key1)
        grp[f"{key1}/ID"][...] = np.array([[12], [13]], dtype=np.int32)
    with Results.from_mpco(path, model_h5=_stub_model_h5_path()) as r:
        slab = r.elements.gauss.get(component="damage_tension")
    assert np.asarray(slab.element_index).tolist() == [10, 11, 12, 13]
    # The copy holds header 0's columns: elements 12, 13 read rows 0, 1.
    want = np.array([[_value(0, t, e, 0, 0) for e in (0, 1, 0, 1)]
                     for t in range(N_STEPS)])
    np.testing.assert_array_equal(np.asarray(slab.values), want)


# ---------------------------------------------------------------------------
# Gauss (catalog stress)
# ---------------------------------------------------------------------------


def test_gauss_stress_reads_plane_stress_and_plane_strain_headers(
    tmp_path: Path,
) -> None:
    """A plane-stress material (3 columns per GP) at header 0 and a
    plane-strain material that adds ``sigma33`` (4 columns) at header 1 of
    one FourNodeQuad class. Each header's width comes from its own
    ``NUM_COLUMNS``, so ``stress_zz`` reads from header 1 alone."""
    path = tmp_path / "mixed_quad.mpco"
    rule = int(IntRule.Quad_GL_2)
    key0 = f"{ELE_TAG_FourNodeQuad}-FourNodeQuad[{rule}:0:0]"
    key1 = f"{ELE_TAG_FourNodeQuad}-FourNodeQuad[{rule}:0:1]"
    with _file(path) as f:
        _gauss_bucket(f, "material.stress", key0, hdr=0, n_gp=4,
                      labels="sigma11,sigma22,sigma12", ids=[1, 2])
        _gauss_bucket(f, "material.stress", key1, hdr=1, n_gp=4,
                      labels="sigma11,sigma22,sigma12,sigma33", ids=[3])
    coords = lookup("FourNodeQuad", IntRule.Quad_GL_2, "stress").natural_coords
    with _reader(path) as r:
        (stage,) = r.stages()
        assert "stress_zz" in r.available_components(stage.id, ResultLevel.GAUSS)
        for col, comp in enumerate(("stress_xx", "stress_yy", "stress_xy")):
            slab = r.read_gauss(stage.id, comp)
            _expect_gauss(slab, parts=[(0, [1, 2]), (1, [3])], n_gp=4, col=col)
            np.testing.assert_array_equal(
                np.asarray(slab.natural_coords), np.tile(coords, (3, 1)))
        zz = r.read_gauss(stage.id, "stress_zz")
        _expect_gauss(zz, parts=[(1, [3])], n_gp=4, col=3)


def _two_quad_headers(path: Path, labels0: str, labels1: str) -> None:
    rule = int(IntRule.Quad_GL_2)
    with _file(path) as f:
        for hdr, labels, ids in ((0, labels0, [1, 2]), (1, labels1, [3])):
            _gauss_bucket(f, "material.stress",
                          f"{ELE_TAG_FourNodeQuad}-FourNodeQuad[{rule}:0:{hdr}]",
                          hdr=hdr, n_gp=4, labels=labels, ids=ids)


def test_gauss_header_whose_labels_contradict_the_catalog_is_skipped(
    tmp_path: Path,
) -> None:
    """The review of #2114: MPCO opens a same-width header only when the
    labels differ, so ``foo,bar,baz`` at header 1 must never be read as
    ``stress_xx, stress_yy, stress_xy``. That bucket is left unread and
    named in MPCOBucketSkippedWarning with its labels; header 0 still
    reads."""
    path = tmp_path / "foo.mpco"
    _two_quad_headers(path, "sigma11,sigma22,sigma12", "foo,bar,baz")
    with _reader(path) as r:
        (stage,) = r.stages()
        with pytest.warns(MPCOBucketSkippedWarning,
                          match=r"FourNodeQuad\[201:0:1\].*'foo'"):
            slab = r.read_gauss(stage.id, "stress_xx")
    _expect_gauss(slab, parts=[(0, [1, 2])], n_gp=4, col=0)


@pytest.mark.parametrize("cls, tag, rule, n_gp, group, labels, comps", [
    # ShellMITC9.cpp:466-473 tags the membrane shear ``p1212``.
    ("ShellMITC9", 54, 202, 9, "stresses",
     "p11,p22,p1212,m11,m22,m12,q1,q2",
     ("membrane_force_xx", "membrane_force_yy", "membrane_force_xy",
      "bending_moment_xx", "bending_moment_yy", "bending_moment_xy",
      "transverse_shear_xz", "transverse_shear_yz")),
    # VonPapaDamage.cpp:1248 tags its strain ``e11,e22,e12``.
    ("FourNodeQuad", ELE_TAG_FourNodeQuad, 201, 4, "material.strain",
     "e11,e22,e12", ("strain_xx", "strain_yy", "strain_xy")),
    # NDMaterial's untagged width: ``UnknownStress``, in catalog order.
    ("FourNodeQuad", ELE_TAG_FourNodeQuad, 201, 4, "material.stress",
     "UnknownStress,UnknownStress,UnknownStress",
     ("stress_xx", "stress_yy", "stress_xy")),
])
def test_gauss_fork_label_spellings_read(
    tmp_path: Path, cls: str, tag: int, rule: int, n_gp: int, group: str,
    labels: str, comps: tuple[str, ...],
) -> None:
    """Header-0 label spellings the fork writes for catalogued classes and
    materials read, by name, under MPCOBucketSkippedWarning as an error."""
    path = tmp_path / "spelling.mpco"
    with _file(path) as f:
        _gauss_bucket(f, group, f"{tag}-{cls}[{rule}:0:0]", hdr=0,
                      n_gp=n_gp, labels=labels, ids=[5, 6])
    with warnings.catch_warnings():
        warnings.simplefilter("error", MPCOBucketSkippedWarning)
        with _reader(path) as r:
            (stage,) = r.stages()
            for col, comp in enumerate(comps):
                _expect_gauss(r.read_gauss(stage.id, comp),
                              parts=[(0, [5, 6])], n_gp=n_gp, col=col)


def test_gauss_columns_read_by_their_meta_names(tmp_path: Path) -> None:
    """A 4-column plane bucket labelled ``sigma11,sigma22,sigma33,sigma12``
    (``sigma33`` third, not last): ``stress_zz`` is column 2 and
    ``stress_xy`` column 3, by name, beside a header 0 in the fork's
    ``stressPlaneStrain`` order (``sigma33`` last)."""
    path = tmp_path / "zz_order.mpco"
    _two_quad_headers(path, "sigma11,sigma22,sigma12,sigma33",
                      "sigma11,sigma22,sigma33,sigma12")
    with _reader(path) as r:
        (stage,) = r.stages()
        zz = r.read_gauss(stage.id, "stress_zz")
        xy = r.read_gauss(stage.id, "stress_xy")
    parts = [(0, [1, 2]), (1, [3])]
    n = 4 * 2
    np.testing.assert_array_equal(
        np.asarray(zz.values)[:, :n],
        np.array([[_value(0, t, e, gp, 3) for e in (0, 1) for gp in range(4)]
                  for t in range(N_STEPS)]))
    np.testing.assert_array_equal(
        np.asarray(zz.values)[:, n:],
        np.array([[_value(1, t, 0, gp, 2) for gp in range(4)]
                  for t in range(N_STEPS)]))
    np.testing.assert_array_equal(
        np.asarray(xy.values)[:, n:],
        np.array([[_value(1, t, 0, gp, 3) for gp in range(4)]
                  for t in range(N_STEPS)]))
    assert np.asarray(zz.element_index).tolist() == [
        i for _, ids in parts for i in ids for _ in range(4)]


def test_gauss_unnamed_header_reads_in_catalog_order(tmp_path: Path) -> None:
    """LadrunoConcrete3D tags no names under ``material.stress``, so MPCO
    writes ``C1..C6`` (dumped from fork build 79e0623670); beside an
    ASDConcrete3D header 0 that names ``sigma11..sigma13`` it opens header
    1, whose columns take RESPONSE_CATALOG's order."""
    path = tmp_path / "unnamed_stress.mpco"
    with _file(path) as f:
        _gauss_bucket(f, "material.stress",
                      f"{ELE_TAG_LadrunoBrick}-LadrunoBrick[0:0:0]", hdr=0,
                      n_gp=8, labels="sigma11,sigma22,sigma33,sigma12,"
                                     "sigma23,sigma13", ids=[7])
        _gauss_bucket(f, "material.stress",
                      f"{ELE_TAG_LadrunoBrick}-LadrunoBrick[0:0:1]", hdr=1,
                      n_gp=8, labels="C1,C2,C3,C4,C5,C6", ids=[8])
    with _reader(path) as r:
        (stage,) = r.stages()
        for col, comp in enumerate(("stress_xx", "stress_yy", "stress_zz",
                                    "stress_xy", "stress_yz", "stress_xz")):
            _expect_gauss(r.read_gauss(stage.id, comp),
                          parts=[(0, [7]), (1, [8])], n_gp=8, col=col)


# ---------------------------------------------------------------------------
# Fibers
# ---------------------------------------------------------------------------


def _fiber_section(f: h5py.File, *, tag: int, pairs: list[tuple[int, int]],
                   y: list[float]) -> None:
    sec = f["MODEL_STAGE[1]/MODEL/SECTION_ASSIGNMENTS"].create_group(
        f"SECTION_{tag}[FiberSection3d]")
    sec.create_dataset("ASSIGNMENT", data=np.array(pairs, np.int32).reshape(-1, 2))
    n = len(y)
    sec.create_dataset("FIBER_DATA", data=np.column_stack(
        [y, np.zeros(n), np.full(n, 0.01 * tag)]))
    sec.create_dataset("FIBER_MATERIALS",
                       data=np.full((n, 1), tag, dtype=np.int32))


def test_fibers_read_both_fiber_count_headers(tmp_path: Path) -> None:
    """A 4-fiber section at header 0 and a 2-fiber section at header 1 of
    one ForceBeamColumn3d class: each header resolves its own fiber count
    from the sections its elements are assigned."""
    path = tmp_path / "mixed_fibers.mpco"
    cls = "ForceBeamColumn3d"
    n_ip = 3
    gp_x = np.array([-1.0, 0.0, 1.0])
    with _file(path) as f:
        conn = f["MODEL_STAGE[1]/MODEL/ELEMENTS"].create_dataset(
            f"{ELE_TAG_ForceBeamColumn3d}-{cls}[{int(IntRule.Custom)}:0]",
            data=np.zeros((3, 3), dtype=np.int32))
        conn.attrs["GP_X"] = gp_x
        _fiber_section(f, tag=1, pairs=[(e, g) for e in (10, 11) for g in range(n_ip)],
                       y=[-0.3, -0.1, 0.1, 0.3])
        _fiber_section(f, tag=2, pairs=[(12, g) for g in range(n_ip)],
                       y=[-0.2, 0.2])
        for hdr, ids, n_fib in ((0, [10, 11], 4), (1, [12], 2)):
            _bucket(
                f, "section.fiber.stress",
                f"{ELE_TAG_ForceBeamColumn3d}-{cls}[{int(IntRule.Custom)}:0:{hdr}]",
                flat=_flat(hdr, len(ids), n_ip, n_fib), ids=ids,
                multiplicity=[n_fib] * n_ip, gauss_ids=list(range(n_ip)),
                n_comp=[1] * n_ip,
                components=";".join("0.1.2.3.4.stress" for _ in range(n_ip)))
    with _reader(path) as r:
        (stage,) = r.stages()
        assert "fiber_stress" in r.available_components(stage.id, ResultLevel.FIBERS)
        slab = r.read_fibers(stage.id, "fiber_stress")
    # FiberSlab packs element-slow, IP-mid, fiber-fast.
    rows = [(0, e, eid, ip, k, 4) for e, eid in enumerate((10, 11))
            for ip in range(n_ip) for k in range(4)]
    rows += [(1, 0, 12, ip, k, 2) for ip in range(n_ip) for k in range(2)]
    assert np.asarray(slab.element_index).tolist() == [r_[2] for r_ in rows]
    assert np.asarray(slab.gp_index).tolist() == [r_[3] for r_ in rows]
    want = np.array([[_value(h, t, e, ip, k) for h, e, _, ip, k, _n in rows]
                     for t in range(N_STEPS)])
    np.testing.assert_array_equal(np.asarray(slab.values), want)
    y = np.asarray(slab.y)
    np.testing.assert_array_equal(y[:24], np.tile([-0.3, -0.1, 0.1, 0.3], 6))
    np.testing.assert_array_equal(y[24:], np.tile([-0.2, 0.2], 3))
    np.testing.assert_array_equal(
        np.asarray(slab.material_tag), [1] * 24 + [2] * 6)


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


def _layered_section(f: h5py.File, *, tag: int, pairs: list[tuple[int, int]],
                     thickness: list[float]) -> None:
    sec = f["MODEL_STAGE[1]/MODEL/SECTION_ASSIGNMENTS"].create_group(
        f"SECTION_{tag}[LayeredShellFiberSection]")
    sec.create_dataset("ASSIGNMENT", data=np.array(pairs, np.int32).reshape(-1, 2))
    n = len(thickness)
    fdata = np.zeros((n, 3))
    fdata[:, 2] = thickness
    sec.create_dataset("FIBER_DATA", data=fdata)
    sec.create_dataset("FIBER_MATERIALS",
                       data=np.full((n, 1), tag, dtype=np.int32))


def test_layers_read_both_layer_count_headers(tmp_path: Path) -> None:
    """#2093 item 2: a 3-layer section at header 0 and a 2-layer section at
    header 1 of one ASDShellQ4 class: both read, with each header's own
    layer count and thicknesses, and no LayerBucketUnreadWarning."""
    path = tmp_path / "mixed_layers.mpco"
    rule = int(IntRule.Quad_GL_2)
    n_sgp = 4
    with _file(path) as f:
        stage = f["MODEL_STAGE[1]"]
        stage["MODEL/ELEMENTS"].create_dataset(
            f"{ELE_TAG_ASDShellQ4}-ASDShellQ4[{rule}:0]",
            data=np.zeros((3, 5), dtype=np.int32))
        la = stage["MODEL"].create_group("LOCAL_AXES")
        la.create_dataset("ID", data=np.array([20, 21, 22], dtype=np.int32))
        la.create_dataset("LOCAL_AXES", data=np.tile([1.0, 0, 0, 0], (3, 1)))
        _layered_section(f, tag=5, pairs=[(e, g) for e in (20, 21)
                                          for g in range(n_sgp)],
                         thickness=[0.01, 0.02, 0.01])
        _layered_section(f, tag=6, pairs=[(22, g) for g in range(n_sgp)],
                         thickness=[0.05, 0.07])
        for hdr, ids, n_lay in ((0, [20, 21], 3), (1, [22], 2)):
            _bucket(
                f, "material.fiber.stress",
                f"{ELE_TAG_ASDShellQ4}-ASDShellQ4[{rule}:0:{hdr}]",
                flat=_flat(hdr, len(ids), n_sgp, n_lay), ids=ids,
                multiplicity=[n_lay] * n_sgp, gauss_ids=list(range(n_sgp)),
                n_comp=[1] * n_sgp, components=None)
    with warnings.catch_warnings():
        warnings.simplefilter("error", LayerBucketUnreadWarning)
        with _reader(path) as r:
            (stage_,) = r.stages()
            assert "fiber_stress" in r.available_components(
                stage_.id, ResultLevel.LAYERS)
            slab = r.read_layers(stage_.id, "fiber_stress")
    rows = [(0, e, eid, g, k, 3) for e, eid in enumerate((20, 21))
            for g in range(n_sgp) for k in range(3)]
    rows += [(1, 0, 22, g, k, 2) for g in range(n_sgp) for k in range(2)]
    assert np.asarray(slab.element_index).tolist() == [r_[2] for r_ in rows]
    assert np.asarray(slab.layer_index).tolist() == [r_[4] for r_ in rows]
    want = np.array([[_value(h, t, e, g, k) for h, e, _, g, k, _n in rows]
                     for t in range(N_STEPS)])
    np.testing.assert_array_equal(np.asarray(slab.values), want)
    np.testing.assert_array_equal(
        np.asarray(slab.thickness),
        [0.01, 0.02, 0.01] * 8 + [0.05, 0.07] * 4)


# ---------------------------------------------------------------------------
# Nodal forces: a header-1 bucket reads when its META spells the catalog
# layout (#2115), and is named in the read's warning when it does not
# ---------------------------------------------------------------------------

_NODAL_GLOBAL_3D = ("Px", "Py", "Pz", "Mx", "My", "Mz")


def _nodal_two_header_file(path: Path, header_1_bases: tuple[str, ...]) -> None:
    """ElasticBeam3d ``globalForce``: elements 1, 2 at header 0 (fork labels)
    and element 3 at header 1, labelled with ``header_1_bases``."""
    with _file(path) as f:
        for hdr, ids, bases in ((0, [1, 2], _NODAL_GLOBAL_3D),
                                (1, [3], header_1_bases)):
            labels = ",".join(f"{c}_{n}" for n in (1, 2) for c in bases)
            _bucket(f, "globalForce",
                    f"{ELE_TAG_ElasticBeam3d}-ElasticBeam3d[1:0:{hdr}]",
                    flat=_flat(hdr, len(ids), 1, 12), ids=ids,
                    multiplicity=[1], gauss_ids=[-1], n_comp=[12],
                    components=f"0.{labels}")


def test_nodal_force_header_1_reads_its_own_values_beside_header_0(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nodal.mpco"
    _nodal_two_header_file(path, _NODAL_GLOBAL_3D)
    with warnings.catch_warnings():
        warnings.simplefilter("error", NodalForceBucketSkippedWarning)
        with _reader(path) as r:
            (stage,) = r.stages()
            slab = r.read_elements(stage.id, "nodal_resisting_force_x")
    assert np.asarray(slab.element_ids).tolist() == [1, 2, 3]
    # (header, element row within its bucket) for elements 1, 2, 3.
    rows = ((0, 0), (0, 1), (1, 0))
    want = np.array([[[_value(h, t, e, 0, 0), _value(h, t, e, 0, 6)]
                      for h, e in rows] for t in range(N_STEPS)])
    np.testing.assert_array_equal(np.asarray(slab.values), want)


def test_nodal_force_misspelled_header_1_is_named_beside_the_header_0_read(
    tmp_path: Path,
) -> None:
    """Px and Py swapped at header 1: its columns contradict the catalog
    order, so it is left unread and named, and header 0 still reads."""
    path = tmp_path / "nodal.mpco"
    _nodal_two_header_file(path, ("Py", "Px", "Pz", "Mx", "My", "Mz"))
    with _reader(path) as r:
        (stage,) = r.stages()
        with pytest.warns(NodalForceBucketSkippedWarning,
                          match=r"ElasticBeam3d\[1:0:1\].*response header 1"
                                r".*do not spell"):
            slab = r.read_elements(stage.id, "nodal_resisting_force_x")
    assert np.asarray(slab.element_ids).tolist() == [1, 2]
    want = np.array([[[_value(0, t, e, 0, 0), _value(0, t, e, 0, 6)]
                      for e in (0, 1)] for t in range(N_STEPS)])
    np.testing.assert_array_equal(np.asarray(slab.values), want)


# ---------------------------------------------------------------------------
# Live, on the fork: ASDConcrete3D and LadrunoConcrete3D on one brick class
# ---------------------------------------------------------------------------


def _column(g, x0: float, name: str) -> None:
    box = g.model.geometry.add_box(x0, 0, 0, 1, 1, 2)
    g.model.sync()
    g.mesh.structured.set_transfinite_box(box, n=2)
    g.physical.add(3, [box], name=name)


@pytest.mark.ladruno_fork
@pytest.mark.live
def test_live_mixed_concrete_damage_reads_both_headers(tmp_path: Path) -> None:
    """Two LadrunoBrick columns, one ASDConcrete3D and one LadrunoConcrete3D,
    each pulled to three times its cracking strain by an imposed top
    displacement. MPCO writes their
    ``material.damage`` under two headers of one class (``d+,d-`` and
    ``C1,C2``); every Gauss point of both reads back equal to
    ``eleResponse(tag, "material", gp, "damage")`` at rtol 1e-10. Their
    ``material.stress`` splits the same way (ASDConcrete3D names
    ``sigma11..sigma13``, LadrunoConcrete3D writes ``C1..C6``), and every
    stress component of both reads back equal to ``eleResponse(tag,
    "stresses")``."""
    from apeGmsh import apeGmsh
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter

    with apeGmsh(model_name="hdr_mixed", verbose=False) as g:
        _column(g, 0.0, "ASD")
        _column(g, 3.0, "CDPM2")
        g.mesh.generation.generate(3)
        fem = g.mesh.queries.get_fem_data(dim=3)
    ids = np.asarray(fem.nodes.ids)
    xyz = np.asarray(fem.nodes.coords)
    base = [int(n) for n, p in zip(ids, xyz) if abs(p[2]) < 1e-9]
    top = [int(n) for n, p in zip(ids, xyz) if abs(p[2] - 2.0) < 1e-9]

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    asd = ops.nDMaterial.ASDConcrete3D(E=30e6, v=0.2, fc=30e3, ft=3e3)
    cdpm2 = ops.nDMaterial.LadrunoConcrete3D(
        E=30e6, nu=0.2, fc=30e3, ft=3e3, Gf=0.1, Gc=10.0,
    )
    ops.element.LadrunoBrick(pg="ASD", material=asd)
    ops.element.LadrunoBrick(pg="CDPM2", material=cdpm2)
    ops.fix(nodes=base, dofs=(1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        for nid in top:
            p.sp(node=nid, dof=3, value=6.0e-4)
    mpco = str(tmp_path / "mixed.mpco")
    ops.recorder.MPCO(file=mpco, nodal_responses=("displacement",),
                      elem_responses=("material.damage", "material.stress"))
    ops.constraints.Transformation()
    ops.numberer.RCM()
    ops.system.UmfPack()     # LadrunoConcrete3D's tangent is unsymmetric
    ops.test.NormDispIncr(tol=1e-8, max_iter=50)
    ops.algorithm.Newton()
    # 100 steps: past cracking, LadrunoConcrete3D refuses a return map that
    # fails, and with 50 or fewer steps one does (at step 17 of 50, 8 of
    # 20). From 80 to 400 steps every step converges to the same cracked
    # state.
    ops.integrator.LoadControl(dlam=0.01)
    ops.analysis.Static()
    model_h5 = str(tmp_path / "model.h5")
    ops.h5(model_h5)
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    for step in range(100):
        assert emitter.analyze(steps=1) == 0, f"step {step + 1}"
    live = emitter.ops

    with h5py.File(model_h5, "r") as f:
        meta = f["/opensees/element_meta"]
        fem_to_ops = {int(e): int(t) for tok in meta for t, e in
                      zip(meta[tok]["ids"][...], meta[tok]["fem_eids"][...])}
    truth = {
        eid: np.array([np.asarray(live.eleResponse(tag, "material", gp,
                                                   "damage"), float)
                       for gp in range(1, 9)])
        for eid, tag in fem_to_ops.items()
    }
    stress_truth = {
        eid: np.asarray(live.eleResponse(tag, "stresses"), float).reshape(8, 6)
        for eid, tag in fem_to_ops.items()
    }
    live.remove("recorders")
    assert len(truth) == 2
    assert all(t.shape == (8, 2) for t in truth.values())
    # Both columns cracked: each header carries a non-zero tension damage.
    assert all(float(t[:, 0].max()) > 0.0 for t in truth.values()), truth

    with h5py.File(mpco, "r") as f:
        stage = next(k for k in f if k.startswith("MODEL_STAGE"))
        keys = {g: sorted(f[f"{stage}/RESULTS/ON_ELEMENTS/{g}"])
                for g in ("material.damage", "material.stress")}
    for got in keys.values():
        assert got == [f"{ELE_TAG_LadrunoBrick}-LadrunoBrick[0:0:0]",
                       f"{ELE_TAG_LadrunoBrick}-LadrunoBrick[0:0:1]"], keys

    with Results.from_mpco(mpco, fem=fem, model_h5=model_h5) as r:
        for k, comp in enumerate(("damage_tension", "damage_compression")):
            slab = r.elements.gauss.get(component=comp)
            index = np.asarray(slab.element_index)
            last = np.asarray(slab.values)[-1]
            assert sorted(set(int(e) for e in index)) == sorted(truth)
            for eid, t in truth.items():
                np.testing.assert_allclose(
                    last[index == eid], t[:, k], rtol=1e-10, atol=1e-14,
                )
        for k, comp in enumerate(("stress_xx", "stress_yy", "stress_zz",
                                  "stress_xy", "stress_yz", "stress_xz")):
            slab = r.elements.gauss.get(component=comp)
            index = np.asarray(slab.element_index)
            last = np.asarray(slab.values)[-1]
            assert sorted(set(int(e) for e in index)) == sorted(stress_truth)
            for eid, t in stress_truth.items():
                np.testing.assert_allclose(
                    last[index == eid], t[:, k], rtol=1e-10, atol=1e-6,
                )
