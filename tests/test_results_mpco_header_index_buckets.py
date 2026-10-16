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
# Nodal forces: a header-1 bucket is named in the read's warning
# ---------------------------------------------------------------------------


def test_nodal_force_header_1_is_named_beside_the_header_0_read(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nodal.mpco"
    labels = ",".join(f"{c}_{n}" for n in (1, 2)
                      for c in ("Px", "Py", "Pz", "Mx", "My", "Mz"))
    with _file(path) as f:
        for hdr, ids in ((0, [1, 2]), (1, [3])):
            _bucket(f, "globalForce",
                    f"{ELE_TAG_ElasticBeam3d}-ElasticBeam3d[1:0:{hdr}]",
                    flat=_flat(hdr, len(ids), 1, 12), ids=ids,
                    multiplicity=[1], gauss_ids=[-1], n_comp=[12],
                    components=f"0.{labels}")
    with _reader(path) as r:
        (stage,) = r.stages()
        with pytest.warns(NodalForceBucketSkippedWarning,
                          match=r"ElasticBeam3d\[1:0:1\].*response header 1"):
            slab = r.read_elements(stage.id, "nodal_resisting_force_x")
    assert np.asarray(slab.element_ids).tolist() == [1, 2]
    want = np.array([[[_value(0, t, e, 0, 0), _value(0, t, e, 0, 6)]
                      for e in (0, 1)] for t in range(N_STEPS)])
    np.testing.assert_array_equal(np.asarray(slab.values), want)
