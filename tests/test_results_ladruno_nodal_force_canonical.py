"""Canonical nodal resisting forces read from a ``.ladruno`` (#2071).

The ``.ladruno`` element level is token-driven: ``globalForce`` reads the
raw ``(T, E, ncol)`` block. A canonical ``nodal_resisting_{force,moment}_*``
request used to return an empty slab with no warning. It now decodes the
``globalForce`` columns per node into the MPCO reader's ``(T, E, n_nodes)``
layout, or refuses by name.

The oracle is constructive. Each synthetic bucket below holds, for element
``e``, node ``n``, quantity ``q`` and step ``t``, the value
``_truth(e, n, q, t)``, laid out in the column order and spelling the
OpenSees class writes (beams ``Px_1``…, quads and bricks ``P<dof>_<node>``,
trusses ``P<node>_<dof>``). A read that takes any column for another
reads a different number. The MPCO parity case transcribes a real
fork-written MPCO ``ElasticBeam3d`` bucket into a ``.ladruno`` and asks
both readers for every component.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.results.readers._ladruno import LadrunoReader
from apeGmsh.results.readers._ladruno_element_io import (
    ElementWidthMismatchWarning,
)
from apeGmsh.results.readers._ladruno_multi import LadrunoMultiPartitionReader
from apeGmsh.results.readers._protocol import ResultLevel

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ladruno"
STAGE_GRP = "MODEL_STAGE[1]"

# Quantity code per canonical name: the value's units digit.
_Q = {
    "nodal_resisting_force_x": 1, "nodal_resisting_force_y": 2,
    "nodal_resisting_force_z": 3, "nodal_resisting_moment_x": 4,
    "nodal_resisting_moment_y": 5, "nodal_resisting_moment_z": 6,
}
_AXIS_NAME = {
    "nodal_resisting_force_x": "Px", "nodal_resisting_force_y": "Py",
    "nodal_resisting_force_z": "Pz", "nodal_resisting_moment_x": "Mx",
    "nodal_resisting_moment_y": "My", "nodal_resisting_moment_z": "Mz",
}
_FORCES = ("nodal_resisting_force_x", "nodal_resisting_force_y",
           "nodal_resisting_force_z")
# The local-frame twins, read from ``localForce`` (#2124), in the
# catalog's per-node order, with the base each name spells.
_QL = {
    "nodal_resisting_force_local_x": 7, "nodal_resisting_force_local_y": 8,
    "nodal_resisting_force_local_z": 9, "nodal_resisting_moment_local_x": 10,
    "nodal_resisting_moment_local_y": 11, "nodal_resisting_moment_local_z": 12,
}
_LOCAL_BASE = {
    "nodal_resisting_force_local_x": "N", "nodal_resisting_force_local_y": "Vy",
    "nodal_resisting_force_local_z": "Vz",
    "nodal_resisting_moment_local_x": "T",
    "nodal_resisting_moment_local_y": "My",
    "nodal_resisting_moment_local_z": "Mz",
}
_LOCAL_2D = ("nodal_resisting_force_local_x", "nodal_resisting_force_local_y",
             "nodal_resisting_moment_local_z")


def _truth(e: int, n: int, q: str, t: int) -> float:
    return 1000.0 * e + 100.0 * n + {**_Q, **_QL}[q] + 0.125 * (t + 1)


# ---------------------------------------------------------------------------
# Synthetic buckets in each class's own spelling
# ---------------------------------------------------------------------------


def _axis_named(n_nodes: int, comps: tuple[str, ...]):
    """Beam spelling ``Px_1, Py_1, Mz_1, Px_2, …`` (node-major)."""
    return [(f"{_AXIS_NAME[c]}_{n}", n, c)
            for n in range(1, n_nodes + 1) for c in comps]


def _dof_node(n_nodes: int, ndf: int, ndm: int):
    """Quad / brick spelling ``P<dof>_<node>``; DOFs above ``ndm`` carry
    no canonical quantity (``None``)."""
    return [(f"P{d}_{n}", n, _FORCES[d - 1] if d <= ndm else None)
            for n in range(1, n_nodes + 1) for d in range(1, ndf + 1)]


def _node_dof(n_nodes: int, ndf: int, ndm: int):
    """Truss / zeroLength spelling ``P<node>_<dof>``."""
    return [(f"P{n}_{d}", n, _FORCES[d - 1] if d <= ndm else None)
            for n in range(1, n_nodes + 1) for d in range(1, ndf + 1)]


def _write_bucket(on_elements: "h5py.Group", token: str, key: str,
                  layout, ids: list[int], n_steps: int) -> None:
    grp = (on_elements[token] if token in on_elements
           else on_elements.create_group(token)).create_group(key)
    names = [name for name, _, _ in layout]
    data = np.zeros((n_steps, len(ids), len(layout)))
    for i, e in enumerate(ids):
        for col, (_, n, q) in enumerate(layout):
            for t in range(n_steps):
                # A column with no canonical quantity holds a value no
                # canonical read may return.
                data[t, i, col] = (_truth(e, n, q, t) if q is not None
                                   else -999.0)
    cm = grp.create_group("COLUMN_MAP")
    for name, value in (("LEVELS", 0), ("GAUSS_ID", -1), ("MULTIPLICITY", 1),
                        ("SECTION_TAG", -1), ("FIBER_ID", -1)):
        cm.create_dataset(name, data=np.array([[value]], dtype=np.int32))
    cm.create_dataset("NUM_COMP", data=np.array([[len(names)]], dtype=np.int32))
    cm.attrs["COMP_NAMES"] = np.bytes_(",".join(names))
    grp.create_dataset("DATA", data=data)
    grp.create_dataset("ID", data=np.array(ids, dtype=np.int32)[:, None])
    grp.create_dataset("STEP", data=np.arange(n_steps, dtype=np.int32))
    grp.create_dataset("TIME", data=0.25 * np.arange(1, n_steps + 1))


def _file_with(tmp_path: Path, source: str, buckets, name: str = "",
               local=()) -> Path:
    """Copy a committed fixture and add ``globalForce`` buckets (and
    ``localForce`` ones, ``local=``) to it."""
    dst = tmp_path / (name or source)
    shutil.copyfile(FIXTURES / source, dst)
    with h5py.File(dst, "r+") as f:
        on_e = f[f"{STAGE_GRP}/RESULTS/ON_ELEMENTS"]
        n_steps = int(next(iter(next(iter(on_e.values())).values()))
                      ["DATA"].shape[0])
        for token, group in (("globalForce", buckets), ("localForce", local)):
            for key, layout, ids in group:
                _write_bucket(on_e, token, key, layout, ids, n_steps)
    return dst


def _local_named(n_nodes: int, comps: tuple[str, ...], *,
                 base=None, sep: str = "_"):
    """``localForce`` spelling ``N_1, Vy_1, …, N_2, …`` (node-major);
    ``base`` overrides the base a name spells, ``sep=""`` drops the
    underscore (``N1``)."""
    base = {**_LOCAL_BASE, **(base or {})}
    return [(f"{base[c]}{sep}{n}", n, c)
            for n in range(1, n_nodes + 1) for c in comps]


def _expected(ids, comp: str, n_nodes: int, n_steps: int) -> np.ndarray:
    return np.array([[[_truth(e, n, comp, t) for n in range(1, n_nodes + 1)]
                      for e in ids] for t in range(n_steps)])


def _read(reader, comp: str, ids=None):
    return reader.read_elements(reader.stages()[0].id, comp,
                                element_ids=None if ids is None
                                else np.asarray(ids))


# 2-D frame: a truss (P<node>_<dof>, ndf 2) beside two elastic beams.
_TRUSS_2D = ("12-Truss[1:0:0]", _node_dof(2, 2, 2), [1, 2])
_BEAM_2D = ("5-ElasticBeam2d[1:0:0]",
            _axis_named(2, ("nodal_resisting_force_x",
                            "nodal_resisting_force_y",
                            "nodal_resisting_moment_z")), [7, 8])
# 3-D: an elastic beam beside an 8-node brick (P<dof>_<node>, ndf 3).
_BEAM_3D = ("5-ElasticBeam3d[1:0:0]", _axis_named(2, tuple(_Q)), [1])
_BRICK_3D = ("56-Brick[1:0:0]", _dof_node(8, 3, 3), [40])


# ---------------------------------------------------------------------------
# Decoding the three spellings
# ---------------------------------------------------------------------------


def test_decode_reads_each_spelling_node_major() -> None:
    from apeGmsh.results.readers._ladruno_element_io import (
        decode_nodal_force_columns,
    )

    cols, undecoded = decode_nodal_force_columns(
        ["Px_1", "Py_1", "Mz_1", "Px_2", "Py_2", "Mz_2"], ndm=2)
    assert {c: v.tolist() for c, v in cols.items()} == {
        "nodal_resisting_force_x": [0, 3], "nodal_resisting_force_y": [1, 4],
        "nodal_resisting_moment_z": [2, 5]}
    assert undecoded == ()
    # Quad: P<dof>_<node>.
    cols, _ = decode_nodal_force_columns(
        ["P1_1", "P2_1", "P1_2", "P2_2", "P1_3", "P2_3"], ndm=2)
    assert cols["nodal_resisting_force_x"].tolist() == [0, 2, 4]
    assert cols["nodal_resisting_force_y"].tolist() == [1, 3, 5]
    # Truss: P<node>_<dof>, the same names in the other order.
    cols, _ = decode_nodal_force_columns(
        ["P1_1", "P1_2", "P2_1", "P2_2"], ndm=2)
    assert cols["nodal_resisting_force_x"].tolist() == [0, 2]
    assert cols["nodal_resisting_force_y"].tolist() == [1, 3]


def test_decode_leaves_indexed_dofs_above_ndm_undecoded() -> None:
    from apeGmsh.results.readers._ladruno_element_io import (
        decode_nodal_force_columns,
    )

    # 2-D, three DOFs per node: P3 may be a rotation, a pore pressure or
    # a PML field, so it is not named.
    cols, undecoded = decode_nodal_force_columns(
        ["P1_1", "P2_1", "P3_1", "P1_2", "P2_2", "P3_2"], ndm=2)
    assert sorted(cols) == ["nodal_resisting_force_x",
                            "nodal_resisting_force_y"]
    assert cols["nodal_resisting_force_y"].tolist() == [1, 4]
    assert undecoded == ("P3_1", "P3_2")


@pytest.mark.parametrize("names, match", [
    (["P1_1", "P2_1"], "both"),                  # 1 node x 2 DOF or 2 x 1
    (["P1_1", "P2_2"], "neither"),
    (["P1_1", "P2_1", "Pp_1"], "neither nodal-force spelling"),
    (["Px_1", "P1_1"], "mix"),
    (["Px_1", "Py_1", "Px_2"], "absent at element node 2"),
])
def test_decode_refuses_what_it_cannot_name(names, match) -> None:
    from apeGmsh.results.readers._ladruno_element_io import (
        decode_nodal_force_columns,
    )

    with pytest.raises(ValueError, match=match):
        decode_nodal_force_columns(names, ndm=2)


# ---------------------------------------------------------------------------
# Single file, 2-D and 3-D
# ---------------------------------------------------------------------------


def test_2d_mixed_classes_read_per_node(tmp_path: Path) -> None:
    path = _file_with(tmp_path, "truss2d.ladruno", [_TRUSS_2D, _BEAM_2D])
    with LadrunoReader(path) as r:
        for comp in ("nodal_resisting_force_x", "nodal_resisting_force_y"):
            slab = _read(r, comp)
            ids = slab.element_ids.tolist()
            assert sorted(ids) == [1, 2, 7, 8]
            np.testing.assert_array_equal(
                slab.values, _expected(ids, comp, 2, 4), err_msg=comp)
        # A truss has no rotation: the moment comes from the beams alone.
        slab = _read(r, "nodal_resisting_moment_z")
        assert slab.element_ids.tolist() == [7, 8]
        np.testing.assert_array_equal(
            slab.values, _expected([7, 8], "nodal_resisting_moment_z", 2, 4))
        # The id filter and the time slice act as on a raw read.
        slab = r.read_elements(r.stages()[0].id, "nodal_resisting_force_y",
                               element_ids=np.array([8, 2]), time_slice=-1)
        ids = slab.element_ids.tolist()
        assert sorted(ids) == [2, 8]
        np.testing.assert_array_equal(
            slab.values, _expected(ids, "nodal_resisting_force_y", 2, 4)[-1:])


def test_2d_indexed_third_dof_is_not_read_as_a_moment(tmp_path: Path) -> None:
    from apeGmsh.results.readers._ladruno_element_io import (
        NodalForceColumnsUndecodedWarning,
    )

    # A 2-D element writing P1_1, P2_1, P3_1 (the PML / u-p / drilling
    # shape): forces x and y are its first two DOFs.
    odd = ("99-ThreeDofQuad[1:0:0]", _dof_node(2, 3, 2), [30])
    path = _file_with(tmp_path, "truss2d.ladruno", [odd])
    with LadrunoReader(path) as r:
        for comp in ("nodal_resisting_force_x", "nodal_resisting_force_y"):
            slab = _read(r, comp)
            np.testing.assert_array_equal(
                slab.values, _expected([30], comp, 2, 4), err_msg=comp)
        assert "nodal_resisting_moment_z" not in r.available_components(
            r.stages()[0].id, ResultLevel.ELEMENTS)
        with pytest.warns(NodalForceColumnsUndecodedWarning,
                          match=r"99-ThreeDofQuad.*P3_1"), \
                pytest.raises(KeyError, match="nodal_resisting_moment_z"):
            _read(r, "nodal_resisting_moment_z")
    # Beside beams, the moment read answers from the beams and still says
    # which bucket it left out.
    path = _file_with(tmp_path, "truss2d.ladruno", [odd, _BEAM_2D],
                      name="odd_and_beams.ladruno")
    with LadrunoReader(path) as r:
        with pytest.warns(NodalForceColumnsUndecodedWarning,
                          match="99-ThreeDofQuad"):
            slab = _read(r, "nodal_resisting_moment_z")
    assert slab.element_ids.tolist() == [7, 8]
    np.testing.assert_array_equal(
        slab.values, _expected([7, 8], "nodal_resisting_moment_z", 2, 4))


def test_3d_beam_reads_every_component(tmp_path: Path) -> None:
    path = _file_with(tmp_path, "truss2d.ladruno", [_BEAM_3D],
                      name="beam3d_global.ladruno")
    with h5py.File(path, "r+") as f:
        f["INFO"].attrs["SPATIAL_DIM"] = np.array([3], dtype=np.int32)
    with LadrunoReader(path) as r:
        for comp in _Q:
            slab = _read(r, comp)
            assert slab.values.shape == (4, 1, 2), comp
            np.testing.assert_array_equal(
                slab.values, _expected([1], comp, 2, 4), err_msg=comp)


def test_3d_brick_beside_a_beam_warns_and_reads_alone(tmp_path: Path) -> None:
    path = _file_with(tmp_path, "beam3d.ladruno", [_BEAM_3D, _BRICK_3D])
    with LadrunoReader(path) as r:
        # Two nodes and eight cannot share one (T, E, n_nodes) slab.
        with pytest.warns(ElementWidthMismatchWarning, match="node count"):
            slab = _read(r, "nodal_resisting_force_z")
        assert slab.values.shape[2] in (2, 8)
        # Scoped to the brick, its eight nodes come back in node order.
        slab = _read(r, "nodal_resisting_force_z", ids=[40])
        np.testing.assert_array_equal(
            slab.values, _expected([40], "nodal_resisting_force_z", 8, 1))
        slab = _read(r, "nodal_resisting_moment_y", ids=[1])
        np.testing.assert_array_equal(
            slab.values, _expected([1], "nodal_resisting_moment_y", 2, 1))


def test_available_components_list_the_canonical_names(tmp_path: Path) -> None:
    path = _file_with(tmp_path, "truss2d.ladruno", [_TRUSS_2D, _BEAM_2D])
    with LadrunoReader(path) as r:
        comps = r.available_components(r.stages()[0].id, ResultLevel.ELEMENTS)
    assert set(comps) == {
        "basicForce", "globalForce", "nodal_resisting_force_x",
        "nodal_resisting_force_y", "nodal_resisting_moment_z"}
    path = _file_with(tmp_path, "beam3d.ladruno", [_BEAM_3D])
    with LadrunoReader(path) as r:
        comps = r.available_components(r.stages()[0].id, ResultLevel.ELEMENTS)
    # The fixture's real ElasticBeam3d ``localForce`` bucket answers the
    # local-frame names too (#2124).
    assert set(comps) == {"localForce", "globalForce", *_Q, *_QL}


def test_raw_global_force_token_reads_unchanged(tmp_path: Path) -> None:
    path = _file_with(tmp_path, "beam3d.ladruno", [_BEAM_3D])
    with LadrunoReader(path) as r:
        slab = _read(r, "globalForce")
    with h5py.File(path, "r") as f:
        raw = f[f"{STAGE_GRP}/RESULTS/ON_ELEMENTS/globalForce/"
                f"5-ElasticBeam3d[1:0:0]/DATA"][...]
    np.testing.assert_array_equal(slab.values, raw)
    assert slab.values.shape == (1, 1, 12)


# ---------------------------------------------------------------------------
# Requests the file is unable to answer: refused by name
# ---------------------------------------------------------------------------


def test_file_without_global_force_refuses_naming_its_tokens() -> None:
    from apeGmsh.results.readers._ladruno_element_io import (
        NodalForceNotRecorded,
    )

    with LadrunoReader(FIXTURES / "quad2d.ladruno") as r:
        with pytest.raises(NodalForceNotRecorded) as exc:
            _read(r, "nodal_resisting_force_x")
        # The raw token read stays as it was.
        assert _read(r, "force").values.shape == (2, 1, 8)
    msg = str(exc.value)
    assert "globalForce" in msg and "'force'" in msg and "'stress'" in msg


def test_component_the_buckets_lack_refuses_naming_what_they_hold(
        tmp_path: Path) -> None:
    from apeGmsh.results.readers._ladruno_element_io import (
        NodalForceNotRecorded,
    )

    path = _file_with(tmp_path, "truss2d.ladruno", [_TRUSS_2D])
    with LadrunoReader(path) as r:
        with pytest.raises(NodalForceNotRecorded,
                           match=r"decode to \['nodal_resisting_force_x', "
                                 r"'nodal_resisting_force_y'\]"):
            _read(r, "nodal_resisting_moment_z")
        # A local-frame name reads localForce, which this file lacks.
        with pytest.raises(NodalForceNotRecorded,
                           match="'localForce', which this file does not "
                                 "record"):
            _read(r, "nodal_resisting_force_local_x")


# ---------------------------------------------------------------------------
# Partition sets
# ---------------------------------------------------------------------------


def _parts(tmp_path: Path, part0, part1) -> list[Path]:
    out = []
    for i, buckets in enumerate((part0, part1)):
        src = f"truss2d.part-{i}.ladruno"
        if buckets:
            out.append(_file_with(tmp_path, src, buckets))
        else:
            out.append(tmp_path / src)
            shutil.copyfile(FIXTURES / src, out[-1])
    return out


def test_partitions_answer_like_the_serial_file(tmp_path: Path) -> None:
    truss = _TRUSS_2D[1]
    paths = _parts(tmp_path, [("12-Truss[1:0:0]", truss, [1])],
                   [("12-Truss[1:0:0]", truss, [2])])
    serial = _file_with(tmp_path, "truss2d.ladruno", [_TRUSS_2D])
    for comp in ("nodal_resisting_force_x", "nodal_resisting_force_y"):
        with LadrunoMultiPartitionReader(paths) as m:
            got = _read(m, comp)
            assert comp in m.available_components(
                m.stages()[0].id, ResultLevel.ELEMENTS)
        with LadrunoReader(serial) as s:
            want = _read(s, comp)
        order = np.argsort(got.element_ids)
        assert got.element_ids[order].tolist() == [1, 2]
        np.testing.assert_array_equal(got.values[:, order], want.values)
        np.testing.assert_array_equal(
            want.values, _expected([1, 2], comp, 2, 4))


def test_partition_without_global_force_drops_out(tmp_path: Path) -> None:
    from apeGmsh.results.readers._ladruno_element_io import (
        NodalForceNotRecorded,
    )

    # Rank 1 owns no element answering globalForce: rank 0 answers alone.
    paths = _parts(tmp_path, [("12-Truss[1:0:0]", _TRUSS_2D[1], [1])], None)
    with LadrunoMultiPartitionReader(paths) as m:
        slab = _read(m, "nodal_resisting_force_x")
    assert slab.element_ids.tolist() == [1]
    np.testing.assert_array_equal(
        slab.values, _expected([1], "nodal_resisting_force_x", 2, 4))
    # No rank records it: the set refuses, naming what it records.
    for p in paths:
        shutil.copyfile(FIXTURES / p.name, p)
    with LadrunoMultiPartitionReader(paths) as m:
        with pytest.raises(NodalForceNotRecorded, match="basicForce"):
            _read(m, "nodal_resisting_force_x")


# ---------------------------------------------------------------------------
# Parity with the MPCO reader on a real fork-written bucket
# ---------------------------------------------------------------------------


def test_matches_the_mpco_reader_on_a_real_elastic_beam(tmp_path: Path) -> None:
    from apeGmsh.results.readers._mpco import MPCOReader

    src = Path(__file__).resolve().parent / "fixtures" / "results"
    mpco = tmp_path / "frame.mpco"
    shutil.copyfile(src / "elasticFrame.mpco", mpco)
    key = "5-ElasticBeam3d[1:0:0]"
    with h5py.File(mpco, "r+") as f:
        on_e = f[f"{STAGE_GRP}/RESULTS/ON_ELEMENTS"]
        # ElasticBeam3d answers force and globalForce with one vector.
        on_e.move("force", "globalForce")
        bucket = on_e[f"globalForce/{key}"]
        names = bucket["META/COMPONENTS"][...].ravel()[0]
        names = (names.decode() if isinstance(names, bytes) else str(names))
        names = names.split(".", 1)[1].split(",")
        steps = sorted(bucket["DATA"], key=lambda s: int(s.split("_")[1]))
        data = np.stack([bucket["DATA"][s][...] for s in steps])
        ids = bucket["ID"][...].ravel().astype(int).tolist()
    assert names[:3] == ["Px_1", "Py_1", "Pz_1"], names

    # The .ladruno twin keeps the fixture's four steps: the first four of
    # the MPCO run.
    n_steps = 4
    lad = _file_with(tmp_path, "truss2d.ladruno", [], name="frame.ladruno")
    with h5py.File(lad, "r+") as f:
        f["INFO"].attrs["SPATIAL_DIM"] = np.array([3], dtype=np.int32)
        on_e = f[f"{STAGE_GRP}/RESULTS/ON_ELEMENTS"]
        layout = [(n, 0, None) for n in names]
        _write_bucket(on_e, "globalForce", key, layout, ids, n_steps)
        on_e[f"globalForce/{key}/DATA"][...] = data[:n_steps]
        del on_e["basicForce"]

    nonzero = set()
    with MPCOReader(mpco) as m, LadrunoReader(lad) as r:
        m_stage = m.stages()[0].id
        for comp in _Q:
            want = m.read_elements(m_stage, comp)
            got = _read(r, comp)
            assert want.values.shape == (data.shape[0], len(ids), 2), comp
            assert got.element_ids.tolist() == want.element_ids.tolist()
            np.testing.assert_array_equal(got.values, want.values[:n_steps],
                                          err_msg=comp)
            if np.abs(want.values[:n_steps]).max() > 0.0:
                nonzero.add(comp)
    # The frame is planar (x-z): its in-plane components carry the check.
    assert nonzero >= {"nodal_resisting_force_x", "nodal_resisting_force_z",
                       "nodal_resisting_moment_y"}, nonzero


# ---------------------------------------------------------------------------
# Local-frame names from ``localForce`` (#2124)
# ---------------------------------------------------------------------------
#
# The same constructive oracle: each synthetic ``localForce`` bucket holds
# ``_truth(e, n, q, t)`` in the column the class's NODAL_FORCE_CATALOG
# layout gives ``q`` at node ``n``, spelled as the fork spells it. The
# real-file cases read a fork-written ``.ladruno`` against its own raw
# columns and a fork-written MPCO bucket against the MPCO reader.

_V_M = {"nodal_resisting_force_local_y": "V",
        "nodal_resisting_moment_local_z": "M"}
_LBEAM_2D = ("5-ElasticBeam2d[1:0:0]", _local_named(2, _LOCAL_2D, base=_V_M),
             [7, 8])
# DispBeamColumn2d drops the underscore: N1, V1, M1, N2, ...
_LDISP_2D = ("62-DispBeamColumn2d[1000:1:0]",
             _local_named(2, _LOCAL_2D, base=_V_M, sep=""), [3])
_LBEAM_3D = ("5-ElasticBeam3d[1:0:0]", _local_named(2, tuple(_QL)), [1])
# A class NODAL_FORCE_CATALOG has no localForce layout for.
_LLINK_2D = ("86-TwoNodeLink2d[1:0:0]", _local_named(2, _LOCAL_2D), [50])
_NAMES_2D = ["N_1", "V_1", "M_1", "N_2", "V_2", "M_2"]
_NAMES_3D = [name for name, _, _ in _LBEAM_3D[1]]


def test_decode_local_reads_each_catalogued_spelling() -> None:
    from apeGmsh.results.readers._ladruno_element_io import (
        decode_local_nodal_force_columns,
    )

    cols = decode_local_nodal_force_columns(_NAMES_3D, "ElasticBeam3d")
    assert {c: v.tolist() for c, v in cols.items()} == {
        c: [k, k + 6] for k, c in enumerate(_QL)}
    want_2d = {"nodal_resisting_force_local_x": [0, 3],
               "nodal_resisting_force_local_y": [1, 4],
               "nodal_resisting_moment_local_z": [2, 5]}
    for names, cls in (
        (_NAMES_2D, "ElasticBeam2d"),
        (["N1", "V1", "M1", "N2", "V2", "M2"], "DispBeamColumn2d"),
        (["N_1", "Vy_1", "Mz_1", "N_2", "Vy_2", "Mz_2"], "ForceBeamColumn2d"),
    ):
        cols = decode_local_nodal_force_columns(names, cls)
        assert {c: v.tolist() for c, v in cols.items()} == want_2d, cls


@pytest.mark.parametrize("names, cls, match", [
    # No catalog layout: refused by class, whatever the names say.
    (_NAMES_2D, "TwoNodeLink2d", "'TwoNodeLink2d' has no 'local_force'"),
    (_NAMES_3D, "ComponentElement3d", "'ComponentElement3d' has no"),
    # 2-D names on a 3-D class: the shear and moment axes are unknown.
    (_NAMES_2D, "ElasticBeam3d", "do not spell"),
    ([n.replace("Vy", "V").replace("Mz", "M") for n in _NAMES_3D],
     "ElasticBeam3d", "do not spell"),
    # A 2-D class whose names disagree with the catalog's order or ends.
    (["V_1", "N_1", "M_1", "V_2", "N_2", "M_2"], "ElasticBeam2d",
     "do not spell"),
    (["N_1", "V_1", "M_1", "N_1", "V_1", "M_1"], "ElasticBeam2d",
     "do not spell"),
    (["N_1", "V_1", "M_1"], "ElasticBeam2d", "do not spell"),
    (["C1", "C2", "C3", "C4", "C5", "C6"], "ElasticBeam2d", "do not spell"),
])
def test_decode_local_refuses_what_the_catalog_does_not_describe(
        names, cls, match) -> None:
    from apeGmsh.results.readers._ladruno_element_io import (
        decode_local_nodal_force_columns,
    )

    with pytest.raises(ValueError, match=match):
        decode_local_nodal_force_columns(names, cls)


def test_local_2d_classes_read_per_node_and_name_the_one_left_out(
        tmp_path: Path) -> None:
    from apeGmsh.results.readers._ladruno_element_io import (
        NodalForceColumnsUndecodedWarning,
    )

    path = _file_with(tmp_path, "truss2d.ladruno", [],
                      local=[_LBEAM_2D, _LDISP_2D, _LLINK_2D])
    with LadrunoReader(path) as r:
        listed = r.available_components(r.stages()[0].id,
                                        ResultLevel.ELEMENTS)
        assert set(_LOCAL_2D) <= set(listed)
        assert "nodal_resisting_force_local_z" not in listed
        for comp in _LOCAL_2D:
            with pytest.warns(NodalForceColumnsUndecodedWarning,
                              match=r"86-TwoNodeLink2d.*NODAL_FORCE_CATALOG"):
                slab = _read(r, comp)
            ids = slab.element_ids.tolist()
            assert sorted(ids) == [3, 7, 8], comp
            np.testing.assert_array_equal(
                slab.values, _expected(ids, comp, 2, 4), err_msg=comp)
        # The id filter and the time slice act as on a raw read.
        with pytest.warns(NodalForceColumnsUndecodedWarning):
            slab = r.read_elements(
                r.stages()[0].id, "nodal_resisting_moment_local_z",
                element_ids=np.array([8, 3]), time_slice=-1)
        ids = slab.element_ids.tolist()
        assert sorted(ids) == [3, 8]
        np.testing.assert_array_equal(
            slab.values,
            _expected(ids, "nodal_resisting_moment_local_z", 2, 4)[-1:])


def test_local_read_of_only_uncatalogued_classes_refuses(
        tmp_path: Path) -> None:
    from apeGmsh.results.readers._ladruno_element_io import (
        NodalForceColumnsUndecodedWarning,
        NodalForceNotRecorded,
    )

    path = _file_with(tmp_path, "truss2d.ladruno", [], local=[_LLINK_2D])
    with LadrunoReader(path) as r:
        assert not [c for c in r.available_components(
            r.stages()[0].id, ResultLevel.ELEMENTS) if "_local_" in c]
        with pytest.warns(NodalForceColumnsUndecodedWarning,
                          match="'TwoNodeLink2d' has no 'local_force'"), \
                pytest.raises(NodalForceNotRecorded,
                              match=r"86-TwoNodeLink2d\[1:0:0\]"):
            _read(r, "nodal_resisting_force_local_x")
        # The raw token read stays as it was.
        assert _read(r, "localForce").values.shape == (4, 1, 6)


def test_local_2d_names_on_a_3d_catalogued_class_are_refused(
        tmp_path: Path) -> None:
    """The ComponentElement3d trap on a class the catalog does list: 2-D
    names on a 3-D layout are left out, never read as the 3-D columns."""
    from apeGmsh.results.readers._ladruno_element_io import (
        NodalForceColumnsUndecodedWarning,
        NodalForceNotRecorded,
    )

    trap = ("5-ElasticBeam3d[1:0:0]", _local_named(2, _LOCAL_2D, base=_V_M),
            [1])
    path = _file_with(tmp_path, "truss2d.ladruno", [], local=[trap],
                      name="trap3d.ladruno")
    with h5py.File(path, "r+") as f:
        f["INFO"].attrs["SPATIAL_DIM"] = np.array([3], dtype=np.int32)
    with LadrunoReader(path) as r:
        with pytest.warns(NodalForceColumnsUndecodedWarning,
                          match="do not spell"), \
                pytest.raises(NodalForceNotRecorded):
            _read(r, "nodal_resisting_force_local_x")


def test_local_3d_beam_reads_every_component(tmp_path: Path) -> None:
    path = _file_with(tmp_path, "truss2d.ladruno", [], local=[_LBEAM_3D],
                      name="lbeam3d.ladruno")
    with h5py.File(path, "r+") as f:
        f["INFO"].attrs["SPATIAL_DIM"] = np.array([3], dtype=np.int32)
    with LadrunoReader(path) as r:
        for comp in _QL:
            slab = _read(r, comp)
            assert slab.values.shape == (4, 1, 2), comp
            np.testing.assert_array_equal(
                slab.values, _expected([1], comp, 2, 4), err_msg=comp)


def test_local_names_do_not_depend_on_the_spatial_dimension(
        tmp_path: Path) -> None:
    """An indexed globalForce bucket makes the global names refuse when
    the dimension is unknown; the localForce layout comes from the class,
    so the local names still read and are still listed."""
    from apeGmsh.results.readers._ladruno import SpatialDimUnknown

    path = _file_with(tmp_path, "truss2d.ladruno", [_TRUSS_2D],
                      local=[_LBEAM_2D])
    with h5py.File(path, "r+") as f:
        del f["INFO"].attrs["SPATIAL_DIM"]
        del f[f"{STAGE_GRP}/MODEL"]
    with LadrunoReader(path) as r:
        with pytest.warns(UserWarning, match="spatial dimension is unknown"):
            listed = r.available_components(r.stages()[0].id,
                                            ResultLevel.ELEMENTS)
        assert set(_LOCAL_2D) <= set(listed)
        assert "nodal_resisting_force_x" not in listed
        with pytest.raises(SpatialDimUnknown):
            _read(r, "nodal_resisting_force_x")
        for comp in _LOCAL_2D:
            slab = _read(r, comp)
            np.testing.assert_array_equal(
                slab.values, _expected([7, 8], comp, 2, 4), err_msg=comp)


def test_local_partitions_answer_like_the_serial_file(tmp_path: Path) -> None:
    key, layout, _ = _LBEAM_2D
    paths = [
        _file_with(tmp_path, f"truss2d.part-{i}.ladruno", [],
                   local=[(key, layout, ids)])
        for i, ids in enumerate(([7], [8]))
    ]
    with LadrunoMultiPartitionReader(paths) as m:
        for comp in _LOCAL_2D:
            got = _read(m, comp)
            order = np.argsort(got.element_ids)
            assert got.element_ids[order].tolist() == [7, 8]
            np.testing.assert_array_equal(
                got.values[:, order], _expected([7, 8], comp, 2, 4))


def test_real_ladruno_local_force_reads_its_own_columns() -> None:
    """``beam3d.ladruno`` is fork-written: its ElasticBeam3d ``localForce``
    bucket reads back columns ``k`` and ``k + 6`` for local component ``k``."""
    with LadrunoReader(FIXTURES / "beam3d.ladruno") as r:
        raw = _read(r, "localForce").values
        assert raw.shape == (1, 1, 12) and np.abs(raw).max() > 0.0
        for k, comp in enumerate(_QL):
            slab = _read(r, comp)
            np.testing.assert_array_equal(slab.values, raw[:, :, [k, k + 6]],
                                          err_msg=comp)


def test_local_matches_the_mpco_reader_on_a_real_elastic_beam(
        tmp_path: Path) -> None:
    from apeGmsh.results.readers._mpco import MPCOReader

    src = Path(__file__).resolve().parent / "fixtures" / "results"
    mpco = tmp_path / "frame.mpco"
    shutil.copyfile(src / "elasticFrame.mpco", mpco)
    key = "5-ElasticBeam3d[1:0:0]"
    with h5py.File(mpco, "r") as f:
        bucket = f[f"{STAGE_GRP}/RESULTS/ON_ELEMENTS/localForce/{key}"]
        names = bucket["META/COMPONENTS"][...].ravel()[0]
        names = (names.decode() if isinstance(names, bytes) else str(names))
        names = names.split(".", 1)[1].split(",")
        steps = sorted(bucket["DATA"], key=lambda s: int(s.split("_")[1]))
        data = np.stack([bucket["DATA"][s][...] for s in steps])
        ids = bucket["ID"][...].ravel().astype(int).tolist()
    assert names == _NAMES_3D, names

    n_steps = 4
    lad = _file_with(tmp_path, "truss2d.ladruno", [], name="lframe.ladruno")
    with h5py.File(lad, "r+") as f:
        f["INFO"].attrs["SPATIAL_DIM"] = np.array([3], dtype=np.int32)
        on_e = f[f"{STAGE_GRP}/RESULTS/ON_ELEMENTS"]
        _write_bucket(on_e, "localForce", key,
                      [(n, 0, None) for n in names], ids, n_steps)
        on_e[f"localForce/{key}/DATA"][...] = data[:n_steps]
        del on_e["basicForce"]

    nonzero = set()
    with MPCOReader(mpco) as m, LadrunoReader(lad) as r:
        m_stage = m.stages()[0].id
        assert set(_QL) <= set(m.available_components(
            m_stage, ResultLevel.ELEMENTS))
        assert set(_QL) <= set(r.available_components(
            r.stages()[0].id, ResultLevel.ELEMENTS))
        for comp in _QL:
            want = m.read_elements(m_stage, comp)
            got = _read(r, comp)
            assert want.values.shape == (data.shape[0], len(ids), 2), comp
            assert got.element_ids.tolist() == want.element_ids.tolist()
            np.testing.assert_array_equal(got.values, want.values[:n_steps],
                                          err_msg=comp)
            if np.abs(want.values[:n_steps]).max() > 0.0:
                nonzero.add(comp)
    # The planar frame loads its local axial, one shear and one moment.
    assert len(nonzero) >= 3, nonzero
