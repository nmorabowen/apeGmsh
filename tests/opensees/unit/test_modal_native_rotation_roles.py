"""#2203: the native modes writers keep a rotation slot only where the
model's DOF roles say it holds a rotation.

``ParallelModalResult.to_native(..., *, model_h5=None)`` and
``EigenResult.to_native(..., *, model_h5=None)`` lay each node's DOFs onto
``ux..rz`` by its own ndf and ask
``_recorder_translate.model_h5_rotation_roles`` (#2153's role model, the
rule ``to_ladruno`` follows since #2178) before writing a ``rotation_*``.

Oracles, each a value the test names:

* the review's reproducer: a deck padded to ndf 6 and a node row
  ``[0.1, 0.2, 0.3, 777, 0, 0]`` (a 3-D u-p node: ux, uy, uz, p). 777 is
  never written as a rotation, with or without ``model_h5``;
* a 2-D u-p node (ux, uy, p) writes ``rotation_z = 0.0``, never its
  pressure, and a 2-D frame node (ux, uy, rz) writes its real
  ``rotation_z``: ndf 3 and ndm 2 cannot tell the two apart, the roles do;
* without ``model_h5`` a slot only the roles could settle is omitted with
  one ``RotationRolesUnavailableWarning``; a deck whose DOF list reaches no
  rotation slot (3-D ndf 3 or 4) writes translations, silently;
* the two writers give the same components and values on the same vectors.

The bridge runs against the fake OpenSees module of
``tests/test_eigen_to_native.py``; no backend is needed.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.analysis.eigen import RotationSlotUnresolvedWarning
from apeGmsh.opensees.analysis.modal import (
    ParallelModalResult,
    RotationRolesUnavailableWarning,
)
from apeGmsh.opensees.emitter import live
from apeGmsh.opensees.opensees_model import OpenSeesModel

from tests.test_eigen_to_native import _build_fem, _FakeOps, _frame, _vectors
from tests.test_parallel_modal_to_native import _NdfFem
from tests.test_results_ladruno_writer_modes import (
    _ForkFake,
    _up_beside_frame_2d,
)

_ROT = ("rotation_x", "rotation_y", "rotation_z")


@pytest.fixture(scope="module")
def fem3() -> Any:
    return _build_fem(3)


@pytest.fixture(scope="module")
def fem2() -> Any:
    return _build_fem(2)


@pytest.fixture
def bind(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Bind ``fake`` as the module the resolver answers."""
    def _bind(fake: _FakeOps) -> _FakeOps:
        monkeypatch.setattr(live, "_get_ops", lambda: fake)
        monkeypatch.setattr(live, "_BACKEND_INFO", None)
        return fake
    return _bind


def _ids(fem: Any) -> np.ndarray:
    return np.asarray(fem.nodes.ids, dtype=np.int64)


def _job(tmp_path: Path, ids: Any, shapes: np.ndarray, ndm: int) -> Path:
    """A replicated-deck run dir holding ``shapes`` (``(k, n, ndf)``)."""
    job = tmp_path / "job"
    job.mkdir()
    k = shapes.shape[0]
    (job / "eigenvalues.out").write_text(
        " ".join(repr(16.0 * (i + 1)) for i in range(k)))
    (job / "mode_shapes.json").write_text(json.dumps({
        "nodes": [int(n) for n in ids], "ndf": int(shapes.shape[2]),
        "ndm": ndm}))
    for i in range(k):
        (job / f"mode_shape_{i + 1}.out").write_text(
            " ".join(repr(float(v)) for v in shapes[i].ravel()))
    return job


def _comps(path: Path, mode: int = 1) -> dict[str, dict[int, float]]:
    """``{component: {node: value}}`` of the native stage of ``mode``."""
    with h5py.File(path, "r") as f:
        (stage,) = [s for s in f["stages"]
                    if int(f["stages"][s].attrs["mode_index"]) == mode]
        grp = f["stages"][stage]["partitions"]["partition_0"]["nodes"]
        ids = [int(n) for n in grp["_ids"][()]]
        return {
            c: dict(zip(ids, np.asarray(grp[c][()])[0].tolist()))
            for c in grp if c != "_ids"
        }


def _native(
    res: ParallelModalResult, path: Path, fem: Any, **kw: Any,
) -> tuple[dict[str, dict[int, float]], list[warnings.WarningMessage]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res.to_native(path, fem, **kw)
    return _comps(path), caught


def _of(caught: list[warnings.WarningMessage], cls: type) -> list[Any]:
    return [w for w in caught if issubclass(w.category, cls)]


# ---------------------------------------------------------------------------
# The review's reproducer: a 3-D u-p row in an ndf-6 deck
# ---------------------------------------------------------------------------

def test_reproducer_without_model_h5_writes_no_rotation(
    tmp_path: Path,
) -> None:
    """Sidecar ndf 6, row ``[0.1, 0.2, 0.3, 777, 0, 0]``, on a fem with
    solid cells and no per-node ndf: column 4 may be a rotation or a
    pressure, so no rotation_* is written, with one warning that names
    the fix, raised from the caller's line."""
    shapes = np.array([[[0.1, 0.2, 0.3, 777.0, 0.0, 0.0]]])
    res = ParallelModalResult.from_job(_job(tmp_path, [1], shapes, 3))

    comps, caught = _native(
        res, tmp_path / "m.h5", _NdfFem([1], {}, solid=True))

    assert set(comps) == {"displacement_x", "displacement_y", "displacement_z"}
    assert [comps[c][1] for c in sorted(comps)] == [0.1, 0.2, 0.3]
    named = _of(caught, RotationRolesUnavailableWarning)
    assert len(named) == 1 and len(_of(caught, UserWarning)) == 1
    assert "model_h5=" in str(named[0].message)
    assert "DOFs 1..6" in str(named[0].message)
    assert Path(named[0].filename) == Path(__file__)


def test_reproducer_with_model_h5_writes_zero_not_777(
    g: Any, tmp_path: Path,
) -> None:
    """A real 3-D model: a LadrunoUP box (ndf 4) beside a frame column
    (ndf 6), the deck padded to ndf 6. Every u-p row is ``[.., .., .., 777,
    0, 0]``: its rotations are 0.0, and the frame keeps its own."""
    G = g.model.geometry
    box = G.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    col = G.add_line(G.add_point(3.0, 0.0, 0.0), G.add_point(3.0, 0.0, 2.0))
    g.model.sync()
    g.physical.add(3, [box], name="Soil")
    g.physical.add(1, [col], name="Col")
    g.mesh.structured.set_transfinite_box("Soil", n=2)
    g.mesh.generation.generate(3)
    g.mesh.partitioning.renumber(base=1)
    fem = g.mesh.queries.get_fem_data()
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    ops.element.LadrunoUP(
        pg="Soil",
        material=ops.nDMaterial.ElasticIsotropic(E=1e4, nu=0.3, rho=2.0),
        Kf=2.2e6, poro=0.4, rhoF=1.0, perm=(1e-4, 1e-4, 1e-4),
    )
    ops.element.elasticBeamColumn(
        pg="Col", transf=ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0)),
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4)
    archive = tmp_path / "model.h5"
    ops.h5(str(archive))
    up = sorted(int(n) for n in fem.nodes.select(pg="Soil").ids)
    frame = sorted(int(n) for n in fem.nodes.select(pg="Col").ids)
    ndf = OpenSeesModel.from_h5(str(archive)).nodes_ndf
    assert {ndf[n] for n in up} == {4} and {ndf[n] for n in frame} == {6}

    ids = _ids(fem)
    row = {int(n): i for i, n in enumerate(ids)}
    shapes = np.zeros((1, ids.size, 6))
    for n in up:
        shapes[0, row[n]] = [0.1, 0.2, 0.3, 777.0, 0.0, 0.0]
    for n in frame:
        shapes[0, row[n]] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    res = ParallelModalResult.from_job(_job(tmp_path, ids, shapes, 3))

    comps, caught = _native(res, tmp_path / "m.h5", fem, model_h5=archive)

    assert caught == []
    for n in up:
        assert [comps[c][n] for c in _ROT] == [0.0, 0.0, 0.0]
        assert comps["displacement_x"][n] == 0.1
    for n in frame:
        assert [comps[c][n] for c in _ROT] == [4.0, 5.0, 6.0]
    assert 777.0 not in {v for c in comps.values() for v in c.values()}


def _frame_and_up_job(tmp_path: Path) -> tuple[ParallelModalResult, np.ndarray]:
    """Nodes 1, 2 a 3-D frame (ux..rz); nodes 3, 4 a 3-D u-p solid
    (ux, uy, uz, p), padded to the deck's ndf 6 with p = 777."""
    shapes = np.array([[
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
        [0.1, 0.2, 0.3, 777.0, 0.0, 0.0],
        [0.4, 0.5, 0.6, 777.0, 0.0, 0.0],
    ]])
    return (ParallelModalResult.from_job(
        _job(tmp_path, [1, 2, 3, 4], shapes, 3)), shapes)


def test_3d_frame_beside_up_without_model_h5_keeps_frame_rotations(
    tmp_path: Path,
) -> None:
    """``fem`` carries each node's ndf: the ndf-6 frame nodes keep their
    rotations, the ndf-4 u-p nodes get 0.0 (never 777), and nothing warns."""
    res, shapes = _frame_and_up_job(tmp_path)
    fem = _NdfFem([1, 2, 3, 4], {1: 6, 2: 6, 3: 4, 4: 4})

    comps, caught = _native(res, tmp_path / "m.h5", fem)

    assert caught == []
    assert set(_ROT) <= set(comps)
    for i, n in enumerate((1, 2)):
        assert [comps[c][n] for c in _ROT] == shapes[0, i, 3:].tolist()
    for i, n in ((2, 3), (3, 4)):
        assert [comps[c][n] for c in _ROT] == [0.0, 0.0, 0.0]
        assert comps["displacement_z"][n] == shapes[0, i, 2]
    assert 777.0 not in {v for c in comps.values() for v in c.values()}


def test_a_node_of_unknown_ndf_writes_no_rotation_and_is_named(
    tmp_path: Path,
) -> None:
    """``fem`` holds no ndf for node 3: it writes 0.0, not 777, and the
    one warning names it; the frame nodes keep their rotations."""
    res, shapes = _frame_and_up_job(tmp_path)
    fem = _NdfFem([1, 2, 3, 4], {1: 6, 2: 6, 4: 4})

    comps, caught = _native(res, tmp_path / "m.h5", fem)

    named = _of(caught, RotationRolesUnavailableWarning)
    assert len(named) == 1 and len(_of(caught, UserWarning)) == 1
    assert "1 node(s) ([3])" in str(named[0].message)
    assert [comps[c][3] for c in _ROT] == [0.0, 0.0, 0.0]
    assert [comps[c][1] for c in _ROT] == shapes[0, 0, 3:].tolist()


def test_2d_without_model_h5_ndf_2_is_silent_ndf_3_is_named(
    tmp_path: Path,
) -> None:
    """2-D deck of ndf 3: a node of ndf 2 reaches no rotation slot
    (silent); a node of ndf 3 may be a frame or a u-p node, so it writes
    no rotation_z and the warning names it."""
    shapes = np.array([[[0.1, 0.2, 0.0], [1.1, 1.2, 777.0]]])
    res = ParallelModalResult.from_job(_job(tmp_path, [1, 2], shapes, 2))

    comps, caught = _native(res, tmp_path / "a.h5", _NdfFem([1, 2], 2))
    assert caught == []
    assert set(comps) == {"displacement_x", "displacement_y"}

    comps, caught = _native(
        res, tmp_path / "b.h5", _NdfFem([1, 2], {1: 2, 2: 3}))
    named = _of(caught, RotationRolesUnavailableWarning)
    assert len(named) == 1 and "([2])" in str(named[0].message)
    assert set(comps) == {"displacement_x", "displacement_y"}


@pytest.mark.parametrize("ndf", [3, 4])
def test_3d_deck_without_a_rotation_slot_is_silent(
    fem3: Any, tmp_path: Path, ndf: int,
) -> None:
    """ndf 3 (solid) or 4 (u-p) in 3-D: the DOF list reaches no rotation
    slot, so ndf settles it; no rotation, no warning, a u-p pressure is
    dropped."""
    shapes = np.array([[[0.1, 0.2, 0.3, 777.0][:ndf]]])
    res = ParallelModalResult.from_job(_job(tmp_path, [1], shapes, 3))

    comps, caught = _native(res, tmp_path / "m.h5", fem3)

    assert set(comps) == {"displacement_x", "displacement_y", "displacement_z"}
    assert caught == []


# ---------------------------------------------------------------------------
# 2-D: a u-p node (ux, uy, p) and a frame node (ux, uy, rz) share ndf 3
# ---------------------------------------------------------------------------

def test_2d_up_beside_frame_parallel(g: Any, tmp_path: Path) -> None:
    """With ``model_h5``: u-p nodes write rotation_z 0.0 (not p), the frame
    keeps its real rotation_z, the element-less node is 0.0 with one
    RotationSlotUnresolvedWarning. Without it: no rotation_z, one
    RotationRolesUnavailableWarning."""
    fem, ops, up, frame, free = _up_beside_frame_2d(g)
    archive = tmp_path / "model.h5"
    ops.h5(str(archive))
    ids = _ids(fem)
    shapes = np.array([[[10.0 + i, 20.0 + i, 30.0 + i]
                        for i in range(ids.size)]])
    row = {int(n): i for i, n in enumerate(ids)}
    res = ParallelModalResult.from_job(_job(tmp_path, ids, shapes, 2))

    comps, caught = _native(res, tmp_path / "a.h5", fem, model_h5=archive)

    assert set(comps) == {"displacement_x", "displacement_y", "rotation_z"}
    for n in up:
        assert comps["rotation_z"][n] == 0.0
        assert comps["displacement_y"][n] == shapes[0, row[n], 1]
    for n in frame:
        assert comps["rotation_z"][n] == shapes[0, row[n], 2]
    assert comps["rotation_z"][free] == 0.0
    unresolved = _of(caught, RotationSlotUnresolvedWarning)
    assert len(unresolved) == 1 and len(_of(caught, UserWarning)) == 1
    assert f"[{free}]" in str(unresolved[0].message)
    assert Path(unresolved[0].filename) == Path(__file__)

    comps, caught = _native(res, tmp_path / "b.h5", fem)

    assert set(comps) == {"displacement_x", "displacement_y"}
    assert len(_of(caught, RotationRolesUnavailableWarning)) == 1
    assert len(_of(caught, UserWarning)) == 1


def test_2d_up_beside_frame_eigen_agrees_with_parallel(
    g: Any, bind: Any, tmp_path: Path,
) -> None:
    """The same vectors through both writers: the same components and the
    same values, node for node (atol 0)."""
    fem, ops, up, frame, free = _up_beside_frame_2d(g)
    ids = _ids(fem)
    vecs = _vectors(ids, 2, {int(n): 3 for n in ids})
    bind(_ForkFake([16.0, 81.0], vecs, build="0" * 40))
    r = ops.eigen(2)
    archive = tmp_path / "model.h5"
    ops.h5(str(archive))

    with pytest.warns(RotationSlotUnresolvedWarning, match=rf"\[{free}\]"):
        eig = r.to_native(tmp_path / "eigen.h5")
    shapes = np.array([[vecs[(int(n), k)] for n in ids] for k in (1, 2)])
    res = ParallelModalResult.from_job(_job(tmp_path, ids, shapes, 2))
    with pytest.warns(RotationSlotUnresolvedWarning, match=rf"\[{free}\]"):
        res.to_native(tmp_path / "par.h5", fem, model_h5=archive)

    for mode in (1, 2):
        a = _comps(tmp_path / "eigen.h5", mode)
        b = _comps(tmp_path / "par.h5", mode)
        assert a == b
        assert set(a) == {"displacement_x", "displacement_y", "rotation_z"}
    a = _comps(tmp_path / "eigen.h5")
    for n in up:
        assert a["rotation_z"][n] == 0.0
    for n in frame:
        assert a["rotation_z"][n] == vecs[(n, 1)][2]


def test_2d_frame_keeps_rotation_z_in_both_writers(
    fem2: Any, bind: Any, tmp_path: Path,
) -> None:
    """A pure 2-D frame (ndf 3): rotation_z is entry 3 of every node's
    vector, in EigenResult.to_native and in ParallelModalResult.to_native
    with the archive, with no warning."""
    ids = _ids(fem2)
    vecs = _vectors(ids, 1, {int(n): 3 for n in ids})
    bind(_FakeOps([16.0], vecs))
    ops = _frame(fem2, ndm=2)
    r = ops.eigen(1)
    archive = tmp_path / "model.h5"
    ops.h5(str(archive))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        eig = _comps(r.to_native(tmp_path / "eigen.h5"))
        shapes = np.array([[vecs[(int(n), 1)] for n in ids]])
        res = ParallelModalResult.from_job(_job(tmp_path, ids, shapes, 2))
        res.to_native(tmp_path / "par.h5", fem2, model_h5=archive)
    par = _comps(tmp_path / "par.h5")

    assert eig == par
    assert eig["rotation_z"] == {int(n): vecs[(int(n), 1)][2] for n in ids}


# ---------------------------------------------------------------------------
# EigenResult: element-less nodes in a u-p model; the model_h5 checks
# ---------------------------------------------------------------------------

def test_eigen_3d_elementless_node_in_a_up_model_gets_no_rotation(
    g: Any, bind: Any, tmp_path: Path,
) -> None:
    """A 3-D u-p box (ndf 4) and an element-less node (ndf 6): the node's
    rotation slots could be tied to a pore pressure, so 0.0 with one
    warning; no node holds a rotation, so no rotation_* is written."""
    G = g.model.geometry
    box = G.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    pt = G.add_point(5.0, 5.0, 5.0)
    g.model.sync()
    g.physical.add(3, [box], name="Soil")
    g.physical.add(0, [pt], name="Free")
    g.mesh.structured.set_transfinite_box("Soil", n=2)
    g.mesh.generation.generate(3)
    g.mesh.partitioning.renumber(base=1)
    fem = g.mesh.queries.get_fem_data()
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    ops.element.LadrunoUP(
        pg="Soil",
        material=ops.nDMaterial.ElasticIsotropic(E=1e4, nu=0.3, rho=2.0),
        Kf=2.2e6, poro=0.4, rhoF=1.0, perm=(1e-4, 1e-4, 1e-4),
    )
    (free,) = [int(n) for n in fem.nodes.select(pg="Free").ids]
    ids = _ids(fem)
    lengths = {int(n): 4 for n in ids}
    lengths[free] = 6
    vecs = _vectors(ids, 1, lengths)
    bind(_ForkFake([16.0], vecs, build="0" * 40))

    with pytest.warns(RotationSlotUnresolvedWarning, match=rf"\[{free}\]"):
        out = ops.eigen(1).to_native(tmp_path / "m.h5")

    comps = _comps(out)
    assert set(comps) == {"displacement_x", "displacement_y", "displacement_z"}
    assert comps["displacement_z"][free] == vecs[(free, 1)][2]


def test_eigen_model_h5_of_another_dimension_is_refused(
    fem3: Any, fem2: Any, bind: Any, tmp_path: Path,
) -> None:
    other = tmp_path / "other.h5"
    _frame(fem2, ndm=2).h5(str(other))
    ids = _ids(fem3)
    bind(_FakeOps([16.0], _vectors(ids, 1, {int(n): 6 for n in ids})))
    r = _frame(fem3).eigen(1)

    with pytest.raises(ValueError, match="EigenResult.to_native: .*different models"):
        r.to_native(tmp_path / "m.h5", model_h5=other)
    assert not (tmp_path / "m.h5").exists()


def test_eigen_model_h5_names_the_roles_archive(
    fem3: Any, bind: Any, tmp_path: Path,
) -> None:
    """``model_h5`` is where the roles come from: the bridge's own archive
    gives the file ``model_h5=None`` gives."""
    ids = _ids(fem3)
    bind(_FakeOps([16.0], _vectors(ids, 1, {int(n): 6 for n in ids})))
    ops = _frame(fem3)
    r = ops.eigen(1)
    archive = tmp_path / "model.h5"
    ops.h5(str(archive))

    a = _comps(r.to_native(tmp_path / "a.h5"))
    b = _comps(r.to_native(tmp_path / "b.h5", model_h5=archive))

    assert a == b
    assert set(_ROT) <= set(a)


# ---------------------------------------------------------------------------
# Without model_h5: the deck's own archive, then the solid-free fallback
# (T's ruling on #2225: a plain 3-D frame keeps the rotations main wrote)
# ---------------------------------------------------------------------------

def _frame_3d(g: Any, *, with_up: bool) -> tuple[Any, apeSees, list[int], list[int]]:
    """A session-built 3-D portal frame (ndf 6), optionally beside a
    LadrunoUP box (ndf 4): ``(fem, ops, frame_nodes, up_nodes)``."""
    G = g.model.geometry
    a, b = G.add_point(3.0, 0.0, 0.0), G.add_point(3.0, 0.0, 2.0)
    c, d = G.add_point(5.0, 0.0, 2.0), G.add_point(5.0, 0.0, 0.0)
    lines = [G.add_line(a, b), G.add_line(b, c), G.add_line(c, d)]
    if with_up:
        box = G.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    g.model.sync()
    g.physical.add(1, lines, name="Frame")
    if with_up:
        g.physical.add(3, [box], name="Soil")
        g.mesh.structured.set_transfinite_box("Soil", n=2)
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(3 if with_up else 1)
    g.mesh.partitioning.renumber(base=1)
    fem = g.mesh.queries.get_fem_data()
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    ops.element.elasticBeamColumn(
        pg="Frame", transf=ops.geomTransf.Linear(vecxz=(0.0, 1.0, 0.0)),
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4)
    up: list[int] = []
    if with_up:
        ops.element.LadrunoUP(
            pg="Soil",
            material=ops.nDMaterial.ElasticIsotropic(E=1e4, nu=0.3, rho=2.0),
            Kf=2.2e6, poro=0.4, rhoF=1.0, perm=(1e-4, 1e-4, 1e-4),
        )
        up = sorted(int(n) for n in fem.nodes.select(pg="Soil").ids)
    frame = sorted(int(n) for n in fem.nodes.select(pg="Frame").ids)
    return fem, ops, frame, up


def _rows(fem: Any, frame: list[int], up: list[int]) -> np.ndarray:
    """One mode: frame rows ``[1..6] + node``, u-p rows ``[.., .., .., 777,
    0, 0]`` (ux, uy, uz, p padded to ndf 6)."""
    ids = _ids(fem)
    out = np.zeros((1, ids.size, 6))
    for i, n in enumerate(ids.tolist()):
        if n in up:
            out[0, i] = [0.1, 0.2, 0.3, 777.0, 0.0, 0.0]
        else:
            out[0, i] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            out[0, i] += 0.01 * n
    return out


def _conventional(fem: Any) -> Path:
    from apeGmsh._core import default_artifact_dir

    return default_artifact_dir() / f"{fem.model_name}.h5"


def _emit_deck(ops: apeSees, tmp_path: Path) -> None:
    ops.modal_deck(str(tmp_path / "deck.tcl"), band=(0.1, 100.0))


def test_found_matching_archive_gives_the_roles(g: Any, tmp_path: Path) -> None:
    """``modal_deck`` leaves the archive at the conventional path with this
    fem's hash: its roles are used. A frame beside a u-p box keeps the
    frame's rotations, the u-p nodes get 0.0 (never 777), no warning."""
    fem, ops, frame, up = _frame_3d(g, with_up=True)
    _emit_deck(ops, tmp_path)
    assert _conventional(fem).is_file()
    shapes = _rows(fem, frame, up)
    res = ParallelModalResult.from_job(_job(tmp_path, _ids(fem), shapes, 3))

    comps, caught = _native(res, tmp_path / "m.h5", fem)

    assert caught == []
    row = {int(n): i for i, n in enumerate(_ids(fem))}
    for n in frame:
        assert [comps[c][n] for c in _ROT] == shapes[0, row[n], 3:].tolist()
    for n in up:
        assert [comps[c][n] for c in _ROT] == [0.0, 0.0, 0.0]
    assert 777.0 not in {v for c in comps.values() for v in c.values()}


def test_stale_archive_is_ignored_with_a_warning(g: Any, tmp_path: Path) -> None:
    """The file at the conventional path carries another fem_hash: it is
    ignored with one StaleModelArchiveWarning naming it, and the call
    falls through to the rules without an archive (a pure frame: the
    solid-free fallback keeps the rotations)."""
    import h5py

    from apeGmsh.opensees.analysis.modal import StaleModelArchiveWarning

    fem, ops, frame, _ = _frame_3d(g, with_up=False)
    _emit_deck(ops, tmp_path)
    archive = _conventional(fem)
    with h5py.File(archive, "r+") as f:
        f["meta/lineage"].attrs["fem_hash"] = "0" * 16
    shapes = _rows(fem, frame, [])
    res = ParallelModalResult.from_job(_job(tmp_path, _ids(fem), shapes, 3))

    comps, caught = _native(res, tmp_path / "m.h5", fem)

    stale = _of(caught, StaleModelArchiveWarning)
    assert len(stale) == 1 and len(_of(caught, UserWarning)) == 1
    assert "fem_hash" in str(stale[0].message)
    assert archive.name in str(stale[0].message)
    assert Path(stale[0].filename) == Path(__file__)
    for i, n in enumerate(_ids(fem).tolist()):
        assert [comps[c][n] for c in _ROT] == shapes[0, i, 3:].tolist()


def test_stale_archive_is_not_read_for_roles(g: Any, tmp_path: Path) -> None:
    """A stale archive whose roles would keep the frame rotations of a
    solid model is still not used: the u-p + frame model with its
    fem_hash changed omits every rotation and warns twice (stale, then
    the roles it lacks)."""
    import h5py

    from apeGmsh.opensees.analysis.modal import StaleModelArchiveWarning

    fem, ops, frame, up = _frame_3d(g, with_up=True)
    _emit_deck(ops, tmp_path)
    with h5py.File(_conventional(fem), "r+") as f:
        f["meta/lineage"].attrs["fem_hash"] = "0" * 16
    shapes = _rows(fem, frame, up)
    res = ParallelModalResult.from_job(_job(tmp_path, _ids(fem), shapes, 3))

    comps, caught = _native(res, tmp_path / "m.h5", fem)

    assert len(_of(caught, StaleModelArchiveWarning)) == 1
    assert len(_of(caught, RotationRolesUnavailableWarning)) == 1
    assert not set(_ROT) & set(comps)


def test_another_bridges_archive_on_the_same_mesh_is_not_used(
    g: Any, tmp_path: Path,
) -> None:
    """The review's two-bridge reproducer (#2225, delta review at
    1e12778e). Bridge A is frame + LadrunoUP, bridge B is frame-only, on
    the same fem and model_name. B's modal_deck overwrites A's archive
    with a matching fem_hash. A's harvest through ``to_native(path, fem)``
    must not read B's roles (the u-p nodes would be element-less there,
    so 777 would land in rotation_x): the archive misses the solid nodes,
    so it is ignored with the warning, and solids are present, so every
    rotation is omitted with the second warning."""
    import h5py

    from apeGmsh.opensees.analysis.modal import StaleModelArchiveWarning

    fem, ops_a, frame, up = _frame_3d(g, with_up=True)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _emit_deck(ops_a, tmp_path / "a")
    ops_b = apeSees(fem)
    ops_b.model(ndm=3, ndf=6)
    ops_b.element.elasticBeamColumn(
        pg="Frame", transf=ops_b.geomTransf.Linear(vecxz=(0.0, 1.0, 0.0)),
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4)
    _emit_deck(ops_b, tmp_path / "b")
    archive = _conventional(fem)
    with h5py.File(archive, "r") as f:     # B's file, A's mesh hash
        assert str(f["meta/lineage"].attrs["fem_hash"]) == str(fem.snapshot_id)
    tokens = {r.type_token for r in OpenSeesModel.from_h5(str(archive)).elements()}
    assert tokens and not any("UP" in t for t in tokens)   # frame-only
    shapes = _rows(fem, frame, up)
    res = ParallelModalResult.from_job(_job(tmp_path, _ids(fem), shapes, 3))

    comps, caught = _native(res, tmp_path / "m.h5", fem)

    stale = _of(caught, StaleModelArchiveWarning)
    assert len(stale) == 1 and "solid" in str(stale[0].message)
    assert len(_of(caught, RotationRolesUnavailableWarning)) == 1
    assert not set(_ROT) & set(comps)
    assert 777.0 not in {v for c in comps.values() for v in c.values()}


@pytest.mark.parametrize("shell", [False, True])
def test_pure_frame_or_shell_without_archive_keeps_rotations(
    g: Any, tmp_path: Path, shell: bool,
) -> None:
    """No archive (the bridge never emitted) and no solid cell: every node
    is a frame or shell node, so DOFs 4-6 are rotations; silent."""
    G = g.model.geometry
    if shell:
        G.add_rectangle(0.0, 0.0, 0.0, 2.0, 1.0)
        g.model.sync()
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(2)
    else:
        G.add_line(G.add_point(0.0, 0.0, 0.0), G.add_point(0.0, 0.0, 3.0))
        g.model.sync()
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(1)
    g.mesh.partitioning.renumber(base=1)
    fem = g.mesh.queries.get_fem_data()
    assert not _conventional(fem).exists()
    ids = _ids(fem)
    shapes = _rows(fem, ids.tolist(), [])
    res = ParallelModalResult.from_job(_job(tmp_path, ids, shapes, 3))

    comps, caught = _native(res, tmp_path / "m.h5", fem)

    assert caught == []
    for i, n in enumerate(ids.tolist()):
        assert [comps[c][n] for c in _ROT] == shapes[0, i, 3:].tolist()


def test_solids_without_archive_omit_rotations_with_the_warning(
    g: Any, tmp_path: Path,
) -> None:
    """A frame beside a u-p box, no archive: solid cells are present and
    the fem carries no ndf, so every rotation is omitted, one warning
    names the fix, and 777 is never written."""
    fem, _ops, frame, up = _frame_3d(g, with_up=True)
    assert not _conventional(fem).exists()
    shapes = _rows(fem, frame, up)
    res = ParallelModalResult.from_job(_job(tmp_path, _ids(fem), shapes, 3))

    comps, caught = _native(res, tmp_path / "m.h5", fem)

    named = _of(caught, RotationRolesUnavailableWarning)
    assert len(named) == 1 and len(_of(caught, UserWarning)) == 1
    assert "model_h5=" in str(named[0].message)
    assert not set(_ROT) & set(comps)
    assert 777.0 not in {v for c in comps.values() for v in c.values()}


def test_2d_ndf_3_without_archive_warns(g: Any, tmp_path: Path) -> None:
    """2-D ndf 3, no archive, no solid cell (a frame column only): still
    ambiguous (``(ux, uy, rz)`` or ``(ux, uy, p)``), so no rotation_z and
    the warning; the solid-free fallback is a 3-D rule only."""
    G = g.model.geometry
    G.add_line(G.add_point(0.0, 0.0, 0.0), G.add_point(0.0, 2.0, 0.0))
    g.model.sync()
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(1)
    fem = g.mesh.queries.get_fem_data()
    ids = _ids(fem)
    shapes = np.array([[[1.0, 2.0, 3.0 + i] for i in range(ids.size)]])
    res = ParallelModalResult.from_job(_job(tmp_path, ids, shapes, 2))

    comps, caught = _native(res, tmp_path / "m.h5", fem)

    assert len(_of(caught, RotationRolesUnavailableWarning)) == 1
    assert set(comps) == {"displacement_x", "displacement_y"}


# ---------------------------------------------------------------------------
# A plain 3-D frame writes exactly what origin/main's writer wrote
# ---------------------------------------------------------------------------

def _main_to_native(
    eigenvalues: Any, nodes: Any, shapes: np.ndarray, ndm: int,
    path: Path, fem: Any,
) -> None:
    """``ParallelModalResult.to_native`` as on origin/main at ``a1dddeac``
    (and ``_write_mode_stages`` with it), frozen here as the oracle: the
    first ``min(3, ndm, ndf)`` columns as displacements and, at ndf >= 6,
    columns 4-6 as rotations, for every node."""
    import math

    from apeGmsh.results.writers._native import NativeWriter

    ndf = int(shapes.shape[2])
    n_trans, rotations = min(3, ndm, ndf), ndf >= 6
    axes = ("x", "y", "z")
    writer = NativeWriter(path)
    writer.open(fem=fem, source_type="parallel_modal",
                source_path="<modal_deck harvest>")
    try:
        for mode_idx, lam in enumerate(eigenvalues, start=1):
            lam_f = float(lam)
            omega = math.sqrt(lam_f) if lam_f > 0 else 0.0
            sid = writer.begin_stage(
                name=f"mode_{mode_idx}", kind="mode", time=np.array([0.0]),
                eigenvalue=lam_f,
                frequency_hz=float(omega / (2.0 * math.pi)),
                period_s=float(2.0 * math.pi / omega) if omega > 0 else 0.0,
                mode_index=mode_idx,
            )
            field = shapes[mode_idx - 1]
            comps = {f"displacement_{axes[a]}": field[:, a][None, :]
                     for a in range(n_trans)}
            if rotations:
                for a in range(3):
                    comps[f"rotation_{axes[a]}"] = field[:, 3 + a][None, :]
            writer.write_nodes(sid, "partition_0",
                               node_ids=np.asarray(nodes), components=comps)
            writer.end_stage()
    finally:
        writer.close()


def _stage_walk(path: Path) -> dict[str, Any]:
    """Every dataset and attr under ``/stages`` (stage ids by mode)."""
    out: dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        stages = f["stages"]
        for sid in stages:
            key = f"mode_{int(stages[sid].attrs['mode_index'])}"
            for a, v in stages[sid].attrs.items():
                out[f"{key}@{a}"] = np.asarray(v).tolist()

            def visit(name: str, obj: Any, key: str = key) -> None:
                if isinstance(obj, h5py.Dataset):
                    out[f"{key}/{name}"] = (
                        obj.dtype.str, np.asarray(obj[()]).tolist())
            stages[sid].visititems(visit)
    return out


@pytest.mark.parametrize("archive", [False, True])
def test_plain_3d_frame_matches_mains_writer(
    g: Any, tmp_path: Path, archive: bool,
) -> None:
    """A plain 3-D frame, ``to_native(path, fem)`` with no ``model_h5``:
    with the deck's archive found or with none, ``/stages`` is dataset
    for dataset and attr for attr what origin/main's writer wrote."""
    fem, ops, frame, _ = _frame_3d(g, with_up=False)
    if archive:
        _emit_deck(ops, tmp_path)
    assert _conventional(fem).is_file() is archive
    ids = _ids(fem)
    shapes = np.concatenate([_rows(fem, frame, []), -_rows(fem, frame, [])])
    res = ParallelModalResult.from_job(_job(tmp_path, ids, shapes, 3))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res.to_native(tmp_path / "ours.h5", fem)
    _main_to_native(res.eigenvalues, ids, shapes, 3,
                    tmp_path / "main.h5", fem)

    ours, main = _stage_walk(tmp_path / "ours.h5"), _stage_walk(
        tmp_path / "main.h5")
    assert sum("rotation_" in k for k in main) == 2 * 3
    assert ours.keys() == main.keys()
    assert {k for k in ours if ours[k] != main[k]} == set()

