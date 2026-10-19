"""#2178 (X slice 9a): the Python ``.ladruno`` writer for modes.

``EigenResult.to_ladruno`` and ``ParallelModalResult.to_ladruno`` write
one ``KIND="eigen"`` ``MODEL_STAGE`` holding every mode, in the fork
recorder's own ``modesOfVibration`` layout, stamped against the bridge's
model archive. The bridge runs against the fake OpenSees module of
``tests/test_eigen_to_native.py`` (no backend needed).

Oracles:

* the fork's own ``ladruno_format.validate()``, fetched at ``FORK_PIN``,
  returns ``[]`` for every file written here;
* ``Results.from_ladruno(path, model_h5=)`` gives the mode shapes,
  eigenvalues, frequencies and periods of ``to_native`` +
  ``from_native`` on the same solve (rtol 1e-12), rotations included
  (the native file writes them where the DOF roles allow, #2203), and the
  2-D rotation_z equals the fake's vectors exactly;
* ``INFO/PROVENANCE`` carries the archive's ``/meta/lineage`` hashes
  byte for byte, ``writer`` names this writer and ``unsupported`` names
  the element buckets it leaves out.

The fork-written file it must equal is
``tests/opensees/integration_ladruno/test_ladruno_writer_modes_live.py``.
"""
from __future__ import annotations

import json
import math
import urllib.error
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

import h5py
import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.analysis.modal import ParallelModalResult
from apeGmsh.opensees.emitter import live
from apeGmsh.opensees.opensees_model import OpenSeesModel
from apeGmsh.results import Results

from tests.opensees.integration_ladruno import _fork_spec
from tests.test_eigen_to_native import _build_fem, _FakeOps, _frame, _vectors

_RTOL = 1e-12
_STAGE = "MODEL_STAGE[1]"
_U, _R = "MODES_OF_VIBRATION(U)", "MODES_OF_VIBRATION(R)"


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


@pytest.fixture(scope="module")
def spec(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ModuleType]:
    """The fork's ``ladruno_format`` at ``FORK_PIN``."""
    try:
        where = _fork_spec.fetch(tmp_path_factory.mktemp("ladruno_spec"))
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        pytest.skip(f"cannot fetch the fork's ladruno_format at FORK_PIN: {e}")
    with _fork_spec.ladruno_format(where) as module:
        yield module


def _ids(fem: Any) -> np.ndarray:
    return np.asarray(fem.nodes.ids, dtype=np.int64)


def _by_index(modes: Any) -> list[Any]:
    return sorted(modes, key=lambda m: m.mode_index)


def _slab(mode: Any, comp: str) -> dict[int, float]:
    s = mode.nodes.get(component=comp)
    assert s.values.shape[0] == 1
    return dict(zip(s.node_ids.tolist(), s.values[0].tolist()))


def _provenance(path: Path) -> dict[str, str]:
    with h5py.File(path, "r") as f:
        return {k: str(v) for k, v in f["INFO/PROVENANCE"].attrs.items()}


def _lineage(path: Path) -> dict[str, str]:
    with h5py.File(path, "r") as f:
        return {k: str(v) for k, v in f["meta/lineage"].attrs.items()}


# ---------------------------------------------------------------------------
# EigenResult.to_ladruno
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ndm", [2, 3])
def test_three_modes_pass_validate_and_equal_the_native_route(
    ndm: int, fem2: Any, fem3: Any, bind: Any, spec: ModuleType,
    tmp_path: Path,
) -> None:
    fem = fem2 if ndm == 2 else fem3
    ndf = 3 if ndm == 2 else 6
    lams = [4.0, 37.5, 1234.5678]
    ids = _ids(fem)
    vecs = _vectors(ids, 3, {int(n): ndf for n in ids})
    bind(_FakeOps(lams, vecs))
    r = _frame(fem, ndm=ndm).eigen(3)

    lad = r.to_ladruno(tmp_path / "modes.ladruno")
    nat = r.to_native(tmp_path / "modes_native.h5")

    assert lad == tmp_path / "modes.ladruno"
    assert spec.validate(str(lad)) == []
    archive = lad.with_suffix(".h5")
    with Results.from_ladruno(lad, model_h5=archive) as rl, \
            Results.from_native(nat, model=OpenSeesModel.from_h5(nat)) as rn:
        got, want = _by_index(rl.modes), _by_index(rn.modes)
        assert [m.mode_index for m in got] == [1, 2, 3]
        assert [m.kind for m in rl.stages if m.kind != "mode"] == ["static"]
        for g, w in zip(got, want):
            for attr in ("eigenvalue", "frequency_hz", "period_s"):
                assert getattr(g, attr) == pytest.approx(
                    getattr(w, attr), rel=_RTOL)
            native = set(w.nodes.available_components())
            assert native <= set(g.nodes.available_components())
            rot = {"rotation_z"} if ndm == 2 else {
                "rotation_x", "rotation_y", "rotation_z"}
            assert rot <= native
            for comp in sorted(native):
                gs, ws = _slab(g, comp), _slab(w, comp)
                assert sorted(gs) == sorted(ws) == sorted(ids.tolist())
                for nid in gs:
                    assert gs[nid] == pytest.approx(ws[nid], rel=_RTOL, abs=0)
        if ndm == 2:
            # Both files carry rotation_z in 2-D (the DOF roles make
            # eigenvector entry 3 a rotation on a frame); pin it to the
            # fake's vectors, not only to the native file.
            for k, g in enumerate(got, start=1):
                rz = _slab(g, "rotation_z")
                assert rz == {int(n): vecs[(int(n), k)][2] for n in ids}


def test_the_layout_is_the_fork_recorders(
    fem2: Any, bind: Any, tmp_path: Path,
) -> None:
    lams = [4.0, 9.0]
    ids = _ids(fem2)
    vecs = _vectors(ids, 2, {int(n): 3 for n in ids})
    bind(_FakeOps(lams, vecs))
    lad = _frame(fem2, ndm=2).eigen(2).to_ladruno(tmp_path / "m.ladruno")

    with h5py.File(lad, "r") as f:
        info = f["INFO"].attrs
        assert info["GENERATOR"][0] == b"Ladruno"
        assert int(info["FORMAT_VERSION"][0]) == 1
        assert int(info["SPATIAL_DIM"][0]) == 2
        assert list(f) == ["INFO", _STAGE]
        stage = f[_STAGE]
        assert stage.attrs["KIND"][0] == b"eigen"
        assert list(stage["MODEL/ELEMENTS"]) == []
        np.testing.assert_array_equal(stage["MODEL/NODES/ID"][()], ids)
        np.testing.assert_array_equal(
            stage["MODEL/NODES/COORDINATES"][()],
            np.asarray(fem2.nodes.coords)[:, :2])
        on_nodes = stage["RESULTS/ON_NODES"]
        assert list(on_nodes) == [_U, _R]
        assert on_nodes[_U].attrs["COMPONENTS"][0] == b"Ux,Uy"
        assert on_nodes[_R].attrs["COMPONENTS"][0] == b"Rz"
        np.testing.assert_array_equal(
            on_nodes[_U]["ID"][()], ids.reshape(-1, 1))
        step = on_nodes[_U]["DATA/STEP_0"]
        assert list(step) == ["MODE_0", "MODE_1"]
        for k, lam in enumerate(lams):
            mode = step[f"MODE_{k}"]
            assert int(mode.attrs["MODE"][0]) == k
            assert float(mode.attrs["LAMBDA"][0]) == lam
            omega = math.sqrt(lam)
            assert float(mode.attrs["OMEGA"][0]) == omega
            assert float(mode.attrs["FREQUENCY"][0]) == omega / (2 * math.pi)
            assert float(mode.attrs["PERIOD"][0]) == pytest.approx(
                2 * math.pi / omega, rel=1e-15)
            np.testing.assert_array_equal(mode[()], np.array(
                [vecs[(int(n), k + 1)][:2] for n in ids]))


def test_the_stamp_pairs_with_the_written_archive(
    fem3: Any, bind: Any, tmp_path: Path,
) -> None:
    import apeGmsh

    ids = _ids(fem3)
    bind(_FakeOps([4.0], _vectors(ids, 1, {int(n): 6 for n in ids})))
    lad = _frame(fem3).eigen(1).to_ladruno(tmp_path / "m.ladruno")

    prov = _provenance(lad)
    lineage = _lineage(lad.with_suffix(".h5"))
    assert prov["model_hash"] == lineage["model_hash"]
    assert prov["fem_hash"] == lineage["fem_hash"]
    assert prov["writer"] == f"apeGmsh eigen->ladruno {apeGmsh.__version__}"
    assert json.loads(prov["unsupported"]) == ["elements"]
    assert "stage_names" not in prov   # the program has no stages
    assert not (tmp_path / "m.ladruno.partial").exists()


def test_mixed_ndf_zero_fills_and_a_pressure_slot_is_not_a_rotation(
    fem3: Any, bind: Any, spec: ModuleType, tmp_path: Path,
) -> None:
    """Envelope ndf 6 over vectors of length 3 (a solid), 4 (a u-p node)
    and 6: the recorder's rule, entries 4-6 only for a 6-entry vector."""
    ids = _ids(fem3)
    lengths = {int(n): 6 for n in ids}
    solid, up = int(ids[0]), int(ids[1])
    lengths[solid], lengths[up] = 3, 4
    vecs = _vectors(ids, 1, lengths)
    bind(_FakeOps([25.0], vecs))
    lad = _frame(fem3).eigen(1).to_ladruno(tmp_path / "m.ladruno")

    assert spec.validate(str(lad)) == []
    with h5py.File(lad, "r") as f:
        u = f[f"{_STAGE}/RESULTS/ON_NODES/{_U}/DATA/STEP_0/MODE_0"][()]
        r = f[f"{_STAGE}/RESULTS/ON_NODES/{_R}/DATA/STEP_0/MODE_0"][()]
    row = {int(n): i for i, n in enumerate(ids)}
    for nid in (solid, up):
        np.testing.assert_array_equal(r[row[nid]], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(u[row[nid]], vecs[(nid, 1)][:3])
    frame = int(ids[-1])
    np.testing.assert_array_equal(r[row[frame]], vecs[(frame, 1)][3:6])


@pytest.mark.parametrize("lam", [0.0, -3.5])
def test_nonpositive_eigenvalue_warns_once_as_to_native_does(
    fem3: Any, bind: Any, spec: ModuleType, tmp_path: Path, lam: float,
) -> None:
    ids = _ids(fem3)
    bind(_FakeOps([4.0, lam, 9.0],
                  _vectors(ids, 3, {int(n): 6 for n in ids})))
    r = _frame(fem3).eigen(3)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lad = r.to_ladruno(tmp_path / "m.ladruno")
    user = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user) == 1
    assert "Mode 2" in str(user[0].message)
    assert Path(user[0].filename) == Path(__file__)   # points at the caller

    assert spec.validate(str(lad)) == []
    with h5py.File(lad, "r") as f:
        mode = f[f"{_STAGE}/RESULTS/ON_NODES/{_U}/DATA/STEP_0/MODE_1"]
        assert float(mode.attrs["LAMBDA"][0]) == lam
        for a in ("OMEGA", "FREQUENCY", "PERIOD"):
            assert float(mode.attrs[a][0]) == 0.0
    # The reader warns for the mode too, and reads 0 Hz / 0 s.
    with pytest.warns(UserWarning, match="mode 2 has a non-positive"):
        with Results.from_ladruno(lad, model_h5=lad.with_suffix(".h5")) as rl:
            m2 = _by_index(rl.modes)[1]
            assert (m2.eigenvalue, m2.frequency_hz, m2.period_s) == (lam, 0.0, 0.0)


def test_default_path_is_a_temp_file_beside_its_archive(
    fem3: Any, bind: Any,
) -> None:
    ids = _ids(fem3)
    bind(_FakeOps([4.0, 9.0], _vectors(ids, 2, {int(n): 6 for n in ids})))
    lad = _frame(fem3).eigen(2).to_ladruno()

    assert lad.name == "modes.ladruno"
    with Results.from_ladruno(lad, model_h5=lad.with_suffix(".h5")) as rl:
        assert len(rl.modes) == 2


def test_a_second_export_to_the_same_path_reuses_the_archive(
    fem3: Any, bind: Any, tmp_path: Path,
) -> None:
    ids = _ids(fem3)
    bind(_FakeOps([4.0], _vectors(ids, 1, {int(n): 6 for n in ids})))
    r = _frame(fem3).eigen(1)
    lad = r.to_ladruno(tmp_path / "m.ladruno")
    before = (lad.with_suffix(".h5")).stat().st_mtime_ns

    assert r.to_ladruno(tmp_path / "m.ladruno") == lad
    assert (lad.with_suffix(".h5")).stat().st_mtime_ns == before


@pytest.mark.parametrize("kind", ["other_bridge", "no_lineage"])
def test_a_foreign_stem_h5_refuses_and_writes_nothing(
    fem2: Any, fem3: Any, bind: Any, tmp_path: Path, kind: str,
) -> None:
    ids = _ids(fem3)
    bind(_FakeOps([4.0], _vectors(ids, 1, {int(n): 6 for n in ids})))
    foreign = tmp_path / "m.h5"
    if kind == "other_bridge":
        # A full /meta/lineage of another model: only the hash comparison
        # tells it from this bridge's archive.
        _frame(fem2, ndm=2).h5(str(foreign))
        assert set(_lineage(foreign)) >= {"model_hash", "fem_hash"}
    else:
        with h5py.File(foreign, "w") as f:
            f.create_group("meta/lineage").attrs["model_hash"] = "x"
    body = foreign.read_bytes()

    with pytest.raises(FileExistsError, match="not this bridge's model archive"):
        _frame(fem3).eigen(1).to_ladruno(tmp_path / "m.ladruno")
    assert not (tmp_path / "m.ladruno").exists()
    assert foreign.read_bytes() == body


def test_an_h5_target_refuses_and_keeps_the_users_archive(
    fem3: Any, bind: Any, tmp_path: Path,
) -> None:
    """``ops.h5("model.h5"); ops.eigen(1).to_ladruno("model.h5")`` would
    replace the archive with the .ladruno (``<stem>.h5`` is the path)."""
    ids = _ids(fem3)
    bind(_FakeOps([4.0], _vectors(ids, 1, {int(n): 6 for n in ids})))
    ops = _frame(fem3)
    archive = tmp_path / "model.h5"
    ops.h5(str(archive))
    body = archive.read_bytes()

    with pytest.raises(ValueError, match="has an .h5 suffix"):
        ops.eigen(1).to_ladruno(archive)
    assert archive.read_bytes() == body
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model.h5"]


@pytest.mark.parametrize("target", ["same", "h5"])
def test_parallel_refuses_a_target_that_is_its_archive(
    fem3: Any, tmp_path: Path, target: str,
) -> None:
    archive = tmp_path / ("model.h5" if target == "h5" else "model.arch")
    _frame(fem3, ndm=3).h5(str(archive))
    body = archive.read_bytes()
    job, _ = _job(tmp_path, _ids(fem3), [16.0], 6, 3)

    with pytest.raises(ValueError, match="to_ladruno: .*(suffix|is the model archive)"):
        ParallelModalResult.from_job(job).to_ladruno(archive, model_h5=archive)
    assert archive.read_bytes() == body


def test_a_stale_domain_refuses_before_writing(
    fem3: Any, bind: Any, tmp_path: Path,
) -> None:
    ids = _ids(fem3)
    bind(_FakeOps([1.0, 4.0, 9.0], _vectors(ids, 3, {int(n): 6 for n in ids})))
    r = _frame(fem3).eigen(3)
    _frame(fem3).eigen(2)

    with pytest.raises(RuntimeError, match="to_ladruno: the live OpenSees"):
        r.to_ladruno(tmp_path / "m.ladruno")
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# ParallelModalResult.to_ladruno
# ---------------------------------------------------------------------------

def _job(
    tmp_path: Path, ids: np.ndarray, lams: list[float], ndf: int, ndm: int,
) -> tuple[Path, np.ndarray]:
    """A replicated-deck run dir; returns it and the ``(k, n, ndf)`` shapes."""
    job = tmp_path / "job"
    job.mkdir()
    (job / "eigenvalues.out").write_text(" ".join(repr(v) for v in lams))
    (job / "mode_shapes.json").write_text(json.dumps(
        {"nodes": ids.tolist(), "ndf": ndf, "ndm": ndm}))
    shapes = np.array([
        [[100.0 * k + row + 0.125 * (d + 1) for d in range(ndf)]
         for row in range(ids.size)]
        for k in range(1, len(lams) + 1)
    ])
    for k in range(len(lams)):
        (job / f"mode_shape_{k + 1}.out").write_text(
            " ".join(repr(float(v)) for v in shapes[k].ravel()))
    return job, shapes


@pytest.mark.parametrize("ndm", [2, 3])
def test_parallel_modes_pass_validate_and_equal_the_native_route(
    ndm: int, fem2: Any, fem3: Any, spec: ModuleType, tmp_path: Path,
) -> None:
    fem = fem2 if ndm == 2 else fem3
    ndf = 3 if ndm == 2 else 6
    ids = _ids(fem)
    archive = tmp_path / "model.h5"
    _frame(fem, ndm=ndm).h5(str(archive))
    job, shapes = _job(tmp_path, ids, [16.0, 81.0, 625.0], ndf, ndm)
    res = ParallelModalResult.from_job(job)

    lad = res.to_ladruno(tmp_path / "modes.ladruno", model_h5=archive)
    nat = tmp_path / "modes_native.h5"
    res.to_native(nat, OpenSeesModel.from_h5(str(archive)).fem,
                  model_h5=archive)

    assert spec.validate(str(lad)) == []
    prov = _provenance(lad)
    assert prov["model_hash"] == _lineage(archive)["model_hash"]
    assert prov["writer"].startswith("apeGmsh modal_deck->ladruno ")
    with Results.from_ladruno(lad, model_h5=archive) as rl, \
            Results.from_native(nat, model=OpenSeesModel.from_h5(str(archive))) as rn:
        got, want = _by_index(rl.modes), _by_index(rn.modes)
        assert len(got) == 3
        for k, (g, w) in enumerate(zip(got, want)):
            for attr in ("eigenvalue", "frequency_hz", "period_s"):
                assert getattr(g, attr) == pytest.approx(
                    getattr(w, attr), rel=_RTOL)
            rot_comps = {"rotation_z"} if ndm == 2 else {
                "rotation_x", "rotation_y", "rotation_z"}
            assert rot_comps <= set(w.nodes.available_components())
            for comp in sorted(w.nodes.available_components()):
                gs, ws = _slab(g, comp), _slab(w, comp)
                assert sorted(gs) == sorted(ws)
                for nid in gs:
                    assert gs[nid] == pytest.approx(ws[nid], rel=_RTOL, abs=0)
            rot = {"rotation_z": 2} if ndm == 2 else {
                "rotation_x": 3, "rotation_y": 4, "rotation_z": 5}
            for comp, col in rot.items():
                assert _slab(g, comp) == {
                    int(n): shapes[k, i, col] for i, n in enumerate(ids)}


def test_parallel_rotations_follow_each_nodes_own_ndf(
    fem3: Any, spec: ModuleType, tmp_path: Path,
) -> None:
    """A 3-D u-p node (ndf 4) under a deck padded to ndf 6: its column 4
    is a pressure, so Rx..Rz are 0.0 there; a frame node keeps them."""
    ids = _ids(fem3)
    archive = tmp_path / "model.h5"
    _frame(fem3, ndm=3).h5(str(archive))
    up = int(ids[1])
    with h5py.File(archive, "r+") as f:
        grp = f["opensees/nodes_ndf"]
        tags = grp["tags"][()]
        ndf = grp["ndf"][()]
        ndf[list(tags).index(up)] = 4
        grp["ndf"][...] = ndf
    job, shapes = _job(tmp_path, ids, [16.0], 6, 3)

    lad = ParallelModalResult.from_job(job).to_ladruno(
        tmp_path / "m.ladruno", model_h5=archive)

    assert spec.validate(str(lad)) == []
    with h5py.File(lad, "r") as f:
        r = f[f"{_STAGE}/RESULTS/ON_NODES/{_R}/DATA/STEP_0/MODE_0"][()]
    row = {int(n): i for i, n in enumerate(ids)}
    np.testing.assert_array_equal(r[row[up]], [0.0, 0.0, 0.0])
    frame = int(ids[-1])
    np.testing.assert_array_equal(r[row[frame]], shapes[0, row[frame], 3:6])


def test_parallel_stamp_warning_points_at_the_caller(
    fem3: Any, tmp_path: Path,
) -> None:
    """A staged archive: the stamp omits ``stage_names`` and warns, from
    the user's line."""
    from apeGmsh.results.readers._ladruno_provenance import (
        StageNamesOmittedWarning,
    )

    ops = _frame(fem3, ndm=3)
    with ops.stage(name="gravity") as s:
        s.analysis(
            test=ops.test.NormDispIncr(tol=1e-6, max_iter=10),
            algorithm=ops.algorithm.Newton(),
            integrator=ops.integrator.LoadControl(dlam=1.0),
            constraints=ops.constraints.Plain(),
            numberer=ops.numberer.RCM(),
            system=ops.system.UmfPack(),
            analysis=ops.analysis.Static(),
        )
        s.run(n_increments=1)
    archive = tmp_path / "model.h5"
    ops.h5(str(archive))
    job, _ = _job(tmp_path, _ids(fem3), [16.0], 6, 3)
    res = ParallelModalResult.from_job(job)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lad = res.to_ladruno(tmp_path / "m.ladruno", model_h5=archive)
    omitted = [w for w in caught
               if issubclass(w.category, StageNamesOmittedWarning)]
    assert len(omitted) == 1
    assert Path(omitted[0].filename) == Path(__file__)
    assert "stage_names" not in _provenance(lad)


@pytest.mark.parametrize("ndm,labels,mask", [
    (1, b"Rx,Ry,Rz", True), (2, b"Rz", False), (3, b"Rx,Ry,Rz", False),
])
def test_the_rotational_group_is_zeros_without_rotations(
    fem3: Any, spec: ModuleType, tmp_path: Path,
    ndm: int, labels: bytes, mask: bool,
) -> None:
    """The recorder writes MODES_OF_VIBRATION(R) for every model when both
    tokens are requested; without rotations it holds zeros (1-D has no
    rotation slot even where the mask allows one)."""
    from apeGmsh.results.readers._ladruno_writer import write_modes

    archive = tmp_path / "model.h5"
    _frame(fem3, ndm=3).h5(str(archive))
    ids = _ids(fem3)
    coords = np.asarray(fem3.nodes.coords, dtype=np.float64)
    field = 1.0 + np.arange(ids.size * 6, dtype=np.float64).reshape(ids.size, 6)
    lad = write_modes(
        tmp_path / "m.ladruno", model_h5=archive, node_ids=ids,
        coords=coords, ndm=ndm, eigenvalues=[4.0],
        fields=lambda k: field, rotations=np.full(ids.size, mask),
        source="test",
    )

    assert spec.validate(str(lad)) == []
    with h5py.File(lad, "r") as f:
        grp = f[f"{_STAGE}/RESULTS/ON_NODES/{_R}"]
        assert grp.attrs["COMPONENTS"][0] == labels
        assert "DIMENSION" not in grp.attrs   # the recorder skips ""
        block = grp["DATA/STEP_0/MODE_0"][()]
        assert block.shape == (ids.size, len(labels.split(b",")))
        assert not block.any()
        u = f[f"{_STAGE}/RESULTS/ON_NODES/{_U}/DATA/STEP_0/MODE_0"][()]
        np.testing.assert_array_equal(u, field[:, :ndm])


def test_parallel_refuses_an_archive_of_another_dimension(
    fem3: Any, tmp_path: Path,
) -> None:
    archive = tmp_path / "model.h5"
    _frame(fem3, ndm=3).h5(str(archive))
    job, _ = _job(tmp_path, _ids(fem3), [16.0], 3, 2)

    with pytest.raises(ValueError, match="different models"):
        ParallelModalResult.from_job(job).to_ladruno(
            tmp_path / "m.ladruno", model_h5=archive)
    assert not (tmp_path / "m.ladruno").exists()


def test_parallel_refuses_a_node_the_archive_does_not_have(
    fem3: Any, tmp_path: Path,
) -> None:
    archive = tmp_path / "model.h5"
    _frame(fem3, ndm=3).h5(str(archive))
    ids = np.append(_ids(fem3), 9999)
    job, _ = _job(tmp_path, ids, [16.0], 6, 3)

    with pytest.raises(ValueError, match=r"\[9999\] are not nodes"):
        ParallelModalResult.from_job(job).to_ladruno(
            tmp_path / "m.ladruno", model_h5=archive)
    assert not (tmp_path / "m.ladruno").exists()


# ---------------------------------------------------------------------------
# Rotation slots follow the model's DOF roles (ruling on #2203)
# ---------------------------------------------------------------------------

def _up_beside_frame_2d(g: Any) -> tuple[Any, apeSees, list[int], list[int], int]:
    """A 2-D LadrunoUP block, a 2-D frame column and one element-less node,
    all at ndf 3: ``(fem, ops, up, frame, free)``. ndf and ndm cannot tell
    the u-p node's (ux, uy, p) from the frame node's (ux, uy, rz)."""
    G = g.model.geometry
    block = G.add_rectangle(0.0, 0.0, 0.0, 1.0, 1.0)
    col = G.add_line(G.add_point(3.0, 0.0, 0.0), G.add_point(3.0, 2.0, 0.0))
    free = G.add_point(6.0, 6.0, 0.0)
    g.model.sync()
    g.physical.add(2, [block], name="Soil")
    g.physical.add(1, [col], name="Col")
    g.physical.add(0, [free], name="Free")
    g.mesh.structured.set_recombine("Soil", dim=2)
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(2)
    g.mesh.partitioning.renumber(base=1)
    fem = g.mesh.queries.get_fem_data()
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=3)
    ops.element.LadrunoUP(
        pg="Soil",
        material=ops.nDMaterial.ElasticIsotropic(E=1e4, nu=0.3, rho=2.0),
        Kf=2.2e6, poro=0.4, rhoF=1.0, perm=(1e-4, 1e-4),
    )
    ops.element.elasticBeamColumn(
        pg="Col", transf=ops.geomTransf.Linear(), A=0.01, E=200e9, Iz=1e-4)
    up = sorted(int(n) for n in fem.nodes.select(pg="Soil").ids)
    frame = sorted(int(n) for n in fem.nodes.select(pg="Col").ids)
    (free_id,) = [int(n) for n in fem.nodes.select(pg="Free").ids]
    return fem, ops, up, frame, free_id


class _ForkFake(_FakeOps):
    """The fake as a fork build that lists no element tags, so the live
    emitter trusts a fork-only element (LadrunoUP) it was asked to make."""

    def getEleTags(self) -> Any:
        raise RuntimeError("this fake records no elements")


def _rot_rows(lad: Path) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    with h5py.File(lad, "r") as f:
        on = f[f"{_STAGE}/RESULTS/ON_NODES"]
        ids = np.asarray(on[_R]["ID"][()]).ravel()
        u = on[_U]["DATA/STEP_0/MODE_0"][()]
        r = on[_R]["DATA/STEP_0/MODE_0"][()]
    return ({int(n): u[i] for i, n in enumerate(ids)},
            {int(n): r[i] for i, n in enumerate(ids)})


def test_parallel_2d_up_deck_never_writes_a_pressure_as_rotation_z(
    g: Any, spec: ModuleType, tmp_path: Path,
) -> None:
    """2-D u-p nodes (ndf 3): Rz is 0.0, silently (the roles say pore
    pressure). The frame column keeps its real rotation_z under an
    envelope ndf below 6. The element-less node in a model with a u-p
    class is ambiguous: 0.0, and exactly one named warning."""
    from apeGmsh.opensees.analysis.eigen import RotationSlotUnresolvedWarning

    fem, ops, up, frame, free = _up_beside_frame_2d(g)
    archive = tmp_path / "model.h5"
    ops.h5(str(archive))
    ids = _ids(fem)
    job, shapes = _job(tmp_path, ids, [16.0], 3, 2)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lad = ParallelModalResult.from_job(job).to_ladruno(
            tmp_path / "m.ladruno", model_h5=archive)
    named = [w for w in caught
             if issubclass(w.category, RotationSlotUnresolvedWarning)]
    assert len(named) == 1
    assert f"[{free}]" in str(named[0].message)
    assert Path(named[0].filename) == Path(__file__)
    assert len([w for w in caught if issubclass(w.category, UserWarning)]) == 1

    assert spec.validate(str(lad)) == []
    u, r = _rot_rows(lad)
    row = {int(n): i for i, n in enumerate(ids)}
    for n in up:
        assert r[n].tolist() == [0.0]                      # p, not rz
        assert u[n].tolist() == shapes[0, row[n], :2].tolist()
    for n in frame:
        assert r[n].tolist() == [shapes[0, row[n], 2]]     # the real rz
    assert r[free].tolist() == [0.0]


def test_eigen_2d_up_deck_never_writes_a_pressure_as_rotation_z(
    g: Any, bind: Any, tmp_path: Path,
) -> None:
    from apeGmsh.opensees.analysis.eigen import RotationSlotUnresolvedWarning

    fem, ops, up, frame, free = _up_beside_frame_2d(g)
    ids = _ids(fem)
    vecs = _vectors(ids, 1, {int(n): 3 for n in ids})
    bind(_ForkFake([16.0], vecs, build="0" * 40))

    with pytest.warns(RotationSlotUnresolvedWarning, match=rf"\[{free}\]"):
        lad = ops.eigen(1).to_ladruno(tmp_path / "m.ladruno")
    _, r = _rot_rows(lad)
    for n in up:
        assert r[n].tolist() == [0.0]
    for n in frame:
        assert r[n].tolist() == [vecs[(n, 1)][2]]
    assert r[free].tolist() == [0.0]


def test_parallel_3d_elementless_node_in_a_up_model_gets_no_rotations(
    g: Any, tmp_path: Path,
) -> None:
    """An element-less node at ndf 6 beside a 3-D u-p block (ndf 4): its
    rotation slots could be tied to a pore pressure, so they are 0.0 with
    the one warning, and the u-p nodes' pressure (column 4) is no Rx."""
    from apeGmsh.opensees.analysis.eigen import RotationSlotUnresolvedWarning

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
    archive = tmp_path / "model.h5"
    ops.h5(str(archive))
    (free,) = [int(n) for n in fem.nodes.select(pg="Free").ids]
    ids = _ids(fem)
    job, shapes = _job(tmp_path, ids, [16.0], 6, 3)

    with pytest.warns(RotationSlotUnresolvedWarning, match=rf"\[{free}\]"):
        lad = ParallelModalResult.from_job(job).to_ladruno(
            tmp_path / "m.ladruno", model_h5=archive)
    u, r = _rot_rows(lad)
    row = {int(n): i for i, n in enumerate(ids)}
    for n in ids.tolist():
        assert r[n].tolist() == [0.0, 0.0, 0.0]
        assert u[n].tolist() == shapes[0, row[n], :3].tolist()
