"""``EigenResult.to_native`` / ``EigenResult.results`` (#1972, brief #1961 r2).

The modes route for pip ``openseespy`` once ``DomainCapture`` is deleted.
Every case here runs the real bridge (``apeSees.eigen`` and
``LiveOpsEmitter``) against a fake OpenSees module, so it needs no
backend: emit calls are no-ops, ``eigen`` and ``nodeEigenvector`` answer
from a table the test sets, and ``ladrunoBuild`` exists only on the fake
fork. The fem is a real one, meshed with gmsh in this module, never the
``_MockFem`` of the capture suite that the deletion removes.

Oracles (brief §6): the vectors written equal the fake's table (atol 0);
mixed ndf (vectors of length 3, 4 and 6 under an envelope of 6) never
raises and zero-fills; a dataset-by-dataset h5 walk against
``DomainCapture.capture_modes`` on the same fake; one ``UserWarning`` for
λ < 0 and none otherwise; staleness through ``domain_epoch()``.
The live solve is ``tests/opensees/live/test_eigen_to_native_live.py``.
"""
from __future__ import annotations

import math
import tempfile
import warnings
from pathlib import Path
from typing import Any

import gmsh
import h5py
import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees.analysis.eigen import EigenResult
from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter import live
from apeGmsh.opensees.opensees_model import OpenSeesModel
from apeGmsh.results import Results

_SHA = "0123456789abcdef0123456789abcdef01234567"
_N_NODES = 4
_L = 3.0


# ---------------------------------------------------------------------------
# The fem and the fake module
# ---------------------------------------------------------------------------

def _build_fem(ndm: int) -> Any:
    with apeGmsh(model_name=f"eigen_to_native_{ndm}d", verbose=False) as g:
        geo = g.model.geometry
        base = geo.add_point(0.0, 0.0, 0.0)
        tip = geo.add_point(_L, 0.0, 0.0)
        line = geo.add_line(base, tip)
        gmsh.model.mesh.setTransfiniteCurve(line, _N_NODES)
        g.physical.add(1, [line], name="Beam")
        g.labels.add(0, [base], "base")
        g.labels.add(0, [tip], "tip")
        g.mesh.generation.generate(dim=1)
        return g.mesh.queries.get_fem_data(dim=1)


@pytest.fixture(scope="module")
def fem3() -> Any:
    return _build_fem(3)


@pytest.fixture(scope="module")
def fem2() -> Any:
    return _build_fem(2)


class _FakeOps:
    """An OpenSees-shaped module. Model-building calls are no-ops;
    ``eigen`` returns ``eigenvalues``; ``nodeEigenvector(tag, k)``
    returns ``vectors[(tag, k)]`` and logs the call."""

    def __init__(
        self,
        eigenvalues: list[float],
        vectors: dict[tuple[int, int], list[float]],
        *,
        build: str | None = None,
    ) -> None:
        self.eigenvalues = list(eigenvalues)
        self.vectors = vectors
        self.vector_calls: list[tuple[int, int]] = []
        if build is not None:
            self.ladrunoBuild = lambda: build

    def eigen(self, *args: Any) -> list[float]:
        return list(self.eigenvalues)

    def nodeEigenvector(self, tag: int, mode: int, *rest: Any) -> list[float]:
        assert not rest, "one call per node, never one per dof"
        self.vector_calls.append((int(tag), int(mode)))
        return list(self.vectors[(int(tag), int(mode))])

    def modalProperties(self, *args: Any) -> dict[str, list[float]]:
        return {}

    def __getattr__(self, name: str) -> Any:
        # Fork-only probes answer "absent", as on stock openseespy.
        if name.startswith("_") or name.lower().startswith("ladruno"):
            raise AttributeError(name)
        return lambda *a, **k: 0


def _vectors(
    node_ids: np.ndarray, n_modes: int, lengths: dict[int, int],
) -> dict[tuple[int, int], list[float]]:
    """Distinct, exactly representable entries per (node, mode, dof)."""
    return {
        (int(nid), k): [
            float(k) * 100.0 + float(row) + 0.125 * (d + 1)
            for d in range(lengths[int(nid)])
        ]
        for row, nid in enumerate(node_ids)
        for k in range(1, n_modes + 1)
    }


@pytest.fixture
def bind(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Bind ``fake`` as the module the resolver answers."""
    def _bind(fake: _FakeOps) -> _FakeOps:
        monkeypatch.setattr(live, "_get_ops", lambda: fake)
        monkeypatch.setattr(live, "_BACKEND_INFO", None)
        return fake
    return _bind


def _frame(fem: Any, *, ndm: int = 3) -> apeSees:
    ops = apeSees(fem, default_orientation=None)
    if ndm == 3:
        ops.model(ndm=3, ndf=6)
        t = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
        ops.element.elasticBeamColumn(
            pg="Beam", transf=t, A=1e-2, E=2e11, Iz=8e-6, Iy=8e-6,
            G=8e10, J=1e-5,
        )
        ops.fix(label="base", dofs=(1, 1, 1, 1, 1, 1))
        ops.mass(label="tip", values=(10.0, 10.0, 10.0, 1.0, 1.0, 1.0))
    else:
        ops.model(ndm=2, ndf=3)
        t = ops.geomTransf.Linear()
        ops.element.elasticBeamColumn(
            pg="Beam", transf=t, A=1e-2, E=2e11, Iz=8e-6,
        )
        ops.fix(label="base", dofs=(1, 1, 1))
        ops.mass(label="tip", values=(10.0, 10.0, 1.0))
    return ops


def _node_ids(fem: Any) -> np.ndarray:
    return np.asarray(fem.nodes.ids, dtype=np.int64)


def _mode_indices(path: Path) -> list[int]:
    with Results.from_native(path, model=OpenSeesModel.from_h5(path)) as r:
        return sorted(m.mode_index for m in r.modes)


def _stage_rows(path: Path) -> list[dict[str, Any]]:
    with h5py.File(path, "r") as f:
        return [
            {k: f["stages"][s].attrs[k] for k in f["stages"][s].attrs}
            for s in sorted(f["stages"])
        ]


def _component(path: Path, stage: str, comp: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        grp = f["stages"][stage]["partitions"]["partition_0"]["nodes"]
        return np.asarray(grp[comp][()])


def _components(path: Path, stage: str) -> set[str]:
    with h5py.File(path, "r") as f:
        grp = f["stages"][stage]["partitions"]["partition_0"]["nodes"]
        return {k for k in grp if k != "_ids"}


# ---------------------------------------------------------------------------
# Invariants 1-3: stages, attrs, vectors
# ---------------------------------------------------------------------------

def test_one_mode_stage_per_eigenvalue_with_exact_attrs(
    fem3: Any, bind: Any, tmp_path: Path,
) -> None:
    lams = [4.0, 37.5, 1234.5678]
    ids = _node_ids(fem3)
    bind(_FakeOps(lams, _vectors(ids, 3, {int(n): 6 for n in ids})))
    r = _frame(fem3).eigen(3)

    out = r.to_native(tmp_path / "modes.h5")

    assert out == tmp_path / "modes.h5"
    rows = _stage_rows(out)
    assert [row["kind"] for row in rows] == ["mode"] * 3
    assert [int(row["mode_index"]) for row in rows] == [1, 2, 3]
    for row, lam in zip(rows, lams):
        assert float(row["eigenvalue"]) == lam          # bit for bit
        omega = math.sqrt(lam)
        assert row["frequency_hz"] == pytest.approx(
            omega / (2.0 * math.pi), rel=1e-15)
        assert row["period_s"] == pytest.approx(
            2.0 * math.pi / omega, rel=1e-15)
    with h5py.File(out, "r") as f:
        assert f.attrs["source_type"] == "eigen"


def test_vectors_are_the_domain_vectors_read_once_per_node(
    fem3: Any, bind: Any, tmp_path: Path,
) -> None:
    ids = _node_ids(fem3)
    vecs = _vectors(ids, 2, {int(n): 6 for n in ids})
    fake = bind(_FakeOps([4.0, 9.0], vecs))
    r = _frame(fem3).eigen(2)

    out = r.to_native(tmp_path / "modes.h5")

    assert sorted(fake.vector_calls) == sorted(
        (int(n), k) for n in ids for k in (1, 2))
    with h5py.File(out, "r") as f:
        stages = sorted(f["stages"])
    for k, stage in enumerate(stages, start=1):
        assert _components(out, stage) == {
            "displacement_x", "displacement_y", "displacement_z",
            "rotation_x", "rotation_y", "rotation_z",
        }
        for d, comp in enumerate(
            ("displacement_x", "displacement_y", "displacement_z",
             "rotation_x", "rotation_y", "rotation_z"),
        ):
            want = np.array([[vecs[(int(n), k)][d] for n in ids]])
            np.testing.assert_array_equal(_component(out, stage, comp), want)


def test_2d_frame_writes_in_plane_translations_only(
    fem2: Any, bind: Any, tmp_path: Path,
) -> None:
    ids = _node_ids(fem2)
    vecs = _vectors(ids, 1, {int(n): 3 for n in ids})
    bind(_FakeOps([16.0], vecs))
    r = _frame(fem2, ndm=2).eigen(1)

    out = r.to_native(tmp_path / "modes.h5")

    with h5py.File(out, "r") as f:
        (name,) = list(f["stages"])
    assert _components(out, name) == {"displacement_x", "displacement_y"}
    np.testing.assert_array_equal(
        _component(out, name, "displacement_y"),
        np.array([[vecs[(int(n), 1)][1] for n in ids]]))


def test_mixed_ndf_never_raises_and_zero_fills(
    fem3: Any, bind: Any, tmp_path: Path,
) -> None:
    """Envelope ndf 6 over nodes of ndf 3 (a solid), 4 (a u-p node) and
    6 (a frame): no raise; the short nodes write 0.0 rotations, and the
    ndf-4 node's 4th entry (a pressure) never lands in rotation_x."""
    ids = _node_ids(fem3)
    lengths = {int(n): 6 for n in ids}
    solid, up = int(ids[0]), int(ids[1])
    lengths[solid] = 3
    lengths[up] = 4
    vecs = _vectors(ids, 1, lengths)
    bind(_FakeOps([25.0], vecs))
    r = _frame(fem3).eigen(1)

    out = r.to_native(tmp_path / "modes.h5")

    with h5py.File(out, "r") as f:
        (stage,) = list(f["stages"])
    cols = {int(n): i for i, n in enumerate(ids)}
    for d, comp in enumerate(("rotation_x", "rotation_y", "rotation_z")):
        got = _component(out, stage, comp)[0]
        assert got[cols[solid]] == 0.0
        assert got[cols[up]] == 0.0
        frame = int(ids[-1])
        assert got[cols[frame]] == vecs[(frame, 1)][3 + d]
    for d, comp in enumerate(
        ("displacement_x", "displacement_y", "displacement_z"),
    ):
        got = _component(out, stage, comp)[0]
        assert got[cols[solid]] == vecs[(solid, 1)][d]
        assert got[cols[up]] == vecs[(up, 1)][d]


def test_a_vector_shorter_than_ndm_zero_fills_translations(
    fem3: Any, bind: Any, tmp_path: Path,
) -> None:
    """ndm 3, one node answering a 2-entry vector: displacement_z is 0.0
    there and the other entries are the vector's."""
    ids = _node_ids(fem3)
    lengths = {int(n): 6 for n in ids}
    short = int(ids[1])
    lengths[short] = 2
    vecs = _vectors(ids, 1, lengths)
    bind(_FakeOps([25.0], vecs))
    r = _frame(fem3).eigen(1)

    out = r.to_native(tmp_path / "modes.h5")

    with h5py.File(out, "r") as f:
        (stage,) = list(f["stages"])
    col = [int(n) for n in ids].index(short)
    assert _component(out, stage, "displacement_x")[0][col] == vecs[(short, 1)][0]
    assert _component(out, stage, "displacement_y")[0][col] == vecs[(short, 1)][1]
    assert _component(out, stage, "displacement_z")[0][col] == 0.0
    for comp in ("rotation_x", "rotation_y", "rotation_z"):
        assert _component(out, stage, comp)[0][col] == 0.0


# ---------------------------------------------------------------------------
# Invariant 4: parity with DomainCapture.capture_modes
# ---------------------------------------------------------------------------

def _walk(path: Path, root: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        def visit(name: str, obj: Any) -> None:
            key = f"{root}/{name}"
            for a in obj.attrs:
                out[f"{key}@{a}"] = np.asarray(obj.attrs[a]).tolist()
            if isinstance(obj, h5py.Dataset):
                out[key] = (obj.dtype.str, np.asarray(obj[()]).tolist())
        grp = f[root]
        for a in grp.attrs:
            out[f"{root}@{a}"] = np.asarray(grp.attrs[a]).tolist()
        grp.visititems(visit)
    return out


def test_stages_match_capture_modes_dataset_by_dataset(
    fem3: Any, bind: Any, tmp_path: Path,
) -> None:
    from apeGmsh.results.capture.spec import DomainCaptureSpec

    ids = _node_ids(fem3)
    lengths = {int(n): 6 for n in ids}
    lengths[int(ids[0])] = 3
    fake = bind(_FakeOps([4.0, 37.5, 1234.5678],
                         _vectors(ids, 3, lengths)))
    ops = _frame(fem3)
    r = ops.eigen(3)
    # Same file name in two directories: the sidecar's model_name
    # ("<stem>.model") then matches too.
    ours = r.to_native(tmp_path / "a" / "modes.h5")

    theirs = tmp_path / "b" / "modes.h5"
    theirs.parent.mkdir()
    with ops.domain_capture(
        DomainCaptureSpec(opensees=ops), path=theirs, ops=fake,
    ) as cap:
        cap.capture_modes(3)

    a, b = _walk(ours, "stages"), _walk(theirs, "stages")
    assert sum("@" not in k for k in a) == 3 * (6 + 2)   # data + _ids + time
    assert a.keys() == b.keys()
    assert {k for k in a if a[k] != b[k]} == set()
    for zone, volatile in (("opensees", set()),
                           ("model", {"model/meta@created_iso"})):
        a, b = _walk(ours, zone), _walk(theirs, zone)
        assert a.keys() == b.keys()
        assert {k for k in a if repr(a[k]) != repr(b[k])} == volatile
    with h5py.File(ours, "r") as f, h5py.File(theirs, "r") as g:
        assert f.attrs["source_type"] == "eigen"
        assert g.attrs["source_type"] == "domain_capture"


# ---------------------------------------------------------------------------
# Warn-as-contract (invariant 2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lam", [-2.5, 0.0])
def test_nonpositive_eigenvalue_warns_once_naming_the_mode(
    fem3: Any, bind: Any, tmp_path: Path, lam: float,
) -> None:
    ids = _node_ids(fem3)
    bind(_FakeOps([4.0, lam, 9.0],
                  _vectors(ids, 3, {int(n): 6 for n in ids})))
    r = _frame(fem3).eigen(3)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = r.to_native(tmp_path / "modes.h5")

    user = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user) == 1
    assert "Mode 2" in str(user[0].message)
    assert Path(user[0].filename) == Path(__file__)   # points at the caller
    rows = _stage_rows(out)
    assert float(rows[1]["eigenvalue"]) == lam
    assert float(rows[1]["frequency_hz"]) == 0.0
    assert float(rows[1]["period_s"]) == 0.0


def test_nonpositive_warning_from_results_points_at_the_caller(
    fem3: Any, bind: Any,
) -> None:
    ids = _node_ids(fem3)
    bind(_FakeOps([-1.0], _vectors(ids, 1, {int(n): 6 for n in ids})))
    r = _frame(fem3).eigen(1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with r.results() as res:
            assert len(res.modes) == 1

    user = [w for w in caught
            if issubclass(w.category, UserWarning) and "Mode 1" in str(w.message)]
    assert len(user) == 1
    assert Path(user[0].filename) == Path(__file__)


# ---------------------------------------------------------------------------
# Invariant 5: staleness through the domain epoch
# ---------------------------------------------------------------------------

def _fake_for(fem: Any, n: int) -> _FakeOps:
    ids = _node_ids(fem)
    return _FakeOps([float(k * k) for k in range(1, n + 1)],
                    _vectors(ids, n, {int(m): 6 for m in ids}))


def test_modal_properties_after_eigen_makes_it_stale(
    fem3: Any, bind: Any, tmp_path: Path,
) -> None:
    bind(_fake_for(fem3, 5))
    ops = _frame(fem3)
    r = ops.eigen(3)
    ops.modal_properties(5)

    with pytest.raises(RuntimeError, match="rebuilt since this eigen"):
        r.to_native(tmp_path / "modes.h5")
    assert not (tmp_path / "modes.h5").exists()


def test_a_second_bridges_eigen_makes_it_stale(
    fem3: Any, bind: Any, tmp_path: Path,
) -> None:
    bind(_fake_for(fem3, 3))
    r = _frame(fem3).eigen(3)
    _frame(fem3).eigen(2)

    with pytest.raises(RuntimeError, match="rebuilt since this eigen"):
        r.to_native(tmp_path / "modes.h5")


def test_a_deck_only_tcl_emit_does_not_invalidate(
    fem3: Any, bind: Any, tmp_path: Path,
) -> None:
    bind(_fake_for(fem3, 3))
    ops = _frame(fem3)
    r = ops.eigen(3)
    ops.tcl(str(tmp_path / "deck.tcl"))

    out = r.to_native(tmp_path / "modes.h5")
    assert len(_stage_rows(out)) == 3


def test_a_hand_built_result_is_refused(tmp_path: Path) -> None:
    r = EigenResult(eigenvalues=np.array([1.0]), _live=None)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="not returned by"):
        r.to_native(tmp_path / "modes.h5")


# ---------------------------------------------------------------------------
# Invariant 6 / 6a: provenance, empty FEAST band
# ---------------------------------------------------------------------------

def test_stock_writes_stock_and_no_build(
    fem3: Any, bind: Any, tmp_path: Path,
) -> None:
    bind(_fake_for(fem3, 1))
    out = _frame(fem3).eigen(1).to_native(tmp_path / "modes.h5")
    with h5py.File(out, "r") as f:
        assert f.attrs["opensees_backend"] == "stock"
        assert "opensees_build" not in f.attrs


def test_fork_writes_fork_and_the_solving_modules_build(
    fem3: Any, bind: Any, tmp_path: Path,
) -> None:
    ids = _node_ids(fem3)
    bind(_FakeOps([4.0], _vectors(ids, 1, {int(n): 6 for n in ids}),
                  build=_SHA))
    out = _frame(fem3).eigen(1).to_native(tmp_path / "modes.h5")
    with h5py.File(out, "r") as f:
        assert f.attrs["opensees_backend"] == "fork"
        assert f.attrs["opensees_build"] == _SHA


def test_empty_feast_band_raises_before_any_filesystem_write(
    fem3: Any, bind: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bind(_FakeOps([], {}, build=_SHA))
    ops = _frame(fem3)
    r = ops.eigen_feast(1.0, 2.0)
    assert len(r.eigenvalues) == 0

    def _no_write(*a: Any, **k: Any) -> Any:
        raise AssertionError("a filesystem write before the refusal")

    from apeGmsh.results.writers._native import NativeWriter

    monkeypatch.setattr(tempfile, "mkdtemp", _no_write)
    monkeypatch.setattr(tempfile, "TemporaryDirectory", _no_write)
    monkeypatch.setattr(NativeWriter, "open", _no_write)
    monkeypatch.setattr(apeSees, "h5", _no_write)
    for path in (None, tmp_path / "modes.h5"):
        with pytest.raises(ValueError, match="no modes"):
            r.to_native(path)
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Invariant 7: the temp default opens alone; results()
# ---------------------------------------------------------------------------

def test_default_path_is_a_temp_file_that_opens_alone(
    fem3: Any, bind: Any,
) -> None:
    bind(_fake_for(fem3, 2))
    p = _frame(fem3).eigen(2).to_native()

    assert Path(tempfile.gettempdir()).resolve() in p.resolve().parents
    assert _mode_indices(p) == [1, 2]


def test_results_opens_the_modes(fem3: Any, bind: Any) -> None:
    bind(_fake_for(fem3, 2))
    with _frame(fem3).eigen(2).results() as r:
        modes = sorted(r.modes, key=lambda m: m.mode_index)
        assert [m.eigenvalue for m in modes] == [1.0, 4.0]
