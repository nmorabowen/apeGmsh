"""``FootfallResult.to_results`` writes ``<stem>.user.h5`` (ADR 0131 D6, slice W3).

Footfall is a state-free user result (ADR 0130 D7), written through
``write_user_file``: one ``"footfall"`` entry on the nodes, three components,
``NaN`` off the response set. The oracle is the reader: the file reads back
through ``UserFile`` and its ``DATA`` equals, bit for bit, the arrays scattered
here by hand from the in-memory result. No native ``results.h5`` is written.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import pytest

from apeGmsh.opensees._target import BackendInfo
from apeGmsh.opensees.analysis.footfall_result import FootfallResult
from apeGmsh.opensees.emitter import live
from apeGmsh.results.user import UserFile, UserResult, UserWriteRefused, write_user_file
from tests.conftest import _MockFem

_FEM_IDS = [10, 20, 30, 40, 50]
_RESPONSE = (40, 10)
_A_P = np.array([0.0123456789, 1e-300])
_RATIO = np.array([1.1772, 0.25])
_F_DOM = np.array([16.326, 4.0])


@pytest.fixture(autouse=True)
def _stock_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    # Never bind a real OpenSees module from this unit file.
    monkeypatch.setattr(live, "get_backend_info", lambda: BackendInfo(
        kind="stock", build=None, version=None, source="test"))


def _result(nodes: tuple[int, ...] = _RESPONSE) -> FootfallResult:
    res = object.__new__(FootfallResult)
    for name, value in {
        "nodes": nodes, "a_p": _A_P[: len(nodes)], "ratio": _RATIO[: len(nodes)],
        "f_dom": _F_DOM[: len(nodes)],
    }.items():
        object.__setattr__(res, name, value)
    return res


def _archive(path: Path, fem: Any, model_hash: Optional[str] = None) -> Path:
    """The model archive of ``fem``: what the pairing reads, ``/meta/lineage``."""
    with h5py.File(path, "w") as f:
        lineage = f.create_group("meta/lineage")
        lineage.attrs["fem_hash"] = fem.snapshot_id
        if model_hash is not None:
            lineage.attrs["model_hash"] = model_hash
    return path


def _expected() -> np.ndarray:
    """``(1, N, 3)`` scattered by hand: rows of 40 and 10, NaN elsewhere."""
    out = np.full((1, len(_FEM_IDS), 3), np.nan)
    for j, tag in enumerate(_RESPONSE):
        row = _FEM_IDS.index(tag)
        out[0, row] = (_A_P[j], _RATIO[j], _F_DOM[j])
    return out


def test_writes_a_state_free_user_result_that_reads_back_bit_for_bit(tmp_path: Path) -> None:
    fem = _MockFem(_FEM_IDS)
    model = _archive(tmp_path / "bay.h5", fem, model_hash="ab" * 16)
    target = tmp_path / "bay.user.h5"

    out = _result().to_results(fem, target, model_h5=model)

    assert out == target
    user = UserFile(out)
    assert (user.fem_hash, user.model_hash) == (fem.snapshot_id, "ab" * 16)
    [entry] = user.results().entries
    assert (entry.name, entry.target, entry.basis, entry.per_state) == (
        "footfall", "nodes", "model", False)
    back = user.read_result("footfall")
    assert back.states is None
    assert back.components == ("footfall_ap", "footfall_ratio", "footfall_fdom")
    assert back.units == ("g", "1", "Hz")
    assert np.asarray(back.ids).ravel().tolist() == _FEM_IDS
    data = np.asarray(back.data)
    assert data.shape == (1, len(_FEM_IDS), 3)
    assert data.tobytes() == _expected().tobytes()
    # Off the response set every component is NaN.
    assert np.isnan(data[0, [1, 2, 4]]).all()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bay.h5", "bay.user.h5"]


def test_no_native_results_file_is_written(tmp_path: Path) -> None:
    fem = _MockFem(_FEM_IDS)
    out = _result().to_results(
        fem, tmp_path / "bay.user.h5", model_h5=_archive(tmp_path / "bay.h5", fem))
    with h5py.File(out, "r") as f:
        assert "stages" not in f and "model" not in f
        assert set(f) == {"meta", "results"}


def test_rewrite_replaces_the_entry_and_keeps_the_others(tmp_path: Path) -> None:
    fem = _MockFem(_FEM_IDS)
    model = _archive(tmp_path / "bay.h5", fem)
    target = tmp_path / "bay.user.h5"
    other = UserResult(
        name="D/C", target="nodes", basis="model", components=("dc",),
        data=np.ones((1, 1, 1)), ids=np.array([10], dtype="i4"))
    write_user_file(target, model, results=(other,))

    _result().to_results(fem, target, model_h5=model)
    _result(nodes=(40,)).to_results(fem, target, model_h5=model)

    user = UserFile(target)
    assert user.revision == 3
    assert sorted(e.name for e in user.results().entries) == ["D/C", "footfall"]
    data = np.asarray(user.read_result("footfall").data)
    assert np.isnan(data[0, 0]).all()  # node 10 left the response set
    assert data[0, 3].tobytes() == np.array([_A_P[0], _RATIO[0], _F_DOM[0]]).tobytes()


def test_model_h5_is_required(tmp_path: Path) -> None:
    fem = _MockFem(_FEM_IDS)
    with pytest.raises(TypeError, match="model_h5"):
        _result().to_results(fem, tmp_path / "bay.user.h5")  # type: ignore[call-arg]
    with pytest.raises(UserWriteRefused, match="model_h5= is required"):
        _result().to_results(fem, tmp_path / "bay.user.h5", model_h5=None)  # type: ignore[arg-type]
    assert list(tmp_path.iterdir()) == []


def test_a_fem_with_no_nodes_refuses(tmp_path: Path) -> None:
    fem = _MockFem([])
    model = _archive(tmp_path / "bay.h5", fem)
    with pytest.raises(UserWriteRefused, match="fem has no nodes"):
        _result().to_results(fem, tmp_path / "bay.user.h5", model_h5=model)
    assert not (tmp_path / "bay.user.h5").exists()


def test_a_response_node_outside_fem_refuses_by_name(tmp_path: Path) -> None:
    fem = _MockFem([10, 20, 30])
    model = _archive(tmp_path / "bay.h5", fem)
    with pytest.raises(UserWriteRefused, match=r"response nodes \[40\] are not in fem"):
        _result().to_results(fem, tmp_path / "bay.user.h5", model_h5=model)
    assert not (tmp_path / "bay.user.h5").exists()


def test_the_archive_of_another_fem_refuses(tmp_path: Path) -> None:
    fem = _MockFem(_FEM_IDS)
    model = _archive(tmp_path / "other.h5", _MockFem(_FEM_IDS, salt=1))
    with pytest.raises(UserWriteRefused, match="is not this fem's archive"):
        _result().to_results(fem, tmp_path / "bay.user.h5", model_h5=model)
    assert not (tmp_path / "bay.user.h5").exists()


def test_a_path_not_named_user_h5_refuses(tmp_path: Path) -> None:
    fem = _MockFem(_FEM_IDS)
    model = _archive(tmp_path / "bay.h5", fem)
    with pytest.raises(UserWriteRefused, match=r"<stem>\.user\.h5"):
        _result().to_results(fem, tmp_path / "footfall.h5", model_h5=model)
    assert not (tmp_path / "footfall.h5").exists()
