"""``FootfallResult.to_results`` writes ``<stem>.user.h5`` (ADR 0131 D6, slice W3).

Footfall is a state-free user result (ADR 0130 D7), written through
``write_user_file``: one ``"footfall"`` entry on the nodes, three components,
``NaN`` off the response set. The oracle is the reader: the file reads back
through ``UserFile`` and its ``DATA`` equals, bit for bit, the arrays scattered
here by hand from the in-memory result. No native ``results.h5`` is written.

#2801: the archive must also be the *model* the evaluation solved. The live
test at the bottom is the reproducer: one fem, a cracked and an uncracked
bridge, and the cracked evaluation refuses the uncracked archive by both
``model_hash`` values.
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
from apeGmsh.results.user import (
    UserFile,
    UserResult,
    UserWriteRefused,
    model_hashes,
    write_user_file,
)
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


# ---------------------------------------------------------------------------
# #2801: the archive's model_hash must be the model footfall_walking solved.
# Two archives of one fem (a cracked and an uncracked stiffness variant) share
# the fem_hash, so the fem check alone accepted either and stamped its hash.
# ---------------------------------------------------------------------------


@pytest.fixture
def _memo(monkeypatch: pytest.MonkeyPatch) -> dict[str, tuple[str, str]]:
    """A fresh, process-local lineage memo for one test."""
    from apeGmsh.opensees._internal import artifact_write

    fresh: dict[str, tuple[str, str]] = {}
    monkeypatch.setattr(artifact_write, "_LINEAGE_BY_DIGEST", fresh)
    return fresh


def _keyed(key: Optional[str]) -> FootfallResult:
    res = _result()
    object.__setattr__(res, "_lineage_key", key)
    return res


def test_the_memoised_model_hash_refuses_an_archive_of_another_model(
    tmp_path: Path, _memo: dict[str, tuple[str, str]],
) -> None:
    fem = _MockFem(_FEM_IDS)
    _memo["k"] = ("aa" * 16, fem.snapshot_id)
    other = _archive(tmp_path / "uncracked.h5", fem, model_hash="bb" * 16)
    target = tmp_path / "cracked.user.h5"
    with pytest.raises(UserWriteRefused) as info:
        _keyed("k").to_results(fem, target, model_h5=other)
    assert "bb" * 16 in str(info.value) and "aa" * 16 in str(info.value)
    assert not target.exists()

    out = _keyed("k").to_results(
        fem, target, model_h5=_archive(tmp_path / "cracked.h5", fem, "aa" * 16))
    assert UserFile(out).model_hash == "aa" * 16


@pytest.mark.parametrize("model_hash", [None, "bb" * 16])
@pytest.mark.parametrize("key", [None, "never-written"])
def test_no_solved_model_hash_or_none_in_the_archive_passes(
    tmp_path: Path, _memo: dict[str, tuple[str, str]],
    key: Optional[str], model_hash: Optional[str],
) -> None:
    """Refuse only when both sides carry a model_hash: a result with no
    lineage key (stub snapshot, built by hand) or one whose archive this
    process never wrote (a reload) passes, and so does a fem_hash-only
    archive."""
    fem = _MockFem(_FEM_IDS)
    _memo["k"] = ("aa" * 16, fem.snapshot_id)
    model = _archive(tmp_path / "bay.h5", fem, model_hash=model_hash)
    out = _keyed(key).to_results(fem, tmp_path / "bay.user.h5", model_h5=model)
    assert UserFile(out).model_hash == model_hash
    if key is None:
        # A fem_hash-only archive passes against a memoised lineage too.
        fem_only = _archive(tmp_path / "fem_only.h5", fem)
        _keyed("k").to_results(fem, tmp_path / "bay.user.h5", model_h5=fem_only)


def _bay(fem: Any, e_mod: float) -> Any:
    """The how-to's 6 m floor bay on ``fem``, at Young's modulus ``e_mod``."""
    from apeGmsh.opensees import apeSees

    ops = apeSees(fem)
    ops.model(ndm=2, ndf=3)
    transf = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    ops.element.elasticBeamColumn(
        pg="Beam", transf=transf, A=0.0083, E=e_mod, Iz=3.5e-4, mass=500.0,
    )
    ops.fix(pg="Left", dofs=(1, 1, 0))
    ops.fix(pg="Right", dofs=(0, 1, 0))
    return ops


def _bay_fem(g: Any) -> Any:
    length = 6.0
    p0 = g.model.geometry.add_point(0.0, 0.0, 0.0)
    pm = g.model.geometry.add_point(length / 2, 0.0, 0.0)
    p1 = g.model.geometry.add_point(length, 0.0, 0.0)
    l0 = g.model.geometry.add_line(p0, pm)
    l1 = g.model.geometry.add_line(pm, p1)
    g.model.sync()
    g.physical.add(1, [l0, l1], name="Beam")
    g.physical.add(0, [p0], name="Left")
    g.physical.add(0, [p1], name="Right")
    g.physical.add(0, [pm], name="Midspan")
    g.mesh.sizing.set_global_size(length / 12.0)
    g.mesh.generation.generate(1)
    return g.mesh.queries.get_fem_data(dim=1)


def _walk(ops: Any) -> FootfallResult:
    return ops.footfall_walking(
        num_modes=2, body_weight=747.0, g=9.81,
        response_nodes=ops.nodes.get(pg="Midspan"), dof=2, damp=0.03,
    )


@pytest.mark.live
@pytest.mark.parametrize("h5_first", [False, True], ids=["h5_after", "h5_before"])
def test_footfall_walking_refuses_the_archive_of_another_model_of_its_fem(
    g: Any, tmp_path: Path, _memo: dict[str, tuple[str, str]], h5_first: bool,
) -> None:
    """The reproducer of #2801, through the real driver and ``ops.h5``.

    One fem, two bridges: cracked (``0.35 E``) and uncracked. Their archives
    share the ``fem_hash`` and differ in ``model_hash``. The cracked
    evaluation pairs with the cracked archive, and with a ``fem_hash``-only
    archive of the fem, but refuses the uncracked one by both hashes and
    writes nothing. ``ops.h5`` runs after the driver (the documented route)
    or before it.
    """
    pytest.importorskip("openseespy.opensees")
    fem = _bay_fem(g)
    cracked_ops, uncracked_ops = _bay(fem, 0.35 * 200e9), _bay(fem, 200e9)
    cracked, uncracked = tmp_path / "cracked.h5", tmp_path / "uncracked.h5"
    if h5_first:
        cracked_ops.h5(str(cracked))
    result = _walk(cracked_ops)
    if not h5_first:
        cracked_ops.h5(str(cracked))
    uncracked_ops.h5(str(uncracked))
    fem_only = tmp_path / "fem_only.h5"
    fem.to_h5(str(fem_only))

    fem_c, hash_c = model_hashes(cracked)
    fem_u, hash_u = model_hashes(uncracked)
    assert fem_c == fem_u == fem.snapshot_id
    assert hash_c is not None and hash_u is not None and hash_c != hash_u
    assert model_hashes(fem_only) == (fem.snapshot_id, None)

    target = tmp_path / "cracked.user.h5"
    with pytest.raises(UserWriteRefused, match="another model of this fem") as info:
        result.to_results(fem, target, model_h5=uncracked)
    assert hash_u in str(info.value) and hash_c in str(info.value)
    assert not target.exists()

    assert UserFile(result.to_results(fem, target, model_h5=cracked)).model_hash == hash_c
    out = result.to_results(fem, tmp_path / "fem_only.user.h5", model_h5=fem_only)
    assert UserFile(out).model_hash is None
