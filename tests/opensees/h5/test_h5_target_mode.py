"""``/opensees@target_mode`` (opensees 2.36.0, #2375; #2278 ruling A).

The archive keeps a pinned ``OpenSeesTarget.mode`` and nothing else of the
target. Each test names its oracle:

* the writer stores exactly the bridge's mode, ``"fork"`` / ``"stock"``, and
  nothing for ``"auto"`` (closed form);
* the reader reads an absent attribute as ``"auto"`` (the pre-bump schema
  corpus file is the reader-floor witness) and refuses anything that is not
  one string of the three modes, the empty string included, naming the
  attribute;
* a pinned archive's replay deck equals the bridge's deck byte for byte,
  stamp line included (the bridge deck is the oracle);
* a ``"fork"`` archive needs the fork on replay: ``build('live')`` refuses a
  stock openseespy, and the stage-close route checks apply the #2444
  fork-reset refusal the bridge applies under ``mode="fork"``;
* ``model_hash`` is the same pinned or on auto (ruling A: same physics, same
  hash).
"""
from __future__ import annotations

import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import pytest

from apeGmsh.opensees import OpenSeesModel, OpenSeesTarget, apeSees
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees._target import BackendInfo
from apeGmsh.opensees.emitter import live
from apeGmsh.opensees.emitter.h5 import H5Emitter
from apeGmsh.opensees.emitter.h5_reader import (
    TARGET_MODES, MalformedH5Error, open as h5_open,
)

from tests.fixtures.schema import OPENSEES_PRIOR_MINOR, OPENSEES_TARGET_MODE_FROM
from tests.opensees.h5.test_h5_solve_stamp import _frame_bridge

_PINNED = ("fork", "stock")
_CORPUS = Path(__file__).parents[2] / "fixtures" / "schema_corpus"


def _pinned_frame(mode: "str | None") -> apeSees:
    ops = _frame_bridge(analysis=True)
    ops._opensees = (None if mode is None
                     else OpenSeesTarget(mode=mode))  # type: ignore[arg-type]
    return ops


def _archive(ops: apeSees, path: Path) -> Path:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops.h5(str(path))
    return path


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [None, "auto", *_PINNED])
def test_the_archive_stores_exactly_the_pinned_mode(
    mode: "str | None", tmp_path: Path,
) -> None:
    path = _archive(_pinned_frame(mode), tmp_path / "m.h5")
    with h5py.File(path, "r") as f:
        attrs = f["opensees"].attrs
        if mode in _PINNED:
            assert attrs["target_mode"] == mode
            assert h5py.check_string_dtype(
                attrs.get_id("target_mode").dtype) is not None
        else:
            assert "target_mode" not in attrs
        # Only the mode: no machine path reaches the archive.
        assert not any(k in attrs for k in ("binary", "python"))


def test_the_writer_refuses_an_unknown_mode() -> None:
    with pytest.raises(ValueError, match="target_mode='Fork'"):
        H5Emitter(target_mode="Fork")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="target_mode=''"):
        H5Emitter(target_mode="")  # type: ignore[arg-type]


def test_binary_and_python_are_never_archived(tmp_path: Path) -> None:
    ops = _frame_bridge(analysis=True)
    ops._opensees = OpenSeesTarget(
        binary=str(tmp_path / "OpenSees.exe"),
        python=str(tmp_path / "python.exe"), mode="fork")
    path = _archive(ops, tmp_path / "m.h5")
    with h5py.File(path, "r") as f:
        def visit(name: str, obj: Any) -> None:
            for value in obj.attrs.values():
                assert str(tmp_path) not in str(value), (name, value)
        f.visititems(visit)
        assert f["opensees"].attrs["target_mode"] == "fork"


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def test_a_pre_bump_archive_reads_as_auto() -> None:
    """The reader-floor witness: the schema corpus file of the minor below
    the bump carries no attribute and reads as ``"auto"``."""
    major, minor, _patch = OPENSEES_PRIOR_MINOR.split(".")
    path = _CORPUS / f"opensees_{major}.{minor}.h5"
    assert OPENSEES_PRIOR_MINOR < OPENSEES_TARGET_MODE_FROM
    with h5py.File(path, "r") as f:
        assert "target_mode" not in f["opensees"].attrs
    assert h5_open(path).target_mode() == "auto"
    assert OpenSeesModel.from_h5(str(path)).target_mode == "auto"


def test_the_corpus_reads_every_mode() -> None:
    assert TARGET_MODES == {"auto", "fork", "stock"}


@pytest.mark.parametrize("bad", [
    "", "Fork", "auto ", "mpi", 1, np.int8(1), np.array(["fork", "fork"],
                                                         dtype=object),
], ids=["empty", "case", "space", "token", "int", "int8", "array"])
def test_a_malformed_mode_raises_naming_the_attribute(
    bad: Any, tmp_path: Path,
) -> None:
    path = _archive(_pinned_frame(None), tmp_path / "m.h5")
    with h5py.File(path, "a") as f:
        if isinstance(bad, np.ndarray):
            f["opensees"].attrs.create(
                "target_mode", bad, dtype=h5py.string_dtype())
        else:
            f["opensees"].attrs["target_mode"] = bad
    with pytest.raises(MalformedH5Error, match="/opensees@target_mode"):
        h5_open(path).target_mode()
    with pytest.raises(MalformedH5Error, match="/opensees@target_mode"):
        OpenSeesModel.from_h5(str(path))


# ---------------------------------------------------------------------------
# Replay: the deck stamps as the bridge does
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["tcl", "py"])
@pytest.mark.parametrize("mode", [None, *_PINNED])
def test_a_pinned_replay_deck_is_the_bridge_deck(
    kind: str, mode: "str | None", tmp_path: Path,
) -> None:
    ops = _pinned_frame(mode)
    deck = tmp_path / f"bridge.{kind}"
    if kind == "tcl":
        ops.tcl(str(deck), flat=True)
    else:
        ops.py(str(deck))
    path = _archive(ops, tmp_path / "m.h5")
    replay = OpenSeesModel.from_h5(str(path)).build(kind, flat=kind == "tcl")
    bridge = deck.read_text(encoding="utf-8")
    assert replay == bridge
    stamped = [ln for ln in bridge.splitlines() if "; backend " in ln]
    assert stamped == ([] if mode is None else [stamped[0]])
    if mode is not None:
        assert stamped[0].endswith(f"; backend {mode}")


@pytest.mark.parametrize("mode", [None, *_PINNED])
def test_to_h5_echoes_the_mode(mode: "str | None", tmp_path: Path) -> None:
    src = _archive(_pinned_frame(mode), tmp_path / "src.h5")
    dst = tmp_path / "dst.h5"
    OpenSeesModel.from_h5(str(src)).to_h5(dst)
    assert h5_open(dst).target_mode() == (mode or "auto")


# ---------------------------------------------------------------------------
# Replay: a fork archive needs the fork
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [None, *_PINNED])
def test_only_a_fork_archive_requires_the_fork(
    mode: "str | None", tmp_path: Path,
) -> None:
    path = _archive(_pinned_frame(mode), tmp_path / "m.h5")
    model = OpenSeesModel.from_h5(str(path))
    assert model.replay_requires() == (("fork",) if mode == "fork" else ())
    # The stamp itself stays the verbs' requirement: the pin is its own
    # attribute, not folded into @requires.
    assert model.solve_stamp is not None
    assert "fork" not in model.solve_stamp.requires


@pytest.fixture
def stock_backend(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """``build('live')`` judged against a stock in-process openseespy; the
    refusal runs before any emitter exists, so nothing is imported."""
    stock = BackendInfo(kind="stock", build=None, version="3.8.0",
                        source="test stand-in")
    monkeypatch.setattr(live, "get_backend_info", lambda *a, **k: stock)

    def no_live(*a: Any, **k: Any) -> None:
        raise AssertionError("the live emitter must not be built")

    monkeypatch.setattr(live, "LiveOpsEmitter", no_live)
    yield


@pytest.mark.usefixtures("stock_backend")
def test_build_live_refuses_a_fork_archive_on_stock(tmp_path: Path) -> None:
    path = _archive(_pinned_frame("fork"), tmp_path / "m.h5")
    with pytest.raises(BridgeError) as exc:
        OpenSeesModel.from_h5(str(path)).build("live")
    msg = str(exc.value)
    # The verbs need no fork: the message names the pin, not @requires.
    assert "the archive is pinned to 'fork' (/opensees@target_mode" in msg, msg
    assert "verbs require" not in msg and "@requires" not in msg, msg


def test_the_live_refusal_names_each_source() -> None:
    """``refuse_live_requires`` attributes each token to where it came
    from: the verbs (``@requires``) or the pin (``@target_mode``)."""
    from apeGmsh.opensees.emitter.caps import SolveStamp
    from apeGmsh.opensees.opensees_model import refuse_live_requires

    stock = BackendInfo(kind="stock", build=None, version="3.8.0", source="x")
    with pytest.raises(BridgeError) as exc:
        refuse_live_requires(SolveStamp(True, requires=("fork",)), stock)
    msg = str(exc.value)
    assert "the archive's verbs require 'fork' (/opensees@requires)" in msg
    assert "target_mode" not in msg
    with pytest.raises(BridgeError) as exc:
        refuse_live_requires(SolveStamp(True, requires=("fork", "mpco")),
                             stock, pinned=("fork",))
    msg = str(exc.value)
    assert ("the archive's verbs require 'mpco' (/opensees@requires) and "
            "the archive is pinned to 'fork' (/opensees@target_mode") in msg


@pytest.mark.usefixtures("stock_backend")
@pytest.mark.parametrize("mode", [None, "stock"])
def test_build_live_does_not_ask_an_unpinned_or_stock_archive_for_the_fork(
    mode: "str | None", tmp_path: Path,
) -> None:
    path = _archive(_pinned_frame(mode), tmp_path / "m.h5")
    with pytest.raises(AssertionError, match="live emitter must not be built"):
        OpenSeesModel.from_h5(str(path)).build("live")


def _pre_fix_fork_reset(
    mode: "str | None", path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """The #2444 model (a second ManzariDafalias activated after a plastic
    flip, on a serial system) archived as the bridge did before the stage
    close refused it: under ``mode="fork"`` the current bridge refuses it,
    and its archive must not replay."""
    from tests.opensees.h5 import test_h5_stage_route_replay as routes

    build: Callable[[], apeSees] = lambda: routes._flip_then_activate(
        own_material=True, mode=mode)
    return routes._write(build, path, monkeypatch, pre_fix=True)


def test_the_bridge_refuses_the_fork_reset_only_under_a_fork_pin(
    tmp_path: Path,
) -> None:
    from tests.opensees.h5 import test_h5_stage_route_replay as routes

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(BridgeError, match="#2444"):
            routes._flip_then_activate(own_material=True, mode="fork").tcl(
                str(tmp_path / "fork.tcl"))
        routes._flip_then_activate(own_material=True).tcl(
            str(tmp_path / "auto.tcl"))


@pytest.mark.parametrize("target", ["tcl", "py"])
def test_a_fork_pinned_pre_fix_archive_keeps_the_fork_reset_refusal(
    target: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    path = _pre_fix_fork_reset("fork", tmp_path / "fork.h5", monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = OpenSeesModel.from_h5(str(path))
        assert model.target_mode == "fork"
        with pytest.raises(BridgeError, match="#2444") as exc:
            model.build(target)
    assert "ManzariDafalias::mElastFlag to 0" in str(exc.value)


def test_the_unpinned_twin_replays(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    path = _pre_fix_fork_reset(None, tmp_path / "auto.h5", monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        deck = OpenSeesModel.from_h5(str(path)).build("tcl")
    assert deck is not None and "updateMaterialStage" in deck


# ---------------------------------------------------------------------------
# model_hash: the pin is not hashed
# ---------------------------------------------------------------------------


def _model_hash(path: Path) -> str:
    with h5py.File(path, "r") as f:
        return str(f["meta"]["lineage"].attrs["model_hash"])


def test_model_hash_is_the_same_pinned_or_on_auto(tmp_path: Path) -> None:
    hashes = {
        mode: _model_hash(_archive(_pinned_frame(mode),
                                   tmp_path / f"{mode}.h5"))
        for mode in (None, *_PINNED)
    }
    assert len(set(hashes.values())) == 1, hashes
    # And the stored hash is the one a reader recomputes (no lineage drift).
    for mode in _PINNED:
        assert OpenSeesModel.from_h5(
            str(tmp_path / f"{mode}.h5")).lineage.warnings == ()
