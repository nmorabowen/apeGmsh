"""Deck provenance stamps at the bridge's deck sites (F2-d phase 2, #1511).

Phase 1 (``test_deck_provenance_stamp.py``) pinned the emitter header.
This file pins what the bridge hands it: every deck path stamps the
target it is built for, through ``emitter.tcl.deck_backend``.  Oracle,
a closed form per row of the contract:

* a pinned ``OpenSeesTarget(mode=...)`` always stamps that kind; the
  build rides along only when the live resolver already answered the
  same kind;
* ``mode="auto"`` (or no target) stamps the resolver's verdict when it
  has answered, and nothing when it has not; the deck path never probes.

Fork-free: the resolver's cache is replaced by a fake verdict.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from apeGmsh import __version__
from apeGmsh.opensees import OpenSeesTarget, apeSees
from apeGmsh.opensees._target import BackendInfo
from apeGmsh.opensees.emitter import live
from apeGmsh.opensees.emitter.tcl import DECK_BANNER, deck_backend
from apeGmsh.opensees.opensees_model import OpenSeesModel

from tests.opensees.golden import builder
from tests.opensees.h5._opensees_model_fixtures import build_simple_frame_h5

_SHA = "a240b9183c0ffee0123456789abcdef012345678"
_FORK = BackendInfo(kind="fork", build=_SHA, version="3.8.0", source="x")
_STOCK = BackendInfo(kind="stock", build=None, version="3.7.1", source="y")

_FORK_LINE = f"# apeGmsh {__version__}; backend fork; build {_SHA}"
_FORK_NO_BUILD = f"# apeGmsh {__version__}; backend fork"
_STOCK_LINE = f"# apeGmsh {__version__}; backend stock"


class _FakeModule:
    """Stands in for the bound openseespy module in the resolver cache."""


@pytest.fixture
def unanswered(monkeypatch: pytest.MonkeyPatch) -> None:
    """The live resolver has not answered in this process."""
    monkeypatch.setattr(live, "_OPS_CACHE", None)
    monkeypatch.setattr(live, "_BACKEND_INFO", None)


def _answer(monkeypatch: pytest.MonkeyPatch, info: BackendInfo) -> None:
    mod = _FakeModule()
    monkeypatch.setattr(live, "_OPS_CACHE", mod)
    monkeypatch.setattr(live, "_BACKEND_INFO", (mod, info))


@pytest.fixture
def answered_fork(monkeypatch: pytest.MonkeyPatch) -> None:
    _answer(monkeypatch, _FORK)


def _model(
    target: OpenSeesTarget | None, *, mode: str = "flat",
    fixture: str = "two_column_frame",
) -> apeSees:
    ops = builder.build_model(fixture, mode, "tcl")
    # build_model pins mode="stock" for the corpus; re-bind the target
    # under test (the constructor's ``opensees=`` slot).
    ops._opensees = target
    return ops


def _header(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()[:2]


def _stamp(path: Path) -> str | None:
    banner, second = _header(path)
    assert banner == DECK_BANNER
    return second if second.startswith("# apeGmsh ") else None


# --------------------------------------------------------------------------
# deck_backend: the closed-form table
# --------------------------------------------------------------------------
@pytest.mark.usefixtures("unanswered")
def test_deck_backend_unanswered() -> None:
    assert deck_backend(None) is None
    assert deck_backend(OpenSeesTarget()) is None
    fork = deck_backend(OpenSeesTarget(mode="fork"))
    assert fork is not None and (fork.kind, fork.build) == ("fork", None)
    stock = deck_backend(OpenSeesTarget(mode="stock"))
    assert stock is not None and (stock.kind, stock.build) == ("stock", None)


def test_deck_backend_answered(monkeypatch: pytest.MonkeyPatch) -> None:
    _answer(monkeypatch, _FORK)
    assert deck_backend(None) is _FORK
    assert deck_backend(OpenSeesTarget(mode="fork")) is _FORK
    stock = deck_backend(OpenSeesTarget(mode="stock"))
    assert stock is not None and (stock.kind, stock.build) == ("stock", None)
    _answer(monkeypatch, _STOCK)
    fork = deck_backend(OpenSeesTarget(require_fork=True))
    assert fork is not None and (fork.kind, fork.build) == ("fork", None)


def test_deck_backend_ignores_a_stale_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A verdict computed for a module that is no longer the bound one.
    monkeypatch.setattr(live, "_OPS_CACHE", _FakeModule())
    monkeypatch.setattr(live, "_BACKEND_INFO", (_FakeModule(), _FORK))
    assert deck_backend(None) is None


def test_deck_backend_never_probes(
    monkeypatch: pytest.MonkeyPatch, unanswered: None,
) -> None:
    def _boom() -> object:
        raise AssertionError("the deck path resolved the backend")

    monkeypatch.setattr(live, "_resolve_ops", _boom)
    assert deck_backend(None) is None
    assert deck_backend(OpenSeesTarget(mode="fork")) is not None


# --------------------------------------------------------------------------
# The four apeSees construction sites
# --------------------------------------------------------------------------
@pytest.mark.usefixtures("answered_fork")
@pytest.mark.parametrize("stream", [False, True])
def test_tcl_stamps_answered_fork(tmp_path: Path, stream: bool) -> None:
    deck = tmp_path / "model.tcl"
    _model(None).tcl(str(deck), flat=True, stream=stream)
    assert _stamp(deck) == _FORK_LINE


@pytest.mark.usefixtures("answered_fork")
def test_tcl_per_rank_driver_stamped(tmp_path: Path) -> None:
    deck = tmp_path / "model.tcl"
    _model(None, mode="per_rank").tcl(str(deck), per_rank=True)
    assert _stamp(deck) == _FORK_LINE
    for frag in (tmp_path / "ranks").glob("*.tcl"):
        assert "; backend " not in frag.read_text(encoding="utf-8")


@pytest.mark.usefixtures("answered_fork")
def test_py_stamps_answered_fork(tmp_path: Path) -> None:
    deck = tmp_path / "model.py"
    _model(None).py(str(deck))
    assert _stamp(deck) == _FORK_LINE


@pytest.mark.usefixtures("unanswered")
def test_pinned_fork_unanswered_has_no_build(tmp_path: Path) -> None:
    ops = _model(OpenSeesTarget(mode="fork"))
    ops.tcl(str(tmp_path / "a.tcl"), flat=True)
    ops.py(str(tmp_path / "a.py"))
    assert _stamp(tmp_path / "a.tcl") == _FORK_NO_BUILD
    assert _stamp(tmp_path / "a.py") == _FORK_NO_BUILD


@pytest.mark.usefixtures("answered_fork")
def test_pinned_stock_wins_over_answered_fork(tmp_path: Path) -> None:
    ops = _model(OpenSeesTarget(mode="stock"))
    ops.tcl(str(tmp_path / "a.tcl"), flat=True)
    ops.py(str(tmp_path / "a.py"))
    assert _stamp(tmp_path / "a.tcl") == _STOCK_LINE
    assert _stamp(tmp_path / "a.py") == _STOCK_LINE


@pytest.mark.usefixtures("unanswered")
def test_auto_unanswered_is_unstamped(tmp_path: Path) -> None:
    ops = _model(OpenSeesTarget())
    ops.tcl(str(tmp_path / "a.tcl"), flat=True)
    ops.py(str(tmp_path / "a.py"))
    assert _stamp(tmp_path / "a.tcl") is None
    assert _stamp(tmp_path / "a.py") is None


@pytest.mark.usefixtures("answered_fork")
def test_modal_deck_feast_stamped(tmp_path: Path) -> None:
    deck = tmp_path / "modal.tcl"
    _model(None).modal_deck(str(deck), band=(0.0, 10.0))
    assert _stamp(deck) == _FORK_LINE


@pytest.mark.usefixtures("answered_fork")
def test_modal_deck_arpack_stamped(tmp_path: Path) -> None:
    deck = tmp_path / "modal.tcl"
    _model(None, mode="partitioned").modal_deck(
        str(deck), solver="arpack", num_modes=1,
    )
    assert _stamp(deck) == _FORK_LINE


# --------------------------------------------------------------------------
# The archive-first path: OpenSeesModel.from_h5(...).build('tcl'/'py')
# --------------------------------------------------------------------------
@pytest.fixture
def archive(tmp_path: Path) -> Path:
    h5, _fem = build_simple_frame_h5(tmp_path)
    return h5


@pytest.mark.parametrize("kind", ["tcl", "py"])
def test_from_h5_build_stamps_answered_fork(
    monkeypatch: pytest.MonkeyPatch, archive: Path, kind: str,
) -> None:
    _answer(monkeypatch, _FORK)
    text = OpenSeesModel.from_h5(str(archive)).build(kind)
    assert text is not None
    assert text.splitlines()[:2] == [DECK_BANNER, _FORK_LINE]


@pytest.mark.parametrize("kind", ["tcl", "py"])
def test_from_h5_build_unanswered_is_unstamped(
    monkeypatch: pytest.MonkeyPatch, archive: Path, kind: str,
) -> None:
    monkeypatch.setattr(live, "_OPS_CACHE", None)
    monkeypatch.setattr(live, "_BACKEND_INFO", None)
    text = OpenSeesModel.from_h5(str(archive)).build(kind)
    assert text is not None
    assert "# apeGmsh " not in text
