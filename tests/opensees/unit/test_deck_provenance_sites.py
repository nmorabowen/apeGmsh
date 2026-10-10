"""Deck provenance stamps at the bridge's deck sites (F2-d phase 2, #1511).

Phase 1 (``test_deck_provenance_stamp.py``) pinned the emitter header.
This file pins what the bridge hands it: every deck path stamps the
target it is built for, through ``emitter.tcl.deck_backend``.  Oracle,
a closed form per row of the contract:

* a pinned ``OpenSeesTarget(mode="fork" | "stock")`` stamps that kind and
  no build;
* ``mode="auto"``, or no target, stamps nothing;
* neither depends on what the live resolver has answered earlier in the
  process: the same model emits the same deck before and after a fork
  verdict is cached.  (A first cut read the cached verdict under
  ``auto``; the suite caught a committed deck golden changing with test
  order.)

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

_FORK_LINE = f"# apeGmsh {__version__}; backend fork"
_STOCK_LINE = f"# apeGmsh {__version__}; backend stock"


class _FakeModule:
    """Stands in for the bound openseespy module in the resolver cache."""


@pytest.fixture(params=["unanswered", "answered_fork"])
def resolver(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch,
) -> str:
    """Run each case with and without a cached fork verdict in the process."""
    if request.param == "unanswered":
        monkeypatch.setattr(live, "_OPS_CACHE", None)
        monkeypatch.setattr(live, "_BACKEND_INFO", None)
    else:
        mod = _FakeModule()
        monkeypatch.setattr(live, "_OPS_CACHE", mod)
        monkeypatch.setattr(live, "_BACKEND_INFO", (mod, _FORK))
    return str(request.param)


def _model(
    target: OpenSeesTarget | None, *, mode: str = "flat",
    fixture: str = "two_column_frame",
) -> apeSees:
    ops = builder.build_model(fixture, mode, "tcl")
    # build_model pins mode="stock" for the corpus; re-bind the target
    # under test (the constructor's ``opensees=`` slot).
    ops._opensees = target
    return ops


def _stamp(path: Path) -> str | None:
    banner, second = path.read_text(encoding="utf-8").splitlines()[:2]
    assert banner == DECK_BANNER
    return second if second.startswith("# apeGmsh ") else None


# --------------------------------------------------------------------------
# deck_backend: the closed-form table
# --------------------------------------------------------------------------
@pytest.mark.usefixtures("resolver")
def test_deck_backend_table() -> None:
    assert deck_backend(None) is None
    assert deck_backend(OpenSeesTarget()) is None
    fork = deck_backend(OpenSeesTarget(mode="fork"))
    assert fork is not None and (fork.kind, fork.build) == ("fork", None)
    assert deck_backend(OpenSeesTarget(require_fork=True)) == fork
    stock = deck_backend(OpenSeesTarget(mode="stock"))
    assert stock is not None and (stock.kind, stock.build) == ("stock", None)


def test_deck_backend_never_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> object:
        raise AssertionError("the deck path resolved the backend")

    monkeypatch.setattr(live, "_OPS_CACHE", None)
    monkeypatch.setattr(live, "_BACKEND_INFO", None)
    monkeypatch.setattr(live, "_resolve_ops", _boom)
    for mode in ("auto", "fork", "stock"):
        deck_backend(OpenSeesTarget(mode=mode))  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# The four apeSees construction sites
# --------------------------------------------------------------------------
@pytest.mark.usefixtures("resolver")
@pytest.mark.parametrize("stream", [False, True])
def test_tcl_stamps_pinned_fork(tmp_path: Path, stream: bool) -> None:
    deck = tmp_path / "model.tcl"
    _model(OpenSeesTarget(mode="fork")).tcl(
        str(deck), flat=True, stream=stream,
    )
    assert _stamp(deck) == _FORK_LINE


@pytest.mark.usefixtures("resolver")
def test_tcl_per_rank_driver_stamped(tmp_path: Path) -> None:
    deck = tmp_path / "model.tcl"
    _model(OpenSeesTarget(mode="fork"), mode="per_rank").tcl(
        str(deck), per_rank=True,
    )
    assert _stamp(deck) == _FORK_LINE
    frags = list((tmp_path / "ranks").glob("*.tcl"))
    assert frags
    for frag in frags:
        assert "; backend " not in frag.read_text(encoding="utf-8")


@pytest.mark.usefixtures("resolver")
def test_py_stamps_pinned_target(tmp_path: Path) -> None:
    _model(OpenSeesTarget(mode="fork")).py(str(tmp_path / "f.py"))
    _model(OpenSeesTarget(mode="stock")).py(str(tmp_path / "s.py"))
    assert _stamp(tmp_path / "f.py") == _FORK_LINE
    assert _stamp(tmp_path / "s.py") == _STOCK_LINE


@pytest.mark.usefixtures("resolver")
@pytest.mark.parametrize("auto", ["none", "auto"])
def test_auto_is_unstamped(tmp_path: Path, auto: str) -> None:
    ops = _model(None if auto == "none" else OpenSeesTarget())
    ops.tcl(str(tmp_path / "a.tcl"), flat=True)
    ops.py(str(tmp_path / "a.py"))
    assert _stamp(tmp_path / "a.tcl") is None
    assert _stamp(tmp_path / "a.py") is None


@pytest.mark.usefixtures("resolver")
def test_modal_deck_feast_stamped(tmp_path: Path) -> None:
    deck = tmp_path / "modal.tcl"
    _model(OpenSeesTarget(mode="fork")).modal_deck(
        str(deck), band=(0.0, 10.0),
    )
    assert _stamp(deck) == _FORK_LINE


@pytest.mark.usefixtures("resolver")
def test_modal_deck_arpack_stamped(tmp_path: Path) -> None:
    deck = tmp_path / "modal.tcl"
    _model(OpenSeesTarget(mode="stock"), mode="partitioned").modal_deck(
        str(deck), solver="arpack", num_modes=1,
    )
    assert _stamp(deck) == _STOCK_LINE


# --------------------------------------------------------------------------
# The archive-first path: OpenSeesModel.from_h5(...).build('tcl'/'py')
# --------------------------------------------------------------------------
@pytest.mark.usefixtures("resolver")
@pytest.mark.parametrize("kind", ["tcl", "py"])
def test_from_h5_build_is_the_auto_case(tmp_path: Path, kind: str) -> None:
    # The archive carries no OpenSeesTarget, so the rebuilt deck is the
    # auto case: unstamped, whatever the resolver has answered.
    h5, _fem = build_simple_frame_h5(tmp_path)
    text = OpenSeesModel.from_h5(str(h5)).build(kind)
    assert text is not None
    assert text.splitlines()[0] == DECK_BANNER
    assert "; backend " not in text
