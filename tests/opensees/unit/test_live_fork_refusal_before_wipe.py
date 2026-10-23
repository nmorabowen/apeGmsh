"""#2440: in-process ``apeSees.analyze`` on stock refuses a fork-only chain
type BEFORE the live emitter wipes the domain.

#2441's gate refuses at the chain command itself, which comes after
``wipe`` and the model emit, so a refused build still destroyed whatever the
user had in the domain.  ``refuse_fork_chain_before_wipe`` reads the
declared chain (each chain primitive's own ``_emit``, plus the constraint
handler the bridge auto-emits) and refuses first.
"""
from __future__ import annotations

import warnings
from typing import Any

import pytest

from tests.opensees.h5.test_h5_solve_stamp import (
    _FORK_MEMBER_ROWS,
    _TRANSIENT,
    _StockOps,
    _chain_frame,
)


@pytest.mark.parametrize("family, row", _FORK_MEMBER_ROWS,
                         ids=[f"{f}-{r}" for f, r in _FORK_MEMBER_ROWS])
def test_stock_refuses_before_the_wipe(
    monkeypatch: pytest.MonkeyPatch, family: str, row: str,
) -> None:
    """Every fork-only chain member: the stand-in sees no call at all."""
    import apeGmsh.opensees.emitter.live as live_mod

    stock = _StockOps()
    monkeypatch.setattr(live_mod, "_get_ops", lambda: stock)
    ops = _chain_frame(family, row, staged=False)
    transient = family == "integrator" and row in _TRANSIENT
    with pytest.raises(RuntimeError, match="requires the Ladruno fork|fork-only"):
        ops.analyze(steps=1, dt=0.01 if transient else None)
    assert stock.calls == [], stock.calls[:5]


def test_a_stock_chain_still_wipes_and_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import apeGmsh.opensees.emitter.live as live_mod

    stock = _StockOps()
    monkeypatch.setattr(live_mod, "_get_ops", lambda: stock)
    assert _chain_frame("integrator", "LoadControl", staged=False).analyze(
        steps=1) == 0
    assert stock.calls[0] == ("wipe",)
    assert ("analyze", 1) in stock.calls


def test_the_fork_passes_a_fork_only_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The resolver's verdict says fork: a fork-only chain passes."""
    import apeGmsh.opensees.emitter.live as live_mod

    class _Fork:
        kind = "fork"

    monkeypatch.setattr(live_mod, "get_backend_info", lambda: _Fork())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bm = _chain_frame("integrator", "LadrunoLoadControl",
                          staged=False).build()
    live_mod.refuse_fork_chain_before_wipe(bm)


def test_a_stock_chain_never_resolves_the_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No fork-only type: the check imports no OpenSees backend, so it runs
    where none is installed (the ``suite`` lane, a stubbed live emitter)."""
    import apeGmsh.opensees.emitter.live as live_mod

    def _no_backend() -> Any:
        raise AssertionError("the check resolved the backend")

    monkeypatch.setattr(live_mod, "get_backend_info", _no_backend)
    monkeypatch.setattr(live_mod, "_get_ops", _no_backend)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bm = _chain_frame("integrator", "LoadControl", staged=False).build()
    live_mod.refuse_fork_chain_before_wipe(bm)


@pytest.mark.live
def test_live_a_refused_stock_analyze_keeps_the_users_node() -> None:
    """Real openseespy: a node the user put in the domain survives a refused
    stock ``analyze``.  Without the check, #2441's gate refused only after
    the wipe, and the node was gone."""
    pytest.importorskip("openseespy.opensees")
    from apeGmsh.opensees.emitter.live import get_backend_name, get_ops

    if get_backend_name() == "ladruno-fork":
        pytest.skip("the fork runs a fork-only chain; nothing to refuse")
    dom = get_ops()
    dom.wipe()
    try:
        dom.model("basic", "-ndm", 2, "-ndf", 2)
        dom.node(9999, 0.0, 0.0)
        ops = _chain_frame("integrator", "LadrunoLoadControl", staged=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(RuntimeError, match="requires the Ladruno fork"):
                ops.analyze(steps=1)
        assert 9999 in list(dom.getNodeTags())
    finally:
        dom.wipe()
