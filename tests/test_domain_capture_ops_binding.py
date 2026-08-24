"""``DomainCapture`` must query the module the bridge is DRIVING.

``_lazy_ops`` used to go straight to ``import openseespy.opensees``. That is a
*different module object* from the ``opensees`` a :class:`LiveOpsEmitter`
imports whenever both are importable — e.g. a locally built fork on
``PYTHONPATH`` beside a stock openseespy in site-packages. Two modules mean two
domains, so the capture sampled an empty one and every ``nodeDisp`` returned
``WARNING no response is found`` while the analysis it was meant to be
sampling had run, and converged, in the other module.

It is invisible in a normally-wired venv, where ``openseespy.opensees is
opensees``. In a mixed one it reads exactly like a solver regression — which
is how it was found: it produced a convincing false "the fork upgrade broke
this" call before the two-module split was spotted.
"""
from __future__ import annotations

import pytest

from apeGmsh.results.capture._domain import DomainCapture


class _Sentinel:
    """Stands in for a live openseespy module."""

    def __init__(self, name: str) -> None:
        self.name = name


class _Emitter:
    def __init__(self, ops: object) -> None:
        self.ops = ops


class _Bridge:
    def __init__(self, live_ops: object | None) -> None:
        self._live_emitter = _Emitter(live_ops) if live_ops is not None else None


def _capture(*, ops=None, bridge=None) -> DomainCapture:
    """A DomainCapture with only the fields ``_lazy_ops`` reads."""
    cap = DomainCapture.__new__(DomainCapture)
    cap._ops = ops
    cap._bridge = bridge
    return cap


def test_prefers_the_module_the_bridge_is_driving():
    live = _Sentinel("fork-opensees")
    assert _capture(bridge=_Bridge(live))._lazy_ops() is live


def test_an_explicitly_injected_ops_still_wins():
    """The ``ops=`` seam is what the rest of the capture tests mock through;
    the bridge must not override it."""
    injected = _Sentinel("injected")
    live = _Sentinel("fork-opensees")
    cap = _capture(ops=injected, bridge=_Bridge(live))
    assert cap._lazy_ops() is injected


def test_falls_back_when_no_analysis_has_run_yet():
    """``domain_capture()`` is entered BEFORE ``analyze()`` builds the live
    emitter, so a bridge with no emitter must not blow up — it falls through
    to the import."""
    pytest.importorskip("openseespy.opensees")
    import openseespy.opensees as stock

    assert _capture(bridge=_Bridge(None))._lazy_ops() is stock


def test_falls_back_with_no_bridge_at_all():
    """``DomainCapture.from_h5`` builds one with ``bridge=None``."""
    pytest.importorskip("openseespy.opensees")
    import openseespy.opensees as stock

    assert _capture()._lazy_ops() is stock


def test_no_bridge_and_no_openseespy_still_names_the_cause(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *a, **kw):
        if name.startswith("openseespy"):
            raise ImportError("blocked for test")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    with pytest.raises(RuntimeError, match="openseespy is not installed"):
        _capture()._lazy_ops()
