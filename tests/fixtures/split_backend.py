"""A fork build beside stock openseespy, faked: two modules, two domains.

The bridge's resolver (``apeGmsh.opensees.emitter.live``) answers one module
while the name ``openseespy.opensees`` binds another. That is the setup in
which code importing openseespy by name talks to a different domain from the
one the bridge built the model in (9ffe6aa2). Nothing real is imported: a dev
box's installed fork may be stale, and most CI lanes have no backend at all.
"""
from __future__ import annotations

import sys
import types

import pytest

from apeGmsh.opensees.emitter import live


def split_backend(monkeypatch: pytest.MonkeyPatch, bound: object, stray: object) -> None:
    """Make the resolver answer ``bound`` and ``openseespy.opensees`` bind ``stray``."""
    monkeypatch.setattr(live, "_get_ops", lambda: bound)
    parent = types.ModuleType("openseespy")
    parent.opensees = stray  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openseespy", parent)
    monkeypatch.setitem(sys.modules, "openseespy.opensees", stray)
