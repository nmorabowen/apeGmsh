"""Shared helpers for viewer diagram tests (ADR 0042, R-B).

``RecordingBackend`` is a no-GL stand-in for a ``RenderBackend``: it
captures the ``SceneLayer``s a migrated diagram emits so tests can
assert on the *emitted layer* rather than pixels — the headless
testability win the render seam delivers.
"""
from __future__ import annotations

from typing import Any

import pytest


class _Handle:
    def __init__(self, layer_id: str) -> None:
        self.layer_id = layer_id
        self.visible = True


class RecordingBackend:
    """Captures emitted layers; satisfies the RenderBackend Protocol."""

    def __init__(self) -> None:
        self.layers: dict[str, Any] = {}
        self.removed: list[str] = []
        self.colors: dict[str, Any] = {}        # layer_id -> ColorSpec
        self.opacities: dict[str, float] = {}   # layer_id -> opacity
        self.scalar_bars: dict[str, Any] = {}   # layer_id -> ScalarBarSpec
        self.bar_formats: dict[str, str] = {}

    def add_layer(self, layer: Any) -> _Handle:
        self.layers[layer.layer_id] = layer
        self.colors[layer.layer_id] = layer.color
        return _Handle(layer.layer_id)

    def update_layer(self, handle: _Handle, layer: Any) -> None:
        self.layers[handle.layer_id] = layer
        self.colors[handle.layer_id] = layer.color

    def remove_layer(self, handle: _Handle) -> None:
        self.layers.pop(handle.layer_id, None)
        self.removed.append(handle.layer_id)

    def set_visibility(self, handle: _Handle, mask: Any) -> None:
        pass

    def set_layer_visible(self, handle: _Handle, visible: bool) -> None:
        handle.visible = bool(visible)

    def set_layer_color(self, handle: _Handle, color: Any) -> None:
        self.colors[handle.layer_id] = color

    def set_layer_opacity(self, handle: _Handle, opacity: float) -> None:
        self.opacities[handle.layer_id] = float(opacity)

    def add_scalar_bar(self, handle: _Handle, spec: Any) -> None:
        self.scalar_bars[handle.layer_id] = spec

    def remove_scalar_bar(self, layer_id: str) -> None:
        self.scalar_bars.pop(layer_id, None)

    def set_scalar_bar_format(self, layer_id: str, fmt: str) -> None:
        self.bar_formats[layer_id] = fmt

    def reset_camera(self) -> None:
        pass

    def render(self) -> None:
        pass

    def screenshot(self, path: Any) -> None:
        pass

    def supports_picking(self) -> bool:
        return False


@pytest.fixture
def backend() -> RecordingBackend:
    return RecordingBackend()


@pytest.fixture
def pv_backend():
    """A real offscreen ``PyVistaQtBackend`` for render-integration tests."""
    import pyvista as pv

    from apeGmsh.viewers.backends import PyVistaQtBackend
    try:
        plotter = pv.Plotter(off_screen=True)
    except Exception:  # pragma: no cover
        pytest.skip("no offscreen render context")
    yield PyVistaQtBackend(plotter)
    plotter.close()


@pytest.fixture
def headless_plotter():
    """Offscreen ``PyVistaQtBackend`` for diagram attach/update tests.

    ADR 0042 R-B.final: a diagram attaches to a ``RenderBackend``, not a
    raw pyvista plotter (the base ``attach`` no longer wraps). This yields
    the backend; the raw plotter behind it is at ``.plotter`` for the few
    tests that assert on plotter-level state (e.g. ``scalar_bars``).

    Named ``headless_plotter`` for continuity with the dozens of diagram
    tests that pass it straight into ``diagram.attach(...)``.
    """
    import pyvista as pv

    from apeGmsh.viewers.backends import PyVistaQtBackend
    try:
        plotter = pv.Plotter(off_screen=True)
    except Exception:  # pragma: no cover
        pytest.skip("no offscreen render context")
    yield PyVistaQtBackend(plotter)
    plotter.close()
