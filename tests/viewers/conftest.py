"""Shared helpers for viewer diagram tests (ADR 0042, R-B).

``RecordingBackend`` is a no-GL stand-in for a ``RenderBackend``: it
captures the ``SceneLayer``s a migrated diagram emits so tests can
assert on the *emitted layer* rather than pixels — the headless
testability win the render seam delivers.
"""
from __future__ import annotations

import functools
from typing import Any

import pytest


# =====================================================================
# Offscreen-GL capability probes
# =====================================================================
#
# A few render tests assert EXACT pixel semantics: a framebuffer that
# is bit-clean after an actor is removed, and wireframe rasterization
# that is bit-identical through two different mapper paths. Mesa on
# the CI runners satisfies both, so the strict assertions are real
# gates there and must stay strict. Some drivers (observed: Windows 11
# desktop GL, 2026-07-31) do not:
#
#   * removing an actor leaves residue in the offscreen buffer
#     (measured 22 stray px on the gizmo scene, 5.5k on a cube);
#   * wireframe through the render-surface fast path vs the plain
#     volumetric mapper differed by ONE pixel at ONE intensity level
#     out of 40 000, on frames that both paint 9 102 px — i.e. the
#     picture is the same, the rasterizer is not bit-deterministic.
#
# Neither is an apeGmsh defect: both reproduce at the commit that
# introduced the tests, and CI is green on the same commits. So the
# tests skip where the platform cannot honour the assertion instead of
# failing red on dev machines — and stay strict everywhere else.
# Probes are cached; each builds and closes its own plotter.


def _probe_plotter():
    import pyvista as pv

    p = pv.Plotter(off_screen=True, window_size=(80, 80))
    p.background_color = "black"
    return p


def _painted_px(plotter) -> int:
    import numpy as np

    plotter.render()
    img = np.asarray(
        plotter.screenshot(return_img=True, transparent_background=False),
    )
    return int((img != 0).any(axis=2).sum())


@functools.lru_cache(maxsize=1)
def gl_clears_removed_actors() -> bool:
    """Does the offscreen buffer go clean when an actor is removed?"""
    try:
        import pyvista as pv

        plotter = _probe_plotter()
        try:
            actor = plotter.add_mesh(pv.Cube(), color="red")
            plotter.camera_position = "iso"
            if _painted_px(plotter) == 0:
                return False            # nothing drawn -> probe is void
            plotter.remove_actor(actor)
            return _painted_px(plotter) == 0
        finally:
            plotter.close()
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def gl_wireframe_is_bit_exact() -> bool:
    """Do two mapper paths rasterize one wireframe bit-identically?

    Mirrors the F-PARITY comparison: the same polydata rendered from a
    pre-extracted surface and from the unstructured grid it came from.
    Uses a TET grid, not a box: the divergence lives in diagonal line
    rasterization, and an axis-aligned cube wireframe agrees even on
    stacks that fail the real comparison.
    """
    try:
        import numpy as np
        import pyvista as pv
        from vtkmodules.vtkFiltersGeneral import vtkDataSetTriangleFilter

        img = pv.ImageData(dimensions=(5, 5, 5))
        tri = vtkDataSetTriangleFilter()
        tri.SetInputData(img.cast_to_unstructured_grid())
        tri.Update()
        grid = pv.UnstructuredGrid(tri.GetOutput())
        grid.point_data["v"] = np.asarray(
            grid.points[:, 2], dtype=np.float64,
        )
        surface = grid.extract_surface()
        frames = []
        # Scalars + colormap matter: the divergence is in the
        # INTERPOLATED colour along a diagonal line, not the geometry.
        # A flat-coloured wireframe agrees on stacks that fail here.
        style = dict(
            scalars="v", cmap="jet", show_scalar_bar=False,
            clim=(0.0, float(np.asarray(grid.points)[:, 2].max())),
            style="wireframe",
        )
        for mesh in (surface, grid):
            plotter = _probe_plotter()
            try:
                plotter.add_mesh(mesh, **style)
                plotter.camera_position = "iso"
                plotter.render()
                frames.append(
                    np.asarray(
                        plotter.screenshot(
                            return_img=True, transparent_background=False,
                        ),
                    ).copy(),
                )
            finally:
                plotter.close()
        return bool(np.array_equal(frames[0], frames[1]))
    except Exception:
        return False


@pytest.fixture
def requires_gl_actor_clear() -> None:
    """Skip unless actor removal leaves a bit-clean framebuffer."""
    if not gl_clears_removed_actors():
        pytest.skip(
            "offscreen GL leaves residue after actor removal on this "
            "platform; the exact painted-pixel assertion cannot hold "
            "(see conftest capability-probe note)",
        )


@pytest.fixture
def requires_gl_wireframe_exact() -> None:
    """Skip unless wireframe rasterizes identically across mappers."""
    if not gl_wireframe_is_bit_exact():
        pytest.skip(
            "offscreen GL wireframe rasterization is not bit-identical "
            "across mapper paths on this platform (see conftest "
            "capability-probe note)",
        )


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
        self.scalar_bars: dict[str, Any] = {}   # bar_key -> ScalarBarSpec
        self.bar_formats: dict[str, str] = {}
        self.moved_bars: list[str] = []
        self.clip_planes: tuple = ()            # ADR 0083 set_clip_planes
        self.viewport: tuple[int, int] = (1280, 800)

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

    def set_clip_planes(self, planes: Any) -> None:
        self.clip_planes = tuple(planes or ())

    def viewport_size(self) -> tuple[int, int]:
        return self.viewport

    def add_scalar_bar(self, handle: _Handle, spec: Any) -> None:
        self.scalar_bars[spec.key] = spec

    def move_scalar_bar(self, bar_key: str, spec: Any) -> bool:
        if bar_key not in self.scalar_bars:
            return False
        self.scalar_bars[bar_key] = spec
        self.moved_bars.append(bar_key)
        return True

    def remove_scalar_bar(self, bar_key: str) -> None:
        self.scalar_bars.pop(bar_key, None)

    def set_scalar_bar_format(self, bar_key: str, fmt: str) -> None:
        self.bar_formats[bar_key] = fmt

    def reset_camera(self) -> None:
        pass

    def render(self) -> None:
        pass

    def screenshot(self, path: Any) -> None:
        pass

    def supports_picking(self) -> bool:
        return False


class PumpFailureCollector:
    """Collects pump failures reported through ``viewers/_failures``.

    Only ``pump.*`` reports are collected — the registry also carries
    ordinary ``safe_slot`` failures, which are not this gate's business.
    """

    def __init__(self) -> None:
        self.failures: list[tuple[str, BaseException]] = []

    def __call__(self, name: str, exc: BaseException) -> None:
        if name.startswith("pump."):
            self.failures.append((name, exc))

    def summary(self) -> str:
        return "\n".join(
            f"  {name}: {type(exc).__name__}: {exc}"
            for name, exc in self.failures
        )


@pytest.fixture(autouse=True)
def pump_failures(request) -> Any:
    """ADR 0084 D4 — a failing pump fails the test, not just the viewport.

    The pump catch sites keep swallowing (the viewport must survive in
    production) but now report through the ``_failures`` registry. A
    bare re-raise inside a pump is useless in tests: pumps run from
    ``QTimer`` callbacks / ``safe_slot`` contexts where the exception
    never reaches the test. So collect during the test and fail at
    teardown instead.

    Opt out with ``@pytest.mark.allow_pump_failures`` for tests that
    exercise a pump failure on purpose.
    """
    from apeGmsh.viewers._failures import (
        register_error_handler,
        unregister_error_handler,
    )

    collector = PumpFailureCollector()
    register_error_handler(collector)
    yield collector
    unregister_error_handler(collector)
    if collector.failures and not request.node.get_closest_marker(
        "allow_pump_failures"
    ):
        pytest.fail(
            f"{len(collector.failures)} pump failure(s) reported during "
            f"this test:\n{collector.summary()}"
        )


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
