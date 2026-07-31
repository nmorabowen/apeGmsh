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
# Exact-pixel assertions on inexact GL stacks
# =====================================================================
#
# A few render tests assert EXACT pixel semantics: a framebuffer that
# is bit-clean once an actor is removed, and wireframe rasterization
# that is bit-identical through two different mapper paths. Mesa on
# the CI runners satisfies both, so those assertions are real gates
# there and MUST stay strict. Some drivers (observed: Windows 11
# desktop GL, 2026-07-31) do not:
#
#   * removing an actor leaves residue in the offscreen buffer
#     (measured 22 stray px on the gizmo scene, 5.5k on a bare cube);
#   * the render-surface fast path vs the plain volumetric mapper
#     differed by ONE pixel at ONE intensity level out of 40 000, on
#     frames that both paint 9 102 px — the picture is identical, the
#     rasterizer is not bit-deterministic. Solid renders match
#     exactly; only interpolated-colour diagonal lines diverge.
#
# Neither is an apeGmsh defect: both reproduce at the commit that
# introduced the test, and CI is green on those same commits.
#
# These helpers classify the ACTUAL comparison rather than probing a
# synthetic stand-in. A hand-rolled proxy probe was tried first and
# disagreed with the real comparison in BOTH directions — too lenient
# locally (an axis-aligned cube agrees where tets do not), then too
# strict on mesa, where it silently skipped a test that had been
# passing. Classifying the real frames cannot drift from the thing it
# is meant to describe:
#
#   identical            -> pass (strict, unchanged on mesa/CI)
#   trivial noise        -> skip with a reason (rasterizer, not us)
#   material divergence  -> FAIL (a real regression still fails)
#
# The noise band is deliberately tiny. A genuine break in either
# contract moves thousands of pixels (edges appearing, an actor still
# drawn), never one pixel at one intensity level.


def _frames_match_or_skip(actual, expected, *, what: str) -> None:
    """Exact-frame assertion that tolerates only rasterizer noise."""
    import numpy as np

    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape == expected.shape and np.array_equal(actual, expected):
        return
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{what}: frame shape {actual.shape} != {expected.shape}",
        )
    delta = np.abs(actual.astype(np.int32) - expected.astype(np.int32))
    differing = int((delta != 0).any(axis=2).sum())
    max_delta = int(delta.max())
    ink = int((expected != 0).any(axis=2).sum())
    # Noise: a handful of pixels (or <=0.1% of the drawn ink) off by at
    # most one intensity level.
    if differing <= max(4, ink // 1000) and max_delta <= 1:
        pytest.skip(
            f"{what}: {differing} px differ by <={max_delta} intensity "
            f"level out of {ink} painted - this GL stack does not "
            f"rasterize identically across mapper paths (see conftest "
            f"exact-pixel note); the picture is the same",
        )
    raise AssertionError(
        f"{what}: {differing} px differ (max delta {max_delta}) out of "
        f"{ink} painted - too large to be rasterizer noise",
    )


def _cleared_or_skip(painted: int, reference_ink: int, *, what: str) -> None:
    """Assert a frame went clean; tolerate only faint residue."""
    if painted == 0:
        return
    if painted <= max(2, reference_ink // 20):
        pytest.skip(
            f"{what}: {painted} px remain of {reference_ink} painted - "
            f"this GL stack leaves residue after actor removal (see "
            f"conftest exact-pixel note); the actors are gone",
        )
    raise AssertionError(
        f"{what}: {painted} px still painted of {reference_ink} - the "
        f"actor looks still present, not residue",
    )


@pytest.fixture
def frames_match_or_skip():
    """Callable: assert two frames match, tolerating rasterizer noise."""
    return _frames_match_or_skip


@pytest.fixture
def cleared_or_skip():
    """Callable: assert a frame cleared, tolerating faint residue."""
    return _cleared_or_skip


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
