"""MotionLOD — hide heavy actors while the camera is moving.

ParaView-style level-of-detail: during an interactive camera gesture
(orbit / pan / wheel-zoom) the most expensive per-frame actors are
hidden so frames stay cheap; ~``settle_ms`` after the last camera
change the full-detail scene is restored with a single render.

One ``ModifiedEvent`` observer on the active camera catches *every*
camera motion path uniformly — the custom Shift/MMB orbit, the VTK
trackball pan, and wheel zoom all modify the camera — so no
per-gesture wiring into ``navigation`` is needed.

Usage::

    lod = MotionLOD(plotter, lambda: registry.dim_node_actors.values())
    lod.install()
"""
from __future__ import annotations

from typing import Any, Callable, Iterable


class MotionLOD:
    """Toggle a set of actors off during camera motion, on at rest.

    Parameters
    ----------
    plotter
        The pyvista/Qt plotter (needs ``.renderer`` and ``.render()``).
    get_actors
        Callable returning the vtk actors to hide while moving. Called
        fresh on every motion-start so it tracks actor swaps (e.g. the
        node cloud is rebuilt on a point-size change).
    settle_ms
        Idle time after the last camera change before full detail is
        restored. 120 ms ≈ imperceptible yet long enough to bridge the
        gaps between mouse-move events during a drag.
    """

    def __init__(
        self,
        plotter: Any,
        get_actors: Callable[[], Iterable[Any]],
        *,
        settle_ms: int = 120,
    ) -> None:
        self._plotter = plotter
        self._get_actors = get_actors
        self._in_motion = False
        # id -> (actor, visibility_before_motion)
        self._saved: dict[int, tuple[Any, int]] = {}
        self._armed = False

        from qtpy import QtCore
        self._timer = QtCore.QTimer()
        self._timer.setSingleShot(True)
        self._timer.setInterval(int(settle_ms))
        self._timer.timeout.connect(self._on_settle)

    # ------------------------------------------------------------------

    def install(self) -> None:
        """Attach the camera observer. Call after the scene is built."""
        try:
            cam = self._plotter.renderer.GetActiveCamera()
            cam.AddObserver("ModifiedEvent", self._on_cam_modified)
            self._armed = True
        except Exception:
            self._armed = False

    def set_enabled(self, enabled: bool) -> None:
        """Pause or resume LOD hiding (e.g. while explode is active)."""
        self._armed = enabled
        if not enabled:
            # Cancel any in-flight settle timer — if it fired after
            # explosion activated it would restore actors that the
            # explode is supposed to keep hidden.
            self._timer.stop()
            self._in_motion = False
            self._saved.clear()

    # ------------------------------------------------------------------

    def _on_cam_modified(self, *_a) -> None:
        if not self._armed:
            return
        if not self._in_motion:
            self._enter_motion()
        # Restart the settle timer on every camera tick — while the
        # user keeps moving, this keeps deferring the full-detail
        # restore. Cheap: a QTimer restart, GUI-thread safe (the VTK
        # render loop runs in the Qt thread under pyvistaqt).
        self._timer.start()

    def _enter_motion(self) -> None:
        self._in_motion = True
        self._saved.clear()
        for act in self._get_actors():
            if act is None:
                continue
            try:
                vis = int(act.GetVisibility())
                self._saved[id(act)] = (act, vis)
                if vis:
                    act.SetVisibility(False)
            except Exception:
                pass
        # No render here — the in-flight interaction render (triggered
        # by navigation right after the camera change) paints the
        # reduced scene, so motion frames are cheap for free.

    def _on_settle(self) -> None:
        for act, vis in self._saved.values():
            try:
                act.SetVisibility(vis)
            except Exception:
                pass
        self._saved.clear()
        self._in_motion = False
        try:
            self._plotter.render()
        except Exception:
            pass
