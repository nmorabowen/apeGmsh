"""``export_animation`` on the session path — a re-step loop (X2-b).

``Results.export_animation`` used to build a Qt ``ResultsViewer`` and
drive its ``ResultsDirector`` through ``viewers/animation.py``. This is
the same verb as a loop over ONE ``ResultsSession`` mesh pane: realize
it once into an offscreen plotter (the ``render_still`` plotter, theme
and camera rules), then for each scheduled step move the session
instant and re-step the pane in place (``can_restep`` /
``restep_pane``, ADR 0098 A4.2), falling back to a full re-realize
when the fast path refuses. Each frame is the plotter's screenshot,
appended to an imageio writer exactly as the old loop did.

Moved, not extended (ADR 0112 D8): no feature is added to the session
path. On a staged model each frame draws only the elements active in
the animated stage, as the Qt path did; ``realize`` applies that rule
to every mesh pane (X2-f, #1722). What the Qt window
drew and this does not (the axes triad, the
window chrome, the viewer's substrate shading) are accepted look gaps
(maintainer ruling 2026-10-11, #1663).

Partial writes: frames are encoded to ``<stem>.partial<suffix>`` and
renamed onto ``path`` only after the last frame is appended and the
writer closed. A failure removes the partial file and propagates, so
no truncated video is ever reported as written.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from .._stills import (
    _DEFAULT_WINDOW,
    _SKIP_ENV,
    _apply_camera,
    _apply_theme,
    _env_skips,
    _open_plotter,
)
from ..render import _CAMERAS, _resolve_camera
from ._realize import can_restep, realize_pane, restep_pane
from ._reconciler import LedgerBackend

if TYPE_CHECKING:
    from apeGmsh.results.session import MeshView, ResultsSession

_SUPPORTED_SUFFIXES = (".mp4", ".gif")

#: ``ResultsDirector`` attributes a pre-X2-b ``setup(plotter, director)``
#: reaches for, none of which a ``MeshView`` has.
_DIRECTOR_ATTRS = frozenset({
    "registry", "geometries", "dispatcher", "results", "stages",
    "stage_id", "set_stage", "set_step", "step_index", "n_steps",
    "scopes",
})


def frame_schedule(n_steps: int, step_stride: int) -> list[int]:
    """``range(0, n, stride)`` plus the last step (the old loop's rule)."""
    stride = max(1, int(step_stride))
    indices = list(range(0, int(n_steps), stride))
    if not indices or indices[-1] != n_steps - 1:
        indices.append(int(n_steps) - 1)
    return indices


def export_session_animation(
    session: "ResultsSession",
    view: "MeshView",
    path: "str | Path",
    *,
    stage_id: str,
    fps: int = 30,
    step_stride: int = 1,
    camera: Any = None,
    window_size: "Optional[tuple[int, int]]" = _DEFAULT_WINDOW,
    setup: "Optional[Callable[[Any, Any], None]]" = None,
) -> Optional[Path]:
    """Animate ``view`` over ``stage_id``; the written Path, or ``None``.

    ``None`` (with the ``[skip viewer]`` notice) under
    ``APEGMSH_SKIP_VIEWER=1`` or with no GL context: no file is written.
    ``setup(plotter, view)`` runs once after the first frame's scene is
    built and the camera applied, before any frame is captured; if it
    changes the view the pane is realized again.
    """
    from apeGmsh.results.session import Instant

    from ..backends import PyVistaQtBackend

    out = Path(path)
    suffix = out.suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise ValueError(
            f"export_animation: unsupported suffix {suffix!r}. "
            f"Use one of {_SUPPORTED_SUFFIXES}."
        )
    if int(fps) <= 0:
        raise ValueError(f"export_animation: fps must be positive, got {fps}.")
    results = session.results
    if results is None or results.fem is None:
        raise RuntimeError(
            "export_animation requires a Results with a bound FEMData "
            "(construct with model= / model_h5= or call results.bind)."
        )
    n_steps = int(results.stage(stage_id).n_steps)
    if n_steps <= 0:
        raise RuntimeError(
            "export_animation: results carry no steps to animate."
        )
    indices = frame_schedule(n_steps, step_stride)
    # A camera token goes through the still rule (planar -> xy); any
    # other value (an explicit position) is assigned to the plotter.
    token = camera is None or (isinstance(camera, str) and camera in _CAMERAS)
    if token:
        camera = _resolve_camera(camera, results.fem.nodes.coords)
    writer_kwargs = _writer_kwargs_for(suffix, fps)
    if _env_skips():
        print(_SKIP_ENV)
        return None

    plotter = _open_plotter(tuple(window_size or _DEFAULT_WINDOW))
    if plotter is None:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".partial" + out.suffix)
    try:
        _apply_theme(plotter)
        backend = LedgerBackend(PyVistaQtBackend(plotter))
        session.time = Instant(stage_id, indices[0])
        realized = realize_pane(session, view, backend)
        if token:
            _apply_camera(plotter, camera)
        else:
            plotter.camera_position = camera
        if setup is not None:
            realized = _run_setup(session, view, plotter, backend, realized, setup)

        import imageio.v2 as imageio  # stable v2 API

        with imageio.get_writer(str(tmp), **writer_kwargs) as writer:
            for n, step in enumerate(indices):
                if n:
                    session.time = Instant(stage_id, step)
                    realized = _step_to(session, view, backend, realized)
                plotter.render()
                frame = plotter.screenshot(
                    return_img=True, transparent_background=False,
                )
                writer.append_data(frame)
        if not tmp.exists():
            # The encoder closed without leaving a file: nothing was
            # written, so never report a path as success.
            raise RuntimeError(
                f"export_animation: the {suffix} encoder wrote no file "
                f"({tmp.name}); nothing was written to {out}."
            )
        tmp.replace(out)
        return out
    except BaseException:
        _unlink(tmp)
        raise
    finally:
        try:
            plotter.close()
        except Exception:
            pass


def _step_to(
    session: "ResultsSession", view: "MeshView", backend: LedgerBackend,
    realized: Any,
) -> Any:
    """Re-step in place when the fast path allows, else re-realize.

    The re-step reuses the realized cell set, which carries the stage's
    activation (X2-f); one export animates ONE stage, and
    ``restep_pane`` refuses a stage change anyway.
    """
    if can_restep(realized, view):
        return restep_pane(session, view, realized, backend)
    return _rerealize(session, view, backend)


def _rerealize(
    session: "ResultsSession", view: "MeshView", backend: LedgerBackend,
) -> Any:
    """Tear down exactly what was emitted (by ledger) and realize again."""
    for bar_key in list(backend.bar_keys):
        backend.remove_scalar_bar(bar_key)
    for handle in list(backend.layer_handles):
        backend.remove_layer(handle)
    return realize_pane(session, view, backend)


def _run_setup(
    session: "ResultsSession", view: "MeshView", plotter: Any,
    backend: LedgerBackend, realized: Any,
    setup: "Callable[[Any, Any], None]",
) -> Any:
    """``setup(plotter, view)``; re-realize if it changed the session."""
    ticks: list[None] = []

    def _tick() -> None:
        ticks.append(None)

    session.subscribe(_tick)
    try:
        try:
            setup(plotter, view)
        except AttributeError as exc:
            # Only a callback reaching for the retired director's surface
            # ON THE VIEW is re-diagnosed; any other AttributeError (a bug
            # in a callback written the new way) propagates untouched.
            if getattr(exc, "obj", None) is not view or (
                getattr(exc, "name", None) not in _DIRECTOR_ATTRS
            ):
                raise
            raise TypeError(
                "export_animation(setup=...) now calls setup(plotter, view) "
                "with the ResultsSession MeshView (e.g. view.contour = "
                "Contour('displacement_x')), not a ResultsDirector: the "
                "callback failed with " f"{type(exc).__name__}: {exc}"
            ) from exc
    finally:
        session.unsubscribe(_tick)
    if ticks:
        return _rerealize(session, view, backend)
    return realized


def _writer_kwargs_for(suffix: str, fps: int) -> dict:
    """imageio writer kwargs for the format (``viewers/animation.py``'s
    rule, copied so this path never imports the legacy module)."""
    if suffix == ".mp4":
        try:
            import imageio_ffmpeg  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "MP4 export requires imageio-ffmpeg. Install with "
                "`pip install apegmsh[animation]` (or directly: "
                "`pip install imageio-ffmpeg`)."
            ) from exc
        return dict(fps=int(fps), codec="libx264", quality=8,
                    macro_block_size=1)
    # GIF (Pillow): per-frame duration in ms; loop=0 loops forever.
    return dict(duration=int(round(1000.0 / int(fps))), loop=0)


def _unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


__all__ = ["export_session_animation", "frame_schedule"]
