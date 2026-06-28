"""Animation export — drive the time scrubber, capture frames, encode.

Programmatic API used by both :class:`ResultsViewer` (interactive
``viewer.export_animation(...)``) and headless scripts. Format is
auto-detected from the path suffix:

* ``.mp4`` → H.264 via ``imageio-ffmpeg`` (compact, smooth — best for
  long histories or sharing as video).
* ``.gif`` → animated GIF via Pillow (portable, embeddable in PRs and
  Slack — large file size for long histories or 24-bit color).

The export loop drives :meth:`ResultsDirector.set_step`, which already
runs the full diagram update + render pipeline, then captures the
plotter's current frame. The user's prior step is restored on
completion.

``imageio-ffmpeg`` is an optional dependency (``apegmsh[animation]``);
GIF works without it. MP4 raises a clear install hint when the wheel
is missing.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .diagrams._director import ResultsDirector


_SUPPORTED_SUFFIXES = (".mp4", ".gif")


def export_animation(
    plotter: Any,
    director: "ResultsDirector",
    path: "str | Path",
    *,
    fps: int = 30,
    step_stride: int = 1,
    progress: "Optional[Callable[[int, int], None]]" = None,
) -> Path:
    """Export the time history as an animated MP4 or GIF.

    Parameters
    ----------
    plotter
        PyVista plotter the diagrams are drawing into. Must already
        have its scene built (the substrate + any active diagrams).
    director
        :class:`ResultsDirector` driving the step state. The export
        walks ``range(0, director.n_steps, step_stride)`` plus a final
        frame at ``n_steps - 1`` so the last step is always captured.
    path
        Output path. Suffix selects the format — ``.mp4`` or ``.gif``.
    fps
        Frames per second of the output. The wall-clock duration is
        ``len(captured_steps) / fps``.
    step_stride
        Skip every N-th step. Useful for long histories — a 10000-step
        run with ``step_stride=20`` yields a 500-frame animation
        (16 seconds at 30 fps) instead of a 5-minute clip.
    progress
        Optional ``callback(done, total)`` fired after each frame is
        captured (``done`` is 1-based, ``total`` is the frame count).
        Used by the interactive viewer to drive a progress dialog and
        by headless callers to log. Exceptions raised by the callback
        propagate — a cancel-via-exception is a supported pattern.

    Returns
    -------
    Path
        The resolved output path.
    """
    if plotter is None:
        raise RuntimeError(
            "export_animation: plotter is None. Call viewer.show() first "
            "or pass a constructed pv.Plotter."
        )
    if director is None:
        raise RuntimeError("export_animation: director is None.")

    n_steps = int(director.n_steps)
    if n_steps == 0:
        raise RuntimeError(
            "export_animation: results carry no steps to animate."
        )

    out_path = Path(path)
    suffix = out_path.suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise ValueError(
            f"export_animation: unsupported suffix {suffix!r}. "
            f"Use one of {_SUPPORTED_SUFFIXES}."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if int(fps) <= 0:
        raise ValueError(f"export_animation: fps must be positive, got {fps}.")
    stride = max(1, int(step_stride))

    # Build the step schedule. Always end on the last step so the
    # final state is visible regardless of stride.
    indices: list[int] = list(range(0, n_steps, stride))
    if not indices or indices[-1] != n_steps - 1:
        indices.append(n_steps - 1)

    writer_kwargs = _writer_kwargs_for(suffix, fps)

    import imageio.v2 as imageio  # stable v2 API

    total = len(indices)
    saved_step = director.step_index
    try:
        with imageio.get_writer(str(out_path), **writer_kwargs) as writer:
            for n, i in enumerate(indices):
                director.set_step(i)
                # ``set_step`` early-returns when already at the target
                # step (and skips the render). Force a render so the
                # first captured frame is correct in that case.
                try:
                    plotter.render()
                except Exception as _render_exc:
                    _log.warning("render() failed at step %d: %s", i, _render_exc)
                frame = plotter.screenshot(
                    return_img=True, transparent_background=False,
                )
                writer.append_data(frame)
                if progress is not None:
                    progress(n + 1, total)
    finally:
        # Restore whatever the user was looking at before the export.
        try:
            director.set_step(saved_step)
        except Exception:
            pass

    return out_path


def _writer_kwargs_for(suffix: str, fps: int) -> dict:
    """Pick imageio writer kwargs for the chosen format."""
    if suffix == ".mp4":
        try:
            import imageio_ffmpeg  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "MP4 export requires imageio-ffmpeg. Install with "
                "`pip install apegmsh[animation]` (or directly: "
                "`pip install imageio-ffmpeg`)."
            ) from exc
        # ``quality`` is on a 0–10 scale; 8 is a reasonable default
        # (visually lossless for typical scenes, ~1–2 MB/sec).
        return dict(fps=int(fps), codec="libx264", quality=8,
                    macro_block_size=1)
    if suffix == ".gif":
        # Pillow plugin handles GIF. ``duration`` is per-frame display
        # time in ms — the modern replacement for ``fps`` on this
        # backend. ``loop=0`` = infinite loop.
        return dict(duration=int(round(1000.0 / int(fps))), loop=0)
    raise ValueError(f"Unsupported suffix {suffix!r}")
