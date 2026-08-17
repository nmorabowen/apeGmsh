"""Offscreen still-shot machinery (relocated from ``render.py``).

The plotter / theme / camera / screenshot half of the ADR 0094 still
path, moved here verbatim (ADR 0098 S1) so the session still client
(``apeGmsh.viewers.session``) and the ``render.py`` verbs share ONE
GL-skip and ONE screenshot discipline. Behaviour is unchanged:
``render.py`` imports these names back into its module namespace.

``APEGMSH_SKIP_VIEWER=1`` or no GL prints the ``[skip viewer]`` notice
and writes no file — the caller returns ``None`` / ``()``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

_SKIP_ENV = "[skip viewer] APEGMSH_SKIP_VIEWER set"
_SKIP_GL = "[skip viewer] no GL context"
# VTK/pyvista raise these when the offscreen window cannot start.
# TypeError / ValueError / AttributeError are bugs and must propagate.
_GL_EXC = (RuntimeError, OSError)
_DEFAULT_WINDOW = (1280, 720)


def _env_skips() -> bool:
    return bool(os.environ.get("APEGMSH_SKIP_VIEWER"))


def _open_plotter(window_size: tuple[int, int]) -> Any:
    try:
        import pyvista as pv
        return pv.Plotter(off_screen=True, window_size=list(window_size))
    except _GL_EXC:
        print(_SKIP_GL)
        return None


def _apply_theme(plotter: Any) -> None:
    try:
        from apeGmsh.viewers.scene.background import apply_background
        from apeGmsh.viewers.ui.theme import THEME
        apply_background(plotter, THEME.current)
    except Exception:
        pass


def _add_substrate(plotter: Any, scene: Any) -> None:
    """Fill + wireframe from the theme substrate colors. No outline."""
    try:
        from apeGmsh.viewers.ui.theme import THEME
        palette = THEME.current
        fill = palette.substrate_color
        edge = palette.substrate_edge_color
    except Exception:
        fill, edge = "#bfbfbf", "#1a1a1a"
    plotter.add_mesh(
        scene.grid,
        color=fill,
        show_edges=True,
        edge_color=edge,
        line_width=1.0,
        lighting=True,
        name="render_substrate",
        reset_camera=True,
    )


def _apply_camera(plotter: Any, camera: str) -> None:
    if camera == "iso":
        plotter.view_isometric()
    elif camera == "xy":
        plotter.view_xy()
    elif camera == "xz":
        plotter.view_xz()
    else:
        plotter.view_yz()
    plotter.reset_camera()


def _screenshot(plotter: Any, path: "str | Path") -> Optional[Path]:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # PyVista only writes known image suffixes; keep .png on the temp.
    tmp = out.with_name(out.stem + ".partial" + out.suffix)
    try:
        plotter.screenshot(str(tmp))
    except _GL_EXC:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        print(_SKIP_GL)
        return None
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise
    if not tmp.exists() or tmp.stat().st_size == 0:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        print(_SKIP_GL)
        return None
    tmp.replace(out)
    return out


def _shoot(
    path: "str | Path",
    *,
    camera: str,
    window_size: tuple[int, int],
    scene: Any = None,
    add_substrate: bool = False,
    populate: Any = None,
) -> Optional[Path]:
    plotter = None
    try:
        plotter = _open_plotter(window_size)
        if plotter is None:
            return None
        _apply_theme(plotter)
        if populate is not None:
            populate(plotter)
        elif add_substrate:
            _add_substrate(plotter, scene)
        _apply_camera(plotter, camera)
        return _screenshot(plotter, path)
    finally:
        if plotter is not None:
            try:
                plotter.close()
            except Exception:
                pass


__all__ = [
    "_DEFAULT_WINDOW",
    "_GL_EXC",
    "_SKIP_ENV",
    "_SKIP_GL",
    "_add_substrate",
    "_apply_camera",
    "_apply_theme",
    "_env_skips",
    "_open_plotter",
    "_screenshot",
    "_shoot",
]
