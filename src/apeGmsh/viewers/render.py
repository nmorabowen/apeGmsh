"""Offscreen stills from the viewer scene builders (ADR 0094 S1).

This module *writes stills* (PNG files). That is a different verb from
ADR 0042 ``RenderBackend.render()``, which pumps a live scene. Callers
hold a broker and want a Path::

    fem.render("mesh.png")
    g.model.render("geom.png")
    g.mesh.render("mesh.png")

``results.render`` / ``results.render_pack`` are drawn on the session
path (``viewers/session/_results_stills.py``), which reuses this
module's closed ``view=`` / ``camera=`` / ``deform=`` validation.

Stills come from ``pv.Plotter(off_screen=True)`` (VTK offscreen, not
Qt ``QT_QPA_PLATFORM=offscreen``) plus ``build_fem_scene`` or the
live BRep / mesh scene builders. There is no ``QMainWindow``, no dock,
and no ``app.exec_()`` / ``win.exec_()`` (INV-2).

``view=`` is a closed set (INV-10): ``mesh`` / ``contour`` /
``deformed`` / ``reactions``. ``setup(plotter, director)`` is not
accepted. Ladder step 2 (hidden ``ResultsViewer.show(run_loop=False)``)
is gated on ADR 0094 open question 2 and is not implemented: if VTK
offscreen cannot get a GL context, figures skip.

``APEGMSH_SKIP_VIEWER=1`` or no GL returns ``None``, prints the
existing ``[skip viewer]`` notice, and writes no file.

``g.model.render`` / ``g.mesh.render`` are live-session only
(INV-5). They use the same scene builders as ModelViewer /
MeshViewer. A ``from_h5`` or closed session raises
``RuntimeError`` — they do not fake a BRep still from FEMData.

Windows (ADR 0128 slice 6, VA6b): ``fem.render`` takes its still from
apeGmshViewer through :func:`apeGmsh.agent_view.render_stills`, an
``apegmsh-view/1`` spec, after writing the in-memory ``FEMData`` to a
temporary ``model.h5`` (parity plan G7, R1). No app is a
``FileNotFoundError`` and a refused shot an ``AgentViewError`` (G9);
only ``APEGMSH_SKIP_VIEWER=1`` still skips. Linux and macOS keep the
VTK path (G2). ``_stills_backend`` is the one switch.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional, Union

# Still-shot machinery (plotter / theme / camera / screenshot + the
# GL-skip) lives in ``_stills`` since ADR 0098 S1 so the session still
# client shares it; imported back here so behaviour (and the names
# this module always had) are unchanged.
from ._stills import (
    _DEFAULT_WINDOW,
    _GL_EXC,  # noqa: F401  (re-exported for continuity)
    _SKIP_ENV,
    _SKIP_GL,  # noqa: F401
    _env_skips,
    _screenshot,  # noqa: F401  (tests/viewers/test_render.py reads it here)
    _shoot,
)

DeformArg = Union[float, tuple[str, float], None]

_VIEWS: frozenset[str] = frozenset({"mesh", "contour", "deformed", "reactions"})
_CAMERAS: frozenset[str] = frozenset({"iso", "xy", "xz", "yz"})

#: The v1 camera tokens as ``apegmsh-view/1`` cameras (parity plan
#: R4-R7): ``xy`` looks along -z, ``xz`` along +y, ``yz`` along -x.
#: ``iso`` becomes the app's own iso (R4: the look changes).
_APP_CAMERAS: dict[str, str] = {"iso": "iso", "xy": "top", "xz": "-y", "yz": "+x"}

#: The ``apegmsh-view/1`` version every spec built here names.
_SPEC_VERSION = "apegmsh-view/1"


def _stills_backend() -> str:
    """``"app"`` (apeGmshViewer, ``apegmsh-view/1``) on Windows, else ``"vtk"``.

    Parity plan G2 (ruled 2026-11-02): v2 cannot capture headless on
    Linux, so Linux and macOS keep the VTK path. The tests patch this
    function to reach either path on any platform.
    """
    return "app" if sys.platform == "win32" else "vtk"


def _app_camera(camera: str) -> str:
    return _APP_CAMERAS[_require_camera(camera)]


def _app_size(window_size: tuple[int, int]) -> list[int]:
    width, height = window_size
    return [int(width), int(height)]


def _shoot_app(
    spec: Mapping[str, Any], targets: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Take every shot of ``spec`` with apeGmshViewer; move each PNG to its target.

    ``targets`` maps a shot name to the path its PNG goes to. The app
    writes into a fresh temporary folder (it refuses one that is not
    empty), so a target folder that already holds files is fine. Returns
    the manifest and the written paths by shot name. Raises
    ``AgentViewError`` on any refusal (G9) and ``FileNotFoundError``
    without the app; nothing is moved then.
    """
    from apeGmsh.agent_view import render_stills

    names = [shot["name"] for shot in spec["shots"]]
    if sorted(names) != sorted(targets):
        raise ValueError(
            f"the spec's shots {names} and the still targets "
            f"{sorted(targets)} differ"
        )
    with tempfile.TemporaryDirectory(prefix="apegmsh-stills-") as tmp:
        out = Path(tmp) / "out"
        manifest = render_stills(spec, out)
        written: dict[str, Path] = {}
        for entry in manifest["shots"]:
            name = entry["name"]
            dest = Path(targets[name])
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(out / entry["png"]), str(dest))
            written[name] = dest
    missing = sorted(set(targets) - set(written))
    if missing:
        raise RuntimeError(
            f"apeGmshViewer reported success but its manifest has no "
            f"entry for the shots {missing}"
        )
    return manifest, written


def render_fem(
    fem: Any,
    path: "str | Path",
    *,
    camera: Optional[str] = None,
    window_size: tuple[int, int] = _DEFAULT_WINDOW,
) -> Optional[Path]:
    """Write one undeformed mesh still of ``fem``. Returns the Path or None.

    ``camera=`` defaults to ``xy`` for a planar model, ``iso`` otherwise
    (ADR 0094 Amendment 3); pass it explicitly to override.

    On Windows the still is apeGmshViewer's (module docstring): ``fem``
    is written to a temporary ``model.h5`` and drawn as one mesh shot.
    """
    camera = _resolve_camera(camera, fem.nodes.coords)
    if _env_skips():
        print(_SKIP_ENV)
        return None
    if _stills_backend() == "app":
        return _render_fem_app(fem, Path(path), camera=camera, window_size=window_size)
    from apeGmsh.viewers.scene.fem_scene import build_fem_scene

    scene = build_fem_scene(fem)
    return _shoot(
        path,
        camera=camera,
        window_size=window_size,
        scene=scene,
        add_substrate=True,
    )


def _app_mesh_spec(
    model_path: "str | Path",
    *,
    camera: str,
    window_size: tuple[int, int],
    name: str = "mesh",
) -> dict[str, Any]:
    """The ``apegmsh-view/1`` spec of one undeformed mesh shot of a model file (R1)."""
    return {
        "spec": _SPEC_VERSION,
        "files": {"model": str(Path(model_path).resolve())},
        "size": _app_size(window_size),
        "shots": [{"name": name, "camera": _app_camera(camera)}],
    }


def _render_fem_app(
    fem: Any,
    path: Path,
    *,
    camera: str,
    window_size: tuple[int, int],
) -> Path:
    """G7: an in-memory ``FEMData`` drawn through a temporary ``model.h5``."""
    with tempfile.TemporaryDirectory(prefix="apegmsh-fem-") as tmp:
        model = Path(tmp) / "model.h5"
        fem.to_h5(str(model))
        spec = _app_mesh_spec(model, camera=camera, window_size=window_size)
        _, written = _shoot_app(spec, {"mesh": path})
    return written["mesh"]


def render_model(
    session: Any,
    path: "str | Path",
    *,
    camera: str = "iso",
    window_size: tuple[int, int] = _DEFAULT_WINDOW,
) -> Optional[Path]:
    """Write one BRep still of the live session. Returns the Path or None.

    Uses ``build_brep_scene`` — the same tessellation ModelViewer
    already runs, including a throwaway coarse 2-D mesh when none
    exists. Raises ``RuntimeError`` on ``from_h5`` / closed sessions.
    """
    camera = _require_camera(camera)
    _require_live_kernel(session, "g.model.render")
    if _env_skips():
        print(_SKIP_ENV)
        return None
    from apeGmsh.viewers.scene.brep_scene import build_brep_scene

    return _shoot(
        path,
        camera=camera,
        window_size=window_size,
        populate=lambda plotter: build_brep_scene(plotter, [0, 1, 2, 3]),
    )


def render_mesh(
    session: Any,
    path: "str | Path",
    *,
    camera: str = "iso",
    window_size: tuple[int, int] = _DEFAULT_WINDOW,
) -> Optional[Path]:
    """Write one undeformed mesh still of the live session.

    Uses ``build_mesh_scene`` (the live-Gmsh scene MeshViewer uses),
    not ``get_fem_data`` + ``build_fem_scene``. Snapshot stills stay
    on ``fem.render``. Raises ``RuntimeError`` on ``from_h5`` /
    closed sessions, or when no mesh has been generated.
    """
    camera = _require_camera(camera)
    _require_live_kernel(session, "g.mesh.render")
    _require_live_mesh()
    if _env_skips():
        print(_SKIP_ENV)
        return None
    from apeGmsh.viewers.scene.mesh_scene import build_mesh_scene

    return _shoot(
        path,
        camera=camera,
        window_size=window_size,
        populate=lambda plotter: build_mesh_scene(plotter, [1, 2, 3]),
    )


# ------------------------------------------------------------------
# Validation / skip
# ------------------------------------------------------------------


def _require_live_kernel(session: Any, who: str) -> None:
    """INV-5: BRep / live-mesh stills need an open Gmsh kernel."""
    if getattr(session, "_active", False):
        return
    raise RuntimeError(
        f"{who} requires a live Gmsh kernel. "
        "from_h5 / closed sessions cannot emit CAD/BRep stills "
        "(INV-5). Use fem.render() for a mesh still from the snapshot."
    )


def _require_live_mesh() -> None:
    import gmsh

    tags, _, _ = gmsh.model.mesh.getNodes()
    if len(tags) == 0:
        raise RuntimeError(
            "g.mesh.render requires a generated mesh "
            "(call g.mesh.generation.generate first). "
            "For geometry stills use g.model.render."
        )


def _require_view(view: str) -> str:
    token = str(view)
    if token not in _VIEWS:
        raise ValueError(
            f"view must be one of {sorted(_VIEWS)}; got {view!r}."
        )
    return token


def _require_camera(camera: str) -> str:
    token = str(camera)
    if token not in _CAMERAS:
        raise ValueError(
            f"camera must be one of {sorted(_CAMERAS)}; got {camera!r}."
        )
    return token


def _resolve_camera(camera: Optional[str], coords: Any) -> str:
    """ADR 0094 Amendment 3: default camera for a planar model is
    ``xy`` (an isometric shot of a flat model is skewed for no
    reason); ``iso`` otherwise. Only used when ``camera=`` is omitted
    — an explicit value (including ``"iso"``) is honored unchanged.
    """
    if camera is None:
        from apeGmsh.results._geometry import is_planar

        camera = "xy" if is_planar(coords) else "iso"
    return _require_camera(camera)


def _require_deform(
    deform: DeformArg,
) -> Optional[tuple[str, Optional[float]]]:
    """Normalize ``deform=`` to ``(field, scale_or_None)``.

    ``None`` → no explicit warp (``view='deformed'`` still auto-scales).
    A number → ``('displacement', scale)``.
    A ``(field, scale)`` pair → as given.
    """
    if deform is None:
        return None
    if isinstance(deform, (int, float)) and not isinstance(deform, bool):
        return ("displacement", float(deform))
    if isinstance(deform, (tuple, list)) and len(deform) == 2:
        field, scale = deform[0], deform[1]
        return (str(field), float(scale))
    raise ValueError(
        "deform must be None, a number (displacement scale), "
        f"or a (field, scale) pair; got {deform!r}."
    )
