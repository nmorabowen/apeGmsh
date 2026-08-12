"""Offscreen stills from the viewer scene / diagram pipeline (ADR 0094 S1).

This module *writes stills* (PNG files). That is a different verb from
ADR 0042 ``RenderBackend.render()``, which pumps a live scene. Callers
hold a broker and want a Path::

    fem.render("mesh.png")
    results.render("uz.png", view="contour", component="displacement_z")

Stills come from ``pv.Plotter(off_screen=True)`` (VTK offscreen, not
Qt ``QT_QPA_PLATFORM=offscreen``) plus ``build_fem_scene`` /
``ResultsDirector`` / a registered diagram attached through
``PyVistaQtBackend``. Deform goes through ``director.geometries``
(ADR 0058). There is no ``QMainWindow``, no dock, and no
``app.exec_()`` / ``win.exec_()`` (INV-2).

``view=`` is a closed set (INV-10): ``mesh`` / ``contour`` /
``deformed`` / ``reactions``. ``setup(plotter, director)`` is not
accepted. Ladder step 2 (hidden ``ResultsViewer.show(run_loop=False)``)
is gated on ADR 0094 open question 2 and is not implemented: if VTK
offscreen cannot get a GL context, figures skip.

``APEGMSH_SKIP_VIEWER=1`` or no GL returns ``None``, prints the
``[skip viewer]`` notice, and writes no file.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union

DeformArg = Union[float, tuple[str, float], None]

_VIEWS: frozenset[str] = frozenset({"mesh", "contour", "deformed", "reactions"})
_CAMERAS: frozenset[str] = frozenset({"iso", "xy", "xz", "yz"})
_SKIP_ENV = "[skip viewer] APEGMSH_SKIP_VIEWER set"
_SKIP_GL = "[skip viewer] no GL context"
# stills.py default: largest |u| becomes this fraction of the model diagonal.
_DEFORM_FRACTION = 0.12
_DEFAULT_WINDOW = (1280, 720)


def render_fem(
    fem: Any,
    path: "str | Path",
    *,
    camera: str = "iso",
    window_size: tuple[int, int] = _DEFAULT_WINDOW,
) -> Optional[Path]:
    """Write one undeformed mesh still of ``fem``. Returns the Path or None."""
    camera = _require_camera(camera)
    if _env_skips():
        print(_SKIP_ENV)
        return None
    from apeGmsh.viewers.scene.fem_scene import build_fem_scene

    scene = build_fem_scene(fem)
    return _shoot(
        path,
        camera=camera,
        window_size=window_size,
        scene=scene,
        add_substrate=True,
    )


def render_results(
    results: Any,
    path: "str | Path",
    *,
    view: str = "contour",
    component: Optional[str] = None,
    step: int = -1,
    deform: DeformArg = None,
    camera: str = "iso",
    window_size: tuple[int, int] = _DEFAULT_WINDOW,
) -> Optional[Path]:
    """Write one still of ``results``. Returns the Path or None."""
    view = _require_view(view)
    camera = _require_camera(camera)
    deform_spec = _require_deform(deform)
    if _env_skips():
        print(_SKIP_ENV)
        return None
    if results.fem is None:
        raise RuntimeError(
            "results.render requires a bound FEMData "
            "(construct with model= / model_h5= or call results.bind)."
        )

    from apeGmsh.viewers.backends import PyVistaQtBackend
    from apeGmsh.viewers.diagrams import DiagramSpec, ResultsDirector
    from apeGmsh.viewers.diagrams._kinds import kind_def
    from apeGmsh.viewers.diagrams._selectors import (
        normalize as normalize_selector,
    )
    from apeGmsh.viewers.diagrams._starter import default_contour_component
    from apeGmsh.viewers.scene.fem_scene import build_fem_scene

    scene = build_fem_scene(results.fem)
    director = ResultsDirector(results)
    _ensure_stage(director)
    step_i = _resolve_step(director, step)

    wants_diagram = view in ("contour", "deformed", "reactions")
    wants_deform = view == "deformed" or (
        view != "mesh" and deform_spec is not None
    )
    # Full-mesh contour occupies the substrate — a second fill z-fights.
    add_substrate = view in ("mesh", "reactions")

    plotter = None
    try:
        plotter = _open_plotter(window_size)
        if plotter is None:
            return None
        _apply_theme(plotter)
        backend = PyVistaQtBackend(plotter)
        director.bind_plotter(
            backend, scene=scene, render_callback=plotter.render,
        )
        if add_substrate:
            _add_substrate(plotter, scene)
        if wants_diagram:
            kind = "reactions" if view == "reactions" else "contour"
            if kind == "contour":
                resolved = component or default_contour_component(director)
                if resolved is None:
                    raise ValueError(
                        "view='contour'/'deformed' needs a nodal component; "
                        "none is recorded. Pass component= explicitly."
                    )
                component = resolved
            else:
                component = component or "reactions"
            kdef = kind_def(kind)
            spec = DiagramSpec(
                kind=kind,
                selector=normalize_selector(component=component),
                style=kdef.make_default_style(component),
            )
            director.registry.add(kdef.diagram_class(spec, director.results))
        if step_i != director.step_index:
            director.set_step(step_i)
        elif director.registry.diagrams():
            director.registry.update_to_step(step_i)
        if wants_deform:
            _apply_deform(director, scene, deform_spec, step_i)
        _apply_camera(plotter, camera)
        return _screenshot(plotter, path)
    finally:
        director.unbind_plotter()
        if plotter is not None:
            try:
                plotter.close()
            except Exception:
                pass


# ------------------------------------------------------------------
# Validation / skip
# ------------------------------------------------------------------


def _env_skips() -> bool:
    return bool(os.environ.get("APEGMSH_SKIP_VIEWER"))


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


# ------------------------------------------------------------------
# Plotter / theme / camera
# ------------------------------------------------------------------


def _open_plotter(window_size: tuple[int, int]) -> Any:
    try:
        import pyvista as pv
        return pv.Plotter(off_screen=True, window_size=list(window_size))
    except Exception:
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
    try:
        plotter.screenshot(str(out))
    except Exception:
        if out.exists():
            try:
                out.unlink()
            except OSError:
                pass
        print(_SKIP_GL)
        return None
    if not out.exists() or out.stat().st_size == 0:
        if out.exists():
            try:
                out.unlink()
            except OSError:
                pass
        print(_SKIP_GL)
        return None
    return out


def _shoot(
    path: "str | Path",
    *,
    camera: str,
    window_size: tuple[int, int],
    scene: Any,
    add_substrate: bool,
) -> Optional[Path]:
    plotter = None
    try:
        plotter = _open_plotter(window_size)
        if plotter is None:
            return None
        _apply_theme(plotter)
        if add_substrate:
            _add_substrate(plotter, scene)
        _apply_camera(plotter, camera)
        return _screenshot(plotter, path)
    finally:
        if plotter is not None:
            try:
                plotter.close()
            except Exception:
                pass


# ------------------------------------------------------------------
# Director helpers
# ------------------------------------------------------------------


def _ensure_stage(director: Any) -> None:
    if director.stage_id is not None:
        return
    stages = list(director.results.stages)
    if not stages:
        raise RuntimeError("results.render requires at least one stage.")
    director.set_stage(stages[-1].id)


def _resolve_step(director: Any, step: int) -> int:
    n = int(director.n_steps)
    if n <= 0:
        return 0
    if step < 0:
        return max(0, n + int(step))
    return min(int(step), n - 1)


def _apply_deform(
    director: Any,
    scene: Any,
    deform_spec: Optional[tuple[str, Optional[float]]],
    step: int,
) -> None:
    """Warp via ``director.geometries`` (ADR 0058), not a point-loop."""
    from apeGmsh.viewers._pump_set import _compose_substrate_points

    if deform_spec is None:
        field, scale = "displacement", None
    else:
        field, scale = deform_spec
    vals = _read_deform_field(director, scene, field, step)
    if vals is None:
        return
    if scale is None:
        max_d = float(abs(vals).max())
        diag = float(scene.model_diagonal)
        scale = (
            _DEFORM_FRACTION * diag / max_d
            if max_d > 0.0 and diag > 0.0
            else 1.0
        )
    geoms = director.geometries
    active = geoms.active or (
        geoms.geometries[0] if geoms.geometries else None
    )
    if active is None:
        return
    geoms.set_deformation(
        active.id, enabled=True, field=field, scale=float(scale),
    )
    pts = _compose_substrate_points(
        scene.reference_points,
        getattr(active, "offset", None),
        vals,
        float(active.deform_scale),
    )
    if pts is None:
        return
    scene.grid.points = pts
    for diagram in director.registry.diagrams():
        if diagram.is_attached:
            diagram.sync_substrate_points(pts, scene)


def _read_deform_field(
    director: Any, scene: Any, field: str, step: int,
) -> Any:
    import numpy as np

    try:
        scoped = director.results
        if director.stage_id is not None:
            scoped = director.results.stage(director.stage_id)
    except Exception:
        return None
    n = int(scene.node_ids.size)
    out = np.zeros((n, 3), dtype=np.float64)
    id_to_idx = scene.node_id_to_idx
    any_axis = False
    for axis, suf in enumerate(("x", "y", "z")):
        try:
            slab = scoped.nodes.get(
                ids=scene.node_ids,
                component=f"{field}_{suf}",
                time=[int(step)],
            )
        except Exception:
            continue
        if slab.values.size == 0:
            continue
        vals = np.asarray(slab.values[0], dtype=np.float64)
        for nid, v in zip(np.asarray(slab.node_ids), vals):
            idx = id_to_idx.get(int(nid))
            if idx is not None:
                out[idx, axis] = float(v)
                any_axis = True
    return out if any_axis else None
