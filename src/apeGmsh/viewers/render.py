"""Offscreen stills from the viewer scene / diagram pipeline (ADR 0094 S1).

This module *writes stills* (PNG files). That is a different verb from
ADR 0042 ``RenderBackend.render()``, which pumps a live scene. Callers
hold a broker and want a Path::

    fem.render("mesh.png")
    results.render("uz.png", view="contour", component="displacement_z")
    results.render_pack("out_dir/")
    g.model.render("geom.png")
    g.mesh.render("mesh.png")

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

``APEGMSH_SKIP_VIEWER=1`` or no GL returns ``None`` / ``()``, prints
the existing ``[skip viewer]`` notice, and writes no file.
``render_pack`` is the canned report set, not a poster.

``g.model.render`` / ``g.mesh.render`` are live-session only
(INV-5). They use the same scene builders as ModelViewer /
MeshViewer. A ``from_h5`` or closed session raises
``RuntimeError`` — they do not fake a BRep still from FEMData.
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
# VTK/pyvista raise these when the offscreen window cannot start.
# TypeError / ValueError / AttributeError are bugs and must propagate.
_GL_EXC = (RuntimeError, OSError)
# stills.py default: largest |u| becomes this fraction of the model diagonal.
_DEFORM_FRACTION = 0.12
_DEFAULT_WINDOW = (1280, 720)
_HISTORY_LABEL = "[matplotlib]"


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
    if results.fem is None:
        raise RuntimeError(
            "results.render requires a bound FEMData "
            "(construct with model= / model_h5= or call results.bind)."
        )
    if _env_skips():
        print(_SKIP_ENV)
        return None

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
    if wants_deform:
        _require_deform_field(director, scene, deform_spec, step_i)
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


def render_pack(
    results: Any,
    out_dir: "str | Path",
    *,
    camera: str = "iso",
    window_size: tuple[int, int] = _DEFAULT_WINDOW,
) -> tuple[Path, ...]:
    """Write the canned report pack. Returns written paths, or ``()``.

    1. ``mesh.png`` — undeformed mesh
    2. ``contour.png`` — last-step primary field (S1
       ``default_contour_component``)
    3. ``deformed.png`` — same, auto-scale 0.12 via
       ``director.geometries``
    4. ``reactions.png`` — static stages only, if reactions recorded
    5. ``history.png`` — optional matplotlib history at the extrema node

    Skip / no GL returns ``()`` and prints the existing
    ``[skip viewer]`` notice. There is no ``fem.render_pack``.
    """
    camera = _require_camera(camera)
    if results.fem is None:
        raise RuntimeError(
            "results.render_pack requires a bound FEMData "
            "(construct with model= / model_h5= or call results.bind)."
        )
    if _env_skips():
        print(_SKIP_ENV)
        return ()

    dest = Path(out_dir)
    written: list[Path] = []

    mesh = render_results(
        results, dest / "mesh.png",
        view="mesh", camera=camera, window_size=window_size,
    )
    if mesh is None:
        return ()
    written.append(mesh)

    try:
        component = _pack_primary_component(results)
    except (RuntimeError, ValueError):
        component = None
    if component is not None:
        contour = render_results(
            results, dest / "contour.png",
            view="contour", component=component, step=-1,
            camera=camera, window_size=window_size,
        )
        if contour is not None:
            written.append(contour)
        try:
            deformed = render_results(
                results, dest / "deformed.png",
                view="deformed", component=component, step=-1, deform=None,
                camera=camera, window_size=window_size,
            )
        except ValueError:
            # B5: no recorded displacement — omit, do not fail the pack.
            deformed = None
        if deformed is not None:
            written.append(deformed)

    if _has_static_reactions(results):
        reactions = render_results(
            results, dest / "reactions.png",
            view="reactions", component="reactions", step=-1,
            camera=camera, window_size=window_size,
        )
        if reactions is not None:
            written.append(reactions)

    history = _try_history(results, dest / "history.png", component)
    if history is not None:
        written.append(history)
    return tuple(written)


# ------------------------------------------------------------------
# Validation / skip
# ------------------------------------------------------------------


def _env_skips() -> bool:
    return bool(os.environ.get("APEGMSH_SKIP_VIEWER"))


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
        raise ValueError("results.render has no steps to index.")
    idx = int(step)
    if idx < 0:
        idx = n + idx
    if idx < 0 or idx >= n:
        raise ValueError(
            f"step={step!r} is out of range for {n} step(s) "
            f"(valid 0..{n - 1}, or Python negatives)."
        )
    return idx


def _require_deform_field(
    director: Any,
    scene: Any,
    deform_spec: Optional[tuple[str, Optional[float]]],
    step: int,
) -> Any:
    """Return the warp field or raise (do not write an undeformed still)."""
    field = "displacement" if deform_spec is None else deform_spec[0]
    vals = _read_deform_field(director, scene, field, step)
    if vals is None:
        raise ValueError(
            f"view='deformed' needs a recorded {field}_* field; "
            "none is present."
        )
    return vals


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
    vals = _require_deform_field(director, scene, deform_spec, step)
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
    from apeGmsh.viewers._pump_set import read_nodal_vector_field

    try:
        scoped = director.results
        if director.stage_id is not None:
            scoped = director.results.stage(director.stage_id)
    except Exception:
        return None
    return read_nodal_vector_field(scoped, scene, field, step)


# ------------------------------------------------------------------
# Pack helpers
# ------------------------------------------------------------------


def _pack_primary_component(results: Any) -> Optional[str]:
    """S1 ``default_contour_component`` (uz, ux, uy, first nodal)."""
    from apeGmsh.viewers.diagrams import ResultsDirector
    from apeGmsh.viewers.diagrams._starter import default_contour_component

    director = ResultsDirector(results)
    _ensure_stage(director)
    return default_contour_component(director)


def _has_static_reactions(results: Any) -> bool:
    """True when a ``kind='static'`` stage recorded a reaction_* field."""
    static = [
        s for s in results.stages if getattr(s, "kind", None) == "static"
    ]
    if not static:
        return False
    for stage in static:
        try:
            names = results.inspect.components(stage=stage.id).get("nodes", [])
        except (RuntimeError, ValueError):
            continue
        for name in names:
            if name == "reactions" or str(name).startswith("reaction_"):
                return True
    return False


def _extrema_node(results: Any, component: str) -> Optional[int]:
    """Node id of max |component| at ``time=-1``. INV-9: last step only."""
    import numpy as np

    try:
        slab = results.nodes.get(component=component, time=-1)
    except Exception:
        return None
    values = np.asarray(slab.values)
    ids = np.asarray(slab.node_ids)
    if values.size == 0 or ids.size == 0:
        return None
    row = values[-1] if values.ndim >= 2 else values
    row = np.asarray(row, dtype=np.float64).reshape(-1)
    if row.size != ids.size:
        return None
    finite = np.isfinite(row)
    if not finite.any():
        return None
    masked = np.where(finite, np.abs(row), -1.0)
    return int(ids[int(np.argmax(masked))])


def _try_history(
    results: Any, path: "str | Path", component: Optional[str],
) -> Optional[Path]:
    """Matplotlib history at the extrema node. Labeled (INV-7)."""
    if component is None:
        return None
    node = _extrema_node(results, component)
    if node is None:
        return None
    try:
        import matplotlib
        try:
            matplotlib.use("Agg")
        except Exception:
            pass
        import matplotlib.pyplot as plt
    except Exception:
        return None
    out = Path(path)
    try:
        ax = results.plot.history(node=node, component=component)
        series_len = 0
        if ax.lines:
            series_len = len(ax.lines[0].get_xdata())
        if series_len < 2:
            plt.close(ax.figure)
            return None
        ax.set_title(
            f"{_HISTORY_LABEL} history at node {node} ({component})"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(out)
        plt.close(ax.figure)
    except Exception:
        return None
    if not out.exists() or out.stat().st_size == 0:
        return None
    return out
