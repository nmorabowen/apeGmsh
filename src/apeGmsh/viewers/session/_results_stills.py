"""``results.render`` / ``results.render_pack`` on the session path (X2-d).

The public still verbs of ADR 0094 keep their signatures and their
closed ``view=`` set; what changed is the picture source. Each
``view=`` token is translated into ONE ``ResultsSession`` mesh view
(slots + pose + style buttons + instant) and drawn by
``ResultsSession.render`` (``_stills.render_still``) — no
``ResultsDirector``, no ``DiagramSpec`` registry, no attach-then-sync.
The session path is reused as it is (ADR 0112 D8): nothing here adds a
session feature.

``view=`` → session view:

| view | slots | pose | style buttons |
|---|---|---|---|
| ``mesh`` | none | none | mesh on, outlines off |
| ``contour`` | ``Contour(component)`` | ``deform=`` if given | both off |
| ``deformed`` | ``Contour(component)`` | ``Deform(field, scale)`` (auto-fit when no scale) | both off |
| ``reactions`` | ``Reactions()`` | ``deform=`` if given | mesh on, outlines off |

The style buttons are the old still's picture, not the session's boot
picture: the old ``view='mesh'`` drew the raw mesh with element edges
and no feature-edge pass, and the old contour drew the bare contour
(``test_session_stills.py`` pins both parities).

The session is built with ``ResultsSession(results=...)`` +
``add_view()`` rather than ``results.session()``: the latter boots
persisted section cuts as view clips, which ``results.render`` never
drew.

The pack helpers (``_has_static_reactions``, ``_extrema_node``,
``_try_history``) are the ``viewers/render.py`` ones carried here so
this path owns them; the originals stay until X3 deletes the director
path.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from .._stills import _DEFAULT_WINDOW, _SKIP_ENV, _env_skips
from ..render import (
    DeformArg,
    _require_deform,
    _require_view,
    _resolve_camera,
)

if TYPE_CHECKING:
    from apeGmsh.results.session import ResultsSession

_HISTORY_LABEL = "[matplotlib]"

#: Reaction tokens that the old reactions diagram read as ONE axis
#: (``_reactions._AXIS_FROM_COMPONENT``). The session ``Reactions`` slot
#: carries nothing (ADR 0098 §4), so it draws every axis; a per-axis
#: still would be a new session feature, so it is refused, not widened.
_PER_AXIS_REACTIONS = frozenset({"reaction_x", "reaction_y", "reaction_z"})

#: The fields a session ``Deform`` pose warps by (``results.session``'s
#: ``Deform.field``). The old path warped by any recorded ``{field}_*``.
_DEFORM_FIELDS = frozenset({"displacement", "velocity", "acceleration"})


# =====================================================================
# Public verbs (called by Results.render / Results.render_pack)
# =====================================================================


def render_results_still(
    results: Any,
    path: "str | Path",
    *,
    view: str = "contour",
    component: Optional[str] = None,
    step: int = -1,
    deform: DeformArg = None,
    camera: Optional[str] = None,
    window_size: tuple[int, int] = _DEFAULT_WINDOW,
) -> Optional[Path]:
    """Write one still of ``results``. Returns the Path or ``None``.

    ``None`` (and the ``[skip viewer]`` notice) only under
    ``APEGMSH_SKIP_VIEWER=1`` or with no GL context.
    """
    view = _require_view(view)
    deform_spec = _require_deform(deform)
    _refuse_unsupported(view, component, deform_spec)
    if results.fem is None:
        raise RuntimeError(
            "results.render requires a bound FEMData "
            "(construct with model= / model_h5= or call results.bind)."
        )
    camera = _resolve_camera(camera, results.fem.nodes.coords)
    if _env_skips():
        print(_SKIP_ENV)
        return None
    session = session_for_view(
        results, view=view, component=component, step=step,
        deform_spec=deform_spec,
    )
    return session.render(path, camera=camera, window_size=window_size)


def render_results_pack(
    results: Any,
    out_dir: "str | Path",
    *,
    camera: Optional[str] = None,
    window_size: tuple[int, int] = _DEFAULT_WINDOW,
) -> tuple[Path, ...]:
    """Write the canned report pack. Returns written paths, or ``()``.

    Same file set and order as ADR 0094 S3: ``mesh.png``,
    ``contour.png`` + ``deformed.png`` (default component, last step),
    ``reactions.png`` (static stages with reactions), ``history.png``
    (matplotlib, at the extrema node).
    """
    if results.fem is None:
        raise RuntimeError(
            "results.render_pack requires a bound FEMData "
            "(construct with model= / model_h5= or call results.bind)."
        )
    camera = _resolve_camera(camera, results.fem.nodes.coords)
    if _env_skips():
        print(_SKIP_ENV)
        return ()

    dest = Path(out_dir)
    written: list[Path] = []

    mesh = render_results_still(
        results, dest / "mesh.png",
        view="mesh", camera=camera, window_size=window_size,
    )
    if mesh is None:
        return ()
    written.append(mesh)

    try:
        component = default_contour_component(results)
    except (RuntimeError, ValueError):
        component = None
    if component is not None:
        contour = render_results_still(
            results, dest / "contour.png",
            view="contour", component=component, step=-1,
            camera=camera, window_size=window_size,
        )
        if contour is not None:
            written.append(contour)
        try:
            deformed = render_results_still(
                results, dest / "deformed.png",
                view="deformed", component=component, step=-1, deform=None,
                camera=camera, window_size=window_size,
            )
        except ValueError:
            # No recorded displacement: omit, do not fail the pack.
            deformed = None
        if deformed is not None:
            written.append(deformed)

    if _has_static_reactions(results):
        reactions = render_results_still(
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


# =====================================================================
# view= token -> one session mesh view
# =====================================================================


def session_for_view(
    results: Any,
    *,
    view: str,
    component: Optional[str],
    step: int,
    deform_spec: Optional[tuple[str, Optional[float]]],
) -> "ResultsSession":
    """The one-pane session that paints ``view`` (table in the module
    docstring). Raises the old path's errors: no stage, a step out of
    range, a missing deform field, no nodal component to contour."""
    from apeGmsh.results.session import (
        Contour,
        Deform,
        Instant,
        MeshStyle,
        Reactions,
        ResultsSession,
    )

    stage_id = _resolve_stage(results)
    step_i = _resolve_step(results, stage_id, step)

    wants_deform = view == "deformed" or (
        view != "mesh" and deform_spec is not None
    )
    pose = None
    if wants_deform:
        field, scale = deform_spec if deform_spec is not None else (
            "displacement", None,
        )
        _require_deform_field(results, stage_id, field)
        pose = Deform(field=field, scale=scale)

    session = ResultsSession(results=results)
    pane = session.add_view()
    session.time = Instant(stage=stage_id, step=step_i)

    if view in ("contour", "deformed"):
        quantity = component or default_contour_component(results)
        if quantity is None:
            raise ValueError(
                "view='contour'/'deformed' needs a nodal component; "
                "none is recorded. Pass component= explicitly."
            )
        pane.style = MeshStyle(mesh=False, outlines=False)
        pane.contour = Contour(quantity)
    elif view == "reactions":
        pane.style = MeshStyle(mesh=True, outlines=False)
        pane.reactions = Reactions()
    else:  # mesh
        pane.style = MeshStyle(mesh=True, outlines=False)
    if pose is not None:
        pane.deform = pose
    return session


def _refuse_unsupported(
    view: str,
    component: Optional[str],
    deform_spec: Optional[tuple[str, Optional[float]]],
) -> None:
    """The two old-path stills the session path does not draw (maintainer
    ruling 2026-10-11, #1665). Refused before the skip check, so the
    answer does not depend on whether a GL context exists."""
    if view == "reactions" and component in _PER_AXIS_REACTIONS:
        raise ValueError(
            f"view='reactions' draws every reaction axis; a one-axis "
            f"still (component={component!r}) is not supported. Use "
            f"component='reactions' (or None) for the reaction arrows, "
            f"and results.nodes.get(component='reaction_force_"
            f"{component[-1]}') for that axis's values."
        )
    if (
        deform_spec is not None
        and view != "mesh"
        and deform_spec[0] not in _DEFORM_FIELDS
    ):
        raise ValueError(
            f"deform= warps by displacement, velocity or acceleration; "
            f"field {deform_spec[0]!r} is not supported. Use "
            f"deform=SCALE (displacement) or deform=(FIELD, SCALE) with "
            f"FIELD one of {sorted(_DEFORM_FIELDS)}."
        )


def default_contour_component(results: Any) -> Optional[str]:
    """The S1 starter heuristic (uz, ux, uy, then the first nodal
    component in canonical-axis order), over every stage's nodal
    components. ``None`` when nothing nodal is recorded."""
    nodal: set[str] = set()
    for stage in results.stages:
        try:
            nodal.update(results.stage(stage.id).nodes.available_components())
        except Exception:
            continue
    if not nodal:
        return None
    for comp in ("displacement_z", "displacement_x", "displacement_y"):
        if comp in nodal:
            return comp
    return sorted(nodal, key=_component_sort_key)[0]


#: Canonical axis order of the S1 starter: the diagram catalog's sort,
#: restated here. Importing it would pull in the retired
#: ``viewers.diagrams`` ontology (``test_diagrams_import_guard.py``), and
#: importing ``opensees._response_catalog.split_canonical_component``
#: would break ADR 0014 (``test_viewers_pure_h5_consumer.py``). The keys
#: are exactly that splitter's suffixes, tensor indices first so
#: ``stress_xz`` does not split as ``z``.
_AXIS_ORDER: dict[str, int] = {
    "xx": 10, "yy": 11, "zz": 12, "xy": 13, "yz": 14, "xz": 15,
    "x": 0, "y": 1, "z": 2,
}


def _component_sort_key(name: str) -> tuple[int, str]:
    for suffix, rank in _AXIS_ORDER.items():
        if name.endswith("_" + suffix):
            return (rank, name)
    return (999, name)


def _resolve_stage(results: Any) -> str:
    """The last stage (the only one, for a single-stage file)."""
    stages = list(results.stages)
    if not stages:
        raise RuntimeError("results.render requires at least one stage.")
    return str(stages[-1].id)


def _resolve_step(results: Any, stage_id: str, step: int) -> int:
    n = int(results.stage(stage_id).n_steps)
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


def _require_deform_field(results: Any, stage_id: str, field: str) -> None:
    """Refuse before drawing: never write an undeformed 'deformed' still."""
    try:
        recorded = set(results.stage(stage_id).nodes.available_components())
    except Exception:
        recorded = set()
    if not any(f"{field}_{axis}" in recorded for axis in ("x", "y", "z")):
        raise ValueError(
            f"view='deformed' needs a recorded {field}_* field; "
            "none is present."
        )


# =====================================================================
# Pack helpers (carried from viewers/render.py; no director)
# =====================================================================


def _has_static_reactions(results: Any) -> bool:
    """True when a ``kind='static'`` stage recorded a reaction_* field."""
    static = [
        s for s in results.stages if getattr(s, "kind", None) == "static"
    ]
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
    """Node id of max |component| at ``time=-1`` (last step only)."""
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
    """Matplotlib history at the extrema node, labeled ``[matplotlib]``."""
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


__all__ = [
    "default_contour_component",
    "render_results_pack",
    "render_results_still",
    "session_for_view",
]
