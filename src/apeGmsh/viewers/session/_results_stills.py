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
(the X2-a image goldens, ``tests/viewers/test_x2_golden_stills.py``,
and the X2-e numeric oracle, ``tests/viewers/test_x2_numeric_oracle.py``,
pin the picture and the drawn data).

A contour of a component the stage cannot produce raises
:class:`ContourDataError`: a ``RuntimeError`` like the old path's
``NoDataError``, and also a ``ValueError`` like the session's refusal.

The session is built with ``ResultsSession(results=...)`` +
``add_view()`` rather than ``results.session()``: the latter boots
persisted section cuts as view clips, which ``results.render`` never
drew.

The pack helpers (``_has_static_reactions``, ``_extrema_node``,
``_try_history``) are the ``viewers/render.py`` ones carried here so
this path owns them; the originals stay until X3 deletes the director
path.

``app_still`` / ``app_pack`` (VA6b) are the Windows path of the same
verbs: each ``view=`` token becomes one ``apegmsh-view/1`` shot drawn
by apeGmshViewer through ``apeGmsh.agent_view`` (the parity plan,
``internal_docs/plan_viewer_agent_stills_parity.md``), from a
``.ladruno`` and its ``<stem>.h5`` (``ladruno_files``). They take the
files explicitly: the CLI holds them, and ``render_results_still`` /
``render_results_pack`` take them from ``Results.path``
(``results_files``, R22b), so ``results.render``, ``render_pack`` and
``assess(figures=True)`` reach the app on Windows. Every source but a
``.ladruno`` is refused by name there. Linux keeps the session path (G2).
A stored Gauss component (R16b) is refused for now: follow-up slice
#2794.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from .._stills import _DEFAULT_WINDOW, _SKIP_ENV, _env_skips
from ..render import (
    _SPEC_VERSION,
    DeformArg,
    _app_camera,
    _app_size,
    _require_deform,
    _require_view,
    _resolve_camera,
    _shoot_app,
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
    _warn_unscoped: bool = True,
) -> Optional[Path]:
    """Write one still of ``results``. Returns the Path or ``None``.

    ``None`` (and the ``[skip viewer]`` notice) only under
    ``APEGMSH_SKIP_VIEWER=1`` or with no GL context. An unscoped
    multi-stage Results draws its last stage and warns (#1750);
    ``_warn_unscoped=False`` is the pack's, which warned once already.

    On Windows the still is apeGmshViewer's (:func:`app_still`, from
    the ``.ladruno`` :attr:`Results.path` names; R22b).
    """
    if _app_backend():
        return app_still(
            results, path, files=results_files(results, "render"),
            view=view, component=component, step=step, deform=deform,
            camera=camera, window_size=window_size,
        )
    if _warn_unscoped:
        _warn_if_unscoped(results, "render")
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

    An unscoped multi-stage Results packs its last stage and warns
    once (#1750).

    On Windows the pack is ONE ``apegmsh-view/1`` spec drawn by
    apeGmshViewer (:func:`app_pack`, G5), from the ``.ladruno``
    :attr:`Results.path` names (R22b).
    """
    if _app_backend():
        return app_pack(
            results, out_dir, files=results_files(results, "render_pack"),
            camera=camera, window_size=window_size,
        )
    _warn_if_unscoped(results, "render_pack")
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
        _warn_unscoped=False,
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
            _warn_unscoped=False,
        )
        if contour is not None:
            written.append(contour)
        try:
            deformed = render_results_still(
                results, dest / "deformed.png",
                view="deformed", component=component, step=-1, deform=None,
                camera=camera, window_size=window_size,
                _warn_unscoped=False,
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
            _warn_unscoped=False,
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
        _require_contour_data(results, stage_id, quantity)
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
    components (the scoped stage's only, on a stage-scoped Results).
    ``None`` when nothing nodal is recorded."""
    nodal: set[str] = set()
    for stage in _pack_stages(results):
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
    """The stage a still draws (X2-g, #1742).

    Mirrors ``Results._resolve_stage`` (``export_animation``): the
    ``results.stage(s)`` / ``results.modes[i]`` scope first, then the
    only stage. Where ``Results._resolve_stage`` raises (several stages,
    none scoped), a still keeps its documented unscoped default: the
    LAST stage. A scope that names no stage in the file raises; a still
    never silently draws another stage.
    """
    stages = list(results.stages)
    if not stages:
        raise RuntimeError("results.render requires at least one stage.")
    scoped = getattr(results, "_stage_id", None)
    if scoped is not None:
        if not any(str(s.id) == str(scoped) for s in stages):
            raise RuntimeError(
                f"results.render: the scoped stage {scoped!r} is not in "
                f"this results file (stages: {[s.id for s in stages]})."
            )
        return str(scoped)
    return str(stages[-1].id)


def _warn_if_unscoped(results: Any, verb: str) -> None:
    """Warn when a still falls back to the last of several stages
    (maintainer ruling on #1750): the picture is one stage of many, and
    the caller never said which."""
    if getattr(results, "_stage_id", None) is not None:
        return
    stages = list(results.stages)
    if len(stages) <= 1:
        return
    last = stages[-1]
    name = getattr(last, "name", None) or last.id
    warnings.warn(
        f"results.{verb}() on a Results with {len(stages)} stages and "
        f"none selected draws the LAST stage, {name!r}. Pick one with "
        f"results.stage({name!r}).{verb}(...) (any stage name or id).",
        UserWarning,
        stacklevel=4,
    )


def _pack_stages(results: Any) -> list[Any]:
    """The stages the pack's choices look at: the scoped stage alone on
    a stage-scoped Results (so the default component and the reactions
    still are that stage's), else every stage (unscoped, unchanged)."""
    scoped = getattr(results, "_stage_id", None)
    stages = list(results.stages)
    if scoped is None:
        return stages
    return [s for s in stages if str(s.id) == str(scoped)]


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


class ContourDataError(RuntimeError, ValueError):
    """No data for the requested contour component.

    The old path raised ``viewers.diagrams.NoDataError``, a
    ``RuntimeError``, from the contour's attach. The session path
    refuses earlier with a ``ValueError`` (``_specs``). This type is
    both, so ``except RuntimeError`` and ``except ValueError`` callers
    keep working. This module cannot subclass ``NoDataError`` itself:
    that would import the retired ``viewers.diagrams`` ontology
    (``test_diagrams_import_guard.py``).
    """


def _require_contour_data(results: Any, stage_id: str, quantity: str) -> None:
    from ._specs import resolve_contour_topology

    try:
        resolve_contour_topology(quantity, results, stage_id)
    except ValueError as exc:
        raise ContourDataError(str(exc)) from exc


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
    """True when a ``kind='static'`` stage recorded a reaction_* field
    (the scoped stage, on a stage-scoped Results)."""
    static = [
        s for s in _pack_stages(results)
        if getattr(s, "kind", None) == "static"
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


# =====================================================================
# apegmsh-view/1 (Windows; ADR 0128 slice 6, VA6b)
# =====================================================================
#
# The parity plan (internal_docs/plan_viewer_agent_stills_parity.md)
# maps each view= token to one apegmsh-view/1 shot. A caller that holds
# the files (the CLI) builds the spec here and draws it with
# viewers.render._shoot_app; Linux keeps the session path above (G2).

#: ``<stem>.part-<N>.ladruno``, a partitioned run's part file (the app's
#: ``main/pairing.ts`` PART_FILE rule).
_LADRUNO_PART = re.compile(r"\.part-\d+\.ladruno$", re.IGNORECASE)
_LADRUNO_SUFFIX = ".ladruno"

#: A Python nodal prefix as the ``.ladruno`` ON_NODES group the app
#: names its field after (fork schema ``ladruno_schema_v1.md`` §7.1.1,
#: §7.1.2; the reader's ``readers/_buckets._NODAL_RESULT_NAME_MAP``
#: read backwards, the static group first as ``_NODAL_PRECEDENCE``
#: orders it). A mode stage records ``MODES_OF_VIBRATION(U|R)`` instead.
_APP_NODAL_FIELD: dict[str, str] = {
    "displacement": "DISPLACEMENT",
    "rotation": "ROTATION",
    "velocity": "VELOCITY",
    "angular_velocity": "ANGULAR_VELOCITY",
    "acceleration": "ACCELERATION",
    "angular_acceleration": "ANGULAR_ACCELERATION",
    "reaction_force": "REACTION_FORCE",
    "reaction_moment": "REACTION_MOMENT",
}
_APP_MODE_FIELD: dict[str, str] = {
    "displacement": "MODES_OF_VIBRATION(U)",
    "rotation": "MODES_OF_VIBRATION(R)",
}
_APP_AXES = ("x", "y", "z")


def ladruno_files(
    results_path: "str | Path", model_h5: "str | Path | None" = None,
) -> dict[str, str]:
    """The spec's ``files`` for a ``.ladruno`` results file (R22b, R31, G8).

    The viewer and v2 stills read ``.ladruno`` only (ruled 2026-11-02 on
    #2655): any other results file is refused by name. The app opens a
    ``.ladruno`` as the set of its stem, so the model is
    ``<stem>.h5`` beside it; a ``model_h5`` that is another file is
    refused rather than silently replaced.
    """
    path = Path(results_path)
    part = _LADRUNO_PART.search(path.name)
    if part is not None:
        stem = path.name[: part.start()]
    elif path.name.lower().endswith(_LADRUNO_SUFFIX) and len(path.name) > len(_LADRUNO_SUFFIX):
        stem = path.name[: -len(_LADRUNO_SUFFIX)]
    else:
        raise ValueError(
            f"apeGmshViewer stills read .ladruno results only (ADR 0122 D1; "
            f"the #2655 ruling); {path.name} is not a .ladruno. Write the "
            f"run's results as <stem>.ladruno beside its <stem>.h5 model."
        )
    model = path.parent / f"{stem}.h5"
    if model_h5 is not None and Path(model_h5).resolve() != model.resolve():
        raise ValueError(
            f"apeGmshViewer opens {path.name} with its own model {model.name} "
            f"(the stem rule); model_h5={str(model_h5)!r} is another file. "
            f"Put the model beside the results as {model.name}."
        )
    if not model.is_file():
        raise FileNotFoundError(
            f"apeGmshViewer opens {path.name} with the model {model} beside "
            f"it, which does not exist. Write the model there "
            f"(ops.h5({model.name!r}) or fem.to_h5)."
        )
    return {"model": str(model.resolve()), "results": str(path.resolve())}


def _app_backend() -> bool:
    """Whether the stills take the Windows path (``render._stills_backend``).

    Looked up on the module at each call, so a test that patches
    ``render._stills_backend`` reaches either path on any platform.
    """
    from .. import render as render_mod

    return render_mod._stills_backend() == "app"


def results_files(results: Any, verb: str) -> dict[str, str]:
    """The spec's ``files`` for a ``Results`` (R22b, ruled 2026-11-02).

    The app draws the file a ``Results`` was read from, so only a
    ``Results`` read from a ``.ladruno`` (:attr:`Results.path`) is
    drawn; every other source is refused with a message naming
    ``.ladruno``. A units view (:meth:`Results.to_units`) is held only
    in memory, so it is refused too: the app would label the file's
    units. The model is the ``<stem>.h5`` beside the file
    (:func:`ladruno_files`).
    """
    source = results.path
    if source is None:
        raise ValueError(
            f"results.{verb} on Windows draws through apeGmshViewer, which "
            f"reads a .ladruno file; these Results have no source file "
            f"(ADR 0122 D1; R22b). Open the run's .ladruno with "
            f"Results.from_ladruno."
        )
    if results.units != results.source_units:
        raise ValueError(
            f"results.{verb} on Windows draws {Path(source).name} as recorded, "
            f"in {results.source_units}; a to_units({results.units!r}) view is "
            f"held only in memory, so it is refused (R22b). Render the "
            f"Results the .ladruno opened, without to_units."
        )
    return ladruno_files(source)


def _stage_index(results: Any, stage_id: str) -> int:
    for i, stage in enumerate(results.stages):
        if str(stage.id) == str(stage_id):
            return i
    raise RuntimeError(f"stage {stage_id!r} is not in this results file")


def _app_contour(results: Any, stage_id: str, component: str) -> dict[str, str]:
    """A Python nodal component as the app's ``{field, component}`` (R13).

    Only a recorded nodal axis component of a field in
    :data:`_APP_NODAL_FIELD` maps; anything else is refused by name
    rather than guessed (R16b and R16c are not drawn here).
    """
    _require_contour_data(results, stage_id, component)
    stage = results.stage(stage_id)
    recorded = set(stage.nodes.available_components())
    prefix, _, axis = component.rpartition("_")
    table = _APP_MODE_FIELD if stage.kind == "mode" else _APP_NODAL_FIELD
    if component not in recorded or axis not in _APP_AXES or prefix not in table:
        raise ContourDataError(
            f"component {component!r} has no apegmsh-view/1 contour here: the "
            f"Windows stills draw a recorded nodal axis component of "
            f"{sorted(table)} (e.g. displacement_z). A stored Gauss "
            f"component (R16b) waits on follow-up slice #2794. For another "
            f"one, write a spec and call apeGmsh.agent_view.render_stills."
        )
    return {"field": table[prefix], "component": axis}


def _auto_deform_scale(results: Any, stage_id: str, step: int) -> float:
    """G3: the session path's auto scale, ``_realize._apply_pose``'s rule.

    The largest |u| component at the drawn step becomes
    ``_DEFORM_FRACTION`` of the model diagonal; 1 when either is zero.
    """
    import numpy as np

    from apeGmsh.results._geometry import model_diagonal

    from ._realize import _DEFORM_FRACTION

    stage = results.stage(stage_id)
    recorded = set(stage.nodes.available_components())
    max_d = 0.0
    for axis in _APP_AXES:
        name = f"displacement_{axis}"
        if name not in recorded:
            continue
        values = np.asarray(stage.nodes.get(component=name, time=step).values, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            max_d = max(max_d, float(np.abs(finite).max()))
    diag = model_diagonal(np.asarray(results.fem.nodes.coords, dtype=float))
    if max_d > 0.0 and diag > 0.0:
        return float(_DEFORM_FRACTION * diag / max_d)
    return 1.0


def _require_fem(results: Any, verb: str) -> None:
    if results.fem is None:
        raise RuntimeError(
            f"results.{verb} requires a bound FEMData "
            "(construct with model= / model_h5= or call results.bind)."
        )


def app_shot_for_view(
    results: Any,
    *,
    name: str,
    view: str,
    component: Optional[str],
    step: int,
    deform: DeformArg,
    camera: str,
) -> dict[str, Any]:
    """One ``apegmsh-view/1`` shot for a ``view=`` token (parity plan R9-R22).

    The stage is the session path's (``_resolve_stage``), named by index
    (R22). ``reactions`` (G1) and a velocity or acceleration warp (G4)
    are dropped, so they are refused. ``deformed`` with no scale takes
    the auto scale (G3).
    """
    view = _require_view(view)
    deform_spec = _require_deform(deform)
    if view == "reactions":
        raise ValueError(
            "view='reactions' has no apegmsh-view/1 still (parity plan G1, "
            "dropped 2026-11-02). Contour reaction_force_<axis> instead."
        )
    if deform_spec is not None and view != "mesh" and deform_spec[0] != "displacement":
        raise ValueError(
            f"apegmsh-view/1 warps by displacement only (parity plan G4, "
            f"dropped 2026-11-02); deform field {deform_spec[0]!r} is refused."
        )
    shot: dict[str, Any] = {"name": name, "camera": _app_camera(camera)}
    if view == "mesh":
        return shot
    stage_id = _resolve_stage(results)
    step_i = _resolve_step(results, stage_id, step)
    quantity = component or default_contour_component(results)
    if quantity is None:
        raise ValueError(
            "view='contour'/'deformed' needs a nodal component; "
            "none is recorded. Pass component= explicitly."
        )
    shot["stage"] = _stage_index(results, stage_id)
    shot["step"] = step_i
    shot["contour"] = _app_contour(results, stage_id, quantity)
    if view == "deformed" or deform_spec is not None:
        _require_deform_field(results, stage_id, "displacement")
        scale = deform_spec[1] if deform_spec is not None else None
        if scale is None:
            scale = _auto_deform_scale(results, stage_id, step_i)
        shot["deform"] = {"scale": float(scale)}
    return shot


def app_still(
    results: Any,
    path: "str | Path",
    *,
    files: dict[str, str],
    view: str = "contour",
    component: Optional[str] = None,
    step: int = -1,
    deform: DeformArg = None,
    camera: Optional[str] = None,
    window_size: tuple[int, int] = _DEFAULT_WINDOW,
) -> Optional[Path]:
    """``results.render`` drawn by apeGmshViewer from ``files`` (Windows).

    ``files`` is the set the app opens (:func:`ladruno_files`). Returns
    the written Path; ``None`` only under ``APEGMSH_SKIP_VIEWER=1``.
    Raises ``AgentViewError`` on a refusal and ``FileNotFoundError``
    without the app (G9).
    """
    _warn_if_unscoped(results, "render")
    _require_fem(results, "render")
    camera = _resolve_camera(camera, results.fem.nodes.coords)
    shot = app_shot_for_view(
        results, name="still", view=view, component=component, step=step,
        deform=deform, camera=camera,
    )
    if _env_skips():
        print(_SKIP_ENV)
        return None
    spec = {"spec": _SPEC_VERSION, "files": dict(files), "size": _app_size(window_size), "shots": [shot]}
    _, written = _shoot_app(spec, {"still": Path(path)})
    return written["still"]


def app_pack_spec(
    results: Any,
    *,
    files: dict[str, str],
    camera: str,
    window_size: tuple[int, int],
) -> dict[str, Any]:
    """G5: the canned pack as ONE ``apegmsh-view/1`` spec (R23-R28).

    ``mesh``, then ``contour`` and ``deformed`` at the last step of the
    pack's stage with the default component; ``deformed`` only when a
    displacement is recorded. ``reactions`` is dropped (G1, R26), and
    ``history.png`` stays matplotlib (G6, R27), outside the spec.
    """
    shots = [app_shot_for_view(
        results, name="mesh", view="mesh", component=None, step=-1,
        deform=None, camera=camera,
    )]
    component = _pack_component(results)
    if component is not None:
        shots.append(app_shot_for_view(
            results, name="contour", view="contour", component=component,
            step=-1, deform=None, camera=camera,
        ))
        # No recorded displacement: the deformed still is left out, as v1.
        if _has_displacement(results, _resolve_stage(results)):
            shots.append(app_shot_for_view(
                results, name="deformed", view="deformed", component=component,
                step=-1, deform=None, camera=camera,
            ))
    return {"spec": _SPEC_VERSION, "files": dict(files), "size": _app_size(window_size), "shots": shots}


def _pack_component(results: Any) -> Optional[str]:
    """The pack's contour component (v1's choice), ``None`` when there is none."""
    try:
        return default_contour_component(results)
    except (RuntimeError, ValueError):
        return None


def _has_displacement(results: Any, stage_id: str) -> bool:
    recorded = set(results.stage(stage_id).nodes.available_components())
    return any(f"displacement_{axis}" in recorded for axis in _APP_AXES)


def app_pack(
    results: Any,
    out_dir: "str | Path",
    *,
    files: dict[str, str],
    camera: Optional[str] = None,
    window_size: tuple[int, int] = _DEFAULT_WINDOW,
) -> tuple[Path, ...]:
    """``results.render_pack`` drawn by apeGmshViewer from ``files`` (Windows).

    Writes ``<shot>.png`` per shot of :func:`app_pack_spec`, the app's
    ``manifest.json`` (the record of what was drawn, ADR 0128 D3), and
    ``history.png``. Returns the PNG paths in pack order; ``()`` only
    under ``APEGMSH_SKIP_VIEWER=1``.
    """
    _warn_if_unscoped(results, "render_pack")
    _require_fem(results, "render_pack")
    camera = _resolve_camera(camera, results.fem.nodes.coords)
    spec = app_pack_spec(results, files=files, camera=camera, window_size=window_size)
    if _env_skips():
        print(_SKIP_ENV)
        return ()
    dest = Path(out_dir)
    targets = {shot["name"]: dest / f"{shot['name']}.png" for shot in spec["shots"]}
    manifest, written = _shoot_app(spec, targets)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paths = [written[shot["name"]] for shot in spec["shots"]]
    history = _try_history(results, dest / "history.png", _pack_component(results))
    if history is not None:
        paths.append(history)
    return tuple(paths)


__all__ = [
    "app_pack",
    "app_pack_spec",
    "app_shot_for_view",
    "app_still",
    "default_contour_component",
    "ladruno_files",
    "render_results_pack",
    "render_results_still",
    "results_files",
    "session_for_view",
]
