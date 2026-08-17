"""``realize`` — one-shot projection of a mesh pane into a backend.

ADR 0098 S1 (S1-A mesh pane, S1-B the rest of §4 + scope + plots).
The session IR (``apeGmsh.results.session``) is truth;
this module turns ONE mesh pane into scene-IR layers through an ADR
0042 ``RenderBackend``, reusing the existing per-kind emit code in
place, director-free (the spec carries ``stage_id``;
``_stage_pin_resolver`` stays unset, the visual store is stamped here
the way ``DiagramRegistry`` stamps it). It lives on the viewers side —
NOT under ``results/session`` — precisely so the session package never
imports ``viewers.diagrams`` (``test_session_pure.py``).

One-shot contract (S1): each call realizes a complete pane into an
empty backend and returns a :class:`RealizedPane` whose layers carry
STABLE keys (``"<pane_id>:substrate"``, ``"<pane_id>:contour"`` …).
The S2 reconciler diffs realizations by that key; the raw backend
``layer_id`` is an emission detail.

Laws realized here:

* **Pose before extraction** (S1 decision 5, warp-before-extract): the
  deform pose is composed onto ``scene.grid.points`` BEFORE any layer
  extracts, so the substrate and every slot extraction copy
  already-posed points and ``sync_substrate_points`` is never needed
  on the session path. Pinned equal to ``render_results``'s
  attach-then-sync output by the parity tests.
* **INV-MESH-1, one cell set**: the view's scope resolves once
  (:mod:`._scope`) and every layer — substrate included — is a
  function of it.
* **INV-MESH-2, one surface**: the grey substrate layer is emitted only
  while no *occluding* slot is occupied (contour, sand — generalized
  from ``Diagram.occludes_substrate``); the occupant IS the surface,
  never a second grey mesh under it.
* **INV-LEGEND-1..5** (S1 decision 4): a null legend-controller is
  adopted for the backend before any diagram attaches, so
  ``ScalarBarSupport`` registrations no-op and the ONLY colour scales
  emitted are the ones this module derives from ``view.legends()`` —
  the pose can never cause a bar, and hiding a scale never clears a
  slot.

Session state with no realization yet refuses loudly
(``NotImplementedError`` naming the owning slice) rather than emitting
a picture that silently drops it: the style buttons, view clips and
the undeformed overlay (S3), and a scoped view carrying the `loads`
slot (the kind draws every node of its pattern regardless of scope,
so honouring INV-MESH-1 needs a change inside that kind).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from .._pump_set import _compose_substrate_points, read_nodal_vector_field
from ..core._legend import LegendController, adopt_controller, key_str
from ..data import ViewerData
from ..diagrams._kinds import kind_def
from ..diagrams._scalar_bar_support import _default_vertical
from ..diagrams._visual_store import VisualDataStore
from ..scene.fem_scene import build_fem_scene
from ..scene_ir import ColorSpec, MeshLayer, PointSet
from ._plots import realize_plot
from ._scope import ScopedSet, resolve_scope
from ._specs import OCCLUDING_KINDS, slot_spec

if TYPE_CHECKING:
    from apeGmsh.results.Results import Results
    from apeGmsh.results.session import MeshView, ResultsSession
    from ..scene.fem_scene import FEMSceneData

# render.py's still rule, shared by parity: the largest |u| becomes
# this fraction of the model diagonal when Deform.scale is None.
_DEFORM_FRACTION = 0.12


# =====================================================================
# Result records — the S2 diff surface
# =====================================================================


@dataclass(frozen=True)
class RealizedLayer:
    """One emitted layer of one pane.

    ``key`` is the STABLE identity (``"<pane_id>:<role>"``) the S2
    reconciler diffs by; ``layer_id`` / ``handle`` are what the backend
    actually holds for update / remove.
    """

    key: str
    layer_id: str
    handle: Any


@dataclass(frozen=True)
class RealizedPane:
    """One pane, realized. ``scalar_bars`` are the bar keys emitted to
    the backend, in ``view.legends()`` order (hidden scales register
    invisible and emit no bar — INV-LEGEND-3).

    ``diagrams`` and ``legend_controller`` keep the emitting objects
    alive for the caller's lifetime management (S2 drives
    ``update_to_step`` through them); tests may introspect, nothing
    else should.
    """

    pane_id: str
    layers: tuple[RealizedLayer, ...]
    scalar_bars: tuple[str, ...]
    diagrams: tuple[Any, ...]
    legend_controller: Optional[LegendController]


# =====================================================================
# Decision 4 — the null legend owner
# =====================================================================


class _NullLegendController:
    """Legend owner that owns nothing.

    Adopted for the realize backend (``adopt_controller``) BEFORE any
    diagram attaches, so every ``ScalarBarSupport`` call —
    ``register`` at attach, ``unregister`` at detach, and the
    ``_legend_mutate`` setters — is a no-op and the session IR stays
    the only author of colour scales. Only the surface the mixin
    actually touches is implemented; anything else is a loud
    ``AttributeError`` by design.
    """

    default_font_scale = 1.0

    def register(self, key: Any, layer_id: str, handle: Any, **_: Any) -> None:
        return None

    def unregister(self, layer_id: str) -> None:
        return None

    def entry(self, key: Any) -> None:
        return None

    def set_visible(self, key: Any, value: Any) -> None:
        return None

    def set_fmt(self, key: Any, value: Any) -> None:
        return None

    def set_vertical(self, key: Any, value: Any) -> None:
        return None

    def set_font_scale(self, key: Any, value: Any) -> None:
        return None

    def set_lut(self, key: Any, value: Any) -> None:
        return None

    def refresh(self) -> None:
        return None


# =====================================================================
# realize
# =====================================================================


def realize_pane(
    session: "ResultsSession", pane: Any, backend: Any = None,
) -> Any:
    """Realize one pane of ``session``.

    A mesh view emits scene-IR layers into ``backend`` and returns a
    :class:`RealizedPane`; a plot pane resolves its series to arrays
    and returns a :class:`~._plots.RealizedPlot` (no backend needed —
    §6: a plot is a pane, and its client draws the numbers).
    """
    from apeGmsh.results.session import MeshView, PlotView

    if isinstance(pane, PlotView):
        _require_member(session, pane)
        return realize_plot(session, pane)
    if not isinstance(pane, MeshView):
        raise TypeError(
            f"realize takes a MeshView or PlotView; got "
            f"{type(pane).__name__}."
        )
    if backend is None:
        raise TypeError(
            "Realizing a mesh pane needs a RenderBackend to emit into."
        )
    return _realize_mesh(session, pane, backend)


def _realize_mesh(
    session: "ResultsSession", view: "MeshView", backend: Any,
) -> RealizedPane:
    results = _require_results(session)
    _require_member(session, view)
    _refuse_unrealized_state(view)
    if results.fem is None:
        raise RuntimeError(
            "session realize requires a bound FEMData "
            "(construct with model= / model_h5= or call results.bind)."
        )

    stage_id, step = _resolve_instant(session, view, results)
    scene = build_fem_scene(results.fem)
    view_data = ViewerData.from_fem(results.fem)
    # INV-MESH-1: ONE cell set; every layer below is a function of it.
    scoped = resolve_scope(view.scope, view_data)

    # Decision 5: pose first — every extraction below copies posed points.
    _apply_pose(view, results, scene, stage_id, step)

    # Decision 4: no diagram may author a colour scale on this backend.
    adopt_controller(backend, _NullLegendController())

    layers: list[RealizedLayer] = []
    diagrams: list[Any] = []
    slot_diagrams: dict[str, Any] = {}
    store = VisualDataStore()

    # Resolve every occupied slot to its kind + spec first: the
    # substrate decision below needs the kinds, and routing a slot
    # twice would re-read the stage's component index.
    resolved = [
        (category, *slot_spec(
            category, record, results, stage_id, scoped, view_data,
        ))
        for category, record in view.slots.items()
    ]

    # INV-MESH-2, one surface: the grey substrate is drawn only while
    # nothing opaque and coincident occupies a slot — generalized from
    # ``Diagram.occludes_substrate`` (contour and sand both do).
    if not any(kind_id in OCCLUDING_KINDS for _c, kind_id, _s in resolved):
        substrate = _substrate_layer(view.id, scene, scoped)
        handle = backend.add_layer(substrate)
        layers.append(RealizedLayer(
            key=f"{view.id}:substrate",
            layer_id=substrate.layer_id,
            handle=handle,
        ))

    for category, kind_id, spec in resolved:
        kdef = kind_def(kind_id)
        assert kdef is not None
        diagram = kdef.diagram_class(spec, results)
        # Same stamping idiom as DiagramRegistry._stamp_visual_store —
        # without it a contour's colour scale collapses to step 0's range.
        diagram._visual_store = store  # noqa: SLF001
        diagram.attach(backend, view_data, scene)
        if step != 0:
            diagram.update_to_step(step)
        for suffix, handle in _emitted_handles(diagram):
            key = f"{view.id}:{category}"
            layers.append(RealizedLayer(
                key=f"{key}.{suffix}" if suffix else key,
                layer_id=handle.layer_id,
                handle=handle,
            ))
        diagrams.append(diagram)
        slot_diagrams[category] = diagram

    bar_keys, legend_controller = _realize_legends(
        backend, view, slot_diagrams,
    )

    return RealizedPane(
        pane_id=view.id,
        layers=tuple(layers),
        scalar_bars=bar_keys,
        diagrams=tuple(diagrams),
        legend_controller=legend_controller,
    )


def _emitted_handles(diagram: Any) -> "list[tuple[str, Any]]":
    """Every backend handle ``diagram`` emitted, as ``(suffix, handle)``.

    Most kinds hold one layer on ``_handle``. ``ReactionsDiagram`` is
    the multi-layer one: it draws force and moment glyphs as separate
    families (``_force`` / ``_moment``, each with its own handle), and
    reading only ``_handle`` would hand the S2 reconciler an empty
    layer list for an occupied slot — layers it could then never
    update or remove.
    """
    handle = getattr(diagram, "_handle", None)
    if handle is not None:
        return [("", handle)]
    out: list[tuple[str, Any]] = []
    for attr, suffix in (("_force", "force"), ("_moment", "moment")):
        family = getattr(diagram, attr, None)
        family_handle = getattr(family, "handle", None)
        if family_handle is not None:
            out.append((suffix, family_handle))
    return out


def _require_results(session: "ResultsSession") -> "Results":
    results = session.results
    if results is None:
        raise RuntimeError(
            "This session has no Results bound — construct it via "
            "results.session() to realize or render."
        )
    return results


def _require_member(session: "ResultsSession", pane: Any) -> None:
    if session.pane(pane.id) is not pane:  # KeyError on a foreign id
        raise ValueError(
            f"Pane {pane.id!r} is not a member of this session."
        )


def _refuse_unrealized_state(view: "MeshView") -> None:
    """State this slice cannot draw refuses loudly — a still that
    silently drops session state is a wrong picture, not a lenient
    one."""
    from apeGmsh.results.session import MeshStyle

    if view.scope is not None and "loads" in view.slots:
        # The loads kind ignores its selector's resolved ids
        # (_loads.py:88-96 reads every record of the pattern), so a
        # scoped view would draw arrows outside its own cell set —
        # an INV-MESH-1 violation. Filtering them needs a change in
        # the diagram kind, which this slice does not touch.
        raise NotImplementedError(
            "A scoped view cannot draw the loads slot yet: the loads "
            "kind renders every node of the pattern regardless of "
            "scope (INV-MESH-1). Clear the scope or the loads slot."
        )
    if view.style != MeshStyle():
        raise NotImplementedError(
            "Style-button realization (mesh / outlines / nodes / gauss) "
            "lands at S3; S1-A draws the boot style."
        )
    if view.overlay:
        raise NotImplementedError(
            "The undeformed overlay realizes in a later slice (S1-B)."
        )
    if view.clips:
        raise NotImplementedError(
            "View clips realize with the pane host slices (S3)."
        )


# =====================================================================
# Instant resolution (§7)
# =====================================================================


def _resolve_instant(
    session: "ResultsSession", view: "MeshView", results: "Results",
) -> tuple[str, int]:
    """``(stage_id, step)`` this pane realizes at.

    A mode pose has no instant (§4/§7): it pins the view to its
    ``kind='mode'`` stage at step 0. Otherwise the §7 law
    (``session.effective_instant``) applies, and no instant at all
    falls back to the last stage's last step — the existing
    ``render_results`` convention.
    """
    if view.is_mode_posed:
        assert view.deform is not None
        return _mode_stage_id(results, view.deform.mode), 0

    instant = session.effective_instant(view)
    if instant is None:
        stages = list(results.stages)
        if not stages:
            raise RuntimeError("session realize requires at least one stage.")
        stage_id = stages[-1].id
        n = int(results.stage(stage_id).n_steps)
        if n <= 0:
            raise ValueError(
                f"Stage {stage_id!r} has no recorded steps to realize."
            )
        return stage_id, n - 1

    scoped = results.stage(instant.stage)  # loud on an unknown stage
    n = int(scoped.n_steps)
    if instant.step >= n:
        raise ValueError(
            f"Instant step {instant.step} is out of range for stage "
            f"{instant.stage!r} with {n} step(s)."
        )
    return instant.stage, instant.step


def _mode_stage_id(results: "Results", mode: Optional[int]) -> str:
    available: list[int] = []
    for stage in results.stages:
        if stage.kind != "mode":
            continue
        if stage.mode_index == mode:
            return stage.id
        if stage.mode_index is not None:
            available.append(int(stage.mode_index))
    raise ValueError(
        f"Deform.mode={mode} matches no kind='mode' stage "
        f"(available mode indices: {sorted(available)})."
    )


# =====================================================================
# Pose (§3 — deform is a pose, never a picture)
# =====================================================================


def _apply_pose(
    view: "MeshView",
    results: "Results",
    scene: "FEMSceneData",
    stage_id: str,
    step: int,
) -> None:
    deform = view.deform
    if deform is None:
        return
    vals = read_nodal_vector_field(
        results.stage(stage_id), scene, deform.field, step,
    )
    if vals is None:
        raise ValueError(
            f"Deform needs a recorded {deform.field}_* field; none is "
            f"present in stage {stage_id!r}."
        )
    scale = deform.scale
    if scale is None:
        max_d = float(np.abs(vals).max())
        diag = float(scene.model_diagonal)
        scale = (
            _DEFORM_FRACTION * diag / max_d
            if max_d > 0.0 and diag > 0.0
            else 1.0
        )
    pts = _compose_substrate_points(
        scene.reference_points, None, vals, float(scale),
    )
    if pts is not None:
        scene.grid.points = pts


# =====================================================================
# Substrate (INV-MESH-2) — real scene-IR emission
# =====================================================================


def _substrate_layer(
    pane_id: str, scene: "FEMSceneData", scoped: "ScopedSet",
) -> MeshLayer:
    """The grey analysis mesh as a scene-IR layer (boot style: fill +
    element edges), replacing the raw ``add_mesh`` of the old paths.

    Restricted to the view's cell set when scoped — the grey mesh is a
    layer of the one cell set like everything else (INV-MESH-1), not a
    backdrop of the whole model.
    """
    from ..backends.pyvista_qt import cellblocks_from_grid

    try:
        from ..ui.theme import THEME
        palette = THEME.current
        fill = palette.substrate_color
        edge = palette.substrate_edge_color
    except Exception:
        fill, edge = "#bfbfbf", "#1a1a1a"
    grid = scene.grid
    if not scoped.is_unscoped:
        cells = np.fromiter(
            (
                scene.element_id_to_cell.get(int(e), -1)
                for e in scoped.element_ids
            ),
            dtype=np.int64,
            count=int(scoped.element_ids.size),
        )
        cells = cells[cells >= 0]
        if cells.size == 0:
            raise ValueError(
                "The view's scope selects no cell of the analysis mesh "
                "— nothing to draw."
            )
        grid = grid.extract_cells(cells)
    return MeshLayer(
        layer_id=f"{pane_id}:substrate",
        points=PointSet(np.asarray(grid.points)),
        cells=cellblocks_from_grid(grid),
        color=ColorSpec(mode="solid", solid_rgb=fill),
        show_edges=True,
        edge_color=edge,
        line_width=1.0,
    )


# =====================================================================
# Legends (§5) — derived from the session IR, never from diagrams
# =====================================================================


def _realize_legends(
    backend: Any,
    view: "MeshView",
    slot_diagrams: dict[str, Any],
) -> tuple[tuple[str, ...], Optional[LegendController]]:
    """Emit exactly the scales ``view.legends()`` derives.

    A REAL ``LegendController`` bound directly (not through
    ``controller_for`` — that slot holds the null controller), so the
    0081/0090 layout and typography are reused unchanged. One entry per
    legend, keyed ``("", field)``, bound to the causing slot's layer;
    ``visible`` is the session's hide chrome, never a diagram style.
    """
    legends = view.legends()
    if not legends:
        return (), None
    controller = LegendController(backend)
    bar_keys: list[str] = []
    for legend in legends:
        diagram = slot_diagrams[legend.categories[0]]
        handle = diagram._handle  # noqa: SLF001
        if handle is None:
            # The slot is occupied but its kind drew nothing (several
            # return from attach without emitting: no elements in the
            # scope carry Gauss rows, no node survives the selector,
            # …). A scale for a picture that is not on screen would be
            # a legend about nothing — INV-LEGEND-1 ties the scale to
            # the *painted* field. The empty layer list is what tells a
            # client the slot came up empty.
            continue
        style = diagram.spec.style
        key = ("", legend.field)
        controller.register(
            key,
            handle.layer_id,
            handle,
            lut=_layer_lutspec(diagram),
            fmt=getattr(style, "fmt", "%.3g") or "%.3g",
            vertical=_default_vertical(
                getattr(style, "scalar_bar_vertical", None)
            ),
            font_scale=getattr(style, "scalar_bar_scale", None),
            visible=not legend.hidden,
        )
        if not legend.hidden:
            bar_keys.append(key_str(key))
    return tuple(bar_keys), controller


def _layer_lutspec(diagram: Any) -> Any:
    """The scale range of the picture actually emitted.

    The emitted layer's ``ColorSpec.lut`` carries the true clim/cmap
    computed without Qt; the mixin's ``_current_lutspec()`` reads the
    Qt LUT mirror, which silently degrades to a default 0–1 ``LutSpec``
    whenever ``_init_lut`` could not build (probe P10) — a wrong scale
    on a correct picture. Mirror kept as the fallback for any kind
    that does not cache its emitted layer.
    """
    layer = getattr(diagram, "_layer", None)
    lut = getattr(getattr(layer, "color", None), "lut", None)
    if lut is not None:
        return lut
    return diagram._current_lutspec()  # noqa: SLF001


__all__ = ["RealizedLayer", "RealizedPane", "realize_pane"]
