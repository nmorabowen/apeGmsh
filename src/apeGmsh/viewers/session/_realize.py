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
* **INV-MESH-3/4, the four style buttons** (S3): ``mesh`` is the
  interior grid ON the one surface — ``show_edges`` of whichever layer
  IS that surface, never a second wire mesh; ``outlines`` are the
  feature edges OF the same cell set, so hiding a cell takes its
  outline with it; ``nodes`` and ``gauss`` are glyph clouds at the
  cell set's nodes and integration points. Gauss-as-button draws
  LOCATIONS and stands down when the `gauss` slot is occupied — the
  slot already draws that cloud, with values, and two clouds is the
  law's named failure.
* **INV-LEGEND-1..5** (S1 decision 4): a null legend-controller is
  adopted for the backend before any diagram attaches, so
  ``ScalarBarSupport`` registrations no-op and the ONLY colour scales
  emitted are the ones this module derives from ``view.legends()`` —
  the pose can never cause a bar, and hiding a scale never clears a
  slot.

Session state with no realization yet refuses loudly
(``NotImplementedError`` naming the owning slice) rather than emitting
a picture that silently drops it: the undeformed overlay, and a scoped
view carrying the `loads` slot (the kind draws every node of its
pattern regardless of scope, so honouring INV-MESH-1 needs a change
inside that kind).
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

    # ADR 0083 machinery, view-owned (§3): the active planes of THIS
    # view, pushed before any layer is added so the backend stamps them
    # onto everything it creates. Always pushed — an empty set is how a
    # view with no clips clears a previous flush's cut.
    _apply_clips(backend, view)

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
    # INV-MESH-4 "Mesh": the interior grid belongs to whichever layer IS
    # the one surface. Unoccluded that is the substrate (handled below);
    # occluded it is the occupant, so the button rides its style rather
    # than adding a second wire mesh over it (INV-MESH-2).
    resolved = [
        (category, kind_id, _with_surface_edges(spec, kind_id, view.style))
        for category, kind_id, spec in resolved
    ]

    # INV-MESH-2, one surface: the grey substrate is drawn only while
    # nothing opaque and coincident occupies a slot — generalized from
    # ``Diagram.occludes_substrate`` (contour and sand both do).
    occluded = any(kind_id in OCCLUDING_KINDS for _c, kind_id, _s in resolved)
    grid = _scoped_grid(scene, scoped)
    if not occluded:
        substrate = _substrate_layer(view.id, grid, view.style.mesh)
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
        if view.deform is not None:
            # Decision-5 addendum (S2 acceptance defect): warp-before-
            # extract only poses kinds that extract from scene.grid.
            # Kinds that OWN their glyph geometry (gauss markers,
            # vector/principal anchors) place it from the REFERENCE
            # view data at attach — the old viewer moves them with the
            # DEFORM pump's sync primitive, which the session path
            # must run too (one-shot, same step→deform order as the
            # dispatcher matrix; the already-posed grid points are the
            # pump's ready-made deformed_pts).
            diagram.sync_substrate_points(scene.grid.points, scene)
        _ensure_true_lut(kind_id, diagram, backend)
        for suffix, handle in _emitted_handles(diagram):
            key = f"{view.id}:{category}"
            layers.append(RealizedLayer(
                key=f"{key}.{suffix}" if suffix else key,
                layer_id=handle.layer_id,
                handle=handle,
            ))
        diagrams.append(diagram)
        slot_diagrams[category] = diagram

    # INV-MESH-3/4 — outlines / nodes / gauss as layers of the SAME cell
    # set, emitted after the slots so they read above the surface.
    for role, layer in _style_layers(
        view, scene, grid, scoped, view_data, results, stage_id,
    ):
        handle = backend.add_layer(layer)
        layers.append(RealizedLayer(
            key=f"{view.id}:{role}",
            layer_id=layer.layer_id,
            handle=handle,
        ))

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


#: Colour-mapped kinds whose auto-fit lookup-table range is seeded at
#: ATTACH (step 0) — or left at a hard placeholder — and re-ranged
#: later by the old window's Qt LUT machinery, which the session
#: path's null legend controller deliberately silences. Realized at
#: any later instant they clamp the whole picture to the stale range
#: (S2 acceptance defect: "stresses all one colour"). ``contour`` is
#: deliberately absent: it computes the real whole-history clim
#: itself through the visual store (S1-A probe P10).
_DEFAULT_LUT_KINDS = frozenset(
    {"gauss_marker", "vector_glyph", "principal_glyph", "sand"}
)


def _ensure_true_lut(kind_id: str, diagram: Any, backend: Any) -> None:
    """Auto-fit the lookup-table range to the DISPLAYED values.

    Only for :data:`_DEFAULT_LUT_KINDS`, and only under their styles'
    ``clim=None`` auto-fit contract — an explicit ``style.clim`` wins
    untouched. These kinds auto-fit at attach, which the session path
    always runs at step 0 before stepping to the realized instant, so
    the picture shows one step through another step's scale; with no
    colour-map editor on this path, the displayed data IS the only
    truthful range. ``principal_glyph`` re-ranges SYMMETRICALLY so its
    diverging colormap stays centred on zero (signed principal law);
    the cmap is style, not range, and survives. Updates the emitted
    layer record too, so ``_realize_legends`` puts the same truthful
    range on the scalar bar.
    """
    from dataclasses import replace

    from ..scene_ir import GlyphLayer

    if kind_id not in _DEFAULT_LUT_KINDS:
        return
    if getattr(diagram.spec.style, "clim", None) is not None:
        return
    layer = getattr(diagram, "_layer", None)
    handle = getattr(diagram, "_handle", None)
    color = getattr(layer, "color", None)
    lut = getattr(color, "lut", None)
    if layer is None or handle is None or lut is None:
        return
    if color.mode != "by_array":
        return
    if isinstance(layer, GlyphLayer):
        values = layer.color_scalar
    else:
        values = next(
            (
                f.values for f in getattr(layer, "fields", ())
                if f.name == color.array_name
            ),
            None,
        )
    if values is None or np.asarray(values).size == 0:
        return
    if kind_id == "principal_glyph":
        span = float(np.max(np.abs(values))) or 1e-6
        vmin, vmax = -span, span
    else:
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if vmin == vmax:
            # A uniform field still deserves a colour: widen so the
            # lookup table has a nonzero span.
            vmax = vmin + (abs(vmin) or 1.0) * 1e-6
    if (float(lut.vmin), float(lut.vmax)) == (vmin, vmax):
        return
    new_color = replace(color, lut=replace(lut, vmin=vmin, vmax=vmax))
    diagram._layer = replace(layer, color=new_color)  # noqa: SLF001
    backend.set_layer_color(handle, new_color)


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
    if view.overlay:
        raise NotImplementedError(
            "The undeformed overlay does not realize yet: it is this "
            "mesh at scale 0 in the Outlines style (§3), which needs "
            "the unposed cell set emitted alongside the posed one."
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


def recorded_components(
    session: "ResultsSession", view: "MeshView",
) -> "tuple[set[str], set[str]]":
    """``(nodal, gauss)`` component tokens the view's instant records.

    The vocabulary of every picker and every data-dependent control in
    the window (the inspector's slot rows; the pane header's Gauss
    style button). Resolved through the SAME §7 instant law realize
    applies, so a control and the picture it describes cannot disagree
    about which stage they mean. Empty sets when anything is missing —
    a disabled control, never an error.
    """
    results = session.results
    if results is None:
        return set(), set()
    try:
        stage_id, _step = _resolve_instant(session, view, results)
        components = results.inspect.components(stage=stage_id)
    except Exception:
        return set(), set()
    return (
        set(components.get("nodes", ())),
        set(components.get("gauss", ())),
    )


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


def _scoped_grid(scene: "FEMSceneData", scoped: "ScopedSet") -> Any:
    """The pyvista grid of the view's ONE cell set (INV-MESH-1).

    Unscoped that is the whole posed analysis mesh; scoped it is the
    extracted subset — the same grid the substrate, the outlines and
    the node cloud are all read from, which is what makes "hide a cell
    → its face, grid line and outline all go" (INV-MESH-3) true by
    construction rather than by three agreeing filters.
    """
    grid = scene.grid
    if scoped.is_unscoped:
        return grid
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
    return grid.extract_cells(cells)


def _palette() -> Any:
    """The live palette, or ``None`` with no theme module (headless
    realize outside the Qt tree). Callers fall back per colour."""
    try:
        from ..ui.theme import THEME
        return THEME.current
    except Exception:
        return None


def _substrate_layer(pane_id: str, grid: Any, show_edges: bool) -> MeshLayer:
    """The grey analysis mesh as a scene-IR layer, replacing the raw
    ``add_mesh`` of the old paths.

    ``show_edges`` is the "Mesh" style button (INV-MESH-4): the
    interior element grid is a property of the one surface, not a
    second layer over it.
    """
    from ..backends.pyvista_qt import cellblocks_from_grid

    palette = _palette()
    fill = getattr(palette, "substrate_color", None) or "#bfbfbf"
    edge = getattr(palette, "substrate_edge_color", None) or "#1a1a1a"
    return MeshLayer(
        layer_id=f"{pane_id}:substrate",
        points=PointSet(np.asarray(grid.points)),
        cells=cellblocks_from_grid(grid),
        color=ColorSpec(mode="solid", solid_rgb=fill),
        show_edges=bool(show_edges),
        edge_color=edge,
        line_width=1.0,
    )


# =====================================================================
# Style buttons (INV-MESH-3 / INV-MESH-4) — view chrome, never slots
# =====================================================================

#: Feature-edge threshold for the Outlines button. Same preference the
#: old viewport's ADR 0089 outline pass reads, so the two draw the same
#: creases; a stripped preferences.json falls back to its default.
_OUTLINE_FEATURE_ANGLE = 25.0

#: Screen-space glyph sizes for the node / Gauss clouds. Screen space,
#: not world space, so a pane that is 240 px wide still shows readable
#: markers — these are pick affordances (§8: click-pick requires the
#: matching style button on), not measured geometry.
_NODE_POINT_SIZE = 7.0
_GAUSS_POINT_SIZE = 6.0


def _with_surface_edges(spec: Any, kind_id: str, style: Any) -> Any:
    """Ride the "Mesh" button onto an occluding slot's own style."""
    from dataclasses import replace

    if kind_id not in OCCLUDING_KINDS:
        return spec
    if not hasattr(spec.style, "show_edges"):
        return spec
    return replace(spec, style=replace(spec.style, show_edges=style.mesh))


def _style_layers(
    view: "MeshView",
    scene: "FEMSceneData",
    grid: Any,
    scoped: "ScopedSet",
    view_data: "ViewerData",
    results: "Results",
    stage_id: str,
) -> "list[tuple[str, Any]]":
    """``(role, layer)`` for every ON style button that draws its own
    layer — "Mesh" is not here, it is ``show_edges`` of the surface."""
    out: list[tuple[str, Any]] = []
    style = view.style
    if style.outlines:
        layer = _outlines_layer(view.id, grid)
        if layer is not None:
            out.append(("outlines", layer))
    if style.nodes:
        layer = _nodes_layer(view.id, scene, scoped)
        if layer is not None:
            out.append(("nodes", layer))
    # "They must not draw two clouds" (INV-MESH-4): the `gauss` SLOT
    # already paints these points, with values and a scale. The button
    # stays lit — it describes what is on screen — and emits nothing.
    if style.gauss and "gauss" not in view.slots:
        layer = _gauss_layer(
            view.id, scene, scoped, view_data, results, stage_id,
            posed=view.deform is not None,
        )
        if layer is not None:
            out.append(("gauss", layer))
    return out


def _outlines_layer(pane_id: str, grid: Any) -> "Optional[MeshLayer]":
    """Element / feature boundaries of the cell set (ADR 0089 D1).

    ``None`` when the cell set has no such edges — a 1-D beam model
    whose surface is already lines. An empty layer would be an actor
    that draws nothing; the missing key is what tells a client the
    button had nothing to draw.
    """
    from ..scene.mesh_scene import _extract_surface_fast
    from ..scene_ir import CellBlocks

    try:
        # The project's own boundary extraction (vtkGeometryFilter in
        # FastMode): the conservative path allocates a point-to-cell
        # hash at ~96 B per point, which is ~466 MB at 607k points —
        # and this now runs once per pane, per flush.
        surface = _extract_surface_fast(grid)
        edges = surface.extract_feature_edges(
            feature_angle=_outline_feature_angle(),
            boundary_edges=True,
            feature_edges=True,
            manifold_edges=False,
            non_manifold_edges=False,
        )
    except Exception:
        return None
    if edges is None or edges.n_points == 0 or edges.n_lines == 0:
        return None
    conn = _two_point_lines(edges)
    if conn is None:
        return None
    palette = _palette()
    color = getattr(palette, "outline_color", None) or "#1a1a1a"
    return MeshLayer(
        layer_id=f"{pane_id}:outlines",
        points=PointSet(np.asarray(edges.points)),
        cells=CellBlocks({"line": conn}),
        color=ColorSpec(mode="solid", solid_rgb=color),
        wireframe=True,
        line_width=2.0,
        pickable=False,
    )


def _two_point_lines(edges: Any) -> "Optional[np.ndarray]":
    """``(n, 2)`` connectivity of a feature-edge ``PolyData``.

    ``PolyData`` carries lines as VTK's flat ``[npts, i, j, npts, ...]``
    array and has no ``cells_dict``, so ``cellblocks_from_grid`` (which
    reads that dict) cannot decompose it. ``extract_feature_edges``
    emits 2-point segments; anything else would need a polyline split
    this layer does not owe, so it declines rather than mis-reading the
    stream.
    """
    flat = np.asarray(edges.lines, dtype=np.int64)
    if flat.size == 0 or flat.size % 3 != 0:
        return None
    triples = flat.reshape(-1, 3)
    if not bool(np.all(triples[:, 0] == 2)):
        return None
    return np.ascontiguousarray(triples[:, 1:])


def _outline_feature_angle() -> float:
    try:
        from ..ui.preferences_manager import PREFERENCES
        return float(PREFERENCES.current.feature_angle)
    except Exception:
        return _OUTLINE_FEATURE_ANGLE


def _nodes_layer(
    pane_id: str, scene: "FEMSceneData", scoped: "ScopedSet",
) -> "Optional[MeshLayer]":
    """Node glyphs of the cell set — the click-pick affordance (§8).

    Read off ``scene.grid.points``, which the pose already moved, so
    the glyphs follow the deformed model without a second sync.
    """
    from ..scene_ir import CellBlocks

    points = np.asarray(scene.grid.points)
    if scoped.is_unscoped:
        rows = np.arange(points.shape[0], dtype=np.int64)
    else:
        rows = np.fromiter(
            (
                scene.node_id_to_idx.get(int(n), -1)
                for n in scoped.node_ids
            ),
            dtype=np.int64,
            count=int(scoped.node_ids.size),
        )
        rows = rows[rows >= 0]
    if rows.size == 0:
        return None
    palette = _palette()
    color = getattr(palette, "node_accent", None) or "#e0e0e0"
    return MeshLayer(
        layer_id=f"{pane_id}:nodes",
        points=PointSet(points[rows]),
        cells=CellBlocks(
            {"vertex": np.arange(rows.size, dtype=np.int64).reshape(-1, 1)},
        ),
        color=ColorSpec(mode="solid", solid_rgb=color),
        point_size=_NODE_POINT_SIZE,
        render_points_as_spheres=True,
        pickable=False,
    )


def _gauss_layer(
    pane_id: str,
    scene: "FEMSceneData",
    scoped: "ScopedSet",
    view_data: "ViewerData",
    results: "Results",
    stage_id: str,
    *,
    posed: bool,
) -> "Optional[MeshLayer]":
    """Integration-point glyphs — LOCATIONS, not values (INV-MESH-4).

    The addresses come from a Gauss slab, which is read per component;
    any recorded component answers the same ``(element_index,
    natural_coords)`` pair, so the first one alphabetically is read
    purely as the address probe and its values are discarded. A stage
    that records no Gauss composite has no integration points to
    place — the pane header disables the button in that case, and this
    returns ``None`` rather than inventing a cloud.
    """
    from apeGmsh.results._gauss_world_coords import (
        compute_global_coords_from_arrays,
    )

    from ..scene_ir import CellBlocks

    probe = _gauss_probe_component(results, stage_id)
    if probe is None:
        return None
    element_ids = (
        scoped.element_ids if not scoped.is_unscoped
        else np.asarray(scene.cell_to_element_id, dtype=np.int64)
    )
    if element_ids.size == 0:
        return None
    try:
        slab = results.stage(stage_id).elements.gauss.get(
            ids=tuple(int(e) for e in element_ids),
            component=probe,
            time=[0],
        )
    except Exception:
        return None
    if slab.element_index.size == 0:
        return None
    try:
        coords = compute_global_coords_from_arrays(
            np.asarray(slab.element_index, dtype=np.int64),
            np.asarray(slab.natural_coords, dtype=np.float64),
            view_data,
            node_coords_override=(
                np.asarray(scene.grid.points) if posed else None
            ),
        )
    except Exception:
        return None
    coords = np.asarray(coords, dtype=np.float64)
    if coords.size == 0:
        return None
    palette = _palette()
    color = getattr(palette, "info", None) or "#7aa2f7"
    n = coords.shape[0]
    return MeshLayer(
        layer_id=f"{pane_id}:gauss",
        points=PointSet(coords),
        cells=CellBlocks(
            {"vertex": np.arange(n, dtype=np.int64).reshape(-1, 1)},
        ),
        color=ColorSpec(mode="solid", solid_rgb=color),
        point_size=_GAUSS_POINT_SIZE,
        render_points_as_spheres=True,
        pickable=False,
    )


def _gauss_probe_component(
    results: "Results", stage_id: str,
) -> Optional[str]:
    """The component read purely for its integration-point addresses."""
    try:
        recorded = results.inspect.components(stage=stage_id).get("gauss", ())
    except Exception:
        return None
    tokens = sorted(str(t) for t in recorded)
    return tokens[0] if tokens else None


# =====================================================================
# Clips (§3 — the 0083 machinery, ownership moved viewer → view)
# =====================================================================


def _apply_clips(backend: Any, view: "MeshView") -> None:
    """Push this view's ACTIVE section planes onto the backend.

    The record's ``offset`` resolves to an origin and ``flipped`` folds
    into the normal's sign — the same resolution
    ``ClipPlane.spec`` does, done here because the session IR owns the
    planes now and the 0083 controller is never constructed on this
    path. Inactive planes are simply absent from the set.
    """
    from ..scene_ir import ClipPlaneSpec

    specs = []
    for clip in view.clips:
        if not clip.active:
            continue
        n = clip.normal
        origin = tuple(c * float(clip.offset) for c in n)
        if clip.flipped:
            n = (-n[0], -n[1], -n[2])
        specs.append(ClipPlaneSpec(origin=origin, normal=n))
    try:
        backend.set_clip_planes(specs)
    except AttributeError:
        # A backend with no clip support (a plain recorder) is not a
        # realize failure — only a view that HAS clips would notice.
        if specs:
            raise


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


__all__ = [
    "RealizedLayer", "RealizedPane", "realize_pane", "recorded_components",
]
