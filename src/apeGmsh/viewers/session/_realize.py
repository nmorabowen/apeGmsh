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
* **§8, selection** (S4-1): the node and Gauss clouds are emitted
  ``pickable`` — they ARE the pick targets — and their addresses ride
  out on :class:`PaneTargets` so a hit can only ever resolve to a
  point this pane is drawing. The session's ONE set is then marked as
  a ``<pane>:selection`` layer over those same posed coords, which is
  what "highlights follow the current pose" costs here: nothing.
  Because the highlight is realize output, the S2 reconciler's
  signature carries a selection term — without it a selection write
  ticks the session, compares equal, and the repaint is skipped.
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

import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from .._pump_set import _compose_substrate_points, read_nodal_vector_field
from ..core._clip_planes import MAX_ACTIVE_PLANES
from ..core._legend import LegendController, adopt_controller, key_str
from ..data import ViewerData
from ..diagrams._kinds import kind_def
from ..diagrams._scalar_bar_support import _default_vertical
from ..diagrams._visual_store import VisualDataStore
from ..scene.fem_scene import build_fem_scene
from ..scene_ir import ColorSpec, MeshLayer, PointSet
from ._gauss_addr import gp_index_within_element, probe_component
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

#: ONE :class:`VisualDataStore` per ``Results`` — the old viewer's
#: ownership rule (``_visual_store.py``: "one Results -> one director ->
#: one store"), which the session path lost.
#:
#: Realize used to construct a store per CALL. The store caches
#: full-time slabs precisely so scrubbing reads RAM instead of HDF5, so
#: a per-call store is a cache that is empty on every frame and dies
#: with it — and because ``ContourDiagram.attach`` pre-warms it, every
#: flush paid a full ``(T, N)`` read and threw it away. N panes meant N
#: copies of the same slab (ADR 0098 A4.1 Bug B).
#:
#: Weak keys so the cache dies with the ``Results`` it describes; the
#: existing ``APEGMSH_VIEWER_CACHE_BYTES`` LRU still bounds each entry.
_STORES: "weakref.WeakKeyDictionary[Any, VisualDataStore]" = (
    weakref.WeakKeyDictionary()
)


def visual_store_for(results: Any) -> VisualDataStore:
    """The one store for ``results``, created on first use."""
    store = _STORES.get(results)
    if store is None:
        store = VisualDataStore()
        _STORES[results] = store
    return store


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
    #: The emitted layer, and (for layers realize owns rather than a
    #: diagram) the rows of its SOURCE point array it was built from.
    #: A cursor-only re-step re-points the layer from those rows
    #: instead of rebuilding it (A4.2). ``rows=None`` means "all rows,
    #: in order".
    layer: Any = None
    rows: "Optional[np.ndarray]" = None


@dataclass(frozen=True, eq=False)
class NodeTargets:
    """The node pick targets of one pane — ids and their POSED coords.

    ``ids[i]`` is the FEM node id at ``coords[i]``. Row-aligned by
    construction: both come off the one cell set (INV-MESH-1) at the
    pose this flush realized, which is what makes §8's "highlights
    follow the current pose" free.
    """

    ids: "np.ndarray"          # (n,) int64 FEM node ids
    coords: "np.ndarray"       # (n, 3) posed world coords
    #: Rows of ``scene.grid.points`` the coords were read from, so a
    #: re-step can re-read them at the new pose (A4.2).
    rows: "Optional[np.ndarray]" = None


@dataclass(frozen=True, eq=False)
class GaussTargets:
    """The Gauss pick targets of one pane.

    ``(element_ids[i], gp_indices[i])`` is the §8 target at
    ``coords[i]``. ``gp_index`` is the row WITHIN that element, in the
    slab's own row order — the encoding ``_plots._gauss_series``
    resolves and ``SessionSelection`` stores.
    """

    element_ids: "np.ndarray"  # (n,) int64
    gp_indices: "np.ndarray"   # (n,) int64
    coords: "np.ndarray"       # (n, 3) posed world coords
    #: The slab's natural coords. With ``element_ids`` these are what
    #: ``compute_global_coords_from_arrays`` needs to re-place the
    #: cloud at a new pose — the same two arrays ``GaussPointDiagram``
    #: keeps for its own ``sync_substrate_points`` (A4.2).
    natural_coords: "Optional[np.ndarray]" = None


@dataclass(frozen=True, eq=False)
class PaneTargets:
    """What a click or a window can hit in this pane (§8).

    Populated per style button, because that is the law: "click-pick
    requires the matching style button on". A target family is ``None``
    when its button is off — not an empty array, which would say "the
    button is on and nothing is there".

    The Gauss family is populated whenever the button is on, INCLUDING
    when the `gauss` slot occupies and suppresses the button's own
    layer — the slot is drawing that cloud, with values, so those
    points are on screen and a click must still address them. In that
    case the targets are read off the SLOT's own cloud, so they cannot
    diverge from it.
    """

    nodes: Optional[NodeTargets] = None
    gauss: Optional[GaussTargets] = None


@dataclass(frozen=True, eq=False)
class PaneRestep:
    """What a cursor-only re-step needs and would otherwise rebuild.

    ADR 0098 A4.2. Everything here is a realize local that used to be
    dropped on the floor: the scene whose points the pose mutates, the
    view's ONE cell set as an extracted grid, and the rows that map the
    second back onto the first.

    ``substrate_rows`` is ``None`` for an unscoped view — there the
    scoped grid IS ``scene.grid`` and no gather is needed.
    ``PaneRestep`` being absent from a ``RealizedPane`` is how realize
    says "I could not record enough; re-realize instead".
    """

    scene: Any
    view_data: Any
    grid: Any
    substrate_rows: "Optional[np.ndarray]" = None
    #: The `gauss` SLOT's diagram when it occupies. The button's pick
    #: targets are read off ITS cloud (INV-MESH-4), so a re-step must
    #: read the same object rather than re-deriving a probe cloud.
    gauss_occupant: Any = None


@dataclass(frozen=True)
class RealizedPane:
    """One pane, realized. ``scalar_bars`` are the bar keys emitted to
    the backend, in ``view.legends()`` order (hidden scales register
    invisible and emit no bar — INV-LEGEND-3).

    ``diagrams`` and ``legend_controller`` keep the emitting objects
    alive for the caller's lifetime management (S2 drives
    ``update_to_step`` through them); tests may introspect, nothing
    else should.

    ``targets`` is S4-1's addition: the node / Gauss addresses this
    flush put on screen, which the pane's pick controller resolves a
    hit against. Derived from the same cell set and the same pose as
    the layers, so a pick can never address a point the pane is not
    drawing.
    """

    pane_id: str
    layers: tuple[RealizedLayer, ...]
    scalar_bars: tuple[str, ...]
    diagrams: tuple[Any, ...]
    legend_controller: Optional[LegendController]
    targets: PaneTargets = PaneTargets()
    #: A4.2 — what a cursor-only flush needs to re-step this pane in
    #: place. ``None`` means the fast path must refuse.
    restep: Optional[PaneRestep] = None
    #: A7 (R1) — ``(xmin, ymin, zmin, xmax, ymax, zmax)`` of this
    #: pane's cell set in REFERENCE geometry, which is what the ADR
    #: 0083 clip gizmo sizes its quad and arrow against.
    #:
    #: Reference, not the posed shape, and not negotiable: a quad
    #: derived from deformed points would breathe on every scrub
    #: frame, so the handle the user is holding would move while the
    #: plane did not. It is also what makes the gizmo free on the A4
    #: fast path — a re-step cannot change it, so nothing recomputes.
    #: SCOPED, because the quad should span what the pane draws rather
    #: than geometry it does not show. ``None`` when realize could not
    #: resolve one; the pane then draws no gizmo rather than guessing.
    reference_bounds: "Optional[tuple[float, float, float, float, float, float]]" = None


#: Layer roles realize OWNS — the ones a re-step has to move itself.
#: Every other key belongs to a slot diagram, which
#: ``sync_substrate_points`` already moved.
_REALIZE_OWNED_ROLES = ("substrate", "outlines", "nodes", "gauss", "selection")


def can_restep(realized: "Optional[RealizedPane]", view: "MeshView") -> bool:
    """Whether ``realized`` can be re-stepped in place (A4.4).

    Refusing is free: the caller falls back to a full realize, which is
    today's cost and today's behaviour. It refuses when realize could
    not record a row map it would need, and when a slot re-fits its LUT
    to the DISPLAYED values per step — reproducing that means
    re-registering the scalar bar, which the fast path does not do.
    ``contour`` is deliberately not such a kind: its scale comes from
    the store's whole-history limits and is fixed at attach.
    """
    if realized is None or realized.restep is None:
        return False
    for diagram in realized.diagrams:
        spec = getattr(diagram, "spec", None)
        if spec is None:
            return False
        if (
            getattr(spec, "kind", None) in _DEFAULT_LUT_KINDS
            and getattr(spec.style, "clim", None) is None
        ):
            return False
    if view.deform is None:
        # Nothing geometric moves, so no row map is needed at all.
        return True
    rs = realized.restep
    if rs.grid is not rs.scene.grid and rs.substrate_rows is None:
        # A scoped grid whose source rows were not recorded cannot be
        # re-pointed from the posed scene.
        return False
    for entry in realized.layers:
        role = entry.key.split(":", 1)[-1]
        if role == "outlines" and entry.rows is None:
            return False
        if role == "nodes" and entry.rows is None:
            return False
    return True


def restep_pane(
    session: "ResultsSession",
    view: "MeshView",
    realized: "RealizedPane",
    backend: Any,
) -> "RealizedPane":
    """Move ONE already-realized pane to a new instant, in place (A4.2).

    The cursor moved and nothing else did, so nothing is torn down and
    nothing is rebuilt: the diagrams take the new step through
    ``update_to_step`` (the call ``RealizedPane.diagrams`` was added
    for), the pose is recomposed onto the scene, and the layers realize
    owns are re-pointed from the row maps it recorded. Same
    step-then-deform order as the slot loop in ``_realize_mesh``, so
    the picture is the one a full realize would have produced — which
    the parity oracle asserts rather than assumes.

    Raises on anything unexpected; the caller reports and falls back to
    a full realize in the same flush, because a HALF re-stepped pane is
    the stale-picture failure this amendment exists to avoid.
    """
    from dataclasses import replace

    results = _require_results(session)
    rs = realized.restep
    if rs is None:
        raise RuntimeError("realize recorded no re-step context")
    stage_id, step = _resolve_instant(session, view, results)
    posed = view.deform is not None

    if posed:
        _apply_pose(view, results, rs.scene, stage_id, step)
    for diagram in realized.diagrams:
        diagram.update_to_step(step)
        if posed:
            diagram.sync_substrate_points(rs.scene.grid.points, rs.scene)

    if not posed:
        # The geometry did not move, so every layer realize owns and
        # every pick target is still exactly right at this instant.
        return realized

    scene_pts = np.asarray(rs.scene.grid.points)
    if rs.substrate_rows is not None:
        # The scoped grid is an extracted COPY; follow the posed scene.
        rs.grid.points = scene_pts[rs.substrate_rows]
    grid_pts = np.asarray(rs.grid.points)

    targets = _restep_targets(realized.targets, rs, scene_pts)

    layers: list[RealizedLayer] = []
    for entry in realized.layers:
        role = entry.key.split(":", 1)[-1]
        if role not in _REALIZE_OWNED_ROLES or entry.layer is None:
            layers.append(entry)          # a diagram owns it; already moved
            continue
        if role == "substrate":
            layer = replace(entry.layer, points=PointSet(grid_pts))
        elif role == "outlines":
            layer = replace(
                entry.layer, points=PointSet(grid_pts[entry.rows]),
            )
        elif role == "nodes" and targets.nodes is not None:
            layer = _nodes_layer(view.id, targets.nodes)
        elif role == "gauss" and targets.gauss is not None:
            layer = _gauss_layer(view.id, targets.gauss)
        elif role == "selection":
            layer = _selection_layer(view.id, session.selection, targets)
        else:
            layers.append(entry)
            continue
        if layer is None:
            layers.append(entry)
            continue
        backend.update_layer(entry.handle, layer)
        layers.append(replace(entry, layer=layer))

    return replace(realized, layers=tuple(layers), targets=targets)


def _restep_targets(
    targets: "PaneTargets", rs: "PaneRestep", scene_pts: "np.ndarray",
) -> "PaneTargets":
    """The §8 pick targets at the new pose.

    They must move with the picture or a click addresses where a point
    USED to be — ``PanePick`` and the window both read them off
    ``reconciler.realized``.
    """
    nodes = targets.nodes
    if nodes is not None and nodes.rows is not None:
        nodes = NodeTargets(
            ids=nodes.ids, coords=scene_pts[nodes.rows], rows=nodes.rows,
        )
    gauss = targets.gauss
    if gauss is not None:
        if rs.gauss_occupant is not None:
            # The slot drew this cloud and its own sync just posed it —
            # read it back so the targets cannot diverge from it.
            gauss = _gauss_targets_of(rs.gauss_occupant) or gauss
        elif gauss.natural_coords is not None:
            from apeGmsh.results._gauss_world_coords import (
                compute_global_coords_from_arrays,
            )

            coords = np.asarray(
                compute_global_coords_from_arrays(
                    gauss.element_ids,
                    gauss.natural_coords,
                    rs.view_data,
                    node_coords_override=scene_pts,
                ),
                dtype=np.float64,
            )
            gauss = GaussTargets(
                element_ids=gauss.element_ids,
                gp_indices=gauss.gp_indices,
                coords=coords,
                natural_coords=gauss.natural_coords,
            )
    return PaneTargets(nodes=nodes, gauss=gauss)


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
    store = visual_store_for(results)

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
    grid, substrate_rows = _scoped_grid(scene, scoped)
    if not occluded:
        substrate = _substrate_layer(view.id, grid, view.style.mesh)
        handle = backend.add_layer(substrate)
        layers.append(RealizedLayer(
            key=f"{view.id}:substrate",
            layer_id=substrate.layer_id,
            handle=handle,
            layer=substrate,
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
    # set, emitted after the slots so they read above the surface. The
    # same pass resolves the §8 pick targets: the glyph clouds ARE the
    # pick targets, so their addresses and their layers come off one
    # computation.
    style_layers, targets = _style_and_targets(
        view, scene, grid, scoped, view_data, results, stage_id,
        slot_diagrams,
    )
    for role, layer, rows in style_layers:
        handle = backend.add_layer(layer)
        layers.append(RealizedLayer(
            key=f"{view.id}:{role}",
            layer_id=layer.layer_id,
            handle=handle,
            layer=layer,
            rows=rows,
        ))

    # §8 — the selection, on THIS pane's pose and cell set. Emitted
    # last so it reads above every glyph it marks.
    highlight = _selection_layer(view.id, session.selection, targets)
    if highlight is not None:
        handle = backend.add_layer(highlight)
        layers.append(RealizedLayer(
            key=f"{view.id}:selection",
            layer_id=highlight.layer_id,
            handle=handle,
            layer=highlight,
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
        targets=targets,
        restep=PaneRestep(
            scene=scene,
            view_data=view_data,
            grid=grid,
            substrate_rows=substrate_rows,
            gauss_occupant=slot_diagrams.get("gauss"),
        ),
        reference_bounds=_reference_bounds(scene, substrate_rows),
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


def _scoped_element_ids(
    session: "ResultsSession", view: "MeshView", results: "Results",
) -> "Optional[np.ndarray]":
    """The element ids this view draws, or ``None`` for all of them.

    Reuses ``_scope.resolve_scope`` — the SAME resolver realize uses —
    so "what the picker offers" and "what the picture can paint" cannot
    disagree about which cells the pane means (A6.6). Any failure
    answers ``None``: the unscoped vocabulary is the old behaviour, and
    a broken scope must not silently empty every picker.
    """
    if view.scope is None:
        return None
    try:
        scoped = resolve_scope(view.scope, ViewerData.from_fem(results.fem))
    except Exception:
        return None
    return scoped.element_ids


def _components_in_scope(
    results: "Results",
    stage_id: str,
    element_ids: "Optional[np.ndarray]",
) -> "dict[str, set[str]]":
    """``{family: components}`` for the elements the view draws.

    Falls back to the whole stage when the reader cannot answer the
    scoped question — only the native reader implements it today, and a
    backend that cannot is no reason to empty a picker.
    """
    whole = {
        family: set(cols)
        for family, cols in results.inspect.components(stage=stage_id).items()
    }
    if element_ids is None:
        return whole
    reader = getattr(results, "_reader", None)
    scoped_fn = getattr(reader, "available_components_for_elements", None)
    if scoped_fn is None:
        return whole
    from apeGmsh.results.readers import ResultLevel

    for level in ResultLevel:
        if level.value not in ("gauss", "line_stations", "elements"):
            continue          # nodal families are not element-scoped
        try:
            whole[level.value] = set(
                scoped_fn(stage_id, level, element_ids)
            )
        except Exception:
            pass              # keep the unscoped answer for that family
    return whole


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

    **Scope-aware since A6.6.** It answered with the whole stage's set
    regardless of what the pane draws, so the inspector offered
    ``stress_zz`` on a view scoped to shells that recorded none and the
    refusal arrived only after the user picked it — measured as an
    identical six-component list under ``soil+raft``, ``slabs+wall`` and
    ``columns+beams`` while only the solids carried Gauss stress. Now
    the element-level families are asked about the view's OWN cells.
    Nodal components are not element-scoped and are unchanged.
    """
    results = session.results
    if results is None:
        return set(), set()
    try:
        stage_id, _step = _resolve_instant(session, view, results)
        components = _components_in_scope(
            results, stage_id, _scoped_element_ids(session, view, results),
        )
    except Exception:
        return set(), set()
    return (
        set(components.get("nodes", ())),
        set(components.get("gauss", ())),
    )


def recorded_line_components(
    session: "ResultsSession", view: "MeshView",
) -> "set[str]":
    """The ``line_stations`` columns the view's instant records.

    A sibling of :func:`recorded_components` rather than a third member
    of its tuple, which three call sites unpack today.

    The line slot reads a family of its own — `axial_force`,
    `bending_moment_*`, `torsion`, the quantities only a *sectioned*
    beam produces — and until this existed nothing sourced it. The
    inspector filtered its Line row against the nodal and Gauss sets,
    where a line component can never appear, so the row offered nothing
    and its Add button was permanently dead while `view.line = Line(...)`
    from a script drew the diagram perfectly well. Same §7 instant law
    as its sibling, the same empty-on-anything-missing rule (a disabled
    control, never an error), and the same A6.6 scope-awareness — a
    line diagram is refused on a scope holding no sectioned beam.
    """
    results = session.results
    if results is None:
        return set()
    try:
        stage_id, _step = _resolve_instant(session, view, results)
        components = _components_in_scope(
            results, stage_id, _scoped_element_ids(session, view, results),
        )
    except Exception:
        return set()
    return set(components.get("line_stations", ()))


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
        # The cell set IS the whole mesh: no extraction, no gather, and
        # the grid the layers read is the one the pose mutates.
        return grid, None
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
    sub_grid = grid.extract_cells(cells)
    # ``extract_cells`` stamps the source point rows; they are what
    # lets a re-step re-point this COPY from the posed scene grid
    # instead of extracting again (A4.2). Absent (older VTK, an
    # extraction that dropped it) -> no re-step for this pane.
    rows = sub_grid.point_data.get("vtkOriginalPointIds")
    return sub_grid, (
        None if rows is None else np.asarray(rows, dtype=np.int64)
    )


def _reference_bounds(
    scene: Any, substrate_rows: "Optional[np.ndarray]",
) -> "Optional[tuple[float, float, float, float, float, float]]":
    """The pane's cell set, bounded in REFERENCE geometry (A7 / R1).

    Read off ``scene.reference_points`` rather than the (possibly
    posed) grid, because the ADR 0083 gizmo quad is sized from this: a
    box that followed the deformation would resize the grab handle on
    every scrub frame.

    ``substrate_rows`` is the scoped grid's map back into the scene's
    point rows — exactly the map A4 already records for the re-step —
    so the scoped case costs one fancy-index and no second extraction.
    ``None`` for an unscoped view means "every point", which is the
    same convention ``_scoped_grid`` uses.
    """
    try:
        pts = np.asarray(scene.reference_points, dtype=float)
        if substrate_rows is not None:
            pts = pts[substrate_rows]
        if pts.size == 0:
            return None
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        return (
            float(lo[0]), float(lo[1]), float(lo[2]),
            float(hi[0]), float(hi[1]), float(hi[2]),
        )
    except Exception:
        # A backend or scene that cannot answer is not a realize
        # failure — only a view with a gizmo would notice, and it
        # degrades to drawing none.
        return None


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

#: The §8 highlight, drawn over whichever cloud it marks — bigger than
#: both so a selected glyph reads as marked, not merely recoloured.
_SELECTION_POINT_SIZE = 12.0


def _with_surface_edges(spec: Any, kind_id: str, style: Any) -> Any:
    """Ride the "Mesh" button onto an occluding slot's own style."""
    from dataclasses import replace

    if kind_id not in OCCLUDING_KINDS:
        return spec
    if not hasattr(spec.style, "show_edges"):
        return spec
    return replace(spec, style=replace(spec.style, show_edges=style.mesh))


def _style_and_targets(
    view: "MeshView",
    scene: "FEMSceneData",
    grid: Any,
    scoped: "ScopedSet",
    view_data: "ViewerData",
    results: "Results",
    stage_id: str,
    slot_diagrams: "dict[str, Any]",
) -> "tuple[list[tuple[str, Any, Any]], PaneTargets]":
    """``(role, layer)`` for every ON style button that draws its own
    layer, plus the §8 pick targets those buttons put on screen.

    "Mesh" is not here — it is ``show_edges`` of the surface. Nodes and
    Gauss are: their glyph clouds ARE the pick targets, so the
    addresses and the layer come off ONE computation of each cloud.
    """
    out: list[tuple[str, Any, Any]] = []
    nodes = gauss = None
    style = view.style
    if style.outlines:
        layer, rows = _outlines_layer(view.id, grid)
        if layer is not None:
            out.append(("outlines", layer, rows))
    if style.nodes:
        nodes = _node_targets(scene, scoped)
        if nodes is not None:
            out.append(("nodes", _nodes_layer(view.id, nodes), nodes.rows))
    if style.gauss:
        # "They must not draw two clouds" (INV-MESH-4): when the `gauss`
        # SLOT is occupied it already paints these points, with values
        # and a scale, so the button emits nothing — it stays lit
        # because it describes what is on screen. The TARGETS then come
        # from the SLOT's own cloud, not from a second read: the probe
        # component and the slot's quantity can live in different Gauss
        # groups (different class_tag / int_rule), and a probe cloud
        # would then address integration points this pane is not
        # drawing — plus it would pay a whole second slab read on every
        # flush.
        occupant = slot_diagrams.get("gauss")
        if occupant is not None:
            gauss = _gauss_targets_of(occupant)
        else:
            gauss = _gauss_targets(
                scene, scoped, view_data, results, stage_id,
                posed=view.deform is not None,
            )
            if gauss is not None:
                out.append(("gauss", _gauss_layer(view.id, gauss), None))
    return out, PaneTargets(nodes=nodes, gauss=gauss)


#: Point-data tag used to carry grid rows through the two extraction
#: hops the outline takes. Same device as the backend's
#: ``_OUTLINE_ROW_ARRAY`` (ADR 0089 D4) — one tag on the GRID survives
#: both ``_extract_surface_fast`` and ``extract_feature_edges``, so the
#: edges come back knowing which grid point each of them came from.
_OUTLINE_ROWS = "_apegmsh_outline_rows"


def _outlines_layer(
    pane_id: str, grid: Any,
) -> "tuple[Optional[MeshLayer], Optional[np.ndarray]]":
    """Element / feature boundaries of the cell set (ADR 0089 D1).

    Returns ``(layer, rows)`` where ``rows`` maps each outline point
    back to its row in ``grid`` — what a re-step scatters the posed
    points onto, exactly as the old viewer's DEFORM pump does through
    ``rs.outline_rows``.

    ``(None, None)`` when the cell set has no such edges — a 1-D beam
    model whose surface is already lines. An empty layer would be an
    actor that draws nothing; the missing key is what tells a client
    the button had nothing to draw.
    """
    from ..scene.mesh_scene import _extract_surface_fast
    from ..scene_ir import CellBlocks

    try:
        # The project's own boundary extraction (vtkGeometryFilter in
        # FastMode): the conservative path allocates a point-to-cell
        # hash at ~96 B per point, which is ~466 MB at 607k points —
        # and this now runs once per pane, per flush.
        grid.point_data[_OUTLINE_ROWS] = np.arange(
            grid.n_points, dtype=np.int64,
        )
        try:
            surface = _extract_surface_fast(grid)
            edges = surface.extract_feature_edges(
                feature_angle=_outline_feature_angle(),
                boundary_edges=True,
                feature_edges=True,
                manifold_edges=False,
                non_manifold_edges=False,
            )
        finally:
            # The tag served the extraction only. Leaving it on would
            # put a stray array on the SCENE grid (unscoped: this grid
            # IS scene.grid) that every later extraction would copy.
            try:
                del grid.point_data[_OUTLINE_ROWS]
            except KeyError:
                pass
    except Exception:
        return None, None
    if edges is None or edges.n_points == 0 or edges.n_lines == 0:
        return None, None
    conn = _two_point_lines(edges)
    if conn is None:
        return None, None
    rows = edges.point_data.get(_OUTLINE_ROWS)
    rows = None if rows is None else np.asarray(rows, dtype=np.int64).copy()
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
    ), rows


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


def _node_targets(
    scene: "FEMSceneData", scoped: "ScopedSet",
) -> "Optional[NodeTargets]":
    """The nodes of the VISIBLE cells, with their posed coords (§8).

    Read off ``scene.grid.points``, which the pose already moved, so
    the glyphs — and the picks and highlights that ride them — follow
    the deformed model without a second sync. ``scoped.node_ids`` is
    the node-of-visible-cells incidence, derived once per flush by
    ``resolve_scope`` and shared by every consumer here.
    """
    points = np.asarray(scene.grid.points)
    node_ids = np.asarray(scene.node_ids, dtype=np.int64)
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
    return NodeTargets(
        ids=node_ids[rows], coords=points[rows], rows=rows,
    )


def _nodes_layer(pane_id: str, targets: "NodeTargets") -> "MeshLayer":
    """Node glyphs — the click-pick affordance (§8).

    ``pickable=True`` because §8 names this cloud as what a click
    hits: a hit on a glyph resolves THROUGH the glyph rather than
    through whatever surface lies behind it. It is not what makes
    picking possible — the one surface stays pickable, so the
    ray-then-snap rule (:mod:`._pick`) answers a click on a face too —
    it is what makes the glyph itself a target, which is the law the
    style button gates.
    """
    from ..scene_ir import CellBlocks

    palette = _palette()
    color = getattr(palette, "node_accent", None) or "#e0e0e0"
    n = int(targets.ids.size)
    return MeshLayer(
        layer_id=f"{pane_id}:nodes",
        points=PointSet(targets.coords),
        cells=CellBlocks(
            {"vertex": np.arange(n, dtype=np.int64).reshape(-1, 1)},
        ),
        color=ColorSpec(mode="solid", solid_rgb=color),
        point_size=_NODE_POINT_SIZE,
        render_points_as_spheres=True,
        pickable=True,
    )


def _gauss_targets(
    scene: "FEMSceneData",
    scoped: "ScopedSet",
    view_data: "ViewerData",
    results: "Results",
    stage_id: str,
    *,
    posed: bool,
) -> "Optional[GaussTargets]":
    """Integration points of the cell set — LOCATIONS, not values.

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

    probe = probe_component(results, stage_id)
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
    element_index = np.asarray(slab.element_index, dtype=np.int64)
    return GaussTargets(
        element_ids=element_index,
        gp_indices=gp_index_within_element(element_index),
        coords=coords,
        natural_coords=np.asarray(slab.natural_coords, dtype=np.float64),
    )


def _gauss_targets_of(diagram: Any) -> "Optional[GaussTargets]":
    """The pick targets of an OCCUPYING `gauss` slot — its own cloud.

    ``GaussPointDiagram`` caches the slab's ``element_index`` and the
    world centers it drew, and ``sync_substrate_points`` keeps those
    centers posed — the same two arrays this module would otherwise
    re-derive. Reading them makes the targets equal to what is on
    screen BY CONSTRUCTION, which no second read of a probe component
    can promise (the probe may resolve to a different Gauss group
    entirely). ``None`` when the slot came up empty: an occupied slot
    that drew nothing has nothing to hit.

    Reaching into the diagram is this module's established idiom
    (``_layer`` / ``_handle`` / ``_visual_store`` are all read the same
    way) — the emitting objects are realize's own output.
    """
    element_index = getattr(diagram, "_gp_element_index", None)
    coords = getattr(diagram, "_coords", None)
    if element_index is None or coords is None:
        return None
    element_index = np.asarray(element_index, dtype=np.int64)
    coords = np.asarray(coords, dtype=np.float64)
    if element_index.size == 0 or coords.shape[0] != element_index.size:
        return None
    return GaussTargets(
        element_ids=element_index,
        gp_indices=gp_index_within_element(element_index),
        coords=coords,
    )


def _gauss_layer(pane_id: str, targets: "GaussTargets") -> "MeshLayer":
    """The integration-point glyph cloud (INV-MESH-4).

    ``pickable=True`` for the same reason the node cloud is (§8 names
    it as what a click hits) — and here it carries more weight:
    integration points sit INSIDE their elements, so a Gauss glyph is
    the only prop that can answer for one.
    """
    from ..scene_ir import CellBlocks

    palette = _palette()
    color = getattr(palette, "info", None) or "#7aa2f7"
    n = int(targets.coords.shape[0])
    return MeshLayer(
        layer_id=f"{pane_id}:gauss",
        points=PointSet(targets.coords),
        cells=CellBlocks(
            {"vertex": np.arange(n, dtype=np.int64).reshape(-1, 1)},
        ),
        color=ColorSpec(mode="solid", solid_rgb=color),
        point_size=_GAUSS_POINT_SIZE,
        render_points_as_spheres=True,
        pickable=True,
    )


# =====================================================================
# Selection highlight (§8) — the one set, on THIS pane's pose
# =====================================================================


def _selection_layer(
    pane_id: str, selection: Any, targets: "PaneTargets",
) -> "Optional[MeshLayer]":
    """The session selection, marked on this pane's own glyph cloud.

    The set is ONE (§8) and every pane marks it, but only over the
    points that pane is actually drawing: a node outside this view's
    scope has no coordinate here, and a set of the kind whose button
    is off has nothing to mark — "query-from-outline may fill the set
    with glyphs off; glyphs on is how you see it".

    Sizes and colour are chrome. What is NOT chrome is that the coords
    come from ``targets``, which this same flush posed: highlights
    follow the current pose, always, with no second sync.
    """
    from ..scene_ir import CellBlocks

    kind = selection.kind
    if kind == "nodes" and targets.nodes is not None:
        wanted = np.asarray(selection.nodes, dtype=np.int64)
        rows = np.flatnonzero(np.isin(targets.nodes.ids, wanted))
        coords = targets.nodes.coords[rows]
    elif kind == "gauss" and targets.gauss is not None:
        pairs = np.asarray(
            selection.gauss, dtype=np.int64,
        ).reshape(-1, 2)
        rows = np.flatnonzero(_pair_isin(
            targets.gauss.element_ids, targets.gauss.gp_indices, pairs,
        ))
        coords = targets.gauss.coords[rows]
    else:
        return None
    if rows.size == 0:
        # Selected, but not in this pane's picture. The missing key is
        # what tells a client the pane marked nothing.
        return None
    palette = _palette()
    color = getattr(palette, "accent", None) or "#ffb000"
    return MeshLayer(
        layer_id=f"{pane_id}:selection",
        points=PointSet(coords),
        cells=CellBlocks(
            {"vertex": np.arange(rows.size, dtype=np.int64).reshape(-1, 1)},
        ),
        color=ColorSpec(mode="solid", solid_rgb=color),
        point_size=_SELECTION_POINT_SIZE,
        render_points_as_spheres=True,
        pickable=False,
    )


def _pair_isin(
    element_ids: "np.ndarray", gp_indices: "np.ndarray",
    wanted: "np.ndarray",
) -> "np.ndarray":
    """Row-wise membership of ``(element_id, gp_index)`` in ``wanted``.

    Flattened to ONE integer key per pair — ``eid * stride + gp`` with
    ``stride`` taken from the data, so the encoding is exact rather
    than a guessed maximum integration-point count.
    """
    if wanted.size == 0 or element_ids.size == 0:
        return np.zeros(element_ids.size, dtype=bool)
    stride = int(max(
        int(gp_indices.max(initial=0)), int(wanted[:, 1].max(initial=0)),
    )) + 1
    have = element_ids * stride + gp_indices
    want = wanted[:, 0] * stride + wanted[:, 1]
    return np.isin(have, want)


# =====================================================================
# Clips (§3 — the 0083 machinery, ownership moved viewer → view)
# =====================================================================


def reclip_pane(view: "MeshView", backend: Any) -> int:
    """Re-cut an already-realized pane in place (ADR 0098 A7 / R1).

    The clip analogue of A4's :func:`restep_pane`, and it exists for a
    measured reason. ``pane.clips`` used to sit in the reconciler's
    STRUCTURE term, so every mouse-move of a section-plane gizmo drag
    tore down and rebuilt every layer, diagram and scalar bar: **240.8
    ms mean per frame** on the bench (``ssi_frame_wall``,
    ``c010_clip_gizmo_drag``, one pane), against the scrubber's 33 ms
    budget and a 17.7 ms scrub on the SAME pane. About 4 fps for a
    direct-manipulation gesture. A4's fast path could never help: it
    fires only when the CURSOR term moved alone, so ``can_restep`` was
    never consulted for a clip edit.

    Nothing is rebuilt because nothing needs to be. A clip edit changes
    which half-spaces cut the scene — never which actors exist — and
    :meth:`RenderBackend.set_clip_planes` re-stamps its set onto every
    live handle (ADR 0083 Part 2), so one call re-cuts the whole pane.
    Gizmo layers are ``clip_exempt`` and are skipped by that stamp, so
    a plane cannot slice its own handle.

    Deliberately delegates to :func:`_apply_clips` rather than
    reimplementing the resolution: ONE writer, so the fast path and a
    full realize cannot disagree about what ``offset`` and ``flipped``
    mean. That equivalence is the fast path's whole safety argument and
    it is gated, not assumed.

    Raises whatever the backend raises; the caller reports and falls
    back to a full realize in the same flush, exactly as A4.4 requires,
    because a HALF re-cut pane is the stale-picture class ADR 0084
    exists to prevent.
    """
    return _apply_clips(backend, view)


def _apply_clips(backend: Any, view: "MeshView") -> int:
    """Push this view's ACTIVE section planes onto the backend.

    The record's ``offset`` resolves to an origin and ``flipped`` folds
    into the normal's sign — the same resolution
    ``ClipPlane.spec`` does, done here because the session IR owns the
    planes now and the 0083 controller is never constructed on this
    path. Inactive planes are simply absent from the set.

    **Capped at** :data:`MAX_ACTIVE_PLANES`, and this is the one place
    that can be. §3 puts the cap here on purpose — "a render-time
    constraint enforced where planes meet a backend, not by the IR" —
    because it is a property of OpenGL, not of what a view may
    describe: `gl_ClipDistance` guarantees only six, and past that the
    driver is FREE TO IGNORE the extras. A seventh active plane
    therefore does not fail, it draws a model that looks cut and is
    not — the worst shape of wrong, because nothing anywhere says so.

    Returns how many active planes were dropped, so a caller with a
    place to say it can. Creation order decides which six survive,
    matching ``ClipPlaneSetController.add``: the first six to be
    activated keep cutting, and a seventh is the one that waits.

    The count is a RETURN rather than a log because this runs on every
    realize and every drag frame — a warning here would fire thirty
    times a second and teach the reader to ignore it. The UI refuses
    the seventh activation up front (the inspector's Section planes
    section), so reaching this cap at all means a script or a restored
    snapshot did it, and the inspector's note is what surfaces it.
    """
    from ..scene_ir import ClipPlaneSpec

    specs = []
    dropped = 0
    for clip in view.clips:
        if not clip.active:
            continue
        if len(specs) >= MAX_ACTIVE_PLANES:
            dropped += 1
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
    return dropped


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
        # A5.3(2) — a hand placement outranks the style's scale. The
        # style seeds size only while the scale still sits in the
        # automatic stack; once dragged, the view's record is what the
        # bar is rebuilt from, or every realize would revert the
        # gesture (A5.2).
        placement = view.legend_placement(legend.field)
        font_scale = getattr(style, "scalar_bar_scale", None)
        if placement is not None and placement.font_scale is not None:
            font_scale = placement.font_scale
        controller.register(
            key,
            handle.layer_id,
            handle,
            lut=_layer_lutspec(diagram),
            fmt=getattr(style, "fmt", "%.3g") or "%.3g",
            vertical=_default_vertical(
                getattr(style, "scalar_bar_vertical", None)
            ),
            font_scale=font_scale,
            visible=not legend.hidden,
        )
        if placement is not None:
            # ``set_anchor`` is what undocks an entry (slot -> None), so
            # it is the placement seed as well as the drag mutator; the
            # layout then honours the anchor instead of a stack slot.
            controller.set_anchor(key, placement.anchor)
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
    "GaussTargets", "NodeTargets", "PaneTargets", "RealizedLayer",
    "RealizedPane", "realize_pane", "reclip_pane", "recorded_components",
    "recorded_line_components",
]
