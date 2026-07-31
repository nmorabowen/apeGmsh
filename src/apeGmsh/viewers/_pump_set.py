"""The results viewer's dispatcher pumps, as a testable unit.

ADR 0084 D7 (PR 7). These four primitives — STEP, DEFORM, GATE,
RESTACK — used to be closures defined inside
``ResultsViewer._show_impl`` and handed to ``Dispatcher.bind()`` a few
hundred lines later. That made them unreachable without a real window:
every headless test of the pump semantics had to re-type the pump body
(see ``tests/viewers/test_gate_effective_visibility.py`` before this
PR), which is the definition of an untested seam.

This is a **mechanical extraction, not a redesign** — the pump bodies
are moved verbatim; the only change is that what they used to capture
from the enclosing ``_show_impl`` frame is now an explicit constructor
parameter or a narrow callback. That parameter list *is* the frozen
interface the rest of ADR 0084 builds on (PR 8's scrub-values tests,
PR 9's batch-exit protocol, PR 10's single STEP path).

Nothing here imports a render backend or touches an actor flag
directly: the pumps talk to diagrams (``update_to_step`` /
``sync_substrate_points`` / ``apply_effective_visibility`` /
``attach``) and to scene point arrays. This module is guarded by the
ADR 0056 AST contract with a zero budget on all three guards
(``tests/viewers/test_viewer_state_contract.py``) so the extraction
cannot become a new home for the direct-render debt it was carved out
of.

``_pump_failed``, ``_gate_visible_layer_ids``,
``_compose_substrate_points`` and ``sync_render_surface_points`` moved
here with the pumps (they were module-level helpers in
``results_viewer.py`` serving only these bodies); ``results_viewer``
re-exports all four, so their existing import paths still resolve.
"""
from __future__ import annotations

from typing import Any, Callable


def _pump_failed(action: str, exc: BaseException, **payload: Any) -> None:
    """Report a dispatcher-pump failure (ADR 0084 D4 — pumps are loud).

    The catch stays (one bad diagram must not kill the viewport), but
    the failure is logged like ``_pump_restack`` does AND pushed
    through the :mod:`._failures` handler registry, where the strict
    test fixture collects it and fails the test at teardown.
    """
    from ._log import log_error
    from ._failures import report
    log_error("dispatch", action, exc, **payload)
    report(f"pump.{action}", exc)


def _gate_visible_layer_ids(geom_mgr: Any) -> "set[int]":
    """Layer ids (``id(layer)``) the composition gate may show.

    ADR 0058 S2b — concurrent rendering: every geometry with
    ``visible=True`` contributes its layers. Per geometry the existing
    composition-gate semantics are preserved: when a composition is
    active there, only that composition's layers; otherwise all of its
    compositions' layers. A layer is then shown iff
    ``layer.is_visible AND id(layer) in this set`` (the GATE pump
    composes the two).

    Module-level (not a ``show()`` closure) so the truth table is
    testable headless.
    """
    visible_layers: set[int] = set()
    for geom in geom_mgr.geometries:
        if not geom.visible:
            continue
        active_comp = geom.compositions.active
        if active_comp is not None:
            visible_layers.update(map(id, active_comp.layers))
        else:
            for c in geom.compositions.compositions:
                visible_layers.update(map(id, c.layers))
    return visible_layers


def _compose_substrate_points(
    reference_points: Any,
    offset: Any,
    field_vals: Any,
    scale: float,
) -> Any:
    """DEFORM pump point composition (ADR 0058 S3a).

    ``reference + offset + scale·field`` — the geometry's spatial
    offset is a pump-time term, never an actor transform and never
    baked into ``reference_points`` (world coordinates stay grid
    coordinates — the S2c picking invariant).

    Returns ``None`` only when there is nothing to apply (no field AND
    zero offset) — the byte-identical legacy fast-path: the pump then
    resets ``grid.points`` to reference and tells diagrams "back to
    reference". A deform-off geometry with a non-zero offset returns
    ``reference + offset``.

    Module-level (not a ``show()`` closure) so the composition rule is
    testable headless.
    """
    import numpy as np

    off = (
        np.asarray(offset, dtype=np.float64)
        if offset is not None else None
    )
    has_offset = off is not None and bool(np.any(off != 0.0))
    if field_vals is None and not has_offset:
        return None
    pts = np.asarray(reference_points, dtype=np.float64).copy()
    if has_offset:
        pts += off
    if field_vals is not None:
        pts += float(scale) * np.asarray(field_vals, dtype=np.float64)
    return pts


def sync_render_surface_points(g_scene: Any, pts: Any = None) -> None:
    """Scatter substrate points onto the pre-extracted render surface.

    The O(surface) half of the substrate fast path: after the DEFORM
    pump writes ``g_scene.grid.points`` (a cheap memcpy now that no
    mapper consumes the grid), this gathers the rows the surface
    actually renders through ``render_surface.point_ids``. ``pts`` is
    the freshly-composed point array when the caller has it; ``None``
    reads the grid's current points. No-op for scenes without a render
    surface (headless builds, extraction fallback). The node cloud
    needs no counterpart — its glyph polydata feeds the mapper
    directly, so ``_sync_node_cloud``'s point write was never paying
    the volumetric re-extraction.
    """
    import numpy as np

    rs = getattr(g_scene, "render_surface", None)
    if rs is None:
        return
    target = np.asarray(g_scene.grid.points if pts is None else pts)
    if target.ndim != 2 or (
        rs.point_ids.size and int(rs.point_ids.max()) >= target.shape[0]
    ):
        return
    rs.surface.points = target[rs.point_ids]


class PumpSet:
    """The four dispatcher pump primitives with frozen, explicit inputs.

    Constructed once per ``show()`` and bound straight into
    ``Dispatcher.bind(pump_step=…, pump_deform=…, pump_gate=…,
    pump_restack=…)``. Every collaborator the closures used to capture
    from ``_show_impl`` is a constructor argument:

    ``director``
        The :class:`ResultsDirector`. The one irreducibly viewer-bound
        dependency — the pumps read ``registry``, ``geometries``,
        ``step_index``, ``local_step_for_stage``, ``scene_for``,
        ``view`` and ``_scene_for_diagram`` off it, and every one of
        those is director-owned state, not shell state.
    ``scene``
        The bound :class:`FEMSceneData` used as the fallback whenever
        ``director.scene_for(geom)`` yields nothing.
    ``read_deform_field(field, step, stage_id=None)``
        Returns the ``(N, 3)`` vector field for one stage, or ``None``.
        A callback because the reader owns the viewer's per-(stage,
        component) visual-store column cache and the bound
        :class:`Results` handle.
    ``render_geometries()``
        The geometries participating in this frame (ADR 0058 S2b:
        every ``visible`` one).
    ``sync_node_cloud(deformed_pts)``
        Pushes the frame onto the active geometry's node-cloud overlay
        (actor-owning, so it stays on the shell side of the seam).
    ``sync_diagram_substrate_points(pts, *, geometry, scene)``
        The deformation fan-out over the registry's layers.
    ``on_failure(action, exc, **payload)``
        ADR 0084 D4 — the loud-failure sink. Defaults to
        :func:`_pump_failed` (``log_error`` + the ``_failures``
        registry the strict pytest fixture collects from).
    """

    __slots__ = (
        "director", "scene", "read_deform_field", "render_geometries",
        "sync_node_cloud", "sync_diagram_substrate_points", "on_failure",
    )

    def __init__(
        self,
        *,
        director: Any,
        scene: Any,
        read_deform_field: Callable[..., Any],
        render_geometries: Callable[[], list],
        sync_node_cloud: Callable[[Any], None],
        sync_diagram_substrate_points: Callable[..., None],
        on_failure: Callable[..., None] = _pump_failed,
    ) -> None:
        self.director = director
        self.scene = scene
        self.read_deform_field = read_deform_field
        self.render_geometries = render_geometries
        self.sync_node_cloud = sync_node_cloud
        self.sync_diagram_substrate_points = sync_diagram_substrate_points
        self.on_failure = on_failure

    # ── Pipeline primitives — see _dispatch.py for the contract ──
    # The dispatcher is the only place that composes these into
    # event-driven sequences. Don't call them directly from observers —
    # fire a dispatcher event instead.

    def compute_deformed_pts(self, geom, step: int) -> Any:
        """Resolve ``geom``'s substrate points at ``step``.

        ADR 0058 S3a: ``reference + offset + scale·field``. Returns
        non-None whenever deformation contributes OR the geometry
        carries a non-zero spatial ``offset`` (a deform-off offset
        geometry yields ``reference + offset``). Returns None only
        for the legacy fast-path — no deformation (disabled / no
        field / unreadable) AND zero offset; the caller then resets
        the substrate to reference.

        ADR 0058 S1: geometry-parameterized — the baseline comes
        from the geometry's own scene (today the single bound
        scene; per-geometry in S2). The field reader stays
        geometry-agnostic: every geometry's scene indexes the same
        model, so ``node_ids`` / the id→row map are shared.

        ADR 0058 S3b: pinned-or-active — a geometry with a stage
        pin reads its PINNED stage's field at the global cursor
        clamped into the pinned range
        (``director.local_step_for_stage``); unpinned geometries
        keep the active stage + raw step.
        """
        director = self.director
        if geom is None:
            return None
        field_vals = None
        if geom.deform_enabled and geom.deform_field:
            pin = getattr(geom, "stage_id", None)
            if pin:
                field_vals = self.read_deform_field(
                    geom.deform_field,
                    int(director.local_step_for_stage(pin)),
                    stage_id=pin,
                )
            else:
                field_vals = self.read_deform_field(
                    geom.deform_field, int(step),
                )
        g_scene = director.scene_for(geom) or self.scene
        return _compose_substrate_points(
            g_scene.reference_points,
            getattr(geom, "offset", None),
            field_vals,
            float(geom.deform_scale),
        )

    def effective_step_for(self, layer) -> int:
        """Step to push to ``layer`` (ADR 0058 S3b — pin-aware).

        A layer owned by a stage-PINNED geometry steps through the
        pinned stage: the global cursor clamped into its range via
        ``director.local_step_for_stage``. Unpinned (or ownerless)
        layers step through the ACTIVE stage — which outside combined
        mode is the global cursor itself, and inside combined mode is
        that cursor translated onto whichever real stage owns it
        (``director.local_step_for_active_stage``).

        ADR 0084 D2: once pumps are bound this is the *only* place a
        step is resolved for a diagram, so both translations have to
        live here. They used to be split — the Director's direct pump
        applied the combined translation and no pin clamp, this pump
        applied the pin clamp and no translation, and whichever ran
        last won. In combined mode that was the Director's raw global
        cursor arriving at a stage-scoped read.
        """
        director = self.director
        geoms = director.geometries
        owner = geoms.geometry_for_layer(layer)
        pin = getattr(owner, "stage_id", None) if owner is not None else None
        if pin:
            return int(director.local_step_for_stage(pin))
        return int(director.local_step_for_active_stage())

    def pump_step(self, layer) -> None:
        """STEP primitive — push current step values.

        ADR 0058 S3b: pin-aware — the full path loops the registry
        pushing each diagram's effective step (clamped for layers
        of pinned geometries, raw otherwise); the layer-scoped
        path resolves its one owner the same way.
        """
        if layer is not None:
            try:
                layer.update_to_step(self.effective_step_for(layer))
            except Exception as exc:
                self.on_failure("step", exc, diagram=id(layer))
            return
        for d in self.director.registry.diagrams():
            if not (d.is_attached and d.is_visible):
                continue
            try:
                d.update_to_step(self.effective_step_for(d))
            except Exception as exc:
                # Loud, but keep pumping — one bad diagram must not
                # strand the rest of the registry at the old step.
                self.on_failure("step", exc, diagram=id(d))

    def pump_deform(self, layer) -> None:
        """DEFORM primitive.

        ``layer=None``: full pump — for every rendering geometry
        (ADR 0058 S2b: every geometry with ``visible=True``),
        recompute its deformed_pts from ITS deform state, mutate
        its scene's ``grid.points``, sync the node cloud (active
        geometry only — it is the editing overlay), and fan out
        to THAT geometry's diagrams. Hidden geometries' diagrams
        are gate-hidden and re-pumped when shown again
        (``GEOMETRY_VISIBILITY_CHANGED`` runs DEFORM), so they
        never show stale positions.

        ``layer=<diagram>``: scoped — sync that one diagram against
        its OWNING geometry's state. Used after a single layer's
        attach / re-attach so existing diagrams aren't re-pumped.
        """
        director = self.director
        step = int(director.step_index)
        geoms = director.geometries
        if layer is not None:
            geom = geoms.geometry_for_layer(layer) or geoms.active
            g_scene = director.scene_for(geom) or self.scene
            deformed_pts = self.compute_deformed_pts(geom, step)
            try:
                layer.sync_substrate_points(deformed_pts, g_scene)
            except Exception as exc:
                self.on_failure("deform", exc, diagram=id(layer))
            return
        for geom in self.render_geometries():
            g_scene = director.scene_for(geom) or self.scene
            deformed_pts = self.compute_deformed_pts(geom, step)
            if deformed_pts is None:
                g_scene.grid.points = g_scene.reference_points.copy()
            else:
                g_scene.grid.points = deformed_pts
            # Fast path: the grid write above no longer reaches a
            # mapper — scatter the frame onto the render surface.
            sync_render_surface_points(g_scene, deformed_pts)
            if geom is geoms.active:
                self.sync_node_cloud(deformed_pts)
            self.sync_diagram_substrate_points(
                deformed_pts, geometry=geom, scene=g_scene,
            )

    def pump_gate(self) -> None:
        """GATE primitive — composition-based actor visibility.

        ADR 0058 S2b: a layer is shown iff ``layer.is_visible AND
        composition gate AND owning_geometry.visible``. Every
        geometry with ``visible=True`` contributes layers (per
        geometry: the active composition's layers when one is
        active there, else all of its compositions' layers — see
        :func:`_gate_visible_layer_ids`). Layers owned by hidden
        geometries are gate-hidden, so toggling a geometry's
        ``visible`` flag drops / restores its diagrams without
        touching their user-intent ``is_visible`` flags.

        No render here; the dispatcher coalesces RENDER per event.
        """
        visible_layers = _gate_visible_layer_ids(self.director.geometries)
        for d in self.director.registry.diagrams():
            in_active = id(d) in visible_layers
            desired = bool(d.is_visible) and in_active
            # Polymorphic: migrated diagrams route backend layer
            # handles, legacy ones flip raw actors. The user-intent
            # flag (is_visible) is preserved — only the rendered
            # artifacts follow the gate.
            try:
                d.apply_effective_visibility(desired)
            except Exception as exc:
                # Loud, but keep gating the remaining layers.
                self.on_failure("gate", exc, diagram=id(d))

    def pump_restack(self) -> None:
        """Re-stack actors so paint order matches the layer order
        in the registry.

        VTK paints actors in the order they were added; reordering
        via the ↑ / ↓ buttons updates the registry list but doesn't
        move the actors. Detach + re-attach each diagram in order
        (cheap when the polydata is cached on the instance).
        """
        # Re-attach through the registry's RenderBackend (ADR 0042
        # R-B.final) — attach injects a backend, not a raw plotter.
        # ADR 0058 S2a: resolve each diagram's scene through its
        # owning geometry so a restack doesn't re-bind a layer to
        # the wrong substrate.
        director = self.director
        backend = director.registry.backend
        for d in list(director.registry.diagrams()):
            if not d.is_attached:
                continue
            try:
                d.detach()
                d.attach(
                    backend, director.view,
                    director._scene_for_diagram(d),  # noqa: SLF001
                )
            except Exception as exc:
                # d is now detached but failed to re-attach; log the
                # error so the broken state is visible rather than
                # silently swallowed.
                from ._log import log_error
                log_error("dispatch", "restack", exc, diagram=id(d))
                continue


__all__ = [
    "PumpSet",
    "sync_render_surface_points",
    "_compose_substrate_points",
    "_gate_visible_layer_ids",
    "_pump_failed",
]
