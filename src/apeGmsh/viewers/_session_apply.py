"""The results viewer's session-restore path, as a testable unit.

ADR 0084 D7 (PR 7), the other half of the ``_show_impl`` carve. This
was ``ResultsViewer._apply_session`` — reachable only through
``show()`` → ``_maybe_restore_session`` → a Qt dialog, which is why the
session round-trip test of PR 8 could not run headless.

**Mechanical extraction, not a redesign.** The body is moved verbatim;
what it used to reach for on ``self`` is now an explicit collaborator.
Two things are preserved to the letter:

* the ``dispatcher.session_batch()`` enter/exit bracketing;
* the ``None``-hole handling in ``session.diagrams`` (ADR 0084 D6 —
  the deserializer pads a hole per unbuildable spec so
  ``CompositionSnapshot.layer_indices`` stay positionally aligned).

ADR 0084 D3 / PR 9 removed the third: the post-batch hand patches
(substrate sync, off-seam clip re-cut, gizmos, cut faces, plane panel)
that re-ran RENDER-lane work the batch exit used to drop. Batch exit
now replays the RENDER lane itself, id-coalesced per handler, so this
path owns no compensation at all.

``_prompt_restore`` deliberately stays in the shell — it is Qt dialog
code.

Collaborators that resolve state late (``legends``, ``clip_planes``)
are zero-argument providers rather than pre-resolved objects, so the
extracted path reads them at exactly the point in the sequence the
closure did.
"""
from __future__ import annotations

from typing import Any, Callable, Optional


def _comp_name_for(gsnap: Any, comp_id: Optional[str]) -> Optional[str]:
    """Look up a composition's name in a snapshot by its saved id."""
    if comp_id is None:
        return None
    for c in gsnap.compositions:
        if c.id == comp_id:
            return c.name
    return None


def _geom_name_for(
    geom_snapshots: Any, geom_id: Optional[str],
) -> Optional[str]:
    if geom_id is None:
        return None
    for g in geom_snapshots:
        if g.id == geom_id:
            return g.name
    return None


def _none() -> Any:
    return None


class SessionController:
    """Applies a saved viewer session onto a live director.

    Every argument is a collaborator the old closure read off the
    ``ResultsViewer`` instance:

    ``director``
        The :class:`ResultsDirector` — owns the dispatcher (and its
        ``session_batch``), the diagram registry, the geometry tree,
        the stage/step cursor and ``bind_results``.
    ``results``
        The bound :class:`Results`; each restored diagram is
        constructed against it.
    ``legends`` / ``clip_planes``
        Zero-arg providers for the colour-scale registry (ADR 0081 L3)
        and the section-plane set (ADR 0083 P5). May return ``None``.

    The five post-batch refresh collaborators PR 7 carried over
    (``sync_substrate_visibility``, ``reapply_clip_planes``,
    ``clip_gizmos``, ``clip_cut_faces``, ``clip_planes_panel``) are
    gone: they were hand patches for RENDER-lane work the batch exit
    dropped, and the batch now replays that lane itself (ADR 0084 D3).
    """

    __slots__ = ("director", "results", "legends", "clip_planes")

    def __init__(
        self,
        *,
        director: Any,
        results: Any,
        legends: Callable[[], Any] = _none,
        clip_planes: Callable[[], Any] = _none,
    ) -> None:
        self.director = director
        self.results = results
        self.legends = legends
        self.clip_planes = clip_planes

    def apply(self, session: Any, win: Any) -> None:
        """Reconstruct each spec into a Diagram and rebuild the
        Geometry → Composition → Layer hierarchy.

        v2 sessions carry the full hierarchy (with deformation state
        per geometry). v1 sessions (no ``geometries`` block) bundle
        every restored layer into one "Restored" composition under
        the active geometry — same fallback as before.
        """
        import dataclasses

        from .diagrams._base import NoDataError
        from .diagrams._kinds import kind_def

        # Suppress per-add registry pumps during the bulk restore.
        # Without this, the registry observer fires K times for K
        # layers, and each fire re-pumps every other layer
        # (K(K+1)/2 cost). The batch context runs one full pump on
        # exit instead. The dispatcher always exists (ADR 0056 Part 3).
        _batch_cm = self.director.dispatcher.session_batch()
        _batch_cm.__enter__()
        # Colour-scale placement (ADR 0081 L3). Parked before the
        # diagrams attach: the legends do not exist yet, and each one
        # claims its snapshot as it registers.
        legends = self.legends()
        if legends is not None:
            try:
                legends.restore(getattr(session, "legends", ()))
            except Exception as exc:
                from ._failures import report
                report("ResultsViewer._apply_session(legends)", exc)
        # Section planes (ADR 0083 P5). Applied straight away: planes
        # belong to no diagram and the backend exists by now
        # (``bind_plotter`` ran before the restore), so stamping covers
        # every layer this restore is about to add. A legacy session
        # carries no block and restores an empty set.
        clip_planes = self.clip_planes()
        if clip_planes is not None:
            try:
                clip_planes.restore(
                    [
                        dataclasses.asdict(cp)
                        for cp in getattr(session, "clip_planes", ()) or ()
                    ],
                    apply_cuts=bool(getattr(session, "clip_apply_cuts", True)),
                    show_gizmos=bool(
                        getattr(session, "clip_show_gizmos", True),
                    ),
                    cut_face=bool(
                        getattr(session, "clip_cut_face", False),
                    ),
                )
            except Exception as exc:
                from ._failures import report
                report("ResultsViewer._apply_session(clip_planes)", exc)
        # ADR 0026 PR-stretch — the director's tag-map source is now
        # derived from the bound :class:`Results`, not a separate
        # ``_model_h5`` path field.  bind_results(self._results)
        # ensures the director's tag_map property resolves correctly
        # at construction time for any SectionCutDiagram restored
        # below.  The session payload's ``model_h5`` field is now
        # informational only (kept in the schema for one cycle so
        # older session JSONs still parse).
        try:
            self.director.bind_results(self.results)
        except Exception as exc:
            from ._failures import report
            report(
                "ResultsViewer._apply_session(bind_results)", exc,
            )
        n_added = 0
        n_skipped = 0
        # Build every Diagram instance and stash it at the same index
        # the session JSON used (None for skipped specs so layer_indices
        # references stay aligned).
        restored_layers: list[Any] = []
        for spec in session.diagrams:
            if spec is None:
                # A hole the deserializer padded for a spec it could not
                # rebuild (ADR 0084 D6) — hold the index, skip the work.
                n_skipped += 1
                restored_layers.append(None)
                continue
            kdef = kind_def(spec.kind)
            cls = kdef.diagram_class if kdef is not None else None
            if cls is None:
                n_skipped += 1
                restored_layers.append(None)
                continue
            try:
                if spec.kind == "section_cut":
                    tag_map = self.director.tag_map
                    diagram = cls(
                        spec, self.results, tag_map=tag_map,
                    )
                else:
                    diagram = cls(spec, self.results)
                self.director.registry.add(diagram)
                restored_layers.append(diagram)
                n_added += 1
            except NoDataError:
                n_skipped += 1
                restored_layers.append(None)
            except Exception as exc:
                from ._failures import report
                report(
                    f"ResultsViewer._apply_session({spec.kind})", exc,
                )
                n_skipped += 1
                restored_layers.append(None)

        geom_mgr = self.director.geometries
        geom_snapshots = getattr(session, "geometries", ()) or ()

        if geom_snapshots:
            # ── v2: rebuild the full geometry tree ────────────────
            # The bootstrap already created one "Geometry 1"; reuse
            # it for the first snapshot and add() for the rest.
            # ADR 0058 S2b — sessions saved before the ``visible``
            # flag existed (snapshot field None) map to "visible iff
            # active", reproducing their previous active-only render.
            # The active pointer is restored after this loop, so the
            # legacy mapping is deferred until then.
            legacy_visible_ids: list[str] = []
            for gi, gsnap in enumerate(geom_snapshots):
                if gi == 0 and len(geom_mgr.geometries) >= 1:
                    geom = geom_mgr.geometries[0]
                    geom_mgr.rename(geom.id, gsnap.name)
                else:
                    geom = geom_mgr.add(name=gsnap.name, make_active=False)
                geom_mgr.set_deformation(
                    geom.id,
                    enabled=bool(gsnap.deform_enabled),
                    field=gsnap.deform_field,
                    scale=float(gsnap.deform_scale),
                )
                # ADR 0058 S3a — restore via the owner mutator (no-op
                # at the zero default; legacy sessions read (0,0,0)).
                geom_mgr.set_offset(geom.id, tuple(gsnap.offset))
                geom_mgr.set_display(
                    geom.id,
                    show_mesh=bool(gsnap.show_mesh),
                    show_nodes=bool(gsnap.show_nodes),
                    display_opacity=float(gsnap.display_opacity),
                )
                if gsnap.visible is None:
                    legacy_visible_ids.append(geom.id)
                else:
                    geom_mgr.set_visible(geom.id, bool(gsnap.visible))
                # Compositions inside this geometry.
                for csnap in gsnap.compositions:
                    comp = geom.compositions.add(
                        name=csnap.name, make_active=False,
                    )
                    for idx in csnap.layer_indices:
                        if 0 <= idx < len(restored_layers):
                            d = restored_layers[idx]
                            if d is not None:
                                geom.compositions.add_layer(comp.id, d)
                # Restore active composition pointer.
                if gsnap.active_composition_id is not None:
                    # Find the composition we just added by its
                    # POSITION — saved ids are stale UUIDs.
                    matches = [
                        c for c in geom.compositions.compositions
                        if c.name == _comp_name_for(
                            gsnap, gsnap.active_composition_id,
                        )
                    ]
                    if matches:
                        geom.compositions.set_active(matches[0].id)
                # Heal stale sessions: pre-#71/#72 the active id could
                # be saved as None (Esc / Geometry-row click). Without
                # an active comp the gate hides every layer on restore,
                # so default to the first composition when the user
                # didn't explicitly pick one.
                if (
                    geom.compositions.active is None
                    and geom.compositions.compositions
                ):
                    first = geom.compositions.compositions[0]
                    geom.compositions.set_active(first.id)
                # ADR 0058 S3b — restore the stage pin via the owner
                # mutator, AFTER the composition/layer loop so the
                # director's reattach observer fires against recorded
                # membership, once, instead of churning per layer.
                # No-op at the None default (legacy sessions).
                geom_mgr.set_stage_pin(
                    geom.id, getattr(gsnap, "stage_id", None),
                )
                # ADR 0084 D1 — the scalar threshold, keyed by the
                # geometry's LIVE id (the saved one is a stale UUID,
                # which is exactly why the spec rides inside the
                # geometry snapshot). Legacy sessions carry None and
                # this is a no-op; the batch exit's STEP pump applies
                # whatever landed here.
                tsnap = getattr(gsnap, "threshold", None)
                if tsnap is not None:
                    self.director.thresholds.set_threshold(
                        geom.id,
                        component=tsnap.component,
                        lo=tsnap.lo, hi=tsnap.hi,
                        topology=tsnap.topology,
                    )
            # Restore active geometry pointer (by name match — saved
            # UUIDs don't survive a re-bootstrap).
            saved_active = _geom_name_for(
                geom_snapshots, session.active_geometry_id,
            )
            if saved_active:
                for g in geom_mgr.geometries:
                    if g.name == saved_active:
                        geom_mgr.set_active(g.id)
                        break
            # Legacy (pre-S2b) snapshots: visible = is-active, now
            # that the active pointer reflects the saved session.
            for gid in legacy_visible_ids:
                geom_mgr.set_visible(gid, gid == geom_mgr.active_id)
        else:
            # ── v1 fallback: bundle into one "Restored" composition ─
            real_layers = [d for d in restored_layers if d is not None]
            if real_layers:
                geom = geom_mgr.active
                if geom is not None:
                    comp = geom.compositions.add(
                        name="Restored", make_active=True,
                    )
                    for d in real_layers:
                        geom.compositions.add_layer(comp.id, d)

        # Restore active stage / step where possible.
        try:
            if session.active_stage_id and (
                self.director.stage_id != session.active_stage_id
            ):
                self.director.set_stage(session.active_stage_id)
            if session.active_step:
                self.director.set_step(int(session.active_step))
        except Exception:
            pass

        # Close the colour-scale restore window (ADR 0081 L3). Every
        # diagram the session carries has attached by now, so anything
        # still parked names a legend this restore did not recreate —
        # left in place it would ambush a diagram the user adds later.
        if legends is not None:
            try:
                legends.end_restore()
            except Exception:
                pass

        # Flush the batch — runs STEP + DEFORM + GATE, replays the
        # RENDER lane, drains the UI lane, then RENDER once, against
        # everything that was added during the loop above.
        #
        # ADR 0084 D3 / PR 9 (fork F-A): every post-batch compensation
        # this method used to run by hand is now the batch's own job.
        #
        # * the substrate sync rides GEOMETRY_ADDED /
        #   GEOMETRY_ACTIVE_CHANGED / GEOMETRY_VISIBILITY_CHANGED —
        #   whichever of those the restore actually caused above. A
        #   one-geometry session causes none of them (it reuses the
        #   bootstrap geometry), and needs none: the only substrate
        #   work it owes is the occlusion recompute for its restored
        #   layers, and that rides ``_apply_geometry_display`` on the
        #   plain ``geometries`` observer chain, which the dispatcher
        #   batch does not suppress;
        # * the off-seam clip re-cut, the gizmos and the cut faces ride
        #   CLIP_PLANES_CHANGED, which ``clip_planes.restore()`` fires
        #   from ``_reconcile_and_fire`` above;
        # * the plane panel is a UI-lane subscriber the exit drains.
        #
        # Do not re-add hand patches here — a missing sync is a missing
        # subscription or a missing owner-fire, and both are global.
        try:
            _batch_cm.__exit__(None, None, None)
        except Exception:
            pass

        try:
            msg = f"Restored {n_added} diagram(s)"
            if n_skipped:
                msg += f"; {n_skipped} skipped"
            win.set_status(msg, timeout=5000)
        except Exception:
            pass


__all__ = ["SessionController"]
