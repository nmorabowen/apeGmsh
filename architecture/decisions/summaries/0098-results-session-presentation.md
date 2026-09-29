# ADR 0098 — Results presentation is a `ResultsSession` of views, not geometries of diagrams (summary)

**Status:** Accepted (2026-08-16) — design ratified in the owner workshop the same day; record
revised the same day after adversarial review. […]

Supersedes ADR 0058's product ontology and one-global-time cursor; amends ADR 0088 (D1–D3,
D6–D7), 0081, 0083, 0094 INV-10 (snapshot realization only) and narrows 0045 pick targets.

## Decision

1. **Two libraries, stacked.** `Results` is the data broker; `ResultsSession`
   (`apeGmsh.results.session`) is presentation: `panes[]` (`MeshView | PlotView`, each with a
   stable `id`), `time`, `time_linked`, `selection` (nodes XOR Gauss), `realize()` → scene IR.
   `s = results.session()`, `s.show()` (Qt), `s.render("a.png")`; `results.viewer()` becomes
   sugar for `results.session().show()` at S6. Snapshots hold panes, slots, time, style, clip,
   plots, selection; an old `schema_version` is skipped and renamed `.legacy`. The director,
   geometries and diagram-kind registry are not public API.
2. **Three authors, one session.** Python and the Qt app write it; Studio MCP realizes
   snapshots. The Qt app must be sufficient alone.
3. **Mesh view: analysis mesh only**, no BRep. **INV-MESH-1** one cell set; **INV-MESH-2**
   one surface (grey or contour); **INV-MESH-3** edges are derived; **INV-MESH-4** four
   style buttons (Mesh, Outlines, Nodes, Gauss). Deform is a pose, never a picture.
4. **Result slots: a closed catalog, unique per category.** `contour` (averaged |
   unaveraged), `vector`, `gauss`, `line`, `sand` (colour-mapped except `line`), `loads`,
   `reactions`. Filling an occupied slot replaces it. Fibers, layers, isochrones and springs
   get no eighth slot; a new slot is an amendment here.
5. **Legends are caused by slots.** **INV-LEGEND-1** cause: only occupied colour-mapped
   slots; **-2** presence; **-3** independence (hide is chrome); **-4** identity; **-5** per
   view. Deform on with every slot empty gives zero legends.
6. **Plot view** is a pane (`kind` history | path | xy, `series[]`, `cursor`).
7. **Time.** An instant is `(stage, step)`; linked means one instant for every pane,
   unlinked means per pane; a mode-posed view has no instant.
8. **Selection: nodes or Gauss only**, in ADR 0045's one `SelectionState` / `SelectionLog`
   (INV-5, no second store); four writers; the set is the plots' `source=`.
9. **Outline:** panes plus model composition. **10. Clients:** `session.show()`,
    `session.render()`, `results.plot`, Studio MCP; the Qt client lives in `apeGmsh.viewers`.
11. **Build order** S0–S6; `viewer()` flips to `session().show()` at S6. ADR 0084's
    one-reconciler discipline applies to session → realize → paint.

## Amendments

- A1 (2026-08-17), pane host: nested `QSplitter`s auto-tiled by `T(N)` in the central
  widget, one `QtInteractor` per mesh pane, no shell interactor. UI, not IR.
- A2 (2026-08-18), the six slotless kinds (`fiber_section`, `layer_stack`, three isochrone
  kinds, `spring_force`) survive internally behind the `show_web` hatch; the catalog stays
  at seven; the disposition expires with the hatch.
- A3 (2026-08-19), **reverses A1.1**: each pane is its own `QDockWidget`. The session is
  authoritative for pane existence, the saved `QSettings` arrangement advisory for placement.
- A4 (2026-08-19), a cursor-only change within one stage re-steps in place
  (`signature = (structure, cursor)`) with parity, zero-realize, one-render, no-bar-churn and
  pose-current-pick invariants; it refuses and falls back to a full realize otherwise.
- A5 (2026-08-19), reaches the IR: `MeshView` gains per-field legend placement
  `(anchor, font_scale)` read via `view.legend_placement(field)`; existence stays derived;
  placements for legends no longer caused are dropped loudly on restore.
- A6 (2026-08-20), reachability gates: G1 capability parity
  (`test_session_capability_parity.py`), G2 the picker law (`test_inspector_picker_law.py`),
  G3 real-gesture test deferred (`manual_legend_gesture.py`), G4 done.

Full text: [../0098-results-session-presentation.md](../0098-results-session-presentation.md).
Precedence (AGENTS.md): the code wins over the ADR, and the ADR wins over this summary.
