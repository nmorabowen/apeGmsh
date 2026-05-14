# Viewer Modernization — Overview

This directory holds the plan for porting widget-level patterns from ParaView into
apeGmsh's three viewers (`model.viewer`, `mesh.viewer`, `results.viewer`).

> **Status — 2026-05-14.** First wave merged to main via
> [PR #168](https://github.com/nmorabowen/apeGmsh/pull/168) (8 commits, +93 net tests,
> suite at 809 passed / 1 skipped). Plans **08, 01, 05** are substantially complete for
> `results.viewer`; plan **04** has steps 1 + 2 done (state coordinator + outline-selection
> migration). Plans 06, 02, 03, 07 are still pending. See per-plan status badges below.

For *visual* reference of the ParaView flows we're adapting, open
[`../paraview-flows/index.html`](../paraview-flows/index.html) — each plan doc links to the
specific action it grounds.

For the *rendering* / plotting-layer comparison (views, representations, render passes,
scalar bars, transparency), see [`plotting-comparison.md`](plotting-comparison.md).

For the *diagram catalog* comparison (fibers, gauss, nodes, contours, averaging — what
gets plotted vs ParaView's conventions), see
[`diagram-catalog-comparison.md`](diagram-catalog-comparison.md).

For the *region tools* comparison (section cuts, scope boxes, clip planes — tools that
restrict what's visible or what's integrated), see
[`region-tools-comparison.md`](region-tools-comparison.md).

For the *engine* comparison (what's actually under the wrapper layers — VTK, PyVista,
ParaView's stack), see [`engine-comparison.md`](engine-comparison.md).

## Three concrete pains, mapped to plan docs

The plan is anchored to three problems the developer is hitting today:

| Pain | Plan docs that fix it |
|---|---|
| **Event workflow is buggy** — selection callbacks, in-place manager mutation in `_rebuild_scene`, three viewers each invent their own callback graph | [`04-active-objects.md`](04-active-objects.md) |
| **Dock and layout handling is poor** — no state persistence, inline construction, no auto-populated View menu, no tabification | [`08-dock-layout-persistence.md`](08-dock-layout-persistence.md) |
| **`results.viewer` is barebones** — diagram settings panel is flat, scrubbing stutters with multiple diagrams, colormap UX is scattered, no shared LUTs | [`05-apply-autoapply.md`](05-apply-autoapply.md), [`06-color-map-editor.md`](06-color-map-editor.md), plus targeted result-side work flagged in [`future/`](future/) |

Everything else is supporting infrastructure or polish.

## Sequenced plan

The order below is calibrated to the pains above. Items higher up are either bug fixes
for current pain or architectural prerequisites for items below them. Items at the
bottom are polish — they ship real value but won't fix what hurts today.

| # | Plan doc | Status | What it ships | Cost |
|---|---|---|---|---|
| 1 | [`08-dock-layout-persistence.md`](08-dock-layout-persistence.md) | ✅ **Landed** (results.viewer) — PR #168 | Dock registry + `QSettings` save/restore + auto-populated View menu | 3–4d |
| 2 | [`04-active-objects.md`](04-active-objects.md) | 🟡 **Step 1 + 2 done** — PR #168; steps 3/4 pending | `ActiveObjects` singleton + Qt-signal cascade | 4d |
| 3 | [`01-output-dock.md`](01-output-dock.md) | ✅ **Landed** (results.viewer) — PR #168; mesh/model + VTK deferred | Output / log dock with Python traceback + VTK warning capture | 2d |
| 4 | [`05-apply-autoapply.md`](05-apply-autoapply.md) | ✅ **Landed** (4 widgets migrated) — PR #168 | Apply / Reset / Auto-Apply on diagram settings | 3d |
| 5 | [`06-color-map-editor.md`](06-color-map-editor.md) | ⏸ Pending — biggest remaining `results.viewer` feature | Shared color-map editor dock + LUT manager | 5d |
| 6 | [`03-outline-eye-icon.md`](03-outline-eye-icon.md) | ⏸ Pending | Clickable eye-icon visibility column on outline trees | 3d |
| 7 | [`02-view-frame-chrome.md`](02-view-frame-chrome.md) | ⏸ Pending (downgraded — see plotting-comparison.md) | Per-viewport toolbar extensibility hook | 0.5d |
| 8 | [`07-property-descriptors-settings.md`](07-property-descriptors-settings.md) | ⏸ Pending (speculative — no concrete trigger) | `Property[T]` descriptors + unified Settings dialog | 1w |

### What's "must-ship" vs "do if there's time"

- **Must-ship** (fixes acknowledged pain): **08, 04, 01, 05, 06.** Roughly **17 days**.
  After PR #168, **08 + 01 + 05 done; 04 half-done; 06 still pending** — the next
  natural item.
- **Polish** (real value, not painful today): **03, 02.** ~5 days.
- **Speculative investment**: **07.** ~1 week. Only worth it if you commit to migrating
  more than one style class — otherwise it's infrastructure that doesn't pay off.

Recommendation: pick up plan 06 next (depends on the `activeLayerChanged` signal from
plan 04 — already wired). Then finish plan 04 steps 3/4 (mesh/model migration) as
opportunistic follow-ups. Re-evaluate 03/02 only after 06 lands. Defer 07 until you
have a concrete second use case beyond `ContourStyle`.

## How to read a plan doc

Every plan doc follows the same template:

1. **Goal** — one sentence: what user-visible thing changes.
2. **Why** — pain it removes; ParaView pattern it borrows.
3. **ParaView reference** — link into [`../paraview-flows/`](../paraview-flows/) and
   file:line citations of the actual ParaView code.
4. **Files to add / modify** — exact paths in `src/apeGmsh/viewers/`.
5. **API sketch** — Python signatures or pseudocode, just enough to commit to a shape.
6. **Risks** — what might go wrong, what we mitigate, what we accept.
7. **Done criteria** — testable, binary. No "feels good."
8. **Out of scope** — explicitly *not* doing in this doc.

## Out of scope for this plan — see [`future/`](future/) for scaffolded stubs

The four comparison docs surfaced ten future candidates, now scaffolded as one-page
placeholders in [`future/`](future/). The full index is in [`future/README.md`](future/README.md).
Summary:

**Rendering layer** (from [`plotting-comparison.md`](plotting-comparison.md))
- [`future/render-quality.md`](future/render-quality.md) — depth peeling + still/interactive (~2d)
- [`future/representation-split.md`](future/representation-split.md) — Diagram → Source + Representation (~1–2 weeks)
- [`future/selection-overlay.md`](future/selection-overlay.md) — selection as overlay layer (~2d)

**Catalog** (from [`diagram-catalog-comparison.md`](diagram-catalog-comparison.md))
- [`future/derived-fields.md`](future/derived-fields.md) — Von Mises / principals / yield / damage (~1–2 weeks). **Highest-value catalog gap.**
- [`future/tensor-glyphs.md`](future/tensor-glyphs.md) — principal directions via `vtkTensorGlyph` (~3–4d)
- [`future/isosurfaces.md`](future/isosurfaces.md) — `isosurface` kind for 3D solids (~2–3d)

**Region tools** (from [`region-tools-comparison.md`](region-tools-comparison.md))
- [`future/scope-box.md`](future/scope-box.md) — 3D box widget filtering cells (~3d)
- [`future/slice-plane.md`](future/slice-plane.md) — drag-the-plane field-driven cut (~2d)
- [`future/clip-plane-everywhere.md`](future/clip-plane-everywhere.md) — generalize clip plane (~2d)
- [`future/interactive-section-cut.md`](future/interactive-section-cut.md) — drag editing with STKO live re-integration (~1 week)

Items not yet scaffolded (lower priority): DAG pipeline, session persistence
unification, plugin system for diagram types, multi-view layout.

## Done criteria for the whole plan

- [ ] Items 08, 04, 01, 05, 06 implemented and merged.
- [ ] `results.viewer` feels noticeably different to use: stable scrubbing, persistent
      layout, central color-map editor, debug-friendly output dock.
- [ ] Each of `model.viewer`, `mesh.viewer`, `results.viewer` uses the shared dock
      registry, layout persistence, output dock, and active-objects singleton.
- [ ] No regression in existing viewer tests.
- [ ] Demo: open `results.viewer`, add three diagrams, scrub time, tile docks the way
      you like, close it, reopen — same layout, stable behavior, tracebacks in the dock.
