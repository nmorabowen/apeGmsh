# Future Plans — Index

Scaffolded ideas surfaced by the four comparison docs. Each is a planning placeholder,
not a committed plan. Pick one up after the must-ship list (08, 04, 01, 05, 06) lands.

Grouped by which comparison doc surfaced it.

## From [`../plotting-comparison.md`](../plotting-comparison.md)

- [`render-quality.md`](render-quality.md) — depth peeling + still-vs-interactive render lifecycle. ~2d.
- [`representation-split.md`](representation-split.md) — split `Diagram` into Source + Representation. **~1–2 weeks.**
- [`selection-overlay.md`](selection-overlay.md) — render selection as a separate overlay layer, fixes picking through transparency. ~2d.

## From [`../diagram-catalog-comparison.md`](../diagram-catalog-comparison.md)

- [`derived-fields.md`](derived-fields.md) — on-the-fly Von Mises / principals / yield / damage. **~1–2 weeks.** *Highest-value catalog gap.*
- [`tensor-glyphs.md`](tensor-glyphs.md) — principal-direction visualization via `vtkTensorGlyph`. ~3–4d.
- [`isosurfaces.md`](isosurfaces.md) — `isosurface` kind for 3D solid fields. ~2–3d.

## From [`../region-tools-comparison.md`](../region-tools-comparison.md)

- [`scope-box.md`](scope-box.md) — 3D box widget filtering visible cells. ~3d. *Standard CAD/BIM tool we're missing.*
- [`slice-plane.md`](slice-plane.md) — drag-the-plane field-driven cutting. ~2d.
- [`clip-plane-everywhere.md`](clip-plane-everywhere.md) — generalize clip plane to all three viewers, allow non-axis-aligned. ~2d.
- [`interactive-section-cut.md`](interactive-section-cut.md) — drag editing of `SectionCutDef` with live STKO re-integration. ~1 week.

## Where the engine comparison lands

[`../engine-comparison.md`](../engine-comparison.md) didn't surface new future docs —
it showed that every item above is a VTK feature already in our box. The engine choice
is correct; this list is just exploitation of what we already depend on.

## Suggested priority if you pick from this list

If you finish the must-ship plan (~17d) and want immediate user-visible wins:

1. **`render-quality.md`** — 4h of depth peeling alone is the single highest-leverage rendering win. Pair with plan 06.
2. **`scope-box.md`** — standard tool, ~3d, hits BIM/CAD muscle memory users expect.
3. **`derived-fields.md`** — the biggest *catalog* gap; biggest impact for stress analysis users.

The architectural ones (`representation-split.md`) are 1–2 weeks; defer until you have a concrete trigger.
