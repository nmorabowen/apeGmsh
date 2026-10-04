### ADDED — apeGmshViewer draws the geometry sibling, shows a stale-geometry notice, and jumps to source from the inspector (ADR 0112 P1, slice V2f phase 2b, #1309)

The store now holds the `/geometry` sibling: a `fileLoaded` event for the
geometry artifact, its render blobs, and a geometry → mesh phase axis with a
phase bar. The viewport draws the sibling's curves, faces and points on the
geometry phase, or alone when no model is open. The sibling is drawn only when
its `/meta/session_id` equals the model's; otherwise the banner says
`Geometry not drawn: … is stale or foreign`, with both session ids.
`/provenance` is read with the model and joined onto the declarations by
declaration path. Every declaration in the inspector's definition chain gets
a source button (`requestSource` → effects → `goToSource`), disabled with the
reason when there is no record. Unnamed `#k` objects and elements are never
joined. A refused geometry or provenance zone is a refusal sentence, and the
model still loads. The app's version check follows ADR 0113 for every zone
through one function and one table (`ZONE_FLOOR`): neutral 2.10, opensees
2.11 (it was 2.0), geometry 1.0 and provenance 1.0. A file below its floor,
or of another major, is refused with a sentence naming the floor. A newer
same-major minor opens with one banner. An older minor at or above the floor
opens silently, which retires the two-version warning window. The zone readers now also refuse: a backslash path in
`/provenance`, a duplicate `seq`, `pg < 1` on a physical-group row, an `ok`
outside {0, 1}, an entity listed twice, and a membership of an unlisted
entity.
