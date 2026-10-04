### ADDED — apeGmshViewer reads the `/geometry` and `/provenance` zones; "Open with" registration; case-safe pairing (ADR 0112 P1, slice V2f phase 2a, #1309)

`apeGmshViewer/src/reader/geometry.ts` reads a `<stem>.geometry.h5` sibling
and pairs it with its model only when both carry an equal `/meta/session_id`,
with the reason when they do not. `src/reader/provenance.ts` reads
`/provenance` and finds the source line of a declaration path. Both follow
`h5-schema.md`: an absent zone is ignored, and a zone outside its version
window is refused. Each also refuses a broken table, an index out of range,
an unknown enumeration or an int64 column, naming the HDF5 path. The
installer now adds the app to the "Open with" list of `.h5` files instead of
making it the default, and the uninstaller removes the keys it wrote. Results
pair as `<stem>.results.h5` only, with no `.mpco` fallback. Pairing and
watching keep the opened file's spelling and compare paths case-insensitively
on Windows and macOS. A file that the renderer opens but that cannot be
paired now ends the old watched set.
