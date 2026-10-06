### ADDED — apeGmshViewer reads /provenance 1.1 and lists synthesised objects (V2g, #1425)

The app's target for `/provenance` is now 1.1, so a file from the current
bridge opens with no "newer than this app" banner for that zone. The reader
takes the `records/origin` column: it is required from 1.1.0 (a 1.1.0 file
without it is malformed), an unknown value is refused, and a 1.0.x file reads
every record as `user`, as the Python reader does. A sources listing
(`selectors.sourcesOf`, `panels/sources.ts`) holds every record of the model
in capture order. The objects the bridge synthesises inside a verb (a stage's
`support:<stage>/hold` series and `support:<stage>` pattern) are listed by
default and marked `synthesised`, and go-to-source on one opens the user's
`s.support(...)` line. New fixture: `fixtures/bridge_provenance.h5`, written by
main's own writer.
