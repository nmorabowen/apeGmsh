### FIXED — apeGmshViewer opens files written by `main` with no neutral or opensees banner (ADR 0112 V2h, #1427)

The app's targets were neutral 2.33 and opensees 2.21, so every file `main`
writes today (neutral 2.35, opensees 2.22) opened under two "newer than this
app" banners. The three minors in between add nothing the app reads: 2.34
makes `/meta/ndm` the declared spatial dimension (the app shows the attribute
and never branched on it), 2.35 adds the `source` column to `/loads/nodal`
(the app does not read `/loads`), and opensees 2.22 adds the
`/opensees/bcs@mass_from_model` marker (the app does not read `/opensees/bcs`).
The targets now read 2.35 and 2.22, `fixtures/shoebuckle.h5` is rewritten by
`main`'s current writer, and the reader tests prove a file carrying the new
column and the new marker opens with no warning, while a later minor still
raises exactly one banner.
