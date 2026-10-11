### ADDED — apeGmshViewer's Recorders window shows every recorder in full (V3c3, #1644)

The Recorders window lists each recorder, global and per stage, in
declaration order. A recorder is shown by its `name=`, or by the `#k` of its
`/opensees/decls` key when it is unnamed. Each row shows the recorder's
type, what it records (responses and dofs), its targets (node and element
counts, and groups by name), its output file and format fields, and the
stage that adds it. Its parameters are listed by field name from
`/opensees/decl_params`; null fields are listed as unset. The window never
reads the positional argv.

Clicking a recorder that targets a group of the file selects that group, and
the row's source button jumps to the call that declared it. A file without
`decl_params` or `/sequence` says so in the row instead of guessing; the
removal stage, for example, is read only from `/sequence`.
