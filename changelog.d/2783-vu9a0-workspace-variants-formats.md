### ADR 0132 (Accepted): the workspace file and the variants manifest

ADR 0132, ratified by the owner on 2026-11-03, defines two files beside a set. No
code writes them yet. `<stem>.workspace.h5` holds what a user makes in apeGmshViewer:
named node and element sets, plot definitions and saved views. It holds
definitions only, never values, and it is separate from `.user.h5`. Sets are
columnar tables of OpenSees tags with their coordinates, and on a re-meshed model
each set reads as `current`, `moved`, `partial` ("k of n") or `lost`, so it is
never dropped silently. Plots and views are JSON text rows, a view being one
`apegmsh-view/1` shot. A save rebuilds the whole file through a partial file per
writer, checks that `/meta@revision` did not move, and replaces the old file. An
unreadable old file is renamed aside rather than overwritten.

`<study>.variants.json` completes ADR 0130 A7. The list key is `variants`, and
each `path` names a variant's model file `<stem>.h5`, relative to the manifest and
confined to its folder. `params` values are strings, numbers or booleans. Under
owner ruling option A, main opens a variant only from a manifest the user opened,
and the page names a row, never a path.

Both files carry a string version read by ADR 0113's floor rule. The layouts are
two new sections of `architecture/h5-schema.md`, and the manifest has a JSON
Schema in `architecture/schemas/` with valid and invalid examples. Nothing in
`src/` or `apeGmshViewer/` changes.
