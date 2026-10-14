### CHANGED — one per-node ndf map when no `ops.ndf` overlay is set (program slice S3-e, #1926)

`BuiltModel.emit` no longer copies the inferred per-node ndf map into a second `effective_ndf` dict
when the `ops.ndf` overlay is empty, which is the common case. It hands the inferred map on as is.
`infer_node_ndf` builds a fresh dict on every call, and no consumer mutates the map. A non-empty
overlay still merges into a fresh dict. This removes about 40 B per global node from the set held
at the first `partition_open` (ADR 0100). The emitted decks are unchanged.
