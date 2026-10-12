### CHANGED — records keep `label=` as provenance; `recorder.Element(label=)` under sequential tags (ADR 0115 amendment R2-f, #1735)

A `fix`, `mass`, `s.fix`, `s.mass`, `recorder.Node` or `recorder.Element`
declared with `label=` now keeps the label names in `model.h5`: the record's
`/opensees/decl_params` row stores `label` (a list of names) beside the ids
the label resolved to, so an archive shows where a support or a recorder
came from. The label is provenance only. The ids stay authoritative, and
replay (`OpenSeesModel.from_h5`, `derive_sequence`) never re-resolves a
label. A declaration made with `pg=` or explicit ids stores `label: null`.
Decks are byte-identical, and `model_hash` is unchanged (`decl_params` is
hash-excluded). The `/opensees` zone takes an additive minor bump; its floor
does not move, and older archives read as before.

`ops.recorder.Element(label=...)` now works under the default
`element_tags="sequential"` too. The label resolves to FEM element ids at the
call, and the build translates them to the emitted OpenSees tags, as it does
for `pg=`. The R2-b refusal is gone.

The `fix_from_model()` clash message no longer names `ops.fix`: since R2-a a
global fix merges with `fix_from_model()`, so only a stage's `s.fix` /
`s.support` can clash with it, and the message says so.
