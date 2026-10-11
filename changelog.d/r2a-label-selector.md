### ADDED — `label=` node selector, `ops.nodes.one`, same-tier fix merge (ADR 0115 D4–D6, program slice R2-a, #1651)

`ops.fix`, `ops.mass`, a stage's `s.fix` / `s.mass`, `ops.recorder.Node` and
`ops.nodes.get` take `label=` (one geometry label or a list) as a third
selector beside `pg=` and `nodes=`; exactly one selector is allowed. The label
resolves on the snapshot at the call, so every refusal (an unknown label,
which lists the labels that exist; an empty or blank one; a label selecting
no node; two selectors) raises before anything registers, and a label never
falls back to a physical group of the same name. `fix(label=X)` emits the
bytes `fix(pg=X)` emits over the same nodes. `ops.nodes.one(pg=|label=)`
returns a `Node` and raises `BridgeError`, naming the count, unless exactly
one node is selected; `ops.nodes.get(pg=|label=)` stays a `NodeSet`.

Behaviour change: identical homogeneous fixes now merge within a tier (the
global pool, which includes `fix_from_model()`, or one stage). The first
declaration of a `(node, DOF)` emits it; a later one emits only the DOFs it
adds on that node, or nothing. Two `fix` calls sharing a corner node used to
be refused on a staged build and to fail inside OpenSees on an unstaged one;
both now build and run. The duplicate fix / mass validator (V2) runs on every
emit, staged or not, so a repeated `mass` on an unstaged model (without
`overwrite=True`) is refused instead of silently overwritten by `setMass`. A
cross-tier fix overlap without `s.remove_sp`, and a `fix` against an
`s.support` HOLD, still raise. Models without an overlap emit byte-identical
decks and archives.
