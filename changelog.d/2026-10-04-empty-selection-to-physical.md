### FIXED — to_physical / to_label on an empty selection raise; "PG not found" lists the real groups (#1335)

`EntitySelection.to_physical` / `.to_label` and `Selection.to_physical` /
`.to_label` used to register nothing, silently, when the selection was empty
(typically an `.in_box(...)` a hair too small: it uses BRep containment, so the
box must enclose each whole entity). They now raise `ValueError` at the call
site, naming the group and the containment rule. A script that relied on the
silent no-op must guard the call on a non-empty selection.

The bridge's "physical group ... not found in FEM snapshot" `BridgeError` (node
and element fan-out) always printed `Available PGs: []`, because it filtered
the `(dim, tag)`-keyed group dict for string keys. It now lists the snapshot's
real names, from `PhysicalGroupSet.names()`.
