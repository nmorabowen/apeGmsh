### ADDED — Assembly couplings: `node`, `equal_dof`, `rigid_link`, `rigid_diaphragm`, `embedded`, `couple` (ADR 0117 D3, AS4-a)

An instance-declared `Assembly` gains the coupling verbs beyond `tie`.
`asm.node("ref", (x, y, z))` declares an assembly-owned reference node: a
bare name (no `.`) in one namespace shared with instance labels and tie
names; `bridge()` adds it to the merged FEM as a decoupled node labelled
`ref` at FEM id `k` (the `k`-th node), below every instance window, so no
instance moves. `equal_dof`, `rigid_link` and `rigid_diaphragm` take two
ports, each `"{instance}.{pg|label}"` or a reference node;
`embedded(host, embedded)` takes two instance ports;
`couple(target, kind="kinematic"|"distributing", reference="ref")` is RBE2 /
RBE3 (`LadrunoKinematicCoupling` / `LadrunoDistributingCoupling`, fork-only
at run time). Each verb validates everything before recording, resolves in
chain phase at `bridge()`, and raises `AssemblyError` naming both ports when
it resolves no record. Every kind, and each reference node, is a row of
`/assembly/ties` (params as canonical JSON, `n_records`) and round-trips
through `Assembly.from_h5`; a row whose params carry other keys is refused
on write and on read. A port naming a dotted source group (`"A.deck.slab"`)
now resolves to the group compose stored (`A/deck.slab`), for `tie` too.
`contact` and `interface` stay instance-internal: `couple(kind="contact")`
on an instance-declared assembly raises.
