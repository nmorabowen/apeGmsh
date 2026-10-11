### ADDED — ADR 0121, Charter v2: B with amendments, one engine policy, a mechanized freeze (Accepted 2026-10-11; program slice P0.2, #1678)

ADR 0121 records the expert panel's fork decision ("B with amendments":
stock first-class, the fork invested, AUTO default target) and the P0.3
rulings in one place, and decides what replaces ADR 0102's prose freeze
and hand-filled Warranted table. Its decisions: one engine policy (a member
is Warranted when its fork test passes in `live-fork` and stock refuses it
or meets an oracle), a census lock over four registries that only an
Accepted ADR can grow, an amendment rule (pointers and status edits land in
the accepting PR), six charter principles each tied to a gate, and
`charter.md` kept as the bridge charter. It records the maintainer's
rulings of 2026-10-11 on its five questions (each the recommended option)
and was accepted by the maintainer on 2026-10-11. As its D5 requires, the
accepting PR also appends a dated amendment to ADR 0102 (D2 and D5
superseded; Status stays Proposed) and gives `architecture/charter.md` a
scope header and two factual fixes: HDF5 is a fourth emit target, and live
runs unstaged models only.
