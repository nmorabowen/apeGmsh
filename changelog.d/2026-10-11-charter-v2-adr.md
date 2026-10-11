### ADDED — ADR 0121, Charter v2: B with amendments, one engine policy, a mechanized freeze (Accepted 2026-10-11; program slice P0.2, #1678)

ADR 0121 records the expert panel's fork decision ("B with amendments":
stock first-class, the fork invested, AUTO default target) and the P0.3
rulings in one place, and proposes what replaces ADR 0102's prose freeze
and hand-filled Warranted table. Its proposals: one engine policy (a member
is Warranted when its fork test passes in `live-fork` and stock refuses it
or meets an oracle), a census lock over four registries that only an
Accepted ADR can grow, an amendment rule (pointers and status edits land in
the accepting PR), six charter principles each tied to a gate, and
`charter.md` kept as the bridge charter. It records the maintainer's
rulings of 2026-10-11 on its five questions (each the recommended option)
and was accepted by the maintainer on 2026-10-11; it edits neither ADR 0102 nor `charter.md`.
