# ADR 0092 — Partitioned contact emit: one owner rank per interaction, the whole interface ghosted (summary)

**Status:** Proposed (2026-08-11) — the emit half of a cross-library effort. The engine half is
fork **ADR-78** (`OpenSees/Ladruno_implementation/78_ladruno_parallel_contact_adr.md`); neither
half ships alone. […]

## Decision

The blanket serial-only refusal is replaced by a **locality contract**: contact is emitted under
partitioning when, and only when, each interaction can be assembled by a single rank with the
whole interface visible to it.

- **INV-1: one owner rank per interaction, chosen on the master side.** Each `ContactRecord` /
  `ContactPlaneRecord` emits its `contactSurface` + `contact` (or `contactPlane`) lines inside
  exactly one rank's block; the per-rank loop emits only when `rank == owner`. Emitting on two
  ranks double-counts the force and converges to a plausible wrong answer with no warning
  (ADR-78 P0.d). As amended: the tally counts only uniquely-owned master nodes; if all are
  shared and one rank leads it is taken; an all-shared tie refuses with a named error. S4 made
  the pick element-exact (`master_backing_element_ids` + `build_element_partition_owner` feed
  `resolve_contact_ownership(master_element_ranks=...)`). A plane is slave-tallied.
- **INV-2: the whole interface is ghosted, not a halo.** Every node of both surfaces the owner
  does not natively own is declared in the owner's block with `node(...)` before the
  `contactSurface` lines, carrying the owner's SP stream per ADR 0027 INV-2.
- **INV-3: `kn="auto"` is preserved; `soft=` is refused.** `-kn auto` is emitted unchanged and
  resolves to the serial value because the master is uncut and owner-local; `soft=` raises a
  named error, since `m_eff` needs mass from both surfaces (fork ADR-78 D4).
- **INV-4: the partitioner may cut the slave side, never the master.** Implemented (S2
  amendment) as a deterministic post-partition override, `partition(n_parts,
  uncuttable_elements=...)`, which moves the master surface's backing solids onto the partition
  already holding most of them (`_Partitioning._enforce_uncuttable`); a no-op when omitted.
- **INV-5: refusals get specific.** Named errors, each naming the interaction: `soft` under
  partitioning, contact plus an MP constraint or equation tie under partitioning, and the
  review's refusals (pattern `sp` on a contact ghost; a cut master with `kn="auto"` and an
  unresolvable facet).
- **INV-6: the read side tolerates single-rank output.** Contact recorders exist only on the
  owner rank; a `.ladruno` read is loud only when every rank lacks the group (`f5358318`).
- **INV-7: a contact ghost carries geometry and SP state only**, never mass, elements or loads
  (a ghost with mass double-counts it, ADR-78 P0.e).

The emit language is the per-rank Tcl target (ADR 0061). Consequence: `soft=` is the one
authoring feature unavailable under partitioning; there is no `rank=` pin and no ghost budget.

## Amendments

- 2026-08-11: INV-1 amended twice (element-side owner not computable from records; then the
  node-majority exactness claim withdrawn after an executed counterexample). INV-3 rewritten.
- 2026-08-11: INV-4 amended during S2 (post-partition override instead of edge weights).
- 2026-08-11: RESOLVED, the Tcl target stands; the fork registered the contact verbs in
  `commands.cpp` (nmorabowen/OpenSees#726). The "Open decision" section is SUPERSEDED.
- 2026-08-12: S3, S4 and S5 landed (S5 is a numeric twin, ≤ 1e-10 vs serial); 2026-08-13: S6.
- 2026-08-13: post-lane review log, findings F1–F8 (F1–F5 and F8 fixed, F6 documented, F7
  accepted).

Full text: [../0092-partitioned-contact-emit.md](../0092-partitioned-contact-emit.md).
Precedence (AGENTS.md): the code wins over the ADR, and the ADR wins over this summary.
