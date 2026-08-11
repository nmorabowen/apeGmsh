# ADR 0092 — Partitioned contact emit: one owner rank per interaction, the whole interface ghosted

**Status:** Proposed (2026-08-11) — the emit half of a cross-library effort.
The engine half is fork **ADR-78**
(`OpenSees/Ladruno_implementation/78_ladruno_parallel_contact_adr.md`);
neither half ships alone. **Adversarially reviewed same day** (see
§Sign-off questions and ADR-78 §Adversarial review log): INV-1 corrected to
master-side ownership, INV-4 sharpened, INV-6 found already satisfied, two of
the three sign-off questions closed.

**ADR-78 P0 ran and PASSED the same day** — a 2-rank contact model matched its
serial twin to 1.6e−14 (implicit, `Mumps`) and **bit-identically** (explicit,
`MPIDiagonal`), with three mutations each failing as designed. The architecture
this ADR is the emit half of is measured, not assumed.

**But P0 surfaced a blocker that lands squarely on this side (P0.a): the Tcl
MPI interpreter has no contact verbs.** `OpenSeesMP` links
`SRC/tcl/commands.cpp`, while `contactSurface`/`contact`/`contactPlane` are
registered only in `TclWrapper.cpp` / `PythonWrapper.cpp` — `OpenSeesMP.exe`
answers `invalid command name "contactSurface"`. Since partitioned emit is a
**Tcl** target (ADR 0061 per-rank `.tcl`), S4 cannot ship until that is
resolved. See the new §Open decision below; S1–S3 are unaffected.

## Context

`g.constraints.contact` / `.contact_plane` are refused outright under
partitioned emit. `_emit_partitioned` raises `BridgeError` as soon as
`fem.elements.contacts` or `.contact_planes` is non-empty
(`opensees/apesees.py:2309`), on the stated grounds that "the fork contact
subsystem is serial-only". `emit_contacts` / `emit_contact_planes`
(`opensees/_internal/build.py:4583`) are only ever called from the flat path
(`apesees.py:1486`); the partitioned path never reaches them. ADR 0073
records the refusal as deliberate and lists the partitioned contact
subsystem as still deferred.

The refusal is the *right* behaviour today, for a sharper reason than the
one written down. The fork handler resolves interface nodes by tag —
`theDomain->getNode(...)` at `LadrunoContactHandler.cpp:637`, `:732`, `:972`,
`:1157` — and a node it cannot resolve is treated as a malformed model:
marked `HUGE_VAL` and backfilled with the segment's first valid coordinate
(`:625–648`), then skipped by the narrow loop. A partitioned deck that
emitted contact today would not crash. It would run a **silently partial
interface**.

What already exists on this side, and is exactly what contact needs:

- **Foreign-node replication.** ADR 0027 declares a non-owned node on the
  referencing rank via `node(tag, *xyz, ndf=...)` and replays the owner's
  ordered `fix` / `remove sp` stream — amended twice (2026-07-27, 2026-07-28)
  after cross-rank MP constraints assembled a singular global matrix without
  it. Contact needs the identical mechanism applied to an interface node set.
- **Per-rank block scoping.** `partition_open(rank)` / `partition_close()`
  and the `_emit_partitioned` per-rank loop (ADR 0027 P4, ADR 0061).
- **Weighted partitioning.** `g.mesh.partitioning` already carries weights,
  so the partitioner can be told what not to cut.
- **A parallel-safe analysis chain.** Auto-emitted `numberer ParallelPlain` +
  `system Mumps` under partitioning (`apesees.py:5537`, `:5595`).

ADR-78 establishes the consequence: because boundary nodes are replicated by
tag and the distributed SOE sums contributions to a shared DOF, a contact
interaction kept **whole on one rank** needs no new communication — and,
critically, keeps the mortar per-slave-node global gap reduction
(`accumulateMortarGap`) rank-local, avoiding an allreduce inside every
residual evaluation.

## Decision

Replace the blanket refusal with a **locality contract**: contact is emitted
under partitioning when — and only when — each interaction can be assembled
by a single rank with the whole interface visible to it.

- **INV-1 — one owner rank per interaction, chosen on the MASTER side.**
  Each `ContactRecord` / `ContactPlaneRecord` emits its `contactSurface` +
  `contact` (or `contactPlane`) lines inside **exactly one** rank's block.
  **Owner = the rank natively owning the master surface's backing solid
  elements**; ties broken by lowest rank index so emission is deterministic.
  Emitting on two ranks would double-count the contact force, so this is
  enforced structurally: the per-rank loop emits when `rank == owner` and
  never otherwise.

  **This is the only available guard, not merely a tidy one.** ADR-78 P0
  measured the duplicate case: it converges to a *plausible wrong answer*
  (exactly half the penetration — double stiffness) while base reactions still
  balance, so equilibrium checks do not catch it. The fork's own
  duplicate-slave-set warning is **rank-local** — each rank's handler sees only
  its own contacts — so a verb emitted on two ranks produces that wrong answer
  with no warning at all (ADR-78 P0.d).

  *Corrected by the 2026-08-11 adversarial review* (this ADR first said
  "majority of slave nodes"). The fork's `-kn auto` resolves the owning solid
  of the **master segment** (`LadrunoContactHandler.cpp:123–172`), so a
  master-side owner keeps penalty auto-sizing native and leaves only
  geometry-only slave nodes to ghost. Nothing else in the narrow phase reads
  elements.

- **INV-2 — the whole interface is ghosted, not a halo.** Every node of both
  surfaces that the owner does not natively own is declared in the owner's
  block via `node(tag, *xyz, ndf=...)` before the `contactSurface` lines,
  carrying the owner's SP stream per ADR 0027 INV-2 (both directions under
  staging). A halo sized from the initial configuration is unsafe: the
  candidate set churns every step and, under ADR-60 finite-sliding re-emit,
  moves — and the engine cannot acquire a node mid-analysis. The ghost set is
  bounded by *interface* size, not model size.

- **INV-3 — `kn="auto"` is preserved; `soft=` is refused.** *Rewritten after
  the review, which split what the first draft lumped together.* Master-side
  ownership (INV-1) plus an uncut master surface (INV-4) leaves the fork's
  `-kn auto` resolving natively — the segment's backing solid is on the owner
  rank, so a partitioned run picks the **same** penalty as its serial twin. We
  therefore emit `-kn auto` unchanged; the first draft's "compute the literal
  in apeGmsh" is withdrawn, because a literal would have silently diverged
  from the serial value. `soft=` **is** refused with a named error: `m_eff`
  needs the assembled mass of **both** surfaces, and the ghosted side
  contributes none on the owner rank, which no owner rule can fix (fork
  ADR-78 D4). P0 confirmed the preserved half: a partitioned `-kn auto` run
  reproduced the serial penalty exactly.

- **INV-7 — a contact ghost carries geometry and SP state only.** Never mass,
  never elements, never loads. ADR-78 P0.e measured the failure: because the
  distributed solver sums the diagonal **mass** as well as the residual over
  shared DOFs, a ghost that declares mass double-counts it and **both ranks
  agree on the same wrong answer** (~20% off in the P0 model). apeGmsh's ADR
  0027 ghost path is already `node` + replayed `fix` with no mass, so this
  invariant holds today — it is stated so it stays that way.

- **INV-4 — the partitioner may cut the slave side, never the master.**
  Contact-pair node adjacency is added as weighted edges to the partition
  graph, with the **master surface and its backing solid elements weighted as
  uncuttable**. A cut master surface leaves some facets' owning solids
  off-rank, which silently disables `-kn auto` for those facets (INV-3, and
  the fork's silent-skip path — ADR-78 D5.2). A cut *slave* surface costs
  only ghost node lines, which INV-2 already pays. This is the sharpened form
  of "don't cut interfaces" from the 2026-08-11 review.

- **INV-5 — refusals get specific.** The blanket `BridgeError` at
  `apesees.py:2309` is replaced by narrower, named errors: `soft`/`kn="auto"`
  under partitioning (INV-3); contact + any MP constraint or equation tie
  under partitioning (the existing serial handler-exclusivity rule at
  `apesees.py:5257`, which partitioned decks hit *by construction* since they
  carry replicated cross-rank MP constraints — fork ADR-78 D6); and a
  contact whose ghost cost exceeds a stated bound. Each error says which
  interaction and why.

- **INV-6 — read side already tolerates single-rank output.** Contact
  recorders exist only on the owner rank. This contract is **already
  satisfied**: `f5358318` (2026-08-10) made a `.ladruno` with no
  `RESULTS/ON_ELEMENTS` group loud at read time, but scoped it so that
  "across partitions the read is loud only when EVERY rank is missing the
  group — a rank owning none of the recorded elements legitimately writes
  none, so `_per_partition` drops it from the stitch and re-raises only when
  nothing survived." Contact-on-one-rank is exactly that shape. Work reduces
  to a regression test pinning it for contact families.

## Implementation plan (slices, each PR-able)

1. **S1 — owner resolution + ghost set, pure kernel.** A resolver that, given
   `fem.partitions` + a `ContactRecord`, returns `(owner_rank, ghost_node_ids)`.
   No emit change. Tests on synthetic partitions: interface wholly inside one
   rank, straddling two, straddling three.
2. **S2 — partition-graph weighting** (INV-4) on `g.mesh.partitioning`, with a
   test that a contact interface is not cut at np=2 for a two-body model.
3. **S3 — the `soft=` refusal** (INV-3), plus a test pinning that a
   partitioned deck emits the *same* `-kn auto` token as its serial twin.
   (Was "compute `kn` in the broker"; the review withdrew that.)
4. **S4 — partitioned emit** (INV-1/INV-2): call `emit_contacts` /
   `emit_contact_planes` inside the owner's block, preceded by ghost `node`
   declarations + SP replay; relax `apesees.py:2309` to the locality check.
   ~~Gated on fork ADR-78 P0.5~~ — **cleared 2026-08-11**, the fork registered
   the contact verbs in the classic Tcl engine and a 2-rank Tcl deck of this
   exact shape now runs. Still gated on **P1** (the missing-node hard error;
   until then a wrong ghost set degrades on a warning rather than aborting,
   which is precisely what this ADR exists to prevent).
5. **S5 — integration test** mirroring
   `tests/opensees/integration/test_emit_partitioned_staged_mp_constraints.py`:
   a 2-rank two-body pounding deck, asserting the `contact` verb appears in
   exactly one rank block and every referenced node is declared in that block.
6. **S6 — `Results` ownership contract** (INV-6).

## Consequences

- Contact models stop being capped by one node's RAM. That is the whole
  point: pounding and progressive-collapse models are exactly the ones that
  need both contact and scale.
- The owner rank does all of an interaction's contact work. Accepted for the
  MVP and measured at ADR-78 P4 — contact is a surface, the mesh is a volume,
  so the ratio is expected to favour us. If measurement disagrees, the fix
  (slave-node-partitioned contact) is ADR-78 P5 and requires the in-Newton
  allreduce this design exists to avoid.
- Ghost declarations add `node` lines to the owner's deck, which interacts
  with ADR 0061's per-rank deck-text budget on production models.
- A partitioned contact deck stays textually comparable to its serial twin:
  `-kn auto` is emitted unchanged and resolves to the same value, because the
  master surface is uncut and owner-local. `soft=` is the one authoring
  feature that becomes unavailable under partitioning.

## Sign-off questions — settled by the 2026-08-11 adversarial review

1. **Owner rule — SETTLED: master-side, no user pin.** The heuristic is not a
   heuristic once you follow `-kn auto` into the fork: it resolves the owning
   solid of the **master** segment, so the master-owning rank is the only
   choice that keeps auto-sizing native. A `rank=` escape hatch would let a
   user silently disable auto-kn, so it is **not** offered. See INV-1.
2. **Ghost budget — WITHDRAWN.** ~50 B per ghost `node` line against ADR
   0061's measured ~250 B/hex puts a 10⁴-node interface at ~0.2% of a 100³
   deck. There is no budget to spend, no refusal threshold, and no halo
   fallback. The real cost of ghosting is the gather-to-rank-0 shared-DOF
   exchange in `DistributedDiagonalSolver::solve()` (fork ADR-78 Q-P0GATHER),
   measured at P4 — not deck text.
3. **First-ship scope — SETTLED: ship restricted, don't wait.** The premise
   that partitioned decks carry MP constraints "by construction" was false —
   ghosts are `node` + replayed `fix` (`emit_ghost_sp_ops`), not MP
   constraints. So MP-constraint-free partitioned contact is a real, useful
   class: two-body pounding, rocking against `contact_plane`, cross-rank
   mortar mesh-ties. Transformation-grade `LadrunoContact` (fork ADR-39
   Q-CONSTR) is a **separate, later** prerequisite for diaphragm-bearing
   models, not a gate on this lane.

**P0 has since run and passed**, so S1–S3 are unblocked. One *new* question
takes its place, and it blocks S4 only.

## RESOLVED 2026-08-11 — the Tcl target stands; the fork registered the verbs

Option (1) below was implemented and verified the same day (fork ADR-78 P0.5,
[nmorabowen/OpenSees#726](https://github.com/nmorabowen/OpenSees/pull/726),
merged to `ladruno`): `SRC/tcl/commands.cpp` + `commands.h` now register the
contact family in the classic engine, so `OpenSeesMP` sees it. **Nothing changes in this repo** — the
existing per-rank Tcl target (ADR 0061) is the emit language for partitioned
contact, and S4 is no longer blocked on an emit-language decision.

Evidence: a 2-rank Tcl deck of the ADR-0092 shape — contact emitted on rank 0
only, slave interface ghosted with `node` + replayed `fix`, no ghost mass —
converged and matched its serial twin to 1.4e−14, with the rank-0 ghost
displacement bit-identical to rank 1's native value. The serial Tcl numbers also
reproduce the Python-lane reference to every digit.

The original decision record follows.

## Open decision (SUPERSEDED) — which language does a partitioned contact deck speak?

ADR-78 P0.a: `OpenSeesMP` links `SRC/tcl/commands.cpp`, which does not register
`contactSurface` / `contact` / `contactPlane` (they live in `TclWrapper.cpp` and
`PythonWrapper.cpp`). Verified by running the deck: `invalid command name
"contactSurface"` on `OpenSeesMP.exe`, on `OpenSees.exe`, and on the installed
Aug-10 build. P0 therefore ran on the `openseesmp` **Python** module. Three ways
out, in increasing cost to this repo:

1. **Fork registers the verbs in `commands.cpp`.** Nothing changes here — the
   existing per-rank Tcl target (ADR 0061) just works. Cheapest for apeGmsh,
   and it also fixes serial Tcl decks, which have the same gap today.
2. **Add a per-rank Python emit target** for partitioned contact — the rank-axis
   sibling of ADR 0061, emitting `.py` driven by `ops.getPID()`. Larger change
   here, and it splits the partitioned emit story across two languages.
3. **Restrict partitioned contact to the live/Python path** and refuse the Tcl
   target. Cheapest overall, narrowest capability.

Recommendation: **(1)**. The gap is a registration omission in the fork, not a
design choice, and it is the only option that leaves apeGmsh's emit architecture
untouched. Needs the fork's sign-off since the work lands there.
