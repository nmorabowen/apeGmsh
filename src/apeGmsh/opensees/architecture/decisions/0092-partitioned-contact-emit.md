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
  **Owner = the rank natively owning the majority of the master surface's
  nodes**; ties broken by lowest rank index so emission is deterministic.

  *Amended 2026-08-11 during S1.* This first read "the rank owning the master
  surface's backing solid **elements**" — which is the property that actually
  matters (the fork's `-kn auto` resolves the owning solid of the master
  segment), but which the resolver's inputs **cannot compute**: a
  `ContactRecord` carries `master_faces` as node connectivity, not element ids,
  so there is no facet → backing-element map at this layer.

  *Amended again 2026-08-11, after the S1/S2 adversarial review.* The first
  amendment claimed node majority was "*exact* whenever INV-4 holds". **That is
  false, and a counterexample was executed against the resolver**, not merely
  argued. `extract_partitions` adds every element's nodes to every partition on
  the entity, so a node on a cut is **replicated onto every touching rank**. In a
  tet mesh, tets that touch the master surface through an edge or vertex only are
  *not* in the uncuttable group (INV-4 moves face-sharing solids), so they stay
  where METIS put them and replicate the master nodes onto a neighbour. With the
  cut hugging the surface — which is exactly the post-override geometry — every
  master node can be replicated, the tally ties M-vs-M, and the lowest-rank break
  hands ownership to a rank holding **zero** backing solids, on a model that
  fully honoured INV-4.

  What the resolver does now:

  1. **Tally only uniquely-owned master nodes.** A node in ≥2 partitions carries
     no locality information. Under INV-4 the backing rank natively owns every
     master node, so if any master node is uniquely owned it is uniquely owned by
     *that* rank — which makes the pick exact, not a proxy.
  2. **If every candidate node is shared but one rank still leads**, take it.
  3. **If every candidate node is shared and the counts tie, refuse** with a
     named error. That case is genuinely undecidable from node data, and the
     executed counterexample lands precisely there. An error at emit time beats a
     partitioned run that dies later inside `-kn auto`.

  The honest residual: this is exact when unique ownership exists and explicit
  when it does not. Making it exact in *all* cases still requires element→rank
  ownership plus a facet→element map — S4 has both and can pass them in, which
  is the real fix rather than a better heuristic.
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

  *Amended 2026-08-11 during S2.* "Weighted edges" turned out not to be the
  lever `g.mesh.partitioning` actually offers: METIS vertex weights (the
  existing `weights=` path, `_Partitioning.partition(..., weights=...,
  backend="pymetis")`) balance partition **size**, they do not bias METIS
  against cutting a specific group — and the wrapper never exposed pymetis's
  `eweights` either, so no edge-weight lever existed to widen. "Never" is also
  a hard invariant, and an edge-weight bias is inherently soft (METIS can
  still cut a heavily-weighted edge if the balance constraint leaves no other
  option). S2 implements INV-4 as a **deterministic post-partition override**
  instead: `partition(n_parts, uncuttable_elements=...)` runs the normal
  partition (native or weighted), tallies which partition already holds the
  most of the named element tags, and reassigns every one of them to that
  partition via the existing `partition_explicit` primitive. The group either
  already sits on one partition, or is moved there outright — never split.
  This composes with `weights=` (the override runs after either flavour) and
  is a no-op when `uncuttable_elements` is omitted, so a model without
  contact declarations partitions exactly as before. See
  `_Partitioning._enforce_uncuttable` in `mesh/_mesh_partitioning.py` and
  `tests/test_partitioning_uncuttable.py`.

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

   *Landed 2026-08-12.* The refusal is a named `BridgeError` in
   `BuiltModel._emit_partitioned` (`opensees/apesees.py`), placed
   immediately **before** the blanket serial-only refusal so (a) a SOFT
   deck gets the real reason today instead of "serial-only", and (b) when
   S4 deletes the blanket, the SOFT check is already standing in the path
   S4's emit necessarily passes through. It is backed by a pure predicate,
   `soft_family_knobs` in `_kernel/resolvers/_contact_ownership.py`
   (S1's module — the same layering), which names the active SOFT-family
   knobs on a record: `soft=` on `ContactRecord` + `ContactPlaneRecord`
   and `edge_soft=` on the mortar edge-edge fallback. `visc=` is
   deliberately *not* refused — viscous stabilisation is a damper on the
   owner-local active set, with no both-sides assembled-mass dependency.
   The error names the interaction (index + `name=`), states the reason
   (k_soft = SOFSCL·4·m_eff/dt² needs the ASSEMBLED mass of both surfaces;
   the ghosted side contributes zero on the owner rank — fork ADR-78 D4),
   and suggests `kn="auto"` / an explicit penalty / serial emit. The fork
   engine independently refuses `-soft`/`-edgeSoft` at handle() time under
   MPI since the same-day fork ADR-78 §P2 (fork PR #739) — the emit-side
   refusal is defense-in-depth that fires at deck-generation time instead
   of at job launch. The `-kn auto` pin ran at the layer that exists today
   (S4's partitioned emit does not): partitioning a model never rewrites
   the record's `kn="auto"` sentinel, and the deck a partition-carrying
   model emits (the sanctioned `flat=True` lane) carries a `contact` verb
   line **byte-identical** to its serial twin's, ending on the literal
   `auto` token. One measured nuance: the twins' `contactSurface` slave
   listings hold the same node *sets* in different *order* (partitioning
   reorders node extraction), so the pin compares the verb line exactly
   and the surface node-sets order-insensitively — S4 extends the same
   assertion to the owner rank's block. Tests:
   `tests/_kernel/resolvers/test_contact_soft_knobs.py` (unit, 13) +
   `tests/opensees/integration/test_contact_soft_partition_refusal.py`
   (refusal fires partitioned / silent serially / named message /
   negative controls distinguishing the S3 error from the blanket, 8).
4. **S4 — partitioned emit** (INV-1/INV-2): call `emit_contacts` /
   `emit_contact_planes` inside the owner's block, preceded by ghost `node`
   declarations + SP replay; relax `apesees.py:2309` to the locality check.
   ~~Gated on fork ADR-78 P0.5~~ — **cleared 2026-08-11**, the fork registered
   the contact verbs in the classic Tcl engine and a 2-rank Tcl deck of this
   exact shape now runs. Still gated on **P1** (the missing-node hard error;
   until then a wrong ghost set degrades on a warning rather than aborting,
   which is precisely what this ADR exists to prevent).

   *Landed 2026-08-12.* The blanket serial-only refusal is gone; the
   locality contract is live. Shape of the change:

   - **Planning before emission.** `BuiltModel._plan_partitioned_contacts`
     (`opensees/apesees.py`) resolves every interaction to
     `(owner_rank, ghost_node_ids)` via the S1 resolver right after
     element ownership is built — every named refusal fires before a
     single line is emitted. The per-rank loop then emits each owned
     interaction at step 7c (after the MP-constraint pass, so
     already-declared ghosts are not re-declared): ghost `node` lines
     with the same inferred/envelope ndf the owner emits, each
     immediately followed by the owner's replayed SP stream (the ADR
     0027 `emit_ghost_sp_ops` machinery, keyed off the same
     `_bucket_fix_targets_by_rank(by_node=...)` view, now also built
     when contacts are present), then the `contactSurface` pair + verb
     via `emit_contacts` / `emit_contact_planes`, which grew a
     `records=` override so one record emits inside one block.
     Emission on one rank is structural: the plan maps each record to
     exactly one owner, and step 7c reads only `plan[rank]`. The
     emitted deck reproduces the fork P0 harness's shape
     (`Ladruno_files/testbed/contact_parallel/mp.tcl`, the one measured
     at 1.4e−14 vs serial): ghosts as `node` + replayed `fix`, both
     surface defs + the verb on the owner rank only, `constraints
     LadrunoContact` + `ParallelPlain`/`Mumps` from the existing
     auto-emit. INV-7 needed no code: mass/loads bucket by primary
     owner and elements by element owner, all of which point at the
     ghost's native rank — pinned by test, not by construction alone.
   - **The owner pick is element-exact, and the element→rank input IS
     reachable at this layer** — the second INV-1 amendment's "real fix"
     shipped, not the proxy. `master_backing_element_ids` (S1's module,
     `_kernel/resolvers/_contact_ownership.py`) intersects each master
     facet's node set against the mesh's element connectivity (which
     `FEMData`'s element groups expose; the record alone still carries
     only nodes): a boundary facet is contained by exactly ONE solid,
     and that solid's rank (via `build_element_partition_owner`) feeds
     `resolve_contact_ownership(master_element_ranks=...)`. The node
     tally survives strictly as the fallback for meshes whose
     connectivity is not exposed (hand-rolled FEM stubs) or whose facet
     resolution is ambiguous (interior face / non-conforming patch) —
     and its undecidable-tie refusal stays, now wrapped in a
     `BridgeError` naming the interaction (INV-5). The executed S1
     counterexample is therefore split in two: with connectivity it
     resolves exactly; without, it refuses loudly.
   - **INV-4 got an emit-time backstop.** The S2 partitioner override is
     opt-in (`uncuttable_elements=`), so a cut master surface can reach
     emit. When the backing solids straddle ranks AND any auto-sizing
     knob is active (`kn`/`eps_n`/`eps_t`/`edge_kn` == `"auto"`), S4
     refuses, naming the ranks and the fix — off-rank backing silently
     skips those facets' auto penalty (fork ADR-78 D5.2). A cut master
     with a fully explicit penalty is allowed: nothing else in the
     narrow phase reads elements (the 2026-08-11 review), and the whole
     interface is on the owner by INV-2.
   - **Ghost-cost bound: none, deliberately.** INV-5's "ghost cost
     exceeds a stated bound" clause is DEAD — sign-off Q2 withdrew the
     budget the same day it was proposed, so there is no threshold to
     enforce and S4 does not invent one. If P4's measurement of the
     shared-DOF gather ever reinstates a bound, it lands as a new
     amendment, not silently in the emitter.
   - **Deviation — staged partitioned contact is refused, named.** Two
     reasons, recorded here as still-open rather than designed around:
     the partitioned staged pipeline skips the analysis-chain auto-emit
     (each stage validates its own chain), so the forced
     `LadrunoContact` handler would never be emitted and the interaction
     would be silently unenforced; and contact-ghost SP sync across
     stage blocks (the ADR 0027 amended machinery) is wired but
     unproven for contact ghosts. Unstaged partitioned contact — the
     first-ship class from sign-off Q3 — is unaffected.
   - Tests: `tests/opensees/integration/test_emit_partitioned_contact.py`
     (one owner block; every referenced node declared; ghosts carry no
     mass/elements/loads; SP replay matches the owner's stream and is
     adjacent to the ghost's `node` line; undecidable tie refuses named;
     S3 soft refusal survives; staged refusal; serial-twin shape parity)
     + the S3 `-kn auto` twin pin extended to the real per-rank fan-out
     (byte-identical verb line, order-insensitive surface node sets) +
     resolver unit coverage for the backing map and the element-exact
     pick. The S3 negative controls that pinned the blanket
     ("serial-only") now pin successful emission instead.
5. **S5 — integration test** mirroring
   `tests/opensees/integration/test_emit_partitioned_staged_mp_constraints.py`:
   a 2-rank two-body pounding deck, asserting the `contact` verb appears in
   exactly one rank block and every referenced node is declared in that block.

   *Landed 2026-08-12 — as the NUMERIC twin, not another shape assertion.*
   S4's own suite already pins the shape, so S5 shipped as the run lane:
   `tests/opensees/subprocess/test_contact_partitioned_numeric_twin.py`
   emits both twins from ONE model definition — the fork P0 geometry
   through the real API (two stacked single-`stdBrick` blocks via
   transfinite hex meshing, 1e-3 initial penetration, lateral-fixed 1-D
   column, 4×(−2500) on the top face, `kn="auto"` + `outward=(0,0,1)`;
   `partition(2)` puts one body per rank, no uncuttable override needed)
   — and RUNS them: the serial `flat=True` deck under `OpenSees.exe`
   (RCM/UmfPack), the partitioned deck under `mpiexec -n 2 OpenSeesMP.exe`
   (ParallelPlain/Mumps). The harness appends only a measurement fragment
   (`analyze` + `puts` + `wipe`); the emitted decks run as-is. Gate:
   relative agreement ≤ 1e−10 per quantity. **Measured 2026-08-12**
   (fork build `d29de577`, the ADR-78 P2 tip):

   | quantity | serial | 2-rank | rel Δ |
   |---|---|---|---|
   | w_top (top-block tip) | −5.6249999999999998e−03 | −5.6249999999998966e−03 | 1.8e−14 |
   | w_slave (interface, native rank) | −5.1249999999999993e−03 | −5.1249999999998979e−03 | 2.0e−14 |
   | w_master (master face) | −5.0000000000000012e−04 | −5.0000000000000001e−04 | 2.2e−16 |
   | ΣR_base | 1.0000000000000000e+04 | 1.0000000000000002e+04 | 1.8e−16 |

   Ghost pin: the owner rank's ghost of the slave interface node printed
   **bit-identical** to the slave rank's native value (asserted < 1e−14;
   measured 0.0) — the same identity fork P0 measured. Physics anchors:
   ΣR = 1e4 balances the applied load exactly; w_master = −5.000e−4 =
   P·h/(E·A) analytically.

   Negative controls (the comparison detects a wrong deck):

   - **no-contact twin** — serial analyze fails (−3, NormDispIncr blows
     up): contact is load-bearing in the comparison;
   - **ghost `fix` replay stripped from the owner block** — Mumps
     "Matrix is Singular Numerically", analyze −3 on both ranks (the
     ADR 0027 INV-2 measured failure mode);
   - **contact verb duplicated into the slave rank's block** (master
     interface ghost-declared there by the mutation — the ADR-78 P0.d
     deck): the fork's P1 hard error tears down all ranks
     ("LadrunoContactHandler: tearing down all 2 MPI ranks…") because
     `-kn auto` cannot resolve the master's backing solid on the
     non-owner. The P0.d silent-wrong case now fails LOUDLY on the
     current fork — better than the measured half-penetration.

   Gating: `subprocess` + `slow` markers with loud skips on
   `APEGMSH_OPENSEES_BIN` (must hold `OpenSees.exe` + `OpenSeesMP.exe`)
   and Intel MPI's `mpiexec` (`I_MPI_ROOT`); the CI suite lane excludes
   `subprocess`, so CI skips it by construction. The MPI launch
   environment is the fork's proven recipe (libfabric under
   `%I_MPI_ROOT%\opt\mpi\libfabric\bin`, compiler runtime, binaries dir
   on PATH, `TCL_LIBRARY` from the sibling `lib/tcl8.6`).

   **Open item surfaced by S5 — auto-emit ordering (real defect, not
   fixed here).** The step-7c auto-emits (`constraints LadrunoContact`,
   the runtime-conditional `numberer ParallelPlain` / `system Mumps`)
   land AFTER a user-declared `analysis Static` in the emitted deck. The
   OpenSees Tcl engine constructs the analysis at the `analysis` command
   with whatever components exist at that moment, and `constraints` /
   `numberer` do **not** retro-propagate into an existing analysis (fork
   `SRC/tcl/commands.cpp` back-propagates only `system` → `setLinearSOE`
   and `algorithm` → `setAlgorithm`). Measured in the S5 harness: a
   contact deck relying on the auto-emit while declaring
   `ops.analysis.Static()` ran **PlainHandler silently** — the serial
   twin diverged, and the 2-rank twin *converged to a plausible-wrong
   answer* (w_top −5.714e−4 vs the true −5.625e−3) with base reactions
   still balancing (1e4), the exact silent-wrong shape this ADR exists
   to prevent. The live (Python) lane constructs the analysis the same
   way, so the hazard is not Tcl-specific. Workaround (what S5 did):
   declare `ops.constraints.LadrunoContact()` + numberer + system
   explicitly so they emit in the pre-`analysis` chain section; the
   auto-emit then re-emits the handler post-`analysis` (inert,
   harmless, warned). The fix — emitting the auto-emit chain pieces
   before any user-declared `Analysis` primitive — reorders emit output
   every lane shares, so it was deliberately its own change, not a
   side-effect of S5.

   **FIXED 2026-08-12 (PR #945).** All three emit paths (`_emit_flat`,
   `_emit_split`, `_emit_partitioned`) now hoist the chain auto-emits
   (constraint handler; plus the INV-5 parallel numberer/system on the
   partitioned path) to immediately BEFORE the first user-declared
   `Analysis` primitive in the pre-element pass, and skip the original
   post-topology site. Decks with no user `analysis` primitive keep
   the original position byte-identically, and the
   `suppress_analysis_chain_auto_emit` seam (ADR 0077 Tier 1B) is
   honored at both sites. The hoisted position is the position the S5
   workaround already proved numerically. Regression: deck-shape pins
   in `tests/opensees/integration/test_emit_partitioned_contact.py`
   (auto handler/numberer/system precede `analysis Static`; the
   no-user-analysis position is unchanged) + the numeric auto-chain
   twins in the S5 harness (`*_auto_chain` — the serial and 2-rank
   hazard decks now converge to the explicit-chain twin's answer
   within 1e-10). One knock-on: the S5 negative control (c) had leaned
   on an accident of the old ordering — the post-verbs handler
   re-construction blew up on the duplicated `-kn auto` verb. With
   every handler line now preceding the verbs, runtime D5.2 silently
   skips auto-kn facets whose backing is off-rank (the duplicate
   contributes nothing and the mutant converges CORRECTLY), so the
   control now forces an explicit kn on the replayed verb to re-create
   the true P0.d double-enforcement class. Measured on the current
   fork, that deck DEADLOCKS (the two would-be owners desynchronize
   the handler's collectives) — the control treats the hang, a loud
   teardown, or a numeric deviation all as detection, via a
   tree-killing bounded runner (`_run_mp_may_hang`; plain
   `subprocess.run(capture_output=True)` blocks past its timeout on
   Windows because orphaned ranks inherit the pipe handles).
6. **S6 — `Results` ownership contract** (INV-6).

   *Landed 2026-08-13* —
   `tests/opensees/subprocess/test_contact_partitioned_results_stitch.py`,
   the S5 model recorded and read back. INV-6 predicted "work reduces to
   a regression test"; writing it found the contract rests on **three**
   things, only one of which was pinned anywhere.

   - **The filename grammar is a CROSS-LIBRARY contract nothing had
     executed.** apeGmsh emits the recorder line ONCE, globally, with the
     name verbatim (there is no `part-` anywhere in `src/`); the
     `<stem>.part-<K>.ladruno` suffix that `discover_partition_files`
     matches is written by the FORK (`SRC/recorder/LadrunoRecorder.cpp`,
     which detects partitioning via `send_self_count` or an MPI
     launcher's `PMI_*` / `OMPI_*` / `SLURM_NTASKS` pair, then rewrites
     the stem). Each side tested its own half. A drift in the fork's
     spelling would make every partitioned read silently return ONE
     rank's slice — a wrong answer shaped like a small model. Now
     asserted against real 2-rank output.
   - **The node stitch dedupes first-write-wins** (`_merge_node_slabs`,
     `_merge_partition_fems`), and the contact ghost IS that duplicate.
     Measured: rank 0 recorded **12** nodes for the 8 it owns — the 4
     extra are exactly the slave-face ghosts {10, 12, 14, 16} — while
     rank 1 recorded its 8. The merge keeps whichever copy it meets
     first and drops the rest **without comparing them**, so the stitch
     is correct only because ghost == native. That equality is a fork
     property (ADR-78 P0 / S5, bit-identical at the solver level); S6
     asserts it survives to the recorder, which is what the reader
     actually consumes, and pins the stitched result against the serial
     twin (16 unique node ids, no NaN, max rel Δ **1.997e−14**, gate
     1e−10; the ghost copy is the one that wins, and it agrees).
   - **The element stitch does NOT dedupe** (`_concat_element_slabs`
     concatenates, assuming rank-disjoint elements), so an INV-7
     violation — a ghost carrying elements — would double-count in
     every partitioned read with no error. Pinned on real output:
     rank 0 {2}, rank 1 {3}, empty intersection, union == the serial
     element set.

   Plus the INV-6 statement proper, now on real contact output rather
   than a synthesized manifest: strip `ON_ELEMENTS` from one rank and
   the stitch still answers; strip it from both and the read is loud.

   Two notes for whoever extends this. `_per_partition` (the
   tolerate-a-missing-rank wrapper) covers `read_elements` /
   `read_line_stations` / `read_gauss` / `read_fibers` but NOT
   `read_layers` / `read_springs` — harmless today only because both
   are always-empty stubs on the `.ladruno` reader that never raise
   `MissingElementResults`; if either ever becomes real, INV-6 breaks
   there. And `opensees_model()` reads **partition 0 only**, so when the
   contact owner is not rank 0 the minimal broker comes from a rank that
   knows nothing about the interaction.

   The suite deliberately does NOT declare a constraint handler: it
   rides the auto-emit, so it is also a live consumer of the S5
   open-item fix (a regressed hoist runs PlainHandler and the numbers
   stop matching). "Contact families" resolves to **ordinary node /
   element results on a model that has contact** — the fork exposes no
   recordable contact response at all (`LadrunoContactHandler` is a
   `ConstraintHandler`, `LadrunoContactFE` an analysis-layer
   `FE_Element`, `LadrunoContactDomain` not even a `TaggedObject`; no
   `setResponse`/`getResponse` anywhere, and `element/contact.py`
   defines no `Element` class, so no tag exists for a recorder to
   target). What contact changes is not *what* is recorded but *who*
   records it — which is the ownership contract itself.

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
