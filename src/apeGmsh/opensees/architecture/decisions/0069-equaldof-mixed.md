# ADR 0069 — Mixed-DOF tie via `equalDOF_Mixed`

**Status:** Accepted (2026-06-24). Extends ADR 0022 (MP-constraint
emission fan-out) with a **seventh** `Emitter` Protocol method
(`equalDOF_mixed`) and a new node-pair constraint kind
(`equal_dof_mixed`). First of the four "Ladruno constraints coverage"
clusters (see `architecture/plan_ladruno_constraints_coverage.md`). No new
geometry machinery — it reuses the `equal_dof` co-location matcher
verbatim. Standard OpenSees command (not fork-only): runs on any build.

## Context

`g.constraints.equal_dof(...)` ties DOF `i` on a retained node to the
**same** DOF `i` on a co-located constrained node — `ops.equalDOF(R, C,
*dofs)`. OpenSees also ships `equalDOF_Mixed R C numDOF RDOF1 CDOF1 …`
(`TclModelBuilder.cpp:3995`), which ties **differently-numbered** DOFs —
e.g. a solid's translation `uz` to a shell's drilling rotation `rz`, or
any cross-DOF coupling at an interface where the two sides expose the
coupled quantity under different DOF indices. apeGmsh had no way to emit
it: `equal_dof` is symmetric and `node_to_surface` is mixed-*ndf*, not
arbitrary DOF pairing. This was the smallest, semantically-unambiguous
gap in the constraints-coverage audit, so it lands first.

## Decision

Mirror the `equal_dof` vertical, carrying a second per-pair DOF list:

* **Kind** — `ConstraintKind.EQUAL_DOF_MIXED = "equal_dof_mixed"`, added to
  `NODE_PAIR_KINDS`.
* **Record** — reuse `NodePairRecord` (no new record type) with one new
  optional field `master_dofs: list[int] | None`. For `equal_dof_mixed`,
  `dofs` holds the **constrained** (CDOF) list and `master_dofs` the
  **retained** (RDOF) list, paired by index (`len` equal). `None` for
  every other kind, where the two are identical and `dofs` alone suffices.
* **Def / API** — `EqualDOFMixedDef` (NOT an `EqualDOFDef` subclass) +
  `g.constraints.equal_dof_mixed(master_label, slave_label, *,
  dof_pairs=[(rdof, cdof), …], tolerance=, …)`. `dof_pairs` is validated
  non-empty with 1-based ints at construction.
* **Resolver** — `ConstraintResolver.resolve_equal_dof_mixed` reuses
  `_match_node_pairs` (identical co-location semantics to `equal_dof`),
  emitting one `NodePairRecord(kind=EQUAL_DOF_MIXED, dofs=cdofs,
  master_dofs=rdofs)` per matched pair. Wired into both dispatch tables
  (`_DISPATCH` / `_RESOLVER_METHOD`, the meshed path) and the chain-phase
  router (`_route_equal_dof_mixed`, the build path). The router checks
  `EqualDOFMixedDef` **before** `EqualDOFDef` (it is not a subclass, but
  the guard documents intent).
* **Protocol (architecture event)** — `equalDOF_mixed(self, master, slave,
  dof_pairs: Sequence[tuple[int, int]])`, implemented on all five concrete
  emitters. tcl/py/live/recording emit `equalDOF_Mixed $R $C $numDOF
  $RDOF1 $CDOF1 …` (master = retained R, slave = constrained C). `build.py
  _emit_equal_dofs` grew the `EQUAL_DOF_MIXED` branch (`zip(master_dofs,
  dofs)`).
* **Persistence** — the canonical FEMData snapshot
  (`_femdata_h5_io.py`) round-trips `master_dofs` via a new
  `master_dofs` vlen-int64 column on `node_pair_payload_dtype`
  (**neutral schema 2.17.0**, additive; empty array ⇒ `None`; reader
  presence-probes for ≤2.16.0 files; the `NODE_PAIR_FIELDS` subset
  dispatch is unperturbed).

## Deferred — OpenSees-deck H5 archival

The OpenSees **deck**-archival emitter (`emitter/h5.py`,
`apeSees(fem).h5(path)`) does **not** archive `equalDOF_Mixed`: its
`equalDOF_mixed` raises a clear `NotImplementedError` rather than silently
dropping the constraint, exactly like the staged-mutator / `rayleigh` /
`modal_damping` H5 deferrals. The canonical constraint round-trip is the
FEMData snapshot (above), which is fully wired; deck-archival parity (a
new `equalDOF_Mixed` compound dataset + reader + emit-index sequencing) is
tracked as a follow-up. tcl / py / live emission is complete, so models
using `equal_dof_mixed` run on every path except `.h5()` deck export.

## Consequences

* New public API `g.constraints.equal_dof_mixed`; no change to existing
  `equal_dof` behaviour or emitted bytes.
* The `Emitter` Protocol gains its seventh MP method; all five emitters
  conform (INV-4). Schema 2.16.0 → 2.17.0 (additive minor, two-version
  reader window per ADR 0023).
* `NodePairRecord` gains one optional field defaulting to `None`, so every
  existing construction and round-trip is byte-stable.
