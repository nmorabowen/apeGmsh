# ADR 0115 — Read-back by label and label-addressed records

**Status:** Accepted, pending the maintainer's merge (2026-10-05; ratified by
the maintainer on #1434, comment 5987895202, with every recommendation
in §Questions decided at ratification)

**Owner:** nmora

**Evidence:** two independent architect briefs (`prog-architect-opus`,
`prog-architect-fable`) written on `38b482bb`, reconciled by the chain R
orchestrator on #1434. The briefs and the reconciliation stay on the
issue; this ADR cites them and does not copy them. Motivation:
`internal_docs/readability/REPORT.md` §5 ranks 1 (read-back) and 3
(supports), and `internal_docs/readability/api_gaps.md`.

**Builds on** [ADR 0114](0114-the-archive-is-the-program.md) (D1–D5) and
K0 (#1341). **Program link:** R1 of chain R (#1372); R2 implements it.

## Context

A model script cannot read a result, or address a support, by the label
it created. The workshop's scripts fell back to `ops_raw.nodeDisp(node.tag, …)`,
`ops_raw.wipe()` to flush recorders, `[node] = ops.nodes.get(pg=…)`,
node-id set algebra for shared corners, and a dim-0 physical group per
point just to make it addressable.

What the code already has, and what it lacks (both briefs, on `38b482bb`):

- Labels resolve on the snapshot (`FEMData.py::NodeComposite.select(label=)`,
  `ElementComposite.select(label=)`), and `Results.nodes.get(label=, component=)`
  already answers by label. The gap is on the bridge side, not the reader.
- `apeSees.fix`/`mass`, `_StageBuilder.fix`/`mass` and `recorder.Node`/`Element`
  take `pg=` or `nodes=` only; `ops.nodes.get(pg=)` always returns a `NodeSet`.
- `apeSees.analyze` returns an `int` and keeps the live emitter; `eigen`
  returns `EigenResult`. Neither has a read-back surface by label.
- Node tags are FEM node ids. Element tags come from the built tag map;
  `TagAllocator.freeze()` (ADR 0114 D4) arrives with K1-3 (#1361).
- `BuiltModel._validate_no_duplicate_fix_mass_across_tiers` refuses every
  repeated `(node, DOF)`, same tier included, and runs only on staged
  builds; an unstaged overlap fails late inside OpenSees.
- ADR 0114 D5's `name=` on `fix`/`mass` is not code yet (K1-6).

Constraints: ADR 0114 D2 (the Protocol's 75 methods are frozen), D3 (no
user `ops.command()`), D4 (tags are archive facts; read-back never
allocates), D5 (`name=` is the record's identity), and ADR 0113 (schema
changes are per-zone floors).

## Decision

### D1. Read-back is read-side, not a verb

Read-back adds no `VERBS` row, no Protocol method and no `command()`
call. Live reads go through `LiveOpsEmitter.ops`, already listed in
`verbs.py::SIDE_CHANNELS["live"]`. `analyze` keeps its `int` return and
`eigen` keeps `EigenResult` (its `.periods` gets documented).

### D2. Instant read on the live route

After a live `analyze`, the bridge keeps a read session: the frozen
`BuiltModel`, its emitter, and a domain epoch.

- `ops.nodes.get(pg=|label=)` returns a `NodeSet`; `ops.nodes.one(pg=|label=)`
  returns a `Node` and raises `BridgeError`, naming the count, unless
  exactly one node is selected.
- `Node.read(component) -> float`; `NodeSet.read(component) -> ndarray (N,)`
  in `.tags` order. A sum over a group is plain numpy:
  `ops.nodes.get(pg="base").read("reaction_force_y").sum()`.
- `ops.elements.get|one(pg=|label=|ids=)` works over FEM ids; `.read`
  returns `ndarray (E,)` and serves, in R2, only components with one value
  per element (Truss `axial_force`). Anything else raises and points to
  `Results`. Element reads wait on K1-3's frozen tag map.
- Components use the results package's canonical names
  (`displacement_x`, `rotation_z`, `reaction_force_y`, …), mapped per node
  through the bridge's own DOF layout (`build.py::_load_dof_layout`), never
  through `capture/_domain.py`'s table (it sends `rotation_z` to DOF 6,
  which fails on `ndf=3`). A reaction read calls `reactions()` first.
  Values are in model units at the last committed state.

### D3. History read: `ops.results(path)` returns a `Results`

The deck and staged routes cannot answer in-process. `ops.results(path)`
returns the existing `Results` after a `py`/`tcl(run=True)` run. `path=` is
required (Q5; ADR 0112 D1). The live route of `ops.results` is deferred (Q4):

- an `MPCO` recorder → `Results.from_mpco`;
- a `recorder.declare(...)` → `Results.from_recorders`;
- typed `recorder.Node` text recorders → a recorder spec synthesised from
  the typed records, read by `Results.from_recorders` (Q3);
- otherwise `BridgeError` naming `recorder.MPCO`, `recorder.declare` and
  `np.loadtxt`.

`NodeSlab.item() -> float` (raises unless `values.size == 1`) replaces
`.values[-1, 0]`. On the live route, `.read` (D2) is the answer. A `.read`
after a deck run raises and names `ops.results`.

### D4. `label=` (target) beside `name=` (identity)

`label=` (a string or a list of strings) is a third selector, beside `pg=`
and `nodes=`. Exactly one selector is allowed. It is accepted on `fix`,
`mass`, `s.fix`, `s.mass`, `recorder.Node`, `recorder.Element`,
`ops.nodes.get|one` and `ops.elements.get|one`. Records keep it, and
`BuiltModel._resolve_node_target` resolves it at build exactly as it
resolves `pg=`, so every validator sees it. An unknown label raises and
lists the labels that exist.

`label=` is an input consumed at build to pick nodes. `name=` (ADR 0114 D5)
is an output key under `/opensees/decls` and never selects. They are
separate keywords that may be combined; neither defaults to the other,
and a label never falls back to a physical group of the same name.

### D5. No cardinality polymorphism

A selector's return type never depends on the mesh. `get` always returns
a set, and `one` is the explicit single-node form. `[node] = ops.nodes.get(pg=…)`
keeps working.

### D6. Identical homogeneous fixes merge within a tier

Within one tier (the global pool, or one stage), homogeneous fixes form a
set of `(node, DOF)` pairs, whether they come from `fix`, `s.fix`,
`Node`/`NodeSet.fix` or `fix_from_model` (Q1). The first declaration emits a pair; a later
one emits only its remaining DOFs on that node, or nothing. A homogeneous
SP is idempotent at zero, so nothing is lost. The duplicate validator
runs on every build, staged or not. These still raise:

- a cross-tier overlap without `s.remove_sp` (the lifecycles differ);
- a `fix` against an `s.support` HOLD (ADR 0052);
- any `mass` overlap without `overwrite=True` (`setMass` is not idempotent);

## Invariants

1. `emitter/verbs.py` and the Protocol do not change; `test_verbs_lock.py`
   passes unedited.
2. A read makes no `TagAllocator.allocate*` call and no `build()` call.
3. Reads take and return FEM ids; a default build and an
   `element_tags="fem"` build read equal values.
4. A bridge-driven wipe after `analyze` (`eigen`, another run, another
   bridge) makes `.read` raise, naming it. `.read` never answers from a
   domain this bridge did not solve live.
5. A component the node lacks raises, naming the node and its ndf.
6. `fix(label=X)` and `fix(pg=X)` over the same nodes emit identical bytes.
7. Every model that builds today emits byte-identical decks and archives.
8. After merging, each `(node, DOF)` has at most one SP per tier.
9. No archive path, dtype or zone version changes (no ADR 0113 floor moves).

## Consequences

- R2 lands in four PRs under `lock:src/apeGmsh/opensees/apesees.py`,
  each with `--base main`:
  - **R2-a:** `label=`, `nodes.one`, the fix merge, the validator on every build;
  - **R2-b:** the instant read (D2); gated on K1-3;
  - **R2-c:** `ops.results(path)` (D3) and `NodeSlab.item`;
  - **R2-d:** the reference scripts, the skill mirror and the docs page.
- Oracles: Pratt-truss bar forces by label against the method of sections,
  with reactions summing to the load; a tag-map case with families declared
  in reverse; the 2-D cantilever tip `rotation_z` at `ndf=3` against PL²/2EI;
  the staged footing whose shared corners build and run; staleness and route
  refusals.
- KPI (`skills/apegmsh/references/model-scripts.md`): `raw-ops:` 2 → 0.
  `API gap` lines go from 10 to 7: the line-59 prose definition plus 9
  markers before, and 6 markers after R2. The ≤ 2 target also needs R3
  (ranks 2, 4, 5) and Q6.
- Additions only. Nothing is deprecated. The only behaviour change: a
  same-tier homogeneous fix overlap is merged instead of refused.

## Not built

`ops.command()` or new `VERBS` rows; a capture inside `analyze`; a
cardinality-dependent `get`; element histories; merging `mass` or merging
across tiers; `label=` on loads, `sp`, supports or regions (Q6); a label
falling back to a PG; unit conversion; staged live execution (K24); new
return values for `analyze` or `py` (R3).

## Questions decided at ratification

| # | Question | Decision |
|---|---|---|
| Q1 | Merge `fix_from_model` with an explicit homogeneous `fix` on the same DOF? | Yes: the physics is the same |
| Q2 | Naming: `nodes.one` (vs `only`); `ops.elements` beside the `ops.element` primitives | `one`; `ops.elements`, mirroring `ops.nodes` |
| Q3 | Typed text recorders on the deck route: synthesise a recorder spec so `ops.results` reads them, or add `ops.recorder.read(name)` (a second `.out` reader, waits on K1-6's `name=`)? | Synthesise the spec: one history reader, `Results` |
| Q4 | `ops.results(path)` on the live route (a one-step capture to file) | Defer: `.read` covers live, and the capture's DOF table must be fixed first |
| Q5 | Is `path=` required? | Required (ADR 0112 D1: every file is explicit) |
| Q6 | `label=` on `p.load` and the `rigid_diaphragm` master | Defer to R3; it removes the frame's `master_point` marker |
| Q7 | Are per-element scalars enough for R2's element read? | Yes |
