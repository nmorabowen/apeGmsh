# ADR 0117 — Assembly (compose v2): instances of model files, rehydrated into one forward bridge

**Status:** Accepted (2026-10-06). Design brief by the Fable architect on
issue #1517 (grounded on `main` `ad515ffe`), ratified by the maintainer on
the same issue, T session, 2026-10-06. The ratification amends the brief at
three points: the carry rule (D4), the migration (D7, replacing the brief's
Q7 and P5) and open question 6 (no deprecation period).

**Owner:** nmora

**Supersedes in part:** [ADR 0038](0038-compose-model-composition.md) (the
host/module asymmetry, the `.`/`/` separator alternation and the materials row
of its merge-semantics table). The amendment to 0038 is made by the removal
slice (chain AS, link AS5), not by this ADR.

**Builds on** ADR 0085 (a `Part` is geometry only; the unit of authorship is a
saved model file), 0086 and 0068 (`tie` with `method=` and `enforce=`), 0112
(D2: a new kind of data is its own zone), 0113 (D1: compatibility is a floor
per zone), 0114 (D2 frozen Protocol; D4 tag law) and 0115 (D4: `label=` is a
target, `name=` an identity). **Program link:** AS0, issue #1520, chain AS
(#1519).

## Context

Today "compose" is a **merge**. `mesh/_compose.py::Compose.compose` calls
`mesh/FEMData.py::FEMData.compose`, which reads the source's neutral zone
only (`_rewrite_source_for_compose` through `read_fem_h5`), offsets every FEM
id by `base - source_min_tag` in 1M windows, prefixes names with `{label}.`,
concatenates (`_merge_bundle_into_fem`) and rebuilds one partition rank per
module (`_rebuild_partitions_from_modules`). `assembly.py::Assembly` is a thin
declarative wrapper (`add`, `couple`, `materialize`; kinds `equal_dof`,
`tied_contact`, `tie`) that fails loud on a zero-record couple.

By inspection of the live code, five things are missing or fixed.

1. **There is a host and there are modules, not instances.** The first file
   is the un-namespaced host (`apeGmsh.from_h5`); every later file is
   prefixed. `Assembly._resolve_port` encodes the asymmetry. A model file
   cannot be instanced twice as an equal, and a bare port silently means
   "the host".
2. **The instance's model content is not carried.** ADR 0038 says materials,
   sections and integration rules import, but `_rewrite_source_for_compose`
   never opens `/opensees/`; `_filter_kind_counts` reads it only to warn. The
   user re-declares every material on the host bridge. `rebar_elements` is
   dropped with `ComposeDroppedStreamWarning` because the bridge-side
   material name was undecided (`_compose.py` L170-181).
3. **The assembly's declarations are not persisted.** `/composed_from/{label}`
   (`mesh/_femdata_h5_io.py::_write_composed_from`) stores placement and
   hashes; ties survive only as resolved `/constraints/{kind}` rows.
4. **The tag law.** ADR 0114 D4 (amended in K1-3d): every tag is planned by
   `opensees/_internal/tag_plan.py::plan_tags`, then
   `TagAllocator.freeze()`; a later mint raises `TagLawError`. The waiver the
   program ledger names, `reinforce-tie-replay`, lives in
   `opensees/_internal/compose.py::_replay_into`. That is the
   archive-to-deck replay (ADR 0018 and 0055, "model.h5 composition"), not ADR
   0038 compose. It mints tie element tags because no archived row carries
   them (ADR 0114 amendment §5: the ledgered reinforce-tie and contact tags
   get their home when K2 flips those rows). Compose v2 must not add a second
   consumer of that fallback.
5. **Fixed contracts.** The Protocol is frozen (ADR 0114 D2); `label=` is a
   target and `name=` an identity (ADR 0115 D4); schema changes are per-zone
   floors and a new kind of data is its own zone (ADR 0113 D1, ADR 0112 D2);
   `Part` stays geometry-only (ADR 0085); new features go in new modules
   (#1202); `tie` is the permanent-bond verb (ADR 0086, 0068).

## Options

**A. Archive merge plus replay.** The assembly is an H5-to-H5 transform: it
relocates every tag-bearing column of each instance's neutral and `/opensees`
zones into one flat archive and emits through the single archive-to-deck path
(K2). Cost: a second rewrite cover set over `/opensees/*` that must track
every K1-4 to K1-8 minor (`commands`, `program`, `decls`, `decl_params`,
per-row parameter tags) while chain K is reshaping that zone. The deck would
come from `_replay_into`, which is secondary and partial today (it replays no
MP family) and is exactly where the waiver lives. The assembly's own ties
would still need a forward emit, so one deck would have two line sources.

**B. Rehydrate plus forward build.** Keep the FEM-side merge (it exists and is
tested). Add a **rehydrator** that reads each instance's `/opensees` model
content through `opensees/opensees_model.py::OpenSeesModel.from_h5` (typed
materials, sections, transforms, beam integrations, elements) and registers it
on one `apeSees` under `{instance}.{name}`. Assembly ties are chain-phase
constraints resolved by `_kernel/resolvers/_chain_phase_router.py::route_def_to_fem`
into ordinary records. One emit path, one plan (`plan_tags` over the assembly
`BuiltModel`), so the tag law holds by construction. Cost: an instance's
OpenSees tags are re-planned in the assembly (no per-instance tag invariance;
FEM ids keep the relocation), and element rehydration from
`/opensees/element_meta` rows needs a per-row selector in the general case.

**C. Status quo plus carry.** Keep host/module and teach `g.compose` to carry
materials. Cost: the asymmetry stays, nothing is persisted, and two
namespacing rules live forever.

## Decision

**Option B.** An assembly is a set of **instances** of **model files**,
joined by **assembly-level constraints declared by label**, built into one
forward bridge.

- **Part** = a saved `model.h5` from a full session (the unit of authorship
  that ADR 0085 already made). No new class; apeGmsh's `Part` stays CAD-only.
  The same file instanced twice is read once, deduplicated by
  `source_fem_hash`.
- **Instance** = file + label + rigid transform. Every instance is
  namespaced; there is no host. Assembly-owned objects (reference nodes,
  ties, fixes) have bare names. An instance label contains no `.`, `/` or
  whitespace (today's `Compose._validate_label`), so a path splits on its
  **first** dot: `pier_1.top`, and `bridge.pier_1.top` when an assembly file
  is itself instanced. ADR 0038's `.`/`/` alternation is not used for v2
  files.
- **Assembly** = a new package `src/apeGmsh/assembly/` replacing
  `assembly.py`; the import path `apeGmsh.assembly` is kept.

```python
from apeGmsh.assembly import Assembly

asm = Assembly("viaduct")
asm.instance("pier_1", "pier.h5")                         # same file, twice
asm.instance("pier_2", "pier.h5", translate=(30, 0, 0))
asm.instance("deck",   "deck.h5", translate=(0, 0, 12))
asm.tie("deck.soffit", "pier_1.top", enforce="equation", method="mortar", name="t1")
asm.tie("deck.soffit", "pier_2.top", enforce="equation", method="mortar", name="t2")
asm.node("abutment", (-1.0, 0.0, 12.0))                   # assembly-owned decoupled node
asm.couple("abutment", "deck.west_edge", kind="kinematic", dofs=[1, 2, 3])

ops = asm.bridge(ndm=3, ndf=6)       # apeSees over the flat FEMData; instance materials,
                                     # sections, transforms, elements registered as
                                     # "pier_1.concrete", "deck.girder", ...
ops.fix(label="pier_1.base", dofs=[1, 2, 3])
ops.fix(label="pier_2.base", dofs=[1, 2, 3])
with ops.pattern.Plain(ts) as p:
    p.from_model("deck.gravity")     # an instance load case, opt-in (as today)
ops.analysis(...); ops.analyze(10)
ops.h5("viaduct.h5")                 # a plain model.h5 plus the /assembly zone
```

### D1 — Model

The user surface is `Assembly.instance`, `tie`, `couple`, `node` and
`bridge`, with the verbs named as above (open question 1, decided: the new
names). `g.compose` remains the engine's primitive (`FEMData.compose`, later
renamed `_merge_instance`), not the user surface; D7 says what happens to it.

### D2 — Namespaces and tags

Two tag spaces, two rules.

- **FEM ids** (nodes, FEM element ids, physical-group tags) are relocated per
  instance in declaration order by today's `_compute_reservation`, and
  recorded per instance in `/assembly/instances` (`fem_id_base`,
  `fem_id_span`).
- **OpenSees tags** are planned by `plan_tags` in registration order, which
  the rehydrator makes deterministic: instance order, then each instance's
  archived family order, then assembly-level objects. No allocation site is
  added, so the K1-3 AST lock extends to `src/apeGmsh/assembly/` unchanged.
  Node tags stay FEM ids (ADR 0115 INV-3).
- `asm.bridge()` defaults to `element_tags="fem"` (open question 2, decided),
  so an element tag minus its instance base is the instance-local id.
- Reinforce and embed ties, interfaces and contacts **inside** an instance are
  neutral-zone records already carried today and already planned from `fem` by
  `tag_plan.py::_plan_mp_elements_and_interfaces`. Assembly ties are the same
  record kinds, so the planner does not change.
- Names: `pier_1.concrete`; ADR 0114 D5 declaration keys become
  `opensees/uniaxial/pier_1.concrete`. Clashes are impossible by
  construction.

### D3 — Ties

Routable in chain phase today (`_chain_phase_router.py`): `tie` (collocation
or mortar; penalty, penalty_al or equation), `equal_dof`, `equal_dof_mixed`,
`rigid_link`, `penalty`, `rigid_diaphragm`, `embedded`, `tied_contact`,
`kinematic_coupling` / `rigid_body` (RBE2) and `distributing_coupling` (RBE3).

v2.0 exposes `tie`, `equal_dof`, `rigid_link`, `rigid_diaphragm`, `embedded`
and `couple(kind="kinematic"|"distributing", reference=<assembly node>)`.
`contact`, `interface`, `g.embed` and `g.reinforce` are not routable
(`ChainPhaseError`); they stay instance-internal in v2.0. Cross-instance
`contact` is phase 6 (ADR 0092's locality rule holds when an instance is a
rank).

Every port is `{instance}.{pg|label}`. A bare port names an assembly object
or raises, listing the instances. Each declaration takes `name=` and reads
back by label (`ops.nodes.get(label=...)`, `Results.nodes.get(label=...)`;
ADR 0115).

### D4 — What an instance carries

**Carry rule, in the Abaqus form: model content travels with the instance;
analysis content (fixes, masses, patterns, recorders, stages, analysis) is
assembly-level.** This is today's neutral-zone verdict table, extended to the
bridge zone.

- **Carried:** mesh, physical groups, labels, selections, parts maps, node
  ndf, decoupled nodes, every intra-instance constraint including
  `/reinforce_ties`, `/embed_ties`, `/interfaces`, `/contacts` and now
  `/rebar_elements` (its bond name becomes `pier_1.<bond>`), load and SP cases
  and masses (opt-in at the bridge, as today), and from `/opensees`:
  materials, sections, transforms, beam integrations, element specs and
  dampings.
- **Filtered, with the existing warnings:** patterns, time series, recorders,
  stages, analysis, initial stress, rayleigh, regions, cuts, partitions,
  results.
- **Not carried:** `/opensees/bcs` fixes and instance `mass` bridge records
  (open question 3, decided). The neutral `bc` and `masses` records travel and
  are opt-in through `fix_from_model` and `mass_from_model`.

### D5 — Persistence

The assembly archive is a plain `model.h5` (neutral plus `/opensees`) written
by `ops.h5()`, so every reader (`FEMData.from_h5`, `OpenSeesModel.from_h5`,
`Results`, the viewers) opens it unchanged, plus one new optional zone (open
question 4, decided: a new `/assembly` zone):

```
/assembly                     @assembly_schema_version "1.0.0" (own key in /meta; ADR 0112 D2)
  /instances   label · source_path · source_fem_hash · source_opensees_hash ·
               translate(3) · rotate(4) · fem_id_base · fem_id_span · partition_rank
  /ties        name · kind · master · slave · params(json) · n_records
/composed_from                still written (floor-compatible; ColorMode.MODULE, module_label)
/provenance/records           assembly/instances/<label>, assembly/ties/<name>  (1.1.0, origin=user)
```

Archive completeness (K0/K1 VERBS): `archive`. Instance files are inputs and
are never re-fetched (ADR 0038 INV-5 kept). `Assembly.from_h5` re-lists
instances and ties; the flat zones stay authoritative. The neutral and
opensees schema versions do not move. The `/composed_from` writer retires when
`ColorMode.MODULE` reads `/assembly`.

### D6 — Partitioned models

Unchanged by construction. The flat FEMData keeps ADR 0038's three-layer rank
model (`partition_rank=` per instance); cross-instance ties are ordinary
cross-partition MP constraints (ADR 0027 replication; S7 rank-invariant
tags); and the resident-graph pass (ADR 0100) sees one flat model. The
two-rank test belongs to P4 (AS4, #1530), not the first slice. Until AS4
settles the rank layout, the merge engine keeps rank 0 for a host that v2 does
not have, so an assembly's default `tcl()` is partitioned with an empty rank 0;
`bridge()` warns, and `tcl(flat=True)` is the serial deck (AS1, #1529).

### D7 — Migration: today's compose is deleted, not deprecated

**This replaces the brief's Q7 and P5.** `g.compose`, `FEMData.compose`,
`apeGmsh.compose` and `Assembly.add`, `Assembly.couple` and
`Assembly.materialize` are **deleted with no deprecation period** once v2
covers their use. The removal is a separate, human-gated slice (chain link
AS5) that runs after v2's parity is proven. The FEM merge engine survives only
as a private helper of `src/apeGmsh/assembly/`.

ADR 0038 is amended by the removal slice, not here (host asymmetry and
separator alternation retired for v2 files; the materials row corrected).
ADR 0041, 0085 and 0086 stand.

The reinforce-tie replay is **not compose v2's to delete**: it stays as
today (S6 option (a)), the ledger row goes when K2 archives `embedded_rebar`
tags, and v2 never calls `_replay_into`, so nothing new depends on the
fallback.

### D8 — Scope and the remaining open questions

Phase 1 is two instances of one file, one `tie`, `bridge()`, an `h5()` round
trip and a live solve (see Phases).

Open question 5 (decided): rotation is `rotate=((ax, ay, az), theta_rad)` with
optional `center=`, through one helper shared with `g.parts.add`; P1 fixes the
`ComposeRecord` docstring, which today says quaternion where `Compose.compose`
documents axis-angle. Open question 7 (decided): P2's element selector takes
the selection route (a mesh selection per rehydrated spec, registered by
`selection=`) if it avoids the `apesees.py` lock; it also keeps the selection
visible in the neutral zone. Otherwise P2 adds an `ids=`/`selection=`
selector under `lock:src/apeGmsh/opensees/apesees.py`. Open question 6 is
superseded by D7.

## Invariants

Each is testable.

1. **INV-1.** Every instance-owned name is `{label}.{local}`; no
   assembly-owned name contains `.`. `instance()` with a label containing `.`,
   `/` or whitespace, or a clashing label, raises `AssemblyError`.
2. **INV-2.** Two runs of the same assembly script yield byte-identical
   `h5dump` of the archive (excluding the volatile stamps `meta@created_iso`,
   `meta@session_id` and `composed_from/*@composed_at`; see the
   [INV-2 amendment](#amendment--inv-2-volatile-stamps-october-2026)) and
   byte-identical decks.
3. **INV-3.** No emit-time mint: `tests/opensees/contract/test_tag_law_lock.py`
   covers `src/apeGmsh/assembly/` (no `TagAllocator(`, no `allocate*`); a
   planted mint raises `TagLawError`.
4. **INV-4.** An assembly of one instance at the identity transform emits a
   serial deck (`tcl(flat=True)`) equal to the instance's own
   `apeSees(fem, element_tags="fem").tcl()` after stripping the `inst.`
   prefix from names and shifting the relocated FEM ids back by
   `fem_id_base - source_min` (rehydration parity).
5. **INV-5.** Carried-set exclusivity: the assembly bridge's `names` equal the
   union of each instance's `OpenSeesModel.from_h5(...).names` for model
   kinds, prefixed, and contain no pattern, recorder, stage or analysis name.
6. **INV-6.** Instancing is exact: for each instance, assembly coordinates
   equal `R·x + t` within 1e-12 and connectivity equals the source's plus
   `fem_id_base - source_min`.
7. **INV-7.** A tie that resolves zero records raises `AssemblyError` naming
   both ports; a bare port that names no assembly object raises, listing the
   instances.
8. **INV-8.** Physics: two stacked instances of one block file, `nu=0`,
   `enforce="equation"` tie gives `K = EA/(2H)` within solver precision;
   `method="mortar"` within 0.1 % (the ADR 0085/0086 rigs).
9. **INV-9.** Partitioned: a two-instance, two-rank emit writes each
   cross-instance MP line on both owning ranks with identical tags, and its
   tag set equals the flat emit's.
10. **INV-10.** `/assembly` is the only new zone; the neutral, opensees and
    provenance versions are unchanged; a file without `/assembly` opens as
    before; `Assembly.from_h5(h5).instances` round-trips.
11. **INV-11.** The `emit-cost-gate` ratio is unchanged (the gate cell has no
    assembly).

## Phases

Each is one PR slice of about ten files, every one `--base main`.

- **P1, the object and the first oracle** (no hub lock):
  `src/apeGmsh/assembly/{__init__,_assembly,_instances,_rehydrate}.py` (v1
  names re-exported), `tests/assembly/test_two_instances_one_tie.py` (INV 1, 4,
  6, 7), `tests/assembly/test_live_stack.py` (INV 8, `live`),
  `tests/opensees/contract/test_tag_law_lock.py` (one more directory) and a
  changelog fragment. P1 rehydrates the common case, one element spec per
  physical group. Stop condition: element rows whose args vary within a group
  go to P2.
- **P2, general rehydration:** per-row element args and the selector of D8,
  dampings, the `/rebar_elements` carry (retires
  `ComposeDroppedStreamWarning`; `tests/mesh/test_compose_dropped_streams.py`
  flips from "warns" to "carried" with the bond name checked), INV-5.
- **P3, persistence:** the `/assembly` writer and reader in
  `src/apeGmsh/assembly/_h5.py`, the key in
  `opensees/_internal/schema_version.py`, `architecture/h5-schema.md`, one
  corpus file (ADR 0113 D8), provenance records, `Assembly.from_h5`, INV 2 and
  10. `tests/opensees/h5/test_h5_schema_compat.py` gains the `/assembly` key.
- **P4, couplings and partitions:** `node`, `couple(kind=kinematic|distributing)`,
  `rigid_diaphragm`, `embedded`, `partition_rank`, INV-9
  (`tests/opensees/integration/`), the docs how-to, and the skill through
  `scripts/sync_skill.py`.
- **P6 (gated on K2):** cross-instance `contact` and an instance-of-assembly
  test. The `reinforce-tie-replay` ledger row is deleted by K2's PR, not here.

There is no P5 deprecation slice: D7 replaces it with the human-gated removal
(chain link AS5).

## Oracles that catch a wrong implementation

- **Parity (INV-4)** catches a rehydrator that drops, reorders or renames a
  family. Mutation probe: comment out section registration; the diff is
  non-empty.
- **Same file twice** catches bare-name re-declaration: both instances of
  `pier.h5` carry `concrete`; the test asserts `pier_1.concrete` and
  `pier_2.concrete` both exist and the deck has two `uniaxialMaterial` lines.
- **Closed form (INV-8)** catches a tie resolved against the wrong face or
  instance (K off by one block) and a penalty default sneaking in (-4.47 % at
  1e18, ADR 0085).
- **Tag law (INV-3)** catches any emit-time mint, including one inside
  `_rehydrate.py`.
- **`h5dump` byte identity (INV-2)** catches dict-order nondeterminism in
  rehydration.
- **Partitioned equality (INV-9)** catches a rank-local counter (the PR #329
  class of bug).
- **Floors:** a diff on the neutral or opensees schema constants fails
  `lock-tests`.

## What not to build

- A meshable `Part` (ADR 0085 stands).
- A `/opensees` H5-to-H5 relocating merge or a second emit or replay path
  (Option A).
- A per-tag `tag_blocks` table (ADR 0114 amendment §5).
- `uncompose` or `recompose`.
- Tree machinery for nesting: an assembly file is just another instance
  source.
- Cross-instance `contact`, `interface`, `embed` or `reinforce` in v2.0.
- Carrying instance patterns, recorders, stages or analysis.
- Automatic tie-pair detection.
- A per-instance split deck (ADR 0043 withdrawn).
- Any change to the Protocol or `VERBS`.
- A deprecation shim for today's compose surfaces (D7).

## Relation to earlier decisions

- **ADR 0038:** superseded in part (header). The merge engine it describes
  survives as a private helper; its host asymmetry, separator alternation and
  materials row are retired for v2 files by the removal slice.
- **ADR 0114:** consistent. No allocation site is added, the Protocol and
  `VERBS` do not change, and the waiver in `_replay_into` keeps one owner (K2).
- **ADR 0112, 0113:** consistent. `/assembly` is a new zone with its own
  version key; no existing floor moves.
- **ADR 0085, 0086, 0115:** stand.

## Amendment — INV-2 volatile stamps (October 2026)

Accepted by the maintainer's ruling on
[#1550](https://github.com/nmorabowen/apeGmsh/issues/1550#issuecomment-6030918108).
INV-2's volatile list gains `composed_from/*@composed_at`, beside
`meta@created_iso` and `meta@session_id`. The compose writer stamps each
`/composed_from` record with the wall-clock time of the merge, so two runs can
never match on that attribute. The writer does not change: `composed_at` stays
a wall-clock stamp until `/composed_from` retires under D5. The exclusion is
matched by exact path (`*` is one path segment), so the same attribute name
anywhere else is still compared.
