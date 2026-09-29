# apeGmsh architectural strains: assessment and agentic-development operating model

**Status:** Assessment, 2026-09-28, against `origin/main` @ `07f757e0`.
The human decisions still open are Charter v2 (§5), the remediation order
(§6) and the workflow operating model (§7).

**Scope:** all of `src/apeGmsh` (632 modules, about 301k LOC). The emphasis
is on the apeSees bridge: declaration, emission, native results capture and
persistence. It also covers the agentic development workflow that produces
the code.

**Method:** nine read-only audits ran in parallel, one per slice (§9). Each
had to cite file:line or commit evidence and to count the touch points of a
representative change. The load-bearing claims were then re-verified
independently (the verification log is in §9). Live probes ran against this
worktree's `src` through `PYTHONPATH`.

**Errata (2026-09-28, expert panel review).** A 9-member panel (see
`internal_docs/plan_expert_panel_2026-09.md`) re-checked this assessment. The
moderator verified the corrections below, which supersede the original text
where the two disagree. The affected lines are also corrected in place.

| # | The original claim | The correction |
|---|---|---|
| E1 | "58% of fix PRs repair files a feature PR touched ≤7 days earlier" (§0, §7.1). | **Withdrawn.** The figure fails its control: 78% of *non-fix* PRs also touch a file changed in the prior 7 days. It measures churn, not merge-then-fix. |
| E2 | Stock and fork strut-and-tie results "differ by 20–30%" (RS7). | **0.79%** at the capped 0.05 mm increment (`CHANGELOG.md:542`). The 18% came from coarse stepping. The `interop/strut_tie.py:585` docstring is stale. |
| E3 | "Equation ties run 71% too soft on stock" (RS7). | **Very likely a misdiagnosis.** The test stacks `stdBrick` on stock `TenNodeTetrahedron`, which is 6× soft upstream (`xsj = Jdet`), and a 6×-soft block in series gives 2/7, i.e. 71.43% soft. A stock run with the plate meshed as tet4 or hex8 is queued to confirm. |
| E4 | "`tens_stiff_c` was 500 against the fork's 200" (RS7). | **Wrong direction.** The fork's `ladruno` branch still defaults to 500. The 200 came from fork PR #873, which was closed without merging. #1184 therefore emits `-crackedNu`/`-betaC`, which `ladruno` silently ignores ("unknown tokens are ignored"). A fix task is queued. |
| E5 | `assess()` depends on the legacy `ResultsViewer` (RS6). | It depends on the *director*, through `viewers/render.py`. Only `export_animation`, `show_web` and `serve_web` instantiate `ResultsViewer`. The total of about 23.9k LOC (44 modules) holds. |
| E6 | "72% of PRs touch an index or generated file" (§7.1). | That figure includes August merge commits that carry no file list. Over squash-era PRs the share is **80–85%**. |
| E7 | Phase 3 splits are "proven by the byte-identity corpus" (§6). | **That corpus does not exist.** There is one 94-line golden deck, and the 67 parity tests are in-process A/B comparisons. The panel requires a committed corpus plus an AST move-proof before any split. |

---

## 0. The answer

apeGmsh does not grow by *extending* abstractions; it grows by
*accretion*. The project actually grows along six axes:

- new OpenSees vocabulary, mostly from the Ladruno fork;
- new declaration kinds;
- new execution modes;
- new result responses and formats;
- new persisted fields;
- new presentations.

None of these axes has an extension point. A feature lands by hand-editing
N parallel enumerations:

- tables keyed by strings;
- per-kind lists;
- per-mode drivers;
- per-format readers;
- hand-written writer/reader pairs;
- Protocol methods implemented once per backend.

N is 8 to 15 today and rising. The enumerations are wired so that a missed
site fails *open*. A `None` capability means "don't check". A
`getattr(..., default)` returns `{}`. A no-op Protocol method drops the
call. A replay skips a stream.

The components themselves are fail-loud; the seams between them are
fail-open. That combination is why the audit found eight silent
wrong-answer defects in shipping code (§3). The "58% of fix PRs" statistic
that first appeared here fails its control, and is withdrawn (Errata E1).

Five structural conditions amplify this:

1. **Semantics are dropped at every lowering boundary** and re-derived
   downstream. There is no IR between declarations, deck and archive; none
   between record intent and the formats; none between results and
   pictures.
2. **Boundaries exist only as convention.** 71% of cross-package runtime
   imports sit in function bodies, 48% of cross-package imports target
   private names, and several layering ADRs are false.
3. **A few hub modules absorb every concern.** 15 files exceed the
   2,000-line agent read window, up from 4 in May. `apesees.py` has 14,181
   lines, about 21% of which is declaration code.
4. **The evolution policy taxes growth.** Archives expire after two schema
   bumps, retired designs are de-published rather than deleted, and shared
   index files collide.
5. **The normative layer describes the system as it was in May.** That
   layer is the charter, the layout doc, the ADR index, the deferral
   register and private memory. Agents are told to judge code against it,
   and it no longer lets them.

The Ladruno fork cuts across all of this. It accounts for half of recent
work, yet no CI lane executes it. About a dozen ad-hoc mechanisms gate it,
and its options leak into the "neutral" core and schema.

The good news: every remedy already has a working instance in this repo
(§4).

- The diagram kind registry cut a new diagram kind to 3 files.
- `tag_rewrite_spec` lets compose walk records generically.
- The results zone uses path constants and has not needed a schema bump in
  four months.
- `RecordingEmitter` is an IR seed already used by 110 test files.

The program in §6 generalizes these patterns, in an order that keeps each
step safe for parallel agents.

For agentic development (§7), the highest-leverage improvement is
architectural: make every growth axis **local-change-safe**. That means one
registration per feature, enumerations derived from it, and completeness
gates that fail closed. Agents are strong at local edits and weak at
global enumeration, and today's architecture needs the latter for
correctness.

These workflow items pay for themselves within days:

- conflict-free index files (72% of PRs touch one);
- a `land_pr.py` script replacing seven prose rules;
- a pre-merge adversarial review on risky paths;
- cutting per-session context (MEMORY.md, the stale onboarding docs) by
  60–80%.

---

## 1. Accretion, not extension: the cost of the next feature

The table gives the touch points needed for a representative change on
each growth axis, measured from real commits.

| Growth axis | Witness | Touch points today | How a miss fails | Working in-repo precedent |
|---|---|---|---|---|
| New element or material | LadrunoUP #793; SANISAND ADR 0101 | About 10 source files. The token spreads over 16–18 files. Facts sit in 21+ string-keyed tables across 4 identity namespaces (Tcl token, C++ class name, class tag, gmsh type). | The capability lookup returns `None`, so the gate is skipped: a "false negative (silent)" (`_element_capabilities.py:729-732`). | LadrunoUP shape tables with derived floors |
| New kwarg on a primitive | #1184 (`07f757e0`); #1097/#1099 | 3 files: the dataclass, the ns wrapper, `_api_index.json`. 938 parameters are restated three times. | The wrapper keeps the old default (the `tan_type` incident) or lacks the parameter entirely (the #1184 TypeError). | None |
| New OpenSees command | `05fd0374`, `10845c14` | The same 8 files every time: base, tcl, py, live, h5, recording, build, apesees. The Protocol grew from 29 to 74 methods between 05-10 and 09-07. | H5 no-ops 12 methods unconditionally. | `_build_embedded_flag_args`, the one shared argument normaliser |
| New stage verb | #1103/#1104/#1106, all on 09-07 | 6–8: StageRecord field, builder verb, freeze, 2 drivers, H5 capture or refusal, staged replay. Making it archivable adds about 7 more. | H5 refuses; replay skips. | The `_StageDampingNS` sink override |
| New declaration kind | ADR 0093 `interface` (47 commits, 18 files); `g.reinforce` (9+ PRs) | About 15: verb, def, record, resolver, factory block, FEMData slot, dtype, writer, reader, version bump, compose bundle/rewrite/merge/verify, hash policy, chain route, cache bump, emit pass, stage claim, viewer tab. | The cache is not bumped (§3 D2). Compose dropped contacts and embed ties (#912/#913): the model "solved as an unbonded free-body problem". | `tag_rewrite_spec`, which compose walks generically |
| New element response | ADR 0108 (#1132 + #1140) | 2 PRs, 29 files, about 1,690 lines. It is still `.ladruno`-only. | Three copies of the nodal DOF map already disagree, and the per-category sets have diverged. | `apeGmsh._vocabulary`, the single name authority |
| New persisted stream | interface, `16c0581c` | 14 files, +1,342 lines. The viewer never sees the new stream. | Positional 46-tuple encoders, "coupled by index alone" (`_record_h5.py:548-553`). | The results zone (path constants, no bump in 4 months) |
| New persisted bridge field | computed_sections #834 | 15 files, 3 of them version pins. | It is written but has no `H5Model` accessor and no doc section. | None |
| New diagram kind | principal glyph #784 | **3 viewer files**, plus separate matplotlib and JavaScript copies. | The JS frame math diverged (§3 D5). | The declarative kind registry (ADR 0058 S0), the model to copy |

Two things follow from the table.

- **Cost tracks enumerations, not physics.** A feature costs as much as the
  number of enumerations that must learn a new name.
- **Where a registry exists, cost is already low.** Diagram kinds show it.
  The problem is missing extension points, not the domain.

---

## 2. The eight root strains

The nine audits found about 50 slice-level strains. Traced to their
mechanisms, they collapse into eight roots. §8 maps each slice finding to
its root. Severity and trend are the auditors' judgement after reconciling
across slices.

### RS1: Facts without a home. Knowledge lives in parallel tables kept in sync by hand. (High, growing)

**Evidence**
- **Element facts** live in 21+ tables spread over:
  - `_element_capabilities.py` and `_response_catalog.py`;
  - `emitter/live.py:158,173` (the fork rosters);
  - `apesees.py:329` and `build.py:4449`;
  - `results/_plane_recovery.py`, `results/_shell_geometry.py` and
    `mesh/_femdata_mpco_io.py`.

  Element classes carry no capability ClassVars at all, and `_ELEM_REGISTRY`
  has 10 keys with no typed class. The claim in `api-design.md` that "the
  class IS the registry entry" is false.
- **Response facts** live in 111 module-level tables across `results/` and
  the four bridge response modules.
  - The nodal canonical-to-DOF map exists three times, and the copies
    disagree on the pressure sentinel: `_recorder_translate.py:54-96`,
    `spec/_emit.py:52-80` and `capture/_domain.py:224-247`.
  - The per-category component sets have diverged: `recorder.py:1260` vs
    `capture/spec.py:116`.
  - 48 minted material-bucket names are missing from `_vocabulary.py`.
- **Geometry facts:**
  - The gmsh element-type tables are inlined per composite. The loads and
    masses `_target_faces` tables omit quad9 (§3 D1).
  - The Gmsh→VTK table exists three times: `fem_scene.py:55`,
    `mesh_scene.py:45` and `viz/VTKExport.py:53`.
  - The beam and local-axis math also exists three times: the viewer, the
    "pure-numpy copy" in `results/plot/_beams.py`, and the JavaScript.
- **Signatures:** 182 of the 200 ns wrappers restate a dataclass's fields.
  That is 938 field-to-parameter pairs, each written three times.
- **Schema:** each neutral field is written 4 to 6 times: dtype, encoder,
  decoder, viewer decoder (`viewers/data/_records.py`, written from the
  prose doc), `h5-schema.md`, and the golden file.

**Mechanism.** Consistency is kept by memory. An agent adds a name to the
tables it finds, and the ones it misses fail open.

**Remedy.** One registry per family of facts: `ElementTraits` and
`MaterialTraits`, `ResponseFamily`, `DeclarationKind`, and a schema
`ColumnSpec`. Today's tables are then *generated* from the registry.
1. Start each registry in shadow mode: derive the table and assert it
   equals the hand-written one.
2. Flip consumers over to the derived table.
3. Add completeness gates that fail closed.

### RS2: Lowering without intermediate representations (High, growing)

**Declarations → deck.**
- No IR type exists: there is no `Phase`, `Command` or `Block`.
- `BuiltModel.emit()` runs about 21 inline checks, then hands off to 7
  hand-written drivers (3,661 LOC) that call primitives and emitters
  imperatively.
- Phase order is encoded at least 7 times, with inconsistent step
  numbering. ADR 0043's phase buckets exist only in the doc.
- There are about 40 path-combination refusals; about 8 are essential.
- Per-rank and split decks are text surgery on the rendered buffer
  (`_write_per_rank_tcl`, `apesees.py:735`).
  - This is implemented twice, as post-hoc slicing and as stream routing
    (`tcl.py:1559-1680`), and kept byte-identical by hand.
  - `stream`+`split` is refused (`apesees.py:10931`), and Python decks have
    no `per_rank`.

**The archive re-derives what the driver already knew.**
- `H5Emitter` (4,161 LOC, 58 private methods) reclassifies node roles from
  the lowered stream (`h5.py:1148-1224`).
- It de-duplicates rank-replicated lines at 18 sites.
- It needs the declarative records pushed back in after emit.
- Five `setattr` side channels (`tag_resolution.py:63-102`) carry what the
  Protocol drops. Two of them have already written wrong `element_meta`.

**Replay is not the inverse of capture.** `OpenSeesModel.build` silently
skips MP constraints, contacts, ties, interfaces, regions and Rayleigh
damping (`compose.py:578-585`).

**Parity gaps between paths cause silent drops.**
- Partitioned decks dropped global Rayleigh damping, so np>1 runs were
  undamped (`1483d172`, #752).
- Staged equation ties were dropped (`00baa892`, #702).
- Staged BC validators were missing on the partitioned path (`ed36cd95`).
- A single ordering invariant (ADR 0099) needed 5 per-path hoists, about 12
  commits and 2 misses.

**Record intent.**
- There are 5+ surfaces for declaring what to record, with no shared intent
  IR.
- The two file-naming conventions are incompatible (`build.py:5247` vs
  `spec/_emit.py:534`).
- The canonical `ops.recorder.declare` path cannot be read back through
  `from_recorders`.

**Pictures.** There is no render-independent picture model, so Qt-free
consumers copy the math (RS1).

**The tests had to invent the IR.** The parity sweep re-parses Tcl/Py text
into `(command, token, *args)` tuples, and `RecordingEmitter` is used by
110 test files.

**Mechanism.** The driver knows each node's role, its owning stage, record
identity and fem_eid. Lowering to method calls discards all of it, and
every consumer has to re-infer it. Each cross-cutting concern must be
placed by hand in every path.

**Remedy.** Explicit IRs at the three busiest boundaries, each introduced
in shadow mode behind the byte-identity corpus:
- **`DeckProgram`**, an annotated tree of `Cmd`, `Block{rank|stage|module}`
  and `AnalyzeLoop` nodes. Tcl and Py become printers, live becomes an
  interpreter and H5 an archiver. Per-rank and split become tree
  transforms.
- **`RecordIntent`**, lowered to capture, file recorder, MPCO and Ladruno.
- **Picture bundles**, `compute() → scene_ir`, drawn by VTK, matplotlib and
  web alike.

### RS3: Seams fail open while components fail loud (High, growing)

**Components are fail-loud.**
- 161 `raise BridgeError` in `build.py` + `apesees.py`.
- 156 `__post_init__` validators on primitives.
- `MissingElementResults`.
- `_add_def` refuses defs it cannot dispatch.

**Seams are fail-open.**
- Capability lookups return `None`, and the gate skips.
- `getattr(bridge, "_primitives", ())` in `results/capture/spec.py:822`
  caused two long outages:
  - class hints were dead from 05-12 to 08-18 (fixed in `d04a7484`);
  - bridge-attached layered capture was dead until #1183 (09-27).
- 402 `except Exception: pass` across 97 viewer files.
- 14 swallow sites in loads and masses, outside the scope of the
  `resolve-swallow` lint.
- H5 no-ops, and replay skips.
- 11 fork build floors are "documented, not enforced". Five of them
  describe builds that silently do the wrong thing.
- OpenSees option parsers silently ignore unknown flags: the
  `-localCS`/`-drillingNT` cases (#1183) and `-crackedNu` (#1184).

**Fix-commit taxonomy** (about 70 fixes in 90 days):
- The largest class, about 21, is "silent drop or missing fail-loud".
- The next, about 12, is interactions between modes (partitioned × staged
  × contact × 2-D ordering), and these are mostly silent too.

**Mechanism.** At any single seam, fail-open is the locally safe choice
("don't crash on an unknown class"), so each author picks it. Across the
system, it turns omissions into wrong answers.

**Remedy.** A fail-closed doctrine at seams:
- completeness gates: every class is classified, and every kind is
  registered in every enumeration;
- loud replay;
- public introspection facades with contract tests, instead of `getattr`
  on private names;
- three oracles: deck round-trip, a FEMData round-trip property test, and a
  results cross-format conformance matrix.

### RS4: Boundaries exist only by convention, so the real dependency graph is hidden (High, growing)

**Lazy imports hide most edges.**
- 71% of cross-package runtime imports are function-level: 377 lazy
  against 152 eager. On 05-20 the share was 55%.
- 92% of those lazy imports neither break a cycle nor defer an optional
  extra. Their effect is to stay invisible to `test_import_dag_polarity.py`,
  which freezes eager edges only ("invisible to this eager tripwire by
  construction"). Commit `eb6055b3` states that motive outright.
- The eager file graph is a clean DAG. Counting lazy edges, 129 modules form
  one strongly connected component, and 13 of the 18 subpackages can all
  reach each other.

**Private names are the inter-package API.**
- 48% of cross-package import statements target `_`-private modules or
  names, up from 37%.
- 62% of test files import private modules or names.
- `opensees._internal.build` is imported by 85 test files.

**Several stated rules are false today.**
- **Charter P9:** the bridge imports from 10 other units, 70 imports in
  all, 38 of them into `_kernel`.
- **ADR 0015:** `_kernel` imports `core` lazily.
- **ADR 0014:** the guard skips relative imports, so
  `viewers/diagrams/_kind_catalog.py:30` imports `opensees._response_catalog`
  at module level undetected.
- **ADR 0026:** the `H5ModelReader` Protocol is indexed as "Accepted …
  shipped" but exists only in the ADR text (`0026:106`).
- **"results → opensees is one-way":** there are 48 statements into 13
  bridge modules, and 7 going back.

**The root `__init__` is eager and has side effects.**
- It prints a banner, sets an environment variable and reads
  `pyproject.toml`.
- Any `import apeGmsh.X` loads 250 of the 632 modules, including the whole
  bridge and gmsh.
- `studio/_mcp.py` promises "No gmsh, no Qt, no viewers"; at runtime that
  is false.

**The bridge reaches up into peripheral products.**
- `_run.py` → studio;
- `opensees_model.py` → assess;
- `apesees.py` → hpc;
- `section/computed.py` → sections;
- `_internal/compose.py` → cuts.

The peripherals, in turn, reach into internals: `sections/_mc.py` drives
`emitter.live._get_ops`.

**Mechanism.** There is no machine-readable layer map. Each ADR adds its
own guard with its own scope, and the guard that passes is the one you can
hide from.

**Remedy.**
1. **One declarative layer contract**, checked over module, lazy and
   `TYPE_CHECKING` imports with relative imports resolved. Seed it with
   today's violations as a shrinking exception list where every entry
   carries a reason.
2. **Per-package API modules** for the most-imported private targets:

   | Private target | Cross-package import sites |
   |---|---|
   | `core._compose_errors` | 48 |
   | `_kernel.records._constraints` | 32 |
   | `core._helpers` | 18 |
   | `opensees._response_catalog` | 17 |
   | `_kernel.records._kinds` | 14 |

3. **PEP 562 lazy `__init__`s.**
4. **A `BridgeIntrospection` facade.**
5. **Invert the bridge → peripheral edges with hooks**: a run-progress
   observer and a builder registry.

### RS5: Hub modules that absorb every concern, beyond the agent read window (High, growing)

**`apesees.py`: 14,181 lines, touched by 64 of the last 408 PRs.**
- `BuiltModel` spans 7,147 lines of emission orchestration.
- `apeSees` spans 4,028 lines. Declaration verbs are about 819 of them, run
  procedures about 1,899 and deck front doors about 877.
- `_StageBuilder` spans 1,810 lines.
- Mapping 72 commits' diff hunks onto the AST shows where the churn is:

  | Region | Lines changed | Commits |
  |---|---|---|
  | Partitioned emit | 1,785 | 33 |
  | Emit drivers | 876 | 33 |
  | Run procedures | 1,502 | 9 |
  | Declaration verbs | 89 | 3 |

- The import block changed in 37 of those 72 commits.

**The other hubs.**

| File | Size | Notes |
|---|---|---|
| `build.py` | 10,308 lines | 149 top-level functions, 18 record types, 25 validators; 51 PRs |
| `results_viewer.py` | 4,941 lines | a retired design; `_show_impl` is 1,926 lines |
| `ModelViewer.show` | 1,792 lines | one method with 65 closures |
| `ConstraintsComposite.py` | 3,746 lines | 22 verbs, 5 def lists |
| `h5.py` | 4,161 lines | `H5Emitter` has 133 methods |

**Size trend.** Files over 2,000 lines went from 4 (05-20) to 12, then 15.
They now hold 22% of the LOC. Reading `apesees.py` costs about 167k tokens
(8 Read calls).

**Merge witness `7d91359e`.** Two parallel branches implemented the same
node-coordinate invariant with opposite architectures, and the merge left
"one invariant with two enforcement points".

**Mechanism.** Four unrelated reasons to change share one module. Parallel
agents collide in it, and no agent ever sees the whole unit it edits.

**Remedy.** Split by reason to change, behind re-export facades, one hub at
a time during a freeze window, proven by the byte-identity corpus:
1. `_StageBuilder` → `_internal/stage.py`.
2. Procedures → `opensees/procedures/`.
3. The deck front doors.
4. `BuiltModel` → `_internal/emit/`.

Add a file-size ceiling ratchet alongside.

### RS6: The evolution policy taxes growth (High, growing)

**Schema versioning makes archives expire.**
- `validate_zone_version` accepts only `{reader.minor-1, reader.minor}`
  (`schema_version.py:290-306`).
- The neutral zone went from 2.5 to 2.33.0 in 30 bumps since 05-12, about
  one minor every 4 days. The measured shelf life of a file is 1 to 4
  weeks.
- `results.h5` files expire through their embedded `/model` zone, even
  though the results zone itself has sat at 1.1.0 since 05-21.
- ADR 0023 INV-5 promised a migrator "before any zone reaches a third minor
  cycle". 28 cycles later it does not exist.
- About 16 of the 22 minors from 2.12 to 2.33 persisted Ladruno options.
- One global counter collides across worktrees: `6192c659` "bump after
  rebase", and main went red at least 4 times.
- Across `1df2d0dc`, the golden-file diff changed only the version stamps,
  yet an external reader still had to redeploy.

**Retiring a design does not delete it.**
- ADR 0098 S6 retired the legacy results window by "de-publication, not
  deletion".
- About 44 modules (about 23.9k LOC) are reachable only through two routes
  (Errata E5):
  - `export_animation` (which studio's MCP `animate` uses), and `show_web`
    / `serve_web`;
  - the director that `render` / `render_pack` still build, which `assess()`
    uses.
- Three reconcile paths coexist, so agents' stills are drawn by a different
  orchestrator than the human window.

**Vestiges survive.**
- `_populate_node_ndf` (about 150 LOC) reads an attribute that was deleted.
  Its data path (`/nodes/ndf`, hash folding, compose threading) is still
  alive.
- Staged live execution has been refused since May, at 10 sites.

**Shared index files collide.**

| File | Size | PRs touching it |
|---|---|---|
| CHANGELOG | 845 KB; 434 sections and 8,949 lines under Unreleased, although v2.0.0 was tagged 08-05 | 60% of all PRs (77% in September) |
| ADR index | 91 KB, rows up to 6.7k characters | 53 |
| `_api_index.json` | 248 KB, with a `"generated"` timestamp line | 37 |
| skill mirror | — | 68 |

72% of PRs touch at least one index or generated file.

**Remedy.**
- A compatibility floor per zone: accept `[floor, current]`, and keep
  INV-4.
- No bump for self-describing additive columns.
- A migrator implemented as `from_h5(floor) → to_h5`, with a golden corpus
  holding one file per past minor.
- A retirement ledger with "finish as deletion" exit criteria.
- Changelog fragments.
- Deterministic, generated index files.

### RS7: The variant axis has no model, no CI lane, and leaks into the neutral core (High, growing)

**Size.**
- About 45 fork-only primitives: about 25% of primitives by count and
  about 35% of primitive class-body lines.
- That is 76% of `element/solid.py` and 81% of `analysis/integrator.py`.
- 183 of 363 PR-level code commits in 90 days touch fork vocabulary, as do
  59 of the 78 code PRs since 09-01.

**Verification.**
- No CI lane executes the fork (`tests.yml:314-317`).
- The 84 `integration_ladruno` tests, and the conditional tests in 11 other
  files, never run in CI.
- The local fork build dates from 06-25.
- ADRs record verification in prose:
  - "7/7 green against 3622d6214";
  - ADR 0108 is Accepted with five assertions never run to green;
  - ADR 0110 says "CI can never exercise".

**Detection.**
- There are at least 12 separate mechanisms.
- Production code uses two different signals for "is this the fork?":
  - `hasattr(ops, "profiler")` at `_target.py:206`;
  - `hasattr(ops, "criticalTimeStep")` at `live.py:411`.
- Rosters exist only for elements and integrators.
- None of the 11 build floors is enforced.
- `OpenSeesTarget` carries no build, and decks carry no build assertion.

**Leakage into the neutral layers.**
- `_kernel/_coupling_control.py:195` emits OpenSees argv.
- `ContactDef` has 33 fields and `ContactRecord` 35, persisted as a
  46-column dtype of fork flags.
- Verbs that look engine-neutral lower only to fork elements:
  `kinematic_coupling` is "BREAKING, replaces the equalDOF expansion".
- Fork facts are copied, and the copies drift. #1184 (`07f757e0`) emits
  `-crackedNu`/`-betaC` and assumes a fork default of 200 for
  `tens_stiff_c`. Both come from fork PR #873, which was closed without
  merging. The `ladruno` branch still defaults to 500 and silently ignores
  the flags (Errata E4).
- The Protocol grew by 8 fork-only commands, each implemented in 6
  emitters.
- Five direct `openseespy` imports bypass the fork-first resolver (§3 D6).

**What is essential.** The two engines differ in results, not only in
commands. Stock `TenNodeTetrahedron` is 6× too soft, stock LAPACK reports a
zero pivot as success, and the DruckerPrager tension cutoff and ASDPlastic
apex are wrong on stock. The fork also has silent paths, because its parsers
ignore unknown tokens (E4). The "71% soft ties" figure is very likely the
tet10 defect in series (E3), and the strut-and-tie gap is 0.79%, not
20–30% (E2).

**Remedy.**
- One `BackendInfo`, owned by the resolver: a single signal plus the build
  stamp.
- A declarative `requires=Fork(feature, since)` on every primitive family.
  Generate from it the live gates, the capability docs map, the test
  markers and an opt-in emit-time `target=` check.
- One table of build floors. Ask the fork to export a feature manifest so
  the floors become enforceable.
- A committed `FORK_PIN`, skip-with-reason on a build mismatch, and a
  scheduled fork lane.
- Accept fork ADRs only with a linked green run.
- A `solver_params` lane that keeps fork options out of neutral defs,
  records and dtypes.
- `emitter.command()` for fork-only verbs.
- `None` means "use the build's default".

### RS8: Normative state is narrated, not derived (High for an agent-built repo, growing)

**The charter.**
- It has had one commit (`ae550e4f`, 2026-05-10) through 108 ADRs.
- Its mission ("writes the deck and stops") and non-goals ("no
  model-of-the-solver-loop", "no automated convergence rescue") are
  contradicted by what shipped:
  - ADR 0057 (the solution-strategy Ladder);
  - ADR 0104 (the Substep controller);
  - per-step hooks, `run_remote`, and footfall FRF analysis.
- P8 and P9 are false (RS2, RS4).

**The architecture docs.**
- `layout.md` lists:
  - `_internal/registry.py` (deleted in `9c077830`);
  - `recipes/` (never built);
  - 5 classes that do not exist.

  It omits 37 of the 86 modules.
- `emitter.md` omits 22 of the 74 Protocol methods and calls the Protocol
  frozen.
- The architecture README says "81 primitives, schema 2.2.0".

**The ADRs.**
- The index is 91 KB, and 7 status lines disagree with the ADR files.
- 12 of the 20 ADRs whose files say "Proposed" have shipped.
- 39 ADRs were edited 3 or more times despite the append-only rule.
  0095 has 19 commits and 1,614 lines; 0098 has 17 and 2,124.
- At least 8 renumbers left stale citations. `RebarComposite.py:2` cites
  0066, which is now H5DRM; the cage ADR is 0067.
- 111 unqualified "ADR NN" citations refer to the fork's ADRs.
- ADR 0102 (freeze new fork wrappers) is Proposed, and four new wrappers
  followed it.
- ADR 0098 A7 is cited by commits but does not exist.

**User docs.** `docs/how-to/choose-results-strategy.md` is built around
`spec.capture(...)` and `ops.tcl(..., recorders=spec)`, and neither exists
(§3 D7). `mkdocs --strict` cannot detect this.

**Registers.**
- `_DEFERRED.md` has not changed since 08-10, and 3 of its entries are
  resolved but not marked.
- 68 ADRs contain 304 deferral mentions; 5 of them link to the register.
- Private memory holds 23 open items, a roadmap that Codex, Cursor and the
  second machine cannot see. It also contradicted the code, until this
  session corrected it: ADR 0052 and 0080 statuses, and the ADR 0026
  Protocol.

**Mechanism.**
- AGENTS.md tells agents to judge new code against the charter, and routes
  them to May-era docs.
- Each ADR becomes local law.
- "Accepted" is disconnected from verification.
- Every feature pays an archaeology cost before it starts.

**Remedy.**
- **Charter v2** (a human decision) naming the run layer, the capability
  model and the real bridge input contract (FEMData + `_kernel.records`).
- **ADR frontmatter** (`status`, `implemented_by`, `verified_by`,
  `supersedes`) feeding a generated index with short rows, plus a
  status-agreement lint.
- **Historical scope docs** moved to `architecture/history/`.
- **A generated `layout.md`.**
- **Doc-API drift checks:** every code identifier in a docs code block must
  resolve.
- **A single in-repo slice ledger** replacing private roadmaps.

---

## 3. Confirmed defects found during the audit

These are consequences of the strains, not strains themselves. Each is a
silent wrong answer, or a false guarantee, in shipping code.

Verification key:
- **R**: re-run by the synthesizer against this worktree.
- **P**: an auditor's live probe output, reviewed.
- **C**: confirmed by reading the code, and the OpenSees C++ where
  relevant.

**Model and results defects**

| # | Defect | Effect | Location | Verified |
|---|---|---|---|---|
| D1 | `g.masses.surface` on quad9 (gmsh etype 10) creates no mass, and `pressure(reduction="tributary")` creates no load. | 0.000 kg instead of the expected 200.000, with no warning. quad4 and quad8 are correct. | The inline etype tables in `MassesComposite._target_faces` (`:734`) and `LoadsComposite._target_faces` (`:842`) omit 10. | R |
| D2 | The FEM snapshot cache is not invalidated by `g.constraints.contact`, `contact_plane` or `interface`, nor by `g.reinforce` or `g.embed`. | `get_fem_data()` returns the stale object with 0 contact planes while the composite holds 1 def. `get_fem_data(dim=3)` returns 1. | There are 7 `_bump_fem_counter` sites, and these verbs have none. `tests/test_fem_cache_invalidation.py` covers only constraints, loads and masses. | R |
| D3 | `g.constraints.bc` and `g.masses.*` never reach the deck unless restated with `ops.fix` / `ops.mass_from_model()`. | 174 SP and 339 mass records produce 0 `fix` and 0 `mass` lines, with no warning. The `bc()` docstring (`ConstraintsComposite.py:1769`) says the nodes "get `ops.fix` downstream". | `_emit_fixes` iterates only `ops.fix` records (`apesees.py:5703-5712`). ADR 0051 §4's reconciliation warning never shipped. | P |
| D4 | `FEMData.select(target=[(dim, tag)])` queries whichever gmsh model is live. | With model B live, model A's selection returned 48 nodes, only 10 of them correct, and raised no error. | `FEMData.py:579-594` | P |
| D5 | `GeomTransfViewer` (Three.js) draws local y and z negated relative to OpenSees, and its default vecxz `[1,0,0]` ignores the Python `default_vecxz` rule. | For a beam along X with vecxz = Z, it shows (−Y, −Z); OpenSees uses (+Y, +Z). The local z points away from the vecxz the user typed. | `viewers/geom_transf_viewer.py:410` computes `ey = cross(ex, vxz)`. OpenSees `LinearCrdTransf3d::getLocalAxes` computes y = v × x. The canonical Python version is `viewers/diagrams/_beam_geometry.py:35,48`. | C |
| D6 | Five direct `import openseespy.opensees` sites bypass the fork-first resolver. | On a machine with the fork, capture and recorders can bind to a different module and domain than the one the model was built in. | `results/capture/_domain.py:1101`, `results/live/_mpco.py:102`, `results/live/_recorders.py:157`, `interop/solve.py:99,141`, instead of `emitter/live.py:285 _resolve_ops`. | C |
| D7 | A published how-to prescribes an API that does not exist. | `spec.capture(...)` and `ops.tcl(..., recorders=spec)` do not exist; `apeSees.tcl`/`py` take no `recorders=`. | `docs/how-to/choose-results-strategy.md:23-24` | C |
| D8 | `OpenSeesModel.build(...)` replay silently skips MP constraints, contacts, embed and equation ties, interfaces, regions and Rayleigh damping. | A "round-tripped" deck describes a different physical model. | `opensees/_internal/compose.py:578-585` ("SCOPED / best-effort") | C |
| D9 | Compose carries the host's `rebar_elements` but drops the source module's. This is a documented deferral with no runtime warning. | Composed modules arrive without their auto-emitted rebar. | `mesh/_compose.py:2714-2719` | C |

**Guard and CI defects**

**D10: guards and gates believed enforced are not.** Verified by reading the
code (C). In each case the next instance of the hazard ships green.
- The ADR 0014 guard skips relative imports, so `_kind_catalog.py:30`
  escapes it.
- A dangling `TYPE_CHECKING` import, `from ...parts.plane_wave_box`,
  resolves to the nonexistent `apeGmsh.opensees.parts`. It sits inside the
  mypy-gated package.
- Outside the gate there are 3 more dangling `TYPE_CHECKING` imports and 3
  F821 undefined names.
- Two non-literal `Group.get` calls remain at `h5_reader.py:1796,1855` (the
  hazard class #1181 fixed).
- The ns-wrapper parity test compares nothing for `LadrunoRCConcrete` or
  `LadrunoRCFiniteStrain`, whose wrappers delegate through `from_fc`.

**D11: CI can cancel the only test of a merged combination.** Verified by
reading the code (C).
- `.github/workflows/tests.yml:11-13` sets
  `concurrency: tests-${{ github.ref }}` with `cancel-in-progress: true`,
  which also cancels runs on **main**.
- Branch protection is `strict: false` (checked with
  `gh api …/required_status_checks`).
- So back-to-back merges can cancel the only CI run of the merged
  combination. 10 of 80 main runs were cancelled.

---

## 4. What already works: the remedies exist in the repo

Preserve these patterns and generalize them. Most of the program in §6 is
"do this everywhere".

| Pattern | Where it already works | Generalize it to |
|---|---|---|
| Declarative kind registry | Diagram kinds (ADR 0058 S0). #784 needed 3 viewer files; #614 removed 562 duplicated lines. | `ElementTraits`, `ResponseFamily`, `DeclarationKind` |
| Per-record metadata walked generically | The `tag_rewrite_spec` ClassVar, consumed by compose (`_compose.py:1050-1078`) | H5 codec, compose carry and hash policy per record kind |
| Facts derived from one table | LadrunoUP shape tables feeding `_up_slot_floors` | All element capability tables |
| Self-describing storage with path constants | The results zone, unchanged at 1.1.0 for 4 months | The neutral and opensees zones, through `ColumnSpec` |
| Narrow protocol plus frozen value types | `ResultsReader` + Slabs (no consumer branches on reader class). The Emitter as primitives see it: 182 `_emit` methods, no emitter sniffing. | Capability sub-protocols; a frozen Emitter with `command()` |
| Qt-free IR with headless tests | `ResultsSession` (175 headless tests, measured fast paths). The `SceneLayer` IR with Recording/Ledger backends. | Picture-compute bundles; `DeckProgram` |
| IR seed | `RecordingEmitter` (110 test files); the parity-sweep tuple parser | `DeckProgram` in shadow mode |
| Plan before emit | `_plan_partitioned_contacts` ("a refused model emits nothing"); ADR 0100's columnar ownership | An ownership-scoped phase table with tag planning |
| Claim-by-name | Stage claims; reinforce references materials by name | A `solver_params` lane |
| Sink override for twin verbs | `_StageDampingNS` overriding `_DampingNS` | One `DeclarationPool` for flat and stage verbs |
| Lesson → mechanical rule | `check_quirks.py` rules, each proven against the commit that had the bug; mutation-tested viewer gates; `test_chain_phase_guard_coverage.py` | Completeness gates for every enumeration |
| Healthy leaves | The eager file graph is a DAG. `_kernel`'s `__init__` closure is 1 module. `fem`, `hpc` and `cuts` are leaves. P2 and P3 hold. | The target shape for the layer contract |
| Emit anywhere, gate only the run | `live.py:154-157`; `get_backend_build()` | `BackendInfo` plus target profiles |
| Slice template | `internal_docs/handoff_adr0109_program.md` | The slice card (§7.3) |

---

## 5. Design principles for the next phase (proposed Charter v2 material)

1. **Local-change safety.** Adding an element, response, declaration kind,
   command, persisted field or diagram is one registration. Every
   enumeration derives from it, and a completeness gate fails closed when
   something does not participate.
2. **Fail closed at seams.** An unknown class, a missing capability, a
   skipped stream or an unconsumed record raises, or warns loudly.
   `getattr(x, "_private", default)` across a package boundary is a lint
   error.
3. **Semantics survive lowering.** Each lowering boundary has an explicit
   IR. Downstream consumers read its annotations instead of re-inferring
   them.
4. **Boundaries are machine-checked over every kind of import.** There is
   one layer contract and there are per-package API modules. No
   cross-package `_private` import exists outside a ratcheted exception
   list.
5. **Variants are data.** One capability model, one table of build floors,
   fork options in a solver-params lane, and a pinned fork lane in CI.
6. **Compatibility is a floor, not a window.** Additive changes do not bump
   the version. A migration is `from_h5(floor) → to_h5`.
7. **State is derived, not narrated.** Status, indexes, layout docs and
   rosters are generated from code or frontmatter and drift-tested.
8. **Retire by deletion.** A retirement gets a ledger entry and an exit
   criterion; de-publication is a stage, not an end state.

---

## 6. Remediation program

The phases are ordered so that each one makes the next safe for parallel
agents:
1. stop silent damage;
2. make drift and boundaries visible;
3. split the hubs;
4. introduce extension points.

Sizes: S is at most one PR-day, M is a few PRs, and L is a program of
several slices.

### Phase 0: stop silent wrong answers (week 1; in parallel, on disjoint files)

Each item is one small PR that ships a regression test shown to fail with
the fix reverted.
- **D1:** use one gmsh element-type table, from `mesh/_element_types`, for
  load and mass face targeting.
- **D2:** add a `_declare(defn)` base (append, route, bump) used by all 12
  entry points. Add an AST gate modelled on
  `test_chain_phase_guard_coverage.py`.
- **D3:** at `apeSees.build()`, warn on unconsumed homogeneous SP and mass
  records (ADR 0051 §4), and correct the `bc()` docstring. This touches
  `apesees.py`, so it must be sequenced with other `apesees.py` PRs.
- **D4:** a raw-DimTag `select` refuses unless its producing session is the
  live one.
- **D5:** match OpenSees (y = v × x, z = x × y) and the Python
  `default_vecxz`. Ideally, pass frames computed in Python to the JS.
- **D6:** route the five sites through `emitter.live._get_ops` /
  `_resolve_ops`.
- **D7:** rewrite the how-to table against the real API: `emit_recorders`,
  `emit_mpco`, capture.
- **D8/D9:** `_replay_into` and compose warn whenever they skip a stream.
- **D10:**
  - fix the ADR 0014 guard (resolve relative imports);
  - delete the dangling `TYPE_CHECKING` imports and fix the F821s;
  - fix the two `.get` calls;
  - extend the ns-parity test to every init field and to `from_fc`
    delegates;
  - add a fail-closed lock test that every `Element` and material subclass
    is classified.

### Phase 1: remove the per-PR tax (weeks 1–2)

See §7.4, lane A:
- stop cancelling CI runs on main;
- a deterministic `_api_index.json`;
- changelog fragments;
- ADR status in one place, with a generated index;
- `land_pr.py`;
- an orphan detector;
- `preflight.py`.

### Phase 2: make boundaries and drift visible (weeks 2–4)

**Layering.** One layer contract over all imports, first in report mode,
then enforcing with a shrinking exception list where every entry has a
reason. Retire the 10 bespoke import guards into it.

**Lint and types.**
- ruff default rules on all of `src`, with a per-directory baseline.
- `ignore_missing_imports` scoped to third-party modules.
- mypy error-count ratchets per package: viewers, results, mesh, core.
- A file-size ceiling ratchet: a file over 2,000 lines may not grow by more
  than 200.

**Oracles.**
- **Deck round-trip:** `tcl() == OpenSeesModel.from_h5(h5()).build('tcl')`
  modulo tags, with an xfail list for known gaps.
- **FEMData field round-trip:** a property test across `to_h5`, `from_h5`
  and compose.
- **Results cross-format conformance matrix:** capture vs MPCO vs
  `.ladruno`, with today's differences marked xfail.
- **Emit mode matrix:** {serial, partitioned} × {flat, staged} × {2-D, 3-D}.

**Table consistency.** Tests for the duplicated facts (σzz sets,
material-only tokens, category sets, DOF maps). They pin today's tables
before consolidation starts.

**Schema policy.** In order:
1. a golden corpus with one file per past minor;
2. a compatibility floor per zone;
3. the additive-column rule;
4. the migrator.

**Normative docs.**
- A Charter v2 ADR (a human decision).
- Historical docs moved to `architecture/history/`.
- `layout.md` and the README status block generated from code.
- ADR frontmatter migration.

**Backend capability model, step 1.**
- `has_fork` uses one signal.
- Rosters plus drift tests for materials, analysis components and
  recorders.
- A `FORK_PIN`, with skip-with-reason on a build mismatch.
- A non-required nightly fork lane on a self-hosted runner.

### Phase 3: split the hubs (weeks 3–6)

One file at a time, in a freeze window, each split proven byte-identical.

**`apesees.py`**, in order:
1. `_StageBuilder` → `_internal/stage.py`.
2. Procedures (modal, footfall, FRF, explicit, contact queries) →
   `opensees/procedures/`, behind delegating methods.
3. The deck front doors.
4. `BuiltModel` → `_internal/emit/`.

**`build.py`**:
- Record types move to one module, merged with their `typed_records`
  twins.
- Validators become a checks registry fed a `ValidationContext` with an
  explicit purpose (solve vs archive). This replaces the `H5Emitter` name
  check.
- Helpers are grouped by concern.

**Package boundaries.**
- Per-package API modules for the top 5 private targets: about 130
  statements, one mechanical PR each.
- PEP 562 lazy `__init__` for the root, `opensees` and `results`.
- The banner moves off import.
- The `_response_catalog` lookup moves to a leaf module.

**Finish ADR 0098 S6 as a deletion**, in order:
1. Session-backed `results.render` tokens.
2. `export_animation` as a restep loop.
3. `show_web` as a session client.
4. Delete `results_viewer.py` and the modules only it uses (about 23.8k
   LOC).

**`ModelViewer.show`** becomes command objects built on the
`SessionWindow` template.

### Phase 4: extension points (weeks 6–16)

Use the strangler pattern, and start each item in shadow mode.

**Emission**, in this order:
1. `EmitOptions`/`TargetCaps` replace the 6 control attributes and the name
   checks.
2. Freeze the Protocol, and add `command(name, *args)` with a verb table
   (H5 policy, return shape, `requires=`).
3. Merge the two staged drivers over a `RankScope`; they already share 27
   of 35 helpers.
4. An ownership-scoped phase table, with tags planned before emission.
5. The `DeckProgram` IR in shadow mode: a shadow Tcl printer that is
   byte-identical across the corpus.
6. Migrate the consumers:
   - **H5:** delete the side channels and the de-duplication.
   - **per-rank and split:** tree transforms. This unblocks ADR 0099 INV-4,
     `stream`+`split`, and Python `per_rank`.
   - **live:** an interpreter. This unblocks staged live.

**Facts.**
- `ElementTraits`/`MaterialTraits`, with `requires=Fork(...)`.
- Derive the existing tables from them.
- Generate the live gates, the capability docs and the test markers.

**Declarations.**
- A `DeclarationKind` registry. Each kind registers:
  - its def types and store, and a `resolve(ctx)`;
  - its broker slot, H5 codec and compose carry;
  - its chain route and emit phase;
  - its clear/summary participation.
- One pure `resolve(defs, source)` over `ResolverSource`.
- A single target resolver that raises an explicit ambiguity error.
- Solver params move out of defs and records (Option B, generalized), and
  `emit_flags` moves to the bridge.
- Delete the vestigial ndf channel.

**Results.**
- A `ResponseFamily` registry that generates the per-route tables.
- `RecordIntent`, with one naming function and a working declare →
  `from_recorders` read-back.
- A `TranslatingReader` decorator.
- Capability sub-protocols (`EnergyReader`, `EnvelopeReader`,
  `LocalAxesReader`, `LineageReader`) replacing the 11 `getattr` probes.
- A `Capturer` protocol.
- A `BridgeIntrospection` facade.

**Persistence.**
- An `_h5` leaf package: child probes, decoders, attribute codecs,
  `SchemaVersion`, atomic write.
- A declarative `ColumnSpec` registry that generates:
  - dtypes and name-keyed codecs;
  - doc tables, the golden file and the viewer decoders;
  - the compose carry list and the hash policy.
- A `solver_params` lane.
- One `ModelArchive` handle shared by `OpenSeesModel`, `NativeReader`, the
  viewer and compose.

**Pictures.** A Qt/VTK-free `compute() → scene_ir` per kind. VTK kinds
become emitters, and matplotlib and web draw the same bundles.

### What not to do

- **No big-bang rewrite** of `apesees.py` or `build.py`. Strangle them
  behind the byte-identity corpus and live results parity.
- **No silent exceptions.** Do not freeze today's violations as "allowed"
  without a recorded reason and an owner.
- **Moratoria:**
  - no new Protocol methods for fork-only commands (use `command()` once it
    exists; until then, justify each in its PR);
  - no new declaration kind outside `_declare()` and the side-list tuple;
  - no neutral-schema bump for an additive column once the floor policy
    lands.
- **Don't let the fork surface grow faster than the fork lane can verify
  it.** ADR 0102 D5 needs a decision, not silence.

---

## 7. Agentic development: the operating model

### 7.1 Metrics snapshot

The window is the 90 days to 09-27, unless noted.

| Metric | Value |
|---|---|
| PRs per month | Apr 44 · May 452 · Jun 263 · Jul 104 · Aug 212 · Sep 92 |
| PR size (last 400) | median 6 files / 449 lines; p90 20 / 1,962; 26% over 1,000 lines |
| Open → merge | median 12 min overall; 24 min in September |
| Reviews on the last 100 PRs | 0 |
| Fix-titled PRs | 79/408 (19%); Jul 9% → Aug–Sep 23% |
| Fix PRs repairing files a feature PR touched ≤7 days earlier | ~~46/79 (58%)~~ **withdrawn:** it fails its control, since 78% of non-fix PRs also touch recently changed files (E1) |
| Work that never reached main | 12 cases, including #858 → #1169 (62 days) |
| Lesson written → mechanical rule | about 4 months: six rules landed 09-25/26 for lessons from 05-16 … 06-20 |
| PRs touching an index or generated file | 72% over all PRs; 80–85% over squash-era PRs (E6) |
| CI wall clock | about 8 min (suite 7.8, live-stock 3.3, static 1.4, emit-cost 1.3, lock 1.1); not the bottleneck |
| Main runs cancelled | 10 of 80 |
| Context loaded before work | always ~8.6k tokens (AGENTS.md 3.2k + MEMORY.md 4.9k + global 0.5k); onboarding "universal context" ~27k; `apesees.py` ~167k |
| Static-gate coverage | 26% of src LOC |
| Local worktrees | 94, of which 90 are on branches deleted on the remote |

### 7.2 Workflow strains

**W1: Index and generated files tax every PR (High).**
- Shared insert points: the CHANGELOG anchor and the ADR index.
- Generated files that are not deterministic: `_api_index.json` carries a
  timestamp.
- Status stored twice: in the ADR file and in the index.
- Result: seven union-merge repairs in three days (09-25 → 09-27).

**W2: Landing a PR is prose, and lessons take about 4 months to become checks (High).**
- 12 cases of lost work.
- Stacked-base mistakes recurred after the lesson was written (#858, #964,
  #1057), and so did push-after-merge (#387, #489, #508, #567, #1097).
- Nothing detects a push after merge.
- `strict: false` plus cancelled main runs leave merged combinations
  untested.

**W3: Merge, then fix (High).**
- Review is optional and happens after merge. Adversarial probes ran
  post-merge: #784 → #785, ADR 0080 B1–B3 → #844, and the ADR 0093 probe
  fixes.
- The two largest failure classes, silent drops and mode interactions,
  have no general test.

**W4: Code that no lane runs (High).**
- The fork: 84 integration tests plus the conditional ones.
- The 16 `subprocess` test files.
- Windows, which is the development platform.
- Python 3.10: `requires-python >= 3.10`, but CI runs 3.11 only.
- Latest dependencies: a pyvista/VTK release turned main red on 09-08.
- The nightly benchmark stayed red for 22 nights.

**W5: Context is expensive, stale, and split between the repo and private memory (Medium).**
- MEMORY.md has 102 entries:
  - about 20 duplicate AGENTS.md or the guides;
  - 35 describe work already shipped or closed;
  - 23 are open items, which makes them a private roadmap;
  - several contradicted the code.
- The onboarding docs have not changed since 05-12. They say nothing about
  `ns/`, the API index or the skill mirror.
- Guides drift within 48 hours, for example "until #1170 merges".
- "Never read apesees.py whole" is written in only one handoff.
- A stale third apegmsh skill is still visible to agents.

**W6: No single view of the work, and ADRs don't close (Medium).**
- 20 ADRs say "Proposed"; 19 are older than 30 days, and at least 9 have
  shipped.
- 46 plan files, 27 with no status line.
- Duplicated work: two agent-surface ports ran on 09-25.
- ADR numbers collided 3 times.

**W7: Hub files defeat "one agent per file" (Medium).** Rule 1 of
`parallel-execution.md` says one agent per file, yet `apesees.py` was
touched by 64 PRs and `build.py` by 51.

**W8: Local verification is noisy (Low–Medium).**
- Local runs have to be diffed against an origin/main baseline.
- The editable install points at another checkout.
- There are 94 worktrees.

### 7.3 The operating model

**Principle.** Agents are strong at local, well-specified edits and weak at
global enumeration and at remembering prose. So invest in four things:
1. architecture that makes changes local (§6, Phase 4);
2. mechanical checks instead of prose;
3. derived state instead of narrated state;
4. small, cheap, accurate context.

**1. Slices.**
- **Slice card**, derived from `handoff_adr0109_program.md`:
  - the files the slice owns;
  - its verification command and stop condition;
  - model and effort;
  - token rules: grep, then read ranges; never read a file over 2,000 lines
    whole.
- **Size:** at most about 10 files and about 800 src+test lines.
- **Ledger:** slices are listed in an in-repo ledger (a GitHub issue plus an
  `in-flight` label).
- **Parallelism:**
  - Slices run in parallel only when their owned files are disjoint.
  - Hub files take a label lock and land one at a time until they are
    split: `apesees.py`, `build.py`, `_response_catalog.py`,
    `ConstraintsComposite.py`, `_femdata_h5_io.py`.

**2. Landing is a script.** `scripts/land_pr.py` checks, in order:
1. The base is main, the PR is open, and `headRefOid` equals the local tip.
2. The required checks passed on that commit, and a changelog fragment is
   present.
3. Re-sync the PR if main has since changed its files or a registered
   shared literal.
4. Squash-merge.
5. Check the `compare` status, and watch the main run.

AGENTS.md "How work lands" shrinks to "run land_pr.py".

Settings:
- Stop cancelling main runs.
- A ruleset that requires `lock-tests` on all branches.
- A nightly detector for orphans and red scheduled lanes, which opens
  issues.

**3. Verification loop.**
- **Preflight:** `scripts/preflight.py` mirrors CI's test selection, adds
  boundary tests for the changed files, and diffs against an origin/main
  baseline.
- **Fix PRs** show their regression test failing with the fix reverted.
- **Pre-merge review:** PRs on risky paths get a read-only adversarial
  review agent **before** merge. The verdict goes in the PR body, and
  `land_pr` checks for it. Risky paths are:
  - `results/readers/**`;
  - `mesh/_compose.py` and `mesh/_femdata_h5_io.py`;
  - partitioned or staged emit;
  - anything over 800 src lines.
- **Lesson SLA:** the second occurrence of a failure class becomes a rule or
  a ratchet within 7 days.

**4. Fitness functions.** All are ratchets, never raised silently.
- File-size ceilings.
- ruff on all of src, and mypy error counts per package.
- One layer contract over all imports.
- No cross-package `getattr` on private names.
- **Completeness gates:** every `Primitive` subclass appears in its `ALL_*`
  list and is classified by traits; every declaration kind appears in every
  enumeration.
- Round-trip and mode-matrix oracles.
- **Doc drift:**
  - ADR index agrees with each file's status;
  - `layout.md` is generated;
  - code identifiers in docs resolve;
  - "until #N merges" lines expire;
  - no developer home paths in src or tests.

**5. Context hygiene.**
- **MEMORY.md** cut to about 5 KB: delete duplicates of AGENTS.md, archive
  the shipped history, and move open items into the ledger so every agent
  and machine sees them.
- **Onboarding:** slice cards replace the ~27k-token "universal context".
- **Code map:** a generated symbol → line-range map for files over 2,000
  lines, until they are split.
- **Large ADRs:** a decision summary of at most 60 lines at the top of any
  ADR over 40 KB, with amendments in sibling files.
- **Housekeeping:** remove the stale skill copy, and prune dead worktrees.

**6. Effort allocation.**

| Kind of work | Model and effort |
|---|---|
| Mechanical splits, table derivations, ratchet PRs | a mid-tier model, with strict parity gates |
| IR design, schema policy, the capability model, Charter v2 drafts | the top model at high effort, then human review |
| Adversarial pre-merge review | the top model, read-only, on risky paths only |
| Auditing and re-measurement | parallel read-only agents, as in this assessment |

**7. Human judgement stays human.**
- Charter v2, and ADR acceptance: a weekly 15-minute triage, whose first
  batch is the 12 ADRs that shipped while still "Proposed".
- The split designs for `apesees.py` and `build.py`.
- Which lessons become rules, and any baseline increase.
- Releases; the first drains the 434-section Unreleased block.
- Fork adoption priorities, against the ADR 0102 freeze.
- Visual acceptance of viewer changes.

### 7.4 First two weeks, as PR-sized steps in three lanes

**Lane A: workflow infrastructure** (mostly serial; touches CI, scripts and
docs)

1. **`ci:`** main runs are no longer cancelled
   (`cancel-in-progress: ${{ github.event_name == 'pull_request' }}`). A
   nightly failure opens an issue. Add a non-required Python 3.10 smoke
   job.
   *Done when* two back-to-back main pushes both complete.
2. **`studio:`** a deterministic `_api_index.json`: drop the timestamp and
   fix the key order.
   *Done when* two regenerations are byte-identical.
3. **`changelog:`** fragments: `changelog.d/<slug>.md` plus
   `scripts/changelog.py --check|--assemble`, and update the guides.
   *Done when* three simulated concurrent PRs merge with no repairs.
4. **`adr:`** status lives in each ADR file's frontmatter.
   `scripts/adr_index.py` generates the README with short rows, and its
   `--check` runs in the quirk lint.
   *Done when* the 7 mismatches are gone and the README is under 20 KB.
5. **`scripts/land_pr.py`** plus the ruleset.
   *Done when* replaying #858, #1057 and #1097 flags each one.
6. **Nightly orphan detector.**
   *Done when* it flags the known orphaned branch.
7. **`scripts/preflight.py`.**
   *Done when* a clean origin/main tree reports 0 new failures.

**Lane B: silent-defect fixes** (fully parallel, on disjoint files)
- D1, D2, D4, D5, D6, D7, and the D8/D9 warnings.
- D3 comes after any in-flight `apesees.py` PR.

**Lane C: guards** (after lane B's files settle)
- The D10 set.
- The fail-closed capability lock.
- The ns-parity extension.
- The layer contract in report mode.
- A ruff baseline over all of src.
- The file-size ceiling.
- The FEMData round-trip property test.

**Human, in parallel**
- Approve the Charter v2 direction.
- Run the first ADR triage batch.
- Curate MEMORY.md and prune worktrees.
- Cut a release.

### 7.5 KPIs to re-measure at day 14 and day 30

| KPI | Now | Target |
|---|---|---|
| Fix-titled share of PRs | 19% | under 10% |
| Fixes within 24 h of a feature | 34 | under 10 |
| Lost-work cases | 12 | 0 |
| CHANGELOG and index repairs | 7 in three days (09-25 → 09-27) | 0 |
| PRs touching an index file | 72% | under 10% |
| Red-main count | 2 in the last 3 weeks | track |
| Median open → merge | 12 min (24 min in September) | track |
| Lazy share of cross-package imports | 71% | falling |
| Private share of cross-package imports | 48% | falling |
| Files over 2,000 lines | 15 | falling |

Also re-measure the §1 touch-point table, axis by axis, as each registry
lands.

---

## 8. Traceability: slice findings → root strains

| Slice | Strain (auditor's label) | Root |
|---|---|---|
| 1 Declaration | S1 `apesees.py` orchestration monolith | RS5 |
| | S2 typed surface restated by hand (dataclass, ns wrapper, index) | RS1 |
| | S3 element facts in 21+ fail-open tables | RS1, RS3 |
| | S4 declared state as per-kind lists threaded through 5 emit modes | RS2 |
| | S5 cross-primitive checks as straight-line code in `emit()` | RS3 |
| | S6 charter silently outgrown | RS8 |
| 2 Emission | S1 no lowering IR | RS2 |
| | S2 Protocol widening (29 → 74), P8 broken | RS2, RS7 |
| | S3 hand-maintained drivers multiply | RS2 |
| | S4 live execution is a runtime posing as an emitter | RS2 |
| | S5 validation interleaved with lowering, keyed on emitter identity | RS3 |
| 3 Results | S1 ~110 response tables | RS1 |
| | S2 5+ declaration surfaces, no read-back | RS2 |
| | S3 level taxonomy and capabilities per format | RS1, RS3 |
| | S4 fem_eid ↔ tag translation replicated per path | RS1 |
| | S5 accretion into Results; private reach into the bridge | RS4, RS5 |
| | S6 duplicated write pipelines | RS1 |
| 4 Persistence | S1 two-version window expires archives | RS6 |
| | S2 no declarative schema | RS1 |
| | S3 archive is not replay's inverse | RS2, RS3 |
| | S4 no shared H5 primitive layer | RS1 |
| | S5 identity tied to the file format | RS6 |
| | S6 fork flags in the neutral zone | RS7 |
| 5 Model core | S1 no declaration-kind protocol | RS1, RS3 |
| | S2 neutral core speaks OpenSees and Ladruno | RS7 |
| | S3 two declaration surfaces for the same physics | RS3 |
| | S4 plural name resolution | RS1 |
| | S5 two resolution pipelines; snapshot not pure | RS2, RS4 |
| | S6 ConstraintsComposite accretion | RS5 |
| 6 Viewers | S1 two presentation designs alive | RS6 |
| | S2 no picture model | RS2, RS1 |
| | S3 giant closure-wired methods | RS5 |
| | S4 concepts re-implemented per window | RS1 |
| | S5 leaky boundaries; guards cover old files only | RS4 |
| 7 Structure | S1 lazy imports hide the graph | RS4 |
| | S2 layering rules false | RS4, RS8 |
| | S3 eager, side-effecting root `__init__` | RS4 |
| | S4 private names as the API | RS4 |
| | S5 peripheral products coupled both ways | RS4 |
| | S6 hub modules | RS5 |
| | S7 static gates cover 26% | RS3 |
| 8 Variant/drift | S1 no capability model | RS7 |
| | S2 fork never executed in CI | RS7 |
| | S3 fork fused into shared seams | RS7 |
| | S4 normative docs describe May | RS8 |
| | S5 registers no longer hold state | RS8 |
| | S6 studio's generated file gates every PR | RS6 |
| 9 Workflow | W1–W8 | §7 |

Each slice's top-three recommendations, as reported:

| Slice | Recommendations |
|---|---|
| 1 Declaration | fail-closed parity and classification gates, plus a deterministic index; split `apesees.py` by reason to change; traits registries, then generated ns wrappers |
| 2 Emission | `EmitContext`/`TargetCaps` plus a frozen Protocol with `command()`; one staged program over `RankScope`, growing into a phase table; `DeckProgram` IR in shadow mode |
| 3 Results | table-consistency tests, then `ResponseFamily`; `TranslatingReader` plus capability sub-protocols plus a conformance matrix; `RecordIntent` with read-back; runner-up: `BridgeIntrospection` |
| 4 Persistence | compatibility floors plus an additive policy plus a migrator; a round-trip oracle plus loud replay; a declarative schema over an `_h5` leaf |
| 5 Model core | plug the four silent channels; a `DeclarationKind` registry; solver params out of geometry, plus unified name resolution |
| 6 Viewers | finish S6 as deletion; a Qt/VTK-free picture layer; point the guards at where code now lives, then split `ModelViewer.show` |
| 7 Structure | one layer contract over all imports; lazy `__init__`s; per-package API modules plus wider static gates |
| 8 Variant/drift | one backend capability model; derive state rather than narrate it; pin and run the fork |
| 9 Workflow | conflict-free indexes; `land_pr.py` plus nightly detectors; pre-merge review on risky paths plus revert-proven fix tests |

---

## 9. Method and verification log

**Auditors.** Each slice had one auditor, all read-only:
1. declaration layer;
2. emission;
3. results pipeline;
4. persistence;
5. model core;
6. viewers;
7. cross-cutting structure (an AST import graph over 632 modules, git
   snapshots at 05-20, 06-29 and 08-10);
8. variant axis and drift;
9. agentic workflow (408 per-PR commits, 1,182 PRs, 500 CI runs, branch
   protection read with `gh api`).

Their scratch scripts and probe outputs are in the session scratchpad, not
the repo.

**Re-verified independently by the synthesizer.**
- **Class sizes:** `BuiltModel` 7,147 LOC / 66 methods; `apeSees` 4,028;
  `_StageBuilder` 1,810.
- **Bridge code:**
  - the `H5Emitter` name check at `apesees.py:1282`;
  - the fail-open docstring at `_element_capabilities.py:725-732`;
  - the fork rosters at `live.py:158,173`;
  - Protocol method count over history (05-10: 29; 05-26: 53; 07-14: 67;
    09-07: 74);
  - the `stream`+`split` refusal;
  - the replay "SCOPED / best-effort" note.
- **Docs and indexes:**
  - the charter's single commit;
  - the `_api_index.json` timestamp;
  - the ADR 0052/0080 statuses (both Accepted);
  - the `RebarComposite` ADR citation;
  - CHANGELOG size (845 KB) and Unreleased length (8,949 lines);
  - the ADR index size (91 KB).
- **Results code:**
  - the two `RecorderRecord` classes;
  - the empty `.ladruno` layers and springs;
  - the capture spec's `getattr` on bridge privates;
  - the how-to's nonexistent API (no `def capture`, no `recorders=` on
    `tcl`/`py`);
  - the date of `d04a7484`.
- **Defects:**
  - the GeomTransfViewer sign, checked against `LinearCrdTransf3d.cpp`;
  - D1 and D2 re-run on this worktree's `src`;
  - the D3 and D4 probe outputs reviewed;
  - the compose `rebar_elements` carry.
- **Persistence:**
  - the schema window rule;
  - the current versions (2.33.0 neutral, 2.21.0 opensees);
  - that no migrator exists;
  - that `H5ModelReader` is defined only in the ADR text;
  - the residual `.get` lines.
- **Imports and guards:**
  - the direct `openseespy` import sites;
  - the dangling `TYPE_CHECKING` import;
  - the ADR 0014 guard skipping relative imports;
  - the two fork signals.
- **CI and repo state:**
  - CI concurrency and `strict: false`;
  - 0 reviews on the last 30 PRs;
  - 94 worktrees.

**Not independently re-verified (auditor evidence only).**
- PR and commit statistics beyond those listed above.
- The per-hunk churn mapping of `apesees.py`.
- The import-graph percentages.
- The fix-commit taxonomy.
- The viewer reachability count.
- The touch-point counts in §1, beyond the commits cited there.
