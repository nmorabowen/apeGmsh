# apeGmsh expert panel: execution, cuts, context, workflow, navigation, and the fork

**Status:** Panel recommendation, 2026-09-28, against `origin/main` @ `07f757e0`. The decisions still owned by the maintainer are listed in §9.

**Input:** `internal_docs/plan_architectural_strains_2026-09.md`, the assessment. Its errata E1–E7 came out of this panel.

**Method:** nine experts, two rounds.
- **Round 1:** independent position papers, each backed by code evidence.
- **Round 2:** a moderated rebuttal plus a 24-item ballot and a 14-item consensus slate.
- **Verification:** the moderator re-checked every load-bearing claim before it entered the ballot (Appendix C).
- **Adversarial pairing:** the fork question was argued by steelman advocates on *different* model families. A red-team seat attacked the whole program.

| Seat | Role | Model |
|---|---|---|
| P1 | Chief Architect (decomposition, extension points) | Opus 5.5 |
| P2 | Product Scope & Cuts | Fable 5.1 |
| P3 | Fork Advocate (steelman: lean into Ladruno) | Fable 5.1 |
| P4 | Vanilla Advocate (steelman: keep stock first-class) | Opus 5.5 |
| P5 | Context & Agentic-Workflow Engineer | Fable 5.1 |
| P6 | Navigation & Agent-DevEx Engineer | Opus 5.5 |
| P7 | Bloat & Simplification Auditor | Opus 5.5 |
| P8 | Red-Team Skeptic | Fable 5.1 |
| P9 | OpenSees Cross-Parity & Execution-Model Architect | Opus 5.5 |

---

## 0. Decisions at a glance

| Question | Panel decision | Vote |
|---|---|---|
| Lean fully into Ladruno, or keep vanilla? | **"B with amendments."** The fork comes first *in investment*; stock stays a first-class, executable profile; the default run target is **auto-detected** from the running binary. | K1 9–0; co-signed by both advocates |
| Keep fork vocabulary out of the neutral core? | **Yes, now,** as a ratchet. `_kernel`/`core`/`mesh`/`results` may not grow fork tokens. The `.ladruno` format readers are exempt. | K2 8–1 |
| Go fork-only at some point? | **As a review trigger, never an automatic flip.** | K3 8–1 |
| Break the Tcl rule? | **Yes, in a narrowed form: the resolved `model.h5` is the program.** It is test-first. Tcl/Py become printers of the archive only after gates pass. The fork later loads the archive natively. | K5 8–1 |
| Where do fork capability facts come from? | **From the binary.** A `ladrunoSchema` dump is generated from the fork's own registries. | K6 8–1 |
| Build registries and IRs now? | **Minimum first.** Extension points are built only when their trigger fires. No DeckProgram, ColumnSpec or picture IR. | K8 5–4; S13 |
| Split the giant classes? | **Yes, as pure moves, starting with one hub.** The `BuiltModel` split is dropped. | K9 8–1 |
| What to cut? | **About 37k LOC deleted** (§3). Studio is frozen, not spun out; interop is kept. | K15–K22 |
| Landing tooling? | **About 80 lines of `land_pr.py`, plus GitHub settings.** `preflight.py` is cut. | K10 9–0, K11 7–1 |
| Review? | **Cross-family, read-only review on label-marked semantic PRs only.** | K12 7–2 |
| Navigation? | **`scripts/nav.py`, AST-only, plus a fail-closed family-completeness gate.** No committed line maps; the atlas is deleted. | S10, S3, K15 9–0 |

---

## 1. The fork: lean in, or keep vanilla?

### The decision: "B with amendments"

Both steelman advocates co-signed this outcome in round 2. Their concessions are the strongest evidence it is right.

**The fork is where investment goes, not the runtime default.**
- The fork gets the native loader, `.ladruno`, per-rank H5 load and the fork command schema.
- The run target is resolved from the binary that is actually imported: `BackendInfo` reads `ladrunoBuild()`, and later `ladrunoSchema`.
- Every deck and every `results.h5` is stamped with provenance: engine, stock version, fork build.
- Reason, from P3's own concession: the fork *also* fails silently (§A4 below). No engine has earned a blind default. And the fork has 0 releases and 0 tags, so it cannot be a pip user's default.

**Stock stays a first-class, executable profile.**
- `live-stock` remains a required CI lane. It is the *only* lane that executes OpenSees today.
- Pin `live-stock` to **Python 3.12**. On 3.11 it resolves openseespy 3.7.x, which predates `equationConstraint`, so CI tests an older stock build than users install (P4, re-checked at `tests.yml:319,333`).
- Tcl/Py decks remain forever: for vanilla users, for review, and for upstream bug reports.

**Neutral layers never speak fork (K2).**
- A per-package ratchet on fork tokens in `_kernel`, `core`, `mesh` and `results`, keyed by package glob so it survives the splits.
- The exemption: `results/readers/_ladruno*.py` is a *file-format* reader, like MPCO.
- Fork options persist only as argv rows in `/opensees` (under K5) or in a `solver_params` lane. They never become neutral-zone columns. About 16 of the last 22 neutral schema minors were forced by fork options; this ends that.
- First target: `_kernel/_coupling_control.py:195`, which emits OpenSees argv from the kernel.

**A two-sided divergence roster, one oracle per row.** Refuse where the answer is wrong; warn where it is degraded.

| Engine | Rows |
|---|---|
| Stock | `TenNodeTetrahedron` 6× soft · DruckerPrager tension cutoff · ASDPlastic apex · LAPACK zero pivot reported as success · `-hall` tangents |
| Fork | three different unknown-token policies (ignore, warn, fatal) · the #873 flags shipped by #1184 |

Every row needs a closed-form or cross-engine oracle that names *which engine is right*. Without one, a roster blames the wrong component, as the "71% soft ties" row did (errata E3).

**Fork-side hygiene.** These are asks of the fork repo, and each is small.
- One **fail-closed unknown-token policy** across all fork parsers, with `since` per option. This is the structural fix for the #1184 class.
- A CI **artifact upload.** Zone-A already builds `opensees.so`; about 10 lines add the upload.
- **Release assets per pin:** an Inno installer and a Linux tarball. There is no Windows wheel; MKL alone is about 300 MB.
- A **`ladrunoSchema` v0** generated from `classTags.h` and the function maps. `manifest.yaml` becomes a checked subset of it; today 17 of its 69 rows are wrong.

**The A-gate is a review trigger (K3).**
- The review fires when `live-fork` has been green for 30 days against a committed `FORK_PIN`, a downloadable artifact exists per pin, and `BackendInfo` enforces the floor table. "Green" means `ladruno_mkl` tests are skip-with-reason and counted (P3).
- P4's distribution, second-committer and upstream criteria are *inputs* to that review, not preconditions. P3 showed they cannot be closed by repo work.
- Under K5 the stock profile costs about one printer, so a fork-only flip saves little.

**Upstream first (K4, 7–2).**
- A fork commit that fixes correctness in an upstream-owned file must link an upstream PR or state why not. It has to be enforced as a fork PR-trailer check; otherwise it is prose.
- A quarterly upstream sync.
- File the tet10 fix upstream first.
- Why: the fork modifies 256 upstream-owned files, is 1,569 commits ahead and 157 behind, and the maintainer has filed 0 upstream PRs. Each merged upstream fix deletes a roster row for every openseespy user and shrinks rebase debt.

### Why not the extremes

**(A) fork-only** would:
- delete the only CI lane that executes anything;
- refuse all pip users (openseespy has ~579k downloads a month);
- bind apeGmsh to an unpublished binary whose parsers are not yet fail-closed.

**(C) as a separate plugin package** would:
- double every registration surface across two repositories;
- split every element family between them;
- add cross-repo sequencing to a workflow that already loses work within one repo.

P4's in-repo version of C turned out to *be* P9's archive boundary, so it merged into B.

**Dissent.**
- **P2 wanted K3(c):** distribution and a second committer as hard preconditions.
- **P8 voted K2(c):** a ratchet baselined in the hundreds never moves, and `requires=` suffices.
- **P5 and P8 voted K4(b):** sync only, because an unmechanized "must link" rule is prose.

---

## 2. Breaking the Tcl rule: the archive is the program

### P9's cross-parity verdict: the hypothesis is half right

**What OpenSees imposes.** OpenSees imposes the *shape* of apeGmsh's worst strains:
- a command vocabulary with no metadata (imperative argv parsers);
- per-kind tag registries;
- interpreter ordering rules. For example, `sp` needs a current pattern, and re-issuing `model` resets the builder;
- one script per MPI rank (OpenSeesMP);
- a string response protocol whose descriptor is discarded (`Domain.cpp:1954`).

**What apeGmsh adds.** The *multiplicity* is apeGmsh's own:
- 5 emitter implementations × a 74-method Protocol;
- 7 hand-written drivers;
- the H5 re-inference;
- replay gaps;
- 21+ fact tables.

**The split.** About 45% of RS2's cost and a third of RS1's is inherited; the rest is self-inflicted. Staged-live refusal is fully self-inflicted, since every staging verb is callable live. The Tcl rule explains why apeGmsh *restates* OpenSees, not why it restates it five times.

### The decision (K5 8–1, K23 6–3)

**The fact it rests on.** The resolved `model.h5` already stores resolved argv rather than declarations: materials as `type/tag/params` (`h5.py:2716-2722`), element tails in `element_meta`, and node-level load rows. P1 re-checked this in round 2 and withdrew its in-apeGmsh DeckProgram, because a separate tree would be a third representation. So the archive is declared *the program*, in three gated tranches.

**Tranche 1: now. apeGmsh only, additive, all tests.**
- **R1a, archive completeness.** Every Emitter verb is either archived or *refused* by `apeSees.h5()`, and an AST lock enforces this.
  - Today global Rayleigh is not archived (`h5.py:2048-2080`).
  - MP constraints, contacts and equation ties are not replayed (`compose.py:578-585`).
  - Under K5(c) those gaps stay silent, which is RS3's fail-open class.
- **R1b, a round-trip oracle.** `tcl() == from_h5(h5()).build('tcl')` on the golden corpus (§4). Rayleigh, MP, contacts, ties and stage mutators go on a **shrink-only expected-failure (xfail) ledger**, which must reach zero within 60 days.
  - If it does not, "printers of the archive" is dropped and R1 stays a gate only (P8's KC4).
- **P1's E4 and P9's archive are one design.**
  - `command(name, *args)` plus a Protocol freeze is the archive's row format.
  - The `VERBS` table's `h5` column records "archive" or "refuse".
  - `will_solve` is stamped into `/opensees`, replacing the `H5Emitter` name check (`apesees.py:1282`).
- **The tag law is settled in this tranche:** tag-stable emission, so the oracle drops its "modulo tags" clause (P8's KC3).

**Tranche 2: the fork loader. Gated, additive, in new fork files.**

The loader, `ladrunoLoad model.h5 -only model`, is fail-closed and feeds archived argv into the fork's *own* parsers, so no second source of model semantics is created. Its gate is `ladrunoDomainDigest`: a `sendSelf`-hash of the domain that must be equal across the Tcl deck, the Py deck and the loader, plus eigen, static and transient results equal to 1e-12 on about 9 models.

**P8's kill criteria are attached (KC1–KC6):**

| # | Criterion |
|---|---|
| KC1 | No loader PR until `live-fork` has been green for 14 consecutive nightly runs. |
| KC2 | The digest must change when any archived parameter is perturbed. There is no skip list. |
| KC3 | The tag law is settled first. |
| KC4 | The xfail ledger may only shrink. |
| KC5 | The fork's single command-registration hook (P9 R10) lands before the loader. |
| KC6 | A cluster run uses the loader within 60 days, or the loader is deleted. |

**Tranche 3: the driver-to-printer flip. Triggered and dated, not scheduled.**

The flip happens only when all of the following hold (P7's safety nets):
- P1's DeckProgram trigger fires: the next invariant needs more than two per-path hoists;
- the oracle has been green for 30 days;
- goldens have been captured from *today's* drivers;
- a dataset-level diff of `model.h5` against the current H5Emitter passes;
- a driver freeze is in effect, with no new modes;
- the flip has a date recorded in the retirement ledger.

**What it pays off:** about 0.7–0.9k LOC of H5 re-inference and up to about 2k LOC of drivers (estimates). Per-rank and split output become printer transforms. In the near term K5 *adds* LOC. Only the flip pays that back.

**One archive-to-deck path only (P2).** `compose.py:582` already makes `from_h5 → forward re-emit` canonical. The printer reads resolved records; it does not re-grow `_replay_into` next to the forward path. K23(b): grow the replay fail-closed, and flip one mode at a time, only on a green corpus.

### What stays OpenSees-shaped

- **The typed primitives stay hand-described** (K7 5–3–1).
  - The fork's `OptSpec` generation is a cross-repo L program (P9 R6), and #1184's root cause was a parser ignoring unknown flags, not missing generation. Fail-closed parsers plus the `live-fork` lane close that class.
  - Parity tests remain the guard against the 938-parameter triple restatement.
  - Generation waits until the parity locks fire twice in a quarter. P2, P4 and P7 preferred generating all ns wrappers from the dataclasses, which would delete about 3k lines now.
- **`ladrunoResponseLayout` is recommended but unscheduled** (P9 R4). It returns the `setResponse` descriptor instead of discarding it. It works for stock and fork classes without touching any parser, which makes it the highest-yield C++ line on the list.

**Dissent (P8, K5c).** "Archive-as-program is DeckProgram serialized." P8 accepts R1a and R1b as *gates*, not as the start of a flip. The KC1–KC6 conditions are P8's price for the majority position.

---

## 3. What to cut

| Item | Decision | Vote | LOC / bytes | Condition |
|---|---|---|---|---|
| Legacy results window (`results_viewer.py`) + director chain + legacy-only diagrams/UI | **CUT** | S9 consensus | ~23.9k src (44 modules) + 2.9k legacy-only tests; 42 test files (~19.9k) need surgery | Capture golden stills and frame counts first. Port `export_animation` (restep over `ResultsSession.render`) and move `render`/`render_pack`/`assess` onto the session, then delete. Drop the eager `diagrams/__init__` re-exports. |
| `show_web` / `serve_web` / trame | **CUT (no port)** | K17 6–0 (3 abstain) | ~0.5k + 6 deps | Notebook fallback becomes a temp `.h5` + `python -m apeGmsh.viewers`; `render()` stills remain. |
| `split=` emit mode (ADR 0043) | **CUT** | K20 8–0 | ~700 | ADR 0043 has been Proposed for 123 days and has 0 examples. Rebuild as a printer transform only if demanded. |
| Plotly `NotebookPreview` + Three.js `geom_transf_viewer` | **CUT**, plus a **two-render-technology rule** (VTK + matplotlib) | K21 7–0 | ~1.2k | The in-flight D5 axis fix becomes moot if cut; keep it as an interim fix only if it is already done. |
| Atlas + `docs/api-flows/flows.json` | **CUT** until a generator exists | K15 9–0 | 2.2 MB | 69% of its 909 pointers are wrong or dead. Fix the false "regenerated by their tool" line in `adr-docs/SKILL.md:60`. |
| `architecture/layout.md` | **CUT** | S12 / P6 | — | `nav pkg` replaces it. |
| May scope docs (`phase-8.x`, `phase-9`, etc.) | **CUT**, with SHA-pinned citations | 5–4 | ~171 KB | Git is the history. Re-point the ~20 inbound references. |
| Certain-tier dead code | **CUT** | S11 | 622 LOC | The ndf populator, the dead `_fv<(2,7,0)` branch, the chimera compose fallback, the `results/_vocabulary.py` shim, the duplicate `GMSH_LINEAR`, the uncollected script, `.bak` files, the 7.7 KB CHANGELOG header line. |
| Incident narrative in comments | **TRIM** with a lint | K22 8–1 | ~3.85k lines / ~60k tokens | Keep MUST/NEVER/CONTRACT/INV and one-line "see ADR" pointers. Trim each hub after its split. The lint covers new code. |
| Likely-tier bloat | **CUT** behind named nets | P7 | ~6.9k LOC + 2.95 MB | Clone families, test helpers, 2.44 MB of archive notebooks, scratch directories. The loads/masses twins wait for the D1/D2 fixes; the tcl/py formatting waits for the corpus. |
| `sensitivity/`, import banner | **CUT** | P2 (not balloted) | ~0.56k | — |
| `ModelData` (ADR 0018) | **Maintainer decision** | K19 3–0 (6 abstain) | 710 | 0 docs, 4 tests, and a documented silent-wrong mode (`model_data.py:51`). |
| Studio | **FREEZE in-tree** | K16 7–2 | 11.2k (kept) | Deterministic `_api_index.json`, a LOC ratchet, and spin-out after 60 untouched days (P8). |
| Interop | **KEEP** behind an extra | K18 4–2 (3 abstain) | 2.5k (kept) | Fix its two direct-openseespy (D6) sites. |
| Retirement-ledger freezes | **FREEZE** with dated expiry | P2 / P7 | ~5k | The section GUI, `from_recorders` transcoders, `hpc/`, `ground_motions/`, and the `rebar/` fold. |
| MPCO readers, Tcl/Py printers, `live-stock`, `RecordingEmitter` | **KEEP** | consensus | — | MPCO covers the layers and springs `.ladruno` returns empty. |

**Totals, reconciled by P7:**
- **about 37k LOC deleted** in total: about 26.8k from the legacy viewer including its tests, 3.1k of product dead-ends, 0.6k certain, and 6.9k likely (3.85k of that is comments);
- **about 3 MB** of non-code bytes;
- **K5 contingent:** a further 2.7–2.9k becomes deletable only after the tranche-3 flip.

**Rule adopted:** every retirement ends in deletion. It gets a ledger row with a *dated* CI expiry, because "one release cycle" is not a date. The `_vocabulary.py` shim has outlived its promise by 139 days.

---

## 4. How to make the improvements

**Program shape: minimum first (K8, 5–4).**
- Build now:
  - Phase-0 fixes (S1);
  - fail-closed gates (S3);
  - the golden corpus and move proof (S4);
  - re-keyed guards (S5);
  - loud replay (S6);
  - the schema floor (S7);
  - the fork lane (S8).
- **E3** (traits plus a completeness gate) *is* S3.
- A **Protocol method-count freeze** has support on both sides of the K8 split.
- **E4** `command()` lands inside K5 tranche 1.
- **E1** (a Preflight registry as an explicit ordered check list) and **E2** (`TargetCaps`, which replaces 4 `getattr` probes, 4 `setattr` type-ignores and the `H5Emitter` name check) are S-size and trigger-gated. P1 argues their triggers have already fired twice; this is the maintainer's call.
- **Deferred with named triggers:** DeckProgram, ColumnSpec, DeclarationKind, ResponseFamily generation, picture-compute, `RankScope`/E5, the migrator, TranslatingReader/BridgeIntrospection/ModelArchive, and lazy `__init__`.

**The proof regime** (resolved between P1 and P8; applied in W0, before anything moves):
1. **A committed golden corpus.** 7 FEMStub fixtures × {flat, partitioned, staged, staged+partitioned, per_rank} × {tcl, py, recording}, plus a canonical H5 dump (path, dtype, shape, sha1; `created_iso` masked). This replaces today's single 94-line deck.
2. **`verify_move.py`.** It checks that the multiset of normalized `ast.dump` over every moved def/class body is equal before and after. That is the right gate for whole-definition relocations.
3. **ruff F821 on every package a codemod touches.** Today ruff and mypy gate only `opensees/`, and `ast.dump` does not resolve free names.
4. **Not pure moves.** `H5Buffer`, `BridgeView` (`self.` becoming `view.`), mixins, `_add_def` becoming `_declare()`, and closure hoisting are semantic changes. They need the corpus and a review.
5. **Why the corpus comes before any split.** A pure move can reorder import-time registrations, and a multiset proof cannot see that.

**Kill criteria (S14).** They are measured weekly from day 14, by the weekly triage:
- the strict fix-prefixed share above 25% for 4 weeks means stop splitting and keep the gates;
- a split PR that needs more than 2 fixes within 7 days is reverted;
- non-refactor PRs below 50% of the trailing mean for 2 months means the program is starving the product;
- any phase past 2× its estimate cancels the next;
- a shadow registry older than 30 days with no flip is deleted. S3's families list is gate configuration and is exempt (P6);
- the fork lane not green within 30 days of the pin forces an explicit C or D decision;
- the index-touch share not below 20% within 30 days of fragments means fragments failed.

**Why this order.** P8's base rates from this repo's own history:
- mechanical relocations (Phase 8.1–8.3, P1-K) finished in days;
- semantic unifications (the three-broker refactor, Phase 9, selection v2) finished fast, but left residue that became the next strain;
- de-publication (ADR 0098 S6) never finished;
- prose promises (the ADR 0023 migrator, the ADR 0051 warning, the ADR 0102 freeze) never shipped.

So the plan is additive gates first, pure moves second, and semantic unification last, only on triggers, with dates.

---

## 5. Cutting the classes into manageable parts

**Scope (K9, 8–1).**
1. Start with one hub: move `_StageBuilder` whole-class to `opensees/stage/`, and move the procedures (modal, FRF, explicit, contact queries) to `opensees/procedures/` as free functions over a `BridgeView` protocol, with delegators kept on `apeSees`.
2. Widen to further hubs only after the first move needs at most one fix within 7 days.
3. **The `BuiltModel` split is dropped** (P1 withdrew it). The per-mode drivers are K5's deletion target, so splitting them first would polish code headed for replacement. **Low-churn hubs** (`Results.py`, `ModelViewer.show`) are split on touch only.

**Target layouts** (from P1; LOC are AST-measured; each file stays under about 1,500 lines, with one reason to change):
- **`_internal/build.py`, 10.3k lines.** It becomes a `build/` package in 5 layers, where each layer imports only from the layers below it. That gives **0 import cycles** after relocating 8 names; a naive split by concern gives about 50.
  - L0: `errors`, `records`.
  - L1: `pg_expand`, `topo`.
  - L2: `ndf`, `transforms`, `ownership`, `validators/*`.
  - L3: `builder_scope`, `element_emit`, `patterns`, `recorders`, `constraints`, `interfaces`, `stage_emit`.
  - L4: `constraints_ranked`.
- **`material/nd.py`, 4.1k lines.** Split by physics family: elastic, sand, asdplastic, concrete, plate, finite_strain, cohesive. Being fork-only is a *trait* of a class, not a folder.
- **`emitter/h5.py`, 4.2k lines.** Separate the capture and write seams through an explicit `H5Buffer`. This is semantic, so it needs the corpus. It aligns with K5 tranche 1.
- **`mesh/_femdata_h5_io.py`, 4.2k lines.** Co-locate encode and decode *per record kind*. That gives ColumnSpec's locality without its registry.
- **`mesh/_compose.py`** becomes a package (labels, tree, rewrite, merge, partitions, verify, api), but only after `CARRY_ALL` and the `compose-streams` lint are re-keyed.
- **`ConstraintsComposite.py`** gets mixins per lane (contact, node_pair, coupling, bc, resolve). This is semantic, and comes after `_declare()` lands.

**The split protocol:**
- **One codemod per hub,** re-run on the latest `main` rather than rebased. A freeze lasts hours.
- **A move map** (qualname → module) feeds `nav.py`. `port_hunks.py` retargets in-flight branches **by qualname**; function name alone is ambiguous where drivers share `_emit_rayleigh`. Import-block hunks cannot be retargeted automatically.
- **Imports are rewritten in-repo in the same PR:** src, tests, docs and the skill. **Underscore modules get no facade.** A public module's facade gets a ledger row with a dated CI expiry (P7's objection, adopted). Nine permanent facades would double every grep and `nav` hit.
- **Guards are re-keyed before anything moves** (S5): `check_quirks.py:46,218` (compose-streams checks two filenames), filename-keyed test budgets, and mkdocstrings `inherited_members: true` on the `apeSees`/`ConstraintsComposite`/`Results` pages. Otherwise a mixin split silently drops public methods from the API docs.
- **A new `stale-patch-target` lint.**

**Guardrails.** All are ratchets that may only shrink:
- **File size:** a new file may have at most 1,500 lines, and existing files over 1,500 may not grow.
- **Class size:** a new class may have at most 40 methods and 1,000 lines. Protocol implementers are exempt; the Protocol is locked at its current count.
- **Function size:** a new function may have at most 200 lines.
- **Ownership:** an `ownership.toml` maps each module glob to one reason to change. Fork-adoption PRs land in validators, primitive files and registry rows, **never in the emit drivers**.

---

## 6. Shaping context

**The context stack:**

| Tier | Budget | Contents |
|---|---|---|
| Always loaded | **≤ 2.5k tokens** (today ~8.6k) | AGENTS.md ≤ 150 lines, including the navigation paragraph (≤ 120 words); global `~/.claude/CLAUDE.md` ≤ 10 lines of machine paths; MEMORY.md ≤ 20 entries |
| Per task | ≤ 1.2k tokens | The **slice card**, which is also the ledger issue body. It lists owned files, verification command, stop condition, model and effort, and token rules, and cites `path::symbol` (resolved by `nav where`), never line ranges. |
| On demand | ≤ 1.5k per answer | `nav.py` answers; task guides (0.8–1.7k each); a ≤ 60-line decision summary at the top of every ADR over 40 KB |
| Never | — | Reading a file over 2,000 lines whole; reading the ADR index whole |

**MEMORY.md policy.** Private memory holds only three things:
- machine facts that are not in the repo (at most 8);
- maintainer preferences (at most 5);
- dated pointers to personal in-flight work that expire in 14 days (at most 5).

Everything about code, decisions, lessons or "shipped in #N" belongs in the repo. The 23 open items move to the ledger, where Codex, Cursor and the second machine can see them.

**Documentation:**
- The **ADR index is generated from the existing `**Status:**` lines** and rendered at build time, not committed (K13 9–0). There is no 110-file frontmatter migration; fix the 7 mismatches by hand.
- **`architecture/` moves out of `src/`** (8–1).
  - It is 2.7 MB of markdown that doubles grep noise.
  - Do it only after the doc-path rule lands, inside a freeze window, with a lint rejecting new files at the old path (P6's conditions).
  - AGENTS.md's "list the directory on origin/main" rule and the `adr-number` lint are updated in the same PR.
- **"until #N merges" lines expire mechanically.**
- **The comment-provenance diet** (§3) takes about 120k tokens of narrative out of the hub files agents read most.

**Maintainer housekeeping** (outside the repo):
- remove the stale `anthropic-skills:apegmsh-helper` skill copy;
- trim the global `CLAUDE.md` stacked-PR essay, which AGENTS.md duplicates;
- curate MEMORY.md;
- prune worktrees (105 today, most on deleted branches).

---

## 7. The agentic workflow

**The ledger.**
- **Slices:** one GitHub issue per slice, with the slice card as its body.
- **Labels:** `slice`, `phase:`, `lane:`, `in-flight`, `lesson`.
- **WIP:** at most 8 open PRs.
- **Freezes:** a small `FREEZE` list with expiry dates, one entry per hub split.
- **Weekly 30-minute triage:** it computes the S14 kill metrics, reviews due `lesson` items, and moves ADR statuses. Batch 1 is the 12 "Proposed" ADRs that have shipped.

**Landing (K10, 9–0 for the minimal form).** `scripts/land_pr.py` is about 80 lines and checks, in order:
1. The base is `main`, and the PR is open and not a draft.
2. `HEAD == headRefOid`, with a clean tree.
3. The PR touches no frozen file.
4. After the merge (`gh pr merge --squash`, never `--auto`), `compare/main...<sha>` reports `behind` or `identical`.

It ships with a four-case replay test covering #858, #1057 and #1097. Every check traces to at least 2 real incidents, except freeze refusal, which is the only thing that makes a declared split window more than prose (P5 answering P8). P8's caution stands: the script checks and verifies, and the merge itself stays an explicit command.

**GitHub settings** (maintainer):
- stop cancelling CI runs on `main`;
- a ruleset requiring `lock-tests` on all branches, so the PR-base check blocks instead of flagging;
- **decide `strict: true`** (see §9).

**Cut from the workflow plan:**
- `preflight.py` (K11, 7–1). CI takes 8 minutes; keep only a 20-line helper that prints CI's exact test selection.
- the 12-check orchestrator;
- the frontmatter migration;
- committed line maps.

**Review (K12, 7–2).**
- Scope: cross-family, read-only adversarial review on **label-marked semantic or refactor PRs** only.
- Pairing: Fable-authored PRs get an Opus reviewer, and vice versa.
- Exemption: AST move-proven PRs.
- Output: the verdict lists what was hunted.
- Kill rule: drop the practice if it finds less than 1 real issue per 10 reviews (P8).

**The lesson-to-rule SLA.** The second occurrence of a failure class becomes a quirk rule, guard test or ratchet within 7 days, proven on the pre-fix tree. The first two cases are the `getattr` lint and the doc-path rule.

**New fork options** ship with a fork test that asserts the option's *effect* (P7). This would have caught #1184.

**Model allocation** (P5, endorsed):

| Work | Model | Effort |
|---|---|---|
| Mechanical slices: pure moves, table derivations, fragments, ratchets | Sonnet | medium; the parity gate, not the model, makes this safe (ADR 0109 precedent) |
| Bulk harvesting: maps, KPIs, claim probes | Haiku fan-out → Sonnet aggregate | low |
| Feature slices with an ADR and an oracle | Fable or Opus, alternating | medium–high |
| Irreversible designs: schema floor, split order, fork policy, charter | Opus ∥ Fable independently → reconcile → human ratifies | high |
| Adversarial pre-merge review | the other family from the author, read-only | high |
| Orchestration: dispatch, read reports, run `land_pr` | Fable | low |
| Panels like this one | 4–9 mixed seats, at most monthly, only for decisions governing ≥ 50 slices | high |

---

## 8. Navigating the library

**`scripts/nav.py`** (S10; P6's working prototype):
- **How it works:** stdlib only and AST-only. It never imports apeGmsh, so the editable-install trap cannot bite. It reads *this* checkout, keeps a per-worktree cache, and bounds every answer.
- **Commands:**
  - `map` (with `--nested`), `at`, `where`, `refs`, `up`: structure, definitions, references and callers;
  - `deps`: eager, lazy and `TYPE_CHECKING` importers;
  - `h5`: who writes, reads or probes an HDF5 path;
  - `impl`, `family`, `pkg`.
- **Measured on five real questions:** 8 calls and about 2.8k tokens, against 27+ calls and about 23k+ tokens with Grep and Read, and nav was more complete. `map` of the 14k-line `apesees.py` costs about 2k tokens; reading the file costs about 167k.
- **Unprompted findings:** `nav family` surfaced the D9 compose gap and the D2 dispatch gap.

**The family completeness gate** (S3):
- A `families` registry declares each family's must-be-complete sites.
- `tests/test_family_completeness.py` fails closed. It starts from today's gaps as a ratcheted exception list with a reason per entry: `ALL_ND` covers 8 of 22, and 54 of 181 concrete primitives are in no `ALL_*` list.
- This absorbs P1's E3 and the classification lock. `nav` finds; the gate proves.
- The **private-`getattr` lint** comes with it. It also flags `getattr` on names nothing defines (P7), which is the mechanism behind the 98- and 138-day capture outages and the `node_ndf` vestige.

**The committed map is invariants only:**
- package `__init__` docstrings (at most 15 lines: purpose, entry points, invariants, hub notes), printed by `nav pkg`;
- a **doc-path rule** in `check_quirks.py`: every backticked path or `path::symbol` in AGENTS.md, the guides, the non-history architecture docs and the package docstrings must resolve. Today 100 of 938 architecture citations are dead.

**Do not build:**
- a maintainer MCP server;
- `python -m apeGmsh.dev`, which would import the wrong checkout;
- committed line-number maps, which would be a fifth merge-magnet index.

**One AST import extractor** feeds `nav deps`, the K2 fork-token ratchet and the layer contract, instead of an eleventh bespoke guard.

**The AGENTS.md paragraph** (P6's text, trimmed to ≤ 120 words): *find before you read*. Run `nav map/at/where/refs/up/h5/deps` first, and `nav family <Base>` before adding an element, material, response or declaration kind. Read only the ranges `nav` names. Never read a file over 2,000 lines without offset and limit.

---

## 9. Open decisions for the maintainer

1. **Ratify "B with amendments"** as a Charter v2 ADR, and amend ADR 0102 D2/D5 to match: a mechanized freeze, and "fork green in `live-fork`; stock honest or refused". This is the panel's central outcome.
2. **K24, staged live execution: tied 4–4.**
   - *Implement it behind the deck-vs-live parity oracle* (P2, P3, P4, P9). The refusal is self-inflicted, and `live.py:1469-1510` already forwards the steps.
   - *Keep refusing until the stage mutators exist and there is demand* (P1, P5, P6, P8).
   - Moderator's lean: keep refusing now, and revisit after K5 tranche 1 archives the stage mutators.
3. **K19, cut `ModelData`:** 3–0, with 6 abstentions. The evidence is 0 docs, 4 tests and a silent-wrong mode.
4. **K8's E1/E2** (Preflight registry, `TargetCaps`): build them now (P1) or at the next trigger (5–4)? Both are S-size.
5. **`strict: true` on `main`.**
   - *For:* guarantees the merged combination ran, closing the #605+#606→#608 class of two green PRs making a red main.
   - *Against:* more rebases at about 90 PRs a month.
   - Recommendation: yes. The median number of concurrent open PRs is 1.
6. **Fork-repo work, in the fork:**
   - the artifact upload;
   - the fail-closed unknown-token policy;
   - the `ladrunoSchema` v0;
   - the R10 registration hook;
   - an upstream PR for tet10;
   - whether #873's `-crackedNu`/`-betaC` will land at all (a fix task is queued).
7. **Personal configuration:** the MEMORY curation, global `CLAUDE.md` trim, stale skill removal and worktree pruning from §6.
8. **Landing these two reports** through a docs PR.

---

## 10. The 30-day plan

**Week 1: safety nets, and stop the silent wrongs.** Everything here is parallel and S-sized.
1. Land the queued fix tasks:
   - D1–D7;
   - the #1184 flag hole;
   - the tie re-diagnosis, with `live-stock` pinned to Python 3.12.
2. CI and settings:
   - no cancelling on `main`;
   - the `lock-tests` ruleset, plus a decision on `strict: true`;
   - a deterministic `_api_index.json`;
   - changelog fragments;
   - an ADR index generated from the Status lines.
3. The S3 family-completeness gate, and the `getattr` lint covering private names and names nothing defines.
4. W0 proofs:
   - the golden corpus with a canonical H5 dump;
   - `verify_move.py`;
   - ruff F821 on codemod-touched packages;
   - re-keyed guards and `inherited_members`.
5. `land_pr.py` (~80 lines) and its replay test.
6. `nav.py`, the AGENTS.md navigation paragraph, and the context diet (AGENTS.md ≤ 150 lines, MEMORY ≤ 20 entries).
7. P7's certain-tier deletions (622 LOC).

**Weeks 2–3: first cuts, the first move, and the fork lane.**

8. The fork lane:
   - the fork PR adding the artifact upload;
   - a committed `FORK_PIN`;
   - a nightly `live-fork` lane;
   - a single-signal `BackendInfo` with an auto default and provenance stamps.
9. The two-sided divergence roster with per-row oracles. The fork PR for a fail-closed unknown-token policy starts with LadrunoRCConcrete.
10. Cuts:
    - `split=`;
    - the plotly and Three.js viewers, plus the two-render-technology ADR;
    - the atlas, `flows.json` and `layout.md`;
    - the banner and `sensitivity/`.
11. The first hub move: `_StageBuilder` and the procedures, under a FREEZE entry, backed by the move proof and the corpus.
12. K5 tranche 1:
    - R1a archive completeness under an AST lock;
    - the E4 `command()` / `VERBS.h5` column;
    - the Protocol count freeze;
    - the tag law.
13. The K2 fork-token ratchet baseline.

**Week 4 and later: the big deletion, the oracle, and wider splits.**

14. Golden stills. Then port `export_animation` and `render`/`render_pack`/`assess` to the session, and delete the legacy window, the director chain and trame (~26.8k LOC including tests).
15. R1b, the round-trip oracle with its shrink-only xfail ledger. Grow the replay fail-closed.
16. Widen the splits to the `build.py` layers, `nd.py` families and the `compose` package, once the first move needed at most 1 fix in 7 days.
17. The comment-provenance lint, with per-hub trims after each split.
18. The retirement ledger with dated CI expiry, covering facades, hatches, xfail lists, shadow paths and the freeze list.

**Gated, not scheduled:**
- `ladrunoLoad` plus the digest, under KC1–KC6;
- `ladrunoSchema` v0 feeding `BackendInfo`;
- `ladrunoResponseLayout`;
- upstream PRs (tet10 first);
- the dated driver-to-printer flip;
- E1/E2 when their triggers fire.

**KPIs, re-measured at day 14 and day 30:**
- the strict fix-prefixed share (September baseline 22%);
- the index-touch share (80–85%, target under 20%);
- lost-work cases;
- red-`main` count;
- files over 2,000 lines (15);
- the lazy share of cross-package imports (71%);
- the private share of cross-package imports (48%);
- touch points per growth axis;
- the `live-fork` green streak.

---

## Appendix A: Round-2 vote matrix

The options for each item are listed in `scratchpad/panel/ROUND2_PACKET.md`. `abs` = abstain.

| K | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 | Tally |
|---|---|---|---|---|---|---|---|---|---|---|
| K1 default target | b | b | b | b | b | b | b | b | b | b 9 |
| K2 neutral-layer rule | a | a | a | a | a | a | a | c | a | a 8, c 1 |
| K3 fork-only | b | c | b | b | b | b | b | b | b | b 8, c 1 |
| K4 upstream-first | a | a | a | a | b | a | a | b | a | a 7, b 2 |
| K5 archive is the program | a | a | a | a | a | a | a | c | a | a 8, c 1 |
| K6 capability source | a | a | a | a | c | a | a | a | a | a 8, c 1 |
| K7 generation | c | b | a | b | c | c | b | c | c | c 5, b 3, a 1 |
| K8 extension points | a | b | a | a | b | b | b | b | a | b 5, a 4 |
| K9 split scope | a | b | b | b | b | b | b | b | b | b 8, a 1 |
| K10 land_pr | b | b | b | b | b | b | b | b | b | b 9 |
| K11 preflight | b | b | b | abs | b | a | b | b | b | b 7, a 1 |
| K12 review | b | b | a | b | b | a | b | b | b | b 7, a 2 |
| K13 ADR index | a | a | a | a | a | a | a | a | a | a 9 |
| K14 architecture docs | a | a | c | c | c | b | a | a | c | out of src 8–1; delete May docs 5–4 |
| K15 atlas | a | a | a | a | a | a | a | a | a | a 9 |
| K16 studio | b | a | b | b | b | b | a | b | b | b 7, a 2 |
| K17 show_web/trame | a | a | a | abs | a | abs | a | a | abs | a 6 |
| K18 interop | b | a | b | abs | b | abs | a | b | abs | b 4, a 2 |
| K19 ModelData | a | a | abs | abs | abs | abs | a | abs | abs | a 3 |
| K20 split= | a | a | a | abs | a | a | a | a | a | a 8 |
| K21 plotly/Three.js | a | a | a | abs | a | a | a | a | abs | a 7 |
| K22 comment diet | a | a | a | a | a | a | a | b | a | a 8, b 1 |
| K23 replay | b | b | b | b | a | a | b | a | b | b 6, a 3 |
| K24 staged live | b | a | a | a | b | b | abs | b | a | 4–4 |

## Appendix B: Consensus slate, as amended

- **S1.** Phase-0 fixes D1–D11, plus the #1184 flag hole and the tie re-diagnosis.
- **S2.** CI hygiene. `strict: true` is added as a maintainer decision (P5).
- **S3.** The fail-closed family-completeness gate plus the `getattr` lint, extended to names nothing defines (P7). The families list is exempt from S14 shadow expiry (P6).
- **S4.** The golden corpus plus `verify_move.py`, committed in W0 before *any* move (P1 conceded). ruff F821 on every codemod-touched package (P8). A canonical H5 dump (P9).
- **S5.** Re-key path-bound guards and set `inherited_members: true` before any move.
- **S6.** Loud replay and compose warnings.
- **S7.** A compatibility floor plus the additive-column rule.
- **S8.** `BackendInfo`, `FORK_PIN`, the artifact upload and a nightly `live-fork`, with three amendments:
  - `live-stock` pinned to Python 3.12 (P4);
  - fork-option effect tests (P7);
  - "green" counts `ladruno_mkl` skips (P3).
- **S9.** Delete the legacy window. `show_web` and trame are deleted, not ported (P2, K17). `export_animation`, `render` and `assess` are ported after golden stills (P7). P8 objected: delete first and port on demand.
- **S10.** `nav.py` plus the navigation paragraph (≤ 120 words).
- **S11.** Certain-tier deletions: 622 LOC, after P7 demoted the fixture merge to the likely tier.
- **S12.** Historical docs leave the normative surface; details in K14.
- **S13.** No DeckProgram, ColumnSpec or picture IR. K5(a) builds no new tree.
- **S14.** Kill criteria, plus KC1–KC6 for K5. Measured by the weekly triage.

## Appendix C: Process notes

**Moderator verification.** The moderator re-checked every correction before it entered the ballot:
- the 58% control;
- the 0.79% strut-and-tie gap;
- the 71% tie as tet10 in series, confirmed from the test's element types and the arithmetic;
- the #873 closed-unmerged PR, and the `ladruno` parser's "unknown tokens are ignored" and `tensStiffC = 500`;
- the fork's CI building `opensees.so` without an upload, with 0 releases and 0 tags;
- the drift in the fork's manifest;
- the star counts;
- the `live-stock` Python version.

**Infrastructure.** An API usage limit and repeated streaming stalls interrupted several panelists.
- Round-1 papers were recovered from files the panelists wrote, or extracted from their transcripts.
- Round 2 ran as fresh instances of the same seat and model. Each read its own round-1 paper, the packet, and the rival papers it had to answer.
- The positions stayed continuous; the contexts went from roughly 200–500k tokens down to 30–40k.

**Fork clone.** `origin/ladruno` in the local fork clone was fast-forwarded by three fetches on 2026-09-28, between 01:21 and 02:02.
- A transcript scan attributes none of them to any panelist, and all three predate the panel's first scratch files (02:34).
- The first fetch's flags match an IDE auto-fetch.
- The clone's HEAD, branch and uncommitted edits (dated 2026-06-18) are untouched.

**Papers.** All 19 documents are archived as comments on the closed issue **#1192**: the round-2 packet, plus nine round-1 and nine round-2 papers. The panel's prototype scripts are in `internal_docs/program/prototypes/`, which expires on 2026-11-30. The program that executes these decisions is described in `internal_docs/program/PROGRAM.md`, with the pinned board at #1203.
