# ADR 0100 — The partitioned-emit resident graph: closing the ledger term that ADR 0065 named and left

**Status:** ACCEPTED — incident measured 2026-08-24 (esmeralda, 60 GB build
node, diag job 145221 with a 5 s RSS sampler); candidate resident terms
identified in `apesees.py::_emit_partitioned` and its ndf-inference
prologue. **Gate G0 is now closed — see Amendment 1 (2026-08-25)**, which
records the measured attribution (G0a = 0.55–0.61 over the originally
authorised terms), **adds R8 to the R-table**, de-prices D5's `class_chunks`
half, and redefines G0b. The attribution below is the pre-measurement
*candidate* list; read it together with Amendment 1, which corrects it. Amends
**ADR 0065** (whose plan, `internal_docs/plan_emit_memory_columnar.md`,
closed with a remaining-ledger flag) and re-arms the **Route E** trigger,
which has now fired. Two adversarial reviews folded 2026-08-24 (§Review
log): the incident, mechanisms, and D0/D1 survived; the first draft's
attribution identity, two route variants, and its expected-effect
arithmetic did not, and are corrected below.

## Context — the incident

A capacity ladder on esmeralda (np=240, the f=10 Hz / P=6 plane-wave family,
`LadrunoBrick -lumped` so no nodal-mass records, no recorders,
`ops.tcl(per_rank=True, stream=True)`) walked global mesh size up from the
18.6 M-hex hero-run scale:

| rung | N_node | build (mesh→emit) | emit outcome |
|---|---|---|---|
| 18.6 M hex | 19.18 M | 1,384 s | clean |
| 26.1 M hex | 26.93 M | 1,937 s | clean |
| 36.5 M hex | 37.63 M | 2,727 s | clean |
| **51.0 M hex** | **52.62 M** | — | **OOM-killed mid-emit** |

The 51.0 M rung (157.8 M DOF) died on the 60,232 MB node with no diagnostic —
bash's bare `Killed`, the OOM-killer signature. An instrumented re-run
(memlog, 5 s cadence) gives the phase-resolved RSS trajectory:

- **Mesh + partition + topology:** bounded, peaking ~31 GB with give-back.
- **Session exit (`with` block closes → `gmsh.finalize()`):** RSS **drops
  31 → 18 GB** — the kernel's ~13 GB freed and pages returned. The emit that
  follows runs against FEMData only; the ADR 0065 caveat's "still-resident
  Gmsh kernel" hypothesis is *not* a term in this script's shape. (It
  becomes one for any script that emits inside the `with` block — see D0.)
- **`build()` + per-rank streaming emit:** monotonic climb 18 → 43 → 50 →
  **56.6 GB plateau** → final creep to **61.3 GB**, `MemAvailable` pinned
  under 1 GB for ~8 minutes, then the kill.

Total peak ≈ **1,200 B/hex of process RSS** (61.3×10⁹ / 51.0×10⁶); the
growth above the post-session 18 GB baseline is ~43 GB ≈ **~850 B/hex**.
The ceiling on a 60 GB node lands at ≈ 50 M hexes — between the rung that
passed and the rung that died, with a 5 s sampler's worth of precision and
no more.

## Context — the prior art, read honestly

Three prior claims meet the measurement:

1. `plan_emit_memory_columnar.md` §"What this does and does not do to RAM"
   projects post-B/C/A emit peak at "~100 B/node ⇒ … ~5 GB at 50M".
   Measured: 61.3 GB. That projection was contradicted a page above by the
   same document's own remaining-ledger line (~715 B/hex) — it was wrong
   within its own pages, not merely scope-limited.
2. The plan's header carries the open ticket: "**Remaining ledger term:** a
   pre-existing transient inside the emit loops themselves (~715 B/hex
   partitioned, **~613 flat** …) — the next slice if [it] still binds."
   The measured ~850 B/hex RSS growth is **consistent in magnitude** with
   that term binding at 51 M hexes — but the two numbers are different
   units (tracemalloc-traced bytes vs process RSS; the v2 plan's own M0 row
   carried "+ RSS ×~3"), and the flat-path ~613 B/hex figure says most of
   the term exists on paths where the partitioned containers (R2–R4 below)
   do not. **Which structures own the growth is exactly what G0 exists to
   establish; this ADR names the candidates and does not pre-announce the
   answer.**
3. The published **1,039 B/hex** came from
   `tests/benchmarks/emit_throughput_profile.py`, whose fixture *is*
   partitioned — so R1–R3 are already inside that traced figure. Its real
   blind spots: (a) it traces only build/emit/write, with session/mesh
   allocations deliberately out of scope (:250-257); (b) its RSS column is
   a Windows-only `PeakWorkingSetSize` — a **process-lifetime** peak that
   the meshing phase dominates, silently `0.0` on Linux (:203-244); (c) its
   `--mem-top` snapshot is taken *after* emit returns (:291-294), when the
   loop-local structures are already dead, so it can attribute none of
   them. The skill reference propagated the resulting overclaim as
   "constant-memory partitioned emit" (`references/opensees-bridge.md:815`),
   which the production path does not satisfy.

Exhaustive search (all 458 refs, 14 worktrees, 6 stashes, pickaxe on
`gmsh.finalize`/`gc.collect`/free/release) confirms **no fix exists
anywhere, in any state**. Separately, the numbering lane is exonerated:
`LadrunoParallelRCM` (fork ADR-74) measured N^0.99 at np=240 through 36.5 M
hexes — the numberer is no longer any part of the large-model ceiling.

## Root cause candidates — what stays resident through the per-rank streaming pass

All in `apesees.py` unless noted. There is no `del`, no `gc.collect()`, no
`.clear()` anywhere in the emit path; every structure below is built before
or during the pass and survives until the last rank block closes.
`_emit_partitioned` is shared by the **tcl, py, h5, live, and recording**
emitters — every change here touches all five.

| # | structure | site | scale |
|---|---|---|---|
| R1 | `node_idx_lookup` — plain `dict[int,int]` over all node ids; **built twice, co-resident, on staged partitioned decks** (`_emit_partitioned` + `_emit_stages_partitioned`) | :2806-2808 **and :3527-3529**; siblings on other paths: :2209 (`_emit_stages_flat`), :1962 (`_emit_split`) | ~84 B/node, **×2 staged** ≈ 8.8 GB at incident scale |
| R2 | `rank_owned_nodes` + `rank_primary_nodes` — `set[int]` per rank, all ranks materialized up front; **consumed after the rank loop** by `_emit_stages_partitioned` (:3733-3771) and passed as set params to regions/patterns emitters (:6273, :6390, :6486) | :3044-3047, :3055-3059 | ~41 B/node × 2 ≈ 4.3 GB (Σ over ranks exceeds N by the shared-boundary factor, which **grows with np**) |
| R3 | `plan_by_rank` — `select_rows` fancy-indexing **copies** eids + connectivity + tags for all ranks before the loop (`build.py:1810-1830`); consumed by the ADR 0099 hoist pre-pass (:2851), the rank loop (:3188), and `_emit_stages_partitioned` (:3750, :3853) | :2809-2812 | ~80 B/elem ≈ 4.1 GB (3rd connectivity copy; 1st = FEMData, 2nd = `_PG_FANOUT_CACHE`, `build.py:1624-1633`) |
| R4 | `model_mass_by_rank` — bucketed `mass_from_model` path re-materializes every transient `MassRecord` into per-rank lists | :3127-3132 | ~340 B/node on nodal-mass models (absent from this incident: `-lumped`) |
| R5 | `BuiltModel.fem` pins the FEMData snapshot (:833) — the floor. **Caveat:** `fem.nodes.ids` is an *object-dtype* ndarray (~36 B/node of boxed ints), not pure numpy; int64-ifying broker ids is a named out-of-scope candidate, bigger than it looks | :833 | ~100+ B/node |
| R6 | ndf inference — `inferred_ndf` (`build.py:504-507`) and `effective_ndf = {**inferred_ndf, **overlay}` (:1263): **two** full per-node dicts, both referenced through the whole emit (:1449) | build.py:504, apesees.py:1263 | ~2×84 B/node ≈ 8 GB at incident scale |
| R7 | `class_chunks` inside `infer_node_ndf` — a fresh boxed connectivity tuple per element (`build.py:420-424`), held until the function returns | build.py:420-424 | ~344 B/hex ≈ **15 GB transient** at 51 M; the likely bulk of the plan's *flat-path* ~613 B/hex term, and an arena-retention driver |

Honest ledger: R1–R3 (the partitioned containers) sum to ~17-20 GB of the
measured ~43 GB growth. R6+R7 plausibly cover much of the rest, with the
×~1.9 allocator/fragmentation multiplier already on ADR 0065's ledger
amplifying both. **The expected G0 outcome is therefore a split
attribution, not a single culprit** — the gates below are written for that.

R4 deserves its own sentence: `MassSet.__iter__` constructs transient
records precisely so nothing stays resident (`record_sets.py:885-902`), and
the flat and non-bucketed partitioned paths honor that — but the bucketed
path production uses collects all N anyway, silently defeating the
mechanism `mass_from_model` shipped to provide. Same defect class, distinct
from this incident. The next model shape will also meet
`_bucket_fix_targets_by_rank` / `_bucket_mass_targets_by_rank`
(:5117-5136), which build boxed per-rank id lists — flagged as R-class for
the follow-up, not routed here.

## Scoped solution routes

Lettering continues the v2 plan (A/B/C shipped, E deferred). Review finding:
D2, D3, and D4 share call surfaces (`:3411-3429`, the `:3440-3441`
signature, the `:3727-3752` stage loop) and are **not separable PRs**; they
ship as one slice, or behind a zero-behavior prep PR that gives
`_emit_stages_partitioned` a single context object. The plan picks the
former.

**D0 — contract, docs, and an honest instrument.** (the only known-wrong
item today; ships first, unblocked)
- Correct `opensees-bridge.md:815`: "constant-memory partitioned emit" →
  "constant deck-**text** memory; the build-side object graph still scales
  with N — see ADR 0100" (repo skill source + sync workflow, never the
  installed copy).
- State the session contract: on large models, close the session before
  `ops.tcl()`. The emit path never touches gmsh (verified); emitting inside
  the `with` block adds the kernel (~13 GB at 51 M hexes) to the peak. The
  partitioning guide's example already does this by accident
  (`guide_partitioning.md:126,187`) — make it an invariant instead of luck.
- Rebuild the bench instrument (this is design work, not a flag): a
  **mid-emit tracemalloc snapshot hook** (bench-side, on first
  `partition_open` — the current post-emit snapshot attributes nothing); a
  **phase-delta RSS sampler** that works on Windows *and* Linux
  (`/proc/self/status` VmRSS/VmHWM; no psutil dependency exists) replacing
  the lifetime `PeakWorkingSetSize`; release of the cross-size `last = (ops,
  sz)` retention (:417) that pins the previous FEMData + its fan-out cache
  through the next measurement; `ranks/` cleanup; ≥3 sizes and ≥2 np values
  (the R-terms are np-dependent); a `--mass from_model` arm so D1 has a
  gate; extrapolation targets updated to 51/71/100/139 M.

**D1 — columnar mass bucketing.** (correctness slice; exempt from the G0
ledger rule since its term is absent from the incident config)
Kill `model_mass_by_rank` (:3127-3132) with a **single-pass columnar index
bucketing** (bucket *indices*, 8 B/node, then emit each rank's records from
its index list in snapshot order — byte-identical, single consumer at
:3212). The naive per-rank rescan is **not a candidate**: np × N
`SortedIntToInt.get` calls ≈ 1.26×10¹⁰ at incident shape ≈ hours, plus as
many transient record constructions. Oracle:
`tests/opensees/integration/test_mass_from_model.py` (streamed vs
explicit-loop, full mass-line order, 4 partitions).

**D2 — columnar rank membership.** (ships with D3/D4)
Replace the R2 dict-of-sets pair with sorted `int64` arrays +
`np.searchsorted` membership — the treatment B2 gave `element_owner`.
**The build-per-rank/drop-at-close variant is dead**: `rank_owned_nodes` is
consumed per-stage *after* the base rank loop by `_emit_stages_partitioned`
and passed into the regions/patterns emitters as sets; lazy recompute would
also reintroduce the O(model × ranks) scan the ADR 0099 hoist measured as
dominant (:2802-2805). The columnar form must therefore serve those
consumers too (membership tests via searchsorted, set-algebra sites
rewritten on arrays).

**D3 — argsort node-index lookup.** (ships with D2/D4)
**Measured fact (review):** `fem.nodes.ids` is object-dtype and **unsorted
on every partitioned mesh** — `_fem_extract.py:117-119` takes
`gmsh.model.mesh.getNodes()` verbatim, which returns entity-grouped tags;
only unpartitioned meshes come out sorted. `searchsorted` on the raw array
is therefore wrong *and* slow (object comparisons). The design is:
materialize `ids_i64` + `order = argsort(ids_i64)` once (16 B/node vs the
dict's ~84), look up through the permutation, preserve missing-id detection
(`.get(nid) is None` is load-bearing at :3164, :2937). **No dict fallback**
— a fallback makes D3 a no-op on exactly the target configuration. Covers
all four sites: :2806, :3527 (co-resident staged pair), :2209, :1962.

**D4 — lazy per-rank element plan.** (ships with D2/D3 — the crux)
Demote the third connectivity copy from O(model) to O(model/np). The lazy
plan must serve **three consumers**: the ADR 0099 hoist pre-pass (:2851 —
needs per-rank counts for all ranks *before* the loop; derive counts from
`element_owner` without materializing rows), the base rank loop (:3188),
and `_emit_stages_partitioned` (:3750, :3853 — after every
`partition_close`, so "release at close" alone is insufficient; the staged
pass gets its own lazy builder). Tag pre-allocation is untouched.
`bucket_pre_allocated_by_rank` is already O(np·N) in *time*
(`build.py:7592-7599`); D4's bench row reports time as well as bytes so a
non-regression isn't misread as one.

**D5 — columnar ndf inference.** (new; routed from review findings R6/R7)
Stream `class_chunks` instead of retaining boxed tuples per element
(`build.py:420-424`), and columnarize `inferred_ndf`/`effective_ndf` (two
full per-node dicts held through emit). Priced **after G0 attributes** —
if R7's ~15 GB transient is confirmed as the flat-path term, this is the
largest single win in the program and may outrank D4.

**E — fork-side binary bulk loader.** (separate fork ADR — trigger FIRED)
The plan gates Route E on "(a) models pass ~30–50 M nodes" — the incident is
52.6 M nodes. E removes deck text entirely (`model.h5` → C++ bulk creation),
killing desktop formatting, the Tcl 2 GB `source` ceiling, and cluster parse
in one move. It stays out of scope here because the D-routes are bounded
Python changes against a measured target — but G0's decision rule (below)
names the outcomes that escalate to E.

### Rejected / dead variants

- **"Just emit on a bigger node."** Esmeralda has no bigger node; renting
  one leaves the defect. Stopgap, not fix.
- **E first, skip D.** Days before weeks; G0's decision rule re-prices E.
- **`gc.collect()` sprinkling.** The containers are referenced, not
  garbage. The fix is scoping, not collection.
- **D2/D4 drop-at-`partition_close`.** Dead on the staged partitioned
  path — the incident's own shape (see D2/D4 above).
- **D3 with a dict fallback.** Dead — the fallback fires on every
  partitioned mesh (see D3).

## Expected effect (bounded, not promised)

If only R1(×2)+R2+R3 die, the named removal is ~17-20 GB at incident scale
— a bound near **~520 B/hex**, which does *not* fit the 100 M-hex rung on a
60 GB node. Reaching further needs D5 (R6/R7) and/or the allocator
amplification tracking the removed terms down. The first draft's
"200–350 B/hex, well past 100 M hexes" is **withdrawn**: the honest
statement is that the post-D ceiling is G3's output, not this section's.

## Gates

> **Superseded in part by Amendment 1 (2026-08-25).** G0 is closed:
> G0a measured 0.55–0.61 over the terms listed here, **R8 was added to
> the numerator** afterwards, and **G0b is redefined as a slope ratio**
> (measured 0.88) — so the "G0b ≈ 2" precondition below was **not
> satisfied** when the branch was chosen. Read this section together
> with Amendment 1.

- **G0 (instrument first, two numbers):**
  - **G0a:** ≥ 80 % of the *traced* emit-phase peak growth attributed to
    R1–R4/R6/R7 by a snapshot taken **at the peak** (mid-emit hook), at ≥ 3
    sizes.
  - **G0b:** the RSS/traced ratio measured and reported at ≥ 2 sizes × ≥ 2
    np values.
  - **Decision rule:** G0a ≥ 80 % → run the D-slices as planned. 40 % ≤
    G0a < 80 % with G0b ≈ 2 → the named terms are real but
    allocator-amplified: ship the merged D2/D3/D4 slice (cheap, largest
    named terms), then **re-price D4-depth and D5 against Route E** before
    further spend. G0a < 40 % → the D program is mis-aimed; escalate to
    Route E with the G0 ledger attached.
- **G1 (byte identity, three named tiers):** (i)
  `tests/opensees/unit/test_emit_streaming_write.py` stream-vs-list —
  necessary, **not sufficient** (a change common to both modes passes it);
  (ii) `tests/benchmarks/test_emit_regression_gate.py::`
  `test_deck_lines_match_baseline_exactly` — the exact deck-shape pin on a
  4-rank partitioned cell; **`emit_gate_baseline.json` must not be
  re-baselined by any D-route PR**; (iii) the h5 + live partitioned parity
  suites (`tests/opensees/h5/test_h5_partitioned_staged_capture.py`,
  `tests/opensees/parity/test_live_parity.py`) — `_emit_partitioned` is
  shared by those emitters. Plus one new ≥ 3-rank mixed-npe parity fixture
  (bench-marked if it costs > 1 s). Emit *time*: the committed 1.7×
  emit/parse ratio gate stays green, and PR bodies report median-of-3 bench
  seconds at 2 sizes — a "10 %" budget sits inside the instrument's
  documented ~1.6× drift and is not a gate.
- **G2 (the incident is the test):** re-run the 51.0 M-hex rung on the same
  60 GB node. It completes, with the memlog trajectory attached to this ADR.
- **G3 (re-price the map):** rerun the ceiling arithmetic from measured
  post-D B/hex; resume the capacity ladder (71.3 M → 100 M → 139 M as far
  as the new ceiling allows); re-evaluate Route E triggers (b)/(c).

## Consequences

- The large-model ceiling story changes shape: after fork ADR-74 the
  binding constraint is single-node **emit-side residency**, now a scoped,
  measured backlog instead of a surprise OOM — with the honest caveat that
  the D-routes bound the named terms only, and the post-D ceiling is
  measured at G3, not promised here.
- `mass_from_model` becomes true on all three paths (D1), not two of three.
- The published memory claim gets a truthful successor: "deck text O(1);
  process peak ≈ FEMData + measured B/hex" with a cross-platform instrument
  behind it.
- Route E stops being hypothetical: trigger (a) is on the record, and G0's
  decision rule + G3 decide whether (b)/(c) join it.

## Review log (2026-08-24)

Two independent adversarial passes before any implementation: a
claims-skeptic on this ADR and a feasibility review on the companion plan
(`internal_docs/plan_emit_residency_d_routes.md`). What they killed, in
this document's first draft:

1. The Context section claimed the measured growth *was* the plan's ledger
   term ("root cause located") — a units conflation (traced vs RSS) that
   pre-announced G0's conclusion. Rewritten as consistency-in-magnitude
   with attribution owed to G0.
2. The R-table missed the ndf-inference terms (now R6/R7) — at ~8 GB +
   ~15 GB they rival or exceed every partitioned container, and the plan's
   own *flat-path* ~613 B/hex figure pointed at them all along. G0's
   original "≥80 % to R1–R4" bar was likely unpassable; the gate now
   includes them and carries a split-attribution decision rule.
3. R1 was undercounted: a fourth, **co-resident** dict in
   `_emit_stages_partitioned` (:3527) doubles it on staged partitioned
   decks; the first draft's sibling annotations were also mislabeled.
4. D2/D4's "drop at partition_close" variant is impossible (post-loop
   staged consumers; the ADR 0099 hoist); D3's "sorted in practice"
   premise is measurably false on every partitioned mesh (and the ids are
   object-dtype). Both routes redesigned; D2/D3/D4 merged into one slice.
5. D1's "measure the rescan" framing hid a 1.26×10¹⁰-call infeasibility;
   the columnar index form is now the design, and D1 is exempted from the
   G0 ledger rule (its term is absent from the incident config).
6. The expected-effect range (200–350 B/hex; "past 100 M hexes") did not
   close against the ADR's own table (~520 B/hex bound) and is withdrawn.
7. The bench critique was partly wrong in the first draft (the per-rank arm
   was a red herring — the traced figure already includes R1–R3); the real
   instrument defects are the post-emit snapshot timing, the
   Windows-lifetime RSS statistic, and cross-size retention — all now in
   D0's contract.
8. Unit inconsistency in the headline (1,260 vs 1,202 B/hex) corrected.

What survived both passes untouched: the incident record and memlog
inference (including the gmsh-kernel exoneration), the `select_rows`-copies
finding (the docstring calling it a view is wrong), R4's mechanism and
D1's byte-identity argument, the no-release greps, the guide/skill
citations, and the Route E trigger arithmetic.

## Amendment 1 (2026-08-25) — G0 measured; the R-table gains R8; D5 re-priced

Gate **G0 is closed.** D0 shipped in two slices — the docs half (#1068) and
the instrument half (#1071, `tests/benchmarks/emit_throughput_profile.py`) —
and the campaign ran 2026-08-24 on the desktop rig against the `box`
recipe, staged,
`--stream`, at four sizes × two np values.

This amendment is written **after** the measurement, and says so on purpose:
it changes the gate's own numerator, and that history has to stay legible
rather than be quietly folded into the original table.

### What G0 measured

**G0a = 0.55–0.61** over the terms this ADR authorised when the decision rule
was written (R1–R4, R6, R7), with the fan-out cache excluded. The gate
statistic is `G0a(conservative)` — Σterms ÷ *true counter-peak* growth —
admitted per cell by an error bound `|sampled − conservative| ≤ 0.02`.

| hexes | np | median G0a | vs cross-repeat max peak |
|---|---|---|---|
| 24,389 | 8 | 0.558–0.600 | 0.558 |
| 29,791 | 4 | 0.605 | 0.581–0.584 |
| 29,791 | 8 | 0.552 | 0.530–0.535 |
| 32,768 | 4 | 0.602 | 0.572 |
| 32,768 | 8 | 0.597 | 0.594 |
| 39,304 | 4, 8 | **discarded 0/6** | error 0.0364–0.0411 > 0.02 (deterministic) |

Where counter-peak growths disagree across repeats, the right-hand column is
the defensible reading: a counter peak is a **maximum**, so a lower-growth
repeat under-read and its G0a is biased **up**. That pulls the honest floor
to ~0.53. Note also what the bound does *not* certify — it tests
snapshot-vs-counter agreement, **not counter-vs-reality**.

One bookkeeping note, so the two tables below reconcile: the gate table
above divides by the *conservative* denominator (`cons_growth_b`, the true
counter-peak growth), while the dual-definition table divides by
`traced_growth_b` at the anchor. The same cell therefore reads 0.597 in one
and 0.600 in the other. Both are stated as measured; neither is a typo.

Per-term ledger at the largest clean cell (32,768 hexes, np 8, growth
31.74 MB):

| term | MB | % of growth |
|---|---|---|
| R2 `rank_owned_nodes` / `rank_primary_nodes` | 8.04 | 25 % |
| **R8** `ops_tag_to_fem_eid` | 6.49 | 20 % |
| R1 `node_idx_lookup` (incl. the staged twin) | 4.62 | 15 % |
| R6 ndf dicts | 3.76 | 12 % |
| R3 `plan_by_rank` | 2.63 | 8 % |
| `_PG_FANOUT_CACHE` | 2.36 | 7 % |
| R7 `class_chunks` | **0.00** | **0 %** |
| R4 `model_mass_by_rank` | 0.00 | 0 % (8.9 MB under `--mass from_model`) |
| unattributed | ~4.0 | 13 % |

### R8 — a new authorised term

`ops_tag_to_fem_eid` (`apesees.py:3674-3677`, filled via `build.py:7221`) is
a full per-element **reverse tag map** built inside
`_emit_stages_partitioned`. Measured across cells at **103–228 B/elem
(~189 average over gate-ok entries)**, i.e. **~5–12 GB at the incident's
51.0 M elements** — larger than several terms this ADR already names, and
absent from the original R-table only because the pre-measurement inventory
missed it. The rate is given as a range because it genuinely varies with np
and mesh size; no single-point figure is defensible from the campaign data.

**Instrument inconsistency, named rather than left standing:** the merged
bench (`emit_throughput_profile.py:337`, `:1202-1206`) still prints
"~128 B/elem ≈ 6.5 GB" for R8. That string is a **stale pre-measurement
estimate** — it predates the campaign and is not supported by any artifact.
A docs amendment cannot change code, so its one-line correction is folded
into **P3's scope**.

It is **added to the R-table and to G0a's numerator.** The effect is the
whole reason this amendment carries a date:

| numerator | G0a | branch under §Gates |
|---|---|---|
| as originally written (R1–R4/R6/R7) | **0.600** | middle |
| + `_PG_FANOUT_CACHE` | 0.675 | middle |
| **+ R8** | **0.805** | **proceed as planned** |
| + both | 0.879 | proceed |

This table is computed from the size-33 / np-8 cell's measured per-term
bytes and is **unaffected by the rate question above** — the rate is a
summary statistic over cells; the branch arithmetic uses one cell's bytes
directly.

**~88 % of emit growth is attributable to identified structures; only ~60 %
to the originally authorised ones.** The branch therefore turned on the
ledger's completeness, not on the measurement. The campaign deliberately
reported the gate over the *original* set and surfaced this table separately,
so that widening the numerator was an explicit decision taken with the
numbers in view — not an instrument quietly grading its own homework.

**Provenance, because this document must not appear to grade itself:** the
decision to authorise R8 and to select the branch was taken by the **project
owner on 2026-08-25**, with the dual-definition table above in view — the
owner selected "Amend + hybrid" from the options put to them. The ruling is
recorded on the amendment's own pull request:
<https://github.com/nmorabowen/apeGmsh/pull/1076#issuecomment-5405842105>.
That record was written by the driving session **on the owner's behalf**,
capturing an interactive decision — it is not a GitHub review by the owner's
account, and should not be read as one. This amendment *records* the ruling;
it does not make it. The measuring party proposed both readings and did not
choose between them.

### `_PG_FANOUT_CACHE` — identified, measured, still not routed

Measured at **72 B/elem** (exact to three digits at three sizes), 7 % of
growth. The plan's risk #5 scoped it out of every D-route as a deliberate
speed cache; that decision **stands**, but it is now a measured quantity
rather than an assumption, and it is recorded here so a future route can
price it.

It is **not** in G0a's numerator. It earns its own line because R7's original
span silently absorbed it: `build.py:423` sits inside the `class_chunks`
block and calls `expand_spec_to_elements`, whose callee allocates the cache
arrays at `build.py:1876`/`:1879`. Attribution that did not separate the two
inflated G0a by ~0.075 at every cell.

### R7 collapses at the peak — D5 is re-priced

**R7 measures 0.00–0.19 MB at the resident peak in every cell.** The
~344 B/hex transient this ADR estimates is real, but it lives at
`infer_node_ndf` time — early in emit — and is gone before the peak.

This ADR speculated that R7 "may be the largest single win in the program"
and might outrank D4. At the peak it contributes **nothing**. Accordingly:

- **D5's `class_chunks` half is de-priced** and drops out of the priced
  program unless G2 shows otherwise.
- **D5's ndf-dict half (R6, 3.76 MB, 12 %) stays a candidate** — that term is
  real at the peak.

### R1 — site-name correction

The R1 row lists four lookup sites. The fourth (`_emit_split`) is spelled
**`node_idx`**, not `node_idx_lookup` — `apesees.py:1962` reads
`node_idx = {int(nid): i for i, nid in enumerate(self.fem.nodes.ids)}`. The
line number was right; the name was not. D3 must cover it under the correct
spelling.

Separately, R1's **×2 co-residency on staged partitioned decks is confirmed
deterministically** — R1 exactly doubles at every repeat and every size, at a
hook firing after the twin's construction. It is *not* the driver of
run-to-run G0a variance, but it **is** a first-order contributor at
**~24–30 %** of the numerator across measured cells. Both halves of that
sentence are load-bearing.

### G0b — redefined, and its precondition fails at bench scale

As written, G0b was a per-cell RSS/traced ratio. That is not a usable
estimator at bench sizes: RSS carries large size-independent components
(interpreter, numpy, gmsh, allocator arenas) that swamp tens of MB of traced
growth, and measured values spanned 0.68–87.35 — including values below 1,
which are unphysical for the quantity intended.

**G0b is redefined as a ratio of slopes**: regress RSS growth and traced
growth on element count across ≥3 sizes and take `slope_rss / slope_traced`,
reporting R² and the points used. ADR 0065's ×~1.9 allocator multiplier was
always a slope, not a level.

Measured: **G0b(slope) = 0.88** (R² 0.996/0.999), paired `--mem` ×
`--rss-only` at a 5 ms poll.

0.88 is **below 1, and well below the `≈2`** this ADR's middle branch assumes.
The explanation is physical, not instrumental: the mesh phase leaves an
**obmalloc reservoir** that absorbs emit allocations at bench scale, so RSS
grows more slowly than traced memory. At incident scale there is no reservoir
and there *is* real memory pressure — the regime the ~1.9 came from. The
caveat cuts both ways, and it means **the middle branch's "G0b ≈ 2"
precondition was not satisfied when the branch was evaluated.** The decision
was taken on G0a alone, and readers of §Gates should treat that precondition
as unverified at desktop scale rather than as met.

### Standing extrapolation caveat

Every measured cell is **~1,300× below the 51 M-hex incident**. An early
hypothesis that G0a rises monotonically with size toward ~0.76–0.78 was
**retracted on fresh data** (a 6,859-hex cell scored 0.678, above a
13,824-hex cell's 0.513 — both from the pre-gate instrument generation; the
mature gated run measures that same 13,824-hex cell at 0.552, so the
direction of the retraction survives the instrument change). There is
therefore **no established basis for
extrapolating G0a to incident scale**, and the largest size attempted
(39,304 hexes) fails deterministically under the error bound. G2/G3 remain
the only measurements that speak to the real ceiling.

### Consequences for the routes

- **P3 proceeds** as the merged D3+D2+D4 slice. R2 is confirmed the largest
  authorised term at every cell (25 %), so the plan's ordering holds.
- **R8 joins the routed program** — same surgical class as D2/D3's targets
  (another `_emit_stages_partitioned` dict).
- **D5** is halved: R6 stays a candidate, `class_chunks` is de-priced.
- **Route E stays deferred.** Its trigger record stands; G2/G3 measure the
  post-fix ceiling that re-prices it.
- The merged instrument is **P3's before/after oracle**: a post-P3 campaign
  must show R1/R2/R3 (and R8) collapse. That is G0's re-run gate.

**Provenance of the numbers above:** every figure comes from the JSON
records the instrument emits (`--json`), not from a checked-in driver
script. The campaign was run as ad hoc invocations of
`emit_throughput_profile.py`; the per-cell JSON records — which carry
`per_term_bytes`, `peak_counter_b`, the gate verdict and the discard reason
— are the authoritative provenance.
