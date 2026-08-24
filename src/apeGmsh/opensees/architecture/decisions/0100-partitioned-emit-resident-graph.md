# ADR 0100 — The partitioned-emit resident graph: closing the ledger term that ADR 0065 named and left

**Status:** PROPOSED — incident measured 2026-08-24 (esmeralda, 60 GB build
node, diag job 145221 with a 5 s RSS sampler); candidate resident terms
identified in `apesees.py::_emit_partitioned` and its ndf-inference
prologue; **attribution is owed to gate G0, not asserted here**. Amends
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
