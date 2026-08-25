# Plan — ADR 0100 D-routes: partitioned-emit resident graph

**Status:** REVIEWED (2026-08-24) — two adversarial passes folded (a
claims-skeptic on the ADR, a feasibility review on this plan; see the ADR's
§Review log). The review re-ordered the slices, re-cut P1, re-tiered P0,
re-priced three estimates by >2×, and added risks 10–20. Companion to
`src/apeGmsh/opensees/architecture/decisions/0100-partitioned-emit-resident-graph.md`.

Token-aware staffing: cheap models for mechanical edits, the strong model
for the instrument design and the merged code slice, skeptics on everything
that merges. The **decision point after P0** is the budget's main control:
the expensive slices are never bought on an unmeasured premise, and the
review's expected G0 outcome (split attribution, ~40–50 % to the named
partitioned containers) has its own branch instead of stalling the program.

## Sequencing and staffing (post-review order)

| slice | what | model · effort | est. tokens | gate |
|---|---|---|---|---|
| **P1 — D0 docs** (was P4; promoted: the only item *known* wrong today) | Fix `skills/apegmsh/references/opensees-bridge.md:815` ("constant-memory partitioned emit" → "constant deck-text memory; build-side graph still scales with N — see ADR 0100") via repo skill source + sync script, never `~/.claude`; session-close contract into `guide_partitioning` + bridge guide; CHANGELOG | Haiku · low | ~30–50k | text lands; no measurement dependency |
| **P2 — D0 instrument** (was P0; re-tiered — it hides the plan's hardest design question: attributing a peak that lives only inside a frame) | Mid-emit tracemalloc snapshot hook (bench-side, first `partition_open`); cross-platform phase-delta RSS sampler (`/proc/self/status` VmRSS/VmHWM + Windows working-set; no psutil dep); release the cross-size `last=(ops,sz)` retention (:417); `ranks/` cleanup; `--per-rank` arm (`stream_to(path, per_rank=True)` — fixture already partitions, the instrument is the work); `--mass from_model` arm; ≥3 sizes × ≥2 np; extrapolation targets 51/71/100/139 M; results table checked in | Fable · high (instrument design) + Sonnet · medium (mechanical flags) | ~150–250k | **G0a/G0b measured** |
| — | **DECISION POINT (three branches, ADR §Gates):** G0a ≥80 % → proceed as planned · 40–80 % with RSS/traced ≈2 → ship P3, then re-price D4-depth/D5 vs Route E · <40 % → escalate to Route E with the ledger attached | main session | — | — |
| **P3 — merged code slice D3+D2+D4** (review: not separable — shared call site :3411-3429, signature :3440-3441, stage loop :3727-3752) | argsort node lookup (all 4 sites incl. the co-resident :3527; **no dict fallback** — ids are object-dtype and unsorted on every partitioned mesh); columnar rank membership serving the post-loop staged/regions/patterns consumers; lazy per-rank plan serving its three consumers (hoist counts derived without rows, base loop, staged pass) | Fable · high (impl) → 2-refuter panel, Opus · high ×2 (lens 1: byte-identity/ordering across all G1 tiers; lens 2: memory-actually-freed, RSS not just traced) | ~400–500k | G1 all three tiers + bench rows (bytes **and** time) |
| **P4 — D1 columnar mass bucketing** (correctness slice; exempt from the G0 ledger rule — its term is absent from the incident config) | Single-pass columnar **index** bucketing (the naive per-rank rescan is 1.26×10¹⁰ `SortedIntToInt.get` calls at incident shape — not a candidate); oracle: `tests/opensees/integration/test_mass_from_model.py` | Sonnet · medium → Opus · high probe (MP-additive semantics, ordering) | ~40–60k | G1 tiers i+ii incl. mass-equivalence |
| **P5 — D5 columnar ndf inference** (new route; priced by G0) | Stream `class_chunks` (build.py:420-424, the ~15 GB transient); columnarize `inferred_ndf`/`effective_ndf` | tier set after G0 (if R7 confirms as the flat-path term, this may outrank D4's payoff) | ~150–250k (provisional) | G0 re-run shows R6/R7 gone |
| **P6 — G2/G3 cluster validation** (re-priced ×3: risk #8's node failures + dead `sacct` are measured facts of this campaign) | Re-run the 51.0 M rung with memlog on the 60 GB node; resume ladder as far as the new ceiling allows; re-price Route E triggers (b)/(c); results into the ADR + Obsidian note | main session (SSH orchestration) | ~150–250k | **G2 completes**; G3 arithmetic |

Rough total: **~950k–1.4M tokens**; everything from P3 down (~75 % of the
spend) sits behind the decision point.

## Slice contracts

- **P1**: docs only; the skill edit goes through the repo skill source +
  sync workflow. No code.
- **P2**: bench + harness only. The mid-emit snapshot hook is the one
  permitted exception to "no library changes" **if** it cannot be done by
  bench-side monkeypatch — prefer the monkeypatch. Deliverable is the
  checked-in attribution table; G0a/G0b are its two numbers, decimal GB
  throughout.
- **P3**: one PR over `_emit_partitioned` + `_emit_stages_partitioned` (+
  the regions/patterns emitter signatures R2 flows through). The flat/split
  sibling sites (:2209, :1962) ride along only for D3 (same lookup, same
  design); any other flat-path change is out of scope. Contract includes:
  missing-id semantics preserved (`.get() is None` is load-bearing at
  :3164, :2937); ADR 0099 hoist gate re-derived from counts, not
  materialized rows; `emit_gate_baseline.json` **not re-baselined**; bench
  rows for bytes and time at 2 sizes, median-of-3.
- **P4**: bucketed path only; flat and non-bucketed paths already stream
  and must not churn; PR body carries the byte-oracle run and the perf row.
- **P5**: scoped after G0; its contract is written then, against the
  measured R6/R7 split.
- **P6**: memlog + collect table appended to the ADR as an implementation
  note; `--no-requeue` on every validation job; assert on artifacts, not rc.

## Risks

1. **Attribution shortfall.** The named partitioned containers likely
   explain ~40–50 % of the growth, with allocator amplification (~×1.9 on
   ADR 0065's ledger) and the R6/R7 ndf terms carrying the rest. Handled
   structurally: the decision point has a branch for exactly this outcome.
2. **Byte-identity breaks.** argsort indirection and lazy slicing must
   preserve emission order and missing-id semantics exactly. Mitigation:
   G1's three tiers; refuter lens 1.
3. **Speed regressions.** searchsorted vs dict in hot loops; O(np·N)
   bucketing time already existed in the eager
   `bucket_pre_allocated_by_rank` (replaced by `LazyRankBuckets`) — report
   time so a non-regression isn't misread. The old "10 % emit-time budget"
   is replaced by the committed 1.7× ratio gate + median-of-3 reporting
   (10 % sits inside the instrument's documented ~1.6× drift).
4. **Memory not actually freed.** Dropping references ≠ returning pages
   (CPython arenas, object-dtype boxes). Refuter lens 2; the RSS sampler
   measures the real statistic.
5. **`_PG_FANOUT_CACHE` temptation.** Connectivity copy #2 is a deliberate
   speed cache (#876); no D-route touches it in v1 — reviews confirmed the
   scoping.
6. **Hot-file collisions.** `apesees.py` is touched by concurrent sessions
   and stash@{4}. Mitigation: the merged P3 is one rebase surface instead
   of three colliding PRs; rebase before each push; never `git add -A`.
7. **Sibling-path drift.** D3 covers all four lookup sites; R2/R3 fixes are
   partitioned-path only — stated in P3's PR body, not silent.
8. **Cluster flakiness for G2/G3.** Node failure mid-job and dead `sacct`
   both measured this campaign; P6 is priced accordingly.
9. **Estimate risk on the crux.** P3 carries the widest error bars
   (400–500k); the decision point and the two-refuter panel are its
   controls.
10. **`fem.nodes.ids` is unsorted and object-dtype** (measured: parts=4/16
    both invert at position 8; `_fem_extract.py:117-119` never sorts).
    Kills any dict-fallback or raw-searchsorted D3 design. Mitigation:
    argsort indirection, no fallback.
11. **The old instrument measures the wrong things.** `--mem-top` snapshots
    after emit returns (terms already dead); `_rss_peak_mb` returns
    process-lifetime peak dominated by meshing. P2 replaces both.
12. **Windows/Linux asymmetry.** The RSS helper is ctypes-Windows inside a
    blanket except → silent `0.0` on Linux; CI (the only Linux oracle)
    cannot run the arm until P2's sampler lands. G0 (Windows) and G2
    (Linux) must report the same statistic definition.
13. **Cross-size retention corrupts the slope.** `last=(ops,sz)` pins the
    previous FEMData + fan-out cache through the next measurement; P2
    releases it (or one size per process).
14. **R-terms are np-dependent.** Shared-boundary duplication grows with
    np; bucketing time is O(np·N). Desktop `--parts 16` under-reports
    np=240 — hence G0b's ≥2-np requirement.
15. **Parity at meaningful size costs suite runtime.** Today's partitioned
    parity fixtures are 2–4 elements / 2 ranks; the new ≥3-rank mixed-npe
    fixture is bench-marked or size-capped.
16. **Shared emitters.** `_emit_partitioned` serves tcl, py, h5, live, and
    recording emitters — G1 tier iii exists because every D-route silently
    changes the archival and in-process paths.
17. **Unlisted R-class terms.** `_bucket_fix_targets_by_rank` /
    `_bucket_mass_targets_by_rank` (:5117-5136) build boxed per-rank id
    lists — absent from this incident, waiting in the next model shape.
    Flagged in the ADR; not routed in v1.
18. **R5's floor isn't numpy.** Object-dtype node ids ≈ 36 B/node of boxed
    ints inside the "legitimate floor"; int64-ifying broker ids is a named
    out-of-scope candidate so it is a decision, not an oversight.
19. **G0 lands in the split-attribution branch** (the review's own
    arithmetic says ~40 %). Not a failure mode — the expected path; the
    decision rule and P5 exist for it.
20. **Route E opportunity cost.** If P2's numbers say the allocator owns
    the remainder, deeper D-work has a ceiling no Python change moves —
    the decision rule escalates rather than grinding.
