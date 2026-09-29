# ADR 0100 — The partitioned-emit resident graph: closing the ledger term that ADR 0065 named and left (summary)

**Status:** ACCEPTED — incident measured 2026-08-24 (esmeralda, 60 GB build node, diag job 145221
with a 5 s RSS sampler); candidate resident terms identified in `apesees.py::_emit_partitioned`
and its ndf-inference prologue. **Gate G0 is closed (Amendment 1) and route P3 has shipped
(Amendment 2, #1077) — read both before acting on anything below.** […]

## Decision

The incident: the 51.0 M-hex rung (52.6 M nodes, np=240) OOM-killed mid-emit. There is no
`del`, `gc.collect()` or `.clear()` in the emit path; build-side structures in
`_emit_partitioned` (shared by the tcl, py, h5, live and recording emitters) stay resident
until the last rank block closes. Candidate terms: **R1** `node_idx_lookup` (built twice on
staged decks), **R2** `rank_owned_nodes` + `rank_primary_nodes`, **R3** `plan_by_rank`
(a third connectivity copy), **R4** `model_mass_by_rank`, **R5** the pinned FEMData (floor),
**R6** `inferred_ndf` / `effective_ndf`, **R7** `class_chunks` in `infer_node_ndf`. Amends
ADR 0065 and re-arms its Route E trigger.

Routes (D2, D3 and D4 share call surfaces and ship as one slice):
- **D0** contract, docs and an honest instrument: "constant deck-text memory", close the
  session before `ops.tcl()` on large models, a mid-emit tracemalloc hook and a
  cross-platform phase-delta RSS sampler.
- **D1** columnar mass bucketing (index buckets, byte-identical output).
- **D2** columnar rank membership (sorted `int64` + `np.searchsorted`); drop-at-close is dead.
- **D3** argsort node-index lookup (`fem.nodes.ids` is unsorted on partitioned meshes);
  missing-id detection preserved; no dict fallback.
- **D4** lazy per-rank element plan serving the ADR 0099 hoist, the rank loop and
  `_emit_stages_partitioned`.
- **D5** columnar ndf inference (stream `class_chunks`, columnarize the ndf dicts).
- **E** fork-side binary bulk loader: a separate fork ADR, trigger fired, out of scope here.
Rejected: a bigger node, E first, `gc.collect()` sprinkling, drop-at-`partition_close`, D3
with a dict fallback. The post-D ceiling is G3's output, not a promise.

Gates: **G0** instrument first (G0a ≥ 80 % of traced peak attributed; G0b RSS/traced ratio;
a decision rule that escalates to Route E below 40 %). **G1** byte identity in three tiers:
`test_emit_streaming_write.py`, `test_deck_lines_match_baseline_exactly` (and
`emit_gate_baseline.json` must not be re-baselined by any D-route PR), and the h5 + live
partitioned parity suites. **G2** re-run the 51.0 M rung. **G3** re-price the ceiling and
resume the capacity ladder.

## Amendments

- 2026-08-25, A1: G0 closed (D0 shipped in #1068 and #1071); G0a measured 0.55–0.61; **R8**
  `ops_tag_to_fem_eid` added (~103–228 B/elem); G0b redefined as a slope ratio (0.88), so
  the "G0b ≈ 2" precondition was not met; P3 proceeds as the merged D3+D2+D4 slice with R8.
- 2026-08-25, A2: P3 shipped (#1077); traced peak −48/−54 %, RSS ≈ −40 % at bench scale.
  R7 is the binding peak term (~370 B/hex), so D5 is re-priced UP and is necessary; P3 alone
  does not close the incident. RSS, not traced bytes, is the OOM-relevant statistic.
- 2026-08-25/26, A3: G2/G3 closed. The 51.0 M rung now builds and emits, but with under 2 %
  RSS improvement; the 71.3 M rung OOM-kills before `get_fem_data()` completes. The
  single-node ceiling has not moved an order of magnitude; Route E is re-priced up.
  Follow-ups: a phase marker after `get_fem_data()`, the G2 solve, and D5 checked at cluster
  scale.
- 2026-08-26, A4: the 51.0 M solve-based emission-correctness check PASSES (job 145623,
  240/240 ranks) after a PATH-bug false start (job 145609). Follow-ups 1 and 3 remain open.

Full text: [../0100-partitioned-emit-resident-graph.md](../0100-partitioned-emit-resident-graph.md).
Precedence (AGENTS.md): the code wins over the ADR, and the ADR wins over this summary.
