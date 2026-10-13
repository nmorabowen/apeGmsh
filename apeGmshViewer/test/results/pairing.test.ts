// Pairing a results file with the open model (V4b, #1728), by content: the
// rule of h5-schema.md (`model_hash`, else `fem_hash`), never the name or the
// path. Each state has its case, and only `paired` puts the results stages on
// the phase axis. The staged fixture pairs by model_hash through the real
// readers (main/zones.ts for the model, main/results.ts for the results).

import assert from "node:assert/strict";
import { test } from "node:test";
import { pairResults } from "../../src/state/results.ts";
import { noticesOf, resultsPairing } from "../../src/state/selectors.ts";
import { initialState, reduce, type State } from "../../src/state/store.ts";
import type { Lineage, ResultsInfo } from "../../src/state/types.ts";
import { stagedModelLoad, stagedResults, stagedState } from "./staged.ts";

const H = (c: string) => c.repeat(32);
const lin = (fem: string | null, model: string | null, results: string | null = H("r")): Lineage => ({ femHash: fem, modelHash: model, resultsHash: results });
const res = (lineage: Lineage | null, openseesVersion: string | null = null): ResultsInfo => ({
  path: "x.results.h5",
  version: "1.1.0",
  sourceType: "domain_capture",
  lineage,
  partial: null,
  stages: [],
  unavailableZones: [],
  openseesVersion,
});
/** The open model as pairResults takes it: its lineage and its `/meta@opensees_schema_version`. */
const mod = (lineage: Lineage | null, openseesVersion: string | null = null) => ({ lineage, openseesVersion });

test("paired by model_hash when the results carry one; by fem_hash only when they do not", () => {
  assert.deepEqual(pairResults(mod(lin(H("a"), H("b"))), res(lin(H("a"), H("b")))), { state: "paired", by: "model_hash" });
  assert.deepEqual(pairResults(mod(lin(H("a"), H("b"))), res(lin(H("a"), null))), { state: "paired", by: "fem_hash" });
});

test("fem_hash fall-through: a model with no model_hash still pairs fem-only results; a model with no fem_hash refuses them", () => {
  assert.deepEqual(pairResults(mod(lin(H("a"), null)), res(lin(H("a"), null))), { state: "paired", by: "fem_hash" });
  const none = pairResults(mod(lin(null, null)), res(lin(H("a"), null)));
  assert.equal(none.state, "mismatch");
  assert.match((none as { reason: string }).reason, /the results name a fem_hash, and the open model has none to compare/);
  const noLineage = pairResults(mod(null), res(lin(H("a"), null)));
  assert.match((noLineage as { reason: string }).reason, /has none to compare/);
});

test("mismatch: another model_hash refuses even when the mesh (fem_hash) is the same", () => {
  const p = pairResults(mod(lin(H("a"), H("b"))), res(lin(H("a"), H("c"))));
  assert.equal(p.state, "mismatch");
  assert.match((p as { reason: string }).reason, /model_hash differs.*results cccccccc…, model bbbbbbbb…/);
  assert.equal(pairResults(mod(lin(H("a"), H("b"))), res(lin(H("d"), null))).state, "mismatch");
  assert.equal(pairResults(mod(null), res(lin(H("a"), H("b")))).state, "mismatch", "a model with no lineage pairs with nothing");
});

test("noModel, unpairable and none", () => {
  assert.deepEqual(pairResults(null, res(lin(H("a"), H("b")))), { state: "noModel" });
  assert.equal(pairResults(mod(lin(H("a"), H("b"))), res(null)).state, "unpairable");
  assert.equal(pairResults(mod(lin(H("a"), H("b"))), res(lin(null, null))).state, "unpairable");
  assert.deepEqual(pairResults(mod(lin(H("a"), H("b"))), null), { state: "none" });
});

test("the staged fixture pairs by model_hash through the real readers; its stages go on the phase axis", async () => {
  const s = await stagedState();
  assert.deepEqual(resultsPairing(s), { state: "paired", by: "model_hash" });
  assert.deepEqual(
    s.phase.axis.map((k) => (k.kind === "results" ? `results ${k.stage}` : k.kind)),
    ["mesh", "results 0", "results 1", "results 2", "results 3"],
  );
  assert.deepEqual(s.phase.at, { kind: "mesh" }, "opening the pair stays on the mesh");
  assert.deepEqual(noticesOf(s), []);
});

test("a results file of another model: no results on the axis, and a banner that says why", async () => {
  const r = await stagedResults();
  const other = { ...r.info, lineage: { ...r.info.lineage!, modelHash: H("e") } };
  let s = reduce(await stagedState(), { type: "resultsLoaded", ...r, info: other });
  assert.equal(resultsPairing(s).state, "mismatch");
  assert.equal(s.phase.axis.some((k) => k.kind === "results"), false);
  assert.match(noticesOf(s).join("\n"), /^Results not drawn: staged_activation\.results\.h5 does not belong to this model \(model_hash differs/m);
  assert.throws(() => reduce(s, { type: "setPhase", at: { kind: "results", stage: 0, step: 0 } }), /not on the phase axis/);
  // The same results re-read with the model's hash: paired again (re-pair on reload).
  s = reduce(s, { type: "resultsLoaded", ...r });
  assert.equal(resultsPairing(s).state, "paired");
});

test("a results file with no model open: not drawn, and the banner says the embedded model is not drawn", async () => {
  const r = await stagedResults();
  let s: State = reduce(initialState, { type: "fileOpened", artifact: "results", path: r.info.path });
  s = reduce(s, { type: "resultsLoaded", ...r });
  assert.deepEqual(resultsPairing(s), { state: "noModel" });
  assert.deepEqual(s.phase.axis, []);
  assert.match(noticesOf(s)[0]!, /no model is open for staged_activation\.results\.h5\. Its embedded \/model is not drawn/);
  // The model arrives: the pair forms.
  s = reduce(s, { type: "fileLoaded", artifact: "model", load: await stagedModelLoad() });
  assert.equal(resultsPairing(s).state, "paired");
});

test("a model that fails or closes keeps the results read; they re-pair when a model loads", async () => {
  const s = await stagedState();
  const failed = reduce(s, { type: "fileFailed", artifact: "model", path: s.artifacts.model!.path, error: "boom" });
  assert.notEqual(failed.results, null);
  assert.equal(resultsPairing(failed).state, "noModel");
  const closed = reduce(s, { type: "fileClosed", artifact: "results" });
  assert.equal(closed.results, null);
  assert.equal(closed.phase.axis.some((k) => k.kind === "results"), false);
});

// V4h (#1856): at opensees 2.31.0 every bridge archive's model_hash changes
// once (declaration ids entered the hashed /opensees/program column). A
// mismatch whose two sides straddle that boundary names the cause; it stays a
// refusal, never a pairing, never a fall-back to fem_hash. Any other mismatch
// reads exactly as before.
const plain = (rv: string, mv: string) => `model_hash differs: the results were written from another model (results ${rv}…, model ${mv}…)`;
const reasonOf = (p: ReturnType<typeof pairResults>): string => {
  assert.equal(p.state, "mismatch");
  return (p as { reason: string }).reason;
};

test("V4h: a 2.30 results file against a re-saved 2.31 model is refused with the 2.31.0 hash change named", () => {
  const reason = reasonOf(pairResults(mod(lin(H("a"), H("b")), "2.31.0"), res(lin(H("a"), H("c")), "2.30.0")));
  assert.match(
    reason,
    /^model_hash differs: the opensees schema 2\.31\.0 hash change \(declaration ids entered the hash; results opensees 2\.30\.0, model opensees 2\.31\.0\); re-run the analysis against the re-saved model/,
  );
  assert.match(reason, /\(results cccccccc…, model bbbbbbbb…\)$/);
});

test("V4h: the reverse direction (2.31 results, 2.30 model) names the cause with the other remedy", () => {
  const reason = reasonOf(pairResults(mod(lin(H("a"), H("b")), "2.30.0"), res(lin(H("a"), H("c")), "2.31.0")));
  assert.match(reason, /the opensees schema 2\.31\.0 hash change .*results opensees 2\.31\.0, model opensees 2\.30\.0\); open the results with the model\.h5 they were run on/);
});

test("V4h: a same-era mismatch keeps today's message exactly", () => {
  const pairs = [["2.30.0", "2.30.0"], ["2.31.0", "2.31.0"], ["2.27.0", "2.30.9"], ["2.31.0", "2.32.0"], ["3.0.0", "2.31"]] as const;
  for (const [r, m] of pairs) {
    assert.equal(reasonOf(pairResults(mod(lin(H("a"), H("b")), m), res(lin(H("a"), H("c")), r))), plain("cccccccc", "bbbbbbbb"), `results ${r}, model ${m}`);
  }
});

test("V4h: the boundary is 2.31 itself (2.30.9 | 2.31.0, and 2.31 without a patch)", () => {
  assert.match(reasonOf(pairResults(mod(lin(H("a"), H("b")), "2.31.0"), res(lin(H("a"), H("c")), "2.30.9"))), /2\.31\.0 hash change/);
  assert.match(reasonOf(pairResults(mod(lin(H("a"), H("b")), "2.31"), res(lin(H("a"), H("c")), "2.30"))), /2\.31\.0 hash change/);
});

test("V4h: an absent (or unreadable) version on either side keeps today's message", () => {
  const pairs = [[null, "2.31.0"], ["2.30.0", null], [null, null], ["2.30.0", "two"]] as const;
  for (const [r, m] of pairs) {
    assert.equal(reasonOf(pairResults(mod(lin(H("a"), H("b")), m), res(lin(H("a"), H("c")), r))), plain("cccccccc", "bbbbbbbb"), `results ${r}, model ${m}`);
  }
});

test("V4h: a matching hash across the boundary still pairs; a cross-era mismatch never falls back to fem_hash", () => {
  assert.deepEqual(pairResults(mod(lin(H("a"), H("b")), "2.31.0"), res(lin(H("a"), H("b")), "2.30.0")), { state: "paired", by: "model_hash" });
  // Same mesh (fem_hash a), another model_hash across the boundary: refused, not paired by fem_hash.
  assert.equal(pairResults(mod(lin(H("a"), H("b")), "2.31.0"), res(lin(H("a"), H("c")), "2.30.0")).state, "mismatch");
});

test("V4h: the results reader carries the embedded opensees stamp; the banner names the cause through the store", async () => {
  const r = await stagedResults();
  assert.equal(r.info.openseesVersion, "2.27.0", "the staged results embed opensees 2.27.0 (reader/results.ts reads it, main/results.ts sends it)");
  const s0 = await stagedState();
  assert.deepEqual(s0.artifacts.model?.zones["opensees"], { status: "ready", version: "2.27.0" });
  // The staged model is opensees 2.27.0; results of another model at 2.31.0 straddle the boundary.
  const other = { ...r.info, openseesVersion: "2.31.0", lineage: { ...r.info.lineage!, modelHash: H("e") } };
  const s = reduce(s0, { type: "resultsLoaded", ...r, info: other });
  assert.equal(resultsPairing(s).state, "mismatch");
  assert.equal(s.phase.axis.some((k) => k.kind === "results"), false);
  assert.match(
    noticesOf(s).join("\n"),
    /^Results not drawn: staged_activation\.results\.h5 does not belong to this model \(model_hash differs: the opensees schema 2\.31\.0 hash change \(declaration ids entered the hash; results opensees 2\.31\.0, model opensees 2\.27\.0\)/m,
  );
});
