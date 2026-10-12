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
const res = (lineage: Lineage | null): ResultsInfo => ({ path: "x.results.h5", version: "1.1.0", sourceType: "domain_capture", lineage, partial: null, stages: [], unavailableZones: [] });

test("paired by model_hash when the results carry one; by fem_hash only when they do not", () => {
  assert.deepEqual(pairResults({ lineage: lin(H("a"), H("b")) }, res(lin(H("a"), H("b")))), { state: "paired", by: "model_hash" });
  assert.deepEqual(pairResults({ lineage: lin(H("a"), H("b")) }, res(lin(H("a"), null))), { state: "paired", by: "fem_hash" });
});

test("mismatch: another model_hash refuses even when the mesh (fem_hash) is the same", () => {
  const p = pairResults({ lineage: lin(H("a"), H("b")) }, res(lin(H("a"), H("c"))));
  assert.equal(p.state, "mismatch");
  assert.match((p as { reason: string }).reason, /model_hash differs.*results cccccccc…, model bbbbbbbb…/);
  assert.equal(pairResults({ lineage: lin(H("a"), H("b")) }, res(lin(H("d"), null))).state, "mismatch");
  assert.equal(pairResults({ lineage: null }, res(lin(H("a"), H("b")))).state, "mismatch", "a model with no lineage pairs with nothing");
});

test("noModel, unpairable and none", () => {
  assert.deepEqual(pairResults(null, res(lin(H("a"), H("b")))), { state: "noModel" });
  assert.equal(pairResults({ lineage: lin(H("a"), H("b")) }, res(null)).state, "unpairable");
  assert.equal(pairResults({ lineage: lin(H("a"), H("b")) }, res(lin(null, null))).state, "unpairable");
  assert.deepEqual(pairResults({ lineage: lin(H("a"), H("b")) }, null), { state: "none" });
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
