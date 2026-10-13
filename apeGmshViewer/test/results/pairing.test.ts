// Pairing a results file with the open model (V4b, #1728), by content: the
// rule of h5-schema.md (`model_hash`, else `fem_hash`), never the name or the
// path. Each state has its case, and only `paired` puts the results stages on
// the phase axis. The staged fixture pairs by model_hash through the real
// readers (main/zones.ts for the model, main/results.ts for the results).

import assert from "node:assert/strict";
import { copyFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import * as h5wasm from "h5wasm/node";
import { pairResults } from "../../src/state/results.ts";
import { noticesOf, resultsPairing } from "../../src/state/selectors.ts";
import { initialState, reduce, type State } from "../../src/state/store.ts";
import type { Lineage, ResultsInfo } from "../../src/state/types.ts";
import { MODEL, RESULTS, stagedModelLoad, stagedResults, stagedState } from "./staged.ts";

const H = (c: string) => c.repeat(32);
const lin = (fem: string | null, model: string | null, results: string | null = H("r")): Lineage => ({ femHash: fem, modelHash: model, resultsHash: results });
/** A side's opensees zone: its stamp and its `/opensees/decls@program_decl` marker as the readers report it; NONE: no /opensees. */
type Ops = { openseesVersion: string | null; programDecl: string | null };
const NONE: Ops = { openseesVersion: null, programDecl: null };
const ops = (openseesVersion: string, programDecl: string): Ops => ({ openseesVersion, programDecl });
/** A file written below 2.31.0 (no marker: the readers report "absent"). */
const LEGACY = ops("2.30.0", "absent");
/** A 2.31 apeSees emit: its hash includes declaration ids. */
const EMIT = ops("2.31.0", "complete");
/** A to_h5 rewrite of a legacy file: stamped 2.31, marker "absent", the old hash. */
const REWRITE = ops("2.31.0", "absent");
const res = (lineage: Lineage | null, o: Ops = NONE): ResultsInfo => ({
  path: "x.results.h5",
  version: "1.1.0",
  sourceType: "domain_capture",
  lineage,
  partial: null,
  stages: [],
  unavailableZones: [],
  ...o,
});
/** The open model as pairResults takes it: its lineage, `/meta@opensees_schema_version` and decls marker. */
const mod = (lineage: Lineage | null, o: Ops = NONE) => ({ lineage, ...o });

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
// mismatch where exactly one side's hash includes declaration ids names the
// cause; it stays a refusal, never a pairing, never a fall-back to fem_hash.
// The era is the `/opensees/decls@program_decl` marker on each side, never the
// version stamp (a to_h5 rewrite of an older file is stamped 2.31 but keeps
// its old hash; Python #1861). Any other mismatch reads exactly as before.
const plain = (rv: string, mv: string) => `model_hash differs: the results were written from another model (results ${rv}…, model ${mv}…)`;
const PLAIN = plain("cccccccc", "bbbbbbbb");
const reasonOf = (p: ReturnType<typeof pairResults>): string => {
  assert.equal(p.state, "mismatch");
  return (p as { reason: string }).reason;
};
const mismatch = (m: Ops, r: Ops) => reasonOf(pairResults(mod(lin(H("a"), H("b")), m), res(lin(H("a"), H("c")), r)));

test("V4h: legacy results against a re-emitted (complete) model are refused with the 2.31.0 hash change named", () => {
  const reason = mismatch(EMIT, LEGACY);
  assert.match(
    reason,
    /^model_hash differs: the opensees schema 2\.31\.0 hash change \(declaration ids entered the hash; results opensees 2\.30\.0, model opensees 2\.31\.0\); re-run the analysis against the re-saved model/,
  );
  assert.match(reason, /\(results cccccccc…, model bbbbbbbb…\)$/);
});

test("V4h: the reverse direction (complete results, legacy model) names the cause with the other remedy", () => {
  assert.match(mismatch(LEGACY, EMIT), /the opensees schema 2\.31\.0 hash change .*results opensees 2\.31\.0, model opensees 2\.30\.0\); open the results with the model\.h5 they were run on/);
});

test("V4h (review reproducer): a legacy to_h5 rewrite stamped 2.31 with no complete marker, against 2.30 results, gives the exact old text", () => {
  assert.equal(mismatch(REWRITE, LEGACY), PLAIN);
  assert.equal(mismatch(LEGACY, REWRITE), PLAIN);
});

test("V4h: the era is the marker, not the stamp: same-marker mismatches keep today's message whatever the stamps", () => {
  const pairs: [Ops, Ops][] = [
    [LEGACY, LEGACY],
    [EMIT, EMIT],
    [ops("2.30.0", "complete"), EMIT],
    [ops("2.32.0", "absent"), LEGACY],
    [REWRITE, ops("2.27.0", "absent")],
  ];
  for (const [m, r] of pairs) assert.equal(mismatch(m, r), PLAIN, `model ${JSON.stringify(m)}, results ${JSON.stringify(r)}`);
  // A rewrite (stamped 2.31, "absent") against a 2.31 emit: the markers differ, so the cause is named.
  assert.match(mismatch(REWRITE, EMIT), /2\.31\.0 hash change/);
});

test("V4h: no opensees zone, or an unknown marker, on either side keeps today's message", () => {
  const pairs: [Ops, Ops][] = [
    [NONE, EMIT],
    [EMIT, NONE],
    [NONE, NONE],
    [ops("2.31.0", "partial"), LEGACY],
    [EMIT, ops("2.31.0", "\"1\"")],
  ];
  for (const [m, r] of pairs) assert.equal(mismatch(m, r), PLAIN, `model ${JSON.stringify(m)}, results ${JSON.stringify(r)}`);
});

test("V4h: a matching hash across eras still pairs; a cross-era mismatch never falls back to fem_hash", () => {
  assert.deepEqual(pairResults(mod(lin(H("a"), H("b")), EMIT), res(lin(H("a"), H("b")), LEGACY)), { state: "paired", by: "model_hash" });
  // Same mesh (fem_hash a), another model_hash across eras: refused, not paired by fem_hash.
  assert.equal(pairResults(mod(lin(H("a"), H("b")), EMIT), res(lin(H("a"), H("c")), LEGACY)).state, "mismatch");
});

// Through the real readers: scratch copies of the staged pair (opensees 2.27.0,
// no marker), edited in read-write mode.
function editedCopy(from: string, name: string, edit: (f: h5wasm.File) => void): string {
  const p = join(mkdtempSync(join(tmpdir(), "agv-v4h-")), name);
  copyFileSync(from, p);
  const f = new h5wasm.File(p, "a");
  try {
    edit(f);
  } finally {
    f.close();
  }
  return p;
}
const setAttr = (f: h5wasm.File, path: string, key: string, value: string) => {
  const node = f.get(path) as h5wasm.Group;
  if (Object.keys(node.attrs).includes(key)) node.delete_attribute(key);
  node.create_attribute(key, value);
};
/** The staged results re-stamped as written from another model; `marker` sets their embedded decls marker. */
const otherResults = (marker: string | null) =>
  editedCopy(RESULTS, "staged_activation.results.h5", (f) => {
    setAttr(f, "meta/lineage", "model_hash", H("e"));
    if (marker !== null) setAttr(f, "opensees/decls", "program_decl", marker);
  });
/** The staged model with its decls marker (and optionally its stamp) set. */
const markedModel = (marker: string | null, stamp: string | null = null) =>
  editedCopy(MODEL, "staged_activation.h5", (f) => {
    if (marker !== null) setAttr(f, "opensees/decls", "program_decl", marker);
    if (stamp !== null) setAttr(f, "meta", "opensees_schema_version", stamp);
  });
async function pairedState(modelPath: string, resultsPath: string): Promise<State> {
  const r = await stagedResults(resultsPath);
  let s: State = reduce(initialState, { type: "fileLoaded", artifact: "model", load: await stagedModelLoad(modelPath) });
  s = reduce(s, { type: "fileOpened", artifact: "results", path: r.info.path });
  return reduce(s, { type: "resultsLoaded", ...r });
}
const BANNER = /^Results not drawn: staged_activation\.results\.h5 does not belong to this model \(model_hash differs: (.*)\)\.$/m;
const bannerReason = (s: State): string => {
  assert.equal(resultsPairing(s).state, "mismatch");
  assert.equal(s.phase.axis.some((k) => k.kind === "results"), false);
  const hit = BANNER.exec(noticesOf(s).join("\n"));
  assert.ok(hit, noticesOf(s).join("\n"));
  return hit[1]!;
};

test("V4h: the readers report a missing marker as absent: the staged pair (2.27.0, no marker) carries 'absent' on both sides", async () => {
  const r = await stagedResults();
  assert.equal(r.info.openseesVersion, "2.27.0");
  assert.equal(r.info.programDecl, "absent");
  const s = await stagedState();
  assert.equal(s.definitions?.decls?.programDecl, "absent");
  assert.deepEqual(s.artifacts.model?.zones["opensees"], { status: "ready", version: "2.27.0" });
});

test("V4h: through the readers, complete results against the legacy staged model name the cause in the banner", async () => {
  const reason = bannerReason(await pairedState(MODEL, otherResults("complete")));
  assert.match(reason, /^the opensees schema 2\.31\.0 hash change \(declaration ids entered the hash; results opensees 2\.27\.0, model opensees 2\.27\.0\); open the results with the model\.h5 they were run on/);
});

test("V4h: through the readers, a complete model (read.ts marker) against legacy results names the cause", async () => {
  const model = markedModel("complete");
  const s = await pairedState(model, otherResults(null));
  assert.equal(s.definitions?.decls?.programDecl, "complete");
  assert.match(bannerReason(s), /^the opensees schema 2\.31\.0 hash change .*; re-run the analysis against the re-saved model/);
});

test("V4h (review reproducer, through the readers): a model stamped 2.31 with no complete marker against legacy results reads as before", async () => {
  const s = await pairedState(markedModel(null, "2.31.0"), otherResults(null));
  assert.equal(bannerReason(s), "the results were written from another model (results eeeeeeee…, model f359ba2f…)");
  // An explicit "absent" (the rewrite's marker) is the same.
  assert.equal(bannerReason(await pairedState(markedModel("absent", "2.31.0"), otherResults(null))), "the results were written from another model (results eeeeeeee…, model f359ba2f…)");
  // An unknown marker names no era.
  assert.equal(bannerReason(await pairedState(markedModel("partial"), otherResults("complete"))), "the results were written from another model (results eeeeeeee…, model f359ba2f…)");
});
