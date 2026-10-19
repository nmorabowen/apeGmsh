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
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { pairingNotice, pairResults, SIDE_CHANNEL_DECL_VERBS, sideChannelProof } from "../../src/state/results.ts";
import { programDecls } from "../../src/reader/read.ts";
import { noticesOf, resultsPairing } from "../../src/state/selectors.ts";
import { initialState, reduce, type State } from "../../src/state/store.ts";
import type { Lineage, ProgramDecls, ResultsInfo } from "../../src/state/types.ts";
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

// -- V4L-b1 (#2127): a .ladruno pairs by its INFO/PROVENANCE stamp (ADR 0122 D3) --
//
// ladruno_frame3d.h5 and ladruno_frame3d.ladruno (#2140) carry one stamp:
// model_hash 0123…, fem_hash fedc… on both sides, written by make_ladruno_fixtures.py.
// ladruno_truss2d.ladruno is the fork's output unstamped (an empty PROVENANCE group).

const FIX = (name: string) => fileURLToPath(new URL(`../../fixtures/${name}`, import.meta.url));
const L_MODEL = FIX("ladruno_frame3d.h5");
const L_STAMPED = FIX("ladruno_frame3d.ladruno");
const L_UNSTAMPED = FIX("ladruno_truss2d.ladruno");
const ladrunoNotice = (s: State): string => noticesOf(s).filter((n) => n.startsWith("Results not drawn")).join("\n");

test("V4L-b1: a .ladruno with no stamp is unpairable and the reason names stamp_provenance; a native file keeps its message", () => {
  const l: ResultsInfo = { ...res(null), path: "run.ladruno", sourceType: "ladruno", lineageReason: "INFO/PROVENANCE has no @model_hash; INFO/PROVENANCE has no @fem_hash: the file is not stamped for pairing (schema §7b.1)" };
  const p = pairResults(mod(lin(H("a"), H("b"))), l);
  assert.equal(p.state, "unpairable");
  assert.match((p as { reason: string }).reason, /has no @model_hash.*Stamp it with stamp_provenance\(path, model_h5\) against the model\.h5 the run was emitted from \(ADR 0122 D3\)/);
  // The same file stamped pairs by model_hash, as a native file does.
  assert.deepEqual(pairResults(mod(lin(H("a"), H("b"))), { ...l, lineage: lin(H("a"), H("b"), null) }), { state: "paired", by: "model_hash" });
  // Native, unchanged: no /meta/lineage names the native reason.
  assert.match((pairResults(mod(lin(H("a"), H("b"))), res(null)) as { reason: string }).reason, /^the results file has no \/meta\/lineage \(results 1\.1\.0\)/);
});

test("V4L-b1: a .ladruno with no model open: not drawn, and the notice asks for the model.h5 (it has no embedded /model)", async () => {
  assert.equal(
    pairingNotice({ state: "noModel" }, "run.ladruno"),
    "Results not drawn: no model is open for run.ladruno; open the model.h5 the run was emitted from.",
  );
  assert.equal(pairingNotice({ state: "noModel" }, "RUN.LADRUNO"), "Results not drawn: no model is open for RUN.LADRUNO; open the model.h5 the run was emitted from.");
  // A native file keeps its sentence.
  assert.match(pairingNotice({ state: "noModel" }, "x.results.h5")!, /^Results not drawn: no model is open for x\.results\.h5\. Its embedded \/model is not drawn/);
  // Through the store: the stamped .ladruno read with no model.
  const r = await stagedResults(L_STAMPED);
  const s = reduce(reduce(initialState, { type: "fileOpened", artifact: "results", path: r.info.path }), { type: "resultsLoaded", ...r });
  assert.deepEqual(resultsPairing(s), { state: "noModel" });
  assert.deepEqual(noticesOf(s).filter((n) => n.startsWith("Results not drawn")), ["Results not drawn: no model is open for ladruno_frame3d.ladruno; open the model.h5 the run was emitted from."]);
});

test("V4L-b1: a stamped .ladruno beside its model pairs by model_hash through the real readers; its stages go on the axis", async () => {
  const s = await pairedState(L_MODEL, L_STAMPED);
  assert.deepEqual(resultsPairing(s), { state: "paired", by: "model_hash" });
  assert.deepEqual(s.results!.lineage, { modelHash: "0123456789abcdef0123456789abcdef", femHash: "fedcba9876543210fedcba9876543210", resultsHash: null });
  // PROVENANCE stage_names ["gravity","lateral"] name the two MODEL_STAGE groups.
  assert.deepEqual(s.results!.stages.map((st) => st.name), ["gravity", "lateral"]);
  assert.ok(s.phase.axis.some((k) => k.kind === "results"));
  assert.equal(ladrunoNotice(s), "");
  // Nodal components reach the store verbatim, as the reader names them.
  assert.ok(s.results!.stages[0]!.nodal.length > 0);
  assert.ok(s.results!.stages[0]!.nodal.every((c) => /^[A-Z_()]+\/\S+$/.test(c)), s.results!.stages[0]!.nodal.join(", "));
  // #2128: the regenerated fixture's groups are fork schema §7.1.1's (ndm 3): DISPLACEMENT and ROTATION apart.
  assert.deepEqual(s.results!.stages[0]!.nodal, ["DISPLACEMENT/Ux", "DISPLACEMENT/Uy", "DISPLACEMENT/Uz", "ROTATION/Rx", "ROTATION/Ry", "ROTATION/Rz"]);
});

test("V4L-b1: an unstamped .ladruno beside a model is not drawn; the banner says why and names the remedy", async () => {
  const s = await pairedState(L_MODEL, L_UNSTAMPED);
  assert.equal(resultsPairing(s).state, "unpairable");
  assert.equal(s.phase.axis.some((k) => k.kind === "results"), false);
  assert.match(ladrunoNotice(s), /^Results not drawn: INFO\/PROVENANCE has no @model_hash; INFO\/PROVENANCE has no @fem_hash: the file is not stamped for pairing \(schema §7b\.1\)\. Stamp it with stamp_provenance\(path, model_h5\)/);
});

test("V4L-b1: a stamp written in part (model_hash deleted) is unpairable, never paired by fem_hash alone", async () => {
  const half = editedCopy(L_STAMPED, "ladruno_frame3d.ladruno", (f) => (f.get("INFO/PROVENANCE") as h5wasm.Group).delete_attribute("model_hash"));
  const s = await pairedState(L_MODEL, half);
  assert.deepEqual(s.results!.lineage, null);
  assert.equal(resultsPairing(s).state, "unpairable");
  assert.match(ladrunoNotice(s), /INFO\/PROVENANCE has no @model_hash: the file is not stamped for pairing .*stamp_provenance/);
});

test("V4L-b1: a .ladruno stamped from another model is a mismatch, not drawn", async () => {
  const other = editedCopy(L_STAMPED, "ladruno_frame3d.ladruno", (f) => setAttr(f, "INFO/PROVENANCE", "model_hash", H("e")));
  const s = await pairedState(L_MODEL, other);
  assert.equal(resultsPairing(s).state, "mismatch");
  assert.equal(s.phase.axis.some((k) => k.kind === "results"), false);
  assert.equal(ladrunoNotice(s), "Results not drawn: ladruno_frame3d.ladruno does not belong to this model (model_hash differs: the results were written from another model (results eeeeeeee…, model 01234567…)).");
  // Against the staged model (another model altogether): the same refusal.
  assert.equal(resultsPairing(await pairedState(MODEL, L_STAMPED)).state, "mismatch");
});

test("V4L-b1: stage_names that do not count the MODEL_STAGE groups leave the file's names, and warn", async () => {
  const miscount = editedCopy(L_STAMPED, "ladruno_frame3d.ladruno", (f) => setAttr(f, "INFO/PROVENANCE", "stage_names", '["only"]'));
  const unnamed = editedCopy(L_STAMPED, "ladruno_frame3d.ladruno", (f) => (f.get("INFO/PROVENANCE") as h5wasm.Group).delete_attribute("stage_names"));
  const own = (await stagedResults(unnamed)).info.stages.map((st) => st.name);
  assert.equal(own.length, 2);
  assert.notDeepEqual(own, ["gravity", "lateral"], "the file's own names are not the stamped ones");
  const r = await stagedResults(miscount);
  assert.deepEqual(r.info.stages.map((st) => st.name), own);
  assert.ok(r.warnings.some((w) => /INFO\/PROVENANCE@stage_names names 1 stage\(s\), the file has 2 MODEL_STAGE group\(s\); the file's own stage names are shown/.test(w)), r.warnings.join("\n"));
});

test("V4L-b1: element_index_space \"fem\" verifies the element rows; absent is unverified with the reason", async () => {
  const stamped = await stagedResults(FIX("ladruno_truss2d.stamped.ladruno"));
  assert.deepEqual(stamped.info.elementIndex, { verified: true });
  const frame = await stagedResults(L_STAMPED);
  assert.equal(frame.info.elementIndex?.verified, false);
  assert.match((frame.info.elementIndex as { reason: string }).reason, /INFO\/PROVENANCE has no @element_index_space = "fem"/);
  const ops = editedCopy(FIX("ladruno_truss2d.stamped.ladruno"), "t.ladruno", (f) => setAttr(f, "INFO/PROVENANCE", "element_index_space", "ops_tag"));
  const bad = await stagedResults(ops);
  assert.equal(bad.info.elementIndex?.verified, false);
  assert.match((bad.info.elementIndex as { reason: string }).reason, /@element_index_space = "ops_tag"; §7b\.1 allows "fem" or no attribute; its element rows are not known/);
});

// -- #2189 (the viewer half of #1911): the opensees 2.32.0 side-channel step --
//
// At 2.32.0 the six side-channel verbs' calls gained their declaration ids in
// the hashed /opensees/program `decl` column (#1853), so an archive with any of
// them hashes differently once more while both `program_decl` markers still
// read "complete". The rule is Python's, as K shipped it (#2198,
// `Results.lineage`): both markers "complete", both programs call a side-channel
// verb, and the `decl >= 0` proof on exactly one side. It reads the column, not
// the version stamps. These cases mirror tests/results/test_lineage_hash_era.py.

const FORWARD = "re-run the analysis against the re-saved model, or open the results with the model.h5 they were run on";
const REVERSE = "open the results with the model.h5 they were run on, or re-run the analysis against this model";
/** Python's line (`_side_channel_hash_cause`), verbatim, inside the viewer's reason. */
const scLine = (remedy: string, rv = "cccccccc", mv = "bbbbbbbb") =>
  `model_hash differs: likely cause: the opensees schema 2.32.0 hash change (side-channel calls' declaration ids entered model_hash); ${remedy} (results ${rv}…, model ${mv}…)`;
/** One side as the readers report it: its stamp, its decls marker, and its program (read.ts `programDecls`). */
type Side = Ops & { program: ProgramDecls | null };
const side = (o: Ops, program: ProgramDecls | null): Side => ({ ...o, program });
/** A 2.32 emit with an `s.zero_velocities`: its side-channel calls carry their declaration. */
const PROVED = side(ops("2.32.0", "complete"), { model: false, node: false, element: true, set_node_vel: true, set_node_accel: true });
/** The same model emitted at 2.31: every side-channel decl is -1. */
const BELOW = side(ops("2.31.0", "complete"), { model: false, node: false, element: true, set_node_vel: false, set_node_accel: false });
/** A to_h5 rewrite of BELOW: stamped 2.32, the old column (and so the old hash). */
const REWRITTEN = side(ops("2.32.0", "complete"), BELOW.program);
/** Archives with no side-channel call, either side of the step. */
const NO_SC_32 = side(ops("2.32.0", "complete"), { model: false, node: false, element: true });
const NO_SC_31 = side(ops("2.31.0", "complete"), { model: false, node: false, element: true });
const scMismatch = (m: Side, r: Side) =>
  reasonOf(pairResults({ lineage: lin(H("a"), H("b")), ...m }, { ...res(lin(H("a"), H("c"))), ...r }));

test("#2189: results run below 2.32 against the re-emitted model with a side-channel call name the 2.32.0 step, Python's line verbatim", () => {
  assert.equal(scMismatch(PROVED, BELOW), scLine(FORWARD));
});

test("#2189: the reverse direction (results at the step, model below it) names the step with the reverse remedy", () => {
  assert.equal(scMismatch(BELOW, PROVED), scLine(REVERSE));
});

test("#2189: each condition of the rule dropped in turn gives the bare reason", () => {
  // 1. Both markers "complete": an unknown marker, or no opensees zone, on either side names nothing.
  const partial = (s: Side): Side => ({ ...s, programDecl: "partial" });
  const noZone = (s: Side): Side => ({ ...s, openseesVersion: null, programDecl: null });
  assert.equal(scMismatch(partial(PROVED), BELOW), PLAIN);
  assert.equal(scMismatch(PROVED, partial(BELOW)), PLAIN);
  assert.equal(scMismatch(PROVED, noZone(BELOW)), PLAIN);
  assert.equal(scMismatch(noZone(PROVED), BELOW), PLAIN);
  // 2. Both programs call a side-channel verb: one side without such a call, or with no program read, names nothing.
  assert.equal(scMismatch(NO_SC_32, BELOW), PLAIN);
  assert.equal(scMismatch(PROVED, NO_SC_31), PLAIN);
  assert.equal(scMismatch(side(PROVED, null), BELOW), PLAIN);
  assert.equal(scMismatch(PROVED, side(BELOW, null)), PLAIN);
  const { program: _dropped, ...unread } = BELOW;
  assert.equal(reasonOf(pairResults({ lineage: lin(H("a"), H("b")), ...PROVED }, { ...res(lin(H("a"), H("c"))), ...unread })), PLAIN, "an info built without a program names no cause");
  // 3. The proof on exactly one side: both proved, or neither, names nothing.
  assert.equal(scMismatch(PROVED, PROVED), PLAIN);
  assert.equal(scMismatch(BELOW, BELOW), PLAIN);
});

test("#2189: a straddling pair with no side-channel call, and a same-version mismatch, keep the bare reason", () => {
  assert.equal(scMismatch(NO_SC_32, NO_SC_31), PLAIN);
  assert.equal(scMismatch(NO_SC_31, NO_SC_32), PLAIN);
  assert.equal(scMismatch(PROVED, side(ops("2.32.0", "complete"), PROVED.program)), PLAIN);
});

test("#2189: the rule reads the hashed column, not the stamps (the to_h5 rewrite seam of #2198)", () => {
  // A rewrite (stamped 2.32, old column) beside results of another 2.31 model: the stamps straddle, the step is not the cause.
  assert.equal(scMismatch(REWRITTEN, BELOW), PLAIN);
  // Results run on the rewrite (old column, stamped 2.32) beside the re-emitted model: the stamps agree, the step is the cause.
  assert.equal(scMismatch(PROVED, REWRITTEN), scLine(FORWARD));
  // Stamps alone never decide: the same columns under swapped stamps read the same.
  assert.equal(scMismatch(side(ops("2.31.0", "complete"), PROVED.program), side(ops("2.32.0", "complete"), BELOW.program)), scLine(FORWARD));
});

test("#2189: the 2.31.0 marker cause takes precedence over the side-channel step", () => {
  // Legacy results (no marker; their side-channel calls all -1) against the proved model.
  const legacy = side(LEGACY, { set_node_vel: false });
  const reason = scMismatch(PROVED, legacy);
  assert.match(reason, /^model_hash differs: the opensees schema 2\.31\.0 hash change \(declaration ids entered the hash;/);
  assert.doesNotMatch(reason, /side-channel/);
});

test("#2189: sideChannelProof reads only the six verbs; a decl of 0 is a proof", () => {
  assert.equal(sideChannelProof(null), null);
  assert.equal(sideChannelProof(undefined), null);
  assert.equal(sideChannelProof({}), null);
  assert.equal(sideChannelProof({ element: true, node: false, fix: true }), null, "other verbs never count");
  assert.equal(sideChannelProof({ step_hook_ramp: false, element: true }), false);
  assert.equal(sideChannelProof({ step_hook_ramp: false, addToParameter: true }), true, "one proved verb is enough");
  assert.equal(SIDE_CHANNEL_DECL_VERBS.length, 6);
  for (const verb of SIDE_CHANNEL_DECL_VERBS) {
    assert.equal(sideChannelProof({ [verb]: true }), true, verb);
    assert.equal(sideChannelProof({ [verb]: false }), false, verb);
  }
  // programDecls: decl 0 proves, -1 does not; a run of any verb with decl >= 0 makes it proved.
  const cols = ["first", "count", "method", "store", "row", "stage", "decl"];
  const t = programDecls("/opensees/program", ["node", "set_node_vel", "update_parameter", "unused"], cols, [
    [1, 4, 0, 0, 0, -1, -1],
    [5, 1, 1, 1, 0, 0, -1],
    [6, 1, 1, 1, 1, 0, 0],
    [7, 1, 2, 2, 0, 0, -1],
  ]);
  assert.deepEqual({ ...(t as ProgramDecls) }, { node: false, set_node_vel: true, update_parameter: false });
  assert.equal(sideChannelProof(t as ProgramDecls), true);
});

test("#2189: programDecls refuses a table that is not the schema's, naming the path", () => {
  const cols = ["first", "count", "method", "store", "row", "stage", "decl"];
  assert.match(programDecls("/opensees/program", ["a"], ["first", "method"], []) as string, /^\/opensees\/program has no method and decl columns/);
  assert.match(programDecls("/opensees/program", undefined, cols, []) as string, /@methods is not a list of strings/);
  assert.match(programDecls("/opensees/program", ["a"], cols, [[1, 1, 3, 0, 0, -1, -1]]) as string, /\[0\]\.method is 3, outside the 1 names of @methods/);
  assert.match(programDecls("/opensees/program", ["a"], cols, [[1, 1, 0, 0, 0, -1, 0.5]]) as string, /\[0\]\.decl is not an integer/);
});

test("#2189: a .ladruno never names an era cause (Python's non-native reader returns None)", () => {
  // The .ladruno read reports no opensees zone and no program; even a program on it would not name the step.
  const l = (program: ProgramDecls | null): ResultsInfo => ({ ...res(lin(H("a"), H("c"))), path: "run.ladruno", sourceType: "ladruno", program });
  assert.equal(reasonOf(pairResults({ lineage: lin(H("a"), H("b")), ...PROVED }, l(null))), PLAIN);
  assert.equal(reasonOf(pairResults({ lineage: lin(H("a"), H("b")), ...PROVED }, l(BELOW.program))), PLAIN);
});

// Through the real readers: the staged pair (opensees 2.27.0) edited so its
// program calls `set_node_vel` (renamed from `remove_element` in @methods),
// with or without decl 0 on those runs, and its markers set.
function sideChannelCall(f: h5wasm.File, proved: boolean): void {
  const ds = f.get("opensees/program") as h5wasm.Dataset;
  const methods = (ds.attrs["methods"]!.value as string[]).map((m) => (m === "remove_element" ? "set_node_vel" : m));
  ds.delete_attribute("methods");
  ds.create_attribute("methods", methods);
  if (!proved) return;
  const dtype = ds.dtype as unknown as [string, string][];
  const names = dtype.map(([n]) => n);
  const im = names.indexOf("method");
  const id = names.indexOf("decl");
  const k = methods.indexOf("set_node_vel");
  (ds.value as number[][]).forEach((row, i) => {
    if (row[im] !== k) return;
    const cols = new Map(dtype.map(([n, t], j) => [n, (t === "<h" ? Int16Array : Int32Array).from([j === id ? 0 : row[j]!])]));
    ds.write_slice([[i, i + 1]], cols);
  });
}
const scModel = (proved: boolean) =>
  editedCopy(MODEL, "staged_activation.h5", (f) => {
    setAttr(f, "opensees/decls", "program_decl", "complete");
    sideChannelCall(f, proved);
  });
const scResults = (proved: boolean) =>
  editedCopy(RESULTS, "staged_activation.results.h5", (f) => {
    setAttr(f, "meta/lineage", "model_hash", H("e"));
    setAttr(f, "opensees/decls", "program_decl", "complete");
    sideChannelCall(f, proved);
  });

test("#2189: both readers read /opensees/program; the staged pair calls no side-channel verb", async () => {
  const s = await stagedState();
  const r = await stagedResults();
  for (const p of [s.definitions?.program, r.info.program]) {
    assert.ok(p);
    assert.equal(p["remove_element"], false);
    assert.equal(p["node"], false, "below 2.31 every decl is -1");
    assert.equal(sideChannelProof(p), null);
  }
});

test("#2189: through the readers, a proved model against unproved results names the step in the banner; the reverse too", async () => {
  const fwd = await pairedState(scModel(true), scResults(false));
  assert.equal(fwd.definitions?.program?.["set_node_vel"], true);
  assert.equal(fwd.results?.program?.["set_node_vel"], false);
  assert.equal(`model_hash differs: ${bannerReason(fwd)}`, scLine(FORWARD, "eeeeeeee", "f359ba2f"));
  const rev = await pairedState(scModel(false), scResults(true));
  assert.equal(`model_hash differs: ${bannerReason(rev)}`, scLine(REVERSE, "eeeeeeee", "f359ba2f"));
  // Both proved, or neither: the bare reason.
  const bare = "the results were written from another model (results eeeeeeee…, model f359ba2f…)";
  assert.equal(bannerReason(await pairedState(scModel(true), scResults(true))), bare);
  assert.equal(bannerReason(await pairedState(scModel(false), scResults(false))), bare);
});

test("#2189: an embedded program the reader cannot read warns and names no cause", async () => {
  const broken = editedCopy(RESULTS, "staged_activation.results.h5", (f) => {
    setAttr(f, "meta/lineage", "model_hash", H("e"));
    setAttr(f, "opensees/decls", "program_decl", "complete");
    sideChannelCall(f, false);
    (f.get("opensees/program") as h5wasm.Dataset).delete_attribute("methods");
  });
  const r = await stagedResults(broken);
  assert.equal(r.info.program, null);
  assert.ok(r.warnings.some((w) => /\/opensees\/program@methods is not a list of strings; the results pairing does not read it/.test(w)), r.warnings.join("\n"));
  assert.equal(bannerReason(await pairedState(scModel(true), broken)), "the results were written from another model (results eeeeeeee…, model f359ba2f…)");
});

test("#2189: through the readers, a .ladruno against a proved, complete model names no era cause", async () => {
  const r = await stagedResults(L_STAMPED);
  assert.equal(r.info.program, null);
  assert.equal(r.info.programDecl, null);
  const s = await pairedState(scModel(true), L_STAMPED);
  assert.equal(resultsPairing(s).state, "mismatch");
  assert.match(ladrunoNotice(s), /^Results not drawn: ladruno_frame3d\.ladruno does not belong to this model \(model_hash differs: the results were written from another model \(results 01234567…, model f359ba2f…\)\)\.$/);
});
