// V3c0 (#1641): the model-definition data in the store, and the window frame's
// registry. The reader loads /opensees/decls (2.25), /opensees/decl_params
// (2.26), the analysis chain, time series, patterns, recorders, fix records and
// /opensees/stages/stage_NNN; the root /sequence zone when present (V3a), and a
// typed "not in this file" state when absent.
//
// Oracles: fixtures/definitions.h5, written by hand by make_zone_fixtures.py
// (`definitions`): every count and value checked here is a literal of that
// script, not a value copied from the app. The malformed cases mirror the
// checks of Python's `H5Model.declarations()` (h5-schema.md, decl_params: "a
// column that differs in length, a decl out of range, a params that is not a
// JSON object, a reference or *_ref to a key the declarations lack, an object
// with an unknown $ tag, or a params_names that is not a list of str").

import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import type { ModelFile } from "../../src/model/types.ts";
import { WINDOWS } from "../../src/panels/windows/registry.ts";
import { openModel } from "../../src/reader/node.ts";
import { readModel, SchemaError, type H5Module } from "../../src/reader/read.ts";
import { BlobStore } from "../../src/state/blobs.ts";
import { loadModel } from "../../src/state/load.ts";
import { initialState, reduce } from "../../src/state/reduce.ts";
import { refusalsOf, warningsOf, windowContentOf } from "../../src/state/selectors.ts";
import { WINDOW_KINDS, type Event, type State, type WindowKind } from "../../src/state/types.ts";
import { tree, writeTree, type Tree } from "./tree.ts";

await h5wasm.ready;
const h5 = h5wasm as unknown as H5Module;
const fixtures = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "fixtures");
const DEFS = join(fixtures, "definitions.h5");
const PLAIN = join(fixtures, "shoebuckle.h5");
const dir = mkdtempSync(join(tmpdir(), "agv-defs-"));

const model: ModelFile = await openModel(DEFS);
const load = (m: ModelFile): State => reduce(initialState, { type: "fileLoaded", artifact: "model", load: loadModel(m, new BlobStore()) });
const state = load(model);
const M = "opensees/uniaxialMaterial/#1", S = "opensees/section/#1", T = "opensees/geomTransf/#1", BI = "opensees/beamIntegration/#1";

/** Fails on anything that is not a record, array, string, number, boolean or null (the D6 snapshot rule). */
function assertPlain(v: unknown, at: string): void {
  if (v === null || typeof v === "string" || typeof v === "number" || typeof v === "boolean") return;
  if (typeof v !== "object") throw new Error(`${at}: ${typeof v} is not plain data`);
  if (ArrayBuffer.isView(v) || v instanceof ArrayBuffer || v instanceof Map || v instanceof Set) throw new Error(`${at}: a ${v.constructor.name} is in the state`);
  for (const [k, x] of Object.entries(v)) assertPlain(x, `${at}.${k}`);
}

test("a 2.26.0 file opens with no banner, and the store holds its definitions as plain data", () => {
  assert.equal(model.opensees?.version, "2.26.0");
  assert.deepEqual(model.warnings, []);
  assert.deepEqual(warningsOf(state), []);
  assert.deepEqual(refusalsOf(state), []);
  assert.deepEqual(state.artifacts.model?.zones["opensees"], { status: "ready", version: "2.26.0" });
  assert.ok(state.definitions);
  assert.equal(state.definitions, model.opensees!.definitions, "the store holds the loaded definitions, not a copy");
  assertPlain(state.definitions, "definitions");
  assertPlain(state.sequence, "sequence");
});

test("decls: 16 keys with family, name and synth; the tag runs and the tagless rows", () => {
  const d = state.definitions!.decls!;
  assert.equal(d.hasParams, true);
  assert.equal(d.records.length, 16);
  assert.deepEqual(d.records.slice(0, 4).map((r) => [r.key, r.family]), [[M, "uniaxialMaterial"], [S, "section"], [T, "geomTransf"], [BI, "beamIntegration"]]);
  assert.deepEqual(d.records.filter((r) => r.name).map((r) => r.name), ["supports", "ramp", "gravity", "crown", "push", "arch"]);
  assert.deepEqual(d.records.filter((r) => r.synth).map((r) => r.key), ["opensees/timeSeries/support:gravity/hold"]);
  assert.equal(d.tags.length, 11);
  assert.deepEqual(d.tags.find((t) => t.kind === "element"), { kind: "element", first: 2, count: 123, decl: 4 });
  assert.deepEqual(Object.keys(d.rows).sort(), ["bcs/fix", "recorders", "stages/stage_000/bcs/fix", "stages/stage_000/recorders"]);
  assert.deepEqual(d.rows["bcs/fix"], [8, 8]);
});

test("decl_params: parameters by field name, one row per owner, refs as declaration keys", () => {
  const recs = state.definitions!.decls!.records;
  const by = (k: string) => recs.find((r) => r.key === k)!;
  assert.deepEqual(by(M).params[0]!.params, { E: 2e11, fy: 3.75e8, fu: 4.8e8, eu: 0.2, fracture: true });
  assert.deepEqual(by(S).params[0]!.params["patches"], [{ $struct: "RectPatch", fields: { material: { $decl: M }, ny: 8, nz: 1 } }]);
  assert.deepEqual(by(BI).params[0]!.paramsNames, ["section", "n"]);
  assert.equal(by(M).params[0]!.paramsNames, null);
  assert.deepEqual(by(BI).params[0]!.refs, { transf: [], integration: [], section: [S] });
  assert.deepEqual(by("opensees/element/#1").params[0]!.refs, { transf: [T], integration: [BI], section: [] });
  // A key with several owners keeps one entry per owner, in call order.
  assert.deepEqual(by("opensees/region/arch").params.map((p) => p.params["elements"]), [[2, 3], [4]]);
  // Every row of the file is somewhere: 17 rows over 16 keys.
  assert.equal(recs.reduce((a, r) => a + r.params.length, 0), 17);
});

test("series, patterns, recorders, fix records and the two stages", () => {
  const defs = state.definitions!;
  assert.equal(defs.analysis, null, "a staged file carries no global /opensees/analysis");
  assert.deepEqual(defs.timeSeries.map((t) => [t.groupName, t.tag]), [["Linear_1", 1], ["Linear_2", 2], ["Linear_3", 3]]);
  assert.deepEqual(defs.patterns.map((p) => [p.groupName, p.attrs["series_ref"]]), [["Plain_1", "/opensees/time_series/Linear_1"]]);
  assert.deepEqual(defs.recorders.map((r) => [r.groupName, r.tag, r.attrs["kind"]]), [["mpco_0", null, "typed"]]);
  assert.equal(defs.bcs.fix?.rows.length, 2);
  assert.deepEqual(defs.stages.map((s) => [s.index, s.name, s.attrs["analyze_steps"]]), [[0, "gravity", 10], [1, "push", 5]]);
  const [g, p] = defs.stages as [typeof defs.stages[0], typeof defs.stages[0]];
  assert.equal(g.analysis?.attrs["integrator"], "LoadControl");
  assert.deepEqual(g.analysis?.attrs["integrator_args"], [0.1]);
  assert.deepEqual(g.patterns.map((x) => [x.groupName, x.tag, x.attrs["emit_index"], x.tables["loads"]?.rows.length]), [["Plain_2", 2, 1, 1]]);
  assert.equal(g.bcs.fix?.rows.length, 1);
  assert.deepEqual(g.recorders.map((r) => [r.groupName, r.attrs["file"]]), [["Node_1", "gravity.out"]]);
  assert.deepEqual(g.other, []);
  assert.equal(p.attrs["analyze_dt"], 0.01);
  assert.equal(p.analysis?.attrs["analysis"], "Transient");
  assert.deepEqual(p.other, ["remove_sp"]);
});

test("no /sequence: a typed 'not in this file' state, no error, no banner", () => {
  assert.deepEqual(model.sequence, { status: "absent" });
  assert.deepEqual(state.sequence, { status: "absent" });
  assert.deepEqual(state.artifacts.model?.zones["sequence"], { status: "absent" });
  assert.match(windowContentOf(state, "stages").note!, /no \/sequence zone/);
});

test("a pre-2.25 file (shoebuckle, opensees 2.22) has no decls and keeps its global analysis", async () => {
  const s = load(await openModel(PLAIN));
  assert.equal(s.definitions!.decls, null);
  assert.equal(s.definitions!.analysis?.attrs["algorithm"], "KrylovNewton");
  assert.deepEqual(s.definitions!.stages, []);
  assert.match(windowContentOf(s, "stages").note!, /^unstaged model: one implicit stage, from \/opensees\/analysis/);
});

// ---- the window frame's registry and the window rows -----------------------

test("the registry has exactly one view per WindowKind", () => {
  assert.deepEqual(Object.keys(WINDOWS).sort(), [...WINDOW_KINDS].sort());
  for (const k of WINDOW_KINDS) assert.equal(typeof WINDOWS[k].view, "function", k);
});

test("openWindow / closeWindow drive the open set; an unknown kind is refused", () => {
  let s = reduce(state, { type: "openWindow", window: "stages" });
  s = reduce(s, { type: "openWindow", window: "materials" });
  assert.deepEqual(s.windows.open, ["stages", "materials"]);
  assert.equal(reduce(s, { type: "openWindow", window: "stages" }), s);
  s = reduce(s, { type: "closeWindow", window: "stages" });
  assert.deepEqual(s.windows.open, ["materials"]);
  assert.throws(() => reduce(s, { type: "openWindow", window: "nope" as WindowKind }), /not a window kind/);
});

test("each window's rows, with the decls each joins", () => {
  const rows = (k: WindowKind) => windowContentOf(state, k).rows;
  assert.deepEqual(rows("stages").map((r) => r.title), ["0 · gravity", "1 · push"]);
  assert.deepEqual(rows("analysis").map((r) => [r.section, r.title]), [["stage 0 · gravity", "Static · LoadControl · Newton"], ["stage 1 · push", "Transient · Newmark · Newton"]]);
  assert.deepEqual(rows("patterns").map((r) => [r.section, r.title, r.declKey]), [
    ["global", "Plain_1", "opensees/pattern/#1"],
    ["stage 0 · gravity", "gravity", "opensees/pattern/gravity"],
    ["stage 1 · push", "push", "opensees/pattern/push"],
    ["time series", "Linear_1", "opensees/timeSeries/#1"],
    ["time series", "ramp", "opensees/timeSeries/ramp"],
    ["time series", "Linear_3", "opensees/timeSeries/#2"],
  ]);
  assert.deepEqual(rows("recorders").map((r) => [r.section, r.title, r.declKey]), [["global", "mpco_0", "opensees/recorder/#1"], ["stage 0 · gravity", "crown", "opensees/recorder/crown"]]);
  const mats = rows("materials");
  // With /opensees/decls, the app keys every object by the archive's own key.
  assert.deepEqual(mats.map((r) => [r.section, r.declKey, r.select]), [["uniaxialMaterial", M, M], ["section", S, S]]);
  assert.match(mats[0]!.detail, /ASDSteel1D tag 1 · E=200000000000 · fy=375000000/);
});

test("a window row that is a declaration selects it through the store", () => {
  const row = windowContentOf(state, "materials").rows[0]!;
  const e: Event = { type: "select", pick: { decl: row.select!, at: null, nodes: null } };
  assert.deepEqual(reduce(state, e).selection.decls, [M]);
});

test("app declaration keys are the archive's /opensees/decls keys (2.25+); a pre-2.25 file keeps the listing keys", async () => {
  // definitions.h5: every unnamed object is keyed as the archive keys it
  // (declaration order from #1), and the beam chain still resolves through them.
  for (const k of [M, S, T, BI]) assert.ok(k in state.decls, k);
  assert.ok(!Object.keys(state.decls).some((k) => /^opensees\/[^/]+\/#0$/.test(k)), "no listing-order #0 key is left");
  const elem = Object.values(state.decls).find((d) => d.kind === "element")!;
  assert.deepEqual(Object.values(elem.refs).sort(), [BI, T]);
  // shoebuckle.h5 (opensees 2.22, no decls): the listing keys stay.
  const old = load(await openModel(PLAIN));
  assert.ok("opensees/uniaxialMaterial/#0" in old.decls);
});

test("the loader never falls back for an object no decls tag run owns: it throws, naming it", () => {
  // The reader refuses such a file (see the fail-closed cases); a ModelFile
  // that skipped it is refused again here, never keyed by the listing order.
  const m: ModelFile = structuredClone(model);
  m.opensees!.definitions.decls!.tags = m.opensees!.definitions.decls!.tags.filter((t) => t.kind !== "section");
  assert.throws(() => load(m), /\/opensees\/sections\/Fiber_1: \/opensees\/decls\/tags has no run for section 1/);
});

test("a declaration that owns several tags (a geomTransf fan-out): the first object takes the key, the rest <key>@<tag>", () => {
  const m: ModelFile = structuredClone(model);
  const ops = m.opensees!;
  const t1 = ops.objects.find((o) => o.family === "geomTransf")!;
  ops.objects.push({ ...t1, tag: 2, path: "/opensees/transforms/Linear_2", groupName: "Linear_2" });
  const run = ops.definitions.decls!.tags.find((t) => t.kind === "geomTransf")!;
  run.count = 2;
  const s = load(m);
  assert.equal(s.decls[T]?.h5, "/opensees/transforms/Linear_1");
  assert.equal(s.decls[`${T}@2`]?.h5, "/opensees/transforms/Linear_2");
  assert.equal(s.decls[`${T}@2`]?.tag, 2);
  // Both are listed: no object overwrote another.
  assert.equal(Object.values(s.decls).filter((d) => d.kind === "geomTransf").length, 2);
});

// ---- fail-closed: malformed decls / decl_params / stages, and /sequence ----

const KEYS = ["opensees/timeSeries/#1", "opensees/pattern/#1"];
function decls(over: Tree = {}): Tree {
  return {
    key: KEYS,
    family: ["timeSeries", "pattern"],
    name: ["", ""],
    synth: new Int8Array([0, 0]),
    tags: { kind: ["timeSeries", "pattern"], first: new Int32Array([1, 1]), count: new Int32Array([1, 1]), decl: new Int32Array([0, 1]) },
    rows: {},
    ...over,
  };
}
/** Two uniaxial materials, tags 1 and 2; the decls own only tag 1 (the review's overwrite case). */
const TWO_MATERIALS: Tree = {
  materials: {
    uniaxial: {
      Elastic_1: { attrs: { type: "Elastic", tag: 1, params: new Float64Array([2e11]) } },
      Elastic_2: { attrs: { type: "Elastic", tag: 2, params: new Float64Array([3e10]) } },
    },
  },
  decls: decls({
    key: ["opensees/uniaxialMaterial/#1"],
    family: ["uniaxialMaterial"],
    name: [""],
    synth: new Int8Array([0]),
    tags: { kind: ["uniaxialMaterial"], first: new Int32Array([1]), count: new Int32Array([1]), decl: new Int32Array([0]) },
  }),
};
function params(over: Tree = {}): Tree {
  const ref = { first: new Int32Array([0, 0]), count: new Int32Array([0, 0]), key: [""] };
  return {
    decl: new Int32Array([0, 1]),
    type: ["Linear", "Plain"],
    params: ['{"factor": 1.0}', '{"series": {"$decl": "opensees/timeSeries/#1"}}'],
    params_names: ["", ""],
    transf_ref: ref,
    integration_ref: ref,
    section_ref: ref,
    ...over,
  };
}
function file(name: string, opensees: Tree, root: Tree = {}, meta: Record<string, string> = {}): string {
  const path = join(dir, name);
  writeTree(h5wasm, path, tree({
    meta: { attrs: { neutral_schema_version: "2.35.0", opensees_schema_version: "2.26.0", ...meta } },
    nodes: { ids: new Int32Array([1]), coords: { data: new Float64Array([0, 0, 0]), shape: [1, 3] } },
    opensees,
    ...root,
  }));
  return path;
}
const read = (p: string) => readModel(h5, p, 0);

test("the synthetic base file reads (so each case below fails on its one defect)", () => {
  const m = read(file("ok.h5", { decls: decls(), decl_params: params() }));
  assert.deepEqual(m.opensees!.definitions.decls!.records.map((r) => r.params.length), [1, 1]);
});

const MALFORMED: [string, Tree, RegExp][] = [
  ["an unknown $ tag", { decls: decls(), decl_params: params({ params: ['{"factor": {"$expr": "2*x"}}', "{}"] }) }, /\/opensees\/decl_params\/params\[0\]\.factor carries the unknown tag\(s\) \["\$expr"\]/],
  ["a $decl to an undeclared key", { decls: decls(), decl_params: params({ params: ["{}", '{"series": {"$decl": "opensees/timeSeries/#9"}}'] }) }, /params\[1\]\.series references "opensees\/timeSeries\/#9", which \/opensees\/decls does not declare/],
  ["a $decl with another field", { decls: decls(), decl_params: params({ params: ['{"x": {"$decl": "opensees/pattern/#1", "y": 1}}', "{}"] }) }, /unknown tag/],
  ["params that is not a JSON object", { decls: decls(), decl_params: params({ params: ["[1, 2]", "{}"] }) }, /params\[0\] is not a JSON object/],
  ["params that is not JSON", { decls: decls(), decl_params: params({ params: ["{factor: 1}", "{}"] }) }, /params\[0\] is not JSON/],
  ["a decl row out of range", { decls: decls(), decl_params: params({ decl: new Int32Array([0, 2]) }) }, /row 1 points at declaration 2, outside the 2 rows/],
  ["params_names that is not a list of str", { decls: decls(), decl_params: params({ params_names: ["[1]", ""] }) }, /params_names\[0\] is not "" or a JSON list of strings/],
  ["columns of different lengths", { decls: decls(), decl_params: params({ type: ["Linear"] }) }, /decl, type, params and params_names lengths differ/],
  ["a *_ref run past its keys", { decls: decls(), decl_params: params({ section_ref: { first: new Int32Array([0, 1]), count: new Int32Array([0, 1]), key: [KEYS[0]!] } }) }, /section_ref row 1 runs 1\.\.2, outside its 1 keys/],
  ["a *_ref key the decls lack", { decls: decls(), decl_params: params({ transf_ref: { first: new Int32Array([0, 0]), count: new Int32Array([1, 0]), key: ["opensees/geomTransf/#1"] } }) }, /transf_ref row 0 is "opensees\/geomTransf\/#1", which \/opensees\/decls does not declare/],
  ["decl_params with no decls", { decl_params: params() }, /\/opensees\/decl_params is present but \/opensees\/decls/],
  ["decls columns of different lengths", { decls: decls({ name: [""] }) }, /key, family, name and synth lengths differ/],
  ["a tag run whose decl is out of range", { decls: decls({ tags: { kind: ["pattern"], first: new Int32Array([1]), count: new Int32Array([1]), decl: new Int32Array([5]) } }) }, /tags\/decl\[0\] is 5, outside the 2 rows/],
  ["a stage count that disagrees with @n_stages", { stages: { attrs: { n_stages: 2 }, stage_000: { attrs: { name: "a" } } } }, /@n_stages is 2 but the group holds 1 stage_NNN groups/],
  ["a stage with no name", { stages: { stage_000: { attrs: { analyze_steps: 1 } } } }, /stage_000 has no string 'name' attribute/],
  // Mirrors of Python's declarations(): tags and rows are required, and each
  // rows/<store> column has one entry per record of the store.
  ["decls with no tags", { decls: decls({ tags: undefined }) }, /\/opensees\/decls\/tags is missing/],
  ["decls with no rows", { decls: decls({ rows: undefined }) }, /\/opensees\/decls\/rows is missing/],
  ["rows for a store the file lacks", { decls: decls({ rows: { recorders: new Int32Array([0]) } }) }, /\/opensees\/decls\/rows\/recorders has 1 entries; \/opensees\/recorders has 0 records/],
  [
    "rows longer than their store",
    { recorders: { Node_1: { attrs: { type: "Node", kind: "typed" } } }, decls: decls({ rows: { recorders: new Int32Array([0, 1]) } }) },
    /\/opensees\/decls\/rows\/recorders has 2 entries; \/opensees\/recorders has 1 records/,
  ],
  ["an object no decls tag run owns", TWO_MATERIALS, /Elastic_2: \/opensees\/decls\/tags has no run for uniaxialMaterial 2/],
];

test("rows that match their store read; a dataset store counts its rows", () => {
  const m = read(file("rows-ok.h5", {
    recorders: { Node_1: { attrs: { type: "Node", kind: "typed" } } },
    bcs: { fix: { data: new Int32Array([1, 2, 3, 4]), shape: [2, 2] } },
    decls: decls({ rows: { recorders: new Int32Array([1]), bcs: { fix: new Int32Array([0, 0]) } } }),
  }));
  assert.deepEqual(m.opensees!.definitions.decls!.rows, Object.assign(Object.create(null), { recorders: [1], "bcs/fix": [0, 0] }));
});

for (const [what, opensees, msg] of MALFORMED) {
  test(`fail-closed: ${what} raises, naming the path`, () => {
    const p = file(`bad-${what.replace(/\W+/g, "-")}.h5`, opensees);
    assert.throws(() => read(p), (e: unknown) => e instanceof SchemaError && msg.test((e as Error).message));
  });
}

const SEQ: Tree = {
  sequence: {
    attrs: { staged: 1 },
    stages: { order: new Int32Array([0, 1]), name: ["gravity", "push"], source: ["/opensees/stages/stage_000", "/opensees/stages/stage_001"] },
    changes: { stage: new Int32Array([0]), op: ["add"], what: ["pattern"], name: [""], count: new Int32Array([1]), source: ["/opensees/stages/stage_000/patterns/Plain_2"] },
  },
};

test("/sequence 1.0.0 present: read into the store, flat tables with their source column", () => {
  const m = read(file("seq.h5", {}, SEQ, { sequence_schema_version: "1.0.0" }));
  assert.equal(m.sequence.status, "ready");
  if (m.sequence.status !== "ready") return;
  assert.deepEqual(m.sequence.stages.columns, ["name", "order", "source"]);
  assert.equal(m.sequence.stages.rows.length, 2);
  assert.equal(m.sequence.changes.rows.length, 1);
  const s = load(m);
  assert.deepEqual(s.artifacts.model?.zones["sequence"], { status: "ready", version: "1.0.0" });
  assert.equal(windowContentOf(s, "stages").note, "the model declares no stage; /sequence 1.0.0: 2 stages, 1 changes");
  assertPlain(s.sequence, "sequence");
});

test("/sequence of another major is refused as a zone; the model opens, and the banner says so", () => {
  const m = read(file("seq-major.h5", {}, SEQ, { sequence_schema_version: "2.0.0" }));
  assert.deepEqual(m.sequence, { status: "refused", version: "2.0.0", accepted: "1.0 and any later 1.x", newer: true, reason: "this app reads major 1 only" });
  assert.deepEqual(refusalsOf(load(m)), [
    "model file written by a newer apeGmsh: its sequence zone is version 2.0.0, this app reads 1.0 and any later 1.x (this app reads major 1 only)",
  ]);
});

test("/sequence with a newer minor opens with one banner; a broken /sequence raises, naming its path", () => {
  const m = read(file("seq-newer.h5", {}, SEQ, { sequence_schema_version: "1.2.0" }));
  assert.equal(m.sequence.status, "ready");
  assert.deepEqual(m.warnings, ["sequence_schema_version 1.2.0 is newer than this app (1.0.x): the file opens, and what that apeGmsh added is not shown"]);
  assert.throws(() => read(file("seq-nokey.h5", {}, SEQ)), /\/meta has no string attribute sequence_schema_version/);
  const short: Tree = { sequence: { stages: SEQ["sequence"].stages, changes: { stage: new Int32Array([0, 1]), op: ["add"] } } };
  assert.throws(() => read(file("seq-ragged.h5", {}, short, { sequence_schema_version: "1.0.0" })), /\/sequence\/changes: column stage has 2 rows, not 1|column op has 1 rows, not 2/);
});
