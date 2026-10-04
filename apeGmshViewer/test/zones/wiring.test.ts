// V2f phase 2b: the /geometry and /provenance zones in the store, the
// session_id pairing (the stale-geometry notice), the phase axis, and
// go-to-source through the effects.
//
// Oracles: the h5py fixtures (zones.h5, zones.geometry.h5: a unit cube whose
// faces total area 6), the pairing rule of h5-schema.md (equal session_id or
// not drawn), and the source lines the fixture records, checked against
// examples/shoebuckle_arch.py in provenance.test.ts.

import assert from "node:assert/strict";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { Effects, parseRefusal, type Bridge } from "../../src/effects.ts";
import type { ModelFile } from "../../src/model/types.ts";
import { readGeometry, type GeometryZone } from "../../src/reader/geometry.ts";
import { openModel } from "../../src/reader/node.ts";
import { readProvenance, type ProvenanceZone } from "../../src/reader/provenance.ts";
import type { H5Module } from "../../src/reader/read.ts";
import { BlobStore } from "../../src/state/blobs.ts";
import { loadGeometry } from "../../src/state/geometry.ts";
import { loadModel } from "../../src/state/load.ts";
import { initialState, reduce } from "../../src/state/reduce.ts";
import { geometryPairing, noticesOf, refusalsOf, sourceFailure, sourceFor, warningsOf } from "../../src/state/selectors.ts";
import { Store } from "../../src/state/store.ts";
import type { State } from "../../src/state/types.ts";

await h5wasm.ready;
const h5 = h5wasm as unknown as H5Module;
const fixtures = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "fixtures");
const MODEL = join(fixtures, "zones.h5");
const GEOMETRY = join(fixtures, "zones.geometry.h5");
const PLAIN = join(fixtures, "shoebuckle.h5");
const SCRIPT = "/work/apeGmsh/examples/shoebuckle_arch.py";

const model: ModelFile = await openModel(MODEL);
const plain: ModelFile = await openModel(PLAIN);
const provenance: ProvenanceZone = readProvenance(h5, MODEL)!;
const geometry: GeometryZone = readGeometry(h5, GEOMETRY)!;

const withModel = (s: State, m: ModelFile, p: ProvenanceZone | null = null) =>
  reduce(s, { type: "fileLoaded", artifact: "model", load: loadModel(m, new BlobStore(), p) });
const withGeometry = (s: State, g: GeometryZone) =>
  reduce(s, { type: "fileLoaded", artifact: "geometry", load: loadGeometry(g, new BlobStore(), 0, 0) });

test("loadGeometry: the cube's segments, triangles and points, and its bounds", () => {
  const blobs = new BlobStore();
  const { geometry: g, info } = loadGeometry(geometry, blobs, 123, 4);
  assert.deepEqual(g.curvePositions.shape, [12 * 31, 6]);
  assert.deepEqual(g.surfacePositions.shape, [12, 9]);
  assert.deepEqual(g.pointPositions.shape, [8, 3]);
  assert.deepEqual(g.center, [0.5, 0.5, 0.5]);
  assert.ok(Math.abs(g.radius - Math.sqrt(3) / 2) < 1e-12);
  assert.equal(info.sessionId, "0f8e5c1a2b3d4e5f8a9b0c1d2e3f4a5b");
  assert.deepEqual(info.zones["geometry"], { status: "ready", version: "1.0.0" });
  // Closed form through the blob the viewport draws: total face area 6.
  const t = blobs.f32(g.surfacePositions);
  let area = 0;
  for (let i = 0; i < t.length; i += 9) {
    const u = [t[i + 3]! - t[i]!, t[i + 4]! - t[i + 1]!, t[i + 5]! - t[i + 2]!];
    const v = [t[i + 6]! - t[i]!, t[i + 7]! - t[i + 1]!, t[i + 8]! - t[i + 2]!];
    area += Math.hypot(u[1]! * v[2]! - u[2]! * v[1]!, u[2]! * v[0]! - u[0]! * v[2]!, u[0]! * v[1]! - u[1]! * v[0]!) / 2;
  }
  assert.ok(Math.abs(area - 6) < 1e-6, `area ${area}`);
});

test("pairing: equal session_ids draw the geometry, and the phase axis is geometry -> mesh", () => {
  for (const s of [withGeometry(withModel(initialState, model), geometry), withModel(withGeometry(initialState, geometry), model)]) {
    assert.deepEqual(geometryPairing(s), { draw: true, notice: null });
    assert.deepEqual(noticesOf(s), []);
    assert.deepEqual(s.phase.axis, [{ kind: "geometry" }, { kind: "mesh" }]);
    assert.deepEqual(s.phase.at, { kind: "mesh" });
  }
});

test("session_id mismatch: the geometry is not drawn, and the notice says why", () => {
  const stale = { ...geometry, sessionId: "ffffffffffffffffffffffffffffffff" };
  const s = withGeometry(withModel(initialState, model), stale);
  const p = geometryPairing(s);
  assert.equal(p.draw, false);
  assert.match(p.notice!, /^Geometry not drawn: zones\.geometry\.h5 is stale or foreign: it was written by session ffff.*, the model by session 0f8e5c1a/);
  assert.deepEqual(noticesOf(s), [p.notice]);
  assert.deepEqual(s.phase.axis, [{ kind: "mesh" }], "a stale geometry is not a phase");
  // A geometry phase already shown leaves the axis when the model it pairs with is replaced.
  let t = withGeometry(withModel(initialState, model), geometry);
  t = reduce(t, { type: "setPhase", at: { kind: "geometry" } });
  t = withModel(t, plain);
  assert.deepEqual(t.phase, { axis: [{ kind: "mesh" }], at: { kind: "mesh" } });
  assert.match(noticesOf(t)[0]!, /model file has no session_id/);
});

test("a geometry opened with no model is drawn alone, as the only phase", () => {
  const s = withGeometry(initialState, geometry);
  assert.deepEqual(geometryPairing(s), { draw: true, notice: null });
  assert.deepEqual(s.phase, { axis: [{ kind: "geometry" }], at: { kind: "geometry" } });
  const closed = reduce(s, { type: "fileClosed", artifact: "geometry" });
  assert.equal(closed.geometry, null);
  assert.deepEqual(closed.phase, { axis: [], at: null });
});

test("provenance joins named declarations; the inspector's source is the declaring line", () => {
  const s = withModel(initialState, model, provenance);
  const sec = s.decls["opensees/section/W_section"]!;
  assert.deepEqual(sec.provenance, { file: SCRIPT, line: 293, function: "declare_model", sha256: provenance.files.sha256[0], script: { file: SCRIPT, line: 451 } });
  assert.equal(s.decls["opensees/uniaxialMaterial/Steel"]!.provenance!.line, 285);
  const r = sourceFor(s, "opensees/section/W_section");
  assert.ok(r.ok);
  assert.equal((r as { label: string }).label, "shoebuckle_arch.py:293");
  assert.deepEqual(s.artifacts.model!.zones["provenance"], { status: "ready", version: "1.0.0" });
});

test("why go-to-source is off: unnamed, element, no zone, no record", () => {
  const s = withModel(initialState, model, provenance);
  assert.match((sourceFor(s, "opensees/geomTransf/#0") as { reason: string }).reason, /unnamed declaration is not joined/);
  assert.match((sourceFor(s, s.mesh!.elements[0]!) as { reason: string }).reason, /an element has no provenance key yet/);
  const none = withModel(initialState, plain);
  assert.match((sourceFor(none, "opensees/section/#0") as { reason: string }).reason, /no \/provenance zone/);
  const pruned: ProvenanceZone = { ...provenance, records: { ...provenance.records, path: provenance.records.path.map((p) => (p.endsWith("W_section") ? "opensees/section/Other" : p)) } };
  assert.match((sourceFor(withModel(initialState, model, pruned), "opensees/section/W_section") as { reason: string }).reason, /no \/provenance record for opensees\/section\/W_section/);
});

test("an unnamed #k is never joined, even when the two keys are spelled alike", () => {
  // The app's #0 is the reader's first listed geomTransf; a record keyed #0
  // would be the first *declared*. Same text, possibly another object.
  const alike: ProvenanceZone = {
    ...provenance,
    records: { ...provenance.records, path: provenance.records.path.map((p) => (p === "opensees/geomTransf/#1" ? "opensees/geomTransf/#0" : p)) },
  };
  const s = withModel(initialState, model, alike);
  assert.ok("opensees/geomTransf/#0" in s.decls);
  assert.equal(s.decls["opensees/geomTransf/#0"]!.provenance, undefined);
});

// ---- effects ----------------------------------------------------------------

function harness(over: Partial<Bridge> = {}) {
  const calls: { goto: [string, number][]; geometry: string[] } = { goto: [], geometry: [] };
  const bridge = {
    openModel: async (path: string) => ({ ok: true, model: path === PLAIN ? plain : model, provenance: { ok: true, zone: path === PLAIN ? null : provenance } }),
    openGeometry: async (path: string) => {
      calls.geometry.push(path);
      return { ok: true, geometry, sizeBytes: 1, readMs: 1 };
    },
    goToSource: async (file: string, line: number) => {
      calls.goto.push([file, line]);
      return { ok: true };
    },
    ...over,
  } as unknown as Bridge;
  const store = new Store();
  const effects = new Effects(store, new BlobStore(), bridge);
  return { store, effects, calls };
}

const settle = () => new Promise((r) => setTimeout(r, 10));

test("effects: an open set reads the model with its provenance and the geometry sibling", async () => {
  const { store, effects, calls } = harness();
  effects.openSet({ model: MODEL, geometry: GEOMETRY, results: null });
  await settle();
  const s = store.get();
  assert.deepEqual(calls.geometry, [GEOMETRY]);
  assert.equal(s.artifacts.geometry?.status, "ready");
  assert.equal(geometryPairing(s).draw, true);
  assert.ok(s.decls["opensees/section/W_section"]!.provenance);
  effects.openSet({ model: MODEL, geometry: null, results: null });
  await settle();
  assert.equal(store.get().geometry, null);
  assert.equal(store.get().artifacts.geometry, null);
});

test("effects: requestSource opens the editor at the declaring line", async () => {
  const { store, effects, calls } = harness();
  await effects.open(MODEL);
  store.dispatch({ type: "requestSource", decl: "opensees/section/W_section" });
  await settle();
  assert.deepEqual(calls.goto, [[SCRIPT, 293]]);
  assert.deepEqual(store.get().source.last, { decl: "opensees/section/W_section", ok: true, reason: null });
  // The same request again is a new request (a second click opens it again).
  store.dispatch({ type: "requestSource", decl: "opensees/section/W_section" });
  await settle();
  assert.equal(calls.goto.length, 2);
});

test("effects: a failed jump, and a declaration with no source, are answered with the reason", async () => {
  const { store, effects, calls } = harness({ goToSource: async () => ({ ok: false, reason: `missing: ${SCRIPT}` }) } as Partial<Bridge>);
  await effects.open(MODEL);
  store.dispatch({ type: "requestSource", decl: "opensees/section/W_section" });
  await settle();
  assert.equal(sourceFailure(store.get(), "opensees/section/W_section"), `missing: ${SCRIPT}`);
  store.dispatch({ type: "requestSource", decl: "opensees/geomTransf/#0" });
  await settle();
  assert.match(sourceFailure(store.get(), "opensees/geomTransf/#0")!, /unnamed declaration/);
  assert.equal(calls.goto.length, 0);
  assert.throws(() => store.dispatch({ type: "requestSource", decl: "opensees/section/nope" }), /not a declaration/);
});

test("effects: a refused geometry zone is a refusal sentence, not a blank view", async () => {
  const { store, effects } = harness({
    openGeometry: async () => ({ ok: false, error: "geometry_schema_version 2.0.0: this app reads major 1 only" }),
  } as Partial<Bridge>);
  effects.openSet({ model: MODEL, geometry: GEOMETRY, results: null });
  await settle();
  const s = store.get();
  assert.equal(s.artifacts.geometry?.status, "failed");
  assert.equal(s.geometry, null);
  assert.ok(refusalsOf(s).some((r) => /geometry file written by a newer apeGmsh: its geometry zone is version 2\.0\.0, this app reads 1\.0 and any later 1\.x/.test(r)), refusalsOf(s).join("\n"));
});

test("effects: a refused /provenance does not stop the model; go-to-source says why", async () => {
  const { store, effects } = harness({
    openModel: async () => ({ ok: true, model, provenance: { ok: false, error: "provenance_schema_version 2.0.0: this app reads major 1 only" } }),
  } as Partial<Bridge>);
  assert.equal(await effects.open(MODEL), true);
  const s = store.get();
  assert.equal(s.artifacts.model?.status, "ready");
  assert.equal(s.artifacts.model?.zones["provenance"]?.status, "refused");
  assert.match((sourceFor(s, "opensees/section/W_section") as { reason: string }).reason, /\/provenance zone was refused: this app reads major 1 only/);
});

test("parseRefusal reads the ADR 0112 zones too", () => {
  assert.deepEqual(parseRefusal("geometry_schema_version 2.1.0: this app reads major 1 only"), {
    zone: "geometry", version: "2.1.0", accepted: "1.0 and any later 1.x", newer: true, reason: "this app reads major 1 only",
  });
  assert.equal(parseRefusal("provenance_schema_version 0.9.0: this app reads major 1 only")?.newer, false);
});

// ---- Fable's review of #1320 (each fails on bc7c56c2) -----------------------

test("review 1: a request after a model re-read is answered (the request seq never resets)", async () => {
  const { store, effects, calls } = harness();
  await effects.open(MODEL);
  store.dispatch({ type: "requestSource", decl: "opensees/section/W_section" });
  await settle();
  await effects.open(MODEL); // D1 rewrites the file every run
  store.dispatch({ type: "requestSource", decl: "opensees/section/W_section" });
  await settle();
  assert.equal(calls.goto.length, 2, "both requests reach the editor");
  assert.deepEqual(store.get().source.last, { decl: "opensees/section/W_section", ok: true, reason: null });
});

test("review 2: a newer same-major /provenance minor opens with exactly one banner", () => {
  const banner = "provenance_schema_version 1.3.0 is newer than this app (1.0.x): the file opens, and what that apeGmsh added is not shown";
  const newer: ProvenanceZone = { ...provenance, version: "1.3.0", warnings: [banner] };
  const s = withModel(initialState, model, newer);
  assert.deepEqual(warningsOf(s), [banner]);
  assert.ok(sourceFor(s, "opensees/section/W_section").ok, "the zone still serves go-to-source");
});

test("review 3: a malformed /provenance reports as malformed, naming the path, never as an older apeGmsh", async () => {
  const error = "/provenance/files/sha256[0] is not a hex sha256";
  const { store, effects } = harness({ openModel: async () => ({ ok: true, model, provenance: { ok: false, error } }) } as Partial<Bridge>);
  assert.equal(await effects.open(MODEL), true);
  const s = store.get();
  assert.deepEqual(s.artifacts.model!.zones["provenance"], { status: "malformed", reason: error });
  assert.deepEqual(refusalsOf(s), [], "no 'written by an older apeGmsh' sentence");
  assert.ok(warningsOf(s).some((w) => w.includes("/provenance is malformed") && w.includes("/provenance/files/sha256[0]")), warningsOf(s).join("\n"));
  assert.match((sourceFor(s, "opensees/section/W_section") as { reason: string }).reason, /\/provenance zone is malformed: \/provenance\/files\/sha256\[0\]/);
});

function rerunHarness(windowMs: number) {
  const state = { sid: "0f8e5c1a2b3d4e5f8a9b0c1d2e3f4a5b", geometrySid: null as string | null };
  const reads = { model: 0, geometry: 0 };
  const bridge = {
    openModel: async () => {
      reads.model++;
      return { ok: true, model: { ...model, meta: { ...model.meta, session_id: state.sid } }, provenance: { ok: true, zone: provenance } };
    },
    openGeometry: async () => {
      reads.geometry++;
      return { ok: true, geometry: { ...geometry, sessionId: state.geometrySid ?? state.sid }, sizeBytes: 1, readMs: 1 };
    },
  } as unknown as Bridge;
  const store = new Store();
  const effects = new Effects(store, new BlobStore(), bridge, { setWindowMs: windowMs });
  return { state, reads, store, effects };
}

test("review 4: a D1 re-run rewrites both siblings; they reload as one set and no stale notice ever shows", async () => {
  const { state, reads, store, effects } = rerunHarness(60);
  effects.openSet({ model: MODEL, geometry: GEOMETRY, results: null });
  await settle();
  assert.equal(geometryPairing(store.get()).draw, true);
  const notices: string[] = [];
  const off = store.subscribe((s) => notices.push(...noticesOf(s)));
  state.sid = "aaaabbbbccccddddeeeeffff00001111"; // the next run's session
  store.dispatch({ type: "fileChanged", path: MODEL });
  await new Promise((r) => setTimeout(r, 20));
  store.dispatch({ type: "fileChanged", path: GEOMETRY });
  await new Promise((r) => setTimeout(r, 250));
  off();
  assert.deepEqual(notices, [], "no state between the two re-reads pairs a new file with an old one");
  const s = store.get();
  assert.equal(s.artifacts.model!.sessionId, state.sid);
  assert.equal(s.geometry!.sessionId, state.sid);
  assert.deepEqual(geometryPairing(s), { draw: true, notice: null });
  assert.deepEqual(reads, { model: 2, geometry: 2 }, "one re-read of each");
  effects.dispose();
});

test("review 4: a model rewritten alone still shows the notice once its window closes", async () => {
  const { state, store, effects } = rerunHarness(30);
  effects.openSet({ model: MODEL, geometry: GEOMETRY, results: null });
  await settle();
  state.geometrySid = state.sid; // the geometry file is not rewritten
  state.sid = "aaaabbbbccccddddeeeeffff00001111";
  store.dispatch({ type: "fileChanged", path: MODEL });
  await new Promise((r) => setTimeout(r, 150));
  assert.match(noticesOf(store.get())[0] ?? "", /stale or foreign/);
  effects.dispose();
});
