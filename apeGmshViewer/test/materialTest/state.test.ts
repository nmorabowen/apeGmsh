// ADR 0135 D4 and D5: the job from the state, the run's life in the store, and
// the effect that carries it to main and back.

import assert from "node:assert/strict";
import { test } from "node:test";
import { Effects, type Bridge } from "../../src/effects.ts";
import type { DeclRecord, Definitions } from "../../src/model/types.ts";
import { BlobStore } from "../../src/state/blobs.ts";
import { materialTestJobOf } from "../../src/state/materialTest.ts";
import { initialState, reduce, Store, type State } from "../../src/state/store.ts";
import type { Decl, MaterialTestJob, MaterialTestProtocol } from "../../src/state/types.ts";

const protocol: MaterialTestProtocol = { kind: "monotonic", peak: 0.01, steps: 2 };
const decl = (path: string, type: string, tag: number, params: (number | string)[], kind = "uniaxialMaterial"): Decl => ({ path, kind, type, name: path, tag, params } as unknown as Decl);
const record = (row: number, key: string, params: Record<string, unknown>, family = "uniaxialMaterial"): DeclRecord =>
  ({ row, key, family, name: "", synth: false, params: [{ row, type: "x", params, paramsNames: null, refs: { transf: [], integration: [], section: [] } }] }) as unknown as DeclRecord;

function modelState(records: DeclRecord[] = [], tags: { kind: string; first: number; count: number; decl: number }[] = []): State {
  const decls = Object.assign(Object.create(null) as Record<string, Decl>, {
    bar: decl("bar", "Steel02", 1, [420, 200000, 0.01]),
    buckled: decl("buckled", "LadrunoRebarBuckling", 2, [1, "-lsr", 8]),
    sec: decl("sec", "Fiber", 5, [], "section"),
  });
  const definitions = { decls: { hasParams: true, records, tags, rows: {} } } as unknown as Definitions;
  return { ...initialState, decls, definitions };
}

const wrapped = () =>
  modelState(
    [record(0, "opensees/uniaxialMaterial/bar", { E: 200000 }), record(1, "opensees/uniaxialMaterial/buckled", { material: { $decl: "opensees/uniaxialMaterial/bar" } })],
    [{ kind: "uniaxialMaterial", first: 1, count: 1, decl: 0 }, { kind: "uniaxialMaterial", first: 2, count: 1, decl: 1 }],
  );

test("a wrapper brings its material first, with the file's tags", () => {
  const a = materialTestJobOf(wrapped(), "buckled", protocol);
  assert.ok("job" in a, JSON.stringify(a));
  assert.equal(a.job.target, 2);
  assert.deepEqual(a.job.materials.map((m) => m.tag), [1, 2]);
  assert.deepEqual(a.job.materials[1], { type: "LadrunoRebarBuckling", tag: 2, args: [1, "-lsr", 8] });
});

test("what cannot be a job says why", () => {
  assert.match((materialTestJobOf(wrapped(), "none", protocol) as { error: string }).error, /not a declaration/);
  assert.match((materialTestJobOf(wrapped(), "sec", protocol) as { error: string }).error, /uniaxial materials/);
  const loop = modelState([record(0, "a", { m: { $decl: "b" } }), record(1, "b", { m: [{ $decl: "a" }] })], [{ kind: "uniaxialMaterial", first: 1, count: 1, decl: 0 }, { kind: "uniaxialMaterial", first: 2, count: 1, decl: 1 }]);
  assert.match((materialTestJobOf(loop, "bar", protocol) as { error: string }).error, /references itself/);
  const notMaterial = modelState([record(0, "a", { m: { $decl: "s" } }), record(1, "s", {}, "section")], [{ kind: "uniaxialMaterial", first: 1, count: 1, decl: 0 }]);
  assert.match((materialTestJobOf(notMaterial, "bar", protocol) as { error: string }).error, /not a uniaxial material/);
});

test("a file without decl_params sends the material's own line", () => {
  const a = materialTestJobOf({ ...wrapped(), definitions: null }, "buckled", protocol);
  assert.ok("job" in a);
  assert.deepEqual(a.job.materials.map((m) => m.tag), [2]);
});

const blobRef = { id: "1:x", path: "x", dtype: "f64", shape: [1] } as const;
const curves = { strain: blobRef, stress: blobRef, tangent: blobRef };

test("the reducer: a run goes running to ready, a stale answer is ignored, ids are never reused", () => {
  let s = reduce(wrapped(), { type: "materialTestRequested", decl: "bar", protocol });
  assert.equal(s.materialTest.run?.status, "running");
  assert.equal(reduce(s, { type: "materialTestFinished", id: 99, build: "b", curves, error: null }), s);
  s = reduce(s, { type: "materialTestFinished", id: 1, build: "b", curves, error: null });
  assert.equal(s.materialTest.run?.status, "ready");
  assert.equal(s.materialTest.run?.build, "b");
  s = reduce(s, { type: "materialTestCleared" });
  assert.equal(s.materialTest.run, null);
  s = reduce(s, { type: "materialTestRequested", decl: "sec", protocol });
  assert.equal(s.materialTest.run?.id, 2);
  assert.equal(s.materialTest.run?.status, "failed");
});

test("opening another model drops the run", () => {
  const s = reduce(wrapped(), { type: "materialTestRequested", decl: "bar", protocol });
  assert.equal(reduce(s, { type: "fileOpened", artifact: "model", path: "other.h5" }).materialTest.run, null);
});

const tick = () => new Promise((f) => setTimeout(f, 0));

function harness(bridge: Partial<Bridge>) {
  const store = new Store();
  const blobs = new BlobStore();
  new Effects(store, blobs, bridge as Bridge);
  // the model's declarations, set through the one test door the reducer has: a state seeded by hand
  const seeded = wrapped();
  (store as unknown as { state: State }).state = seeded;
  return { store, blobs };
}

test("the effect runs the job once and stores the curves as blobs", async () => {
  const jobs: MaterialTestJob[] = [];
  const { store, blobs } = harness({
    materialTestRun: async (job, seq) => {
      jobs.push(job);
      return { ok: true, seq, build: "b", strain: Float64Array.of(0, 1), stress: Float64Array.of(0, 2), tangent: Float64Array.of(2, 2) };
    },
  });
  store.dispatch({ type: "materialTestRequested", decl: "buckled", protocol });
  await tick();
  assert.equal(jobs.length, 1);
  const run = store.get().materialTest.run!;
  assert.equal(run.status, "ready");
  assert.deepEqual([...blobs.get(run.curves!.stress)], [0, 2]);
});

test("a failure reaches the state; clearing a run in flight cancels it and ignores its answer", async () => {
  let cancels = 0;
  let answer: (a: { ok: false; seq: number; build: null; error: string }) => void = () => undefined;
  const { store } = harness({ materialTestRun: (_j, _s) => new Promise((f) => (answer = f)), materialTestCancel: async () => void cancels++ });
  store.dispatch({ type: "materialTestRequested", decl: "bar", protocol });
  await tick();
  answer({ ok: false, seq: 1, build: null, error: "bad" });
  await tick();
  assert.equal(store.get().materialTest.run?.error, "bad");
  store.dispatch({ type: "materialTestRequested", decl: "bar", protocol });
  await tick();
  store.dispatch({ type: "materialTestCleared" });
  assert.equal(cancels, 1);
  answer({ ok: false, seq: 2, build: null, error: "late" });
  await tick();
  assert.equal(store.get().materialTest.run, null);
});

test("a window with no door says so", async () => {
  const { store } = harness({});
  store.dispatch({ type: "materialTestRequested", decl: "bar", protocol });
  await tick();
  assert.match(store.get().materialTest.run!.error!, /cannot run/);
});
