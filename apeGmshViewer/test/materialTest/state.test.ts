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
  assert.match((materialTestJobOf(wrapped(), "sec", protocol) as { error: string }).error, /the tester runs uniaxial and nD materials/);
  const loop = modelState([record(0, "a", { m: { $decl: "b" } }), record(1, "b", { m: [{ $decl: "a" }] })], [{ kind: "uniaxialMaterial", first: 1, count: 1, decl: 0 }, { kind: "uniaxialMaterial", first: 2, count: 1, decl: 1 }]);
  assert.match((materialTestJobOf(loop, "bar", protocol) as { error: string }).error, /references itself/);
  const notMaterial = modelState([record(0, "a", { m: { $decl: "s" } }), record(1, "s", {}, "section")], [{ kind: "uniaxialMaterial", first: 1, count: 1, decl: 0 }]);
  assert.match((materialTestJobOf(notMaterial, "bar", protocol) as { error: string }).error, /which is not a material/);
});

test("a file without decl_params sends the material's own line", () => {
  const a = materialTestJobOf({ ...wrapped(), definitions: null }, "buckled", protocol);
  assert.ok("job" in a);
  assert.deepEqual(a.job.materials.map((m) => m.tag), [2]);
});

const blobRef = { id: "1:x", path: "x", dtype: "f64", shape: [1] } as const;
const curves = { strain: blobRef, stress: blobRef, tangent: blobRef };
const points = { step: [0, 1], strain: [0, 1], stress: [0, 2], tangent: [2, 2], energy: [0, 1] };
const last = (s: State) => s.materialTest.runs.at(-1);

test("the reducer: a run goes running to ready, a stale answer is ignored, ids are never reused", () => {
  let s = reduce(wrapped(), { type: "materialTestRequested", decl: "bar", protocol });
  assert.equal(last(s)?.status, "running");
  assert.equal(reduce(s, { type: "materialTestFinished", id: 99, build: "b", curves, points, error: null }), s);
  s = reduce(s, { type: "materialTestFinished", id: 1, build: "b", curves, points, error: null });
  assert.equal(last(s)?.status, "ready");
  assert.equal(last(s)?.build, "b");
  assert.equal(last(s)?.points, points);
  s = reduce(s, { type: "materialTestCleared" });
  assert.deepEqual(s.materialTest.runs, []);
  s = reduce(s, { type: "materialTestRequested", decl: "sec", protocol });
  assert.equal(last(s)?.id, 2);
  assert.equal(last(s)?.status, "failed");
});

test("runs are kept for overlay: oldest first, a running one superseded, at most six, one removed by id", () => {
  let s = wrapped();
  const finish = (id: number) => (s = reduce(s, { type: "materialTestFinished", id, build: "b", curves, points, error: null }));
  s = reduce(s, { type: "materialTestRequested", decl: "bar", protocol });
  finish(1);
  s = reduce(s, { type: "materialTestRequested", decl: "bar", protocol });
  s = reduce(s, { type: "materialTestRequested", decl: "buckled", protocol });
  assert.deepEqual(s.materialTest.runs.map((r) => [r.id, r.status]), [[1, "ready"], [3, "running"]], "the request supersedes run 2, still going; the ready run stays");
  finish(3);
  for (let i = 0; i < 6; i++) {
    s = reduce(s, { type: "materialTestRequested", decl: "bar", protocol });
    finish(s.materialTest.nextId - 1);
  }
  assert.equal(s.materialTest.runs.length, 6);
  assert.deepEqual(s.materialTest.runs.map((r) => r.id), [4, 5, 6, 7, 8, 9], "the oldest leave first");
  s = reduce(s, { type: "materialTestRemoved", id: 6 });
  assert.deepEqual(s.materialTest.runs.map((r) => r.id), [4, 5, 7, 8, 9]);
  assert.equal(reduce(s, { type: "materialTestRemoved", id: 6 }), s, "removing a run not kept is no change");
});

test("a save is asked for a ready run only, one at a time, and answered for that run", () => {
  let s = reduce(wrapped(), { type: "materialTestRequested", decl: "bar", protocol });
  assert.equal(reduce(s, { type: "materialTestSaveRequested", id: 1 }), s, "a running run has nothing to save");
  s = reduce(s, { type: "materialTestFinished", id: 1, build: "b", curves, points, error: null });
  s = reduce(s, { type: "materialTestSaveRequested", id: 1 });
  assert.equal(s.materialTest.saving, 1);
  assert.equal(reduce(s, { type: "materialTestSaveRequested", id: 1 }), s, "one save at a time");
  assert.equal(reduce(s, { type: "materialTestSaved", id: 7, notice: "x" }), s, "an answer for another save is ignored");
  s = reduce(s, { type: "materialTestSaved", id: 1, notice: "saved run 1 to x.csv" });
  assert.equal(s.materialTest.saving, null);
  assert.equal(s.materialTest.notice, "saved run 1 to x.csv");
  const asked = reduce(s, { type: "materialTestSaveRequested", id: 1 });
  assert.equal(reduce(asked, { type: "materialTestRemoved", id: 1 }).materialTest.saving, null, "a save of a removed run is forgotten");
});

test("opening another model drops the runs, the form and the notice", () => {
  const s = reduce(wrapped(), { type: "materialTestRequested", decl: "bar", protocol });
  const next = reduce(s, { type: "fileOpened", artifact: "model", path: "other.h5" }).materialTest;
  assert.deepEqual([next.runs, next.form, next.saving, next.notice], [[], null, null, null]);
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
  const run = last(store.get())!;
  assert.equal(run.status, "ready");
  assert.deepEqual([...blobs.get(run.curves!.stress)], [0, 2]);
  assert.deepEqual(run.points, { step: [0, 1], strain: [0, 1], stress: [0, 2], tangent: [2, 2], energy: [0, 1] }, "the drawn curve, with the work done, reaches the state");
});

test("a failure reaches the state; clearing a run in flight cancels it and ignores its answer", async () => {
  let cancels = 0;
  let answer: (a: { ok: false; seq: number; build: null; error: string }) => void = () => undefined;
  const { store } = harness({ materialTestRun: (_j, _s) => new Promise((f) => (answer = f)), materialTestCancel: async () => void cancels++ });
  store.dispatch({ type: "materialTestRequested", decl: "bar", protocol });
  await tick();
  answer({ ok: false, seq: 1, build: null, error: "bad" });
  await tick();
  assert.equal(last(store.get())?.error, "bad");
  store.dispatch({ type: "materialTestRequested", decl: "bar", protocol });
  await tick();
  store.dispatch({ type: "materialTestCleared" });
  assert.equal(cancels, 1);
  answer({ ok: false, seq: 2, build: null, error: "late" });
  await tick();
  assert.deepEqual(store.get().materialTest.runs, []);
});

const answered = (seq: number) => ({ ok: true as const, seq, build: "b", strain: Float64Array.of(0, 1), stress: Float64Array.of(0, 2), tangent: Float64Array.of(2, 2) });

test("a finished run is not cancelled; removing the running one is", async () => {
  let cancels = 0;
  let answer: (a: ReturnType<typeof answered>) => void = () => undefined;
  const { store } = harness({ materialTestRun: (_j, _s) => new Promise((f) => (answer = f)), materialTestCancel: async () => void cancels++ });
  store.dispatch({ type: "materialTestRequested", decl: "bar", protocol });
  await tick();
  answer(answered(1));
  await tick();
  assert.equal(cancels, 0, "it finished: nothing to stop");
  store.dispatch({ type: "materialTestRequested", decl: "bar", protocol });
  await tick();
  store.dispatch({ type: "materialTestRemoved", id: 2 });
  assert.equal(cancels, 1);
  assert.deepEqual(store.get().materialTest.runs.map((r) => r.id), [1]);
});

test("the effect saves the asked run's full arrays as CSV through main, and says where", async () => {
  const saved: [string, string][] = [];
  const { store } = harness({
    materialTestRun: async (_j, seq) => answered(seq),
    materialTestSaveCsv: async (text, name) => {
      saved.push([text, name]);
      return { ok: true, path: "C:/out/bar-run1.csv" };
    },
  });
  store.dispatch({ type: "materialTestRequested", decl: "bar", protocol });
  await tick();
  store.dispatch({ type: "materialTestSaveRequested", id: 1 });
  await tick();
  await tick();
  assert.equal(saved.length, 1);
  assert.equal(saved[0]![1], "bar-run1.csv");
  assert.match(saved[0]![0], /^# bar \(Steel02\)\n# fork build b; monotonic to 0.01 in 2 steps/);
  assert.match(saved[0]![0], /step,strain,stress,tangent,work\n0,0,0,2,0\n1,1,2,2,1\n$/);
  assert.equal(store.get().materialTest.notice, "saved run 1 to C:/out/bar-run1.csv");
  assert.equal(store.get().materialTest.saving, null);
});

test("a cancelled save, and a window that cannot save, say so", async () => {
  const a = harness({ materialTestRun: async (_j, seq) => answered(seq), materialTestSaveCsv: async () => ({ ok: false, cancelled: true, error: "cancelled" }) });
  a.store.dispatch({ type: "materialTestRequested", decl: "bar", protocol });
  await tick();
  a.store.dispatch({ type: "materialTestSaveRequested", id: 1 });
  await tick();
  await tick();
  assert.equal(a.store.get().materialTest.notice, "the save was cancelled");
  const b = harness({ materialTestRun: async (_j, seq) => answered(seq) });
  b.store.dispatch({ type: "materialTestRequested", decl: "bar", protocol });
  await tick();
  b.store.dispatch({ type: "materialTestSaveRequested", id: 1 });
  await tick();
  await tick();
  assert.equal(b.store.get().materialTest.notice, "this window cannot save files");
});

test("a window with no door says so", async () => {
  const { store } = harness({});
  store.dispatch({ type: "materialTestRequested", decl: "bar", protocol });
  await tick();
  assert.match(last(store.get())!.error!, /cannot run/);
});

test("F8: a failed or closed model keeps the run ids going, as another model does", () => {
  const s = reduce(wrapped(), { type: "materialTestRequested", decl: "bar", protocol });
  const failed = reduce(s, { type: "fileFailed", artifact: "model", path: "other.h5", error: "bad" }).materialTest;
  assert.deepEqual([failed.runs, failed.nextId], [[], 2]);
  const loaded = reduce(s, { type: "fileOpened", artifact: "model", path: "s.h5" });
  const closed = reduce({ ...loaded, artifacts: { ...loaded.artifacts, model: { ...loaded.artifacts.model!, status: "ready" } } }, { type: "fileClosed", artifact: "model" }).materialTest;
  assert.deepEqual([closed.runs, closed.nextId], [[], 2]);
});

test("F8: a run cancelled by a failed model open cannot finish the next run with its late answer", async () => {
  const answers: ((a: ReturnType<typeof answered>) => void)[] = [];
  const { store } = harness({ materialTestRun: (_j, _s) => new Promise((f) => answers.push(f)), materialTestCancel: async () => undefined });
  store.dispatch({ type: "materialTestRequested", decl: "bar", protocol });
  await tick();
  store.dispatch({ type: "fileFailed", artifact: "model", path: "other.h5", error: "bad" });
  // the model's declarations again (a later open), with the tester state the failure left
  (store as unknown as { state: State }).state = { ...wrapped(), materialTest: store.get().materialTest };
  store.dispatch({ type: "materialTestRequested", decl: "bar", protocol });
  await tick();
  assert.equal(last(store.get())?.id, 2, "the id is not reused");
  answers[0]!(answered(1)); // the cancelled run's late answer
  await tick();
  assert.equal(last(store.get())?.status, "running", "the late answer finished nothing");
  answers[1]!({ ...answered(2), seq: 7 }); // an answer for another seq
  await tick();
  assert.equal(last(store.get())?.status, "running", "an answer is matched on its seq");
});
