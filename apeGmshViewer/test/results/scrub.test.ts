// The time axis (V4b, #1728): the results position `{stage, step}` on the
// phase axis, the lazy step fetch (latest wins: a slow reply for step k never
// paints over step k+1), and the reload seams (a re-read clamps the step, or
// leaves a stage that no longer exists).

import assert from "node:assert/strict";
import { test } from "node:test";
import { Effects, type Bridge } from "../../src/effects.ts";
import { BlobStore } from "../../src/state/blobs.ts";
import { timelineOf } from "../../src/state/selectors.ts";
import { reduce, Store } from "../../src/state/store.ts";
import { stagedModelLoad, stagedResults, stagedState } from "./staged.ts";

const base = await stagedState();
const r = await stagedResults();
const load = await stagedModelLoad();

test("setPhase to a results stage, then setResultStep moves within it; out-of-range steps are refused", () => {
  let s = reduce(base, { type: "setPhase", at: { kind: "results", stage: 2, step: 0 } });
  s = reduce(s, { type: "setResultStep", step: 3 });
  assert.deepEqual(s.phase.at, { kind: "results", stage: 2, step: 3 });
  const t = timelineOf(s)!;
  assert.deepEqual([t.stage, t.step, t.steps, t.time, t.kind], [2, 3, 4, 1, "static"]);
  assert.deepEqual(t.stages.map((x) => x.label), ["base", "erect", "unprop", "mode_1"]);
  assert.throws(() => reduce(s, { type: "setResultStep", step: 4 }), RangeError);
  assert.throws(() => reduce(s, { type: "setResultStep", step: -1 }), RangeError);
  assert.throws(() => reduce(s, { type: "setResultStep", step: 1.5 }), RangeError);
  assert.throws(() => reduce(base, { type: "setResultStep", step: 0 }), /not a results stage/);
  assert.throws(() => reduce(s, { type: "setPhase", at: { kind: "results", stage: 0, step: 9 } }), RangeError);
  assert.equal(timelineOf(reduce(s, { type: "setPhase", at: { kind: "mesh" } })), null);
});

test("reload: a re-read keeps the stage and clamps a step past its end; a stage that is gone leaves the results phase", () => {
  let s = reduce(base, { type: "setPhase", at: { kind: "results", stage: 2, step: 0 } });
  s = reduce(s, { type: "setResultStep", step: 3 });
  // The same file re-read with stage 2 shorter: clamped to its last step.
  const shorter = { ...r.info, stages: r.info.stages.map((st, i) => (i === 2 ? { ...st, time: st.time.slice(0, 2) } : st)) };
  const clamped = reduce(s, { type: "resultsLoaded", ...r, info: shorter });
  assert.deepEqual(clamped.phase.at, { kind: "results", stage: 2, step: 1 });
  // Re-read with index 2 now another stage (renamed): the phase leaves it for the mesh.
  const renamed = { ...r.info, stages: r.info.stages.map((st, i) => (i === 2 ? { ...st, name: "other" } : st)) };
  assert.deepEqual(reduce(s, { type: "resultsLoaded", ...r, info: renamed }).phase.at, { kind: "mesh" });
  // Re-read with fewer stages: the same.
  const fewer = { ...r.info, stages: r.info.stages.slice(0, 2) };
  assert.deepEqual(reduce(s, { type: "resultsLoaded", ...r, info: fewer }).phase.at, { kind: "mesh" });
});

test("a step status for a step no longer shown is dropped by the reducer", () => {
  let s = reduce(base, { type: "setPhase", at: { kind: "results", stage: 0, step: 0 } });
  s = reduce(s, { type: "setResultStep", step: 2 });
  assert.equal(reduce(s, { type: "resultStep", status: { stage: 0, step: 1, status: "ready", error: null } }), s);
  assert.equal(reduce(s, { type: "resultStep", status: { stage: 0, step: 2, status: "ready", error: null } }).resultStep?.status, "ready");
});

/** A bridge whose step replies the test releases by hand, in any order. */
function harness() {
  let opens = 0;
  type Reply = Awaited<ReturnType<NonNullable<Bridge["readResultsStep"]>>>;
  const pending: { req: { stage: string; step: number; seq: number }; resolve: (r: Reply) => void }[] = [];
  const bridge = {
    readResultsStep: (req: { path: string; stage: string; step: number; seq: number }) =>
      new Promise<Reply>((resolve) => pending.push({ req, resolve })),
    openResults: async () => {
      opens++;
      return { ok: true as const, meta: r.info, warnings: [], sizeBytes: r.sizeBytes, readMs: 0 };
    },
  } as unknown as Bridge;
  const store = new Store();
  // The loaded pair, as the effects leave it.
  store.dispatch({ type: "fileLoaded", artifact: "model", load });
  store.dispatch({ type: "fileOpened", artifact: "results", path: r.info.path });
  store.dispatch({ type: "resultsLoaded", ...r });
  const effects = new Effects(store, new BlobStore(), bridge);
  const reply = (i: number) => {
    const p = pending[i]!;
    p.resolve({ ok: true, seq: p.req.seq, stage: p.req.stage, step: p.req.step, nodal: { displacement_x: Float64Array.of(p.req.step) } });
  };
  return { store, effects, pending, reply, opens: () => opens };
}

const tick = () => new Promise((res) => setTimeout(res, 0));

test("latest wins: an out-of-order reply for step k never paints over step k+1", async () => {
  const { store, effects, pending, reply } = harness();
  store.dispatch({ type: "setPhase", at: { kind: "results", stage: 2, step: 0 } });
  await tick();
  store.dispatch({ type: "setResultStep", step: 1 });
  await tick();
  store.dispatch({ type: "setResultStep", step: 2 });
  await tick();
  assert.deepEqual(pending.map((p) => p.req.step), [0, 1, 2], "one fetch per step shown");
  assert.equal(store.get().resultStep?.status, "loading");
  // The newest reply lands first, then the two older ones, slow.
  reply(2);
  await tick();
  assert.equal(effects.currentStep()?.step, 2);
  assert.equal(store.get().resultStep?.status, "ready");
  reply(1);
  reply(0);
  await tick();
  assert.equal(effects.currentStep()?.step, 2, "the slow reply for step 1 did not replace step 2");
  assert.deepEqual(Array.from(effects.currentStep()!.nodal["displacement_x"]!), [2]);
  assert.deepEqual(store.get().resultStep, { stage: 2, step: 2, status: "ready", error: null });
  effects.dispose();
});

test("latest wins also across a step that comes back: k, k+1, k, with the first k replying late", async () => {
  const { store, effects, pending, reply } = harness();
  store.dispatch({ type: "setPhase", at: { kind: "results", stage: 2, step: 0 } });
  await tick();
  store.dispatch({ type: "setResultStep", step: 1 });
  await tick();
  store.dispatch({ type: "setResultStep", step: 0 });
  await tick();
  assert.deepEqual(pending.map((p) => p.req.seq), [1, 2, 3]);
  // The first request (step 0, seq 1) replies after the third was sent: it is not the latest.
  reply(0);
  await tick();
  assert.equal(effects.currentStep(), null, "a reply to an older request is dropped even when it names the step shown");
  assert.equal(store.get().resultStep?.status, "loading");
  reply(2);
  await tick();
  assert.equal(effects.currentStep()?.step, 0);
  effects.dispose();
});

test("a `changed` reply (the file is not the one indexed) re-indexes the results and fetches the step again", async () => {
  const { store, effects, pending, reply, opens } = harness();
  store.dispatch({ type: "setPhase", at: { kind: "results", stage: 2, step: 1 } });
  await tick();
  pending[0]!.resolve({ ok: false, seq: pending[0]!.req.seq, error: "changed on disk", changed: true });
  await tick();
  await tick();
  assert.equal(opens(), 1, "the results were read again");
  assert.equal(store.get().artifacts.results?.status, "ready");
  assert.equal(pending.length, 2, "the step is fetched again under the new index");
  reply(1);
  await tick();
  assert.equal(effects.currentStep()?.step, 1);
  effects.dispose();
});

test("leaving the results phases drops the step data; a failed fetch is a visible status", async () => {
  const { store, effects, pending } = harness();
  store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 0 } });
  await tick();
  pending[0]!.resolve({ ok: false, seq: pending[0]!.req.seq, error: "step 0 is outside stage stage_0" });
  await tick();
  assert.deepEqual(store.get().resultStep, { stage: 0, step: 0, status: "failed", error: "step 0 is outside stage stage_0" });
  assert.match(timelineOf(store.get())!.fetch, /^failed: step 0 is outside/);
  store.dispatch({ type: "setPhase", at: { kind: "mesh" } });
  await tick();
  assert.equal(effects.currentStep(), null);
  assert.equal(store.get().resultStep, null);
  effects.dispose();
});
