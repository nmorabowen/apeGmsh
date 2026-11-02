// ADR 0128 D3, INV-6: the manifest entry against Python's answer. The oracle,
// test/spec/manifest.oracle.json, is written and kept current by
// tests/viewers/test_agent_manifest_oracle.py, which reads the same fixtures
// through `Results`, never through this app. Each case applies a shot through
// the one compiler and the app's own read path (test/contour/harness.ts) and
// compares the entry: values, ids, positions, no-data counts, file sizes and
// hashes. The Electron run in test/spec/render.e2e.ts repeats the comparison
// on the `manifest.json` a real run writes.

import assert from "node:assert/strict";
import { statSync } from "node:fs";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import { compileShot, type ShotPlan } from "../../src/spec/compile.ts";
import { describeSet } from "../../src/spec/describe.ts";
import { manifestEntry, refusedEntry, type ManifestEntry } from "../../src/spec/manifest.ts";
import { validateSpec, type SpecShot } from "../../src/spec/spec.ts";
import { viewVerdict } from "../../src/spec/verdict.ts";
import { takeShots, type RenderDeps } from "../../src/renderer/agentRender.ts";
import { opened, until } from "../contour/harness.ts";
import { caseShot, ORACLE, sameAsOracle } from "./manifestOracle.ts";

const fx = (name: string) => fileURLToPath(new URL(`../../fixtures/${name}`, import.meta.url));

type Opened = Awaited<ReturnType<typeof opened>>;
const openSets = new Map<string, Opened>();
after(() => {
  for (const h of openSets.values()) {
    h.effects.dispose();
    h.host.close();
  }
});
async function setOf(name: string): Promise<Opened> {
  let h = openSets.get(name);
  if (!h) {
    h = await opened(fx(`${name}.h5`), fx(`${name}.results.h5`));
    openSets.set(name, h);
  }
  return h;
}

function shotOf(shot: Record<string, unknown>): SpecShot {
  const v = validateSpec({ spec: "apegmsh-view/1", files: { model: "m.h5" }, shots: [{ name: "s", ...shot }] });
  if (!v.ok) throw new Error(`${v.key}: ${v.reason}`);
  return v.spec.shots[0]!;
}

/** Apply a shot as the render mode does, wait until its view is on hand, and build its entry. */
async function entryOf(h: Opened, shot: SpecShot): Promise<{ entry: ManifestEntry; plan: ShotPlan }> {
  const c = compileShot(shot, describeSet(h.store.get(), []));
  if (!c.ok) throw new Error(`the shot did not compile: ${c.reason}`);
  for (const e of c.plan.events) h.store.dispatch(e);
  // As the render mode's `contourOnScreen`: a contour is on hand only once its step's values are.
  const stepHere = () => {
    const at = h.store.get().phase.at;
    const step = h.effects.currentStep();
    return at?.kind === "results" && step?.stage === at.stage && step.step === at.step;
  };
  await until(() => viewVerdict(h.store.get(), shot, c.plan, { contour: stepHere(), deform: true }).kind === "shown", `shot ${shot.name} to be shown`);
  const entry = manifestEntry(h.store.get(), {
    shot,
    png: `${shot.name}.png`,
    image: [1600, 1000],
    camera: { eye: [1, 1, 1], target: [0, 0, 0], up: [0, 0, 1], fov: 30 },
    layout: null,
    blobs: h.blobs,
    step: h.effects.currentStep(),
  });
  return { entry, plan: c.plan };
}

for (const c of ORACLE.cases) {
  test(`D3: ${c.name} equals Python's Results on ${c.set}`, async () => {
    const h = await setOf(c.set);
    const { entry } = await entryOf(h, shotOf(caseShot(c)));
    sameAsOracle(entry, c);
  });
}

test("the recorded extrema are the values the step range is measured on", async () => {
  const h = await setOf("contour_staged");
  const { entry } = await entryOf(h, shotOf({ stage: "erect", step: 1, contour: { field: "displacement", component: "displacement_z", range: "step" } }));
  const k = entry.contour;
  assert.ok(k?.drawn);
  assert.equal(k.range.mode, "step");
  assert.equal(k.extrema.recorded!.min.value, k.range.min);
  assert.equal(k.extrema.recorded!.max.value, k.range.max);
});

test("D3: the range states its mode and whether the history scan finished", async () => {
  const h = await setOf("contour_staged");
  const { entry } = await entryOf(h, shotOf({ stage: "erect", step: 1, contour: { field: "displacement", component: "displacement_z", range: "history" } }));
  const k = entry.contour;
  assert.ok(k?.drawn);
  assert.equal(k.range.mode, "history");
  assert.deepEqual(k.range.history_scan, { status: "done", done: 2, total: 2 });
  const { entry: fixed } = await entryOf(h, shotOf({ stage: "erect", step: 1, contour: { field: "displacement", component: "displacement_z", range: { min: -1, max: 1 } } }));
  assert.ok(fixed.contour?.drawn);
  assert.deepEqual([fixed.contour.range.mode, fixed.contour.range.min, fixed.contour.range.max], ["fixed", -1, 1]);
  assert.equal(fixed.contour.extrema.recorded!.max.value, k.extrema.recorded!.max.value, "a fixed range leaves the field's extrema alone");
});

test("D3: a deformed shape states its scale and the nodes it draws undeformed, and the banner's text", async () => {
  const h = await setOf("contour_nodata");
  const { entry } = await entryOf(h, shotOf({ stage: "erect", step: 1, deform: { scale: 10 } }));
  assert.ok(entry.deform?.state === "partial");
  assert.equal(entry.deform.scale, 10);
  const undeformed = entry.deform.undeformed!;
  // Python's count: the drawn nodes the file records no displacement for.
  assert.equal(undeformed, ORACLE.cases.find((c) => c.name === "nodal_no_data")!.no_data.count);
  assert.ok(undeformed > 0, "the Top box's own nodes have no displacement");
  assert.ok(entry.banner.some((b) => b.startsWith(`Deformed shape: ${undeformed.toLocaleString()} of`)), `the banner names the count: ${entry.banner.join(" | ")}`);
  assert.equal(entry.contour, null, "no contour asked for, none stated");
});

test("INV-7: the chrome, the layers and the background change no number of the entry", async () => {
  const h = await setOf("contour_staged");
  const shot = { stage: "erect", step: 1, contour: { field: "displacement", component: "displacement_z" } };
  const app = await entryOf(h, shotOf(shot));
  const model = await entryOf(h, shotOf({ ...shot, chrome: "model", background: "transparent", layers: { scale_bar: false } }));
  const strip = (e: ManifestEntry) => ({ ...e, shot: null });
  assert.deepEqual(strip(model.entry), strip(app.entry));
  assert.equal(model.entry.shot.layers["scale_bar"], false, "the entry records the layer turned off");
  assert.equal(model.entry.shot.chrome, "model");
  assert.equal(model.entry.shot.background, "transparent");
});

test("the same shot twice gives the same bytes", async () => {
  const h = await setOf("contour_staged");
  const shot = shotOf({ stage: "erect", step: 1, contour: { field: "displacement", component: "magnitude" } });
  const a = JSON.stringify((await entryOf(h, shot)).entry);
  const other = shotOf({ stage: "base", step: 0, contour: { field: "displacement", component: "displacement_x" } });
  await entryOf(h, other);
  assert.equal(JSON.stringify((await entryOf(h, shot)).entry), a);
});

test("fail closed: an entry that cannot state the view throws, so the render mode refuses the shot and writes no still", async () => {
  const h = await setOf("contour_staged");
  const shot = shotOf({ stage: "erect", step: 1, contour: { field: "displacement", component: "displacement_z" } });
  await entryOf(h, shot);
  const base = { shot, png: "s.png", image: [1600, 1000] as [number, number], camera: { eye: [1, 1, 1] as [number, number, number], target: [0, 0, 0] as [number, number, number], up: [0, 0, 1] as [number, number, number], fov: 30 }, layout: null, blobs: h.blobs };
  const s = h.store.get();
  assert.throws(() => manifestEntry(s, { ...base, step: null }), /not the step the view shows/);
  const step = h.effects.currentStep()!;
  assert.throws(() => manifestEntry(s, { ...base, step: { ...step, step: step.step + 1 } }), /not the step the view shows/);
  assert.throws(() => manifestEntry(s, { ...base, step: { ...step, nodal: {} } }), /the step data has no displacement_z/);
  assert.throws(() => manifestEntry({ ...s, mesh: { ...s.mesh!, nodes: undefined } }, { ...base, step }), /built without its node table/);
});

test("INV-6: a section plane cutting the view: `drawn` is null with its reason, `recorded` stands, and the counts state their limit", async () => {
  const h = await setOf("contour_staged");
  // The review's repro: this plane cuts away z > 1, and node 17 at z = 3.5 holds the uncut drawn maximum.
  const shot = { stage: "erect", step: 1, contour: { field: "displacement", component: "displacement_x" } };
  const whole = (await entryOf(h, shotOf(shot))).entry.contour;
  assert.ok(whole?.drawn);
  assert.equal(whole.extrema.drawn!.max.position[2], 3.5, "uncut, the drawn maximum sits above the plane");
  assert.equal(whole.no_data.limit, null);
  const c = shotOf({ ...shot, clip: [{ normal: [0, 0, -1], offset: -1 }] });
  const compiled = compileShot(c, describeSet(h.store.get(), []));
  assert.ok(compiled.ok);
  await entryOf(h, shotOf(shot));
  h.store.dispatch({ type: "addClipPlane", normal: [0, 0, -1] });
  const planeId = h.store.get().clip.planes.at(-1)!.id;
  h.store.dispatch({ type: "setClipOffset", id: planeId, offset: -1 });
  const inputs = { shot: c, png: "s.png", image: [1600, 1000] as [number, number], camera: { eye: [1, 1, 1] as [number, number, number], target: [0, 0, 0] as [number, number, number], up: [0, 0, 1] as [number, number, number], fov: 30 }, layout: null, blobs: h.blobs, step: h.effects.currentStep() };
  const k = manifestEntry(h.store.get(), inputs).contour;
  assert.ok(k?.drawn);
  assert.equal(k.extrema.drawn, null, "no number from a part the still has cut away stands as drawn");
  assert.match(k.extrema.drawn_reason!, /section planes are cutting/);
  assert.deepEqual(k.extrema.recorded, whole.extrema.recorded, "the recorded field does not depend on the cut");
  assert.match(k.no_data.limit!, /counted before the section planes/);
  assert.equal(k.no_data.count, whole.no_data.count);
  // Cuts switched off, or the plane inactive: the view is whole again.
  h.store.dispatch({ type: "setApplyCuts", on: false });
  const off = manifestEntry(h.store.get(), inputs).contour;
  assert.ok(off?.drawn);
  assert.deepEqual(off.extrema.drawn, whole.extrema.drawn);
  assert.equal(off.no_data.limit, null);
  h.store.dispatch({ type: "setApplyCuts", on: true });
  h.store.dispatch({ type: "removeClipPlane", id: planeId });
});

test("D3: a deformed shape's counts state the same limit under a cut", async () => {
  const h = await setOf("contour_nodata");
  const shot = shotOf({ stage: "erect", step: 1, deform: { scale: 10 } });
  const { entry } = await entryOf(h, shot);
  assert.ok(entry.deform?.state === "partial");
  assert.equal(entry.deform.limit, null);
  h.store.dispatch({ type: "addClipPlane", normal: [0, 0, -1] });
  const id = h.store.get().clip.planes.at(-1)!.id;
  const cut = manifestEntry(h.store.get(), { shot, png: "s.png", image: [1, 1], camera: entry.camera!, layout: null, blobs: h.blobs, step: h.effects.currentStep() });
  h.store.dispatch({ type: "removeClipPlane", id });
  assert.ok(cut.deform?.state === "partial");
  assert.match(cut.deform.limit!, /counted before the section planes/);
});

test("D3: the entry states the still, the camera and the theme it was given", async () => {
  const h = await setOf("contour_staged");
  const { entry } = await entryOf(h, shotOf({ stage: "erect", step: 1 }));
  assert.deepEqual(entry.camera, { eye: [1, 1, 1], target: [0, 0, 0], up: [0, 0, 1], fov: 30 });
  assert.equal(entry.png, "s.png");
  assert.deepEqual(entry.image, [1600, 1000]);
  assert.equal(entry.theme, h.store.get().theme);
});

test("D3: the pairing is the app's own answer, for a set that pairs and for one that does not", async () => {
  const paired = await setOf("contour_staged");
  assert.deepEqual((await entryOf(paired, shotOf({}))).entry.pairing, { state: "paired", reason: null });
  // The same mismatch render.e2e.ts refuses: these results were not run on this model.
  const h = await opened(fx("contour_staged.h5"), fx("elem_mixed.results.h5"));
  openSets.set("mismatch", h);
  const { entry } = await entryOf(h, shotOf({}));
  assert.notEqual(entry.pairing.state, "paired");
  assert.match(entry.pairing.reason!, /model_hash differs/);
  const refused = refusedEntry(h.store.get(), shotOf({}), "no", null);
  assert.deepEqual(refused.pairing, entry.pairing, "a refused shot states the pairing too");
});

test("INV-6: a shot that throws is refused and the run reaches its manifest, so earlier stills keep their entries", async () => {
  const h = await setOf("contour_staged");
  const frame = globalThis.requestAnimationFrame;
  globalThis.requestAnimationFrame = ((cb: FrameRequestCallback) => setTimeout(() => cb(0), 0)) as unknown as typeof requestAnimationFrame;
  const written: string[] = [];
  const canvas = { width: 640, height: 480, getBoundingClientRect: () => ({ x: 0, y: 0, width: 640, height: 480 }) };
  const deps = {
    store: h.store,
    blobs: h.blobs,
    effects: h.effects,
    viewport: { renderer: { domElement: canvas }, camera: {}, renderNow: () => {}, currentPose: () => ({ eye: [1, 1, 1], target: [0, 0, 0], up: [0, 0, 1] }) },
    bridge: { renderCapture: async (name: string) => (written.push(name), { path: name, width: 640, height: 480 }) },
    layout: {
      apply: () => {
        if (written.length === 1 && !boomed) {
          boomed = true;
          throw new Error("a DOM check failed");
        }
      },
      inLayout: () => [],
    },
  } as unknown as RenderDeps;
  let boomed = false;
  try {
    const v = validateSpec({ spec: "apegmsh-view/1", files: { model: "m.h5" }, size: [640, 480], shots: [{ name: "a" }, { name: "boom" }, { name: "c" }] });
    assert.ok(v.ok);
    const shots = await takeShots(deps, v.spec, describeSet(h.store.get(), []), null);
    assert.deepEqual(shots.map((s) => [s.entry.name, s.entry.status, s.entry.png]), [["a", "ok", "a.png"], ["boom", "refused", null], ["c", "ok", "c.png"]]);
    assert.match(shots[1]!.entry.reason!, /the shot failed: a DOM check failed/);
    assert.deepEqual(written, ["a", "c"], "every PNG written has an ok entry, and the refused shot has none");
  } finally {
    globalThis.requestAnimationFrame = frame;
  }
});

test("a refused shot's entry: the reason, no still, and no view it never drew", async () => {
  const h = await setOf("contour_nodata");
  // A view from an earlier shot is on screen, with a banner line; a shot that did not compile must not report it.
  const earlier = await entryOf(h, shotOf({ stage: "erect", step: 1, deform: { scale: 10 } }));
  assert.ok(earlier.entry.banner.length > 0, "the earlier view has a banner line");
  const e = refusedEntry(h.store.get(), shotOf({ stage: "nope" }), "no stage named nope", null);
  assert.equal(e.status, "refused");
  assert.equal(e.reason, "no stage named nope");
  assert.deepEqual([e.png, e.image, e.camera, e.contour, e.deform, e.stage, e.step, e.time, e.layout], [null, null, null, null, null, null, null, null, null]);
  assert.deepEqual(e.banner, []);
  assert.equal(e.pairing.state, "paired");
  assert.equal(e.files.results!.size, statSync(fx("contour_nodata.results.h5")).size);
});
