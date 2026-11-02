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

test("a refused shot's entry: the reason, no still, and no view it never drew", async () => {
  const h = await setOf("contour_staged");
  // A view from an earlier shot is on screen; a shot that did not compile must not report it.
  await entryOf(h, shotOf({ stage: "erect", step: 1, contour: { field: "displacement", component: "displacement_z" } }));
  const e = refusedEntry(h.store.get(), shotOf({ stage: "nope" }), "no stage named nope", null);
  assert.equal(e.status, "refused");
  assert.equal(e.reason, "no stage named nope");
  assert.deepEqual([e.png, e.image, e.camera, e.contour, e.deform, e.stage, e.step, e.time, e.layout], [null, null, null, null, null, null, null, null, null]);
  assert.deepEqual(e.banner, []);
  assert.equal(e.pairing.state, "paired");
  assert.equal(e.files.results!.size, statSync(fx("contour_staged.results.h5")).size);
});
