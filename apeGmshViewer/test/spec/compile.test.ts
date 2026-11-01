// ADR 0128 slice 2 (VA2): the spec compiler and the view verdict on the
// fixtures, read through the app's own path in Node (test/contour/harness.ts:
// the real Effects over main's readers). Each D2 shot refusal refuses with a
// sentence that names what is missing; what only the drawn view can tell is
// refused with the app's own reason (src/spec/verdict.ts). A refusal does
// not depend on the chrome, the layers or the background (D9, INV-7).
//
// The render mode (renderer/agentRender.ts) takes a still only for a shot
// that compiled and whose verdict is `shown`; the Electron run in
// test/spec/render.e2e.ts checks that a refused shot leaves no PNG.

import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { after, test } from "node:test";
import { BASE_EVENTS, compileShot, type ShotPlan } from "../../src/spec/compile.ts";
import { describeSet, SPEC_MAGNITUDE, type SetDescription } from "../../src/spec/describe.ts";
import { validateSpec, type SpecShot } from "../../src/spec/spec.ts";
import { viewVerdict } from "../../src/spec/verdict.ts";
import { MAGNITUDE } from "../../src/state/contour.ts";
import { contourView } from "../../src/state/selectors.ts";
import type { State } from "../../src/state/types.ts";
import { opened, STAGED, STAGED_RESULTS, until } from "../contour/harness.ts";
import { makeReactionSet, REACTION_COMPONENTS } from "./reactions.ts";

const FIX = fileURLToPath(new URL("../../fixtures/", import.meta.url));
const tmp = mkdtempSync(join(tmpdir(), "agv-compile-"));
after(() => rmSync(tmp, { recursive: true, force: true }));

type Opened = Awaited<ReturnType<typeof opened>>;
const sets: Opened[] = [];
after(() => {
  for (const h of sets) {
    h.effects.dispose();
    h.host.close();
  }
});
async function open(model: string, results: string): Promise<Opened> {
  const h = await opened(model, results);
  sets.push(h);
  return h;
}

/** One validated shot, with the spec's other keys at their defaults. */
function shotOf(shot: Record<string, unknown>, top: Record<string, unknown> = {}): SpecShot {
  const v = validateSpec({ spec: "apegmsh-view/1", files: { model: "m.h5" }, shots: [{ name: "s", ...shot }], ...top });
  if (!v.ok) throw new Error(`${v.key}: ${v.reason}`);
  return v.spec.shots[0]!;
}

const describe = (s: State): SetDescription => describeSet(s, []);

function plan(shot: SpecShot, d: SetDescription): ShotPlan {
  const c = compileShot(shot, d);
  if (!c.ok) throw new Error(`expected the shot to compile: ${c.reason}`);
  return c.plan;
}

function refusal(shot: Record<string, unknown>, d: SetDescription, reason: RegExp): void {
  // D9: the same refusal, word for word, in both chromes and with any layers or background.
  const reasons = [
    compileShot(shotOf(shot), d),
    compileShot(shotOf({ ...shot, chrome: "model" }), d),
    compileShot(shotOf({ ...shot, chrome: "model", background: "transparent", layers: { axes: false } }), d),
  ].map((c) => (c.ok ? null : c.reason));
  assert.ok(reasons[0] !== null, `expected a refusal matching ${reason}`);
  assert.match(reasons[0]!, reason);
  assert.deepEqual(reasons, [reasons[0], reasons[0], reasons[0]], "a refusal does not depend on the chrome, the layers or the background");
}

// ---- describe --------------------------------------------------------------------

test("describe: the set's stages, steps, times, fields and groups, as the spec names them", async () => {
  const h = await open(STAGED, STAGED_RESULTS);
  const d = describe(h.store.get());
  assert.equal(d.pairing.state, "paired");
  assert.deepEqual(
    d.stages.map((s) => [s.index, s.name, s.steps]),
    [[0, "base", 3], [1, "erect", 2], [2, "unprop", 4]],
  );
  assert.deepEqual(d.stages[2]!.times, [0.25, 0.5, 0.75, 1]);
  const disp = d.stages[0]!.fields.find((f) => f.name === "displacement")!;
  assert.deepEqual(disp.components, [
    { key: SPEC_MAGNITUDE, label: "magnitude" },
    { key: "displacement_x", label: "x" },
    { key: "displacement_y", label: "y" },
    { key: "displacement_z", label: "z" },
  ]);
  assert.deepEqual(d.groups.map((g) => g.name), ["Col", "Prop", "Top"]);
  assert.deepEqual(d.offers.layers, { app: ["header", "phase", "viewcube", "left_rail", "rail", "windows", "banner"], model: ["scale_bar", "axes", "banner", "caption"] });
});

// ---- the four ADR 0094 views compile and reach their state --------------------------

test("the four ADR 0094 views (mesh, contour, deformed, reactions) compile into the state they name", async () => {
  const set = await makeReactionSet(join(tmp, "reactions"));
  const h = await open(set.model, set.results);
  const d = describe(h.store.get());
  const reactions = d.stages[2]!.fields.find((f) => f.name === "reaction_force");
  assert.ok(reactions, "the reaction set offers reaction_force");
  assert.deepEqual(reactions.components.map((c) => c.key), [SPEC_MAGNITUDE, ...REACTION_COMPONENTS]);

  const apply = (p: ShotPlan) => {
    for (const e of p.events) h.store.dispatch(e);
    return h.store.get();
  };
  // mesh: the mesh phase, coloured by group.
  let s = apply(plan(shotOf({ camera: "iso" }), d));
  assert.deepEqual(s.phase.at, { kind: "mesh" });
  assert.equal(s.visibility.colourBy, "group");
  assert.equal(s.deform.on, false);
  // contour: displacement z, at the last step of "unprop", over the whole history.
  const contour = plan(shotOf({ stage: "unprop", contour: { field: "displacement", component: "z" } }), d);
  assert.deepEqual(contour.stage, { index: 2, id: d.stages[2]!.id, name: "unprop" });
  assert.equal(contour.step, 3, "the default step is the last");
  s = apply(contour);
  assert.deepEqual(s.phase.at, { kind: "results", stage: 2, step: 3 });
  assert.deepEqual([s.visibility.colourBy, s.contour.field, s.contour.component, s.contour.source, s.contour.range, s.contour.cmap], ["contour", "displacement", "displacement_z", "nodal", "history", "jet"]);
  // deformed: the deformed shape at scale 10, coloured by group (no reference ghost: spec v1 refuses it).
  s = apply(plan(shotOf({ stage: 1, step: "first", deform: { scale: 10 } }), d));
  assert.deepEqual(s.phase.at, { kind: "results", stage: 1, step: 0 });
  assert.deepEqual([s.visibility.colourBy, s.deform.on, s.deform.scale, s.deform.ghost], ["group", true, 10, false]);
  // reactions: the reaction magnitude, a fixed range, another colormap; the deformed shape is off again (each shot starts from the base view).
  s = apply(plan(shotOf({ stage: "unprop", step: 2, contour: { field: "reaction_force", component: "magnitude", range: { min: 0, max: 50 }, cmap: "viridis" } }), d));
  assert.deepEqual([s.contour.field, s.contour.component, s.contour.range, s.contour.fixed, s.contour.cmap, s.deform.on], ["reaction_force", MAGNITUDE, "fixed", { min: 0, max: 50 }, "viridis", false]);
  // A component may be named by its key or by the legend's label.
  assert.equal(plan(shotOf({ stage: "base", contour: { field: "reaction_force", component: "reaction_force_y" } }), d).contour!.component, "reaction_force_y");
  assert.equal(plan(shotOf({ stage: "base", contour: { field: "reaction_force", component: "y" } }), d).contour!.component, "reaction_force_y");
  // isolate a group by its path.
  s = apply(plan(shotOf({ isolate: [d.groups[0]!.path] }), d));
  assert.deepEqual(s.visibility.hidden, d.groups.slice(1).map((g) => g.path));
});

test("reactions: the set records them at the held nodes only (NaN elsewhere); the contour's history range is the closed form over those nodes", async () => {
  const set = await makeReactionSet(join(tmp, "reactions-range"));
  assert.equal(set.held.size, 144, "the Col group's nodes, held in x, y and z");
  const h = await open(set.model, set.results);
  const shot = shotOf({ stage: "unprop", contour: { field: "reaction_force", component: "magnitude" } });
  const p = plan(shot, describe(h.store.get()));
  for (const e of p.events) h.store.dispatch(e);
  await until(() => h.store.get().contour.scan?.status === "done", "the history scan");
  const scan = h.store.get().contour.scan!;
  // Stage 2 (unprop), steps k = 0..3: |R| = 3 (k + 1) sqrt(1 + 4 + 9) (1 + n mod 4), over held nodes n only.
  const factors = [...set.held].map((n) => 1 + (n % 4));
  const min = 3 * 1 * Math.sqrt(14) * Math.min(...factors);
  const max = 3 * 4 * Math.sqrt(14) * Math.max(...factors);
  assert.ok(Math.abs(scan.min! - min) < 1e-9 * max && Math.abs(scan.max! - max) < 1e-9 * max, `range ${scan.min}..${scan.max}, expected ${min}..${max}`);
  assert.ok(scan.min! > 0, "no node outside the supports contributes a zero reaction");
  // The step the app reads: a value at each held node, and none (NaN, drawn as no data) at every other node.
  const step = await h.host.readStep({ path: set.results, stage: d0(h).stages[2]!.id, step: 3, seq: 1 });
  assert.ok(step.ok, step.ok ? "" : step.error);
  if (!step.ok) return;
  const ids = Array.from(h.blobs.get(h.store.get().results!.stages[2]!.nodeIds!) as Float64Array);
  const rz = step.nodal!["reaction_force_z"]!;
  const finite = ids.filter((_, j) => Number.isFinite(rz[j]!));
  assert.deepEqual(finite.sort((x, y) => x - y), [...set.held].sort((x, y) => x - y), "reactions exactly at the held nodes");
});
const d0 = (h: Opened) => describe(h.store.get());

test("INV-2/INV-7: one compile per shot; chrome, layers and background change no event and no camera", async () => {
  const h = await open(STAGED, STAGED_RESULTS);
  const d = describe(h.store.get());
  const shot = { stage: "erect", contour: { field: "displacement", component: "magnitude" }, deform: { scale: 5 }, camera: "+y" };
  const a = plan(shotOf(shot), d);
  const b = plan(shotOf({ ...shot, chrome: "model", background: "transparent", caption: "A caption.", layers: { caption: true, axes: false } }), d);
  assert.deepEqual(a, b);
  assert.deepEqual(a.events.slice(0, BASE_EVENTS.length), [...BASE_EVENTS], "every shot starts from the base view");
});

// ---- D2: each shot refusal names what is missing (no fallback) ----------------------

test("D2: a stage the results do not have, by name or index, is refused; never the nearest stage", async () => {
  const d = describe((await open(STAGED, STAGED_RESULTS)).store.get());
  refusal({ stage: "gravity" }, d, /the results have no stage named "gravity"; their stages are "base", "erect", "unprop"/);
  refusal({ stage: 3 }, d, /the results have 3 stages \(indices 0 to 2\); stage 3 is not one of them/);
});

test("D2: a step out of range is refused, naming the step; never the step before", async () => {
  const d = describe((await open(STAGED, STAGED_RESULTS)).store.get());
  refusal({ stage: "erect", step: 2 }, d, /stage "erect" has 2 steps \(0 to 1\); step 2 is not one of them/);
});

test("D2: a field or a component the stage does not have is refused, naming it and listing what there is", async () => {
  const d = describe((await open(STAGED, STAGED_RESULTS)).store.get());
  refusal({ stage: "base", contour: { field: "velocity", component: "x" } }, d, /stage "base" has no field "velocity"; its fields are displacement \(nodal\)/);
  refusal({ stage: "base", contour: { field: "displacement", component: "rz" } }, d, /nodal field "displacement" of stage "base" has no component "rz"; its components are magnitude, displacement_x \(x\)/);
  refusal({ stage: "base", contour: { field: "displacement", component: "x", source: "element" } }, d, /has no element field "displacement"/);
});

test("D2: a physical group the model does not have is refused", async () => {
  const d = describe((await open(STAGED, STAGED_RESULTS)).store.get());
  refusal({ isolate: ["mesh/physical_group/Beams"] }, d, /the model has no physical group mesh\/physical_group\/Beams; its groups are mesh\/physical_group\/Col/);
});

test("D2: a stage of results that do not pair with the model is refused with the pairing's reason", async () => {
  const d = describe((await open(STAGED, join(FIX, "elem_mixed.results.h5"))).store.get());
  assert.equal(d.pairing.state, "mismatch");
  assert.deepEqual(d.stages, [], "an unpaired file's stages are not offered");
  refusal({ stage: 0 }, d, /the results are not drawn on this model: .*model_hash differs/);
});

test("D2: a stage named in a set with no results file is refused", async () => {
  const h = await open(STAGED, STAGED_RESULTS);
  const d = { ...describe(h.store.get()), files: { model: STAGED, geometry: null, results: null } };
  refusal({ stage: "base" }, d, /the set has no results file, so it has no stage "base"/);
});

// ---- D2: what only the drawn view can tell, with the app's own reason ----------------

const ON = { contour: true, deform: true };

test("D2: a contour with no results step is refused with the app's reason", async () => {
  const h = await open(STAGED, STAGED_RESULTS);
  const shot = shotOf({ contour: { field: "displacement", component: "z" } });
  const p = plan(shot, describe(h.store.get()));
  for (const e of p.events) h.store.dispatch(e);
  const v = viewVerdict(h.store.get(), shot, p, ON);
  assert.deepEqual(v, { kind: "waiting", why: "the contour is not drawn: contours colour a results step: choose results in the phase bar", loading: false });
});

test("D2: a deformed shape the banner refuses is refused with the banner's sentence", async () => {
  const h = await open(join(FIX, "staged_activation.h5"), join(FIX, "staged_activation.results.h5"));
  const shot = shotOf({ stage: "base", deform: { scale: 10 } });
  const p = plan(shot, describe(h.store.get()));
  for (const e of p.events) h.store.dispatch(e);
  await until(() => h.store.get().resultStep?.status !== "loading", "the step");
  const v = viewVerdict(h.store.get(), shot, p, ON);
  assert.equal(v.kind, "waiting");
  if (v.kind !== "waiting") return;
  assert.equal(v.why, "Deformed shape not drawn: stage base records no displacement_y: the deformed shape needs every translational axis.");
  assert.ok(h.store.get().phase.at?.kind === "results", "the view asked for is set: nothing falls back to the mesh");
});

for (const [name, model, reason] of [
  ["rows that do not join the mesh", "elem_badid", /element field stress: _element_index names 1 element id the model has no cell for \(999999\)/],
  ["an unverified file", "elem_mixed_unstamped", /does not prove which elements its rows are/],
] as const) {
  test(`D2: an element contour of ${name} is refused with the app's reason`, async () => {
    const h = await open(join(FIX, `${model}.h5`), join(FIX, `${model}.results.h5`));
    const shot = shotOf({ stage: 0, contour: { field: "stress", component: "stress_xx" } });
    const p = plan(shot, describe(h.store.get()));
    assert.equal(p.contour!.source, "element", "a field only element results carry is an element field");
    for (const e of p.events) h.store.dispatch(e);
    await until(() => contourView(h.store.get()).state === "refused", "the element refusal");
    const v = viewVerdict(h.store.get(), shot, p, ON);
    assert.equal(v.kind, "waiting");
    if (v.kind === "waiting") assert.match(v.why ?? "", reason);
  });
}

test("D2: a contour whose whole-history range could not be read is refused; never drawn at the step's range", async () => {
  const h = await open(STAGED, STAGED_RESULTS);
  const shot = shotOf({ stage: "base", contour: { field: "displacement", component: "z" } });
  const p = plan(shot, describe(h.store.get()));
  for (const e of p.events) h.store.dispatch(e);
  await until(() => h.store.get().contour.scan?.status === "done", "the history scan");
  const v0 = viewVerdict(h.store.get(), shot, p, ON);
  assert.deepEqual(v0, { kind: "shown" }, "the view is shown once the step, the scan and the colours are in");
  assert.deepEqual(viewVerdict(h.store.get(), shot, p, { contour: false, deform: true }).kind, "waiting", "colours not yet applied: not shown");
  const scan = h.store.get().contour.scan!;
  h.store.dispatch({ type: "contourScan", scan: { ...scan, status: "failed", min: null, max: null, error: "the file changed during the scan" } });
  assert.deepEqual(viewVerdict(h.store.get(), shot, p, ON), { kind: "refused", reason: "the contour's whole-history range could not be read: the file changed during the scan" });
  // A step range asks for no scan: the same failure does not refuse it.
  const stepShot = shotOf({ stage: "base", contour: { field: "displacement", component: "z", range: "step" } });
  assert.equal(viewVerdict(h.store.get(), stepShot, plan(stepShot, describe(h.store.get())), ON).kind, "shown");
});

test("D2: a step that could not be read is refused", async () => {
  const h = await open(STAGED, STAGED_RESULTS);
  const shot = shotOf({ stage: "base", step: 1 });
  const p = plan(shot, describe(h.store.get()));
  for (const e of p.events) h.store.dispatch(e);
  h.store.dispatch({ type: "resultStep", status: { stage: 0, step: 1, status: "failed", error: "truncated dataset" } });
  assert.deepEqual(viewVerdict(h.store.get(), shot, p, ON), { kind: "refused", reason: 'step 1 of stage "base" could not be read: truncated dataset' });
});
