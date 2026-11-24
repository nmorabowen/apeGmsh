// The capture harness's reset between steps (renderer/harness/reset.ts `resetView`; the 2026-11-24 architecture
// review, finding 3). A step used to start from what the step before it left: #3079's clip still inherited an element
// field a file refuses, and the combine stills ran against the fake `<model>-other.h5` the workspace step opened. Here
// a store is left as badly as a sequence can leave it, and `resetView` must bring back the view the model opened on.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import type { HarnessContext } from "../src/renderer/harness/context.ts";
import { pickStageDefault, resetView } from "../src/renderer/harness/reset.ts";
import { defaultPick } from "../src/state/contour.ts";
import type { CameraPose, State } from "../src/state/types.ts";
import { opened, STAGED } from "./contour/harness.ts";

const POSE: CameraPose = { eye: [10, -10, 10], target: [0, 0, 0], up: [-0.4082482904638631, 0.4082482904638631, 0.8164965809277261] };

/** What a test compares: the parts of the view the reset owns. */
const view = (s: State) => ({
  model: s.artifacts.model?.path,
  status: s.artifacts.model?.status,
  phase: s.phase.at,
  selection: s.selection.decls,
  stage: s.stage,
  windows: s.windows.open,
  hidden: s.visibility.hidden,
  colourBy: s.visibility.colourBy,
  opacity: s.visibility.opacity,
  extrude: s.visibility.extrudeFrames,
  deform: { on: s.deform.on, scale: s.deform.scale, ghost: s.deform.ghost },
  contour: { source: s.contour.source, field: s.contour.field, component: s.contour.component, range: s.contour.range, gauss: s.contour.gauss, cmap: s.contour.cmap, units: s.contour.targetUnits },
  lineDiagram: s.lineDiagram.on,
  cracks: s.cracks.on,
  hinges: s.hinges.on,
  nodes: s.nodes.on,
  planes: s.clip.planes.length,
  camera: s.camera.pose,
});

test("resetView: a store left by every kind of step comes back to the view the model opened on", async () => {
  const h = await opened();
  try {
    const ctx = { store: h.store, blobs: h.blobs, effects: h.effects, bridge: {}, viewport: {}, baseline: { model: STAGED, camera: POSE, width: 1600, height: 900 } } as unknown as HarnessContext;
    const { store } = h;
    // The first reset from a fresh open is the reference every later reset must match.
    await resetView(ctx);
    const fresh = view(store.get());
    assert.deepEqual(fresh.phase, { kind: "mesh" });
    assert.equal(fresh.colourBy, "group");
    assert.deepEqual(fresh.deform, { on: false, scale: 1, ghost: false });
    assert.deepEqual(fresh.camera, POSE);
    const first = defaultPick(store.get().results!.stages[0]);
    assert.ok(first);
    assert.equal(fresh.contour.field, first.field);
    assert.equal(fresh.contour.component, first.component);

    // What the steps leave: the last stage at its last step, another component over the step's range in another
    // colour map, translucent, deformed with the ghost, a window and a plane, a stage card selected, the line diagram
    // and cracks on, the camera turned away, and (the workspace step's set switch) another model's path open.
    const last = store.get().results!.stages.length - 1;
    store.dispatch({ type: "setPhase", at: { kind: "results", stage: last, step: store.get().results!.stages[last]!.time.length - 1 } });
    store.dispatch({ type: "setColourBy", by: "contour" });
    store.dispatch({ type: "setContourField", field: first.field, component: "displacement_x", source: "nodal" });
    store.dispatch({ type: "setContourRange", range: "step" });
    store.dispatch({ type: "setContourCmap", cmap: "viridis" });
    store.dispatch({ type: "setOpacity", value: 0.25 });
    store.dispatch({ type: "setDeform", on: true });
    store.dispatch({ type: "setDeformScale", scale: 10 });
    store.dispatch({ type: "setReferenceGhost", on: true });
    store.dispatch({ type: "openWindow", window: "materials" });
    store.dispatch({ type: "addClipPlane", normal: [0, 0, 1] });
    store.dispatch({ type: "selectStage", index: 0 });
    store.dispatch({ type: "setLineDiagram", on: true });
    store.dispatch({ type: "setCracks", on: true });
    store.dispatch({ type: "setNodes", on: true });
    store.dispatch({ type: "setCamera", to: { eye: [0, -20, 0], target: [0, 0, 0], up: [0, 0, 1] } });
    store.dispatch({ type: "fileOpened", artifact: "model", path: STAGED.replace(/(\.h5)$/i, "-other$1") });
    assert.notDeepEqual(view(store.get()), fresh);

    await resetView(ctx);
    assert.deepEqual(view(store.get()), fresh);
  } finally {
    h.effects.dispose();
  }
});

test("resetView: refuses to run without a baseline, and when the model does not read again", async () => {
  const h = await opened();
  try {
    const base = { store: h.store, blobs: h.blobs, effects: h.effects, bridge: {}, viewport: {} };
    await assert.rejects(resetView(base as unknown as HarnessContext), /no baseline/);
    const gone = { ...base, baseline: { model: STAGED.replace(/(\.h5)$/i, "-gone$1"), camera: POSE, width: 1, height: 1 } };
    await assert.rejects(resetView(gone as unknown as HarnessContext), /did not read again/);
  } finally {
    h.effects.dispose();
  }
});

test("pickStageDefault: the shown stage's own default pick, nothing on the mesh phase", async () => {
  const h = await opened();
  try {
    const { store } = h;
    store.dispatch({ type: "setPhase", at: { kind: "mesh" } });
    assert.equal(pickStageDefault(store), false);
    const last = store.get().results!.stages.length - 1;
    store.dispatch({ type: "setPhase", at: { kind: "results", stage: last, step: 0 } });
    store.dispatch({ type: "setContourField", field: "displacement", component: "displacement_x", source: "nodal" });
    assert.equal(pickStageDefault(store), true);
    const want = defaultPick(store.get().results!.stages[last])!;
    assert.equal(store.get().contour.field, want.field);
    assert.equal(store.get().contour.component, want.component);
  } finally {
    h.effects.dispose();
  }
});

test("runCapture resets before every capture step it runs", () => {
  const src = readFileSync(fileURLToPath(new URL("../src/renderer/harness/capture.ts", import.meta.url)), "utf8");
  const body = src.slice(src.indexOf("export async function runCapture"));
  // Every call of a capture step inside the sequence goes through `step`, which calls resetView first.
  const calls = [...body.matchAll(/await (?:step\(\(\) => )?(capture[A-Z]\w*)\(/g)];
  assert.ok(calls.length >= 17, `found ${calls.length} capture steps`);
  for (const c of calls) assert.ok(c[0].includes("step(() =>"), `${c[1]} runs without resetView`);
  assert.match(body, /const step = async[^]*?await resetView\(ctx\);/);
});
