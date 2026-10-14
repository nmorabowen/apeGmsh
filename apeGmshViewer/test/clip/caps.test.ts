// V4e3 (#1904): the cut-face caps, ADR 0083 S3's F-* contracts as tests in
// Node (the viewport does not run here; its pieces do: CapLayer, the real
// `renderNow`, `wireInput` and `settleCaps` on minimal receivers, and the
// app's Effects over the contour fixtures):
//
// - F-FILLED: a solid whose field is 0 on every face shows its interior on the cap;
// - F-EXEMPT: no plane cuts a cap (renderNow stamps the model, never the caps);
// - F-LUT: the cap's clim, colormap and no-data colour are the contour's; a
//   colormap change recolours the caps without slicing again;
// - F-SETTLED: 0 slices during a gizmo drag (the caps hidden), 1 on release,
//   1 per step of a scrub;
// - F-TOGGLE: "Contour on cut face" off removes the caps, on brings them back;
// - F-TRIM: each cap lies in the other planes' kept half-spaces;
// - the no-data rule end to end (contour_nodata): a cut point next to a node
//   with no data is no data, never 0;
// - the deformed step: the cap is the slice of the deformed solid (the
//   generator's closed form, deformed here, sliced independently);
// - the stage's activation and the legend's hidden rows take cells off the cap;
// - a shell model gets no cap and says so.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as THREE from "three";
import { CAP_FILL, CAP_STEP, capColours, CapLayer, releaseAware, solidTable } from "../../src/renderer/caps.ts";
import { sliderBox, SectionClip } from "../../src/renderer/clip.ts";
import { toLinear } from "../../src/renderer/contour.ts";
import { GizmoInteractor, PlaneGizmos } from "../../src/renderer/gizmo.ts";
import { Viewport } from "../../src/renderer/viewport.ts";
import type { BlobStore } from "../../src/state/blobs.ts";
import { cutsOf, gizmoPlanes, type Cut } from "../../src/state/clip.ts";
import { NO_DATA_RGB, rampT, sampleCmap } from "../../src/state/contour.ts";
import { initialState, reduce } from "../../src/state/reduce.ts";
import { contourView, cutFaceNote, deformView, NO_SOLID_CELLS, type ContourView } from "../../src/state/selectors.ts";
import { SOLID_SHAPES, sliceSolids, type Solids } from "../../src/state/slice.ts";
import type { Store } from "../../src/state/store.ts";
import type { GaussMode, State } from "../../src/state/types.ts";
import { closedForm, NODATA, NODATA_RESULTS, opened, STAGES, until } from "../contour/harness.ts";

const proto = Viewport.prototype as unknown as {
  renderNow(this: unknown): void;
  wireInput(this: unknown): { gizmo: GizmoInteractor };
  settleCaps(this: unknown, s?: State): void;
  aim: unknown;
  selectAt: unknown;
};

const ELEM_MIXED = fileURLToPath(new URL("../../fixtures/elem_mixed.h5", import.meta.url));

type H = Awaited<ReturnType<typeof opened>>;

/** A plane through the model: `normal` (kept side), offset along it. */
function plane(h: H, normal: [number, number, number], offset: number): string {
  h.store.dispatch({ type: "addClipPlane", normal });
  const id = h.store.get().clip.planes.at(-1)!.id;
  h.store.dispatch({ type: "setClipOffset", id, offset });
  return id;
}

/** The caps as the viewport would draw them now (no drag). */
function capsNow(h: H, caps = new CapLayer()): CapLayer {
  caps.sync(h.store.get(), h.blobs, h.effects.currentStep(), false);
  return caps;
}

/** The step shown is here and its range measured (what the caps wait for). */
async function stepShown(h: H): Promise<void> {
  await until(() => {
    const s = h.store.get();
    const v = contourView(s);
    const at = s.phase.at;
    const st = h.effects.currentStep();
    return (v.state !== "ready" || v.stepMeasured) && at?.kind === "results" && st?.stage === at.stage && st.step === at.step && s.resultStep?.status === "ready";
  }, "the step shown");
}

const area = (pos: ArrayLike<number>): number => {
  let a = 0;
  for (let t = 0; t < pos.length / 9; t++) {
    const e1 = [0, 1, 2].map((k) => pos[9 * t + 3 + k]! - pos[9 * t + k]!), e2 = [0, 1, 2].map((k) => pos[9 * t + 6 + k]! - pos[9 * t + k]!);
    a += 0.5 * Math.hypot(e1[1]! * e2[2]! - e1[2]! * e2[1]!, e1[2]! * e2[0]! - e1[0]! * e2[2]!, e1[0]! * e2[1]! - e1[1]! * e2[0]!);
  }
  return a;
};
const capArea = (c: CapLayer) => c.drawn.reduce((a, x) => a + x.slice.area, 0);

// ---- F-FILLED --------------------------------------------------------------------

test("F-FILLED: a solid whose field is 0 on every face shows its interior on the cap", () => {
  // A 3 x 3 x 3 grid of unit hexes: every node on the boundary holds 0, the 8 interior nodes hold 1.
  const N = 4;
  const at = (i: number, j: number, k: number) => (k * N + j) * N + i;
  const xyz: number[] = [], value: number[] = [];
  for (let k = 0; k < N; k++) for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    xyz.push(i, j, k);
    value.push(i > 0 && i < N - 1 && j > 0 && j < N - 1 && k > 0 && k < N - 1 ? 1 : 0);
  }
  const slot: number[] = [], start = [0];
  for (let k = 0; k < 3; k++) for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) {
    slot.push(at(i, j, k), at(i + 1, j, k), at(i + 1, j + 1, k), at(i, j + 1, k), at(i, j, k + 1), at(i + 1, j, k + 1), at(i + 1, j + 1, k + 1), at(i, j + 1, k + 1));
    start.push(slot.length);
  }
  const solids: Solids = { shape: new Uint8Array(27).fill(SOLID_SHAPES.indexOf("hex")), start: Int32Array.from(start), slot: Int32Array.from(slot) };
  // The drawn faces are the boundary: every vertex of them is 0. Uncut, or cut with no cap, nothing shows the 1.
  const boundary = value.filter((_, n) => xyz[3 * n] === 0 || xyz[3 * n] === 3 || xyz[3 * n + 1] === 0 || xyz[3 * n + 1] === 3 || xyz[3 * n + 2] === 0 || xyz[3 * n + 2] === 3);
  assert.ok(boundary.length === 56 && boundary.every((v) => v === 0), "the skin is 0 everywhere");
  // The cap at z = 1.5 carries the interior.
  const corners = Float64Array.from(slot, (n) => value[n]!);
  const cap = sliceSolids(solids, Float64Array.from(xyz), corners, { normal: [0, 0, -1], constant: 1.5 })!;
  assert.ok(Math.abs(cap.area - 9) < 1e-12, "the whole 3 x 3 section");
  assert.equal(Math.max(...cap.values!), 1, "the interior's 1 is on the cap");
  const view = { state: "ready", clim: { min: 0, max: 1 }, cmap: "jet" } as unknown as ContourView;
  const rgb = capColours(cap, view);
  const zero = sampleCmap("jet", 0).map(toLinear), one = sampleCmap("jet", 1).map(toLinear);
  let ones = 0;
  for (let v = 0; v < cap.values!.length; v++) {
    const c = [rgb[3 * v]!, rgb[3 * v + 1]!, rgb[3 * v + 2]!];
    if (cap.values![v] === 1) {
      ones++;
      assert.ok(c.every((x, k) => Math.abs(x - one[k]!) < 1e-6), "an interior vertex wears the top of the ramp");
    }
  }
  assert.ok(ones > 0 && rgb.some((x, i) => Math.abs(x - zero[i % 3]!) > 1e-3), "the cap is not the skin's uniform colour");
});

// ---- F-EXEMPT ----------------------------------------------------------------------

test("F-EXEMPT: renderNow cuts the model and never a cap, by its own plane or another", async () => {
  const h = await opened();
  plane(h, [0, 0, -1], -2.2);
  plane(h, [1, 0, 0], 0.5);
  const caps = capsNow(h);
  assert.equal(caps.count, 2, "two planes through the column: two caps");
  const clip = new SectionClip();
  clip.sync(h.store.get().clip);
  const scene = new THREE.Scene();
  const model = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshStandardMaterial());
  scene.add(model, caps.root);
  const receiver = {
    clip, scene, camera: new THREE.PerspectiveCamera(),
    renderer: { render: () => {} }, highlight: new THREE.Group(), halo: { pulsing: false, render: () => {} }, lastRenderMs: 0, requestRender: () => {},
  };
  proto.renderNow.call(receiver);
  assert.equal((model.material as THREE.Material).clippingPlanes, clip.planes, "the model is cut");
  for (const c of caps.drawn) assert.ok(!(c.mesh.material as THREE.Material).clippingPlanes?.length, "no plane cuts a cap");
  // Each cap sits CAP_STEP of its extent on the discarded side of its own plane: under the gizmo quad, never on it.
  for (const c of caps.drawn) {
    const p = c.mesh.position;
    const along = c.cut.normal[0] * p.x + c.cut.normal[1] * p.y + c.cut.normal[2] * p.z;
    assert.ok(along < 0, "into the discarded half");
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < c.slice.positions.length; i += 3) for (let k = 0; k < 3; k++) {
      lo[k] = Math.min(lo[k]!, c.slice.positions[i + k]!);
      hi[k] = Math.max(hi[k]!, c.slice.positions[i + k]!);
    }
    assert.ok(Math.abs(-along - CAP_STEP * Math.hypot(hi[0]! - lo[0]!, hi[1]! - lo[1]!, hi[2]! - lo[2]!)) < 1e-6, "by CAP_STEP of its own extent");
  }
});

// ---- F-LUT and the no-data rule ------------------------------------------------------

test("F-LUT: the cap's clim and colormap are the contour's, and its values are the field at the cut (closed form)", async () => {
  const h = await opened();
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 2, step: 1 } });
  h.store.dispatch({ type: "setColourBy", by: "contour" });
  h.store.dispatch({ type: "setContourField", field: "displacement", component: "displacement_x" });
  h.store.dispatch({ type: "setContourRange", range: "step" });
  plane(h, [0, 0, -1], -2.2);
  await stepShown(h);
  const caps = capsNow(h);
  assert.equal(caps.count, 1);
  const cap = caps.drawn[0]!;
  assert.ok(Math.abs(cap.slice.area - 1) < 1e-12, `the column's 1 x 1 section (area ${cap.slice.area})`);
  // The values: displacement_x = s z, linear, so the value at every cut point is the closed form there.
  const T = STAGES[2].T;
  for (let v = 0; v < cap.slice.values!.length; v++) {
    const [x, , z] = [cap.slice.positions[3 * v]!, cap.slice.positions[3 * v + 1]!, cap.slice.positions[3 * v + 2]!];
    const want = closedForm(x, z, 2, 1, T)[0];
    assert.ok(Math.abs(cap.slice.values![v]! - want) <= 1e-9 * Math.max(1, Math.abs(want)), `value ${cap.slice.values![v]} vs ${want}`);
  }
  // The colours: the contour's clim and ramp, through the same sRGB-to-linear step as the surfaces.
  const view = contourView(h.store.get());
  assert.ok(view.state === "ready");
  const colour = (v: number, cmap: "jet" | "viridis") => sampleCmap(cmap, rampT(v, view.clim.min!, view.clim.max!)).map(toLinear);
  const attr = () => cap.mesh.geometry.getAttribute("color").array as Float32Array;
  for (let v = 0; v < cap.slice.values!.length; v++) {
    const want = colour(cap.slice.values![v]!, "jet");
    for (let k = 0; k < 3; k++) assert.ok(Math.abs(attr()[3 * v + k]! - want[k]!) < 1e-6, "the jet colour of the value");
  }
  // A colormap change recolours the same cap (no new slice).
  const slices = caps.slices;
  h.store.dispatch({ type: "setContourCmap", cmap: "viridis" });
  assert.equal(caps.sync(h.store.get(), h.blobs, h.effects.currentStep(), false), true, "recoloured");
  assert.equal(caps.slices, slices, "no slice for a colormap change");
  for (let v = 0; v < cap.slice.values!.length; v++) {
    const want = colour(cap.slice.values![v]!, "viridis");
    for (let k = 0; k < 3; k++) assert.ok(Math.abs(attr()[3 * v + k]! - want[k]!) < 1e-6, "the viridis colour of the value");
  }
  // With no contour, one flat fill that is not the no-data colour.
  h.store.dispatch({ type: "setColourBy", by: "group" });
  capsNow(h, caps);
  const fill = CAP_FILL.map(toLinear);
  const flat = caps.drawn[0]!.mesh.geometry.getAttribute("color").array as Float32Array;
  assert.ok(Array.from(flat).every((x, i) => Math.abs(x - fill[i % 3]!) < 1e-6), "the flat fill");
  assert.ok(Math.hypot(...CAP_FILL.map((c, k) => c - NO_DATA_RGB[k]!)) > 0.15, "the fill is far from the no-data colour");
});

test("no data: a cut point next to a node with no data is no data on the cap, never 0 (contour_nodata)", async () => {
  // The results record only the nodes with z <= 3: the top box's upper nodes have none.
  const h = await opened(NODATA, NODATA_RESULTS);
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 2, step: 0 } });
  h.store.dispatch({ type: "setColourBy", by: "contour" });
  h.store.dispatch({ type: "setContourField", field: "displacement", component: "displacement_x" });
  plane(h, [0, -1, 0], -0.5); // keep y < 0.5: through the column and the top
  await stepShown(h);
  const caps = capsNow(h);
  const cap = caps.drawn[0]!;
  const v = cap.slice.values!;
  let nan = 0;
  for (let i = 0; i < v.length; i++) {
    const z = cap.slice.positions[3 * i + 2]!;
    if (z > 3 + 1e-9) {
      assert.ok(Number.isNaN(v[i]!), `a cut point at z = ${z} lies on an edge to a node with no data`);
      nan++;
    } else if (z < 3 - 1e-9) assert.ok(Number.isFinite(v[i]!), `a cut point at z = ${z} has data`);
    if (z > 1e-9) assert.notEqual(v[i], 0, "never 0 (displacement_x = s z is 0 only at z = 0)");
  }
  assert.ok(nan > 0, "the top's upper half is cut");
  const rgb = cap.mesh.geometry.getAttribute("color").array as Float32Array;
  const grey = NO_DATA_RGB.map(toLinear);
  for (let t = 0; t < v.length / 3; t++) {
    if (![0, 1, 2].some((k) => Number.isNaN(v[3 * t + k]!))) continue;
    for (let k = 0; k < 9; k++) assert.ok(Math.abs(rgb[9 * t + k]! - grey[k % 3]!) < 1e-6, "a triangle with a no-data vertex is the no-data colour whole");
  }
});

test("no data: a contour that cannot colour (off the results phases) leaves the cap no data, as the surfaces are, never 0", async () => {
  const h = await opened();
  h.store.dispatch({ type: "setColourBy", by: "contour" });
  plane(h, [0, 0, -1], -2.2);
  const v = contourView(h.store.get());
  assert.equal(v.state, "unavailable");
  const caps = capsNow(h);
  const cap = caps.drawn[0]!;
  assert.ok(cap.slice.values!.every((x) => Number.isNaN(x)), "every cut point is no data");
  const grey = NO_DATA_RGB.map(toLinear);
  const rgb = cap.mesh.geometry.getAttribute("color").array as Float32Array;
  assert.ok(Array.from(rgb).every((x, i) => Math.abs(x - grey[i % 3]!) < 1e-6), "the no-data colour");
});

// ---- F-SETTLED -------------------------------------------------------------------------

/** The viewport's input wiring and settle path (`wireInput`, `settleCaps`) on a receiver over the harness store. */
function dragRig(h: H) {
  const W = 800;
  const canvas = Object.assign(new EventTarget(), {
    clientHeight: W,
    style: { cursor: "" },
    setPointerCapture: () => {},
    hasPointerCapture: () => false,
    releasePointerCapture: () => {},
    getBoundingClientRect: () => ({ left: 0, top: 0, width: W, height: W }),
  });
  const g = globalThis as unknown as { window?: unknown; document?: unknown };
  g.window ??= new EventTarget();
  g.document = { elementFromPoint: () => canvas };
  const s0 = h.store.get();
  const box = sliderBox(null, s0.mesh!, h.blobs)!;
  const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 1000);
  camera.up.set(0, 0, 1);
  camera.position.set(9, -12, 8);
  camera.lookAt(...s0.mesh!.center);
  camera.updateMatrixWorld();
  camera.updateProjectionMatrix();
  const gizmos = new PlaneGizmos();
  const caps = new CapLayer();
  const receiver: Record<string, unknown> = {
    renderer: { domElement: canvas },
    raycaster: new THREE.Raycaster(),
    camera,
    gizmos,
    store: h.store,
    blobs: h.blobs,
    caps,
    sliding: false,
    stepSource: () => h.effects.currentStep(),
    geometryInfo: null,
    aim: proto.aim,
    selectAt: proto.selectAt,
    settleCaps: proto.settleCaps,
    pickAt: () => null,
    pointAt: () => null,
    requestRender: () => {},
  };
  const { gizmo } = proto.wireInput.call(receiver);
  receiver["gizmoInput"] = gizmo;
  // The viewport's onState: the gizmos follow the planes, then the caps settle.
  const onState = () => {
    gizmos.sync(gizmoPlanes(h.store.get().clip), box);
    proto.settleCaps.call(receiver);
  };
  const off = (h.store as Store).subscribe(onState);
  onState();
  const pixel = (p: readonly number[]): [number, number] => {
    const v = new THREE.Vector3(p[0], p[1], p[2]).project(camera);
    return [((v.x + 1) / 2) * W, ((1 - v.y) / 2) * W];
  };
  const send = (type: string, [x, y]: [number, number]) =>
    canvas.dispatchEvent(Object.assign(new globalThis.Event(type, { cancelable: true }), { button: 0, shiftKey: false, clientX: x, clientY: y, pointerId: 1 }));
  return { caps, gizmos, gizmo, pixel, send, off, receiver };
}

test("F-SETTLED: no slice while a gizmo drags (the caps hidden), one on release; a click changes nothing", async () => {
  const h = await opened();
  const id = plane(h, [0, 0, -1], -2.2);
  const r = dragRig(h);
  try {
    assert.equal(r.caps.slices, 1, "the plane's cap, once");
    assert.equal(r.caps.count, 1);
    const g = r.gizmos.geometries().get(id)!;
    const along = (t: number) => r.pixel([g.centre[0] + t * g.normal[0], g.centre[1] + t * g.normal[1], g.centre[2] + t * g.normal[2]]);
    r.send("pointerdown", along(0));
    assert.equal(r.gizmo.dragging, true, "the press grabbed the quad");
    const offsets: number[] = [];
    for (const t of [0.1, 0.2, 0.3, 0.4, 0.5]) {
      r.send("pointermove", along(t));
      offsets.push(h.store.get().clip.planes[0]!.offset);
      assert.equal(r.caps.count, 0, "mid-drag the caps are hidden");
    }
    assert.equal(new Set(offsets).size, 5, "five moves moved the plane five times");
    assert.equal(r.caps.slices, 1, "0 slices during the drag");
    r.send("pointerup", along(0.5));
    assert.equal(r.gizmo.dragging, false);
    assert.equal(r.caps.slices, 2, "1 slice on release");
    assert.equal(r.caps.count, 1, "the cap is back, at the released plane");
    const z = -h.store.get().clip.planes[0]!.offset;
    assert.ok(r.caps.drawn[0]!.slice.positions.every((x, i) => i % 3 !== 2 || Math.abs(x - z) < 1e-9), "on the plane where it was released");
    // A press and release with no move (a click through the quad): nothing to slice.
    r.send("pointerdown", along(0.5));
    r.send("pointerup", along(0.5));
    assert.equal(r.caps.slices, 2, "a click slices nothing");
    // A held panel slider likewise: no slice per tick, one on release.
    (r.receiver["setSliding"] = (Viewport.prototype as unknown as { setSliding: unknown }).setSliding);
    const slide = r.receiver["setSliding"] as (this: unknown, on: boolean) => void;
    slide.call(r.receiver, true);
    for (const o of [-2.0, -1.8, -1.6]) h.store.dispatch({ type: "setClipOffset", id, offset: o });
    assert.equal(r.caps.slices, 2, "0 slices while the slider is held");
    assert.equal(r.caps.count, 0);
    slide.call(r.receiver, false);
    assert.equal(r.caps.slices, 3, "1 on its release");
  } finally {
    r.off();
  }
});

test("F-SETTLED: a scrub slices once per step, and only once the step is all here", async () => {
  const h = await opened();
  h.store.dispatch({ type: "setColourBy", by: "contour" });
  h.store.dispatch({ type: "setDeform", on: true });
  plane(h, [0, 0, -1], -2.2);
  const caps = new CapLayer();
  const off = h.store.subscribe(() => caps.sync(h.store.get(), h.blobs, h.effects.currentStep(), false));
  try {
    const T = STAGES[2].T;
    const before = caps.slices;
    for (let k = 0; k < T; k++) {
      h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 2, step: k } });
      await stepShown(h);
      await until(() => caps.count === 1 && caps.slices === before + k + 1, `the cap of step ${k}`);
    }
    // Settled: no more slices once every step is shown.
    await new Promise((r) => setTimeout(r, 50));
    assert.equal(caps.slices, before + T, `${T} steps, ${T} slices`);
  } finally {
    off();
  }
});

// ---- F-TOGGLE, F-TRIM ---------------------------------------------------------------------

test("F-TOGGLE: Contour on cut face off removes every cap; on brings them back", async () => {
  const h = await opened();
  assert.equal(h.store.get().clip.cutFace, true, "on by default: a cut through a solid is never an empty box");
  plane(h, [0, 0, -1], -2.2);
  const caps = capsNow(h);
  assert.equal(caps.count, 1);
  h.store.dispatch({ type: "setCutFace", on: false });
  capsNow(h, caps);
  assert.equal(caps.count, 0);
  assert.equal(caps.root.children.length, 0, "no cap mesh kept");
  h.store.dispatch({ type: "setCutFace", on: true });
  capsNow(h, caps);
  assert.equal(caps.count, 1);
  // Apply cuts off: no plane cuts, so no cap either.
  h.store.dispatch({ type: "setApplyCuts", on: false });
  capsNow(h, caps);
  assert.equal(caps.count, 0);
  assert.throws(() => reduce(initialState, { type: "setCutFace", on: "yes" as unknown as boolean }), RangeError);
});

test("F-TRIM: two planes trim each other's caps: each lies in the other's kept half", async () => {
  const h = await opened();
  plane(h, [0, 0, -1], -2.2); // keep z < 2.2
  plane(h, [1, 0, 0], 0.5); // keep x > 0.5
  const caps = capsNow(h);
  assert.equal(caps.count, 2);
  const cuts = cutsOf(h.store.get().clip);
  for (const [i, c] of caps.drawn.entries()) {
    const other = cuts.find((x) => JSON.stringify(x) !== JSON.stringify(c.cut))!;
    const p = c.slice.positions;
    for (let v = 0; v < p.length; v += 3) assert.ok(other.normal[0] * p[v]! + other.normal[1] * p[v + 1]! + other.normal[2] * p[v + 2]! + other.constant >= -1e-9, `cap ${i} vertex in the other's kept half`);
  }
  // z = 2.2 keeps x in [0.5, 1] of the column's 1 x 1 section; x = 0.5 keeps z in [0, 2.2] of its 1 x 3 (no top: z > 3).
  assert.ok(Math.abs(caps.drawn[0]!.slice.area - 0.5) < 1e-12, `z cap ${caps.drawn[0]!.slice.area}`);
  assert.ok(Math.abs(caps.drawn[1]!.slice.area - 2.2) < 1e-12, `x cap ${caps.drawn[1]!.slice.area}`);
});

// ---- deformed, activation, hidden, shells --------------------------------------------------

test("deformed step: the cap is the slice of the deformed solid (the generator's closed form, deformed independently)", async () => {
  const h = await opened();
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 2, step: 3 } });
  h.store.dispatch({ type: "setDeform", on: true });
  h.store.dispatch({ type: "setDeformScale", scale: 40 });
  const cut: Cut = { normal: [1, 0, 0], constant: -0.5 };
  plane(h, [1, 0, 0], 0.5);
  await stepShown(h);
  const s = h.store.get();
  const d = deformView(s);
  assert.ok(d.state === "loading" || d.state === "ready" || d.state === "partial", d.state);
  const caps = capsNow(h);
  assert.equal(caps.count, 1);
  // Independently: every node moved by the closed form at true x 40, sliced by the same plane.
  const t = solidTable(s, h.blobs);
  assert.ok(!("none" in t));
  const T = STAGES[2].T;
  const xyz = new Float64Array(t.xyz.length);
  for (let n = 0; n < t.ids.length; n++) {
    const [x, y, z] = [t.xyz[3 * n]!, t.xyz[3 * n + 1]!, t.xyz[3 * n + 2]!];
    const u = closedForm(x, z, 2, 3, T);
    xyz.set([x + 40 * u[0], y + 40 * u[1], z + 40 * u[2]], 3 * n);
  }
  const want = sliceSolids(t.solids, xyz, null, cut)!;
  const got = caps.drawn[0]!.slice;
  assert.ok(Math.abs(got.area - want.area) < 1e-9 * want.area, `area ${got.area} vs ${want.area}`);
  assert.equal(got.cell.length, want.cell.length);
  assert.ok(got.positions.every((x, i) => Math.abs(x - want.positions[i]!) < 1e-9), "the same triangles");
  // And not the undeformed section: the deformation moved it.
  const base = sliceSolids(t.solids, t.xyz, null, cut)!;
  assert.ok(Math.abs(got.area - base.area) > 1e-3, `deformed ${got.area} vs undeformed ${base.area}`);
  assert.ok(Math.abs(area(caps.drawn[0]!.mesh.geometry.getAttribute("position").array) - got.area) < 1e-4, "the drawn mesh is the slice");
});

test("activation: a stage's inactive cells and the legend's hidden rows are off the cap", async () => {
  const h = await opened();
  plane(h, [1, 0, 0], 0.5); // x = 0.5 through the column (z 0..3) and the top (z 3..3.5)
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 1, step: 0 } }); // erect: everything active
  const caps = capsNow(h);
  assert.ok(Math.abs(capArea(caps) - 3.5) < 1e-12, `column + top ${capArea(caps)}`);
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 0 } }); // base: the top is not activated yet
  capsNow(h, caps);
  assert.ok(Math.abs(capArea(caps) - 3) < 1e-12, `column only ${capArea(caps)}`);
  // The legend hides the column: only the top's 1 x 0.5 is left (on the erect stage).
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 1, step: 0 } });
  const col = h.store.get().mesh!.legend.find((e) => e.name === "Col")!.decl!;
  h.store.dispatch({ type: "setHidden", decls: [col], hidden: true });
  capsNow(h, caps);
  assert.ok(Math.abs(capArea(caps) - 0.5) < 1e-12, `top only ${capArea(caps)}`);
});

test("a shell model gets no cap and says so; never a fake fill", async () => {
  const h = await opened(ELEM_MIXED, fileURLToPath(new URL("../../fixtures/elem_mixed.results.h5", import.meta.url)));
  assert.equal(h.store.get().mesh!.counts.solidCells, 0);
  plane(h, [0, 0, 1], 0);
  const caps = capsNow(h);
  assert.equal(caps.count, 0);
  assert.equal(caps.root.children.length, 0);
  assert.equal(caps.note, NO_SOLID_CELLS);
  assert.equal(cutFaceNote(h.store.get()), NO_SOLID_CELLS, "the panel says it");
  const solid = await opened();
  assert.equal(cutFaceNote(solid.store.get()), null);
});

test("the node table rides with the model read, and the blob GC keeps it", async () => {
  const h = await opened();
  const m = h.store.get().mesh!;
  assert.ok(m.nodes, "MeshInfo.nodes is set");
  const ids = h.blobs.get(m.nodes.ids), xyz = h.blobs.get(m.nodes.xyz);
  assert.ok(ids instanceof Float64Array && xyz instanceof Float64Array && xyz.length === 3 * ids.length);
  // A results read retains blobs (effects `blobRefsOf`): the node table survives it.
  assert.ok((h.blobs as BlobStore).has(m.nodes.ids) && h.blobs.has(m.nodes.xyz));
});

// ---- element fields --------------------------------------------------------------------------

const fx = (name: string) => fileURLToPath(new URL(`../../fixtures/${name}`, import.meta.url));
const EXTRAP = JSON.parse(readFileSync(fx("extrap.oracle.json"), "utf8")) as Record<
  string,
  Record<string, { gauss_node: { node_ids: number[]; values: number[][] }; gauss_node_discrete: { element_ids: number[]; corner_node_ids: number[][]; /** [element][step][corner] */ values: number[][][] } }>
>;

/** Colour by element field `component` in `mode` at step `k`, range `this step`; wait until it is measured. */
async function pickElement(h: H, component: string, mode: GaussMode, k: number): Promise<void> {
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 0 } });
  if (k) h.store.dispatch({ type: "setResultStep", step: k });
  h.store.dispatch({ type: "setColourBy", by: "contour" });
  h.store.dispatch({ type: "setContourRange", range: "step" });
  h.store.dispatch({ type: "setContourField", field: "stress", component, source: "element" });
  h.store.dispatch({ type: "setContourGauss", mode });
  await until(() => {
    const v = contourView(h.store.get());
    const st = h.effects.currentStep();
    return v.state === "element" && v.stepMeasured && st?.step === k && st.element === component;
  }, `${component} ${mode} at step ${k}`);
}

for (const name of ["extrap_solid", "extrap_tet"] as const) {
  test(`element fields (${name}): the node-located modes carry Python's extrapolated values onto the cap; a flat mode colours per cell`, async () => {
    const h = await opened(fx(`${name}.h5`), fx(`${name}.results.h5`));
    h.store.dispatch({ type: "addClipPlane", normal: [0.31, 0.47, 0.83] }); // oblique, through the centre
    const cut = cutsOf(h.store.get().clip)[0]!;
    const k = 1;
    const oracle = EXTRAP[name]!["stress_xx"]!;
    // gauss_node: each corner its node's averaged value (Python's), carried linearly to the cut.
    await pickElement(h, "stress_xx", "node", k);
    let caps = capsNow(h);
    const t = solidTable(h.store.get(), h.blobs);
    assert.ok(!("none" in t));
    const nodal = new Map(oracle.gauss_node.node_ids.map((id, i) => [id, oracle.gauss_node.values[k]![i]!]));
    const want = sliceSolids(t.solids, t.xyz, Float64Array.from(t.cornerNodes, (id) => nodal.get(id) ?? NaN), cut)!;
    const got = caps.drawn[0]!.slice;
    assert.equal(got.values!.length, want.values!.length);
    assert.ok(got.values!.every((v, i) => Math.abs(v - want.values![i]!) <= 1e-9 * Math.max(1, Math.abs(want.values![i]!))), "gauss_node values on the cap");
    // gauss_node_discrete: each corner its own element's corner value (Python's), not the node's average.
    await pickElement(h, "stress_xx", "node_discrete", k);
    caps = capsNow(h);
    const d = oracle.gauss_node_discrete;
    const corner = new Float64Array(t.cornerNodes.length);
    for (let c = 0; c + 1 < t.solids.start.length; c++) {
      const e = d.element_ids.indexOf(t.fem[c]!);
      for (let q = t.solids.start[c]!; q < t.solids.start[c + 1]!; q++) corner[q] = d.values[e]![k]![d.corner_node_ids[e]!.indexOf(t.cornerNodes[q]!)]!;
    }
    const wantD = sliceSolids(t.solids, t.xyz, corner, cut)!;
    const gotD = caps.drawn[0]!.slice;
    assert.ok(gotD.values!.every((v, i) => Math.abs(v - wantD.values![i]!) <= 1e-9 * Math.max(1, Math.abs(wantD.values![i]!))), "gauss_node_discrete values on the cap");
    // A flat mode: every triangle in its cell's one value (the mean of its Gauss points, from the step rows by FEM id).
    await pickElement(h, "stress_xx", "cell", k);
    caps = capsNow(h);
    const v = contourView(h.store.get());
    assert.ok(v.state === "element" && !v.topology.includes("node"), v.state === "element" ? v.topology : v.state);
    const step = h.effects.currentStep()!;
    const mean = new Map<number, number>();
    for (const g of v.groups) {
      const eids = h.blobs.get(g.eids) as Float64Array;
      const rows = step.gauss[g.id]!;
      for (let e = 0; e < g.nElements; e++) {
        let sum = 0;
        for (let q = 0; q < g.nGauss; q++) sum += rows[e * g.nGauss + q]!;
        mean.set(eids[e]!, sum / g.nGauss);
      }
    }
    const flat = caps.drawn[0]!.slice;
    for (let tri = 0; tri < flat.cell.length; tri++) {
      const m = mean.get(t.fem[flat.cell[tri]!]!)!;
      for (let q = 0; q < 3; q++) assert.ok(Math.abs(flat.values![3 * tri + q]! - m) <= 1e-12 * Math.max(1, Math.abs(m)), "the cell's value at every vertex of its triangles");
    }
  });
}

for (const source of ["nodal", "element"] as const) {
  test(`F-LUT (${source}): a fixed range set with no step change recolours the caps to it, without slicing again`, async () => {
    const h = source === "nodal" ? await opened() : await opened(fx("extrap_solid.h5"), fx("extrap_solid.results.h5"));
    if (source === "nodal") {
      h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 2, step: 1 } });
      h.store.dispatch({ type: "setColourBy", by: "contour" });
      h.store.dispatch({ type: "setContourField", field: "displacement", component: "displacement_x" });
      h.store.dispatch({ type: "setContourRange", range: "step" });
      plane(h, [0, 0, -1], -2.2);
      await stepShown(h);
    } else {
      h.store.dispatch({ type: "addClipPlane", normal: [0.31, 0.47, 0.83] });
      await pickElement(h, "stress_xx", "cell", 1);
    }
    const caps = capsNow(h);
    assert.equal(caps.count, 1);
    const slices = caps.slices;
    const cap = caps.drawn[0]!;
    const vals = cap.slice.values!.filter((x) => Number.isFinite(x));
    const lo = Math.min(...vals), hi = Math.max(...vals);
    // A fixed range narrower than the cap's values, so the colours must change.
    const fixed = { min: lo + 0.25 * (hi - lo || 1), max: lo + 0.75 * (hi - lo || 1) };
    h.store.dispatch({ type: "setContourRange", range: "fixed", fixed });
    const v = contourView(h.store.get());
    assert.ok((v.state === "ready" || v.state === "element") && v.clim.min === fixed.min && v.clim.max === fixed.max, "the surfaces' clim is the fixed range");
    capsNow(h, caps);
    assert.equal(caps.slices, slices, "no slice for a range change");
    assert.equal(caps.drawn[0], cap, "the same cap, recoloured");
    const rgb = cap.mesh.geometry.getAttribute("color").array as Float32Array;
    for (let i = 0; i < cap.slice.values!.length; i++) {
      const x = cap.slice.values![i]!;
      if (!Number.isFinite(x)) continue;
      const want = sampleCmap(v.cmap, rampT(x, fixed.min, fixed.max)).map(toLinear);
      for (let k = 0; k < 3; k++) assert.ok(Math.abs(rgb[3 * i + k]! - want[k]!) < 1e-6, `vertex ${i}: the colour of ${x} in the fixed range`);
    }
  });
}

test("releaseAware: the release listeners run first, then `after`; removing one removes its wrapper", () => {
  const t = new EventTarget();
  const log: string[] = [];
  const w = releaseAware(t, () => log.push("after"));
  const up = () => log.push("up");
  w.addEventListener("pointerup", up);
  w.addEventListener("pointermove", () => log.push("move"));
  t.dispatchEvent(new Event("pointermove"));
  t.dispatchEvent(new Event("pointerup"));
  assert.deepEqual(log, ["move", "up", "after"]);
  w.removeEventListener("pointerup", up);
  t.dispatchEvent(new Event("pointerup"));
  assert.deepEqual(log, ["move", "up", "after"], "removed with its wrapper");
});
