// V4e2 (#1903): the plane gizmo, ADR 0083 S2's contracts as tests in Node, at
// parity with the Qt tests/viewers/test_clip_gizmo.py. The viewport's own call
// sites run on minimal receivers through `Viewport.prototype` (a WebGL renderer
// cannot run here): `renderNow` (the clip stamping), `wireInput` (the gizmo's
// listener ahead of the navigator's, on one canvas), `selectAt` and `pickAt`.
//
// - G-EXEMPT: the gizmo is not cut while the scene is, by its own plane or another;
// - G-HIT / G-MISS: a press on a gizmo is claimed (the navigator never sees it);
//   a miss is not touched; Shift and a hidden gizmo are left alone;
// - G-DRAG: one `setClipOffset` per move, the offset following the cursor along
//   the normal (the ray through `centre + s n` moves it by exactly `s`);
// - G-ROTATE: one `setClipPose` per move; the plane pivots about the quad
//   centre, not the world origin; a flipped plane keeps the arrow on the cursor;
// - G-EVENTS: pointer-leave ends a drag; the legend over the press wins it; a
//   plane deleted mid-drag ends it; a raise ends it; a degenerate pose is refused;
// - G-PICK: a click through the quad picks the model under it, and no scene
//   raycast ever hits a gizmo.

import assert from "node:assert/strict";
import { test } from "node:test";
import * as THREE from "three";
import { planeRows } from "../../src/panels/planes.ts";
import { SectionClip } from "../../src/renderer/clip.ts";
import { axisParam, gizmoGeometry, GizmoInteractor, PlaneGizmos, rayHit, tipPoint, type GizmoGeometry } from "../../src/renderer/gizmo.ts";
import type { Navigator } from "../../src/renderer/navigation.ts";
import { Viewport } from "../../src/renderer/viewport.ts";
import { BlobStore } from "../../src/state/blobs.ts";
import { gizmoPlanes, type Box } from "../../src/state/clip.ts";
import { initialState, reduce } from "../../src/state/reduce.ts";
import type { ClipPlane, Event, MeshInfo, State } from "../../src/state/types.ts";

const proto = Viewport.prototype as unknown as {
  renderNow(this: unknown): void;
  wireInput(this: unknown): { gizmo: GizmoInteractor; nav: Navigator };
  syncGizmos(this: unknown, s: State, prev: State | null): void;
  aim: unknown;
  selectAt: unknown;
  pickAt: unknown;
  pickOf: unknown;
};

const W = 800;
const BOX: Box = { lo: [0, 0, 0], hi: [2, 2, 2] };

const CLIP_EVENTS = new Set(["addClipPlane", "removeClipPlane", "setClipActive", "setClipFlipped", "setClipOffset", "setClipNormal", "setApplyCuts", "setClipPose", "setShowGizmos", "setClipGizmo"]);

/** A store: the clip events reduce for real, every event is logged. */
function storeOf(s0: State) {
  let s = s0;
  const log: Event[] = [];
  return {
    log,
    get: () => s,
    /** Replace the state outright (a phase change the clip reducer does not own). */
    set: (next: State) => {
      s = next;
    },
    dispatch: (e: Event) => {
      log.push(e);
      if (CLIP_EVENTS.has(e.type)) s = reduce(s, e);
    },
  };
}

/** A state whose mesh has this box (the clip reducer reads its centre), with `planes` added and placed. */
function stateWith(box: Box, ...planes: { normal: number[]; offset?: number; flipped?: boolean }[]): State {
  const center = [0, 1, 2].map((k) => (box.lo[k]! + box.hi[k]!) / 2) as [number, number, number];
  const radius = Math.hypot(box.hi[0] - box.lo[0], box.hi[1] - box.lo[1], box.hi[2] - box.lo[2]) / 2;
  let s: State = { ...initialState, mesh: { center, radius } as unknown as MeshInfo };
  planes.forEach((p, i) => {
    s = reduce(s, { type: "addClipPlane", normal: p.normal });
    const id = `plane-${i + 1}`;
    if (p.offset !== undefined) s = reduce(s, { type: "setClipOffset", id, offset: p.offset });
    if (p.flipped) s = reduce(s, { type: "setClipFlipped", id, flipped: true });
  });
  return s;
}

function cameraAt(pos: readonly number[], at: readonly number[]): THREE.PerspectiveCamera {
  const c = new THREE.PerspectiveCamera(40, 1, 0.01, 1000);
  c.up.set(0, 0, 1);
  c.position.set(pos[0]!, pos[1]!, pos[2]!);
  c.lookAt(at[0]!, at[1]!, at[2]!);
  c.updateMatrixWorld();
  c.updateProjectionMatrix();
  return c;
}

const pixel = (camera: THREE.Camera, p: readonly number[]): [number, number] => {
  const v = new THREE.Vector3(p[0], p[1], p[2]).project(camera);
  return [((v.x + 1) / 2) * W, ((1 - v.y) / 2) * W];
};

/** The canvas: an EventTarget with what the navigator and the gizmo read of a canvas. */
function canvasOf() {
  const t = new EventTarget() as EventTarget & Record<string, unknown>;
  return Object.assign(t, {
    clientHeight: W,
    style: { cursor: "" },
    setPointerCapture: () => {},
    hasPointerCapture: () => false,
    releasePointerCapture: () => {},
    getBoundingClientRect: () => ({ left: 0, top: 0, width: W, height: W }),
  });
}

/** A pointer event Node's EventTarget can deliver (Node has no PointerEvent). */
function pointer(type: string, x: number, y: number, extra: { button?: number; shiftKey?: boolean } = {}): globalThis.Event {
  return Object.assign(new globalThis.Event(type, { cancelable: true }), { button: extra.button ?? 0, shiftKey: extra.shiftKey ?? false, clientX: x, clientY: y, pointerId: 1 });
}

interface Rig {
  store: ReturnType<typeof storeOf>;
  canvas: ReturnType<typeof canvasOf>;
  camera: THREE.PerspectiveCamera;
  gizmos: PlaneGizmos;
  gizmo: GizmoInteractor;
  nav: Navigator;
  picks: [number, number][];
  geom: (id?: string) => GizmoGeometry;
  at: (p: readonly number[]) => [number, number];
  send: (type: string, x: number, y: number, extra?: { button?: number; shiftKey?: boolean }) => globalThis.Event;
  /** Put the state on the geometry phase (or back), through the viewport's own syncGizmos. */
  phase: (kind: "geometry" | "results" | null) => void;
  receiver: Record<string, unknown>;
}

/** The viewport's real input wiring (`wireInput`) on a receiver holding the gizmos synced to the state. */
function rig(s0: State, camPos: readonly number[], box: Box = BOX, chrome: ((x: number, y: number) => boolean) | null = null): Rig {
  const store = storeOf(s0);
  const canvas = canvasOf();
  const camera = cameraAt(camPos, s0.mesh!.center);
  const gizmos = new PlaneGizmos();
  const picks: [number, number][] = [];
  // Node has no window (the navigator's key target) and no document (chromeAt's elementFromPoint): this file's process gets stand-ins.
  const g = globalThis as unknown as { window?: unknown; document?: unknown };
  g.window ??= new EventTarget();
  // The real chromeAt reads document.elementFromPoint: the canvas everywhere, the "legend" where `chrome` says.
  const legend = { id: "legend" };
  g.document = { elementFromPoint: (x: number, y: number) => (chrome?.(x, y) ? legend : canvas) };
  const receiver: Record<string, unknown> = {
    renderer: { domElement: canvas },
    raycaster: new THREE.Raycaster(),
    camera,
    gizmos,
    store,
    geometryInfo: null,
    aim: proto.aim,
    selectAt: proto.selectAt,
    // V4e3: a release re-settles the caps (test/clip/caps.test.ts drives the real one).
    settleCaps: () => {},
    pickAt: (x: number, y: number) => {
      picks.push([x, y]);
      return null;
    },
    pointAt: () => null,
    requestRender: () => {},
    // The box syncGizmos reads (the viewport's is the reference mesh's, renderer/clip.ts sliderBox).
    sliderBox: () => box,
  };
  const { gizmo, nav } = proto.wireInput.call(receiver);
  // Every state change goes through the viewport's own syncGizmos, as its store subscription (onState) does.
  const raw = store.dispatch;
  store.dispatch = (e: Event) => {
    const prev = store.get();
    raw(e);
    proto.syncGizmos.call(receiver, store.get(), prev);
  };
  proto.syncGizmos.call(receiver, store.get(), null);
  return {
    store, canvas, camera, gizmos, gizmo, nav, picks, receiver,
    geom: (id = "plane-1") => gizmos.geometries().get(id)!,
    at: (p) => pixel(camera, p),
    send: (type, x, y, extra) => {
      const e = pointer(type, x, y, extra);
      canvas.dispatchEvent(e);
      return e;
    },
    phase: (kind) => {
      const prev = store.get();
      store.set({ ...prev, phase: { ...prev.phase, at: kind === null ? null : ({ kind } as State["phase"]["at"]) } });
      proto.syncGizmos.call(receiver, store.get(), prev);
    },
  };
}

const add = (a: readonly number[], d: readonly number[], t: number) => [a[0]! + t * d[0]!, a[1]! + t * d[1]!, a[2]! + t * d[2]!];
const near = (a: number, b: number, tol: number, what: string) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b}`);

// ---- G-EXEMPT ------------------------------------------------------------------

test("G-EXEMPT: renderNow cuts the model and never a gizmo, by its own plane or another", () => {
  const s = stateWith(BOX, { normal: [0, 0, 1] }, { normal: [1, 0, 0] });
  const clip = new SectionClip();
  clip.sync(s.clip);
  const gizmos = new PlaneGizmos();
  gizmos.sync(gizmoPlanes(s.clip), BOX);
  assert.equal(gizmos.count, 2);
  const scene = new THREE.Scene();
  const model = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshStandardMaterial());
  scene.add(model, gizmos.root);
  const receiver = {
    clip, scene, camera: new THREE.PerspectiveCamera(),
    renderer: { render: () => {} }, highlight: new THREE.Group(), halo: { pulsing: false, render: () => {} }, lastRenderMs: 0, requestRender: () => {},
  };
  proto.renderNow.call(receiver);
  assert.equal((model.material as THREE.Material).clippingPlanes, clip.planes, "the model is cut");
  assert.equal(clip.planes.length, 2);
  const gizmoMaterials: THREE.Material[] = [];
  gizmos.root.traverse((o) => {
    const m = (o as THREE.Mesh).material as THREE.Material | undefined;
    if (m) gizmoMaterials.push(m);
  });
  assert.ok(gizmoMaterials.length >= 8, "two gizmos: quad, outline, shaft and cone each");
  for (const m of gizmoMaterials) assert.ok(!m.clippingPlanes?.length, "no plane cuts a gizmo material");
  // A plane moved and a third one added: still exempt (the stamp runs every frame).
  let s2 = reduce(s, { type: "setClipOffset", id: "plane-1", offset: 0.5 });
  s2 = reduce(s2, { type: "addClipPlane", normal: [0, 1, 0] });
  clip.sync(s2.clip);
  gizmos.sync(gizmoPlanes(s2.clip), BOX);
  proto.renderNow.call(receiver);
  gizmos.root.traverse((o) => {
    const m = (o as THREE.Mesh).material as THREE.Material | undefined;
    if (m) assert.ok(!m.clippingPlanes?.length, "after a plane change and a new gizmo, still not cut");
  });
  assert.equal((model.material as THREE.Material).clippingPlanes!.length, 3);
});

// ---- geometry -------------------------------------------------------------------

test("geometry: the quad centre sits on the plane and the quad spans the box's cross-section", () => {
  const box: Box = { lo: [10, 20, 30], hi: [14, 22, 31] };
  const s = stateWith(box, { normal: [0, 0, 1], offset: 30.25 });
  const g = gizmoGeometry(s.clip.planes[0]!, box);
  assert.deepEqual(g.centre, [12, 21, 30.25]);
  near(g.halfU * 2, 4 * 1.06, 1e-9, "the quad reaches past the box along x");
  near(g.halfV * 2, 2 * 1.06, 1e-9, "and along y");
  const flipped = gizmoGeometry({ ...s.clip.planes[0]!, flipped: true }, box);
  assert.deepEqual(flipped.direction, [-0, -0, -1], "a flipped plane's arrow points at the kept side");
  assert.deepEqual(flipped.normal, [0, 0, 1], "and drags along the stored normal");
});

// ---- G-HIT / G-MISS --------------------------------------------------------------

test("G-MISS: a press off every gizmo is not touched: the navigator takes it, and the click picks", () => {
  const r = rig(stateWith(BOX, { normal: [0, 0, 1] }), [1, -5, 4]);
  const e = r.send("pointerdown", 5, 5);
  assert.equal(e.defaultPrevented, false, "a miss is not claimed");
  assert.equal(r.gizmo.dragging, false);
  assert.equal(r.nav.dragging, true, "the navigator saw the press");
  r.send("pointerup", 5, 5);
  assert.deepEqual(r.picks, [[5, 5]], "the click picks through the navigator, as before");
  assert.deepEqual(r.store.log.map((e) => e.type), ["clearSelection"], "the click's pick (a miss here) and no plane event");
});

test("G-HIT: a press on the quad starts a translate, on the tip a rotate; the navigator never sees either", () => {
  const r = rig(stateWith(BOX, { normal: [0, 0, 1] }), [1, -5, 4]);
  const g = r.geom();
  const [qx, qy] = r.at(add(g.centre, g.u, 0.6));
  assert.equal(r.gizmos.hit({ origin: r.camera.position.toArray(), direction: new THREE.Vector3(...add(g.centre, g.u, 0.6)).sub(r.camera.position).toArray() })?.mode, "translate");
  const e = r.send("pointerdown", qx, qy);
  assert.equal(e.defaultPrevented, true, "claimed");
  assert.equal(r.gizmo.mode, "translate");
  assert.equal(r.nav.dragging, false, "the navigator never saw the press");
  r.send("pointerleave", qx, qy);
  const [tx, ty] = r.at(tipPoint(g));
  r.send("pointerdown", tx, ty);
  assert.equal(r.gizmo.mode, "rotate", "the tip is the rotate handle");
  assert.equal(r.nav.dragging, false);
});

test("G-HIT: Shift over a gizmo, a closed eye and the footer toggle off each leave the press to the navigator", () => {
  const r = rig(stateWith(BOX, { normal: [0, 0, 1] }), [1, -5, 4]);
  const [x, y] = r.at(add(r.geom().centre, r.geom().u, 0.6));
  r.send("pointerdown", x, y, { shiftKey: true });
  assert.equal(r.gizmo.dragging, false, "Shift: not taken");
  assert.equal(r.nav.dragging, true);
  r.send("pointerup", x, y);
  r.store.dispatch({ type: "setClipGizmo", id: "plane-1", visible: false });
  r.send("pointerdown", x, y);
  assert.equal(r.gizmo.dragging, false, "eye closed: no gizmo, no hit");
  assert.equal(r.gizmos.count, 0);
  r.send("pointerup", x, y);
  r.store.dispatch({ type: "setClipGizmo", id: "plane-1", visible: true });
  r.store.dispatch({ type: "setShowGizmos", on: false });
  r.send("pointerdown", x, y);
  assert.equal(r.gizmo.dragging, false, "footer toggle off: no hit");
  r.send("pointerup", x, y);
  r.store.dispatch({ type: "setShowGizmos", on: true });
  r.send("pointerdown", x, y);
  assert.equal(r.gizmo.dragging, true, "both on again: the press is the gizmo's");
});

// ---- G-DRAG ----------------------------------------------------------------------

test("G-DRAG: one setClipOffset per move, the offset following the cursor along the normal; nothing inside the click slop", () => {
  const r = rig(stateWith(BOX, { normal: [0, 0, 1] }), [1, -5, 4]);
  const g = r.geom();
  const [x0, y0] = r.at(g.centre);
  r.send("pointerdown", x0, y0);
  assert.equal(r.gizmo.mode, "translate", "the press at the quad centre is a translate (asserted before it is tested)");
  r.send("pointermove", x0 + 1, y0 - 1);
  assert.equal(r.store.log.length, 0, "inside the click slop: no event");
  const steps = [0.1, 0.2, 0.3, 0.45, -0.25];
  steps.forEach((t, i) => {
    const [x, y] = r.at(add(g.centre, g.normal, t));
    r.send("pointermove", x, y);
    assert.equal(r.store.log.length, i + 1, "exactly one event per move");
    const e = r.store.log.at(-1)!;
    assert.equal(e.type, "setClipOffset");
    near(r.store.get().clip.planes[0]!.offset, 1 + t, 1e-6, `the offset after a move to centre + ${t} n`);
  });
  r.send("pointerup", ...r.at(add(g.centre, g.normal, -0.25)));
  assert.equal(r.gizmo.dragging, false);
  assert.deepEqual(r.picks, [], "a drag is not a click");
  assert.ok(r.store.log.every((e) => e.type === "setClipOffset"), "a translate makes only offset events");
});

test("G-DRAG: looking nearly along the normal the gain saturates instead of throwing the plane off the model", () => {
  const g = gizmoGeometry(stateWith(BOX, { normal: [0, 0, 1] }).clip.planes[0]!, BOX);
  // A ray 1 degree off the axis, offset by a pixel's worth: the exact projection runs away, the floored one does not.
  const o = [1.001, 1, 10], dir = [Math.sin(Math.PI / 180), 0, -Math.cos(Math.PI / 180)];
  const exact = axisParam(g.centre, g.normal, { origin: o, direction: dir })!;
  const floored = axisParam(g.centre, g.normal, { origin: o, direction: dir }, 0.06)!;
  assert.ok(Math.abs(floored) < Math.abs(exact) / 10, `floored ${floored} vs exact ${exact}`);
});

// ---- G-ROTATE --------------------------------------------------------------------

test("G-ROTATE: one setClipPose per move; the plane pivots about the quad centre, not the world origin", () => {
  const box: Box = { lo: [10, 10, 10], hi: [12, 12, 12] };
  const r = rig(stateWith(box, { normal: [0, 0, 1] }), [11, 5, 16], box);
  const g = r.geom();
  const c0 = g.centre;
  assert.deepEqual(c0, [11, 11, 11]);
  r.send("pointerdown", ...r.at(tipPoint(g)));
  assert.equal(r.gizmo.mode, "rotate", "the press on the tip is a rotate (asserted before it is tested)");
  const tilts = [0.15, 0.3, 0.5];
  tilts.forEach((a, i) => {
    const want = [0, -Math.sin(a), Math.cos(a)];
    r.send("pointermove", ...r.at(add(c0, want, g.arrowLen)));
    assert.equal(r.store.log.length, i + 1, "exactly one event per move");
    assert.equal(r.store.log.at(-1)!.type, "setClipPose", "the normal and the offset in one event");
    const p = r.store.get().clip.planes[0]!;
    for (let k = 0; k < 3; k++) near(p.normal[k]!, want[k]!, 1e-6, `normal[${k}] at tilt ${a}`);
    near(p.normal[0]! * c0[0] + p.normal[1]! * c0[1] + p.normal[2]! * c0[2] - p.offset, 0, 1e-6, "the press's quad centre stays on the plane");
  });
  r.send("pointerup", ...r.at(add(c0, [0, -Math.sin(0.5), Math.cos(0.5)], g.arrowLen)));
  assert.ok(!r.store.log.some((e) => e.type === "setClipNormal" || e.type === "setClipOffset"), "never a separate normal and offset pair");
});

test("G-ROTATE: a flipped plane keeps the arrow on the cursor (the stored normal folds the flip back out)", () => {
  const box: Box = { lo: [10, 10, 10], hi: [12, 12, 12] };
  const r = rig(stateWith(box, { normal: [0, 0, -1], flipped: true }), [11, 5, 16], box);
  const g = r.geom();
  assert.deepEqual(g.direction, [-0, -0, 1], "flipped: the arrow points up, at the kept side");
  r.send("pointerdown", ...r.at(tipPoint(g)));
  assert.equal(r.gizmo.mode, "rotate");
  const want = [0, -Math.sin(0.3), Math.cos(0.3)];
  r.send("pointermove", ...r.at(add(g.centre, want, g.arrowLen)));
  const p = r.store.get().clip.planes[0]!;
  for (let k = 0; k < 3; k++) near(p.normal[k]!, -want[k]!, 1e-6, `stored normal[${k}]`);
  assert.equal(p.flipped, true);
  const after = gizmoGeometry(p, box);
  for (let k = 0; k < 3; k++) near(after.direction[k]!, want[k]!, 1e-6, `the arrow follows the cursor [${k}]`);
});

// ---- G-EVENTS and the seams ----------------------------------------------------------

test("G-EVENTS: pointer-leave ends a drag; the moves after it change nothing", () => {
  const r = rig(stateWith(BOX, { normal: [0, 0, 1] }), [1, -5, 4]);
  const g = r.geom();
  r.send("pointerdown", ...r.at(g.centre));
  r.send("pointermove", ...r.at(add(g.centre, g.normal, 0.2)));
  assert.equal(r.store.log.length, 1);
  r.send("pointerleave", 0, 0);
  assert.equal(r.gizmo.dragging, false, "leave ends the drag");
  r.send("pointermove", ...r.at(add(g.centre, g.normal, 0.5)));
  r.send("pointerup", ...r.at(add(g.centre, g.normal, 0.5)));
  assert.equal(r.store.log.length, 1, "no event after the leave");
  near(r.store.get().clip.planes[0]!.offset, 1.2, 1e-6, "the plane stays where the last move put it");
});

test("G-EVENTS: the legend over a gizmo wins the press (chromeAt, through document.elementFromPoint)", () => {
  const s = stateWith(BOX, { normal: [0, 0, 1] });
  const probe = rig(s, [1, -5, 4]);
  const [x, y] = probe.at(probe.geom().centre);
  const r = rig(s, [1, -5, 4], BOX, (px, py) => Math.abs(px - x) < 40 && Math.abs(py - y) < 40);
  const e = r.send("pointerdown", x, y);
  assert.equal(e.defaultPrevented, false);
  assert.equal(r.gizmo.dragging, false, "the legend's press is not the gizmo's");
  // Outside the legend, the same gizmo still takes a press.
  r.send("pointerup", x, y);
  const [qx, qy] = r.at(add(r.geom().centre, r.geom().u, 0.9));
  assert.ok(Math.abs(qx - x) >= 40 || Math.abs(qy - y) >= 40, "the second press is outside the legend");
  r.send("pointerdown", qx, qy);
  assert.equal(r.gizmo.dragging, true);
});

test("seam: a plane deleted mid-drag ends the drag (no event for a plane that is gone)", () => {
  const r = rig(stateWith(BOX, { normal: [0, 0, 1] }), [1, -5, 4]);
  const g = r.geom();
  r.send("pointerdown", ...r.at(g.centre));
  r.store.dispatch({ type: "removeClipPlane", id: "plane-1" });
  const before = r.store.log.length;
  r.send("pointermove", ...r.at(add(g.centre, g.normal, 0.3)));
  assert.equal(r.gizmo.dragging, false);
  assert.equal(r.store.log.length, before, "nothing dispatched for the deleted plane");
});

test("seam: a raise while dispatching a move ends the drag before it propagates", () => {
  const s = stateWith(BOX, { normal: [0, 0, 1] });
  const g = gizmoGeometry(s.clip.planes[0]!, BOX);
  const it = new GizmoInteractor({
    element: new EventTarget(),
    rayAt: () => ({ origin: [1, 1, 5], direction: [0, 0.6, -1] }),
    hit: () => ({ id: "plane-1", mode: "translate", t: 1 }),
    geometry: () => g,
    plane: () => s.clip.planes[0],
    dispatch: () => {
      throw new Error("boom");
    },
    chromeAt: () => false,
    select: () => {},
  });
  const call = it as unknown as { onDown(e: unknown): void; onMove(e: unknown): void };
  call.onDown(pointer("pointerdown", 100, 100));
  assert.equal(it.dragging, true);
  assert.throws(() => call.onMove(pointer("pointermove", 100, 140)), /boom/);
  assert.equal(it.dragging, false, "no drag outlives the raise");
});

test("seam: a degenerate rotate is refused by the reducer and the pose is kept", () => {
  const s = stateWith(BOX, { normal: [0, 0, 1] });
  const p0 = s.clip.planes[0]!;
  for (const normal of [[0, 0, 0], [NaN, 0, 1], [Infinity, 0, 0]]) {
    const next = reduce(s, { type: "setClipPose", id: "plane-1", normal, offset: 1 });
    assert.deepEqual(next.clip.planes[0], p0, `${JSON.stringify(normal)}: the pose is kept`);
    assert.match(next.clip.notice ?? "", /needs a direction/);
  }
  const nanOffset = reduce(s, { type: "setClipPose", id: "plane-1", normal: [0, 1, 0], offset: NaN });
  assert.deepEqual(nanOffset.clip.planes[0], p0);
  assert.match(nanOffset.clip.notice ?? "", /finite/);
  // A normal of any length: the same plane, scaled to the unit normal.
  const scaled = reduce(s, { type: "setClipPose", id: "plane-1", normal: [0, 2, 0], offset: 3 });
  assert.deepEqual(scaled.clip.planes[0]!.normal, [0, 1, 0]);
  assert.equal(scaled.clip.planes[0]!.offset, 1.5);
  // Through the interactor: a ray that cannot place the trackball point (non-finite) reaches the reducer, which refuses it.
  const store = storeOf(s);
  const g = gizmoGeometry(p0, BOX);
  let ray = { origin: [1, -5, 4], direction: new THREE.Vector3(...tipPoint(g)).sub(new THREE.Vector3(1, -5, 4)).toArray() as number[] };
  const it = new GizmoInteractor({
    element: new EventTarget(),
    rayAt: () => ray,
    hit: (q) => (rayHit(g, q)?.mode === "rotate" ? { id: "plane-1", mode: "rotate", t: 1 } : null),
    geometry: () => g,
    plane: (id) => store.get().clip.planes.find((p) => p.id === id),
    dispatch: (e) => store.dispatch(e),
    chromeAt: () => false,
    select: () => {},
  });
  const call = it as unknown as { onDown(e: unknown): void; onMove(e: unknown): void };
  call.onDown(pointer("pointerdown", 100, 100));
  assert.equal(it.mode, "rotate");
  ray = { origin: [NaN, -5, 4], direction: [0, 1, -0.5] };
  call.onMove(pointer("pointermove", 200, 200));
  assert.equal(store.log.length, 1, "the move made its one event");
  assert.deepEqual(store.get().clip.planes[0], p0, "refused: the pose is kept");
  assert.notEqual(store.get().clip.notice, null);
});

test("panel state: the footer toggle and the eyes decide which planes get a gizmo", () => {
  let s = stateWith(BOX, { normal: [0, 0, 1] }, { normal: [1, 0, 0] });
  assert.equal(s.clip.showGizmos, true, "gizmos show by default (the Qt default)");
  assert.deepEqual(gizmoPlanes(s.clip).map((p: ClipPlane) => p.id), ["plane-1", "plane-2"]);
  s = reduce(s, { type: "setClipGizmo", id: "plane-1", visible: false });
  assert.deepEqual(gizmoPlanes(s.clip).map((p) => p.id), ["plane-2"]);
  assert.deepEqual(planeRows(s.clip.planes, BOX).map((r) => r.gizmo), [false, true], "the rows' eyes");
  s = reduce(s, { type: "setClipActive", id: "plane-2", active: false });
  assert.deepEqual(gizmoPlanes(s.clip).map((p) => p.id), ["plane-2"], "an inactive plane keeps its gizmo (move it, then cut)");
  s = reduce(s, { type: "setShowGizmos", on: false });
  assert.deepEqual(gizmoPlanes(s.clip), []);
  assert.throws(() => reduce(s, { type: "setClipGizmo", id: "plane-9", visible: true }), /no section plane/);
});

test("syncGizmos: a hidden gizmo takes no press and moves no plane (geometry phase, eye closed, footer off), and comes back", () => {
  const r = rig(stateWith(BOX, { normal: [0, 0, 1] }), [1, -5, 4]);
  const g = r.geom();
  const [x, y] = r.at(g.centre);
  const offset0 = r.store.get().clip.planes[0]!.offset;
  const tryDrag = () => {
    r.send("pointerdown", x, y);
    const taken = r.gizmo.dragging;
    r.send("pointermove", ...r.at(add(g.centre, g.normal, 0.3)));
    r.send("pointerup", ...r.at(add(g.centre, g.normal, 0.3)));
    return taken;
  };
  r.phase("geometry");
  assert.equal(r.gizmos.root.visible, false, "the geometry phase hides the gizmos");
  assert.equal(tryDrag(), false, "geometry phase: the hidden gizmo takes no press");
  assert.equal(r.store.get().clip.planes[0]!.offset, offset0, "and no plane moved");
  r.phase(null);
  assert.equal(r.gizmos.root.visible, true);
  r.store.dispatch({ type: "setClipGizmo", id: "plane-1", visible: false });
  assert.equal(r.gizmos.count, 0, "the eye closed: syncGizmos drops the gizmo");
  assert.equal(tryDrag(), false, "eye closed: no press taken");
  r.store.dispatch({ type: "setClipGizmo", id: "plane-1", visible: true });
  r.store.dispatch({ type: "setShowGizmos", on: false });
  assert.equal(r.gizmos.count, 0, "footer off: no gizmo");
  assert.equal(tryDrag(), false, "footer off: no press taken");
  assert.equal(r.store.get().clip.planes[0]!.offset, offset0, "no plane moved through any hidden gizmo");
  r.store.dispatch({ type: "setShowGizmos", on: true });
  r.store.dispatch({ type: "addClipPlane", normal: [1, 0, 0] });
  assert.equal(r.gizmos.count, 2, "a plane added after load gets its gizmo");
  assert.equal(tryDrag(), true, "shown again: the press is the gizmo's");
  near(r.store.get().clip.planes[0]!.offset, offset0 + 0.3, 1e-6, "and the drag moved the plane");
  r.store.dispatch({ type: "setClipOffset", id: "plane-1", offset: 1.5 });
  near(r.geom().centre[2], 1.5, 1e-12, "the gizmo follows an offset typed in the panel");
});

test("G-EVENTS: pointercancel mid-drag ends the drag; the next hover move moves no plane", () => {
  const r = rig(stateWith(BOX, { normal: [0, 0, 1] }), [1, -5, 4]);
  const g = r.geom();
  r.send("pointerdown", ...r.at(g.centre));
  r.send("pointermove", ...r.at(add(g.centre, g.normal, 0.2)));
  assert.equal(r.store.log.length, 1);
  r.send("pointercancel", 0, 0);
  assert.equal(r.gizmo.dragging, false, "cancel ends the drag");
  r.send("pointermove", ...r.at(add(g.centre, g.normal, 0.5)));
  assert.equal(r.store.log.length, 1, "the hover move after the cancel sends nothing");
  near(r.store.get().clip.planes[0]!.offset, 1.2, 1e-6, "the plane stays where the last drag move put it");
});

// ---- G-PICK --------------------------------------------------------------------------

test("G-PICK: a click through the quad picks the model under it; no scene raycast hits a gizmo", () => {
  // A 2 x 2 face at z = 0.5, under a plane at z = 1 flipped to keep z <= 1 (the face is kept, the quad is above it).
  const blobs = new BlobStore();
  const mesh = {
    center: [1, 1, 1], radius: Math.sqrt(3), elements: ["mesh/element/1"],
    lineElement: blobs.put("lineElement", new Int32Array(0)), triElement: blobs.put("triElement", Int32Array.from([0, 0])),
  } as unknown as MeshInfo;
  const face = new Float32Array([0, 0, 0.5, 2, 0, 0.5, 2, 2, 0.5, 0, 0, 0.5, 2, 2, 0.5, 0, 2, 0.5]);
  const faces = new THREE.Mesh(new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(face, 3)), new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }));
  faces.updateMatrixWorld();
  const s = { ...stateWith(BOX, { normal: [0, 0, 1], flipped: true }), mesh, decls: Object.create(null) } as State;
  const r = rig(s, [1.3, 0.4, 8]);
  const clip = new SectionClip();
  clip.sync(s.clip);
  Object.assign(r.receiver, {
    blobs, faces, lines: null, clip,
    drawn: { lineMap: new Int32Array(0), triMap: Int32Array.from([0, 1]), edgeMask: null },
    pickAt: proto.pickAt, pickOf: proto.pickOf,
  });
  const g = r.geom();
  const target = add(g.centre, g.u, 0.5);
  const [x, y] = r.at(target);
  assert.ok(r.gizmos.hit({ origin: r.camera.position.toArray(), direction: new THREE.Vector3(...target).sub(r.camera.position).toArray() }), "the click lands on the quad");
  r.send("pointerdown", x, y);
  assert.equal(r.gizmo.mode, "translate", "the press is the gizmo's");
  r.send("pointerup", x, y);
  const select = r.store.log.find((e) => e.type === "select");
  assert.ok(select && select.type === "select", "the click went through to the pick");
  assert.equal(select.pick.decl, "mesh/element/1");
  near(select.pick.at![2]!, 0.5, 1e-6, "on the face below the quad, not on the quad");
  // A scene-wide raycast (what any future pick over the scene would do): the gizmo is never hit.
  const scene = new THREE.Scene();
  scene.add(r.gizmos.root, faces);
  // What a render does first: the gizmo objects at their placed transforms.
  scene.updateMatrixWorld(true);
  const rc = new THREE.Raycaster();
  rc.set(r.camera.position.clone(), new THREE.Vector3(...target).sub(r.camera.position).normalize());
  assert.deepEqual(rc.intersectObject(r.gizmos.root, true), [], "no gizmo object answers a raycast");
  const hits = rc.intersectObjects(scene.children, true);
  assert.equal(hits[0]?.object, faces, "the first hit through the quad is the model");
});
