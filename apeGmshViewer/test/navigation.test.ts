// Navigation: the bindings table against the maintainer's CAD convention
// (#1285 addendum), the pivot fallback, and the turntable invariants, each
// checked against a closed form (the pivot keeps its pixel, Z stays vertical
// on screen, a grabbed point follows the cursor, a zoom point keeps its pixel).

import assert from "node:assert/strict";
import { test } from "node:test";
import * as THREE from "three";
import { bindingFor, buttonOf, keyAction, POINTER_BINDINGS, WHEEL_ACTION, type Button } from "../src/renderer/bindings.ts";
import {
  clampZoom, headingOf, Navigator, nearestHit, orbitAbout, orientationOf, panBy, pivotFor, zoomToward,
  ZOOM_MAX, ZOOM_MIN, type Heading, type ListenTarget, type NavHost,
} from "../src/renderer/navigation.ts";

const W = 800, H = 600;

function camera(at: THREE.Vector3, h: Heading): THREE.PerspectiveCamera {
  const c = new THREE.PerspectiveCamera(35, W / H, 0.01, 1000);
  c.up.set(0, 0, 1);
  c.position.copy(at);
  c.quaternion.copy(orientationOf(h));
  c.updateMatrixWorld();
  return c;
}

/** Pixel (x right, y down) of a world point. */
function pixel(c: THREE.PerspectiveCamera, p: THREE.Vector3): [number, number] {
  c.updateMatrixWorld();
  const n = p.clone().project(c);
  return [((n.x + 1) / 2) * W, ((1 - n.y) / 2) * H];
}

const close = (a: number, b: number, tol: number, what: string) =>
  assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b}`);

// ---- bindings ---------------------------------------------------------------

test("the bindings table has every (button, shift) pair exactly once", () => {
  for (const button of ["left", "middle", "right"] as Button[]) {
    for (const shift of [false, true]) {
      assert.equal(POINTER_BINDINGS.filter((b) => b.button === button && b.shift === shift).length, 1, `${button} ${shift}`);
    }
  }
  assert.equal(POINTER_BINDINGS.length, 6);
});

test("CAD convention: left selects and never orbits, right orbits, middle and shift+right pan", () => {
  for (const shift of [false, true]) {
    assert.equal(bindingFor("left", shift).click, "select");
    assert.equal(bindingFor("left", shift).drag, null, "a left-drag never orbits or pans");
    assert.equal(bindingFor("middle", shift).drag, "pan");
    assert.equal(bindingFor("right", shift).click, null, "right-click does not select");
    assert.equal(bindingFor("middle", shift).click, null);
  }
  assert.equal(bindingFor("right", false).drag, "orbit");
  assert.equal(bindingFor("right", true).drag, "pan");
  assert.equal(POINTER_BINDINGS.filter((b) => b.drag === "orbit").length, 1, "one way to orbit");
  assert.equal(WHEEL_ACTION, "zoom-to-cursor");
});

test("DOM buttons map to the table; back/forward are unbound; F fits", () => {
  assert.equal(buttonOf(0), "left");
  assert.equal(buttonOf(1), "middle");
  assert.equal(buttonOf(2), "right");
  assert.equal(buttonOf(3), null);
  assert.equal(buttonOf(4), null);
  assert.equal(keyAction("f"), "fit");
  assert.equal(keyAction("F"), "fit");
  assert.equal(keyAction("g"), null);
});

// ---- pivot ------------------------------------------------------------------

function plateAndRay(origin: THREE.Vector3, dir: THREE.Vector3) {
  // A 2 x 2 plate in z = 0 centred at the origin.
  const plate = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }));
  plate.updateMatrixWorld();
  return { plate, ray: new THREE.Raycaster(origin, dir.clone().normalize()) };
}

test("pivot: the mesh point under the cursor when the ray hits", () => {
  // Ray from (0.3, -0.2, 5) straight down: hits the plate at (0.3, -0.2, 0).
  const { plate, ray } = plateAndRay(new THREE.Vector3(0.3, -0.2, 5), new THREE.Vector3(0, 0, -1));
  const p = pivotFor(nearestHit(ray, [plate]), new THREE.Vector3(9, 9, 9));
  close(p.x, 0.3, 1e-9, "x");
  close(p.y, -0.2, 1e-9, "y");
  close(p.z, 0, 1e-9, "z");
});

test("pivot: falls back to the bounding-box centre on a miss", () => {
  // Ray from (5, 5, 5) straight down passes beside the plate.
  const { plate, ray } = plateAndRay(new THREE.Vector3(5, 5, 5), new THREE.Vector3(0, 0, -1));
  assert.equal(nearestHit(ray, [plate]), null);
  const centre = new THREE.Vector3(1, 2, 3);
  const p = pivotFor(nearestHit(ray, [plate]), centre);
  assert.deepEqual(p.toArray(), [1, 2, 3]);
  assert.notEqual(p, centre, "a copy, so the drag cannot move the model's centre");
  assert.deepEqual(pivotFor(nearestHit(ray, []), centre).toArray(), [1, 2, 3], "no model drawn");
});

// ---- turntable --------------------------------------------------------------

test("heading round-trips a view direction; heading {0,0} is plan with +Y up", () => {
  const f = new THREE.Vector3(-1.0, 1.3, -0.9).normalize();
  const h = headingOf(f);
  const back = new THREE.Vector3(0, 0, -1).applyQuaternion(orientationOf(h));
  for (const k of [0, 1, 2]) close(back.getComponent(k), f.getComponent(k), 1e-12, `forward[${k}]`);
  const plan = orientationOf({ yaw: 0, tilt: 0 });
  assert.deepEqual(new THREE.Vector3(0, 0, -1).applyQuaternion(plan).toArray().map((v) => Math.round(v * 1e12) / 1e12), [0, 0, -1]);
  assert.deepEqual(new THREE.Vector3(0, 1, 0).applyQuaternion(plan).toArray().map((v) => Math.round(v * 1e12) / 1e12), [0, 1, 0]);
});

test("orbit: the pivot keeps its pixel and world Z stays vertical on screen", () => {
  const h0 = headingOf(new THREE.Vector3(-1.0, 1.3, -0.9));
  const c = camera(new THREE.Vector3(10, -13, 9), h0);
  const pivot = new THREE.Vector3(0.7, 0.4, -0.5); // off the screen centre
  const before = pixel(c, pivot);
  let h = h0;
  for (const [dy, dt] of [[0.8, 0.3], [-2.1, -0.6], [3.0, 0.2]] as const) {
    h = orbitAbout(c, h, pivot, dy, dt);
    const after = pixel(c, pivot);
    close(after[0], before[0], 1e-6, "pivot x px");
    close(after[1], before[1], 1e-6, "pivot y px");
    // No roll: the camera's right axis is horizontal and its up has +Z.
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(c.quaternion);
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(c.quaternion);
    close(right.z, 0, 1e-12, "right.z");
    assert.ok(up.z >= -1e-12, `up.z ${up.z}`);
    // +Z points up the screen at the pivot.
    const a = pixel(c, pivot), b = pixel(c, pivot.clone().add(new THREE.Vector3(0, 0, 0.1)));
    assert.ok(b[1] < a[1], "+Z is up the screen");
  }
});

test("orbit: tilt is clamped so the camera never flips over the pole", () => {
  const c = camera(new THREE.Vector3(0, -10, 0), { yaw: 0, tilt: Math.PI / 2 });
  const h = orbitAbout(c, { yaw: 0, tilt: Math.PI / 2 }, new THREE.Vector3(), 0, -5);
  assert.equal(h.tilt, 0);
  const h2 = orbitAbout(c, h, new THREE.Vector3(), 0, 10);
  assert.equal(h2.tilt, Math.PI);
});

test("pan: the grabbed point follows the cursor pixel for pixel", () => {
  const c = camera(new THREE.Vector3(10, -13, 9), headingOf(new THREE.Vector3(-1.0, 1.3, -0.9)));
  const p = new THREE.Vector3(0.5, 0.2, 0.1);
  const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(c.quaternion);
  const depth = p.clone().sub(c.position).dot(forward);
  const before = pixel(c, p);
  panBy(c, depth, 37, -21, H);
  const after = pixel(c, p);
  close(after[0] - before[0], 37, 1e-6, "dx");
  close(after[1] - before[1], -21, 1e-6, "dy");
});

test("zoom: the point under the cursor keeps its pixel and the distance scales", () => {
  const c = camera(new THREE.Vector3(10, -13, 9), headingOf(new THREE.Vector3(-1.0, 1.3, -0.9)));
  const p = new THREE.Vector3(1.5, -0.4, 0.3);
  const before = pixel(c, p);
  const d0 = c.position.distanceTo(p);
  zoomToward(c, p, 1 / 1.15);
  const after = pixel(c, p);
  close(after[0], before[0], 1e-6, "x px");
  close(after[1], before[1], 1e-6, "y px");
  close(c.position.distanceTo(p), d0 / 1.15, 1e-9, "distance");
});

// ---- zoom limits (V1 finding) -----------------------------------------------

test("zoom: the distance to the zoom point never leaves [ZOOM_MIN, ZOOM_MAX] x radius", () => {
  const r = 10;
  // Repeated zoom-in from 5 r stops at ZOOM_MIN r; repeated zoom-out stops at ZOOM_MAX r.
  let d = 5 * r;
  for (let i = 0; i < 200; i++) d *= clampZoom(d, 1 / 1.15, r);
  close(d, ZOOM_MIN * r, 1e-9, "min distance");
  for (let i = 0; i < 200; i++) d *= clampZoom(d, 1.15, r);
  close(d, ZOOM_MAX * r, 1e-9, "max distance");
  // Inside the band a step is unclamped; at a limit the outward step is refused (keep 1) and the inward one allowed.
  assert.equal(clampZoom(r, 1.15, r), 1.15);
  assert.equal(clampZoom(ZOOM_MAX * r, 1.15, r), 1);
  assert.equal(clampZoom(ZOOM_MIN * r, 1 / 1.15, r), 1);
  assert.ok(clampZoom(ZOOM_MAX * r, 1 / 1.15, r) < 1);
  // Outside the band (a fit on a tiny model) only re-entry is allowed.
  assert.equal(clampZoom(100 * r, 1.15, r), 1);
  assert.ok(clampZoom(100 * r, 1 / 1.15, r) < 1);
});

// ---- listeners: the stale-drag fix and dispose (V1 findings) -----------------

type Listener = (e: unknown) => void;
class FakeTarget implements ListenTarget {
  readonly listeners = new Map<string, Listener[]>();
  readonly removed: string[] = [];
  clientHeight = 600;
  captured = new Set<number>();
  addEventListener(type: string, fn: (e: Event) => void): void {
    (this.listeners.get(type) ?? this.listeners.set(type, []).get(type)!).push(fn as Listener);
  }
  removeEventListener(type: string, fn: (e: Event) => void): void {
    const list = this.listeners.get(type) ?? [];
    const i = list.indexOf(fn as Listener);
    if (i < 0) throw new Error(`remove of a listener never added: ${type}`);
    list.splice(i, 1);
    this.removed.push(type);
  }
  fire(type: string, e: object): void {
    for (const fn of this.listeners.get(type) ?? []) fn(e);
  }
  setPointerCapture(id: number): void { this.captured.add(id); }
  hasPointerCapture(id: number): boolean { return this.captured.has(id); }
  releasePointerCapture(id: number): void { this.captured.delete(id); }
  get count(): number { return [...this.listeners.values()].reduce((a, l) => a + l.length, 0); }
}

function navigator() {
  const element = new FakeTarget(), keys = new FakeTarget();
  const selected: [number, number][] = [];
  const host: NavHost = {
    camera: camera(new THREE.Vector3(0, -10, 0), { yaw: 0, tilt: Math.PI / 2 }),
    element, keys,
    bounds: () => ({ center: new THREE.Vector3(), radius: 1 }),
    hitAt: () => null,
    rayAt: () => new THREE.Ray(),
    select: (x, y) => selected.push([x, y]),
    fit: () => {},
    changed: () => {},
  };
  return { nav: new Navigator(host), element, keys, selected };
}

const press = (id: number, button: number, x: number, y: number) => ({ pointerId: id, button, shiftKey: false, clientX: x, clientY: y });

test("a left press is captured, so a release off the canvas still ends the press and a new one can start", () => {
  const { nav, element, selected } = navigator();
  element.fire("pointerdown", press(1, 0, 100, 100));
  assert.ok(nav.dragging);
  assert.ok(element.hasPointerCapture(1), "a click-only press is captured too");
  // The release arrives through the capture, wherever the pointer went.
  element.fire("pointerup", press(1, 0, 900, 900));
  assert.ok(!nav.dragging, "the press ended");
  assert.deepEqual(selected, [], "a release far from the press is not a click");
  element.fire("pointerdown", press(2, 0, 10, 10));
  assert.ok(nav.dragging, "the next press is not ignored");
  element.fire("pointerup", press(2, 0, 11, 11));
  assert.deepEqual(selected, [[11, 11]]);
});

test("a lost capture or a cancel clears the drag state", () => {
  const { nav, element } = navigator();
  element.fire("pointerdown", press(1, 2, 100, 100));
  assert.ok(nav.dragging);
  element.fire("lostpointercapture", { pointerId: 1 });
  assert.ok(!nav.dragging);
  element.fire("pointerdown", press(3, 1, 100, 100));
  element.fire("pointercancel", { pointerId: 3 });
  assert.ok(!nav.dragging);
});

test("dispose removes every listener it added, on the canvas and on the key target", () => {
  const { nav, element, keys } = navigator();
  assert.equal(element.count, 7, "pointerdown/move/up/cancel, lostpointercapture, contextmenu, wheel");
  assert.equal(keys.count, 1, "keydown");
  nav.dispose();
  assert.equal(element.count, 0);
  assert.equal(keys.count, 0);
  assert.equal(element.removed.length + keys.removed.length, 8);
  element.fire("pointerdown", press(1, 0, 0, 0));
  assert.ok(!nav.dragging, "a disposed navigator ignores events");
});
