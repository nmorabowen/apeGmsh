// Navigation: the bindings table against the maintainer's CAD convention
// (#1285 addendum), the pivot fallback, and the turntable invariants, each
// checked against a closed form (the pivot keeps its pixel, Z stays vertical
// on screen, a grabbed point follows the cursor, a zoom point keeps its pixel).

import assert from "node:assert/strict";
import { test } from "node:test";
import * as THREE from "three";
import { bindingFor, buttonOf, keyAction, POINTER_BINDINGS, WHEEL_ACTION, type Button } from "../src/renderer/bindings.ts";
import { headingOf, nearestHit, orbitAbout, orientationOf, panBy, pivotFor, zoomToward, type Heading } from "../src/renderer/navigation.ts";

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
