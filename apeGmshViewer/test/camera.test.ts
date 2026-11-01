// VA1 (ADR 0128 slice 1): the camera and the theme in the one store. The
// reducer records `setCamera` (named view, explicit pose, turn), `cameraMoved`
// and `setTheme` as pure transitions; the pose math the viewport applies them
// with is checked against closed forms (each named view's view direction and
// screen up, the pose round trip, every pixel of a placed pose); and the
// theme effect styles the document through `applyTheme`.

import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import * as THREE from "three";
import { Effects, type Bridge } from "../src/effects.ts";
import { openModel } from "../src/reader/node.ts";
import {
  headingOf, headingOfPose, ISO_FORWARD, Navigator, orientationOf, placePose, poseOf, syncOrthoFrustum, VIEW_HEADINGS, zoomToward,
  type HeadingView, type ListenTarget, type ViewCamera,
} from "../src/renderer/navigation.ts";
import { headingForZone, zoneById } from "../src/renderer/viewcube.ts";
import { BlobStore } from "../src/state/blobs.ts";
import { loadModel } from "../src/state/load.ts";
import { canonicalPose, initialState, reduce } from "../src/state/reduce.ts";
import { Store } from "../src/state/store.ts";
import { NAMED_VIEWS, type CameraPose, type CameraRequest, type ModelLoad, type State, type Vec3 } from "../src/state/types.ts";
import { cssVariables } from "../src/theme/tokens.ts";

const FIXTURE = fileURLToPath(new URL("../fixtures/shoebuckle.h5", import.meta.url));
const model = await openModel(FIXTURE);
const load: ModelLoad = loadModel(model, new BlobStore());
const loaded: State = reduce(initialState, { type: "fileLoaded", artifact: "model", load });

function deepFreeze<T>(v: T): T {
  if (v && typeof v === "object" && !Object.isFrozen(v)) {
    Object.freeze(v);
    for (const x of Object.values(v as object)) deepFreeze(x);
  }
  return v;
}

const near = (a: readonly number[], b: readonly number[], tol: number, what: string) =>
  a.forEach((x, i) => assert.ok(Math.abs(x - b[i]!) <= tol, `${what}[${i}]: ${x} vs ${b[i]}`));
const unit = (v: readonly number[]): Vec3 => {
  const l = Math.hypot(...v);
  return [v[0]! / l, v[1]! / l, v[2]! / l];
};
const setCamera = (s: State, to: CameraRequest) => reduce(s, { type: "setCamera", to });

// ---- the reducer ------------------------------------------------------------

test("setCamera: each named view is a pure transition that records the request; the viewport writes the pose back", () => {
  const frozen = deepFreeze(structuredClone(loaded));
  const selected = reduce(frozen, { type: "select", pick: { decl: loaded.mesh!.elements[0]!, at: null, nodes: null } });
  for (const view of NAMED_VIEWS) {
    const s = setCamera(view === "fit-selection" ? selected : frozen, view);
    assert.deepEqual(s.camera, { pose: null, request: { to: view, seq: 1 }, seq: 1 }, view);
    assert.deepEqual(JSON.parse(JSON.stringify(s.camera)), s.camera, `${view}: the camera state is plain data`);
  }
  const twice = setCamera(setCamera(frozen, "fit"), "fit");
  assert.deepEqual(twice.camera.request, { to: "fit", seq: 2 }, "the same view asked again is a new request (the viewport applies it again)");
});

test("setCamera: an explicit eye/target/up is the camera at once, with up made the unit screen-up", () => {
  const s = setCamera(loaded, { eye: [0, -10, 0], target: [0, 0, 0], up: [0, 1, 5] });
  // The part of up along the view direction (0, 1, 0) is dropped, and the rest made unit.
  assert.deepEqual(s.camera.pose, { eye: [0, -10, 0], target: [0, 0, 0], up: [0, 0, 1] });
  assert.deepEqual(s.camera.request, { to: s.camera.pose, seq: 1 });
  // Closed form, a tilted view: up is (Z - fz f) / |Z - fz f|, with f = (-3, 4, -5) / sqrt(50).
  const t = setCamera(loaded, { eye: [3, -4, 5], target: [0, 0, 0], up: [0, 0, 1] });
  near(t.camera.pose!.up, unit([-0.3, 0.4, 0.5]), 1e-15, "tilted up");
  const again = canonicalPose(t.camera.pose!, "again");
  for (const k of ["eye", "target", "up"] as const) near(again[k], t.camera.pose![k], 1e-15, `canonical form is a fixed point: ${k}`);
});

test("setCamera: the same event sequence gives the same camera state twice; a turn's tilt is clamped to [0, pi]", () => {
  const seq: CameraRequest[] = ["iso", { eye: [10, -12, 8], target: [1, 2, 3], up: [0, 0, 1] }, { yaw: 1, tilt: 4 }, "fit", "+x", { yaw: -7, tilt: -1 }];
  const run = () => seq.reduce(setCamera, loaded).camera;
  assert.deepEqual(run(), run());
  assert.deepEqual(setCamera(loaded, { yaw: 1, tilt: 4 }).camera.request?.to, { yaw: 1, tilt: Math.PI });
  assert.deepEqual(setCamera(loaded, { yaw: -7, tilt: -1 }).camera.request?.to, { yaw: -7, tilt: 0 });
});

test("setCamera refuses, naming why, and changes nothing", () => {
  const bad: [unknown, RegExp][] = [
    ["front", /"front" is not a named view/],
    [{ eye: [0, 0, NaN], target: [0, 0, 0], up: [0, 0, 1] }, /eye must be three finite numbers/],
    [{ eye: [0, 0], target: [0, 0, 0], up: [0, 0, 1] }, /eye must be three finite numbers/],
    [{ eye: [1, 2, 3], target: [1, 2, 3], up: [0, 0, 1] }, /eye and target are the same point/],
    [{ eye: [0, -10, 0], target: [0, 0, 0], up: [0, 2, 0] }, /is along the view direction/],
    [{ eye: [0, -10, 0], target: [0, 0, 0], up: [0, 0, 0] }, /is along the view direction/],
    [{ eye: [0, -10, 0], target: [0, 0, 0], up: [1, 0, 0] }, /rolls the camera by 90\.00 degrees/],
    [{ eye: [0, -10, 0], target: [0, 0, 0], up: [0, 0, -1] }, /rolls the camera by 180\.0 degrees/],
    [{ eye: [0, -10, 0], target: [0, 0, 0], up: [1e-5, 0, 1] }, /rolls the camera by/],
    [{ eye: [0, -10, 0], target: [0, 0, 0], up: [0, 0, 1], fov: 35 }, /exactly eye, target and up/],
    [{ yaw: 1, tilt: 1, roll: 0 }, /exactly yaw and tilt/],
    [{ yaw: Infinity, tilt: 1 }, /finite yaw and tilt/],
    [42, /not a named view, a pose \(eye, target, up\) or a turn/],
  ];
  const frozen = deepFreeze(structuredClone(loaded));
  for (const [to, why] of bad) assert.throws(() => setCamera(frozen, to as CameraRequest), why, JSON.stringify(to));
  // Within the roll tolerance (1e-6 rad) an up is accepted, and made exact.
  const ok = setCamera(frozen, { eye: [0, -10, 0], target: [0, 0, 0], up: [1e-8, 0, 1] });
  near(ok.camera.pose!.up, [1e-8, 0, 1], 1e-15, "a sub-tolerance up");
});

test("looking straight down or up, any up across the view is allowed and sets the heading", () => {
  for (const [eye, up] of [[[0, 0, 10], [1, 0, 0]], [[0, 0, 10], [0, -1, 0]], [[0, 0, -10], [1, 0, 0]], [[0, 0, -10], [-0.6, 0.8, 0]]] as [Vec3, Vec3][]) {
    const pose = setCamera(loaded, { eye, target: [0, 0, 0], up }).camera.pose!;
    const cam = new THREE.PerspectiveCamera();
    placePose(cam, pose);
    near(new THREE.Vector3(0, 0, -1).applyQuaternion(cam.quaternion).toArray(), [0, 0, -Math.sign(eye[2])], 1e-15, `${eye} forward`);
    near(new THREE.Vector3(0, 1, 0).applyQuaternion(cam.quaternion).toArray(), up, 1e-15, `${eye} ${up} screen up`);
  }
});

test("fit on an empty scene, any named view or turn with nothing drawn, and fit-selection with nothing selected, are no-ops", () => {
  for (const view of NAMED_VIEWS) assert.equal(setCamera(initialState, view), initialState, `${view} with nothing drawn`);
  assert.equal(setCamera(initialState, { yaw: 1, tilt: 1 }), initialState, "a turn with nothing drawn");
  assert.equal(setCamera(loaded, "fit-selection"), loaded, "fit-selection with nothing selected");
  // An explicit pose needs no scene; it is recorded as given (finite, no NaN).
  const s = setCamera(initialState, { eye: [1, 1, 1], target: [0, 0, 0], up: [0, 0, 1] });
  assert.ok([...s.camera.pose!.eye, ...s.camera.pose!.target, ...s.camera.pose!.up].every(Number.isFinite));
});

test("a re-read of the same model keeps the camera; another model opens with none (and the request count stands)", () => {
  const moved = reduce(setCamera(loaded, "iso"), { type: "cameraMoved", pose: { eye: [9, -9, 9], target: [1, 0, 0], up: [0, 0, 1] } });
  const reread = reduce(moved, { type: "fileLoaded", artifact: "model", load: loadModel(model, new BlobStore()) });
  assert.notEqual(reread.mesh, moved.mesh, "the re-read built a new mesh");
  assert.equal(reread.camera, moved.camera, "the re-read keeps the camera");
  const other = reduce(moved, { type: "fileLoaded", artifact: "model", load: { ...load, info: { ...load.info, path: "other.h5" } } });
  assert.deepEqual(other.camera, { pose: null, request: null, seq: 1 });
  for (const e of [{ type: "fileClosed", artifact: "model" }, { type: "fileFailed", artifact: "model", path: load.info.path, error: "x" }] as const) {
    const t = reduce(reduce(moved, { type: "setTheme", theme: "light" }), e);
    assert.equal(t.camera, moved.camera, `${e.type} keeps the camera`);
    assert.equal(t.theme, "light", `${e.type} keeps the theme`);
  }
});

test("a watch reload through a failed read keeps the camera; a failed open of another file drops it", () => {
  const moved = reduce(loaded, { type: "cameraMoved", pose: { eye: [9, -9, 9], target: [1, 0, 0], up: [0, 0, 1] } });
  const p = load.info.path;
  let s = reduce(moved, { type: "fileChanged", path: p });
  s = reduce(s, { type: "fileOpened", artifact: "model", path: p });
  s = reduce(s, { type: "fileFailed", artifact: "model", path: p, error: "half-written" });
  assert.equal(s.artifacts.model?.status, "failed");
  s = reduce(s, { type: "fileOpened", artifact: "model", path: p });
  s = reduce(s, { type: "fileLoaded", artifact: "model", load: loadModel(model, new BlobStore()) });
  assert.equal(s.camera, moved.camera, "the retry's fileLoaded keeps the camera, as any re-read of the open file does");
  let t = reduce(moved, { type: "fileOpened", artifact: "model", path: "other.h5" });
  assert.deepEqual(t.camera, { pose: null, request: null, seq: 0 }, "opening another file drops the camera");
  t = reduce(t, { type: "fileFailed", artifact: "model", path: "other.h5", error: "x" });
  t = reduce(t, { type: "fileOpened", artifact: "model", path: "other.h5" });
  t = reduce(t, { type: "fileLoaded", artifact: "model", load: { ...load, info: { ...load.info, path: "other.h5" } } });
  assert.equal(t.camera.pose, null, "the other file opens framed, not at the first file's camera");
});

test("the write-back of a near-vertical view far from the origin is accepted and places back to the same view", () => {
  // A georeferenced model (centre near 5e5, 4e6): target - eye loses most of the
  // tiny horizontal part of a near-vertical view direction to rounding.
  for (const c of [[0, 0, 0], [5e5, 4e6, 100]] as const) {
    for (const tilt of [0, 1e-12, 1e-9, 1e-6, 1e-4, 1e-2, Math.PI - 1e-9]) {
      const cam = new THREE.PerspectiveCamera(35, 4 / 3, 0.01, 1e7);
      cam.up.set(0, 0, 1);
      const nav = navOn(cam);
      nav.target.set(c[0], c[1], c[2]);
      cam.position.set(c[0], c[1], c[2] + 50);
      nav.snapTo({ yaw: 0.3, tilt });
      const pose = canonicalPose(poseOf(cam, nav.target, 5e-3, "perspective", 50), "cameraMoved");
      const again = new THREE.PerspectiveCamera(35, 4 / 3, 0.01, 1e7);
      placePose(again, pose);
      const what = `centre ${c} tilt ${tilt}`;
      near(again.position.toArray(), cam.position.toArray(), 0, `${what} eye`);
      near(forwardOf(again.quaternion), forwardOf(cam.quaternion), 1e-9, `${what} forward`);
      near(upOf(again.quaternion), upOf(cam.quaternion), 1e-9, `${what} screen up`);
    }
  }
});

test("cameraMoved writes the pose back without a request; the same pose again is a no-op; a bad pose is refused", () => {
  const pose: CameraPose = { eye: [9, -9, 9], target: [1, 0, 0], up: [0, 0, 2] };
  const s = reduce(loaded, { type: "cameraMoved", pose });
  assert.deepEqual(s.camera, { pose: canonicalPose(pose, "x"), request: null, seq: 0 });
  assert.equal(reduce(s, { type: "cameraMoved", pose }), s);
  assert.throws(() => reduce(s, { type: "cameraMoved", pose: { ...pose, up: [1, 1, 0] } }), /cameraMoved: up .* rolls the camera/);
});

test("setTheme: dark or light; the same theme is a no-op; anything else is refused; a re-read keeps it", () => {
  assert.equal(initialState.theme, "dark", "the app's default stays dark (D8's light default is for batch stills)");
  const light = reduce(loaded, { type: "setTheme", theme: "light" });
  assert.equal(light.theme, "light");
  assert.equal(reduce(light, { type: "setTheme", theme: "light" }), light);
  assert.throws(() => reduce(light, { type: "setTheme", theme: "sepia" as "light" }), /"sepia" is not a theme \(dark, light\)/);
  assert.equal(reduce(light, { type: "fileLoaded", artifact: "model", load }).theme, "light");
});

// ---- the theme effect -------------------------------------------------------

function fakeDocument() {
  const props: Record<string, string> = {};
  const root = { style: { colorScheme: "", setProperty: (k: string, v: string) => void (props[k] = v) }, dataset: {} as Record<string, string> };
  // `theme-fonts` present: applyTheme adds no font element.
  return { doc: { documentElement: root, getElementById: () => ({}) } as unknown as Document, root, props };
}

test("the theme effect styles the attached document with applyTheme on attach and on every setTheme", () => {
  const store = new Store();
  const effects = new Effects(store, new BlobStore(), {} as Bridge);
  const { doc, root, props } = fakeDocument();
  effects.setThemeTarget(doc);
  assert.equal(root.dataset["theme"], "dark");
  assert.deepEqual(props, cssVariables("dark"));
  store.dispatch({ type: "setTheme", theme: "light" });
  assert.equal(root.dataset["theme"], "light");
  assert.equal(root.style.colorScheme, "light");
  assert.deepEqual(props, cssVariables("light"));
  effects.dispose();
});

test("setTheme with no theme target attached raises instead of leaving the page unstyled", () => {
  const store = new Store();
  const effects = new Effects(store, new BlobStore(), {} as Bridge);
  assert.throws(() => store.dispatch({ type: "setTheme", theme: "light" }), /setTheme: no theme target is attached/);
  effects.dispose();
});

// ---- the pose math the viewport applies ------------------------------------

const forwardOf = (q: THREE.Quaternion) => new THREE.Vector3(0, 0, -1).applyQuaternion(q).toArray();
const upOf = (q: THREE.Quaternion) => new THREE.Vector3(0, 1, 0).applyQuaternion(q).toArray();

test("VIEW_HEADINGS: each fixed view looks along its axis with Z up (Y for top and bottom), as the ViewCube's faces do", () => {
  const expect: Record<HeadingView, [Vec3, Vec3]> = {
    iso: [unit(ISO_FORWARD.toArray()), [0, 0, 0]],
    "+x": [[-1, 0, 0], [0, 0, 1]],
    "-x": [[1, 0, 0], [0, 0, 1]],
    "+y": [[0, -1, 0], [0, 0, 1]],
    "-y": [[0, 1, 0], [0, 0, 1]],
    top: [[0, 0, -1], [0, 1, 0]],
    bottom: [[0, 0, 1], [0, -1, 0]],
  };
  for (const [view, [f, up]] of Object.entries(expect) as [HeadingView, [Vec3, Vec3]][]) {
    const q = orientationOf(VIEW_HEADINGS[view]);
    near(forwardOf(q), f, 1e-15, `${view} forward`);
    if (view !== "iso") near(upOf(q), up, 1e-15, `${view} up`);
  }
  assert.deepEqual(VIEW_HEADINGS.iso, headingOf(ISO_FORWARD.clone()), "iso is the heading a 3-D model opens on");
  // The ViewCube's face clicks (from yaw 0) land on the same orientations.
  for (const [view, face] of [["+x", "right"], ["-x", "left"], ["+y", "back"], ["-y", "front"], ["top", "top"], ["bottom", "bottom"]] as const) {
    const a = orientationOf(VIEW_HEADINGS[view]), b = orientationOf(headingForZone(zoneById(face)!, 0));
    assert.ok(Math.abs(Math.abs(a.dot(b)) - 1) < 1e-15, `${view} is the ViewCube's ${face} face`);
  }
});

const POSES: CameraPose[] = (
  [
    { eye: [10, -12, 8], target: [1, 2, 3], up: [0, 0, 1] },
    { eye: [0, -7, 0], target: [0, 0, 0], up: [0, 0, 1] },
    { eye: [2, 3, 40], target: [2, 3, -1], up: [0.6, -0.8, 0] },
    { eye: [-5, 4, -9], target: [1, -1, 2], up: [0, 0, 1] },
  ] satisfies CameraPose[]
).map((p) => canonicalPose(p, "fixture"));

test("placePose then poseOf gives the pose back, perspective and orthographic", () => {
  for (const pose of POSES) {
    const d = Math.hypot(...pose.eye.map((x, i) => x - pose.target[i]!));
    for (const [cam, projection] of [[new THREE.PerspectiveCamera(), "perspective"], [new THREE.OrthographicCamera(), "orthographic"]] as const) {
      placePose(cam, pose);
      const back = poseOf(cam, new THREE.Vector3(...pose.target), 1e-9, projection, d);
      for (const k of ["eye", "target", "up"] as const) near(back[k], pose[k], 1e-12 * Math.max(1, d), `${projection} ${k}`);
    }
  }
});

const W = 800, H = 600;
const pixel = (c: ViewCamera, p: THREE.Vector3): [number, number] => {
  c.updateMatrixWorld();
  const n = p.clone().project(c);
  return [((n.x + 1) / 2) * W, ((1 - n.y) / 2) * H];
};

/** A navigator on `camera` with no DOM: listeners go nowhere, the scene is a sphere of radius 5 at the origin. */
function navOn(camera: ViewCamera): Navigator {
  const nowhere: ListenTarget = { addEventListener() {}, removeEventListener() {} };
  return new Navigator({
    camera,
    element: { ...nowhere, clientHeight: H, setPointerCapture() {}, hasPointerCapture: () => false, releasePointerCapture() {} },
    keys: nowhere,
    bounds: () => ({ center: new THREE.Vector3(), radius: 5 }),
    hitAt: () => null,
    rayAt: () => new THREE.Ray(),
    select() {},
    fit() {},
    changed() {},
  });
}

test("the written-back pose of an off-axis orbit and zoom puts its target on the view axis, passes canonicalPose, and placed again shows every pixel where it was", () => {
  const points = [[0, 0, 0], [4, -2, 1], [-3, 3, 2], [1, 5, -4], [2, 2, 2]].map((p) => new THREE.Vector3(...p));
  for (const projection of ["perspective", "orthographic"] as const) {
    const make = (): ViewCamera => {
      const c = projection === "perspective" ? new THREE.PerspectiveCamera(35, W / H, 0.01, 1000) : new THREE.OrthographicCamera();
      c.up.set(0, 0, 1);
      return c;
    };
    const cam = make();
    const nav = navOn(cam);
    nav.place(canonicalPose({ eye: [12, -15, 10], target: [0, 0, 0], up: [0, 0, 1] }, "start"));
    // Orbit about a pivot off the axis, then zoom toward another point: the pivot (`nav.target`) is left off the axis.
    nav.orbit(new THREE.Vector3(3, 1, 2), 0.4, 0.3);
    zoomToward(cam, new THREE.Vector3(2, -1, 1), 0.6);
    nav.orthoDistance *= 0.6;
    if (projection === "orthographic") syncOrthoFrustum(cam as THREE.OrthographicCamera, nav.orthoDistance, W, H, 0, W, H, 35);
    const pose = poseOf(cam, nav.target, 1e-3 * 5, projection, nav.orthoDistance);
    const canon = canonicalPose(pose, "writeback");
    for (const k of ["eye", "target", "up"] as const) near(canon[k], pose[k], 1e-14 * 30, `${projection} canonical ${k}`);
    const axis = new THREE.Vector3(...pose.target).sub(new THREE.Vector3(...pose.eye)).normalize().toArray();
    near(axis, forwardOf(cam.quaternion), 1e-14, `${projection}: the target is on the view axis`);
    const again = make();
    const nav2 = navOn(again);
    nav2.place(pose);
    if (projection === "orthographic") syncOrthoFrustum(again as THREE.OrthographicCamera, nav2.orthoDistance, W, H, 0, W, H, 35);
    else (again as THREE.PerspectiveCamera).updateProjectionMatrix();
    for (const p of points) near(pixel(again, p), pixel(cam, p), 1e-9, `${projection} pixel of ${p.toArray()}`);
  }
});

test("a ViewCube drag dispatched as a turn reproduces the turntable orbit about the pivot, tilt clamp included", () => {
  for (const [dYaw, dTilt] of [[0.3, -0.2], [-1.1, 0.5], [0.2, 9]] as const) {
    const a = new THREE.PerspectiveCamera(), b = new THREE.PerspectiveCamera();
    const na = navOn(a), nb = navOn(b);
    const start = canonicalPose({ eye: [12, -15, 10], target: [1, 2, 0], up: [0, 0, 1] }, "start");
    na.place(start);
    nb.place(start);
    na.orbit(na.target, dYaw, dTilt);
    const to = setCamera(loaded, { yaw: nb.heading.yaw + dYaw, tilt: nb.heading.tilt + dTilt }).camera.request!.to as { yaw: number; tilt: number };
    nb.snapTo(to);
    near(b.position.toArray(), a.position.toArray(), 1e-12, `position after (${dYaw}, ${dTilt})`);
    near(b.quaternion.toArray(), a.quaternion.toArray(), 1e-15, `orientation after (${dYaw}, ${dTilt})`);
  }
});

test("headingOfPose off the vertical is headingOf of the view direction", () => {
  for (const pose of POSES) {
    const f = new THREE.Vector3(...pose.target).sub(new THREE.Vector3(...pose.eye));
    if (Math.hypot(f.x, f.y) < 1e-9) continue;
    const a = headingOfPose(f, new THREE.Vector3(...pose.up)), b = headingOf(f);
    near([a.yaw, a.tilt], [b.yaw, b.tilt], 1e-15, "heading");
  }
});
