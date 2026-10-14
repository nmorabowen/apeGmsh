// V4e1 (#1834): section planes, ADR 0083's contracts as tests in Node (the
// viewport does not run here; its pieces do: the reducer, the three.js
// planes and their stamping, the pick filter, the slider's box):
//
// - C-CUT: the kept and discarded corners of a brick for a plane through it,
//   flip inverting them, and the GPU's own rule (three.js's clipping shader
//   chunk, emulated in view space) agreeing with the pick filter's;
// - C-PICK: a hit on the discarded side is a miss; the same hit with cuts off
//   is a hit; a cut-away line leaves the faces to answer;
// - C-DEFORM: for a fixed plane, the kept vertices of the deformed mesh change
//   between two steps (contour_staged, through the app's own effects);
// - C-LIMIT: a 7th active plane is refused, with a notice;
// - C-REBUILD: an object created after a plane exists is clipped at its first
//   render; a plane change moves the shared planes in place;
// - refusals: a zero or non-finite normal, a non-finite offset; the offset is
//   never clamped; the slider maps the reference (undeformed) box;
// - reload: planes survive re-reads, a failed and a closed model.

import assert from "node:assert/strict";
import { test } from "node:test";
import * as THREE from "three";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import { planeRows } from "../../src/panels/planes.ts";
import { SectionClip, sliderBox } from "../../src/renderer/clip.ts";
import { basePositions, deformFull, positionsFor } from "../../src/renderer/deform.ts";
import { hitElement } from "../../src/renderer/selection.ts";
import { boxOf, cutsOf, isClipped, LIMIT_NOTICE, MAX_ACTIVE_PLANES, offsetRange, unitNormal } from "../../src/state/clip.ts";
import { initialState, reduce } from "../../src/state/reduce.ts";
import { deformView } from "../../src/state/selectors.ts";
import type { ClipState, Event, MeshInfo, State } from "../../src/state/types.ts";
import { opened, until } from "../contour/harness.ts";

/** A state with a mesh centred at `center` (only the centre is read by the clip reducer). */
const withCenter = (center: [number, number, number]): State => ({ ...initialState, mesh: { center } as unknown as MeshInfo });
const run = (s: State, ...events: Event[]): State => events.reduce(reduce, s);
const planesOf = (s: State) => s.clip.planes;

/** The corners of the brick [0, 2] x [0, 4] x [0, 6]. */
const BRICK: [number, number, number][] = [];
for (const x of [0, 2]) for (const y of [0, 4]) for (const z of [0, 6]) BRICK.push([x, y, z]);

/** The corners a clip state keeps (index list). */
const kept = (c: ClipState, pts: readonly (readonly number[])[]) => pts.flatMap((p, i) => (isClipped(cutsOf(c), p) ? [] : [i]));

// ---- C-CUT ----------------------------------------------------------------------

test("C-CUT: a Z plane through a brick keeps the corners its normal points to; flip keeps the others", () => {
  let s = run(withCenter([1, 2, 3]), { type: "addClipPlane", normal: [0, 0, 1] });
  const p = planesOf(s)[0]!;
  assert.deepEqual(p.normal, [0, 0, 1]);
  assert.equal(p.offset, 3, "a new plane passes through the reference centre");
  const up = kept(s.clip, BRICK);
  assert.deepEqual(up.map((i) => BRICK[i]![2]), [6, 6, 6, 6], "the four corners above z = 3");
  s = run(s, { type: "setClipFlipped", id: p.id, flipped: true });
  const down = kept(s.clip, BRICK);
  assert.deepEqual(down.map((i) => BRICK[i]![2]), [0, 0, 0, 0], "flipped: the four below");
  assert.deepEqual([...up, ...down].sort(), BRICK.map((_, i) => i), "flip inverts the kept set exactly");
  // Two planes: the kept region is the intersection of the kept halves (ADR 0083's mapper clip planes; three.js's default).
  s = run(s, { type: "setClipFlipped", id: p.id, flipped: false }, { type: "addClipPlane", normal: [1, 0, 0] });
  assert.deepEqual(kept(s.clip, BRICK).map((i) => BRICK[i]), [[2, 0, 6], [2, 4, 6]], "a corner cut keeps x > 1 and z > 3");
  // Apply cuts off: nothing is cut, the planes stay.
  const off = run(s, { type: "setApplyCuts", on: false });
  assert.equal(kept(off.clip, BRICK).length, 8);
  assert.equal(off.clip.planes.length, 2);
});

test("C-CUT: the GPU's discard rule (three.js clipping_planes_fragment, in view space) agrees with the pick filter", () => {
  // The rule three.js r186 compiles into every clipped material: a fragment is discarded when
  // dot(vClipPosition, plane.xyz) > plane.w, with vClipPosition = -mvPosition.xyz and the plane moved into view space
  // (WebGLClipping.projectPlanes). A three.js upgrade that changes it fails here first.
  const chunk = THREE.ShaderChunk.clipping_planes_fragment;
  assert.match(chunk, /if \( dot\( vClipPosition, plane\.xyz \) > plane\.w \) discard;/);
  assert.match(THREE.ShaderChunk.clipping_planes_vertex, /vClipPosition = - mvPosition\.xyz;/);
  const s = run(
    withCenter([1, 2, 3]),
    { type: "addClipPlane", normal: [0, 0, 1] },
    { type: "addClipPlane", normal: [1, 1, 0] },
    { type: "addClipPlane", normal: [-1, 0.3, 2] },
  );
  const flipped = run(s, { type: "setClipFlipped", id: "plane-2", flipped: true }, { type: "setClipOffset", id: "plane-3", offset: -0.7 });
  const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 100);
  camera.position.set(9, -7, 5);
  camera.lookAt(1, 2, 3);
  camera.updateMatrixWorld();
  const view = camera.matrixWorldInverse;
  const viewNormal = new THREE.Matrix3().getNormalMatrix(view);
  let seed = 7;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 8 - 3;
  for (const state of [s, flipped]) {
    const clip = new SectionClip();
    clip.sync(state.clip);
    const viewPlanes = clip.planes.map((p) => p.clone().applyMatrix4(view, viewNormal));
    for (let i = 0; i < 400; i++) {
      const p = new THREE.Vector3(rnd(), rnd(), rnd());
      const clipPos = p.clone().applyMatrix4(view).negate();
      const gpu = viewPlanes.some((vp) => clipPos.dot(vp.normal) > vp.constant + 1e-9);
      const near = viewPlanes.some((vp) => Math.abs(clipPos.dot(vp.normal) - vp.constant) < 1e-6);
      if (near) continue;
      assert.equal(clip.clipped(p, 0), gpu, `point ${p.toArray()}`);
      assert.equal(isClipped(cutsOf(state.clip), p.toArray()), gpu);
    }
  }
});

// ---- C-PICK ---------------------------------------------------------------------

/** A 2 x 2 x 2 box's top face (z = 2) as two triangles, a beam above it along x at z = 3, and a camera looking down. */
function pickScene() {
  const top = new Float32Array([0, 0, 2, 2, 0, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 0, 2, 2]);
  const faces = new THREE.Mesh(new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(top, 3)), new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }));
  const lg = new LineSegmentsGeometry();
  lg.setPositions(new Float32Array([0, 1, 3, 2, 1, 3]));
  const lm = new LineMaterial({ linewidth: 3, worldUnits: false });
  lm.resolution.set(800, 800);
  const lines = new LineSegments2(lg, lm);
  const camera = new THREE.OrthographicCamera(-4, 4, 4, -4, 0.1, 100);
  camera.position.set(1, 1, 10);
  camera.lookAt(1, 1, 0);
  camera.updateMatrixWorld();
  faces.updateMatrixWorld();
  lines.updateMatrixWorld();
  const ray = new THREE.Raycaster();
  ray.params.Line2 = { threshold: 7 };
  const elements = ["beam", "shell"];
  const drawn = { lineMap: Int32Array.from([0]), triMap: Int32Array.from([0, 1]) };
  const at = (x: number, y: number, clip: SectionClip) => {
    const ndc = new THREE.Vector3(x, y, 2).project(camera);
    ray.setFromCamera(new THREE.Vector2(ndc.x, ndc.y), camera);
    const hit = hitElement(ray, lines, faces, drawn, elements, Int32Array.from([0]), Int32Array.from([1, 1]), 2, (p) => clip.clipped(p, 1e-6 * 2));
    return hit ? elements[hit.element]! : null;
  };
  return { at };
}

test("C-PICK: a hit on the cut-away side is a miss; the same click with cuts off is a hit", () => {
  const { at } = pickScene();
  const clip = new SectionClip();
  // A plane keeping z <= 1 (normal -Z): the top face (z = 2) and the beam (z = 3) are cut away.
  let s = run(withCenter([1, 1, 1]), { type: "addClipPlane", normal: [0, 0, -1] });
  clip.sync(s.clip);
  assert.equal(at(0.5, 0.5, clip), null, "the face under the cursor is cut away: a miss, never the hidden face");
  assert.equal(at(1, 1, clip), null, "the beam and the face are cut away: a miss");
  s = run(s, { type: "setApplyCuts", on: false });
  clip.sync(s.clip);
  assert.equal(at(0.5, 0.5, clip), "shell", "cuts off: the face");
  assert.equal(at(1, 1, clip), "beam", "cuts off: lines win over faces");
  // A plane keeping z <= 2.5: the beam is cut away, the face is kept; the faces answer the beam's click.
  s = run(s, { type: "setApplyCuts", on: true }, { type: "setClipOffset", id: "plane-1", offset: -2.5 });
  clip.sync(s.clip);
  assert.equal(at(1, 1, clip), "shell", "a cut-away line leaves the faces to answer");
  assert.equal(at(0.5, 0.5, clip), "shell", "a hit on the cut face itself (within the on-plane band) is kept");
});

// ---- C-DEFORM -------------------------------------------------------------------

test("C-DEFORM: for a fixed plane, the kept vertices of the deformed mesh change between step 0 and the last step", async () => {
  const h = await opened();
  const last = 3; // stage "unprop", 4 steps
  const show = async (k: number) => {
    h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 2, step: 0 } });
    if (k !== 0) h.store.dispatch({ type: "setResultStep", step: k });
    h.store.dispatch({ type: "setDeform", on: true });
    h.store.dispatch({ type: "setDeformScale", scale: 50 });
    await until(() => {
      const v = deformView(h.store.get());
      return (v.state === "ready" || v.state === "partial") && h.effects.currentDisplacement()?.key === v.keys.disp;
    }, `the deformed shape of step ${k}`);
    const s = h.store.get();
    return positionsFor(s, s.mesh!, h.blobs, h.effects.currentDisplacement(), null)!;
  };
  const center = h.store.get().mesh!.center;
  h.store.dispatch({ type: "addClipPlane", normal: [0, 0, 1] });
  const cuts = cutsOf(h.store.get().clip);
  assert.equal(cuts.length, 1);
  assert.equal(-cuts[0]!.constant, center[2], "the plane through the reference centre, fixed in world space");
  const keptOf = (tri: Float32Array) => {
    const out: number[] = [];
    for (let v = 0; v < tri.length / 3; v++) if (!isClipped(cuts, [tri[3 * v]!, tri[3 * v + 1]!, tri[3 * v + 2]!])) out.push(v);
    return out;
  };
  const k0 = keptOf((await show(0)).tri);
  const kN = keptOf((await show(last)).tri);
  assert.ok(k0.length > 0 && kN.length > 0);
  assert.notDeepEqual(k0, kN, "the deformed shape moves through the world-fixed plane");
  // The plane itself did not move: the same cut, the same state.
  assert.deepEqual(cutsOf(h.store.get().clip), cuts);
  h.effects.dispose();
});

// ---- C-LIMIT --------------------------------------------------------------------

test("C-LIMIT: six planes cut; a 7th is created inactive with a notice, and activating it is refused", () => {
  let s = withCenter([0, 0, 0]);
  for (let i = 0; i < MAX_ACTIVE_PLANES; i++) s = run(s, { type: "addClipPlane", normal: [1, i, 0] });
  assert.equal(s.clip.planes.filter((p) => p.active).length, 6);
  assert.equal(s.clip.notice, null);
  s = run(s, { type: "addClipPlane", normal: [0, 0, 1] });
  const seventh = s.clip.planes[6]!;
  assert.equal(seventh.active, false, "created, inactive");
  assert.equal(s.clip.notice, LIMIT_NOTICE);
  assert.equal(cutsOf(s.clip).length, 6);
  const refused = run(s, { type: "setClipActive", id: seventh.id, active: true });
  assert.equal(refused.clip.planes[6]!.active, false, "activating a 7th is refused");
  assert.equal(refused.clip.notice, LIMIT_NOTICE);
  assert.deepEqual(refused.clip.planes, s.clip.planes, "a refusal changes no plane");
  // Deactivate one, then the 7th may cut; the notice clears.
  const swapped = run(s, { type: "setClipActive", id: "plane-1", active: false }, { type: "setClipActive", id: seventh.id, active: true });
  assert.equal(swapped.clip.planes[6]!.active, true);
  assert.equal(swapped.clip.notice, null);
  assert.equal(cutsOf(swapped.clip).length, 6);
});

// ---- C-REBUILD ------------------------------------------------------------------

test("C-REBUILD: an object created after a plane exists is clipped at its first render; a plane change moves the shared planes", () => {
  const clip = new SectionClip();
  const scene = new THREE.Scene();
  const content = new THREE.Group();
  scene.add(content, new THREE.HemisphereLight());
  content.add(new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshStandardMaterial()));
  let s = run(withCenter([0, 0, 0]), { type: "addClipPlane", normal: [0, 0, 1] });
  assert.equal(clip.sync(s.clip), true);
  assert.equal(clip.beforeRender(scene), 1);
  // A rebuild after the plane exists: new faces, thick lines, ghost lines and the markers (points), each new material.
  const lg = new LineSegmentsGeometry();
  lg.setPositions(new Float32Array([0, 0, 0, 1, 0, 0]));
  const late = [
    new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshStandardMaterial()),
    new LineSegments2(lg, new LineMaterial({ linewidth: 3 })),
    new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial()),
    new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial()),
  ];
  const ghost = new THREE.Group();
  ghost.add(late[2]!);
  content.add(late[0]!, late[1]!, late[3]!);
  scene.add(ghost);
  assert.equal(clip.beforeRender(scene), 4, "the next render stamps the four new materials");
  for (const o of late) {
    const m = (o as THREE.Mesh).material as THREE.Material;
    assert.equal(m.clippingPlanes, clip.planes, `${o.type}: the shared planes`);
    assert.equal(m.clippingPlanes!.length, 1);
  }
  assert.equal((late[1] as LineSegments2).material.clipping, true, "LineMaterial compiles the clipping chunks");
  // A plane change moves the shared array in place: no re-stamp, the same Plane objects.
  const before = clip.planes[0]!;
  s = run(s, { type: "setClipOffset", id: "plane-1", offset: 2.5 });
  assert.equal(clip.sync(s.clip), true);
  assert.equal(clip.planes[0], before);
  assert.equal(before.constant, -2.5);
  assert.equal(clip.beforeRender(scene), 0);
  // Cuts off: the shared array empties (three.js compiles no clipping), the materials keep pointing at it.
  s = run(s, { type: "setApplyCuts", on: false });
  clip.sync(s.clip);
  assert.equal(clip.planes.length, 0);
  assert.equal((late[0] as THREE.Mesh).material instanceof THREE.Material && ((late[0] as THREE.Mesh).material as THREE.Material).clippingPlanes, clip.planes);
});

// ---- refusals, unclamped offsets, the slider's box ------------------------------

test("a zero or non-finite normal is refused with a notice; a normal is normalised on entry", () => {
  let s = run(withCenter([0, 0, 0]), { type: "addClipPlane", normal: [0, 0, 0] });
  assert.equal(s.clip.planes.length, 0);
  assert.match(s.clip.notice!, /needs a direction/);
  s = run(s, { type: "addClipPlane", normal: [2, 0, 0] });
  assert.deepEqual(s.clip.planes[0]!.normal, [1, 0, 0], "normalised");
  assert.equal(s.clip.notice, null, "an accepted event clears the notice");
  for (const bad of [[0, 0, 0], [NaN, 1, 0], [Infinity, 0, 0], [1, 0]]) {
    const r = run(s, { type: "setClipNormal", id: "plane-1", normal: bad });
    assert.deepEqual(r.clip.planes, s.clip.planes, `${JSON.stringify(bad)} changes no plane`);
    assert.match(r.clip.notice!, /needs a direction/);
  }
  assert.equal(unitNormal([1e-300, 0, 0]), null, "too short to normalise");
  const r = run(s, { type: "setClipNormal", id: "plane-1", normal: [0, 3, 4] });
  assert.deepEqual(r.clip.planes[0]!.normal, [0, 0.6, 0.8]);
});

test("a new normal keeps the plane through the point of the old plane nearest the reference centre", () => {
  let s = run(withCenter([1, 2, 3]), { type: "addClipPlane", normal: [0, 0, 1] }, { type: "setClipOffset", id: "plane-1", offset: 5 });
  s = run(s, { type: "setClipNormal", id: "plane-1", normal: [1, 0, 0] });
  // The old plane's point nearest the centre is (1, 2, 5); the X plane through it has offset 1.
  assert.equal(s.clip.planes[0]!.offset, 1);
});

test("the offset is never clamped; a non-finite offset is refused with a notice", () => {
  let s = run(withCenter([0, 0, 0]), { type: "addClipPlane", normal: [0, 0, 1] });
  s = run(s, { type: "setClipOffset", id: "plane-1", offset: 1e9 });
  assert.equal(s.clip.planes[0]!.offset, 1e9, "far outside any model, accepted as typed");
  for (const bad of [NaN, Infinity, -Infinity]) {
    const r = run(s, { type: "setClipOffset", id: "plane-1", offset: bad });
    assert.equal(r.clip.planes[0]!.offset, 1e9);
    assert.match(r.clip.notice!, /finite number/);
  }
  assert.throws(() => run(s, { type: "setClipOffset", id: "plane-9", offset: 1 }), /no section plane/, "an unknown id is a programming error");
  // The panel's row: the field shows the offset itself, the slider sits at its end, flagged outside.
  const box = { lo: [0, 0, 0], hi: [2, 4, 6] } as const;
  const [row] = planeRows(s.clip.planes, box);
  assert.equal(row!.offset, 1e9);
  assert.deepEqual([row!.slider!.min, row!.slider!.max, row!.slider!.value], [0, 6, 6]);
  assert.equal(row!.outside, true);
});

test("offsetRange: the box's extent along the normal", () => {
  const box = { lo: [0, 0, 0], hi: [2, 4, 6] } as const;
  assert.deepEqual(offsetRange([0, 0, 1], box), { min: 0, max: 6 });
  assert.deepEqual(offsetRange([-1, 0, 0], box), { min: -2, max: 0 });
  const d = offsetRange([Math.SQRT1_2, Math.SQRT1_2, 0], box);
  assert.ok(Math.abs(d.min - 0) < 1e-12 && Math.abs(d.max - 6 * Math.SQRT1_2) < 1e-12);
});

test("the slider maps the reference (undeformed) box, never the deformed one", async () => {
  const h = await opened();
  const s = h.store.get();
  const info = s.mesh!;
  const base = basePositions(info, h.blobs);
  const ref = boxOf(base.line, base.tri)!;
  // A deformed shape far outside the reference box (every vertex moved +10 in z).
  const disp = { key: "d", line: { d: new Float32Array(base.line.length).fill(10), missing: new Uint8Array(0), count: 0 }, tri: { d: new Float32Array(base.tri.length).fill(10), missing: new Uint8Array(0), count: 0 }, edge: { d: new Float32Array(base.edge.length).fill(10), missing: new Uint8Array(0), count: 0 } };
  const deformed = deformFull(base, disp as never, 1, "d|scale 1");
  const moved = boxOf(deformed.line, deformed.tri)!;
  assert.notDeepEqual(moved, ref, "the test can tell the two boxes apart");
  assert.deepEqual(sliderBox(deformed, info, h.blobs), ref);
  assert.deepEqual(sliderBox(null, info, h.blobs), ref);
  h.effects.dispose();
});

// ---- reload ---------------------------------------------------------------------

test("reload: the planes survive a model re-read, a failed and a closed model, and a results re-read", async () => {
  const h = await opened();
  h.store.dispatch({ type: "addClipPlane", normal: [0, 0, 1] });
  h.store.dispatch({ type: "setClipOffset", id: "plane-1", offset: 1.25 });
  h.store.dispatch({ type: "addClipPlane", normal: [1, 0, 0] });
  h.store.dispatch({ type: "setClipFlipped", id: "plane-2", flipped: true });
  const clip = h.store.get().clip;
  const mesh = h.store.get().mesh!;
  const oldBox = sliderBox(null, mesh, h.blobs);
  const model = h.store.get().artifacts.model!.path;
  // A D1 re-run: the same files read again (a new MeshInfo, so the slider's box is derived again).
  h.store.dispatch({ type: "fileChanged", path: model });
  await h.effects.open(model);
  await until(() => h.store.get().mesh !== mesh && h.store.get().mesh !== null, "the re-read mesh");
  assert.equal(h.store.get().clip, clip, "a model re-read keeps the planes and their offsets");
  const newBox = sliderBox(null, h.store.get().mesh!, h.blobs);
  assert.notEqual(newBox, oldBox, "the box is derived again from the new read (the old read's blobs are released)");
  assert.deepEqual(newBox, oldBox, "the same file: the same extent");
  const results = h.store.get().artifacts.results!.path;
  await h.effects.openResults(results);
  assert.equal(h.store.get().clip, clip, "a results re-read keeps them");
  const failed = reduce(h.store.get(), { type: "fileFailed", artifact: "model", path: model, error: "x" });
  assert.equal(failed.clip, clip, "a failed model read keeps them");
  const closed = reduce(h.store.get(), { type: "fileClosed", artifact: "model" });
  assert.equal(closed.clip, clip, "a closed model keeps them");
  h.effects.dispose();
});
