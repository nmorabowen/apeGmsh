// ADR 0138 D5: a click takes the element drawn at the pixel, from an ID image,
// with a line within CLICK_LINE_PX winning over the face under the cursor.
//
// The two faults of the ray cast clicks used before, measured on the slab
// model (internal_docs/spike_viewer_selection.md, "Clicks"), rebuilt here:
// a beam hidden behind a slab took the click from the slab, and on a beam
// split into segments shorter than about 10 px the segment nearer the eye
// took the click from the one under the cursor.
//
// The GPU draws the ID image in the app; here `idWindow` stands in for it,
// from the same geometry: a face covers a pixel whose centre ray meets it, a
// line the pixels whose centre is within half a pixel of it (1 px wide, as
// the image draws lines), the nearest wins, and a face sits a little behind
// so a line drawn on a face wins there. The capture stills prove the GPU part.

import assert from "node:assert/strict";
import { test } from "node:test";
import * as THREE from "three";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import { CLICK_LINE_PX, clickId } from "../src/renderer/picking.ts";
import { hitElement } from "../src/renderer/selection.ts";

const W = 1584, H = 961;

/** A scene as the viewport draws it: one face element (triangles) and line elements (segments), with the objects the ray cast reads. */
function scene(tris: number[], segs: number[], camera: THREE.Camera) {
  const faces = new THREE.Mesh(new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(new Float32Array(tris), 3)), new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }));
  const g = new LineSegmentsGeometry();
  g.setPositions(segs);
  const m = new LineMaterial({ linewidth: 3, worldUnits: false });
  m.resolution.set(W, H);
  const lines = new LineSegments2(g, m);
  faces.updateMatrixWorld();
  lines.updateMatrixWorld();
  const nT = tris.length / 9, nL = segs.length / 6;
  // Segment i is element i; every triangle is the face element, nL.
  const lineElement = Int32Array.from({ length: nL }, (_, i) => i), triElement = new Int32Array(nT).fill(nL);
  const drawn = { lineMap: Int32Array.from({ length: nL }, (_, i) => i), triMap: Int32Array.from({ length: nT }, (_, i) => i) };
  const elements = [...Array.from({ length: nL }, (_, i) => `mesh/element/beam${i}`), "mesh/element/slab"];
  const ray = new THREE.Raycaster();
  ray.params.Line2 = { threshold: 7 };
  const aim = (x: number, y: number) => ray.setFromCamera(new THREE.Vector2(((x + 0.5) / W) * 2 - 1, -(((y + 0.5) / H) * 2 - 1)), camera);
  const cast = (x: number, y: number, only: number | null) => {
    aim(x, y);
    return hitElement(ray, lines, faces, drawn, elements, lineElement, triElement, 10, () => false, null, null, null, only);
  };
  const px = (p: THREE.Vector3) => {
    const c = p.clone().project(camera);
    return [((c.x + 1) / 2) * W - 0.5, ((1 - c.y) / 2) * H - 0.5] as const;
  };
  /** The ID image's window of (2 r + 1)^2 pixels around canvas pixel (x, y) from the top left, row 0 at the bottom; an ID is element + 1. */
  const idWindow = (x: number, y: number, r: number) => {
    const n = 2 * r + 1, ids = new Uint32Array(n * n), a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3(), hit = new THREE.Vector3(), on = new THREE.Vector3();
    for (let wy = 0; wy < n; wy++) {
      for (let wx = 0; wx < n; wx++) {
        const cx = x - r + wx, cy = y + r - wy;
        aim(cx, cy);
        let best = 0, depth = Infinity;
        for (let t = 0; t < nT; t++) {
          a.fromArray(tris, 9 * t), b.fromArray(tris, 9 * t + 3), c.fromArray(tris, 9 * t + 6);
          if (!ray.ray.intersectTriangle(a, b, c, false, hit)) continue;
          const d = hit.distanceTo(ray.ray.origin) * (1 + 1e-6);
          if (d < depth) [best, depth] = [nL + 1, d];
        }
        for (let l = 0; l < nL; l++) {
          a.fromArray(segs, 6 * l), b.fromArray(segs, 6 * l + 3);
          const [ax, ay] = px(a), [bx, by] = px(b);
          const len2 = (bx - ax) ** 2 + (by - ay) ** 2, s = len2 ? Math.max(0, Math.min(1, ((cx - ax) * (bx - ax) + (cy - ay) * (by - ay)) / len2)) : 0;
          if (Math.hypot(ax + s * (bx - ax) - cx, ay + s * (by - ay) - cy) > 0.5) continue;
          ray.ray.distanceSqToSegment(a, b, undefined, on);
          const d = on.distanceTo(ray.ray.origin);
          if (d <= depth) [best, depth] = [l + 1, d];
        }
        ids[wy * n + wx] = best;
      }
    }
    return ids;
  };
  /** What a click at canvas pixel (x, y) takes now: the element from the ID image, then where the ray meets it. */
  const click = (x: number, y: number) => {
    const r = CLICK_LINE_PX, e = clickId(idWindow(x, y, r), 2 * r + 1, 2 * r + 1, r, r, r, (el) => el < nL);
    return e < 0 ? null : cast(x, y, e);
  };
  return { click, cast, px, nL };
}

test("clickId: the line nearest the centre within r wins, else the element at the centre; the alpha byte is not part of an ID", () => {
  // A 5 x 5 window, centre (2, 2): a face (element 9, ID 10, alpha byte set) everywhere, a line (element 3) at (4, 2).
  const ids = new Uint32Array(25).fill(10 | (0xff << 24));
  ids[2 * 5 + 4] = 4;
  const isLine = (e: number) => e === 3 || e === 5;
  assert.equal(clickId(ids, 5, 5, 2, 2, 2, isLine), 3, "a line 2 px away takes the click from the face under the cursor");
  assert.equal(clickId(ids, 5, 5, 2, 2, 1, isLine), 9, "no line within 1 px: the face at the centre");
  ids[1 * 5 + 2] = 6;
  assert.equal(clickId(ids, 5, 5, 2, 2, 2, isLine), 5, "a line 1 px away beats one 2 px away: the nearest wins");
  assert.equal(clickId(new Uint32Array(25), 5, 5, 2, 2, 2, isLine), -1, "nothing drawn: nothing");
});

test("a beam split into segments shorter than the ray cast's tolerance: a click takes the segment under the cursor", () => {
  // The spike's reproduction: a straight beam sloping toward the eye, 400 segments of about 5.6 px on screen.
  const camera = new THREE.PerspectiveCamera(30, W / H, 0.1, 1000);
  camera.position.set(0, -12, 8);
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld();
  const n = 400, a = new THREE.Vector3(-6, -3, 0), b = new THREE.Vector3(6, 3, 0), segs: number[] = [];
  for (let i = 0; i < n; i++) segs.push(...a.clone().lerp(b, i / n).toArray(), ...a.clone().lerp(b, (i + 1) / n).toArray());
  const { click, cast, px } = scene([], segs, camera);
  let now = 0, before = 0, clicks = 0;
  for (let i = 5; i < n - 5; i += 10) {
    const mid = new THREE.Vector3().fromArray(segs, 6 * i).add(new THREE.Vector3().fromArray(segs, 6 * i + 3)).multiplyScalar(0.5);
    const [x, y] = px(mid).map(Math.round) as [number, number];
    clicks++;
    const hit = click(x, y);
    if (hit?.element === i) now++;
    if (cast(x, y, null)?.element === i) before++;
    if (hit) assert.equal(hit.prim.kind, "line");
  }
  assert.equal(now, clicks, `every click takes the segment under the cursor (${now} of ${clicks})`);
  assert.equal(before, 0, "the ray cast alone took a neighbour on every one of these clicks (why clicks moved to the ID image)");
});

test("a beam hidden behind a slab never takes the click; a beam drawn on the slab takes it within CLICK_LINE_PX", () => {
  // Looking down at a 10 x 10 slab at z = 0. One beam along x under it (z = -0.05, hidden), one on its surface (z = 0) 2 units away in y.
  const camera = new THREE.PerspectiveCamera(30, W / H, 0.1, 1000);
  camera.position.set(0, 0, 30);
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld();
  const slab = [-5, -5, 0, 5, -5, 0, 5, 5, 0, -5, -5, 0, 5, 5, 0, -5, 5, 0];
  const segs = [-4, 0, -0.05, 4, 0, -0.05, -4, 2, 0, 4, 2, 0];
  const { click, cast, px, nL } = scene(slab, segs, camera);
  const [hx, hy] = px(new THREE.Vector3(0, 0, -0.05)).map(Math.round) as [number, number];
  for (const dy of [0, 2, 3]) {
    assert.equal(cast(hx, hy + dy, null)?.element, 0, `${dy} px from the hidden beam, the ray cast alone took it`);
    assert.equal(click(hx, hy + dy)?.element, nL, `${dy} px from the hidden beam: the slab drawn there`);
  }
  const [vx, vy] = px(new THREE.Vector3(0, 2, 0)).map(Math.round) as [number, number];
  assert.equal(click(vx, vy)?.element, 1, "on the beam drawn on the slab: the beam");
  assert.equal(click(vx, vy + CLICK_LINE_PX - 1)?.element, 1, "within CLICK_LINE_PX of it: the beam, as easy to hit as before");
  assert.equal(click(vx, vy + CLICK_LINE_PX + 2)?.element, nL, "beyond it: the slab under the cursor");
  const hit = click(vx, vy + 2)!;
  assert.equal(hit.prim.kind, "line");
  assert.ok(Math.abs(hit.point.y - 2) < 0.1 && Math.abs(hit.point.z) < 0.01, "the point is by the beam (on the ray, where it passes nearest the beam)");
});
