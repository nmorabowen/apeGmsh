// V4e1 (#1834, review of #1900): the two viewport call sites the section
// planes rely on, pinned through the real `Viewport.prototype` methods (a
// WebGL renderer cannot run in Node, so each runs on a minimal receiver that
// holds only what the method reads):
//
// - `renderNow` puts the planes on every material before it renders, so an
//   object made after the plane is cut on its first frame;
// - `pickAt` passes the clip check to `hitElement`, so a click on cut-away
//   geometry is a miss, and the same click with the cuts off is a hit.

import assert from "node:assert/strict";
import { test } from "node:test";
import * as THREE from "three";
import { SectionClip } from "../../src/renderer/clip.ts";
import { Viewport } from "../../src/renderer/viewport.ts";
import { BlobStore } from "../../src/state/blobs.ts";
import { initialState, reduce } from "../../src/state/reduce.ts";
import type { MeshInfo, State } from "../../src/state/types.ts";

const proto = Viewport.prototype as unknown as {
  renderNow(this: unknown): void;
  pickAt(this: unknown, x: number, y: number): { decl: string } | null;
};

const withPlane = (normal: number[], offset: number): State => {
  let s = reduce({ ...initialState, mesh: { center: [0, 0, 0] } as unknown as MeshInfo }, { type: "addClipPlane", normal });
  s = reduce(s, { type: "setClipOffset", id: "plane-1", offset });
  return s;
};

test("renderNow: a material created after a plane exists carries the planes when the scene is rendered", () => {
  const clip = new SectionClip();
  clip.sync(withPlane([0, 0, 1], 1).clip);
  const scene = new THREE.Scene();
  const late = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshStandardMaterial());
  scene.add(late);
  let seen: THREE.Plane[] | null | undefined;
  const receiver = {
    clip,
    scene,
    camera: new THREE.PerspectiveCamera(),
    renderer: { render: () => (seen = (late.material as THREE.Material).clippingPlanes) },
    highlight: new THREE.Group(),
    halo: { pulsing: false, render: () => {} },
    lastRenderMs: 0,
    requestRender: () => {},
  };
  proto.renderNow.call(receiver);
  assert.equal(seen, clip.planes, "the render saw the shared planes on the new material");
  assert.equal(seen!.length, 1);
});

test("pickAt: a click on cut-away geometry is a miss; the same click with the cuts off picks it", () => {
  // A 2 x 2 face at z = 2 seen from above; one element.
  const blobs = new BlobStore();
  const mesh = {
    center: [1, 1, 1],
    radius: 2,
    elements: ["mesh/element/1"],
    lineElement: blobs.put("lineElement", new Int32Array(0)),
    triElement: blobs.put("triElement", Int32Array.from([0, 0])),
  } as unknown as MeshInfo;
  const top = new Float32Array([0, 0, 2, 2, 0, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 0, 2, 2]);
  const faces = new THREE.Mesh(new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(top, 3)), new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }));
  faces.updateMatrixWorld();
  const camera = new THREE.OrthographicCamera(-4, 4, 4, -4, 0.1, 100);
  camera.position.set(1, 1, 10);
  camera.lookAt(1, 1, 0);
  camera.updateMatrixWorld();
  // A plane keeping z <= 1 (normal -Z, offset -1): the face is cut away.
  let state: State = { ...withPlane([0, 0, -1], -1), mesh, decls: Object.create(null) };
  const clip = new SectionClip();
  const receiver = {
    store: { get: () => state },
    blobs,
    drawn: { lineMap: new Int32Array(0), triMap: Int32Array.from([0, 1]), edgeMask: null },
    lines: null,
    faces,
    clip,
    camera,
    raycaster: new THREE.Raycaster(),
    renderer: { domElement: { getBoundingClientRect: () => ({ left: 0, top: 0, width: 800, height: 800 }) } },
    aim: (Viewport.prototype as unknown as { aim: unknown }).aim,
    pickOf: (Viewport.prototype as unknown as { pickOf: unknown }).pickOf,
  };
  // Client point over (0.5, 0.5, 2).
  const p = new THREE.Vector3(0.5, 0.5, 2).project(camera);
  const x = ((p.x + 1) / 2) * 800, y = ((1 - p.y) / 2) * 800;
  clip.sync(state.clip);
  assert.equal(proto.pickAt.call(receiver, x, y), null, "the face under the click is cut away: a miss");
  state = { ...state, clip: { ...state.clip, applyCuts: false } };
  clip.sync(state.clip);
  assert.equal(proto.pickAt.call(receiver, x, y)?.decl, "mesh/element/1", "cuts off: the same click picks the face");
});
