// The node markers: which vertices stand for which node, where they sit, the
// chosen size, and the state's rules for keeping or choosing it again.

import assert from "node:assert/strict";
import { test } from "node:test";
import { loadsInputsChanged } from "../src/panels/loads.ts";
import { nodePositions, uniqueNodeVertices } from "../src/renderer/nodes.ts";
import { smartNodeSize } from "../src/state/nodes.ts";
import { Store } from "../src/state/store.ts";

test("uniqueNodeVertices: one vertex per node across lines and triangles, drawn primitives only", () => {
  // Segments 1-2, 2-3 (undrawn: 3-9); triangles 2-3-4, 4-5-6 (undrawn).
  const line = Float64Array.from([1, 2, 2, 3, 3, 9]);
  const tri = Float64Array.from([2, 3, 4, 4, 5, 6]);
  const v = uniqueNodeVertices(line, tri, Int32Array.from([0, 1]), Int32Array.from([0]));
  assert.deepEqual([...v.line], [0, 3, 9], "nodes 1, 2, 3 from the first vertex that has them");
  assert.deepEqual([...v.tri], [6], "node 4; nodes 2 and 3 are already placed; node 9, 5 and 6 are not drawn");
});

test("nodePositions: reads the drawn positions, so a deformed set moves the markers", () => {
  const v = { line: Int32Array.from([3]), tri: Int32Array.from([6]) };
  const line = Float32Array.from([0, 0, 0, 1, 2, 3]);
  const tri = Float32Array.from([0, 0, 0, 0, 0, 0, 7, 8, 9]);
  assert.deepEqual([...nodePositions(v, line, tri)], [1, 2, 3, 7, 8, 9]);
  const out = new Float32Array(6);
  nodePositions(v, Float32Array.from([0, 0, 0, 4, 5, 6]), tri, out);
  assert.deepEqual([...out], [4, 5, 6, 7, 8, 9], "written in place");
});

test("the material injects the depth pull after project_vertex, and the pull branches on the projection", async () => {
  const THREE = await import("three");
  const { makeNodePoints, NODE_DEPTH_GLSL, NODE_DEPTH_FLOOR, NODE_DEPTH_ORTHO, NODE_DEPTH_PULL } = await import("../src/renderer/nodes.ts");
  const m = makeNodePoints(new Float32Array(3), 6, 0xffffff, 0).material as InstanceType<typeof THREE.PointsMaterial>;
  const shader = { uniforms: {}, vertexShader: "void main() {\n#include <project_vertex>\n}", fragmentShader: "void main() {\n#include <color_fragment>\n}" };
  m.onBeforeCompile(shader as never, undefined as never);
  assert.ok(shader.vertexShader.includes(`#include <project_vertex>${NODE_DEPTH_GLSL}`));
  for (const part of ["isPerspectiveMatrix(projectionMatrix)", `${NODE_DEPTH_PULL}`, `${NODE_DEPTH_FLOOR}`, `${NODE_DEPTH_ORTHO}`]) {
    assert.ok(NODE_DEPTH_GLSL.includes(part), `the shader's pull has no ${part}`);
  }
});

test("perspective depth pull is a share of eye distance: a marker behind a wall stays behind it, even with near at 1e-3", async () => {
  const THREE = await import("three");
  const { nodeClipZ } = await import("../src/renderer/nodes.ts");
  const cam = new THREE.PerspectiveCamera(45, 1, 1e-3, 100);
  cam.updateProjectionMatrix();
  const p = cam.projectionMatrix.elements;
  const faceZ = (d: number) => {
    const v = new THREE.Vector4(0, 0, -d, 1).applyMatrix4(cam.projectionMatrix);
    return v.z / v.w;
  };
  for (const [wall, behind] of [[2, 0.3], [5, 3], [0.5, 0.01], [40, 1]] as const) {
    assert.ok(nodeClipZ(p, [0, 0, -(wall + behind)]) > faceZ(wall), `marker ${behind} m behind a wall at ${wall} m shows through`);
  }
  for (const d of [0.01, 2, 99]) {
    // Far off, the share alone is under a depth step; the floor keeps the marker in front of its face.
    assert.ok(faceZ(d) - nodeClipZ(p, [0, 0, -d]) >= 2.4e-7, `a marker on a face ${d} m away is not drawn in front of it`);
  }
});

test("orthographic depth pull is fixed: a marker wins against its face even behind the eye, with a negative near", async () => {
  const THREE = await import("three");
  const { nodeClipZ } = await import("../src/renderer/nodes.ts");
  const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, -10, 10);
  cam.updateProjectionMatrix();
  const p = cam.projectionMatrix.elements;
  const faceZ = (z: number) => new THREE.Vector4(0, 0, z, 1).applyMatrix4(cam.projectionMatrix).z;
  for (const z of [-9, -2, -0.01, 0.5, 2, 9]) {
    assert.ok(nodeClipZ(p, [0, 0, z]) < faceZ(z), `a marker on a face at view z ${z} loses to it`);
    assert.ok(nodeClipZ(p, [0, 0, z - 0.01]) > faceZ(z), `a marker 0.01 behind a face at view z ${z} shows through`);
  }
});

test("mountLoads: a marker size or a hover never reads the phase; a new phase does", async () => {
  const { mountLoads } = await import("../src/panels/loads.ts");
  const real = new Store().get();
  let reads = 0;
  const counted = (s: typeof real): typeof real => {
    const at = s.phase.at;
    const phase = { ...s.phase };
    Object.defineProperty(phase, "at", { get: () => (reads++, at), enumerable: true });
    return { ...s, phase };
  };
  const s0 = counted(real);
  let sub: ((s: typeof real, prev: typeof real) => void) | null = null;
  const fake = { get: () => s0, subscribe: (fn: typeof sub) => ((sub = fn), () => {}), dispatch: () => {} };
  const panel = { hidden: false, replaceChildren: () => {} };
  const prevDoc = (globalThis as { document?: unknown }).document;
  (globalThis as { document?: unknown }).document = { getElementById: () => panel };
  try {
    const unmount = mountLoads(fake as never);
    reads = 0;
    sub!({ ...s0, nodes: { ...s0.nodes, size: 9 } }, s0);
    sub!({ ...s0, hover: "/x" }, s0);
    assert.equal(reads, 0, "a marker size or a hover read the phase");
    sub!(counted(real), s0);
    assert.ok(reads > 0, "a new phase was not read");
    unmount();
  } finally {
    (globalThis as { document?: unknown }).document = prevDoc;
  }
});

test("loadsInputsChanged: a marker size or a hover skips the loads summary; a phase, the loads or the units do not", () => {
  const store = new Store();
  const s0 = store.get();
  store.dispatch({ type: "setNodeSize", size: 6 });
  assert.equal(loadsInputsChanged(store.get(), s0), false);
  assert.equal(loadsInputsChanged({ ...s0, hover: "/x" }, s0), false);
  assert.equal(loadsInputsChanged({ ...s0, phase: { ...s0.phase } }, s0), true);
  assert.equal(loadsInputsChanged({ ...s0, loads: { ...s0.loads } }, s0), true);
  assert.equal(loadsInputsChanged({ ...s0, contour: { ...s0.contour } }, s0), true);
  assert.equal(loadsInputsChanged({ ...s0, ndm: { ...s0.ndm } }, s0), true);
});

const segments = (n: number, length: number) => {
  const s = new Float32Array(n * 6);
  for (let i = 0; i < n; i++) {
    s[i * 6 + 3] = length;
  }
  return s;
};

test("smartNodeSize: a coarse model gets a large marker, a dense one a small marker, within bounds", () => {
  assert.equal(smartNodeSize(segments(24, 3000), 7000, 900), 10, "a frame: long elements");
  assert.equal(smartNodeSize(segments(20000, 200), 30000, 900), 3, "a dense slab: clamps at the floor");
  assert.equal(smartNodeSize(segments(100, 1000), 10000, 900), 9, "45 px spacing");
});

test("smartNodeSize: nothing to measure gives the largest size, never NaN", () => {
  assert.equal(smartNodeSize(new Float32Array(0), 10, 900), 10);
  assert.equal(smartNodeSize(segments(4, 0), 10, 900), 10);
  assert.equal(smartNodeSize(segments(4, 1), 0, 900), 10);
});

test("setNodes and setNodeSize: an automatic size is chosen again on, the user's own stays", () => {
  const store = new Store();
  const nodes = () => store.get().nodes;
  assert.equal(nodes().on, false, "off by default");
  store.dispatch({ type: "setNodes", on: true });
  assert.equal(nodes().size, null, "asks for a choice");
  store.dispatch({ type: "setNodeSize", size: 7, auto: true });
  assert.deepEqual([nodes().size, nodes().auto], [7, true]);
  store.dispatch({ type: "setNodes", on: false });
  store.dispatch({ type: "setNodes", on: true });
  assert.equal(nodes().size, null, "on again: chosen again");
  store.dispatch({ type: "setNodeSize", size: 12 });
  assert.deepEqual([nodes().size, nodes().auto], [12, false]);
  store.dispatch({ type: "setNodes", on: false });
  store.dispatch({ type: "setNodes", on: true });
  assert.equal(nodes().size, 12, "the user's size stays");
  store.dispatch({ type: "setNodeSize", size: 999 });
  assert.equal(nodes().size, 24, "clamped");
  store.dispatch({ type: "setNodeSize", size: null });
  assert.deepEqual([nodes().size, nodes().auto], [null, true], "auto asks again");
});
