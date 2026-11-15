// renderer/particles.ts, the MPM points layer (ADR 0134 D5; chain M, M2-S4 #2983).
//
// Oracles: the contour ramp itself (state/contour.ts `sampleCmap`, linearised as the
// contour texture is), the no-value colour `NO_DATA_RGB`, a 2-D box's four edges, and
// three.js's own dispose events for the reload seam (no leaked buffers).

import assert from "node:assert/strict";
import { test } from "node:test";
import * as THREE from "three";
import { toLinear } from "../src/renderer/colormapTexture.ts";
import { componentOf, fillColours, fillPositions, gridBoxSegments, particleBounds, ParticleLayer, particleRange, type ParticleFrame } from "../src/renderer/particles.ts";
import { particleFitSphere, railKeptPx } from "../src/renderer/viewport.ts";
import { NO_DATA_RGB, sampleCmap } from "../src/state/contour.ts";

const LOOK = { plain: 0xe69f00, rim: 0x121418, grid: 0x8b919c };
const lin = (c: readonly number[]) => c.map((x) => toLinear(x));
const close = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  assert.equal(a.length, b.length);
  for (let i = 0; i < a.length; i++) assert.ok(Math.abs(a[i]! - b[i]!) < 1e-6, `${i}: ${a[i]} vs ${b[i]}`);
};

const frame = (over: Partial<ParticleFrame> = {}): ParticleFrame => ({
  stageKey: "results|x.ladruno|1|stage_0",
  ndm: 2,
  positions: new Float64Array([0, 0, 1, 0, 1, 1]),
  scalar: new Float64Array([0, 0.5, NaN]),
  range: { min: 0, max: 1 },
  cmap: "jet",
  grid: null,
  fit: false,
  ...over,
});

test("colours: the ramp's ends at the range's ends, the no-value colour for NaN, one plain colour with no field", () => {
  const out = new Float32Array(9);
  const none = fillColours(3, new Float64Array([0, 1, NaN]), { min: 0, max: 1 }, "jet", [0, 0, 0], out);
  assert.equal(none, 1);
  close(out.subarray(0, 3), lin(sampleCmap("jet", 0)));
  close(out.subarray(3, 6), lin(sampleCmap("jet", 1)));
  close(out.subarray(6, 9), lin(NO_DATA_RGB));
  // Every value NaN: the range has no finite end, and every particle is no value.
  const all = new Float64Array([NaN, NaN, NaN]);
  assert.deepEqual(particleRange(all), { range: null, noValue: 3 });
  assert.equal(fillColours(3, all, null, "jet", [0, 0, 0], out), 3);
  for (let i = 0; i < 3; i++) close(out.subarray(3 * i, 3 * i + 3), lin(NO_DATA_RGB));
  assert.equal(fillColours(3, null, null, "jet", [0.25, 0.5, 0.75], out), 0);
  close(out, [0.25, 0.5, 0.75, 0.25, 0.5, 0.75, 0.25, 0.5, 0.75]);
  assert.throws(() => fillColours(3, new Float64Array(2), null, "jet", [0, 0, 0], out), /2 values for 3 particles/);
});

test("positions: 2-D rows lie at z = 0; a component of row-major values; bounds", () => {
  const out = new Float32Array(9);
  assert.equal(fillPositions(new Float64Array([1, 2, 3, 4, 5, 6]), 2, out), 3);
  assert.deepEqual(Array.from(out), [1, 2, 0, 3, 4, 0, 5, 6, 0]);
  assert.throws(() => fillPositions(new Float64Array(5), 2, out), /not whole 2-D rows/);
  assert.deepEqual(Array.from(componentOf(new Float64Array([1, 2, 3, 4, 5, 6]), 3, 1)), [2, 5]);
  assert.throws(() => componentOf(new Float64Array(6), 3, 3), /component 3 of 3/);
  const b = particleBounds(new Float64Array([0, 0, 2, 0]), 2)!;
  assert.deepEqual(b.center, [1, 0, 0]);
  assert.equal(b.radius, 1);
  assert.equal(particleBounds(new Float64Array(0), 3), null);
});

test("the grid's wire box: 4 edges of the rectangle in 2-D, 12 of the box in 3-D, from origin + cellSize * numCells", () => {
  const g2 = gridBoxSegments({ origin: [0, 0], cellSize: [0.5, 0.5], numCells: [4, 4] }, 2);
  assert.equal(g2.length, 4 * 6);
  const pts = new Set<string>();
  for (let i = 0; i < g2.length; i += 3) pts.add(`${g2[i]},${g2[i + 1]},${g2[i + 2]}`);
  assert.deepEqual([...pts].sort(), ["0,0,0", "0,2,0", "2,0,0", "2,2,0"]);
  assert.equal(gridBoxSegments({ origin: [1, 1, 1], cellSize: [1, 1, 1], numCells: [1, 2, 3] }, 3).length, 12 * 6);
});

test("one Points per stage: an output reuses its buffers; another stage disposes the old geometry and material first; null disposes all", () => {
  const layer = new ParticleLayer();
  layer.show(frame(), LOOK);
  const first = layer.drawn!;
  assert.equal(layer.root.children.length, 1);
  assert.equal(layer.count, 3);
  const disposed: string[] = [];
  first.geometry.addEventListener("dispose", () => disposed.push("geometry"));
  (first.material as THREE.Material).addEventListener("dispose", () => disposed.push("material"));
  // The next output of the same stage, with fewer particles: the same Points, a shorter draw range.
  layer.show(frame({ positions: new Float64Array([5, 5, 6, 6]), scalar: new Float64Array([1, 0]) }), LOOK);
  assert.equal(layer.drawn, first);
  assert.equal(layer.count, 2);
  close((first.geometry.getAttribute("position") as THREE.BufferAttribute).array.slice(0, 6), [5, 5, 0, 6, 6, 0]);
  assert.deepEqual(disposed, []);
  // An output with no particle draws none.
  layer.show(frame({ positions: new Float64Array(0), scalar: new Float64Array(0), range: null }), LOOK);
  assert.equal(layer.count, 0);
  // A re-read (a new stage key): the old buffers go before the new are made.
  layer.show(frame({ stageKey: "results|x.ladruno|2|stage_0" }), LOOK);
  assert.deepEqual(disposed, ["geometry", "material"]);
  assert.notEqual(layer.drawn, first);
  assert.equal(layer.root.children.length, 1);
  // The grid box, then nothing at all.
  layer.show(frame({ stageKey: "results|x.ladruno|2|stage_0", grid: { origin: [0, 0], cellSize: [1, 1], numCells: [1, 1] } }), LOOK);
  assert.equal(layer.root.children.length, 2);
  const left = [layer.drawn!, ...layer.root.children.filter((c) => c instanceof THREE.LineSegments)] as (THREE.Points | THREE.LineSegments)[];
  const gone: string[] = [];
  for (const o of left) o.geometry.addEventListener("dispose", () => gone.push(o.name));
  layer.show(null, LOOK);
  assert.deepEqual(gone.sort(), ["particle grid", "particles"]);
  assert.equal(layer.root.children.length, 0);
  assert.equal(layer.drawn, null);
});

test("#2992 review: the rim follows the theme; the same stage re-shown with another look recolours its rim (and its plain colour)", () => {
  const layer = new ParticleLayer();
  layer.show(frame(), LOOK);
  const pts = layer.drawn!;
  const rim = () => ((pts.material as THREE.PointsMaterial).userData["nodeUniforms"] as { nodeRim: { value: THREE.Color } }).nodeRim.value.getHex();
  assert.equal(rim(), new THREE.Color(LOOK.rim).getHex());
  const light = { plain: 0x0072b2, rim: 0xf4f6f8, grid: 0x454d59 };
  layer.show(frame({ scalar: null, range: null }), light);
  assert.equal(layer.drawn, pts, "the same stage: the same Points");
  assert.equal(rim(), new THREE.Color(light.rim).getHex());
  const plain = new THREE.Color(light.plain);
  close((pts.geometry.getAttribute("color") as THREE.BufferAttribute).array.slice(0, 3), [plain.r, plain.g, plain.b]);
  layer.dispose();
});

test("the colour attribute carries the ramp, and the material colours by vertex", () => {
  const layer = new ParticleLayer();
  layer.show(frame(), LOOK);
  const pts = layer.drawn!;
  assert.equal((pts.material as THREE.PointsMaterial).vertexColors, true);
  const col = (pts.geometry.getAttribute("color") as THREE.BufferAttribute).array as Float32Array;
  close(col.subarray(6, 9), lin(NO_DATA_RGB));
  close(col.subarray(3, 6), lin(sampleCmap("jet", 0.5)));
  layer.dispose();
});

// -- owner D7 on #2992: the cloud centred, the camera fit to its own bounds -------------

test("particles drawn alone keep no width for the right rail, so the cloud is framed in the middle of the window", () => {
  // The rail's 470 + 28 px is kept beside a model in a wide window (as before), and never when collapsed or narrow.
  assert.equal(railKeptPx(1280, false, false), 498);
  assert.equal(railKeptPx(1280, true, false), 0);
  assert.equal(railKeptPx(900, false, false), 0);
  // Particles with no mesh and no geometry: the rail shows nothing, so nothing is kept.
  assert.equal(railKeptPx(1280, false, true), 0);
});

test("the camera fits the cloud's own bounds, not the grid's box, whether the box is drawn or not", () => {
  // A cloud spanning 0.625 .. 1.375 in x and y inside a 0 .. 2 grid: centre (1, 1, 0), radius half the
  // cloud's diagonal times the 1.3 margin. The box's own sphere would be half of 2·√2 times 1.3.
  const xy = new Float64Array([0.625, 0.625, 1.375, 0.625, 1, 1, 0.625, 1.375, 1.375, 1.375]);
  const grid = { origin: [0, 0], cellSize: [0.5, 0.5], numCells: [4, 4] };
  const radius = (Math.hypot(0.75, 0.75) / 2) * 1.3;
  for (const g of [null, grid]) {
    const s = particleFitSphere(frame({ positions: xy, scalar: null, range: null, grid: g }))!;
    close(s.center, [1, 1, 0]);
    assert.ok(Math.abs(s.radius - radius) < 1e-12, `grid ${g ? "on" : "off"}: radius ${s.radius}, want ${radius}, the box's ${(Math.hypot(2, 2) / 2) * 1.3}`);
  }
  assert.equal(particleFitSphere(frame({ positions: new Float64Array(0), scalar: null, range: null })), null);
});
