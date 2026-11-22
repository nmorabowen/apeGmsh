// Traced crack lines on layered shells (state/crackLines.ts), on hand-made slabs where the answer is known, then end
// to end on the fork-run wall (fixtures/ladruno_shells3d.ladruno) through the Cracks layer.

import assert from "node:assert/strict";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import { CracksLayer } from "../src/renderer/cracks.ts";
import { buildCrackLines, buildRibbons, invertQuad } from "../src/state/crackLines.ts";
import { crackHalfSize } from "../src/state/cracks.ts";
import { shellPitchOf } from "../src/state/shellCrackPoints.ts";
import { FACE_RGB, type ShellTickGroup } from "../src/state/shellCracks.ts";
import { opened, until } from "./contour/harness.ts";

test("shellPitchOf: half the shells' size, where the points' bounding box overstates it for a building", () => {
  // A 10 x 10 x 10 m box of 1 m wall panels: four walls, 400 elements of 1 m2, 2 x 2 Gauss points each.
  const corners = new Map<number, Float64Array>();
  const gauss: number[] = [];
  const g = 0.5 - 0.5 / Math.sqrt(3);
  let id = 0;
  for (const [ax, at] of [[0, 0], [0, 10], [1, 0], [1, 10]] as const)
    for (let a = 0; a < 10; a++)
      for (let z = 0; z < 10; z++) {
        const p = (u: number, w: number) => (ax === 0 ? [at, a + u, z + w] : [a + u, at, z + w]);
        corners.set(++id, Float64Array.from([...p(0, 0), ...p(1, 0), ...p(1, 1), ...p(0, 1)]));
        for (const [u, w] of [[g, g], [1 - g, g], [1 - g, 1 - g], [g, 1 - g]]) gauss.push(...p(u!, w!));
      }
  const pitch = shellPitchOf([Float64Array.from(corners.keys())], (e) => corners.get(e) ?? null)!;
  assert.ok(Math.abs(pitch - 0.5) < 1e-12, `${pitch}`);
  // The bounding-box estimate the lines used before: the points fill 4 planes of the box, not its volume.
  const old = crackHalfSize(Float32Array.from(gauss)) / 0.4;
  assert.ok(old > 1.5 * pitch, `the box estimate ${old} against ${pitch}`);
  assert.equal(shellPitchOf([Float64Array.of(1)], () => null), null);
});

// -- a hand-made slab ------------------------------------------------------------------------

const G = 1 / Math.sqrt(3);
const GP = [[-G, -G], [G, -G], [G, G], [-G, G]];

/**
 * An nx x ny slab of unit quads in z = 0 (identity frames: e1 = x, e2 = y, e3 = +z), 4 Gauss points each, two layers
 * read (the -e3 and +e3 outer layers), thickness `t`; `crack(x, y, layer)` gives C1 (local) at a Gauss point.
 */
function slab(nx: number, ny: number, crack: (x: number, y: number, layer: 0 | 1) => [number, number, number], t = 0.2) {
  const E = nx * ny;
  const base: number[] = [], group: number[] = [], element: number[] = [], gauss: number[] = [];
  const values = new Float64Array(E * 4 * 2 * 9);
  const corners = new Map<number, Float64Array>();
  for (let j = 0; j < ny; j++)
    for (let i = 0; i < nx; i++) {
      const e = j * nx + i;
      corners.set(e + 1, Float64Array.from([i, j, 0, i + 1, j, 0, i + 1, j + 1, 0, i, j + 1, 0]));
      GP.forEach(([a, b], g) => {
        const x = i + 0.5 * (1 + a!), y = j + 0.5 * (1 + b!);
        base.push(x, y, 0);
        group.push(0);
        element.push(e);
        gauss.push(g);
        for (const layer of [0, 1] as const) values.set(crack(x, y, layer), ((e * 4 + g) * 2 + layer) * 9);
      });
    }
  const g: ShellTickGroup = {
    values,
    layers: [0, 5],
    nLayers: 6,
    nGauss: 4,
    quaternions: Float64Array.from({ length: 4 * E }, (_, k) => (k % 4 === 0 ? 1 : 0)),
    recorded: new Uint8Array(E).fill(1),
    thickness: new Float64Array(E).fill(t),
  };
  return {
    input: {
      base: Float32Array.from(base),
      points: { group: Int32Array.from(group), element: Int32Array.from(element), gauss: Int32Array.from(gauss) },
      groups: [g],
      elementIds: [Float64Array.from({ length: E }, (_, e) => e + 1)],
      corners: (id: number) => corners.get(id) ?? null,
    },
    corners,
  };
}

/** A vertex's mid-surface position from its element's corners and natural coordinates. */
function place(corners: Map<number, Float64Array>, id: number, xi: number, eta: number): number[] {
  const c = corners.get(id)!;
  const N = [0.25 * (1 - xi) * (1 - eta), 0.25 * (1 + xi) * (1 - eta), 0.25 * (1 + xi) * (1 + eta), 0.25 * (1 - xi) * (1 + eta)];
  return [0, 1, 2].map((k) => N.reduce((a, n, j) => a + n * c[3 * j + k]!, 0));
}

test("invertQuad: the natural coordinates of a point, back through the bilinear map of a skewed quad", () => {
  const c = [0, 0, 2, 0.2, 2.3, 1.7, -0.2, 1.5];
  for (const [xi, eta] of [[0, 0], [0.7, -0.4], [-1, 1], [0.95, 0.95]]) {
    const N = [0.25 * (1 - xi!) * (1 - eta!), 0.25 * (1 + xi!) * (1 - eta!), 0.25 * (1 + xi!) * (1 + eta!), 0.25 * (1 - xi!) * (1 + eta!)];
    const x = N.reduce((a, n, k) => a + n * c[2 * k]!, 0), y = N.reduce((a, n, k) => a + n * c[2 * k + 1]!, 0);
    const r = invertQuad(c, x, y)!;
    assert.ok(Math.abs(r[0] - xi!) < 1e-10 && Math.abs(r[1] - eta!) < 1e-10, `${r} vs ${xi},${eta}`);
  }
});

test("a slab cracked with normals along x: lines along y on the +z face (red), each vertex inside its element", () => {
  const { input, corners } = slab(12, 12, (_x, _y, layer) => (layer === 1 ? [0.5, 0, 0] : [0, 0, 0]));
  const lines = buildCrackLines(input, { face: "both", minOpening: 0.05, pitch: 0.5, eps: 0.001 });
  assert.equal(lines.panels, 1);
  assert.ok(lines.face.length >= 4, `${lines.face.length} lines`);
  assert.ok(lines.face.every((f) => f === "plus"), "the -z layer is uncracked: no blue line");
  for (let v = 0; v < lines.w.length; v++) {
    const xi = lines.xi[2 * v]!, eta = lines.xi[2 * v + 1]!;
    assert.ok(Math.abs(xi) <= 1 && Math.abs(eta) <= 1);
    assert.ok(Math.abs(lines.lift[3 * v + 2]! - (0.1 + 0.001)) < 1e-6, "lifted to the +z face, t/2 + eps");
  }
  for (let l = 0; l + 1 < lines.lineStart.length; l++) {
    const a = lines.lineStart[l]!, b = lines.lineStart[l + 1]! - 1;
    const pa = place(corners, lines.elementId[a]!, lines.xi[2 * a]!, lines.xi[2 * a + 1]!);
    const pb = place(corners, lines.elementId[b]!, lines.xi[2 * b]!, lines.xi[2 * b + 1]!);
    assert.ok(Math.abs(pa[0]! - pb[0]!) < 1e-6, `a line along y keeps its x (${pa[0]} .. ${pb[0]})`);
    assert.ok(Math.abs(pb[1]! - pa[1]!) > 8, "and runs across the slab");
  }
});

test("faces: a crack on the -z layer only is blue, lifted to -z; face \"plus\" then draws nothing", () => {
  const { input } = slab(8, 8, (_x, _y, layer) => (layer === 0 ? [0, 0.4, 0] : [0, 0, 0]));
  const both = buildCrackLines(input, { face: "both", minOpening: 0.05, pitch: 0.5, eps: 0 });
  assert.ok(both.face.length > 0 && both.face.every((f) => f === "minus"));
  assert.ok(Array.from({ length: both.w.length }, (_, v) => both.lift[3 * v + 2]!).every((z) => Math.abs(z + 0.1) < 1e-6));
  assert.equal(buildCrackLines(input, { face: "plus", minOpening: 0.05, pitch: 0.5, eps: 0 }).face.length, 0);
});

test("ribbons: two triangles per segment, lying in the face, wider where the crack opens more, in the face's colour", () => {
  const { input, corners } = slab(10, 4, (x, _y, layer) => (layer === 1 ? [0, 0.05 + 2 * x, 0] : [0, 0, 0]));
  const lines = buildCrackLines(input, { face: "both", minOpening: 0.01, pitch: 0.5, eps: 0 });
  const at = new Float32Array(3 * lines.w.length);
  for (let v = 0; v < lines.w.length; v++) at.set(place(corners, lines.elementId[v]!, lines.xi[2 * v]!, lines.xi[2 * v + 1]!), 3 * v);
  const r = buildRibbons(lines, at, 0.01, 0.1, 0.01);
  assert.equal(r.positions.length, r.segments * 18);
  for (let k = 2; k < r.positions.length; k += 3) assert.ok(Math.abs(r.positions[k]! - 0.1) < 1e-6, "on the +z face");
  for (let k = 0; k < r.colours.length; k += 3) assert.deepEqual(Array.from(r.colours.subarray(k, k + 3)).map((c) => Math.round(c * 255)), FACE_RGB.plus.map((c) => Math.round(c * 255)));
  // Width across each segment (along y for lines along x): narrow at small x, wide at large x.
  const widths = Array.from({ length: r.segments }, (_, s) => ({ x: r.positions[18 * s]!, w: Math.abs(r.positions[18 * s + 1]! - r.positions[18 * s + 16]!) }));
  const lo = widths.filter((s) => s.x < 2).map((s) => s.w), hi = widths.filter((s) => s.x > 8).map((s) => s.w);
  assert.ok(lo.length && hi.length && Math.max(...lo) < Math.min(...hi), "wider where the opening is larger");
});

test("scale: a 24 000-point fully cracked slab (6 000 elements) traces both faces; a runaway guard, the time is logged", () => {
  const { input } = slab(100, 60, (x, y, layer) => (layer === 1 ? [Math.cos(0.02 * y), Math.sin(0.02 * y), 0] : [0.3 * Math.sin(0.1 * x), 0.3, 0]).map((v) => v * (1 + 0.01 * x)) as [number, number, number]);
  const t0 = performance.now();
  const lines = buildCrackLines(input, { face: "both", minOpening: 0.05, pitch: 0.5, eps: 0.001 });
  const ms = performance.now() - t0;
  assert.ok(lines.face.length > 50, `${lines.face.length} lines`);
  // A guard against an accidentally quadratic path, not a budget: ~1 s alone on the maintainer's machine (2026-11-20).
  assert.ok(ms < 20000, `traced in ${ms.toFixed(0)} ms`);
  console.log(`# crackLines: 24000 points, ${lines.face.length} lines, ${lines.w.length} vertices, ${ms.toFixed(0)} ms`);
});

// -- end to end on the fork-run wall ------------------------------------------------------------

const fixture = (name: string) => fileURLToPath(new URL(`../fixtures/${name}`, import.meta.url));
const h = await opened(fixture("ladruno_shells3d.h5"), fixture("ladruno_shells3d.ladruno"));
after(() => {
  h.effects.dispose();
  h.host.close();
});

test("the layer draws the wall's crack lines as ribbons on its faces: red at +y, blue at -y; ticks hide them", async () => {
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 5 } });
  h.store.dispatch({ type: "setCracks", on: true, minOpening: 0.01, style: "lines" });
  await until(() => !!h.effects.currentStep()?.layers, "the layer step");
  const d = h.effects.currentStep()!;
  const layer = new CracksLayer();
  try {
    assert.equal(layer.sync(h.store.get(), h.blobs, null, d), true);
    assert.ok(layer.drawnLines.lines > 0 && layer.drawnLines.segments > 0, JSON.stringify(layer.drawnLines));
    assert.equal(layer.drawnLines.panels, 1, "one wall, one panel");
    const mesh = layer.root.children[2] as unknown as { visible: boolean; geometry: { getAttribute(n: string): { array: Float32Array } } };
    assert.equal(mesh.visible, true);
    const pos = mesh.geometry.getAttribute("position").array, col = mesh.geometry.getAttribute("color").array;
    const red = FACE_RGB.plus.map((c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
    for (let v = 0; v < pos.length / 3; v++) {
      const isRed = Math.abs(col[3 * v]! - red[0]!) < 1e-6;
      assert.ok(isRed ? pos[3 * v + 1]! > 100 : pos[3 * v + 1]! < -100, `vertex ${v} y = ${pos[3 * v + 1]}: on its face, t/2 off the mid-surface`);
    }
    h.store.dispatch({ type: "setCracks", style: "ticks" });
    layer.sync(h.store.get(), h.blobs, null, d);
    assert.equal(mesh.visible, false);
    assert.ok(layer.drawnShell.ticks > 0);
  } finally {
    layer.dispose();
    h.store.dispatch({ type: "setCracks", on: false, minOpening: 0, style: "lines" });
  }
});
