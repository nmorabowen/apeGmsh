// VF-a (#2939; #2924 item 1): the contour's colormap as a 1-D texture,
// sampled per fragment. A vertex colour attribute interpolates RGB inside a
// primitive, so a one-element quad under jet showed a purple diagonal: the
// average of jet's dark blue and dark red, a colour jet never has (#2908 D7).
// The oracle is state/contour.ts `sampleCmap` (matplotlib's segment data),
// never the texture itself:
//
// - the texels are the ramp, linearised, and one no-data texel;
// - a vertex coordinate is its value's ramp position, at a texel centre;
// - `primitiveRamp` keeps `primitiveColours`' no-data rule, vertex for vertex;
// - inside a one-element quad, a fragment shows the ramp at the interpolated
//   value, where interpolated RGB is measurably off the ramp.

import assert from "node:assert/strict";
import { test } from "node:test";
import * as THREE from "three";
import { colormapTexture, NO_DATA_U, noDataRamp, primitiveRamp, RAMP_TEXELS, rampTexels, rampU, rampUv, shownAt, TEXTURE_WIDTH, toLinear } from "../../src/renderer/colormapTexture.ts";
import { CMAP_NAMES, NO_DATA_RGB, primitiveColours, rampT, sampleCmap } from "../../src/state/contour.ts";
import type { ContourCmap } from "../../src/state/types.ts";

const lin = (c: readonly number[]): number[] => c.map(toLinear);
const dist = (a: readonly number[], b: readonly number[]): number => Math.max(...a.map((x, k) => Math.abs(x - b[k]!)));

/** `got` is the ramp's linear colour at `t` to one texel (the nearest sample, or a neighbour). */
function onRampAt(got: readonly number[], cmap: ContourCmap, t: number): number {
  const i = Math.round(Math.min(1, Math.max(0, t)) * (RAMP_TEXELS - 1));
  return Math.min(...[-1, 0, 1].map((k) => dist(got, lin(sampleCmap(cmap, Math.min(1, Math.max(0, (i + k) / (RAMP_TEXELS - 1))))))));
}

/** How far a linear colour is from every colour of the ramp (sampled finely). */
function offRamp(c: readonly number[], cmap: ContourCmap): number {
  let best = Infinity;
  for (let i = 0; i <= 4096; i++) best = Math.min(best, Math.hypot(...lin(sampleCmap(cmap, i / 4096)).map((x, k) => x - c[k]!)));
  return best;
}

test("the texture is the ramp, linearised, sample for sample, then one no-data texel; NEAREST, no mipmaps, linear colour space", () => {
  for (const cmap of CMAP_NAMES) {
    const data = rampTexels(cmap);
    assert.equal(data.length, 4 * TEXTURE_WIDTH);
    for (const i of [0, 1, 511, 1023, 1500, RAMP_TEXELS - 1]) {
      assert.ok(dist(Array.from(data.subarray(4 * i, 4 * i + 3)), lin(sampleCmap(cmap, i / (RAMP_TEXELS - 1)))) < 1e-6, `${cmap} texel ${i}`);
      assert.equal(data[4 * i + 3], 1);
    }
    assert.ok(dist(Array.from(data.subarray(4 * RAMP_TEXELS, 4 * RAMP_TEXELS + 3)), lin(NO_DATA_RGB)) < 1e-6, `${cmap} no-data texel`);
    const t = colormapTexture(cmap);
    assert.equal(t, colormapTexture(cmap), "one texture per colormap");
    assert.equal(t.image.width, TEXTURE_WIDTH);
    assert.equal(t.image.height, 1);
    assert.equal(t.magFilter, THREE.NearestFilter);
    assert.equal(t.minFilter, THREE.NearestFilter);
    assert.equal(t.generateMipmaps, false);
    assert.equal(t.colorSpace, THREE.NoColorSpace, "the texels are linear; three.js must not decode them again");
    assert.equal(t.type, THREE.FloatType);
  }
});

test("a coordinate sits on a texel centre: the ends of the ramp on its end texels, no data on the last, clamped outside [0, 1]", () => {
  const centre = (i: number) => (i + 0.5) / TEXTURE_WIDTH;
  assert.equal(rampU(0), centre(0));
  assert.equal(rampU(1), centre(RAMP_TEXELS - 1));
  assert.equal(rampU(-3), rampU(0));
  assert.equal(rampU(7), rampU(1));
  assert.equal(NO_DATA_U, centre(RAMP_TEXELS));
  const jet = colormapTexture("jet");
  for (const t of [0, 0.125, 0.35, 0.5, 0.66, 0.89, 1, Math.PI / 4]) assert.ok(onRampAt(shownAt(jet, rampU(t)), "jet", t) < 1e-6, `jet at ${t}`);
  assert.ok(dist(shownAt(jet, NO_DATA_U), lin(NO_DATA_RGB)) < 1e-6);
  // No ramp coordinate reaches the no-data texel, and none the ramp's from the other side.
  assert.ok(Math.floor(rampU(1) * TEXTURE_WIDTH) === RAMP_TEXELS - 1);
  assert.deepEqual(Array.from(noDataRamp(4)), Array(4).fill(Math.fround(NO_DATA_U)));
  assert.deepEqual(Array.from(rampUv(Float32Array.of(0.25, 0.75))), [0.25, 0.5, 0.75, 0.5]);
});

test("primitiveRamp keeps primitiveColours' rule: the same no-data primitives, and the same colour at every vertex", () => {
  // Values with NaNs, over a drawn subset in another order; then a range with no finite end.
  const perPrim = 3;
  const n = 40;
  const values = Float64Array.from({ length: n * perPrim }, (_, i) => (i % 17 === 5 ? NaN : Math.sin(i * 0.7) * 3 + 1));
  const map = Int32Array.from({ length: 25 }, (_, d) => (d * 7) % n);
  for (const clim of [{ min: -1.5, max: 3.2 }, { min: 2, max: 2 }, { min: null, max: 4 }]) {
    for (const cmap of ["jet", "viridis", "RdBu_r"] as const) {
      const rgb = primitiveColours(values, map, perPrim, clim, cmap);
      const ramp = primitiveRamp(values, map, perPrim, clim);
      assert.equal(ramp.noData, rgb.noData, `${JSON.stringify(clim)} ${cmap}: no-data count`);
      const tex = colormapTexture(cmap);
      for (let v = 0; v < map.length * perPrim; v++) {
        const got = shownAt(tex, ramp.u[v]!);
        const want = lin(Array.from(rgb.rgb.subarray(3 * v, 3 * v + 3)));
        if (ramp.u[v] === Math.fround(NO_DATA_U)) assert.ok(dist(got, want) < 1e-6 && dist(want, lin(NO_DATA_RGB)) < 1e-6, `vertex ${v}: no data both ways`);
        else {
          const x = values[map[Math.floor(v / perPrim)]! * perPrim + (v % perPrim)]!;
          assert.ok(onRampAt(got, cmap, rampT(x, clim.min!, clim.max!)) < 1e-6, `vertex ${v}: the ramp at ${x}`);
          assert.ok(dist(got, want) < 3e-3, `vertex ${v}: what the vertex colour was, to half a texel`);
        }
      }
    }
  }
});

test("a one-element quad under jet: every fragment is the ramp at the interpolated value; the interpolated RGB was the purple diagonal", () => {
  // The quad A B C D in two triangles (A B C) and (A C D); a bending field: 0 at A and C, 1 at B and D.
  // Along the diagonal A-C both triangles hold 0, and B, D hold 1: the classic one-element contour.
  const values = Float64Array.of(0, 1, 0, 0, 0, 1);
  const map = Int32Array.of(0, 1);
  const clim = { min: 0, max: 1 };
  const { u } = primitiveRamp(values, map, 3, clim);
  const { rgb } = primitiveColours(values, map, 3, clim, "jet");
  const jet = colormapTexture("jet");
  let worstRgb = 0;
  for (let tri = 0; tri < 2; tri++) {
    for (let i = 0; i <= 10; i++) {
      for (let j = 0; j <= 10 - i; j++) {
        const w = [i / 10, j / 10, (10 - i - j) / 10];
        const at = w.reduce((a, wk, k) => a + wk * u[3 * tri + k]!, 0);
        const value = w.reduce((a, wk, k) => a + wk * values[3 * tri + k]!, 0);
        // Per fragment: the ramp at the interpolated value.
        assert.ok(onRampAt(shownAt(jet, at), "jet", rampT(value, 0, 1)) < 1e-6, `triangle ${tri} at ${w}: jet(${value})`);
        // Per vertex (the old path): the blend of the corner colours.
        const blend = [0, 1, 2].map((c) => w.reduce((a, wk, k) => a + wk * toLinear(rgb[3 * (3 * tri + k) + c]!), 0));
        worstRgb = Math.max(worstRgb, offRamp(blend, "jet"));
      }
    }
  }
  // The centroid of (A B C) holds 1/3: jet's cyan, where the RGB blend of two dark blues and a dark red is purple.
  const centroid = shownAt(jet, (u[0]! + u[1]! + u[2]!) / 3);
  assert.ok(onRampAt(centroid, "jet", 1 / 3) < 1e-6);
  assert.ok(centroid[2]! > 0.9 && centroid[1]! > 0.5 && centroid[0]! < 0.01, `jet(1/3) is cyan: ${centroid}`);
  assert.ok(worstRgb > 0.1, `the interpolated RGB leaves the ramp (${worstRgb}): the test tells the two paths apart`);
  const blend = [0, 1, 2].map((c) => [0, 1, 2].reduce((a, k) => a + toLinear(rgb[3 * k + c]!), 0) / 3);
  assert.ok(Math.hypot(...blend.map((x, k) => x - centroid[k]!)) > 0.5, `the RGB blend at the centroid (${blend}) is nowhere near jet(1/3)`);
  assert.ok(blend[0]! > 0.05 && blend[2]! > 0.05 && blend[1]! === 0, `and it is purple: red and blue, no green (${blend})`);
});
