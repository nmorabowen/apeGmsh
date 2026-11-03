// Principal stress and strain of a fork-written .ladruno (state/principal.ts) against Python.
//
// ladruno_quad2d.ladruno (see ladrunoGauss.test.ts): six plane-stress FourNodeQuad elements with 2 x 2
// stress and strain Gauss buckets. ladruno_quad2d.principal.oracle.json, written by
// tests/fixtures/ladruno/make_viewer_principal_oracle.py, is apeGmsh's `principal_frame` (the function
// Qt's PrincipalDirectionDiagram draws from) over Python's own read of the same buckets: per Gauss point
// the values p1 >= p2 >= p3 and their sign-canonical unit vectors. The app reads the step through main
// (`tensor`), joins it to the Gauss points the markers place, and must give the same frame.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { ARROW, ARROW_STRIDE } from "../../src/state/loads.ts";
import { buildPrincipal3D, principalPointsOf, principalTensors } from "../../src/state/principal.ts";
import { eigenSym3, tensorFields } from "../../src/state/tensor.ts";
import { gaussBasePositions } from "../../src/state/gaussPoints.ts";
import { opened } from "../contour/harness.ts";

const fixture = (name: string) => fileURLToPath(new URL(`../../fixtures/${name}`, import.meta.url));
const ORACLE = JSON.parse(readFileSync(fixture("ladruno_quad2d.principal.oracle.json"), "utf8")) as {
  frames: { family: string; stage: string; step: number; points: { element: number; natural: number[]; values: number[]; vectors: number[][] }[] }[];
};

const close = (a: number, b: number, scale: number, what: string) => assert.ok(Math.abs(a - b) <= 1e-9 * scale, `${what}: ${a} vs ${b}`);

test("oracle: each Gauss point's principal values and directions equal Python's principal_frame, for the stress and the strain", async () => {
  const h = await opened(fixture("ladruno_quad2d.h5"), fixture("ladruno_quad2d.ladruno"));
  try {
    for (const frame of ORACLE.frames) {
      h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: frame.step } });
      const s = h.store.get();
      const field = tensorFields(s.results!.stages[0]!.gauss).find((f) => f.family === frame.family)!;
      assert.ok(field, `a ${frame.family} field`);
      assert.deepEqual(field.refused, []);
      const points = principalPointsOf(s, h.blobs, field)!;
      assert.equal(points.layout.count, frame.points.length);
      // The app's points are element-major, Python's slab Gauss-point-major: match them by element and natural coordinates.
      const g = s.results!.stages[0]!.gauss.find((x) => x.id === field.groups[0]!.id)!;
      const eids = h.blobs.get(g.elementIds!) as Float64Array;
      const key = (e: number, nat: readonly number[]) => `${e}|${nat.map((x) => x.toFixed(9)).join(",")}`;
      const at = new Map(frame.points.map((p, i) => [key(p.element, p.natural), i]));
      const order = [...points.row].map((r) => at.get(key(eids[Math.floor(r / g.nGauss)]!, g.naturalCoords![r % g.nGauss]!)));
      assert.ok(order.every((i) => i !== undefined) && new Set(order).size === order.length, "every Python point once");
      const step = h.host.readStep({ path: s.results!.path, stage: s.results!.stages[0]!.id, step: frame.step, seq: 1, tensor: field.components });
      assert.ok(step.ok, step.ok ? "" : step.error);
      if (!step.ok) return;
      const tensors = principalTensors(points, field, step.tensor!);
      const scale = Math.max(...frame.points.flatMap((p) => p.values.map(Math.abs)));
      order.forEach((j, i) => {
        const p = frame.points[j!]!;
        const e = eigenSym3(tensors.subarray(6 * i, 6 * i + 6));
        p.values.forEach((v, k) => close(e.values[k]!, v, scale, `${frame.family} point ${i} p${k + 1}`));
        // A repeated value has no unique direction; this file's three values are distinct at every point.
        p.vectors.forEach((v, k) => v.forEach((x, c) => close(e.vectors[k]![c]!, x, 1, `${frame.family} point ${i} v${k + 1}[${c}]`)));
      });
    }
  } finally {
    h.effects.dispose();
  }
});

test("the arrows: tail-anchored at the points, two per principal on an axis, the colour by sign; a plane element's zero zz draws none", async () => {
  const h = await opened(fixture("ladruno_quad2d.h5"), fixture("ladruno_quad2d.ladruno"));
  try {
    h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 1 } });
    const s = h.store.get();
    const field = tensorFields(s.results!.stages[0]!.gauss).find((f) => f.family === "stress")!;
    const points = principalPointsOf(s, h.blobs, field)!;
    const step = h.host.readStep({ path: s.results!.path, stage: s.results!.stages[0]!.id, step: 1, seq: 1, tensor: field.components });
    assert.ok(step.ok);
    if (!step.ok) return;
    const tensors = principalTensors(points, field, step.tensor!);
    const nodes = s.mesh!.nodes!;
    const base = gaussBasePositions(points.layout, h.blobs.get(nodes.ids) as Float64Array, h.blobs.get(nodes.xyz) as Float64Array);
    const frame = ORACLE.frames.find((f) => f.family === "stress" && f.step === 1)!;
    // Each point has one zero principal (zz of plane stress) and two non-zero in-plane ones.
    const nonzero = frame.points.reduce((n, p) => n + p.values.filter((v) => v !== 0).length, 0);
    const opts = { show: [true, true, true] as const, head: "axis" as const, cmap: "RdBu_r" as const, scale: 1, radius: s.mesh!.radius };
    const axis = buildPrincipal3D(base, tensors, opts);
    assert.equal(axis.arrows.length / ARROW_STRIDE, 2 * nonzero);
    assert.equal(axis.points, 24);
    const one = buildPrincipal3D(base, tensors, { ...opts, head: "arrow" });
    assert.equal(one.arrows.length / ARROW_STRIDE, nonzero);
    const maxAbs = Math.max(...frame.points.flatMap((p) => p.values.map(Math.abs)));
    close(one.maxAbs, maxAbs, maxAbs, "maxAbs");
    const rec = (a: Float32Array, j: number) => a.subarray(j * ARROW_STRIDE, (j + 1) * ARROW_STRIDE);
    for (let j = 0; j < nonzero; j++) {
      const r = rec(one.arrows, j);
      assert.equal(r[ARROW.anchor], 1, "tail-anchored");
      // The two of an axis go opposite ways from the same point.
      const a = rec(axis.arrows, 2 * j), b = rec(axis.arrows, 2 * j + 1);
      assert.deepEqual([...a.subarray(0, 3)], [...b.subarray(0, 3)]);
      for (let c = 0; c < 3; c++) assert.equal(a[ARROW.dir + c], -b[ARROW.dir + c]!);
      assert.equal(a[ARROW.level], b[ARROW.level], "one point thins as one");
    }
    // Tension red, compression blue on RdBu_r: a uniaxial tension and a uniaxial compression of one size.
    const signed = buildPrincipal3D(Float32Array.from([0, 0, 0, 1, 0, 0]), Float64Array.from([2, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0]), { ...opts, head: "arrow" });
    assert.equal(signed.arrows.length / ARROW_STRIDE, 2);
    const [t, c] = [rec(signed.arrows, 0), rec(signed.arrows, 1)];
    assert.ok(t[ARROW.rgb]! > t[ARROW.rgb + 2]! && c[ARROW.rgb + 2]! > c[ARROW.rgb]!, "red tension, blue compression");
    assert.deepEqual([...t.subarray(ARROW.dir, ARROW.dir + 3)], [1, 0, 0]);
    // p1 alone: one arrow per point that has a non-zero p1.
    const p1 = buildPrincipal3D(base, tensors, { ...opts, head: "arrow", show: [true, false, false] });
    assert.equal(p1.arrows.length / ARROW_STRIDE, frame.points.filter((p) => p.values[0] !== 0).length);
  } finally {
    h.effects.dispose();
  }
});
