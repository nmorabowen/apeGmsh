// #3138: a zero-length segment (a zeroLength spring between coincident nodes)
// stays zero-length in the deformed shape. Moving each end by its own node's
// displacement stretched 17 793 soil springs into stubs from each pile node to
// its fixed anchor, and the piles drew as broken lines.

import assert from "node:assert/strict";
import { test } from "node:test";
import type { DisplacementData } from "../../src/effects.ts";
import { deformFull, type FullPositions } from "../../src/renderer/deform.ts";
import { collapseZeroLength } from "../../src/state/deform.ts";

test("#3138: a segment of coincident ends keeps its J end on its deformed I end; a real segment deforms end by end", () => {
  const base = new Float32Array([
    0, 0, -1, 0, 0, -1, // a spring: pile node and anchor at one point
    0, 0, -1, 0, 0, -2, // a pile segment
  ]);
  const deformed = new Float32Array([
    0.5, 0, -1, 0, 0, -1, // the pile node moved, the anchor did not
    0.5, 0, -1, 0.4, 0, -2,
  ]);
  collapseZeroLength(base, deformed);
  assert.deepEqual([...deformed].map((v) => Math.round(v * 1e6) / 1e6), [0.5, 0, -1, 0.5, 0, -1, 0.5, 0, -1, 0.4, 0, -2]);
  assert.throws(() => collapseZeroLength(new Float32Array(6), new Float32Array(5)), RangeError);
});

test("#3138: deformFull draws a spring between a moved pile node and its fixed anchor as nothing", () => {
  const vd = (d: number[]) => ({ d: Float64Array.from(d), missing: new Uint8Array(d.length / 3), count: 0 });
  const base: FullPositions = {
    line: new Float32Array([2, 3, -5, 2, 3, -5, 2, 3, -5, 2, 3, -6]),
    tri: new Float32Array(0),
    edge: new Float32Array(0),
    clearLine: new Float32Array([2, 3, -5, 2, 3, -5, 2, 3, -5, 2, 3, -6]),
    key: "base",
    disp: null,
  };
  // The pile node (vertex 0 of the spring, vertex 0 of the pile) moves 0.05 in x; the anchor and the pile's lower node do not.
  const disp = { key: "k", line: vd([0.05, 0, 0, 0, 0, 0, 0.05, 0, 0, 0, 0, 0]), tri: vd([]), edge: vd([]) } as unknown as DisplacementData;
  const full = deformFull(base, disp, 10, "k|scale 10");
  for (const buf of [full.line, full.clearLine!]) {
    const spring = [...buf.slice(0, 6)];
    assert.deepEqual(spring.slice(0, 3), spring.slice(3, 6), "the spring is zero-length deformed, as undeformed");
    assert.ok(Math.abs(spring[0]! - 2.5) < 1e-6, "at the moved pile node");
    assert.deepEqual([...buf.slice(6, 12)].map((v) => Math.round(v * 1e6) / 1e6), [2.5, 3, -5, 2, 3, -6], "the pile segment deforms end by end");
  }
});
