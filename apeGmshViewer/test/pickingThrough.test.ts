// ADR 0138 D7: a box through the model reads projected positions, not the ID
// image: every drawn target in the box, whether something hides it or not;
// what a section plane cuts away (NaN) is still left out.

import assert from "node:assert/strict";
import { test } from "node:test";
import { selectionBarLines } from "../src/panels/selectionBar.ts";
import { boxOf, elementsThrough, pointsThrough } from "../src/renderer/picking.ts";
import { initialState, reduce } from "../src/state/reduce.ts";
import type { MeshInfo, State } from "../src/state/types.ts";

// Element 0: a triangle inside the box (10..50). Element 1: a triangle straddling its right side.
// Element 2: a segment wholly to the left. Element 3: a segment crossing the box with both ends outside.
// Element 4: a triangle inside but for one vertex a section plane cuts away.
const xy = Float32Array.from([
  20, 20, 30, 20, 25, 30, //   prim 0, element 0
  40, 20, 70, 20, 45, 30, //   prim 1, element 1
  0, 20, 5, 25, //             prim 2, element 2
  0, 40, 60, 40, //            prim 3, element 3
  15, 45, 20, 45, NaN, 0, //   prim 4, element 4
]);
const element = Int32Array.from([0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4]);
const prim = Int32Array.from([0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4]);

test("through, left to right: the elements with every kept vertex inside; a cut-away vertex does not count as outside", () => {
  assert.deepEqual(elementsThrough(xy, element, prim, boxOf(10, 10, 50, 50), 5), [0, 4]);
});

test("through, right to left: every element with a primitive not wholly beyond one side, the long segment across the box included", () => {
  assert.deepEqual(elementsThrough(xy, element, prim, boxOf(50, 10, 10, 50), 5), [0, 1, 3, 4]);
});

test("through, nodes: the points inside the box; a cut-away one is left out", () => {
  assert.deepEqual(pointsThrough(Float32Array.from([20, 20, 60, 20, NaN, 0, 49, 49]), boxOf(10, 10, 50, 50)), [0, 3]);
});

test("the selection bar's Through toggle: off by default, and a store value", () => {
  let s: State = { ...initialState, mesh: { elements: [] } as unknown as MeshInfo };
  assert.equal(selectionBarLines(s)!.through, false);
  s = reduce(s, { type: "setSelectThrough", on: true });
  assert.equal(selectionBarLines(s)!.through, true);
  assert.equal(reduce(s, { type: "setSelectThrough", on: true }), s, "no change: the same state");
  assert.equal(s.selectionUndo.past.length, 0, "the toggle is no selection edit");
});
