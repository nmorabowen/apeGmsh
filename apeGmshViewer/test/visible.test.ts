// The viewport's rebuild decision, without a GPU: what is drawn is compared
// by contents, so isolating one group after another of the same size
// redraws (the V2e review's "isolate swap" finding).

import assert from "node:assert/strict";
import { test } from "node:test";
import { edgeMask, sameDrawn, visibleMaps } from "../src/renderer/visible.ts";
import type { LegendEntry } from "../src/state/types.ts";

const legend: LegendEntry[] = [
  { decl: "mesh/physical_group/LeftColumn", name: "LeftColumn", color: [1, 0, 0], elements: 2, cue: null, slot: null },
  { decl: "mesh/physical_group/RightColumn", name: "RightColumn", color: [0, 1, 0], elements: 2, cue: null, slot: null },
  { decl: null, name: "(no physical group)", color: [0.5, 0.5, 0.5], elements: 1, cue: null, slot: null },
];
// segments 0,1 are LeftColumn; 2,3 RightColumn; 4 in no group
const lineGroup = Int32Array.from([0, 0, 1, 1, 2]);
const triGroup = Int32Array.from([]);

test("isolate swap: LeftColumn only, then RightColumn only, draws different segments of the same count", () => {
  const left = visibleMaps(legend, ["mesh/physical_group/RightColumn"], lineGroup, triGroup);
  const right = visibleMaps(legend, ["mesh/physical_group/LeftColumn"], lineGroup, triGroup);
  assert.deepEqual(Array.from(left.lineMap), [0, 1, 4]);
  assert.deepEqual(Array.from(right.lineMap), [2, 3, 4]);
  assert.equal(left.lineMap.length, right.lineMap.length, "equal counts: a count comparison would not redraw");
  assert.equal(sameDrawn(left, right), false);
  assert.equal(sameDrawn(left, visibleMaps(legend, ["mesh/physical_group/RightColumn"], lineGroup, triGroup)), true);
});

test("nothing hidden draws everything in order; a synthetic row cannot be hidden", () => {
  const all = visibleMaps(legend, [], lineGroup, triGroup);
  assert.deepEqual(Array.from(all.lineMap), [0, 1, 2, 3, 4]);
  assert.deepEqual(Array.from(visibleMaps(legend, ["nowhere"], lineGroup, triGroup).lineMap), [0, 1, 2, 3, 4]);
});

test("edge mask: an outline edge stays iff it is a side of a visible triangle", () => {
  // Two unit squares side by side, each a fan of two triangles; the outline has 7 edges (the shared one once).
  const q = (x0: number) => [
    x0, 0, 0, x0 + 1, 0, 0, x0 + 1, 1, 0,
    x0, 0, 0, x0 + 1, 1, 0, x0, 1, 0,
  ];
  const tri = new Float32Array([...q(0), ...q(1)]);
  const edges = new Float32Array([
    0, 0, 0, 1, 0, 0,  1, 0, 0, 1, 1, 0,  1, 1, 0, 0, 1, 0,  0, 1, 0, 0, 0, 0, // left square
    1, 0, 0, 2, 0, 0,  2, 0, 0, 2, 1, 0,  2, 1, 0, 1, 1, 0, // right square, shared edge not repeated
  ]);
  assert.deepEqual(Array.from(edgeMask(tri, edges, Int32Array.from([0, 1, 2, 3]))), [1, 1, 1, 1, 1, 1, 1]);
  // Only the left square visible: its 4 sides stay, including the shared edge; the right square's 3 own edges go.
  assert.deepEqual(Array.from(edgeMask(tri, edges, Int32Array.from([0, 1]))), [1, 1, 1, 1, 0, 0, 0]);
  assert.deepEqual(Array.from(edgeMask(tri, edges, Int32Array.from([2, 3]))), [0, 1, 0, 0, 1, 1, 1]);
});
