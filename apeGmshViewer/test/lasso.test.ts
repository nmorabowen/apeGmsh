// ADR 0138 D7: the lasso, a mode of the selection bar. Its path is filled
// into a pixel mask (even-odd, pixel centres); drawn clockwise it takes what
// is fully inside, counter-clockwise what it touches, as a box drawn left to
// right or right to left does.

import assert from "node:assert/strict";
import { test } from "node:test";
import { selectionBarLines } from "../src/panels/selectionBar.ts";
import { captureBox, lassoAround } from "../src/renderer/harness/selection.ts";
import { elementsInLasso, idsInLasso, inLasso, lassoClockwise, lassoMask } from "../src/renderer/picking.ts";
import { regionLabel } from "../src/renderer/viewport.ts";
import { initialState, reduce } from "../src/state/reduce.ts";
import type { MeshInfo, State } from "../src/state/types.ts";

// An L: the square 0..20 without its top-right quarter (10..20 x 0..10), clockwise on screen (y down).
const L = Float32Array.from([0, 0, 10, 0, 10, 10, 20, 10, 20, 20, 0, 20]);

test("the mask holds the pixels whose centres the path encloses, a concave one included; device pixels at a ratio of 2", () => {
  const m = lassoMask(L, 1);
  assert.deepEqual([inLasso(m, 5, 5), inLasso(m, 15, 15), inLasso(m, 5, 15)], [true, true, true]);
  assert.equal(inLasso(m, 15, 5), false, "the cut-out quarter is outside");
  assert.equal(inLasso(m, 25, 5), false);
  const m2 = lassoMask(L, 2);
  assert.equal(m2.w, 41);
  assert.deepEqual([inLasso(m2, 9.6, 9.6), inLasso(m2, 10.4, 9.6)], [true, false]);
  assert.equal(lassoMask(Float32Array.from([0, 0, 5, 5]), 1).w, 0, "two points enclose nothing");
});

test("drawn clockwise on screen, a window lasso; counter-clockwise, a crossing one", () => {
  assert.equal(lassoClockwise(L), true);
  const back = new Float32Array(L.length);
  for (let i = 0; i < L.length / 2; i++) back.set(L.subarray(2 * i, 2 * i + 2), L.length - 2 - 2 * i);
  assert.equal(lassoClockwise(back), false);
});

test("the IDs under the mask, each once, from an image whose row 0 is the bottom", () => {
  // A 30 x 30 image: ID 3 (element 2) in the cut-out quarter, ID 5 (element 4) inside the L, ID 9 (element 8) outside it.
  const w = 30, h = 30, ids = new Uint32Array(w * h);
  const put = (x: number, y: number, id: number) => (ids[(h - 1 - y) * w + x] = id);
  put(15, 5, 3);
  put(5, 15, 5);
  put(6, 15, 5);
  put(25, 25, 9);
  const found: number[] = [];
  idsInLasso(ids, w, h, lassoMask(L, 1), new Uint32Array(10), 1, found);
  assert.deepEqual(found, [4]);
});

test("through the model: clockwise takes the elements wholly inside, counter-clockwise any reaching in; a cut-away vertex does not count", () => {
  // Element 0 inside the L; element 1 straddling into the cut-out; element 2 inside but for a cut-away vertex.
  const xy = Float32Array.from([2, 2, 4, 2, 3, 4, 8, 4, 14, 4, 8, 6, 2, 15, 4, 15, NaN, 0]);
  const element = Int32Array.from([0, 0, 0, 1, 1, 1, 2, 2, 2]);
  const m = lassoMask(L, 1);
  assert.deepEqual(elementsInLasso(xy, element, m, 3, false), [0, 2]);
  assert.deepEqual(elementsInLasso(xy, element, m, 3, true), [0, 1, 2]);
});

test("the capture's lasso holds its whole box, clockwise, so it takes what the box takes on a coarse wall too", () => {
  // The flat-wall fixture's capture (1584 x 1113 canvas): the box from 30 % to 60 % across and 30 % to 70 % down took the
  // top-right shell, corners as drawn. The fixed lasso used before held none of the six shells whole and took nothing.
  const a = { x: 475.2, y: 333.9 }, b = { x: 950.4, y: 779.1 };
  const loop = lassoAround(a, b), path = Float32Array.from(loop.flatMap((p) => [p.x, p.y]));
  assert.equal(lassoClockwise(path), true, "a window lasso");
  const m = lassoMask(path, 1);
  for (let y = Math.ceil(a.y); y <= b.y; y += 3) for (let x = Math.ceil(a.x); x <= b.x; x += 3) assert.ok(inLasso(m, x, y), `box pixel ${x},${y} outside the lasso`);
  const shell = [[660, 338], [935, 410], [905, 690], [655, 598]] as const;
  assert.deepEqual(elementsInLasso(Float32Array.from(shell.flat()), new Int32Array(4), m, 1, false), [0]);
});

test("the capture box takes an element whole: the middle box, or a wider one where the middle holds none (hinge fixtures)", () => {
  // ladruno_hinges3d's capture (1584 x 1113 canvas): four beams from one tip fanning up and left, as drawn. None lies whole
  // in the middle box, so it took nothing, the clear after it was no undo step, and undo put back the click still's beam.
  const r = { left: 0, top: 0, width: 1584, height: 1113 };
  const tip = [878, 772], beams = [[250, 205], [262, 375], [275, 531], [287, 674]].map((p) => [tip, p]);
  type P = { x: number; y: number };
  const takes = (segs: number[][][]) => (a: P, b: P) => segs.filter((s) => s.every(([x, y]) => x! >= a.x && x! <= b.x && y! >= a.y && y! <= b.y)).length;
  assert.equal(takes(beams)({ x: 475.2, y: 333.9 }, { x: 950.4, y: 779.1 }), 0, "the middle box holds none of the beams whole");
  const { a, b } = captureBox(r, takes(beams));
  assert.deepEqual([a, b].map((p) => [Math.round(p.x), Math.round(p.y)]), [[48, 33], [1536, 1080]]);
  assert.equal(takes(beams)(a, b), 4);
  // ladruno_hinges2d's capture: two beams framed out to 7 % of the width; only the inset canvas holds them.
  const fan2d = [[[975, 556], [110, 340]], [[975, 556], [112, 772]]];
  const wide = captureBox(r, takes(fan2d));
  assert.equal(takes(fan2d)(wide.a, wide.b), 2);
  const shell = [[[660, 338], [935, 410], [905, 690], [655, 598]]];
  const kept = captureBox(r, takes(shell));
  assert.deepEqual([kept.a, kept.b].map((p) => [Math.round(p.x), Math.round(p.y)]), [[475, 334], [950, 779]], "a solid model keeps the middle box");
  assert.throws(() => captureBox(r, () => 0), /no capture box takes an element whole/);
});

test("the label says what a box or lasso takes, and how", () => {
  assert.equal(regionLabel(1, "element", "replace", false, false), "1 element inside");
  assert.equal(regionLabel(1234, "element", "add", true, true), "+ 1,234 elements touching · through");
  assert.equal(regionLabel(3, "node", "remove", true, false), "− 3 nodes");
});

test("the selection bar's Lasso mode: off by default, a store value, no selection edit", () => {
  let s: State = { ...initialState, mesh: { elements: [] } as unknown as MeshInfo };
  assert.equal(selectionBarLines(s)!.lasso, false);
  s = reduce(s, { type: "setSelectLasso", on: true });
  assert.equal(selectionBarLines(s)!.lasso, true);
  assert.equal(s.selectionUndo.past.length, 0);
  assert.match(selectionBarLines(s)!.hint, /clockwise takes what is fully inside/);
});
