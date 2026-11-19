// ADR 0138 D7: the box a drag covers and the IDs an ID image holds inside it.

import assert from "node:assert/strict";
import { test } from "node:test";
import { boxOf, idsInBox } from "../src/renderer/picking.ts";
import { selectModeOf } from "../src/ui/dom.ts";

test("a drag toward the right is a window box, toward the left a crossing box", () => {
  assert.deepEqual(boxOf(10, 20, 50, 5), { x0: 10, y0: 5, x1: 50, y1: 20, crossing: false });
  assert.deepEqual(boxOf(50, 20, 10, 40), { x0: 10, y0: 20, x1: 50, y1: 40, crossing: true });
});

test("the IDs in a box, each once, from an image whose row 0 is the bottom (as WebGL reads it)", () => {
  // A 4 x 3 image: ID 7 (element 6) fills the top row, ID 2 (element 1) the bottom-left pixel.
  const w = 4, h = 3, ids = new Uint32Array(w * h);
  ids[2 * w + 0] = ids[2 * w + 1] = ids[2 * w + 2] = ids[2 * w + 3] = 7;
  ids[0] = 2 | (0xff << 24);
  const seen = new Uint32Array(8);
  const top: number[] = [];
  idsInBox(ids, w, h, { x0: 0, y0: 0, x1: 3, y1: 0, crossing: false }, 1, seen, 1, top);
  assert.deepEqual(top, [6], "the top row in canvas pixels is the image's last row; each ID once");
  const all: number[] = [];
  idsInBox(ids, w, h, { x0: 0, y0: 0, x1: 3, y1: 2, crossing: false }, 1, seen, 2, all);
  assert.deepEqual(all.sort(), [1, 6], "the alpha byte is not part of the ID");
  const scaled: number[] = [];
  idsInBox(ids, w, h, { x0: 0, y0: 1, x1: 0.4, y1: 1.4, crossing: false }, 2, seen, 3, scaled);
  assert.deepEqual(scaled, [1], "CSS pixels times the device pixel ratio");
});

test("ADR 0138 D6: no modifier replaces, Shift adds, Ctrl or Cmd removes", () => {
  assert.equal(selectModeOf({}), "replace");
  assert.equal(selectModeOf({ shiftKey: true }), "add");
  assert.equal(selectModeOf({ ctrlKey: true }), "remove");
  assert.equal(selectModeOf({ metaKey: true, shiftKey: true }), "remove");
});
