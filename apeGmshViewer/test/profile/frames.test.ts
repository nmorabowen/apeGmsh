import { test } from "node:test";
import assert from "node:assert/strict";
import { fingerprint, flickerFrames } from "../../scripts/profile/frames.ts";

const flat = (v: number) => {
  const rgba = new Uint8Array(4 * 4 * 4).fill(v);
  return fingerprint({ width: 4, height: 4, rgba }, { x: 0, y: 0, width: 4, height: 4 }, 2, 2);
};

test("a frame that jumps away and comes back is a flicker", () => {
  assert.deepEqual(flickerFrames([flat(100), flat(100), flat(10), flat(100), flat(100)]), [2]);
});

test("steady motion is not a flicker, however fast", () => {
  assert.deepEqual(flickerFrames([flat(0), flat(50), flat(100), flat(150), flat(200)]), []);
});

test("one change that stays is not a flicker", () => {
  assert.deepEqual(flickerFrames([flat(100), flat(100), flat(200), flat(200)]), []);
});

test("a change below the visible threshold is not a flicker", () => {
  assert.deepEqual(flickerFrames([flat(100), flat(101), flat(100)]), []);
});
