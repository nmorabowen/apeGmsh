// R2 (#1283): the generated group palette, against the closed forms of its
// construction: n evenly spaced hues (one 30-degree family each up to 12),
// a legend order whose neighbours are far apart on the wheel, and sRGB
// values inside the gamut.

import assert from "node:assert/strict";
import { test } from "node:test";
import { HUE_FAMILIES, hueOf, hueOrder, oklchToSrgb, paletteFor, SAME_FAMILY_DEG, TONE_GUARANTEE, TONES } from "../src/state/palette.ts";

const circ = (a: number, b: number) => Math.min(Math.abs(a - b), 360 - Math.abs(a - b));

test("hueOrder is a permutation whose consecutive entries are at least floor(n/2) - 1 slots apart", () => {
  for (let n = 1; n <= 24; n++) {
    const order = hueOrder(n);
    assert.deepEqual([...order].sort((a, b) => a - b), [...Array(n).keys()], `n=${n} permutation`);
    const minStep = Math.max(0, Math.floor(n / 2) - 1);
    for (let i = 1; i < n; i++) {
      const d = Math.min(Math.abs(order[i]! - order[i - 1]!), n - Math.abs(order[i]! - order[i - 1]!));
      assert.ok(d >= minStep, `n=${n}: entries ${i - 1},${i} are ${d} slots apart (< ${minStep})`);
    }
  }
});

test("up to 12 groups: no two share a hue family (>= 30 degrees apart), neighbours contrast", () => {
  for (let n = 2; n <= HUE_FAMILIES; n++) {
    const hues = [...Array(n).keys()].map((i) => hueOf(i, n).hue);
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) assert.ok(circ(hues[i]!, hues[j]!) >= 360 / n - 1e-9, `n=${n}: ${i},${j}`);
    for (let i = 1; i < n; i++) {
      const expected = Math.max(1, Math.floor(n / 2) - 1) * (360 / n);
      assert.ok(circ(hues[i]!, hues[i - 1]!) >= expected - 1e-9, `n=${n}: neighbours ${i - 1},${i}: ${circ(hues[i]!, hues[i - 1]!)} < ${expected}`);
    }
  }
});

test("up to 24 groups: two hues that read as one family (< 45 degrees apart) never share a tone", () => {
  // The V1 gate on San Ramon (11 groups): two pinks, orange vs salmon, cyan vs teal.
  assert.equal(TONE_GUARANTEE, 24);
  for (let i = 0; i < TONES.length; i++) for (let j = i + 1; j < TONES.length; j++) assert.ok(Math.abs(TONES[i]!.L - TONES[j]!.L) >= 0.14, `tones ${i},${j}`);
  for (let n = 2; n <= TONE_GUARANTEE; n++) {
    const e = [...Array(n).keys()].map((i) => hueOf(i, n));
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      if (circ(e[i]!.hue, e[j]!.hue) < SAME_FAMILY_DEG) {
        assert.notEqual(e[i]!.tone, e[j]!.tone, `n=${n}: entries ${i} (${e[i]!.hue}) and ${j} (${e[j]!.hue}) share tone ${e[i]!.tone}`);
        assert.ok(Math.abs(TONES[e[i]!.tone]!.L - TONES[e[j]!.tone]!.L) >= 0.14, `n=${n}: lightness gap`);
      }
    }
  }
});

test("beyond 12 groups a family repeats in another tone, never the same tone (up to 24)", () => {
  for (const n of [13, 20, 24]) {
    const seen = new Set<string>();
    for (let i = 0; i < n; i++) {
      const { hue, tone } = hueOf(i, n);
      const key = `${Math.round(hue)}:${tone}`;
      assert.ok(!seen.has(key), `n=${n}: ${key}`);
      seen.add(key);
    }
  }
  // Up to 12, the family is unique by itself.
  assert.equal(new Set([...Array(12).keys()].map((i) => Math.round(hueOf(i, 12).hue))).size, 12);
});

test("every colour is inside sRGB, and two groups never get the same colour", () => {
  for (const n of [1, 3, 6, 12, 20]) {
    const p = paletteFor(n);
    assert.equal(p.length, n);
    for (const c of p) for (const v of c) assert.ok(v >= 0 && v <= 1, `${c}`);
    assert.equal(new Set(p.map((c) => c.join(","))).size, n, `n=${n} distinct`);
  }
  // Grey (chroma 0) at L = 1 is white; OKLCH's own closed form.
  assert.deepEqual(oklchToSrgb(1, 0, 0).map((v) => Math.round(v * 100) / 100), [1, 1, 1]);
  assert.deepEqual(oklchToSrgb(0, 0, 0), [0, 0, 0]);
});
