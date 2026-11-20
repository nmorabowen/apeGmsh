// The Cracks card end to end over fixtures/ladruno_cracks3d.ladruno (three
// ASDConcrete3D bricks pulled along x, the third cracked): the crack-component
// spellings, the disc geometry (state/cracks.ts), the reducer, the step fetch of
// the nine components (effects), the layer (renderer/cracks.ts) and the card
// (panels/cracks.ts).

import assert from "node:assert/strict";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import { renderCracks } from "../src/panels/cracks.ts";
import { CracksLayer } from "../src/renderer/cracks.ts";
import { buildCracks3D, crackHalfSize, crackSlotsOf, planeBasis } from "../src/state/cracks.ts";
import { initialCracks } from "../src/state/reduce.ts";
import { crackFieldOf } from "../src/state/selectors.ts";
import { initialState, reduce } from "../src/state/store.ts";
import { opened, until } from "./contour/harness.ts";

const fixture = (name: string) => fileURLToPath(new URL(`../fixtures/${name}`, import.meta.url));
const h = await opened(fixture("ladruno_cracks3d.h5"), fixture("ladruno_cracks3d.ladruno"));
after(() => {
  h.effects.dispose();
  h.host.close();
});

const LADRUNO = [1, 2, 3].flatMap((k) => ["x", "y", "z"].map((a) => `material.crackPattern/C${k}${a}`));
/** The last step. */
const LAST = 3;
// Bricks 1 and 2 never crack; brick 3 has cracked from step 0 (5.3e-4) to step 3 (3.1e-3).

async function atStep(step: number, withTensor: boolean) {
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step } });
  const here = () => {
    const d = h.effects.currentStep();
    return d && d.stage === 0 && d.step === step && (!withTensor || d.tensor) ? d : null;
  };
  await until(() => here() !== null, `step ${step}`);
  return here()!;
}

test("crackSlotsOf: the three spellings, slot-major; a part of one is refused; none is null", () => {
  const canonical = [1, 2, 3].flatMap((k) => ["x", "y", "z"].map((a) => `crack_c${k}_${a}`));
  assert.deepEqual(crackSlotsOf([...canonical].reverse()), { slots: canonical });
  assert.deepEqual(crackSlotsOf(["other/x", ...LADRUNO]), { slots: LADRUNO });
  const generic = Array.from({ length: 9 }, (_, i) => `material.crackPattern/C${i + 1}`);
  assert.deepEqual(crackSlotsOf(generic), { slots: generic });
  const part = crackSlotsOf(LADRUNO.slice(0, 4));
  assert.ok(part && "partial" in part && /4 of the 9/.test(part.partial), JSON.stringify(part));
  assert.equal(crackSlotsOf(["stress/sigma11", "material.crackWidth/cw"]), null);
});

test("planeBasis: two unit vectors normal to n and to each other", () => {
  for (const n of [[1, 0, 0], [0, 0, -1], [0.6, 0.8, 0], [1 / Math.sqrt(3), -1 / Math.sqrt(3), 1 / Math.sqrt(3)]] as const) {
    const { u, v } = planeBasis(n[0], n[1], n[2]);
    const dot = (a: readonly number[], b: readonly number[]) => a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;
    assert.ok(Math.abs(dot(u, u) - 1) < 1e-12 && Math.abs(dot(v, v) - 1) < 1e-12, "unit");
    assert.ok(Math.abs(dot(u, n)) < 1e-12 && Math.abs(dot(v, n)) < 1e-12 && Math.abs(dot(u, v)) < 1e-12, "orthogonal");
  }
});

test("crackHalfSize: 0.4 x the pitch on a line, in a plane (Python's default_half_length) and in a volume", () => {
  const line = Float32Array.from([0, 0, 0, 1, 0, 0, 2, 0, 0]);
  assert.ok(Math.abs(crackHalfSize(line) - 0.4) < 1e-6);
  // A 2 x 2 grid of pitch 1 in the xz plane: sqrt(area / n) = sqrt(1 / 4) = 0.5.
  const plane = Float32Array.from([0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1]);
  assert.ok(Math.abs(crackHalfSize(plane) - 0.4 * 0.5) < 1e-6);
  // Stacked points (the same x, z twice) count once, as Python's distinct projected positions do.
  const cube = Float32Array.from([0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 2, 0, 0, 0, 2, 2, 0, 2, 0, 2, 2, 2, 2, 2]);
  assert.ok(Math.abs(crackHalfSize(cube) - 0.4 * Math.cbrt(8 / 8)) < 1e-6);
});

test("buildCracks3D: a disc per opening > 0 and >= the threshold, in its crack plane, coloured over minOpening..max", () => {
  const base = Float32Array.from([0, 0, 0, 5, 0, 0, NaN, 0, 0]);
  // Point 0: a 2.0 crack along x and a 0.5 one along y; point 1: a 1.0 crack along z, a zero slot; point 2 unplaced.
  const v = new Float64Array(27);
  v.set([2, 0, 0, 0, 0.5, 0, 0, 0, 0], 0);
  v.set([0, 0, -1, 0, 0, 0, 0, 0, 0], 9);
  v.set([9, 0, 0, 0, 0, 0, 0, 0, 0], 18);
  const all = buildCracks3D(base, v, { minOpening: 0, cmap: "jet", halfSize: 0.25, sides: 4 });
  assert.deepEqual([all.discs, all.points, all.maxOpening], [3, 2, 2]);
  assert.equal(all.positions.length, 3 * 4 * 3 * 3, "4 triangles of 3 vertices per disc");
  // Disc 0 lies in x = 0 (normal x), at most 0.25 from its point.
  for (let i = 0; i < 12; i++) {
    const [x, y, z] = [all.positions[3 * i]!, all.positions[3 * i + 1]!, all.positions[3 * i + 2]!];
    assert.ok(Math.abs(x) < 1e-7 && Math.hypot(y, z) <= 0.25 + 1e-6, `${x} ${y} ${z}`);
  }
  // Disc 2 lies in z = 0 around (5, 0, 0).
  for (let i = 24; i < 36; i++) assert.ok(Math.abs(all.positions[3 * i + 2]!) < 1e-7);
  const kept = buildCracks3D(base, v, { minOpening: 1, cmap: "jet", halfSize: 0.25, sides: 4 });
  assert.deepEqual([kept.discs, kept.points], [2, 2], "the 0.5 crack is under the threshold");
  // The colour range is minOpening..max: the opening 1 at the bottom of jet, 2 at its top.
  const lo = Array.from(kept.colours.subarray(36, 39)), hi = Array.from(kept.colours.subarray(0, 3));
  assert.notDeepEqual(lo, hi);
  assert.equal(buildCracks3D(base, v, { minOpening: 10, cmap: "jet", halfSize: 0.25 }).discs, 0);
});

test("setCracks: defaults, a change, and refusals of a bad value", () => {
  assert.deepEqual(initialState.cracks, initialCracks);
  assert.deepEqual(initialCracks, { on: false, minOpening: 0, cmap: "turbo", scale: 1, face: "both" });
  const s = reduce(initialState, { type: "setCracks", on: true, minOpening: 1e-3 });
  assert.deepEqual([s.cracks.on, s.cracks.minOpening], [true, 1e-3]);
  assert.equal(reduce(s, { type: "setCracks", minOpening: 1e-3 }), s, "the same value: no change");
  for (const bad of [{ minOpening: -1 }, { minOpening: NaN }, { scale: 0 }, { cmap: "rainbow" }, { on: "yes" }, { face: "top" }]) {
    assert.throws(() => reduce(initialState, { type: "setCracks", ...(bad as object) } as never), RangeError, JSON.stringify(bad));
  }
});

test("the stage's crack field is the file's crackPattern; the step carries its nine components only while the card is on", async () => {
  await atStep(LAST, false);
  const f = crackFieldOf(h.store.get())!;
  assert.deepEqual(f.components, LADRUNO);
  assert.equal(f.groups.length, 1);
  assert.deepEqual(f.refused, []);
  assert.equal(h.effects.currentStep()!.tensor, undefined, "off: no crack read");
  h.store.dispatch({ type: "setCracks", on: true });
  const d = await atStep(LAST, true);
  assert.deepEqual(Object.keys(d.tensor!), LADRUNO);
  h.store.dispatch({ type: "setCracks", on: false });
});

test("the layer: the third brick's eight x-normal cracks lie in their Gauss planes x = 2.21 and 2.79; off hides them", async () => {
  h.store.dispatch({ type: "setCracks", on: true, minOpening: 1e-3 });
  const d = await atStep(LAST, true);
  const layer = new CracksLayer();
  try {
    assert.equal(layer.sync(h.store.get(), h.blobs, null, d), true);
    assert.equal(layer.root.visible, true);
    assert.equal(layer.drawn.discs, 8, "one primary crack per Gauss point of brick 3; the secondary ones are under 1e-3");
    assert.equal(layer.drawn.points, 8);
    assert.ok(layer.drawn.maxOpening > 2.5e-3 && layer.drawn.maxOpening < 3.5e-3, `${layer.drawn.maxOpening}`);
    const pos = (layer.root.children[0] as unknown as { geometry: { getAttribute(n: string): { array: Float32Array } } }).geometry.getAttribute("position").array;
    const g = 1 / Math.sqrt(3);
    const planes = [2.5 - 0.5 * g, 2.5 + 0.5 * g];
    for (let i = 0; i < pos.length / 3; i++) assert.ok(planes.some((x) => Math.abs(pos[3 * i]! - x) < 1e-3), `vertex x ${pos[3 * i]}`);
    assert.equal(layer.sync(h.store.get(), h.blobs, null, d), false, "the same step draws nothing new");
    h.store.dispatch({ type: "setCracks", minOpening: 0 });
    assert.equal(layer.sync(h.store.get(), h.blobs, null, d), true);
    assert.deepEqual([layer.drawn.discs, layer.drawn.points], [24, 8], "the two secondary cracks of each point too");
    // Step 0: brick 3 has opened 5.3e-4, under the 1e-3 threshold: nothing is drawn.
    h.store.dispatch({ type: "setCracks", minOpening: 1e-3 });
    const d0 = await atStep(0, true);
    layer.sync(h.store.get(), h.blobs, null, d0);
    assert.equal(layer.drawn.discs, 0);
    assert.equal(layer.root.visible, false);
    h.store.dispatch({ type: "setCracks", on: false });
    layer.sync(h.store.get(), h.blobs, null, d0);
    assert.equal(layer.root.visible, false);
  } finally {
    layer.dispose();
    h.store.dispatch({ type: "setCracks", on: false, minOpening: 0 });
  }
});

test("a deformed step moves the discs with the Gauss points", async () => {
  h.store.dispatch({ type: "setCracks", on: true, minOpening: 1e-3 });
  const d = await atStep(LAST, true);
  const layer = new CracksLayer();
  try {
    const xs = () => Array.from((layer.root.children[0] as unknown as { geometry: { getAttribute(n: string): { array: Float32Array } } }).geometry.getAttribute("position").array).filter((_, i) => i % 3 === 0);
    layer.sync(h.store.get(), h.blobs, null, d);
    const before = xs();
    const ids = h.blobs.get(h.store.get().mesh!.nodes!.ids) as Float64Array;
    const column = new Map([...ids].map((id, i) => [id, i]));
    const one = new Float64Array(ids.length).fill(1), zero = new Float64Array(ids.length);
    h.store.dispatch({ type: "setDeformScale", scale: 2 } as never);
    layer.sync(h.store.get(), h.blobs, { key: "probe", disp: { nodal: { column, rows: [one, zero, null] } } } as never, d);
    const k = h.store.get().deform.scale;
    xs().forEach((x, i) => assert.ok(Math.abs(x - (before[i]! + k)) < 1e-5, `${x} vs ${before[i]} + ${k}`));
  } finally {
    layer.dispose();
    h.store.dispatch({ type: "setCracks", on: false, minOpening: 0 });
  }
});

test("the card shows on a stage with a crack pattern; its switch and threshold dispatch setCracks", async () => {
  await atStep(LAST, false);
  const prevDoc = (globalThis as { document?: unknown }).document;
  class El {
    className = "";
    title = "";
    value = "";
    type = "";
    step = "";
    min = "";
    selected = false;
    dataset: Record<string, string> = {};
    style: Record<string, string> = {};
    children: (El | string)[] = [];
    listeners: (() => void)[] = [];
    invalid = "";
    append(...xs: (El | string)[]) {
      this.children.push(...xs);
    }
    setAttribute() {}
    setCustomValidity(m: string) {
      this.invalid = m;
    }
    addEventListener(_: string, fn: () => void) {
      this.listeners.push(fn);
    }
    removeEventListener() {}
    set textContent(v: string) {
      this.children = [v];
    }
    all(pred: (e: El) => boolean): El[] {
      return [...(pred(this) ? [this] : []), ...this.children.flatMap((c) => (typeof c === "string" ? [] : c.all(pred)))];
    }
  }
  (globalThis as { document?: unknown }).document = { documentElement: { dataset: { theme: "dark" } }, createElement: () => new El(), createElementNS: () => new El() };
  try {
    const events: unknown[] = [];
    const box = renderCracks(h.store.get(), (e) => events.push(e)) as unknown as El | null;
    assert.ok(box, "the card shows: the stage records a crack pattern");
    const one = (cls: string) => {
      const found = box.all((e) => e.className.split(" ")[0] === cls || e.className.split(" ").includes(cls));
      assert.equal(found.length, 1, cls);
      return found[0]!;
    };
    for (const fn of one("cracks-toggle").listeners) fn();
    const min = one("cracks-min");
    min.value = "0.001";
    for (const fn of min.listeners) fn();
    min.value = "-1";
    for (const fn of min.listeners) fn();
    assert.match(min.invalid, /0 or more/);
    assert.deepEqual(events, [{ type: "setCracks", on: true }, { type: "setCracks", minOpening: 0.001 }]);
    h.store.dispatch({ type: "setPhase", at: { kind: "mesh" } });
    assert.equal(renderCracks(h.store.get()), null, "no results step on screen: no card");
  } finally {
    (globalThis as { document?: unknown }).document = prevDoc;
  }
});
