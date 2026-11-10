// Zero-length spring force arrows (reader/ladruno.ts `readSpringStep`,
// state/springs.ts `buildSprings3D`, renderer/springs.ts, panels/springs.ts):
// one arrow per spring of the step on screen, its tip on the spring, along node
// j's global force. The fixture is the fork-written ladruno_springs3d
// (fixtures/make_ladruno_fixtures.py --springs): spring 1 (nodes 1-2 at the
// origin, -dir 1 2 3) and spring 2 (nodes 3-4 at (2, 0, 0), -orient (1, 1, 0)),
// loads (10, -5, 4) and (10, -5, 0), LoadControl 0.5, so at load factor t
// node j's force is t times its load.

import assert from "node:assert/strict";
import { test, after } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import type * as THREE from "three";
import { ResultsHost } from "../src/main/results.ts";
import { renderSprings } from "../src/panels/springs.ts";
import { nodeJTranslations, openLadruno } from "../src/reader/ladruno.ts";
import type { ResultsH5 } from "../src/reader/results.ts";
import { basePositions } from "../src/renderer/deform.ts";
import { SpringsLayer } from "../src/renderer/springs.ts";
import { ANCHOR, ARROW, ARROW_STRIDE, BAR } from "../src/state/loads.ts";
import { spotlightViewOf } from "../src/state/spotlight.ts";
import { buildSprings3D, springCountOf } from "../src/state/springs.ts";
import { reduce } from "../src/state/store.ts";
import { opened, until } from "./contour/harness.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as ResultsH5;
const fixture = (name: string) => fileURLToPath(new URL(`../fixtures/${name}`, import.meta.url));
const MODEL = fixture("ladruno_springs3d.h5");
const RESULTS = fixture("ladruno_springs3d.ladruno");
const LOAD = [
  [10, -5, 4],
  [10, -5, 0],
];
const TIP = [
  [0, 0, 0],
  [2, 0, 0],
];

const h = await opened(MODEL, RESULTS);
after(() => {
  h.effects.dispose();
  h.host.close();
});

async function atStep(step: number) {
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step } });
  const here = () => {
    const d = h.effects.currentStep();
    return d && d.stage === 0 && d.step === step && (!h.store.get().springs.on || d.springs !== undefined) ? d : null;
  };
  await until(() => here() !== null, `step ${step}`);
  return here()!;
}

const close = (a: number, b: number, what: string) => assert.ok(Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(b)), `${what}: ${a}, expected ${b}`);

test("the reader finds the zero-length springs of a force bucket and reads node j's global force, a skewed spring as an axis-aligned one", () => {
  const f = openLadruno(H5, RESULTS);
  try {
    const id = f.index.stages[0]!.id;
    const warnings: string[] = [];
    assert.equal(f.springCount(id, warnings), 2);
    assert.deepEqual(warnings, []);
    for (const [step, t] of [[0, 0.5], [1, 1]] as const) {
      const r = f.readSpringStep(id, step);
      assert.deepEqual(Array.from(r.elementIds), [1, 2]);
      assert.deepEqual(Array.from(r.nodes), [1, 2, 3, 4]);
      assert.equal(r.time, t);
      for (let k = 0; k < 2; k++) for (let c = 0; c < 3; c++) close(r.force[k * 3 + c]!, t * LOAD[k]![c]!, `spring ${k + 1} ${"xyz"[c]} at t ${t}`);
    }
    assert.throws(() => f.readSpringStep(id, 2), /outside stage/);
  } finally {
    f.close();
  }
});

test("a force bucket of trusses, quads or beams holds no spring: no two-node element of theirs is zero length", () => {
  for (const name of ["ladruno_truss2d.ladruno", "ladruno_quad2d.ladruno", "ladruno_portal3d.ladruno", "ladruno_frame3d.ladruno"]) {
    const f = openLadruno(H5, fixture(name));
    try {
      const warnings: string[] = [];
      assert.equal(f.springCount(f.index.stages[0]!.id, warnings), 0, name);
      assert.deepEqual(warnings, [], name);
    } finally {
      f.close();
    }
  }
});

test("node j's translation columns come from the column order, in either OpenSees spelling; an ambiguous or short list is refused", () => {
  // P<node>_<dof> (zeroLength, trusses), 3-D with three DOFs a node.
  assert.deepEqual(nodeJTranslations(["P1_1", "P1_2", "P1_3", "P2_1", "P2_2", "P2_3"], 3), [3, 4, 5]);
  // P<dof>_<node> (continua): the same vector, the other way round.
  assert.deepEqual(nodeJTranslations(["P1_1", "P2_1", "P3_1", "P1_2", "P2_2", "P3_2"], 3), [3, 4, 5]);
  // Six DOFs a node in 3-D: the translations are DOFs 1..3, the rotations are left out.
  assert.deepEqual(nodeJTranslations(["P1_1", "P1_2", "P1_3", "P1_4", "P1_5", "P1_6", "P2_1", "P2_2", "P2_3", "P2_4", "P2_5", "P2_6"], 3), [6, 7, 8]);
  // Mixed ndf (3, 4): a u-p node j still names its translations.
  assert.deepEqual(nodeJTranslations(["P1_1", "P1_2", "P1_3", "P2_1", "P2_2", "P2_3", "P2_4"], 3), [3, 4, 5]);
  // 2-D: x and y, no z.
  assert.deepEqual(nodeJTranslations(["P1_1", "P1_2", "P2_1", "P2_2"], 2), [2, 3, -1]);
  // One DOF a node reads one way only (node 1 then node 2).
  assert.deepEqual(nodeJTranslations(["P1_1", "P2_1"], 1), [1, -1, -1]);
  assert.equal(nodeJTranslations(["P1", "P2"], 3), null, "basicForce slots name no node");
  assert.equal(nodeJTranslations(["P1_1", "P2_1"], 3), null, "node j records one DOF: no 3-D translation");
  assert.equal(nodeJTranslations(["P1_1", "P1_2", "P1_3"], 3), null, "one node only");
});

test("main sends each stage's spring count, and the springs' step only when asked", async () => {
  const host = new ResultsHost(H5, h5wasm.ready, { idleMs: 50 });
  try {
    const res = await host.open(RESULTS);
    assert.ok(res.ok);
    assert.equal(res.meta.stages[0]!.springs, 2);
    const without = host.readStep({ path: RESULTS, stage: res.meta.stages[0]!.id, step: 1, seq: 1 });
    assert.ok(without.ok && !("springs" in without), "not asked: not read");
    const withS = host.readStep({ path: RESULTS, stage: res.meta.stages[0]!.id, step: 1, seq: 2, springs: true });
    assert.ok(withS.ok && withS.springs);
    for (let i = 0; i < 6; i++) close(withS.springs.force[i]!, LOAD.flat()[i]!, `force[${i}]`);
  } finally {
    host.close();
  }
});

test("the effects fetch the springs with the step while the card is on, and not otherwise", async () => {
  assert.equal(springCountOf(h.store.get()), 0, "no results step on screen yet");
  const off = await atStep(1);
  assert.equal(springCountOf(h.store.get()), 2);
  assert.equal(off.springs, undefined, "the card is off: the step carries no springs");
  h.store.dispatch({ type: "setSprings", on: true });
  const on = await atStep(1);
  assert.ok(on.springs, "the card is on: the step's springs come with it");
  h.store.dispatch({ type: "setSprings", on: false });
});

test("one arrow per spring, its tip on the spring, along node j's force, no mark; lengths follow the step's largest", async () => {
  h.store.dispatch({ type: "setSprings", on: true });
  const data = await atStep(1);
  const s = h.store.get();
  const r = buildSprings3D(s, h.blobs, data.springs!);
  assert.equal(r.count, 2);
  assert.equal(r.unplaced, 0);
  const mags = LOAD.map((l) => Math.hypot(...l));
  close(r.maxMag, Math.max(...mags), "the largest force");
  const lens: number[] = [];
  for (let k = 0; k < 2; k++) {
    const o = k * ARROW_STRIDE;
    for (let c = 0; c < 3; c++) {
      assert.ok(Math.abs(r.arrows[o + ARROW.tip + c]! - TIP[k]![c]!) < 1e-6, `spring ${k + 1}: tip on the spring`);
      assert.ok(Math.abs(r.arrows[o + ARROW.dir + c]! - LOAD[k]![c]! / mags[k]!) < 1e-6, `spring ${k + 1}: along its force`);
    }
    assert.equal(r.arrows[o + ARROW.bar], BAR.none);
    assert.equal(r.arrows[o + ARROW.anchor], ANCHOR.tip, "the tip, not the tail, on the spring");
    assert.equal(r.arrows[o + ARROW.heads], 1);
    lens.push(r.arrows[o + ARROW.len]!);
  }
  // Length = base (0.35 + 0.65 ratio): spring 1 is the largest.
  const ratio = mags[1]! / mags[0]!;
  assert.ok(Math.abs(lens[1]! / lens[0]! - (0.35 + 0.65 * ratio)) < 1e-6, `lengths ${lens}`);
  h.store.dispatch({ type: "setSprings", on: false });
});

test("components along the global axes: xyz draws each non-zero component, one axis draws ±that axis; a spring's components thin together", async () => {
  h.store.dispatch({ type: "setSprings", on: true, component: "xyz" });
  const data = await atStep(1);
  const all = buildSprings3D(h.store.get(), h.blobs, data.springs!);
  assert.equal(all.count, 5, "three components of spring 1, two of spring 2 (its z force is 0)");
  const levels = new Map<string, Set<number>>();
  for (let i = 0; i < all.count; i++) {
    const tip = [0, 1, 2].map((c) => all.arrows[i * ARROW_STRIDE + c]).join(",");
    levels.set(tip, (levels.get(tip) ?? new Set()).add(all.arrows[i * ARROW_STRIDE + ARROW.level]!));
  }
  for (const [tip, l] of levels) assert.equal(l.size, 1, `spring at ${tip}: one level for its components`);
  h.store.dispatch({ type: "setSprings", component: "y" });
  const y = buildSprings3D(h.store.get(), h.blobs, data.springs!);
  assert.equal(y.count, 2);
  for (let i = 0; i < 2; i++) assert.deepEqual([0, 1, 2].map((c) => y.arrows[i * ARROW_STRIDE + ARROW.dir + c]), [0, -1, 0], "the y force is -5: along -y");
  h.store.dispatch({ type: "setSprings", on: false, component: "resultant" });
});

test("noise is not a force, and a spring the model on screen has no node of is counted, not drawn", async () => {
  const s = h.store.get();
  const noisy = { nodes: Float64Array.from([1, 2, 3, 4]), force: Float64Array.from([10, 0, 0, 1e-9, 0, 0]) };
  assert.equal(buildSprings3D(s, h.blobs, noisy).count, 1, "1e-9 against 10 is solver noise");
  const away = { nodes: Float64Array.from([1, 2, 98, 99]), force: Float64Array.from([10, 0, 0, 5, 0, 0]) };
  const r = buildSprings3D(s, h.blobs, away);
  assert.equal(r.count, 1);
  assert.equal(r.unplaced, 1);
  // Node j missing but node i there: the arrow sits at node i (the two coincide).
  const iOnly = { nodes: Float64Array.from([3, 99]), force: Float64Array.from([0, 5, 0]) };
  const one = buildSprings3D(s, h.blobs, iOnly);
  assert.equal(one.count, 1);
  assert.ok(Math.abs(one.arrows[ARROW.tip]! - 2) < 1e-6, "at node 3");
});

test("the Springs layer: off by default, drawn when on, rebuilt at a new step, hidden by another stage's or a failed read", async () => {
  const layer = new SpringsLayer();
  try {
    const off = await atStep(0);
    assert.equal(layer.sync(h.store.get(), h.blobs, null, off), false, "off by default: nothing drawn");
    h.store.dispatch({ type: "setSprings", on: true, headSize: 16 });
    const data = await atStep(0);
    assert.equal(layer.sync(h.store.get(), h.blobs, null, data), true);
    assert.equal((layer.cones.geometry as THREE.InstancedBufferGeometry).instanceCount, 2);
    assert.equal(layer.uniforms.uHeadPx.value, 16, "its own head size");
    assert.equal(layer.sync(h.store.get(), h.blobs, null, data), false, "the same step draws nothing new");
    const next = await atStep(1);
    assert.equal(layer.sync(h.store.get(), h.blobs, null, next), true, "a new step rebuilds them");
    assert.equal(layer.root.visible, true);
    layer.sync(h.store.get(), h.blobs, null, { ...next, springs: null });
    assert.equal(layer.root.visible, false, "a failed springs read hides the last step's arrows");
    h.store.dispatch({ type: "setSprings", on: false });
    layer.sync(h.store.get(), h.blobs, null, next);
    assert.equal(layer.root.visible, false, "off again: hidden");
  } finally {
    layer.dispose();
  }
});

test("the Springs card shows on a stage with springs; its switch and component menu dispatch setSprings; Spotlight offers the toggle", async () => {
  await atStep(0);
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
    append(...xs: (El | string)[]) {
      this.children.push(...xs);
    }
    setAttribute() {}
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
    const box = renderSprings(h.store.get(), (e) => events.push(e)) as unknown as El | null;
    assert.ok(box, "the card shows: the stage records springs");
    const one = (cls: string) => {
      const found = box.all((e) => e.className.split(" ")[0] === cls);
      assert.equal(found.length, 1, cls);
      return found[0]!;
    };
    for (const fn of one("springs-toggle").listeners) fn();
    const menu = one("springs-component");
    assert.deepEqual(menu.all((e) => e !== menu).map((o) => o.value), ["resultant", "xyz", "x", "y", "z"]);
    menu.value = "xyz";
    for (const fn of menu.listeners) fn();
    assert.deepEqual(events, [{ type: "setSprings", on: true }, { type: "setSprings", component: "xyz" }]);
    const springsCmd = () => spotlightViewOf(reduce(h.store.get(), { type: "openSpotlight", query: "springs" })).items.find((i) => i.id === "cmd:toggle-springs");
    assert.deepEqual(springsCmd()?.events, [{ type: "setSprings", on: true }], "Spotlight offers the toggle");
    h.store.dispatch({ type: "setPhase", at: { kind: "mesh" } });
    assert.equal(renderSprings(h.store.get()), null, "no results step on screen: no card");
    assert.equal(springsCmd(), undefined, "nor the Spotlight toggle");
  } finally {
    (globalThis as { document?: unknown }).document = prevDoc;
  }
});

test("setSprings refuses a non-boolean switch, a scale or head at or below 0, an unknown menu value; an unchanged value changes nothing", () => {
  const s = h.store.get();
  for (const bad of [{ on: "yes" }, { scale: -1 }, { headSize: 0 }, { scale: Number.NaN }, { component: "w" }, { colorBy: "type" }, { thin: 1 }, { notice: 5 }]) {
    assert.throws(() => reduce(s, { type: "setSprings", ...(bad as object) } as never), RangeError, JSON.stringify(bad));
  }
  assert.equal(reduce(s, { type: "setSprings", component: s.springs.component }), s);
});

test("on the deformed shape the arrows sit at the drawn nodes, and the layer rebuilds when the drawn shape changes", async () => {
  h.store.dispatch({ type: "setSprings", on: true });
  const data = await atStep(1);
  const s = h.store.get();
  const moved = buildSprings3D(s, h.blobs, data.springs!, { deformedNodeCoords: new Map([[2, [0, 0, 1] as const], [4, [2, 0, 3] as const]]) });
  assert.deepEqual([0, 1].map((k) => moved.arrows[k * ARROW_STRIDE + 2]), [1, 3], "each tip at its node j's drawn place");
  const layer = new SpringsLayer();
  try {
    const base = basePositions(s.mesh!, h.blobs);
    assert.equal(layer.sync(s, h.blobs, base, data), true);
    const lifted = { ...base, line: base.line.map((v, i) => (i % 3 === 2 ? v + 5 : v)), key: "probe|scale 1" };
    assert.equal(layer.sync(s, h.blobs, lifted, data), true, "another drawn shape rebuilds the arrows");
    assert.equal(layer.sync(s, h.blobs, lifted, data), false, "the same one does not");
    const tipZ = (layer.lines.geometry as THREE.InstancedBufferGeometry).getAttribute("aTip")!.getZ(0);
    assert.equal(tipZ, 5, "the tip follows the drawn line vertices");
  } finally {
    layer.dispose();
    h.store.dispatch({ type: "setSprings", on: false });
  }
});

test("a failed springs read says why on the card, and the next good read clears it; turning the card off re-reads nothing", async () => {
  h.store.dispatch({ type: "setSprings", on: true });
  await atStep(0);
  const real = h.host.readStep.bind(h.host);
  h.host.readStep = (req) => {
    const r = real(req);
    return r.ok && req.springs ? { ...r, springs: null, springsError: "probe: DATA is short" } : r;
  };
  try {
    await atStep(1);
    await until(() => h.store.get().springs.notice === "probe: DATA is short", "the notice");
  } finally {
    h.host.readStep = real;
  }
  const prevDoc = (globalThis as { document?: unknown }).document;
  const texts: string[] = [];
  (globalThis as { document?: unknown }).document = {
    documentElement: { dataset: { theme: "dark" } },
    createElement: () => {
      const e = { className: "", style: {}, dataset: {}, title: "", value: "", children: [] as unknown[], append: (...xs: unknown[]) => e.children.push(...xs), addEventListener() {}, removeEventListener() {}, setAttribute() {} };
      Object.defineProperty(e, "textContent", { set: (v: string) => texts.push(v) });
      return e;
    },
  };
  try {
    renderSprings(h.store.get());
    assert.ok(texts.includes("probe: DATA is short"), "the card shows the notice");
  } finally {
    (globalThis as { document?: unknown }).document = prevDoc;
  }
  await atStep(0);
  await until(() => h.store.get().springs.notice === null, "a good read clears it");
  const fetched = h.effects.currentStep();
  assert.ok(fetched?.springs);
  h.store.dispatch({ type: "setSprings", on: false });
  await new Promise((r) => setTimeout(r, 30));
  assert.equal(h.effects.currentStep(), fetched, "off: the step held is kept, not fetched again");
});
