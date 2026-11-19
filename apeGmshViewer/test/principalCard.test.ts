// The Principal card end to end over fixtures/ladruno_quad2d.ladruno: its
// reducer, the step fetch of the tensor components (effects), the layer
// (renderer/loads.ts `PrincipalLayer`) and the card (panels/principal.ts).

import assert from "node:assert/strict";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import type * as THREE from "three";
import { renderPrincipal } from "../src/panels/principal.ts";
import { PrincipalLayer } from "../src/renderer/loads.ts";
import { ARROW, ARROW_STRIDE } from "../src/state/loads.ts";
import { initialPrincipal } from "../src/state/reduce.ts";
import { principalFieldOf, principalFieldsOf } from "../src/state/selectors.ts";
import { initialState, reduce } from "../src/state/store.ts";
import { opened, until } from "./contour/harness.ts";

const fixture = (name: string) => fileURLToPath(new URL(`../fixtures/${name}`, import.meta.url));
const h = await opened(fixture("ladruno_quad2d.h5"), fixture("ladruno_quad2d.ladruno"));
after(() => {
  h.effects.dispose();
  h.host.close();
});

async function atStep(step: number, withTensor: boolean) {
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step } });
  const here = () => {
    const d = h.effects.currentStep();
    return d && d.stage === 0 && d.step === step && (!withTensor || d.tensor) ? d : null;
  };
  await until(() => here() !== null, `step ${step}`);
  return here()!;
}

test("setPrincipal: defaults, a change of one setting, and refusals of a bad value", () => {
  assert.deepEqual(initialState.principal, initialPrincipal);
  assert.deepEqual(initialPrincipal, { on: false, field: null, show: [true, true, true], head: "axis", cmap: "RdBu_r", scale: 1, headSize: 10, thin: true });
  const s = reduce(initialState, { type: "setPrincipal", on: true, show: [true, false, false], head: "arrow" });
  assert.deepEqual([s.principal.on, s.principal.show, s.principal.head], [true, [true, false, false], "arrow"]);
  assert.equal(reduce(s, { type: "setPrincipal", show: [true, false, false] }), s, "the same show: no change");
  for (const bad of [{ scale: 0 }, { headSize: -1 }, { show: [true, true] }, { head: "both" }, { cmap: "rainbow" }, { on: "yes" }, { field: 3 }]) {
    assert.throws(() => reduce(initialState, { type: "setPrincipal", ...(bad as object) } as never), RangeError, JSON.stringify(bad));
  }
});

test("the stage's fields: stress and strain; the step carries their components only while the card is on", async () => {
  await atStep(1, false);
  assert.deepEqual(principalFieldsOf(h.store.get()).map((f) => [f.name, f.family, f.components]), [
    ["stress", "stress", ["stress/sigma11", "stress/sigma22", "stress/sigma12"]],
    ["strain", "strain", ["strain/eta11", "strain/eta22", "strain/eta12"]],
  ]);
  assert.equal(h.effects.currentStep()!.tensor, undefined, "off: no tensor read");
  h.store.dispatch({ type: "setPrincipal", on: true });
  const d = await atStep(1, true);
  assert.deepEqual(Object.keys(d.tensor!), ["stress/sigma11", "stress/sigma22", "stress/sigma12"]);
  h.store.dispatch({ type: "setPrincipal", field: "strain" });
  assert.equal(principalFieldOf(h.store.get())!.name, "strain");
  await until(() => !!h.effects.currentStep()?.tensor?.["strain/eta12"], "the strain read");
  h.store.dispatch({ type: "setPrincipal", on: false, field: null });
});

test("the layer: tail-anchored arrows at the 24 Gauss points, two per in-plane principal; a scrub keeps them, off hides them", async () => {
  h.store.dispatch({ type: "setPrincipal", on: true, head: "axis" });
  const d = await atStep(1, true);
  const layer = new PrincipalLayer();
  try {
    assert.equal(layer.sync(h.store.get(), h.blobs, null, d), true);
    const g = layer.cones.geometry as THREE.InstancedBufferGeometry;
    // Plane stress: zz = 0 is one principal at every point, below the noise floor; the two in-plane ones draw.
    assert.equal(g.instanceCount, 24 * 2 * 2);
    assert.equal(layer.drawnPoints, 24);
    const anchor = g.getAttribute("aAnchor");
    assert.ok(anchor, "the shader reads the anchor");
    for (let i = 0; i < g.instanceCount; i++) assert.equal(anchor.getX(i), 1);
    assert.equal(layer.sync(h.store.get(), h.blobs, null, d), false, "the same step draws nothing new");
    h.store.dispatch({ type: "setPrincipal", head: "arrow", show: [true, false, false] });
    assert.equal(layer.sync(h.store.get(), h.blobs, null, d), true);
    assert.ok(g !== layer.cones.geometry);
    const n = (layer.cones.geometry as THREE.InstancedBufferGeometry).instanceCount;
    assert.ok(n > 0 && n <= 24, `${n}: at most one p1 arrow per point`);
    // The next step is not here yet: the last arrows stay.
    h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 0 } });
    layer.sync(h.store.get(), h.blobs, null, null);
    assert.equal(layer.root.visible, true);
    h.store.dispatch({ type: "setPrincipal", on: false });
    layer.sync(h.store.get(), h.blobs, null, d);
    assert.equal(layer.root.visible, false);
    assert.equal(layer.drawnPoints, 0);
  } finally {
    layer.dispose();
    h.store.dispatch({ type: "setPrincipal", head: "axis", show: [true, true, true] });
  }
});

test("picking a field that cannot draw hides the last field's arrows; the default prefers a stress", async () => {
  h.store.dispatch({ type: "setPrincipal", on: true, field: null });
  const d = await atStep(1, true);
  const layer = new PrincipalLayer();
  try {
    assert.equal(layer.sync(h.store.get(), h.blobs, null, d), true);
    assert.equal(layer.root.visible, true);
    // The same stage with one more RESULT whose only group lacks yy and xy: a field kept for its refusal.
    const s0 = h.store.get();
    const stage = s0.results!.stages[0]!;
    const bad = { ...stage.gauss[0]!, id: "bad/31-FourNodeQuad[201:0:0]", components: ["bad/sigma11"] };
    const results = { ...s0.results!, stages: [{ ...stage, gauss: [bad, ...stage.gauss] }] };
    const withBad = { ...s0, results };
    assert.equal(principalFieldOf(withBad)!.name, "stress", "the default skips a field that cannot draw, and prefers a stress");
    const picked = { ...withBad, principal: { ...withBad.principal, field: "bad" } };
    assert.deepEqual(principalFieldOf(picked)!.groups, []);
    // The effects ask for no tensor for it: the step comes back without one.
    layer.sync(picked, h.blobs, null, { ...d, tensor: undefined });
    assert.equal(layer.root.visible, false, "no stress arrows left under a field that draws nothing");
    assert.equal(layer.drawnPoints, 0);
  } finally {
    layer.dispose();
    h.store.dispatch({ type: "setPrincipal", on: false });
  }
});

test("a deformed step moves the arrows' tails with the Gauss points", async () => {
  h.store.dispatch({ type: "setPrincipal", on: true, head: "arrow" });
  const d = await atStep(1, true);
  const layer = new PrincipalLayer();
  try {
    layer.sync(h.store.get(), h.blobs, null, d);
    const tails = (l: PrincipalLayer) => {
      const buf = ((l.cones.geometry as THREE.InstancedBufferGeometry).getAttribute("aTip") as THREE.InterleavedBufferAttribute).data.array as Float32Array;
      return Array.from({ length: buf.length / ARROW_STRIDE }, (_, i) => buf[i * ARROW_STRIDE + ARROW.tip]!);
    };
    const before = tails(layer);
    // A FullPositions stand-in: every node moved +1 in x, at scale 2.
    const ids = h.blobs.get(h.store.get().mesh!.nodes!.ids) as Float64Array;
    const column = new Map([...ids].map((id, i) => [id, i]));
    const one = new Float64Array(ids.length).fill(1), zero = new Float64Array(ids.length);
    h.store.dispatch({ type: "setDeformScale", scale: 2 } as never);
    const full = { key: "probe", disp: { nodal: { column, rows: [one, zero, null] } } } as never;
    layer.sync(h.store.get(), h.blobs, full, d);
    const after_ = tails(layer);
    const k = h.store.get().deform.scale;
    after_.forEach((x, i) => assert.ok(Math.abs(x - (before[i]! + k)) < 1e-5, `${x} vs ${before[i]} + ${k}`));
  } finally {
    layer.dispose();
    h.store.dispatch({ type: "setPrincipal", on: false, head: "axis" });
  }
});

test("the card shows on a stage with a tensor; its switch, p buttons and field menu dispatch setPrincipal", async () => {
  await atStep(1, false);
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
    setCustomValidity() {}
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
    const box = renderPrincipal(h.store.get(), (e) => events.push(e)) as unknown as El | null;
    assert.ok(box, "the card shows: the stage has stress and strain tensors");
    const one = (cls: string) => {
      const found = box.all((e) => e.className.split(" ")[0] === cls);
      assert.equal(found.length, 1, cls);
      return found[0]!;
    };
    for (const fn of one("principal-toggle").listeners) fn();
    for (const fn of one("principal-p2").listeners) fn();
    const menu = one("principal-field");
    assert.deepEqual(menu.all((e) => e !== menu).map((o) => o.value), ["stress", "strain"]);
    menu.value = "strain";
    for (const fn of menu.listeners) fn();
    assert.deepEqual(events, [{ type: "setPrincipal", on: true }, { type: "setPrincipal", show: [true, false, true] }, { type: "setPrincipal", field: "strain" }]);
    const text = (e: El): string => e.children.map((c) => (typeof c === "string" ? c : text(c))).join(" ");
    assert.match(text(box), /zz not recorded: taken as 0 \(plane stress\)/);
    h.store.dispatch({ type: "setPhase", at: { kind: "mesh" } });
    assert.equal(renderPrincipal(h.store.get()), null, "no results step on screen: no card");
  } finally {
    (globalThis as { document?: unknown }).document = prevDoc;
  }
});
