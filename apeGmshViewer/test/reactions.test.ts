// Reaction-force arrows (state/loads.ts `buildReactions3D`, renderer/loads.ts):
// one arrow per held node of the step on screen, tip on the node and pointing
// along the reaction, marked with the 45° tail tick, drawn by the load arrows'
// layer. The set is test/spec/reactions.ts: contour_staged with
// reaction_force_{x,y,z} = (stage + 1)(k + 1)(c + 1)(1 + n mod 4) at the held nodes.

import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";
import type * as THREE from "three";
import { renderLoads } from "../src/panels/loads.ts";
import { LoadsLayer } from "../src/renderer/loads.ts";
import { ARROW, ARROW_STRIDE, BAR, buildReactions3D, reactionComponentsOf } from "../src/state/loads.ts";
import { opened, until } from "./contour/harness.ts";
import { makeReactionSet, reactionAt } from "./spec/reactions.ts";

const tmp = mkdtempSync(join(tmpdir(), "agv-reactions-"));
const set = await makeReactionSet(join(tmp, "set"));
const h = await opened(set.model, set.results);
after(() => {
  h.effects.dispose();
  h.host.close();
  rmSync(tmp, { recursive: true, force: true });
});

async function atStep(stage: number, step: number) {
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage, step } });
  const here = () => {
    const d = h.effects.currentStep();
    return d && d.stage === stage && d.step === step ? d : null;
  };
  await until(() => here() !== null, `step ${stage}:${step}`);
  return here()!;
}

test("one reaction arrow per held node, its tip on the node, pointing along the reaction, with the tail tick", async () => {
  const data = await atStep(2, 3);
  const s = h.store.get();
  assert.deepEqual(reactionComponentsOf(s), ["reaction_force_x", "reaction_force_y", "reaction_force_z"]);
  const ids = h.blobs.get(s.results!.stages[2]!.nodeIds!) as Float64Array;
  const r = buildReactions3D(s, h.blobs, data.nodal, ids);
  assert.equal(r.count, set.held.size, "every held node, and only those (NaN elsewhere)");
  assert.equal(r.arrows.length, set.held.size * ARROW_STRIDE);
  // At node n: R = 3 · 4 · (1, 2, 3) · (1 + n mod 4), so the direction is (1, 2, 3)/√14 at every node.
  const unit = [1, 2, 3].map((v) => v / Math.sqrt(14));
  const maxMag = 3 * 4 * Math.sqrt(14) * Math.max(...[...set.held].map((n) => 1 + (n % 4)));
  assert.ok(Math.abs(r.maxMag - maxMag) < 1e-9 * maxMag, `largest reaction ${r.maxMag}, expected ${maxMag}`);
  const xyz = new Map<number, number[]>();
  const nodeIds = h.blobs.get(s.mesh!.nodes!.ids) as Float64Array;
  const nodeXyz = h.blobs.get(s.mesh!.nodes!.xyz) as Float64Array;
  for (let i = 0; i < nodeIds.length; i++) xyz.set(nodeIds[i]!, [nodeXyz[3 * i]!, nodeXyz[3 * i + 1]!, nodeXyz[3 * i + 2]!]);
  const heldTips = new Set([...set.held].map((n) => xyz.get(n)!.map((v) => v.toFixed(5)).join(",")));
  for (let o = 0; o < r.arrows.length; o += ARROW_STRIDE) {
    const a = r.arrows;
    assert.ok(heldTips.has([0, 1, 2].map((c) => a[o + c]!.toFixed(5)).join(",")), "the tip is on a held node");
    for (let c = 0; c < 3; c++) assert.ok(Math.abs(a[o + ARROW.dir + c]! - unit[c]!) < 1e-6, "pointing along the reaction");
    assert.equal(a[o + ARROW.bar], BAR.tick);
    assert.equal(a[o + ARROW.heads], 1);
  }
  assert.equal(reactionAt(2, 3, 0, 0), 12, "the closed form the fixture writes");
});

test("the layer draws reactions with the step: on with the switch, alone or with the loads, and again at a new step", async () => {
  const data = await atStep(2, 1);
  const layer = new LoadsLayer();
  let s = h.store.get();
  assert.equal(layer.sync(s, h.blobs, null, data), false, "off by default: nothing drawn");
  h.store.dispatch({ type: "setLoads", showReactions: true });
  s = h.store.get();
  assert.equal(layer.sync(s, h.blobs, null, data), true);
  const count = () => (layer.cones.geometry as THREE.InstancedBufferGeometry).instanceCount;
  assert.equal(count(), set.held.size, "the reactions alone, the loads switch being off");
  assert.equal(layer.sync(s, h.blobs, null, data), false, "the same step draws nothing new");
  const next = await atStep(2, 2);
  assert.equal(layer.sync(h.store.get(), h.blobs, null, next), true, "a new step rebuilds them");
  assert.equal(count(), set.held.size);
  h.store.dispatch({ type: "setLoads", showReactions: false });
  assert.equal(layer.sync(h.store.get(), h.blobs, null, next), true);
  assert.equal(layer.root.visible, false, "off again: hidden");
  layer.dispose();
});

test("the Loads card offers the reactions switch when the step records reactions, and it dispatches showReactions", async () => {
  await atStep(2, 0);
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
    const box = renderLoads(h.store.get(), (e) => events.push(e)) as unknown as El | null;
    assert.ok(box, "the card shows: the step records reactions");
    const btn = box.all((e) => e.className.startsWith("loads-reactions"));
    assert.equal(btn.length, 1, "one reactions button");
    for (const fn of btn[0]!.listeners) fn();
    assert.deepEqual(events, [{ type: "setLoads", showReactions: true }]);
  } finally {
    (globalThis as { document?: unknown }).document = prevDoc;
  }
});
