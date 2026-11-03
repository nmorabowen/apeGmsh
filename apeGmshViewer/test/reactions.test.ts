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
import { renderReactions } from "../src/panels/reactions.ts";
import { LoadsLayer, ReactionsLayer } from "../src/renderer/loads.ts";
import { ARROW, ARROW_STRIDE, BAR, buildReactions3D, reactionComponentsOf, supportedNodesOf } from "../src/state/loads.ts";
import { opened, until } from "./contour/harness.ts";
import { makeReactionSet, reactionAt } from "./spec/reactions.ts";
import { reduce } from "../src/state/store.ts";

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

test("components along the global axes: three arrows per node for xyz, one for x, y or z, each along ±its axis", async () => {
  const data = await atStep(2, 3);
  const ids = () => h.blobs.get(h.store.get().results!.stages[2]!.nodeIds!) as Float64Array;
  const build = () => buildReactions3D(h.store.get(), h.blobs, data.nodal, ids());
  const dirs = (r: ReturnType<typeof build>) => new Set(Array.from({ length: r.count }, (_, i) => [0, 1, 2].map((c) => r.arrows[i * ARROW_STRIDE + ARROW.dir + c]).join(",")));
  h.store.dispatch({ type: "setReactions", component: "xyz" });
  const all = build();
  assert.equal(all.count, 3 * set.held.size, "three arrows at each held node");
  assert.deepEqual([...dirs(all)].sort(), ["0,0,1", "0,1,0", "1,0,0"]);
  // The longest arrow is the largest component, z: 3 · 4 · 3 · (1 + n mod 4).
  const maxZ = 3 * 4 * 3 * Math.max(...[...set.held].map((n) => 1 + (n % 4)));
  assert.ok(Math.abs(all.maxMag - maxZ) < 1e-9 * maxZ, `largest arrow ${all.maxMag}, expected ${maxZ}`);
  // Thinning is per node: the three arrows sharing a tip share a level, so the smaller components are never thinned by their own node.
  const levelAt = new Map<string, Set<number>>();
  for (let i = 0; i < all.count; i++) {
    const tip = [0, 1, 2].map((c) => all.arrows[i * ARROW_STRIDE + c]).join(",");
    levelAt.set(tip, (levelAt.get(tip) ?? new Set()).add(all.arrows[i * ARROW_STRIDE + ARROW.level]!));
  }
  assert.equal(levelAt.size, set.held.size);
  for (const [tip, levels] of levelAt) assert.equal(levels.size, 1, `node at ${tip}: its components have levels ${[...levels]}`);
  for (const [c, d] of [["x", "1,0,0"], ["y", "0,1,0"], ["z", "0,0,1"]] as const) {
    h.store.dispatch({ type: "setReactions", component: c });
    const one = build();
    assert.equal(one.count, set.held.size, `${c}: one arrow per held node`);
    assert.deepEqual([...dirs(one)], [d]);
  }
  h.store.dispatch({ type: "setReactions", component: "resultant" });
});

test("the Reactions layer: its own switch, independent of the loads, rebuilt at a new step", async () => {
  const data = await atStep(2, 1);
  const layer = new ReactionsLayer();
  const loads = new LoadsLayer();
  assert.equal(h.store.get().reactions.on, false);
  assert.equal(layer.sync(h.store.get(), h.blobs, null, data), false, "off by default: nothing drawn");
  h.store.dispatch({ type: "setReactions", on: true, headSize: 16 });
  const s = h.store.get();
  assert.equal(s.loads.on, false);
  assert.equal(layer.sync(s, h.blobs, null, data), true);
  assert.equal(loads.sync(s, h.blobs), false, "the loads layer stays off");
  const count = () => (layer.cones.geometry as THREE.InstancedBufferGeometry).instanceCount;
  assert.equal(count(), set.held.size);
  assert.equal(layer.uniforms.uHeadPx.value, 16, "its own head size");
  assert.equal(layer.sync(s, h.blobs, null, data), false, "the same step draws nothing new");
  const next = await atStep(2, 2);
  assert.equal(layer.sync(h.store.get(), h.blobs, null, next), true, "a new step rebuilds them");
  h.store.dispatch({ type: "setReactions", on: false });
  assert.equal(layer.sync(h.store.get(), h.blobs, null, next), true);
  assert.equal(layer.root.visible, false, "off again: hidden");
  layer.dispose();
  loads.dispose();
});

test("the Reactions card shows when the step records reactions; its switch and component menu dispatch setReactions", async () => {
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
    const box = renderReactions(h.store.get(), (e) => events.push(e)) as unknown as El | null;
    assert.ok(box, "the card shows: the step records reactions");
    const one = (cls: string) => {
      const found = box.all((e) => e.className.split(" ")[0] === cls);
      assert.equal(found.length, 1, cls);
      return found[0]!;
    };
    for (const fn of one("reactions-toggle").listeners) fn();
    const menu = one("reactions-component");
    assert.deepEqual(menu.all((e) => e !== menu).map((o) => o.value), ["resultant", "xyz", "x", "y", "z"]);
    menu.value = "xyz";
    for (const fn of menu.listeners) fn();
    assert.deepEqual(events, [{ type: "setReactions", on: true }, { type: "setReactions", component: "xyz" }]);
    h.store.dispatch({ type: "setPhase", at: { kind: "mesh" } });
    assert.equal(renderReactions(h.store.get()), null, "no results step on screen: no card");
  } finally {
    (globalThis as { document?: unknown }).document = prevDoc;
  }
});

test("supports only by default: a free node's residual or a constraint's force draws no arrow unless 'all nodes' is on", async () => {
  // The model's supports are the sp_holds of stage 0: exactly the held nodes.
  assert.deepEqual([...supportedNodesOf(h.store.get())].sort((a, b) => a - b), [...set.held].sort((a, b) => a - b));
  // A second set that also records a value at every free node: a tiny residual, and a large "constraint force" at one.
  const loud = Math.max(...[...set.held]) + 1;
  assert.ok(!set.held.has(loud), "the loud node is a free one");
  const noisy = await makeReactionSet(join(tmp, "noisy"), (n) => (n === loud ? 500 : 1e-9));
  const g = await opened(noisy.model, noisy.results);
  try {
    g.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 2, step: 3 } });
    await until(() => g.effects.currentStep()?.step === 3, "the noisy step");
    const data = g.effects.currentStep()!;
    const ids = g.blobs.get(g.store.get().results!.stages[2]!.nodeIds!) as Float64Array;
    const supportsOnly = buildReactions3D(g.store.get(), g.blobs, data.nodal, ids);
    assert.equal(supportsOnly.count, noisy.held.size, "only the supports");
    g.store.dispatch({ type: "setReactions", allNodes: true });
    const all = buildReactions3D(g.store.get(), g.blobs, data.nodal, ids);
    // 'all nodes' adds the free node's large constraint force; the 1e-9 residuals stay under the noise floor.
    assert.equal(all.count, noisy.held.size + 1, "the supports, plus the one large free-node force");
  } finally {
    g.effects.dispose();
    g.host.close();
  }
});

test("the supports of a model with fix rows: a node fixed in some DOF, not one whose mask is all zero, whatever the mask's form", () => {
  const s = h.store.get();
  const free = (h.blobs.get(s.mesh!.nodes!.ids) as Float64Array).filter((n) => !set.held.has(n));
  const [a, b, c, d] = [free[0]!, free[1]!, free[2]!, free[3]!];
  const withFix = (rows: unknown[][]) =>
    supportedNodesOf({
      ...s,
      definitions: { ...s.definitions!, bcs: { fix: { path: "/opensees/bcs/fix", columns: ["target_kind", "target", "dofs"], rows: rows as never }, mass: null } },
    });
  const nodes = withFix([["node", String(a), [1, 0, 0]], ["node", String(b), [0, 0, 0]], ["node", String(c), new Float64Array([0, 0, 0])], ["node", String(d), "0 1 0"]]);
  assert.ok(nodes.has(a), "fixed in x");
  assert.ok(!nodes.has(b), "an all-zero mask fixes nothing");
  assert.ok(!nodes.has(c), "nor does an all-zero typed-array mask");
  assert.ok(nodes.has(d), "a string mask fixed in y");
});

test("solver noise is not a reaction: a component under 1e-6 of the step's largest draws nothing", async () => {
  const data = await atStep(2, 3);
  const s = h.store.get();
  const ids = h.blobs.get(s.results!.stages[2]!.nodeIds!) as Float64Array;
  const noisy = { ...data.nodal, reaction_force_x: data.nodal["reaction_force_x"]!.map((v) => (Number.isFinite(v) ? 1e-12 : v)) };
  h.store.dispatch({ type: "setReactions", component: "xyz" });
  assert.equal(buildReactions3D(h.store.get(), h.blobs, noisy, ids).count, 2 * set.held.size, "y and z at every held node; no x");
  h.store.dispatch({ type: "setReactions", component: "resultant" });
});

test("supports follow the stage: a stage's remove_sp releases a node from that stage on", async () => {
  await atStep(2, 3);
  const s = h.store.get();
  const node = [...set.held][0]!;
  const stages = s.definitions!.stages.map((st, k) =>
    k === 1 ? { ...st, removeSp: { path: "/opensees/stages/stage_001/remove_sp", columns: ["node_tag", "dof"], rows: [[node, 1], [node, 2], [node, 3]] } } : st,
  );
  const released = { ...s, definitions: { ...s.definitions!, stages } };
  const name = (k: number) => s.results!.stages[k]!.name;
  assert.equal(s.definitions!.stages[1]!.name, name(1), "the model's stage 1 is the results' stage 1");
  assert.ok(!supportedNodesOf(released).has(node), `released at ${name(1)}, so not a support at ${name(2)}`);
  const atBase = { ...released, phase: { ...released.phase, at: { kind: "results" as const, stage: 0, step: 0 } } };
  assert.ok(supportedNodesOf(atBase).has(node), `still a support at ${name(0)}, before the release`);
});

test("another stage's arrows never stay: a stage change hides them until its step arrives, and so does a failed read", async () => {
  const data = await atStep(2, 3);
  h.store.dispatch({ type: "setReactions", on: true });
  const layer = new ReactionsLayer();
  assert.equal(layer.sync(h.store.get(), h.blobs, null, data), true);
  assert.equal(layer.root.visible, true);
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 1, step: 0 } });
  layer.sync(h.store.get(), h.blobs, null, null);
  assert.equal(layer.root.visible, false, "stage 1 shows nothing of stage 2");
  const back = await atStep(2, 3);
  layer.sync(h.store.get(), h.blobs, null, back);
  assert.equal(layer.root.visible, true);
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 2, step: 1 } });
  h.store.dispatch({ type: "resultStep", status: { stage: 2, step: 1, status: "failed", error: "probe" } });
  layer.sync(h.store.get(), h.blobs, null, null);
  assert.equal(layer.root.visible, false, "a failed read hides the last step's arrows");
  h.store.dispatch({ type: "setReactions", on: false });
  layer.dispose();
});

test("a single axis the stage does not record draws the resultant, as the card shows", async () => {
  const data = await atStep(2, 3);
  const s = h.store.get();
  const ids = h.blobs.get(s.results!.stages[2]!.nodeIds!) as Float64Array;
  const stages = s.results!.stages.map((st, k) => (k === 2 ? { ...st, nodal: st.nodal.filter((c) => c !== "reaction_force_z") } : st));
  const noZ = { ...s, results: { ...s.results!, stages }, reactions: { ...s.reactions, component: "z" as const } };
  const r = buildReactions3D(noZ, h.blobs, data.nodal, ids);
  assert.equal(r.count, set.held.size, "one resultant arrow per held node");
  assert.ok(Math.abs(r.arrows[ARROW.dir + 2]!) < 1e-6, "in the x-y plane: the z rows are not read");
});

test("setReactions refuses what setLoads refuses: a non-boolean switch, a scale or head at or below 0, an unknown menu value", () => {
  const s = h.store.get();
  for (const bad of [{ on: "yes" }, { scale: -1 }, { headSize: 0 }, { component: "w" }, { colorBy: "rainbow" }, { allNodes: 1 }]) {
    assert.throws(() => reduce(s, { type: "setReactions", ...(bad as object) } as never), RangeError, JSON.stringify(bad));
  }
  assert.equal(reduce(s, { type: "setReactions", component: s.reactions.component }), s, "an unchanged value changes nothing");
});
