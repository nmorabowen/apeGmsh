// Reaction moments (state/loads.ts `buildReactions3D` arcs, renderer/loads.ts `ArcGlyphs`):
// a half circle about each recorded moment's axis, the resultant or its
// global-axis components, with the reaction tick; the card's F and M switches.
// No results file: the model is fixtures/loads3d.h5, and the results stage is
// set in the state (its nodal names and node ids) with the step's rows passed
// in, which is all the builder and the layer read (the #2079 plan retires the
// native results reader, so no fixture targets it here).

import assert from "node:assert/strict";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import type * as THREE from "three";
import { withNodes } from "../src/effects.ts";
import { renderReactions } from "../src/panels/reactions.ts";
import { openModel } from "../src/reader/node.ts";
import { ReactionsLayer } from "../src/renderer/loads.ts";
import { BlobStore } from "../src/state/blobs.ts";
import { loadModel } from "../src/state/load.ts";
import { ARROW, ARROW_STRIDE, BAR, buildReactions3D, reactionMomentComponentsOf } from "../src/state/loads.ts";
import { initialState, reduce, type State } from "../src/state/store.ts";

const fixtures = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures");
const model = await openModel(join(fixtures, "loads3d.h5"));
const blobs = new BlobStore();
const loaded: State = reduce(initialState, { type: "fileLoaded", artifact: "model", load: withNodes(loadModel(model, blobs), model, blobs) });
const ids = blobs.get(loaded.mesh!.nodes!.ids) as Float64Array;
const N = ids.length;
const NAMES = ["reaction_force_x", "reaction_force_y", "reaction_force_z", "reaction_moment_x", "reaction_moment_y", "reaction_moment_z"];

/** The model with one results stage recording `names`, at step 0, the Reactions card on for every node. */
function withStage(names: readonly string[], reactions: Partial<State["reactions"]> = {}): State {
  const nodeIds = blobs.put("test/reactions/ids", ids);
  const stage = { id: "s0", name: "", kind: "static", time: [0], nodal: [...names], refusedNodal: [], nodeIds, gauss: [], unread: [] };
  return {
    ...loaded,
    results: { path: "r.ladruno", version: "test", sourceType: null, lineage: null, partial: null, stages: [stage] } as never,
    phase: { ...loaded.phase, at: { kind: "results", stage: 0, step: 0 } },
    reactions: { ...loaded.reactions, on: true, allNodes: true, ...reactions },
  };
}

/** Rows at the first `k` nodes: force (0, 0, 10) and moment (0, m·(1 + i), 0) at node i, NaN elsewhere. */
function rows(k: number, m = 3): Record<string, Float64Array> {
  const out: Record<string, Float64Array> = {};
  for (const [c, name] of NAMES.entries()) {
    const a = new Float64Array(N).fill(Number.NaN);
    for (let i = 0; i < k; i++) a[i] = c === 2 ? 10 : c === 4 ? m * (1 + i) : 0;
    out[name] = a;
  }
  return out;
}

test("a recorded moment draws one arc per node about its axis, marked with the reaction tick; forces stay arrows", () => {
  const s = withStage(NAMES);
  assert.deepEqual(reactionMomentComponentsOf(s), ["reaction_moment_x", "reaction_moment_y", "reaction_moment_z"]);
  const r = buildReactions3D(s, blobs, rows(4), ids);
  assert.equal(r.count, 4, "one force arrow per node");
  assert.equal(r.arcCount, 4, "one moment arc per node");
  assert.equal(r.maxMoment, 12, "the largest moment, 3 × 4, sets the largest arc");
  for (let o = 0; o < r.arcs.length; o += ARROW_STRIDE) {
    assert.deepEqual([0, 1, 2].map((c) => r.arcs[o + ARROW.dir + c]), [0, 1, 0], "the arc turns about +Y (right-hand rule)");
    assert.equal(r.arcs[o + ARROW.bar], BAR.tick);
  }
  const lens = Array.from({ length: r.arcCount }, (_, i) => r.arcs[i * ARROW_STRIDE + ARROW.len]!);
  assert.ok(lens.every((l, i) => i === 0 || l > lens[i - 1]!), "a larger moment, a larger arc");
});

test("a negative moment turns the other way: its axis flips", () => {
  const data = rows(1);
  data["reaction_moment_y"]![0] = -5;
  const r = buildReactions3D(withStage(NAMES), blobs, data, ids);
  assert.deepEqual([0, 1, 2].map((c) => r.arcs[ARROW.dir + c]), [0, -1, 0]);
});

test("F and M switch each kind; a stage recording only moments still draws its arcs", () => {
  assert.equal(buildReactions3D(withStage(NAMES, { forces: false }), blobs, rows(3), ids).count, 0);
  assert.equal(buildReactions3D(withStage(NAMES, { forces: false }), blobs, rows(3), ids).arcCount, 3);
  assert.equal(buildReactions3D(withStage(NAMES, { moments: false }), blobs, rows(3), ids).arcCount, 0);
  const momentsOnly = NAMES.filter((n) => n.startsWith("reaction_moment"));
  const r = buildReactions3D(withStage(momentsOnly), blobs, rows(3), ids);
  assert.deepEqual([r.count, r.arcCount], [0, 3]);
});

test("components: xyz draws an arc per recorded non-zero axis; noise under 1e-6 of the largest draws none", () => {
  const data = rows(2);
  data["reaction_moment_x"]!.set([2, 2]);
  data["reaction_moment_z"]!.set([1e-12, 1e-12]);
  const r = buildReactions3D(withStage(NAMES, { component: "xyz" }), blobs, data, ids);
  const axes = new Set(Array.from({ length: r.arcCount }, (_, i) => [0, 1, 2].map((c) => r.arcs[i * ARROW_STRIDE + ARROW.dir + c]).join(",")));
  assert.deepEqual([...axes].sort(), ["0,1,0", "1,0,0"], "x and y arcs; the 1e-12 z is noise");
  assert.equal(r.arcCount, 4);
});

test("the layer draws the arcs on the card's root, with no force arrows when only moments are on", () => {
  const s = withStage(NAMES, { forces: false });
  const layer = new ReactionsLayer();
  const step = { stage: 0, step: 0, nodal: rows(3) };
  assert.equal(layer.sync(s, blobs, null, step), true);
  assert.equal(layer.root.visible, true, "the root shows for the arcs");
  assert.equal((layer.arcs.lines.geometry as THREE.InstancedBufferGeometry).instanceCount, 3);
  assert.equal(layer.cones.visible, false, "no force heads drawn");
  assert.equal(layer.lines.visible, false, "no force shafts drawn");
  layer.dispose();
});

test("the card offers M when the stage records moments, and F only when it records forces", () => {
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
    get text(): string {
      return this.children.map((c) => (typeof c === "string" ? c : c.text)).join("");
    }
    all(pred: (e: El) => boolean): El[] {
      return [...(pred(this) ? [this] : []), ...this.children.flatMap((c) => (typeof c === "string" ? [] : c.all(pred)))];
    }
  }
  (globalThis as { document?: unknown }).document = { documentElement: { dataset: { theme: "dark" } }, createElement: () => new El(), createElementNS: () => new El() };
  try {
    const kinds = (s: State) => {
      const box = renderReactions(s) as unknown as El | null;
      assert.ok(box, "the card shows");
      return box.all((e) => /^reactions-(forces|moments)/.test(e.className)).map((e) => e.text);
    };
    assert.deepEqual(kinds(withStage(NAMES)), ["F", "M"]);
    assert.deepEqual(kinds(withStage(NAMES.filter((n) => n.startsWith("reaction_moment")))), ["M"], "a moments-only stage shows the card, with M only");
    const events: unknown[] = [];
    const box = renderReactions(withStage(NAMES), (e) => events.push(e)) as unknown as El;
    for (const fn of box.all((e) => e.className.startsWith("reactions-moments"))[0]!.listeners) fn();
    assert.deepEqual(events, [{ type: "setReactions", moments: false }]);
  } finally {
    (globalThis as { document?: unknown }).document = prevDoc;
  }
});

test("each glyph shader compiles to its own program: arrows, comb bars and arcs have distinct three.js cache keys", () => {
  // three caches a program by customProgramCacheKey(), by default onBeforeCompile.toString(), which is the
  // same text for every material instanced() patches: the arcs and the comb bars were drawn by the arrows' program.
  const layer = new ReactionsLayer();
  const key = (o: { material: THREE.Material | THREE.Material[] }) => (o.material as THREE.Material).customProgramCacheKey();
  const keys = [key(layer.lines), key(layer.combs), key(layer.arcs.lines)];
  assert.equal(new Set(keys).size, 3, "three line materials, three programs");
  assert.notEqual(key(layer.cones), key(layer.arcs.heads), "arrow heads and arc heads differ too");
  assert.equal(key(layer.lines), key(layer.cones), "shafts and heads share the arrow program by design");
  layer.dispose();
});
