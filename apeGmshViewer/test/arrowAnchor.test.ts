// The arrow record's anchor (state/loads.ts `ARROW.anchor`, `ANCHOR`; renderer/loads.ts
// `ARROW_VERTEX`): a record's point is its arrow's tip (a load or a reaction, pointing at
// its node) or its tail (a nodal vector, starting at it). Every builder on main writes
// the tip; the shader and the CPU twin `expandArrows` move a tail-anchored tip by the
// drawn shaft length. No results file: the model is fixtures/loads3d.h5, with a results
// stage set in the state for the reaction records.

import assert from "node:assert/strict";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import type * as THREE from "three";
import { withNodes } from "../src/effects.ts";
import { openModel } from "../src/reader/node.ts";
import { ArrowLayer, arrowShaderSource, ReactionsLayer, SIZING_EXPR } from "../src/renderer/loads.ts";
import { BlobStore } from "../src/state/blobs.ts";
import { loadModel } from "../src/state/load.ts";
import { ANCHOR, ARROW, ARROW_STRIDE, buildLoads3D, buildReactions3D, buildSupports3D, expandArrows, THIN_LEVELS } from "../src/state/loads.ts";
import { initialState, reduce, type State } from "../src/state/store.ts";

const fixtures = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures");
const model = await openModel(join(fixtures, "loads3d.h5"));
const blobs = new BlobStore();
const loaded: State = reduce(initialState, { type: "fileLoaded", artifact: "model", load: withNodes(loadModel(model, blobs), model, blobs) });
const ids = blobs.get(loaded.mesh!.nodes!.ids) as Float64Array;
const RF = ["reaction_force_x", "reaction_force_y", "reaction_force_z"];
const near = (a: ArrayLike<number>, b: readonly number[], tol = 1e-6) => a.length === b.length && b.every((x, i) => Math.abs(a[i]! - x) <= tol);

/** One record at `at` along +x, `len` long, anchored as given. */
function record(at: readonly number[], len: number, anchor: number): Float32Array {
  const r = new Float32Array(ARROW_STRIDE);
  r.set(at, ARROW.tip);
  r.set([1, 0, 0], ARROW.dir);
  r[ARROW.len] = len;
  r.set([1, 1, 1], ARROW.rgb);
  r[ARROW.heads] = 1;
  r[ARROW.anchor] = anchor;
  return r;
}

test("the record ends with its anchor; tip is 0 and tail is 1", () => {
  assert.equal(ARROW_STRIDE, 15);
  assert.equal(ARROW.anchor, ARROW_STRIDE - 1);
  assert.deepEqual(ANCHOR, { tip: 0, tail: 1 });
});

test("every builder writes tip-anchored records: the loads and the reactions point at their nodes", () => {
  const loads = buildLoads3D(reduce(loaded, { type: "setLoads", on: true }), blobs);
  assert.ok(loads.arrows.length > 0, "the fixture has load arrows");
  const nodeIds = blobs.put("test/anchor/ids", ids);
  const stage = { id: "s0", name: "", kind: "static", time: [0], nodal: RF, refusedNodal: [], nodeIds, gauss: [], unread: [] };
  const s: State = {
    ...loaded,
    results: { path: "r.ladruno", version: "test", sourceType: null, lineage: null, partial: null, stages: [stage] } as never,
    phase: { ...loaded.phase, at: { kind: "results", stage: 0, step: 0 } },
    reactions: { ...loaded.reactions, on: true, allNodes: true },
  };
  const rows = Object.fromEntries(RF.map((n, c) => [n, new Float64Array(ids.length).fill(c + 1)]));
  const reactions = buildReactions3D(s, blobs, rows, ids);
  assert.ok(reactions.arrows.length > 0);
  for (const a of [loads.arrows, reactions.arrows]) {
    for (let o = 0; o < a.length; o += ARROW_STRIDE) assert.equal(a[o + ARROW.anchor], ANCHOR.tip);
  }
  // The layer hands the field to the shader.
  const layer = new ReactionsLayer();
  layer.sync(s, blobs, null, { stage: 0, step: 0, nodal: rows });
  const g = layer.lines.geometry as THREE.InstancedBufferGeometry;
  assert.equal(g.getAttribute("aAnchor").count, reactions.count);
  assert.equal(g.getAttribute("aAnchor").getX(0), ANCHOR.tip);
  layer.dispose();
});

test("the support glyphs' per-node thinning records follow the record layout", () => {
  // buildSupports3D thins one record per supported node; a literal one stride short would shift every
  // record after the first. Each glyph takes its node's level, so a level outside 0..THIN_LEVELS means a shifted read.
  const s = reduce(loaded, { type: "setSupports", on: true });
  const g = buildSupports3D(s, blobs);
  assert.ok(g.arrows.length + g.arcs.length > 0, "the fixture has supports");
  for (const a of [g.arrows, g.arcs]) {
    for (let o = 0; o < a.length; o += ARROW_STRIDE) {
      assert.ok(a[o + ARROW.level]! >= 1 && a[o + ARROW.level]! <= THIN_LEVELS, `level ${a[o + ARROW.level]}`);
      assert.equal(a[o + ARROW.anchor], ANCHOR.tip);
    }
  }
});

test("the CPU twin: a tip-anchored arrow ends at its point, a tail-anchored one starts there", () => {
  const none = new Float32Array(0);
  const tip = expandArrows({ arrows: record([5, 0, 0], 2, ANCHOR.tip), combs: none }, 0.5);
  assert.ok(near(tip.conePositions.subarray(0, 3), [5, 0, 0]), "the head's tip is the point");
  assert.ok(near(tip.linePositions.subarray(0, 3), [3, 0, 0]), "the shaft starts len behind it");
  const tail = expandArrows({ arrows: record([5, 0, 0], 2, ANCHOR.tail), combs: none }, 0.5);
  assert.ok(near(tail.linePositions.subarray(0, 3), [5, 0, 0]), "the shaft starts at the point");
  assert.ok(near(tail.conePositions.subarray(0, 3), [7, 0, 0]), "the head's tip is len ahead of it");
});

test("the shader moves a tail-anchored tip by the drawn shaft L, so SIZING_EXPR stays the one sizing source", () => {
  const src = arrowShaderSource();
  assert.match(src, /vec4 mvAt = modelViewMatrix \* vec4\( aTip, 1\.0 \);/);
  assert.match(src, /float L = shaftLength\( aLen, wpp \);/);
  assert.match(src, /vec4 mvTip = vec4\( mvAt\.xyz \+ dV \* \( aAnchor \* L \), 1\.0 \);/);
  assert.ok(src.includes(SIZING_EXPR.shaft), "shaftLength is SIZING_EXPR.shaft");
  // The pixel scale is measured at the record's point, not at the moved tip.
  assert.match(src, /float wpp = worldPerPixel\( mvAt \);/);
  // And the moved tip is what every vertex is placed from: assigned once, never reset to the point,
  // and the only base of the drawn position (a mutation to `mvTip = mvAt` or to `mvAt.xyz - ...` fails here).
  assert.equal(src.match(/\bmvTip\s*=/g)?.length, 1, "mvTip is assigned once");
  assert.match(src, /vec4 mvPosition = vec4\( mvTip\.xyz - dV \* \( aShape\.x \* L \+ aShape\.y \* H \) \+ across \* \( aShape\.z \* H \), 1\.0 \);/);
  assert.equal(src.match(/mvAt\.xyz/g)?.length, 1, "the point is used only to place the tip");
});

test("the layer uploads a tail-anchored record's anchor to the shader", () => {
  class Probe extends ArrowLayer {
    draw(a: Float32Array) {
      return this.upload(a, new Float32Array(0), 1);
    }
  }
  const layer = new Probe();
  assert.equal(layer.draw(new Float32Array([...record([0, 0, 0], 1, ANCHOR.tip), ...record([1, 0, 0], 1, ANCHOR.tail)])), true);
  for (const o of [layer.lines, layer.cones]) {
    const a = (o.geometry as THREE.InstancedBufferGeometry).getAttribute("aAnchor");
    assert.deepEqual([a.getX(0), a.getX(1)], [ANCHOR.tip, ANCHOR.tail]);
  }
  layer.dispose();
});
