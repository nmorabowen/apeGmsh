// Nodal vector glyphs (state/vectors.ts `buildVectors3D`, renderer/loads.ts
// `VectorsLayer`, panels/vectors.ts): arrows starting AT each node and pointing
// along the step's vector (tail-anchored records), half circles for a rotational
// field, the noise floor and per-node thinning, the card, and setVectors. No
// native results file (the "all in ladruno" ruling): the unit tests set a
// `.ladruno`-named stage in the state of fixtures/loads3d.h5 and pass the step's
// rows in; the end-to-end test opens the fork-written ladruno_frame3d pair.

import assert from "node:assert/strict";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import type * as THREE from "three";
import { withNodes } from "../src/effects.ts";
import { renderVectors } from "../src/panels/vectors.ts";
import { openModel } from "../src/reader/node.ts";
import { arrowShaderSource, ReactionsLayer, VectorsLayer } from "../src/renderer/loads.ts";
import { BlobStore } from "../src/state/blobs.ts";
import { sampleCmap } from "../src/state/contour.ts";
import { loadModel } from "../src/state/load.ts";
import { ANCHOR, ARROW, ARROW_STRIDE, expandArrows, loadsColourPalette } from "../src/state/loads.ts";
import { initialState, reduce, type State } from "../src/state/store.ts";
import { buildVectors3D, VECTOR_FLOOR, vectorFieldOf, vectorFieldsOf } from "../src/state/vectors.ts";
import { opened, until } from "./contour/harness.ts";

const fixtures = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures");
const model = await openModel(join(fixtures, "loads3d.h5"));
const blobs = new BlobStore();
const loaded: State = reduce(initialState, { type: "fileLoaded", artifact: "model", load: withNodes(loadModel(model, blobs), model, blobs) });
const ids = blobs.get(loaded.mesh!.nodes!.ids) as Float64Array;
const xyz = blobs.get(loaded.mesh!.nodes!.xyz) as Float64Array;
const N = ids.length;
const U = ["DISPLACEMENT/Ux", "DISPLACEMENT/Uy", "DISPLACEMENT/Uz"];
const R = ["ROTATION/Rx", "ROTATION/Ry", "ROTATION/Rz"];
const V = ["VELOCITY/Vx", "VELOCITY/Vy", "VELOCITY/Vz"];

/** The model with one `.ladruno`-named results stage recording `names` at step 0, the Vectors card on. */
function withStage(names: readonly string[], vectors: Partial<State["vectors"]> = {}, refusedNodal: { component: string; reason: string }[] = []): State {
  const nodeIds = blobs.put("test/vectors/ids", ids);
  const stage = { id: "s0", name: "", kind: "static", time: [0], nodal: [...names], refusedNodal, nodeIds, gauss: [], unread: [] };
  return {
    ...loaded,
    results: { path: "r.ladruno", version: "test", sourceType: null, lineage: null, partial: null, stages: [stage] } as never,
    phase: { ...loaded.phase, at: { kind: "results", stage: 0, step: 0 } },
    vectors: { ...loaded.vectors, on: true, thin: false, ...vectors },
  };
}

/** Rows for `names` (x, y, z) at the first `k` nodes: (1, 2, 2)·(1 + i) at node i (|v| = 3(1 + i)), NaN elsewhere. */
function rows(names: readonly string[], k: number): Record<string, Float64Array> {
  const out: Record<string, Float64Array> = {};
  for (const [c, name] of names.entries()) {
    const a = new Float64Array(N).fill(Number.NaN);
    for (let i = 0; i < k; i++) a[i] = [1, 2, 2][c]! * (1 + i);
    out[name] = a;
  }
  return out;
}

const rec = (a: Float32Array, i: number, f: number, n = 1) => Array.from(a.subarray(i * ARROW_STRIDE + f, i * ARROW_STRIDE + f + n));
const near = (a: readonly number[], b: readonly number[], tol = 1e-5) => a.length === b.length && a.every((x, i) => Math.abs(x - b[i]!) <= tol);

test("the stage's vector fields: each nodal field with two or more axes, the .ladruno groups by their labels, rotations marked", () => {
  const s = withStage([...U, ...R, ...V, "PRESSURE/p"]);
  assert.deepEqual(
    vectorFieldsOf(s).map((f) => [f.name, f.axes, f.rotational]),
    [
      ["DISPLACEMENT", U, false],
      ["ROTATION", R, true],
      ["VELOCITY", V, false],
    ],
    "a scalar group is no vector",
  );
  assert.equal(vectorFieldOf(s)?.name, "DISPLACEMENT", "null picks the first");
  assert.equal(vectorFieldOf(withStage([...U, ...V], { field: "VELOCITY" }))?.name, "VELOCITY");
  assert.equal(vectorFieldOf(withStage(U, { field: "VELOCITY" }))?.name, "DISPLACEMENT", "a field this stage lacks falls back to the first");
  // ndm 2: Ux, Uy is a vector with no z; the lone Rz may be a pore pressure (§7.1.1), so it is none.
  const two = vectorFieldsOf(withStage(["DISPLACEMENT/Ux", "DISPLACEMENT/Uy", "ROTATION/Rz"]));
  assert.deepEqual(two.map((f) => [f.name, f.axes]), [["DISPLACEMENT", ["DISPLACEMENT/Ux", "DISPLACEMENT/Uy", null]]]);
  // A refused axis: the vector of the axes left is not the vector.
  assert.deepEqual(vectorFieldsOf(withStage([...U, ...V], {}, [{ component: "VELOCITY/Vz", reason: "a partial write" }])).map((f) => f.name), ["DISPLACEMENT"]);
  // Off a results step, nothing.
  assert.deepEqual(vectorFieldsOf({ ...withStage(U), phase: { ...loaded.phase, at: null } }), []);
});

test("one arrow per node, its TAIL on the node, pointing along the vector; lengths follow the step's largest", () => {
  const s = withStage(U);
  const r = buildVectors3D(s, blobs, rows(U, 4), ids);
  assert.equal(r.count, 4);
  assert.equal(r.arcs.length, 0);
  assert.equal(r.maxMag, 12, "|(1, 2, 2)·4| = 12");
  for (let i = 0; i < 4; i++) {
    assert.deepEqual(rec(r.arrows, i, ARROW.anchor), [ANCHOR.tail], "tail-anchored");
    assert.ok(near(rec(r.arrows, i, ARROW.tip, 3), [xyz[3 * i]!, xyz[3 * i + 1]!, xyz[3 * i + 2]!]), "the record's point is the node");
    assert.ok(near(rec(r.arrows, i, ARROW.dir, 3), [1 / 3, 2 / 3, 2 / 3]), "along the vector");
  }
  const lens = [0, 1, 2, 3].map((i) => r.arrows[i * ARROW_STRIDE + ARROW.len]!);
  assert.ok(lens.every((l, i) => i === 0 || l > lens[i - 1]!), "a larger vector, a longer arrow");
});

test("the tail anchor puts the tip ahead of the node, in the CPU twin and in the shader", () => {
  const r = buildVectors3D(withStage(U), blobs, rows(U, 1), ids);
  const len = r.arrows[ARROW.len]!;
  const node = [xyz[0]!, xyz[1]!, xyz[2]!];
  const { linePositions, conePositions } = expandArrows({ arrows: r.arrows, combs: new Float32Array(0) }, len / 4);
  assert.ok(near(Array.from(linePositions.subarray(0, 3)), node), "the shaft starts at the node");
  assert.ok(near(Array.from(conePositions.subarray(0, 3)), [0, 1, 2].map((c) => node[c]! + [1 / 3, 2 / 3, 2 / 3][c]! * len)), "the head's tip is len along the vector");
  // A tip-anchored record (a load, a reaction) is unchanged: its tip is the point.
  const tipRec = new Float32Array(r.arrows);
  tipRec[ARROW.anchor] = ANCHOR.tip;
  assert.ok(near(Array.from(expandArrows({ arrows: tipRec, combs: new Float32Array(0) }, len / 4).conePositions.subarray(0, 3)), node));
  const src = arrowShaderSource();
  assert.match(src, /vec4 mvTip = vec4\( mvAt\.xyz \+ dV \* \( aAnchor \* L \), 1\.0 \);/, "the shader moves the tip by the drawn shaft L, the SIZING_EXPR length");
});

test("components: xyz draws one arrow per recorded non-zero axis; one axis draws that axis; z of a 2-D field draws none", () => {
  const xyzR = buildVectors3D(withStage(U, { component: "xyz" }), blobs, rows(U, 2), ids);
  assert.equal(xyzR.count, 6);
  const dirs = new Set(Array.from({ length: 6 }, (_, i) => rec(xyzR.arrows, i, ARROW.dir, 3).join(",")));
  assert.deepEqual([...dirs].sort(), ["0,0,1", "0,1,0", "1,0,0"]);
  const y = buildVectors3D(withStage(U, { component: "y" }), blobs, rows(U, 2), ids);
  assert.equal(y.count, 2);
  assert.equal(y.maxMag, 4);
  const U2 = U.slice(0, 2);
  assert.equal(buildVectors3D(withStage(U2, { component: "z" }), blobs, rows(U2, 2), ids).count, 0);
  assert.equal(buildVectors3D(withStage(U2, { component: "xyz" }), blobs, rows(U2, 2), ids).count, 4);
});

test("noise: a vector under 1e-6 of the step's largest, a zero one, and a NaN row draw nothing", () => {
  const data = rows(U, 3);
  for (const n of U) data[n]![1] = data[n]![0]! * VECTOR_FLOOR * 1e-3;
  for (const n of U) data[n]![2] = 0;
  const r = buildVectors3D(withStage(U), blobs, data, ids);
  assert.equal(r.count, 1, "node 0 only (nodes 3.. are NaN)");
});

test("a rotational field draws half circles centred on the nodes about their axes", () => {
  const r = buildVectors3D(withStage([...U, ...R], { field: "ROTATION" }), blobs, rows(R, 3), ids);
  assert.equal(r.arrows.length, 0);
  assert.equal(r.arcs.length, 3 * ARROW_STRIDE);
  for (let i = 0; i < 3; i++) {
    assert.ok(near(rec(r.arcs, i, ARROW.tip, 3), [xyz[3 * i]!, xyz[3 * i + 1]!, xyz[3 * i + 2]!]));
    assert.ok(near(rec(r.arcs, i, ARROW.dir, 3), [1 / 3, 2 / 3, 2 / 3]), "the rotation axis, right-hand rule");
  }
});

test("the arrows follow the deformed shape: the drawn node positions are the anchors", () => {
  const moved = new Map([[ids[0]!, [100, 200, 300] as const]]);
  const r = buildVectors3D(withStage(U), blobs, rows(U, 1), ids, { deformedNodeCoords: moved });
  assert.deepEqual(rec(r.arrows, 0, ARROW.tip, 3), [100, 200, 300]);
});

test("thinning is per node: a node's components share one level", () => {
  const data = rows(U, Math.min(N, 40));
  const r = buildVectors3D(withStage(U, { component: "xyz", thin: true }), blobs, data, ids);
  const byNode = new Map<string, Set<number>>();
  for (let i = 0; i < r.count; i++) {
    const k = rec(r.arrows, i, ARROW.tip, 3).join(",");
    byNode.set(k, (byNode.get(k) ?? new Set()).add(r.arrows[i * ARROW_STRIDE + ARROW.level]!));
  }
  assert.ok(byNode.size > 1);
  for (const levels of byNode.values()) assert.equal(levels.size, 1);
  assert.ok([...byNode.values()].every((l) => [...l][0]! >= 1), "every kept node has a level");
});

test("colour: one colour, the translation (force) or rotation (moment) colour by type, or jet by magnitude", () => {
  const pal = loadsColourPalette("dark");
  const colour = (s: State, field = U) => {
    const r = buildVectors3D(s, blobs, rows(field, 2), ids);
    return rec(r.arrows.length ? r.arrows : r.arcs, 1, ARROW.rgb, 3);
  };
  assert.ok(near(colour(withStage(U)), pal.mono));
  assert.ok(near(colour(withStage(U, { colorBy: "type" })), pal.force));
  assert.ok(near(colour(withStage([...U, ...R], { colorBy: "type", field: "ROTATION" }), R), pal.moment));
  assert.ok(near(colour(withStage(U, { colorBy: "magnitude" })), sampleCmap("jet", 1)), "the step's largest is jet's top");
});

test("the layer draws tail-anchored instances, keeps them through a scrub, and hides when off or on another stage", () => {
  const s = withStage(U);
  const layer = new VectorsLayer();
  const step = { stage: 0, step: 0, nodal: rows(U, 3) };
  assert.equal(layer.sync(s, blobs, null, step), true);
  assert.equal(layer.root.visible, true);
  const g = layer.lines.geometry as THREE.InstancedBufferGeometry;
  assert.equal(g.instanceCount, 3);
  assert.equal(g.getAttribute("aAnchor").getX(0), ANCHOR.tail, "the shader reads the record's anchor");
  // A scrub: the next step is not read yet, so the last arrows stay.
  const scrub = { ...s, phase: { ...s.phase, at: { kind: "results" as const, stage: 0, step: 1 } } };
  layer.sync(scrub, blobs, null, step);
  assert.equal(layer.root.visible, true);
  assert.equal(layer.sync({ ...s, vectors: { ...s.vectors, on: false } }, blobs, null, step), true, "turning off hides");
  assert.equal(layer.root.visible, false);
  // A rotational field rides the arcs.
  layer.sync(withStage([...U, ...R], { field: "ROTATION" }), blobs, null, { stage: 0, step: 0, nodal: rows(R, 2) });
  assert.equal((layer.arcs.lines.geometry as THREE.InstancedBufferGeometry).instanceCount, 2);
  assert.equal(layer.lines.visible, false);
  layer.dispose();
  // The reactions keep their tip on the node.
  const reactions = new ReactionsLayer();
  const rs = { ...withStage(["reaction_force_x", "reaction_force_y", "reaction_force_z"]), reactions: { ...loaded.reactions, on: true, allNodes: true } };
  reactions.sync(rs, blobs, null, { stage: 0, step: 0, nodal: rows(["reaction_force_x", "reaction_force_y", "reaction_force_z"], 2) });
  assert.equal((reactions.lines.geometry as THREE.InstancedBufferGeometry).getAttribute("aAnchor").getX(0), ANCHOR.tip);
  reactions.dispose();
});

test("the card: the field picker, the components the field records, and its events", () => {
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
    setCustomValidity() {}
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
    assert.equal(renderVectors(withStage(["PRESSURE/p"])), null, "no vector field, no card");
    assert.equal(renderVectors({ ...withStage(U), phase: { ...loaded.phase, at: null } }), null, "off a results step, no card");
    const box = renderVectors(withStage([...U, ...R])) as unknown as El;
    const options = (cls: string) => box.all((e) => e.className === cls)[0]!.children.map((c) => (c as El).value);
    assert.deepEqual(options("vectors-field"), ["DISPLACEMENT", "ROTATION"]);
    assert.deepEqual(options("vectors-component"), ["resultant", "xyz", "x", "y", "z"]);
    const two = renderVectors(withStage(["DISPLACEMENT/Ux", "DISPLACEMENT/Uy"])) as unknown as El;
    assert.deepEqual(two.all((e) => e.className === "vectors-component")[0]!.children.map((c) => (c as El).value), ["resultant", "xyz", "x", "y"], "no z on a 2-D field");
    const events: unknown[] = [];
    const live = renderVectors(withStage([...U, ...R]), (e) => events.push(e)) as unknown as El;
    const field = live.all((e) => e.className === "vectors-field")[0]!;
    field.value = "ROTATION";
    for (const fn of field.listeners) fn();
    for (const fn of live.all((e) => e.className.startsWith("vectors-toggle"))[0]!.listeners) fn();
    assert.deepEqual(events, [{ type: "setVectors", field: "ROTATION" }, { type: "setVectors", on: false }]);
  } finally {
    (globalThis as { document?: unknown }).document = prevDoc;
  }
});

test("setVectors: merges, keeps the state on a no-op, resets the field with null, and refuses bad values", () => {
  const a = reduce(initialState, { type: "setVectors", on: true, field: "VELOCITY", component: "xyz", colorBy: "magnitude", scale: 2, headSize: 9, thin: false });
  assert.deepEqual(a.vectors, { on: true, field: "VELOCITY", component: "xyz", colorBy: "magnitude", scale: 2, headSize: 9, thin: false });
  assert.equal(reduce(a, { type: "setVectors", on: true }), a, "nothing changed, the same state");
  assert.equal(reduce(a, { type: "setVectors", field: null }).vectors.field, null);
  for (const bad of [{ on: "yes" }, { thin: 1 }, { scale: 0 }, { headSize: Number.NaN }, { field: "" }, { field: 3 }, { colorBy: "rainbow" }, { component: "w" }]) {
    assert.throws(() => reduce(initialState, { type: "setVectors", ...(bad as object) } as never), RangeError, JSON.stringify(bad));
  }
});

test("end to end on a fork-written .ladruno: the step's DISPLACEMENT draws one tail-anchored arrow per moving node", async () => {
  const h = await opened(join(fixtures, "ladruno_frame3d.h5"), join(fixtures, "ladruno_frame3d.ladruno"));
  try {
    const s0 = h.store.get();
    const stage = s0.results!.stages.findIndex((st) => st.nodal.includes("DISPLACEMENT/Ux"));
    assert.ok(stage >= 0);
    const step = s0.results!.stages[stage]!.time.length - 1;
    h.store.dispatch({ type: "setPhase", at: { kind: "results", stage, step } });
    h.store.dispatch({ type: "setVectors", on: true });
    await until(() => {
      const d = h.effects.currentStep();
      return !!d && d.stage === stage && d.step === step;
    }, "the step");
    const s = h.store.get();
    assert.deepEqual(vectorFieldsOf(s).map((f) => [f.name, f.rotational]), [["DISPLACEMENT", false], ["ROTATION", true]]);
    const data = h.effects.currentStep()!;
    const nodeIds = h.blobs.get(s.results!.stages[stage]!.nodeIds!) as Float64Array;
    const r = buildVectors3D(s, h.blobs, data.nodal, nodeIds);
    const mags = Array.from(nodeIds, (_, i) => Math.hypot(...U.map((n) => data.nodal[n]![i]!)));
    const max = Math.max(...mags);
    assert.ok(max > 0, "the fixture's last step moves");
    assert.equal(r.count, mags.filter((m) => m >= VECTOR_FLOOR * max && m > 1e-30).length, "every node above the floor, the fixed bases dropped");
    for (let i = 0; i < r.count; i++) assert.equal(r.arrows[i * ARROW_STRIDE + ARROW.anchor], ANCHOR.tail);
    // ROTATION is all zeros in this fixture: nothing to draw.
    assert.equal(buildVectors3D(reduce(s, { type: "setVectors", field: "ROTATION" }), h.blobs, data.nodal, nodeIds).count, 0);
  } finally {
    h.effects.dispose();
    h.host.close();
  }
});
