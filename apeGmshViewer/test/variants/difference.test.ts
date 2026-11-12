// VU10 (#2694; ADR 0132 D9.5): the difference of two variants, end to end on
// the renderer side, over the committed synthetic pair (test/fixtures/vu10,
// scripts/make_viewer_variants_fixture.py). The window set (Uncracked) is opened through the real
// Effects; the compare set (Cracked) is read by a second ResultsHost behind a
// VariantDoor, as main's compare host serves it. The oracle is the fixture's
// closed form, restated here: at step k, s = 0.01 (k + 1) and
//
//     Uy(x) = 0.5 s x^2 / r,   r = 1 (Uncracked), 0.35 (Cracked)
//
// so the difference Uncracked minus Cracked at a node is 0.5 s x^2 (1 - 1/0.35).
// The refusals are judged on the same state: a User Combination (a combination
// across variants), another mesh (fem_hash), a magnitude, an element field;
// and a difference is never offered as a deformed shape.

import assert from "node:assert/strict";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { compareStagesOf, ResultsHost } from "../../src/main/results.ts";
import { openModel } from "../../src/reader/node.ts";
import type { ResultsH5 } from "../../src/reader/results.ts";
import { differenceHeld, VariantRunner, type VariantDoor } from "../../src/renderer/contour.ts";
import { differenceColumn, MAGNITUDE } from "../../src/state/contour.ts";
import { contourView, deformOf, differencePlanOf, noticesOf } from "../../src/state/selectors.ts";
import { reduce } from "../../src/state/reduce.ts";
import type { CompareIn, State, VariantsIn } from "../../src/state/types.ts";
import { harness, until } from "../contour/harness.ts";

const FIX = (p: string) => fileURLToPath(new URL(`../fixtures/vu10/${p}`, import.meta.url));
const UNCRACKED = { model: FIX("plate_uncracked.h5"), results: FIX("plate_uncracked.ladruno") };
const CRACKED = { model: FIX("cracked/plate_cracked.h5"), results: FIX("cracked/plate_cracked.ladruno") };
const RATIO = 0.35;
const LAST = 2;
const uy = (x: number, k: number, r: number) => (0.5 * 0.01 * (k + 1) * x * x) / r;

const STUDY: VariantsIn = {
  manifest: FIX("study.variants.json"),
  version: "1.0.0",
  newer: null,
  rows: [
    { index: 0, label: "Uncracked", path: "plate_uncracked.h5", params: { "Ie/Ig": 1 } },
    { index: 1, label: "Cracked", path: "cracked/plate_cracked.h5", params: { "Ie/Ig": 0.35 } },
  ],
  refused: [],
  active: 0,
  compare: null,
};

/** A VariantDoor over a second results host, as main's compare door and host answer. */
function compareDoor(path = CRACKED.results) {
  const host = new ResultsHost(h5wasm as unknown as ResultsH5, h5wasm.ready, { idleMs: 50 });
  const asked: unknown[] = [];
  const door: VariantDoor = {
    variantsSwitch: async () => ({ ok: false, reason: "not in this test" }),
    variantsCompare: async (index, label) => {
      const a = await host.open(path);
      if (!a.ok) return { ok: false, reason: a.error };
      const compare: CompareIn = { index, label, results: path, femHash: a.meta.lineage?.femHash ?? null, stages: compareStagesOf(a.meta) };
      return { ok: true, compare, nodeIds: a.nodeIds };
    },
    compareClose: async () => host.close(),
    readCompareStep: async (req) => {
      asked.push(req);
      const r = host.readStep({ ...req, path });
      return r.ok ? { ok: true, seq: r.seq, nodal: r.nodal } : { ok: false, seq: r.seq, error: r.error };
    },
    // VU19: this compare set's model and user file are not under test here; main names no slot for them.
    compareModel: async () => ({ ok: false, slot: null, reason: "not in this test" }),
    compareUserRead: async () => ({ slot: null, path: null, read: { ok: false, error: "not in this test" } }),
    compareUserReadResult: async () => ({ slot: null, read: { ok: false, reason: "refused", detail: "not in this test" } }),
  };
  return { door, asked };
}

/** The window set open, the study read, the compare set open, the contour on Uy at the last step. */
async function compared() {
  const h = harness();
  if (!(await h.effects.open(UNCRACKED.model))) throw new Error("the window model did not load");
  if (!(await h.effects.openResults(UNCRACKED.results))) throw new Error("the window results did not load");
  const { door, asked } = compareDoor();
  const runner = new VariantRunner(h.store, h.blobs, door, () => h.effects.currentStep());
  const off = runner.start();
  h.store.dispatch({ type: "variantsRead", study: STUDY });
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 0 } });
  h.store.dispatch({ type: "setResultStep", step: LAST });
  h.store.dispatch({ type: "setColourBy", by: "contour" });
  h.store.dispatch({ type: "setContourField", field: "DISPLACEMENT", component: "DISPLACEMENT/Uy" });
  h.store.dispatch({ type: "variantCompare", row: { index: 1, label: "Cracked" } });
  await until(() => h.store.get().variants.compare?.status === "open", "the compare set");
  return { h, off, asked };
}

test("the difference is the window variant minus the compare variant, node by node, at the step shown (closed form)", async () => {
  const { h, off, asked } = await compared();
  h.store.dispatch({ type: "setDifference", on: true });
  await until(() => contourView(h.store.get()).state === "difference", "the difference view");
  const v = contourView(h.store.get());
  assert.equal(v.state, "difference");
  if (v.state !== "difference") return;
  assert.equal(v.window, "Uncracked");
  assert.equal(v.compare, "Cracked");
  assert.equal(v.missing, 0);
  const model = await openModel(UNCRACKED.model);
  const x = new Map<number, number>();
  model.nodeIds.forEach((id, i) => x.set(id, model.nodeCoords[3 * i]!));
  const held = differenceHeld.current!;
  assert.equal(held.key, v.key);
  assert.equal(held.ids.length, 15);
  held.ids.forEach((id, i) => {
    const want = uy(x.get(id)!, LAST, 1) - uy(x.get(id)!, LAST, RATIO);
    assert.ok(Math.abs(held.values[i]! - want) < 1e-12, `node ${id}: ${held.values[i]} vs ${want}`);
  });
  // The range is the difference's own: 0 at x = 0, 0.06 (1 - 1/0.35) at x = 2.
  assert.ok(Math.abs(v.clim.min! - 0.06 * (1 - 1 / RATIO)) < 1e-12 && v.clim.max === 0, JSON.stringify(v.clim));
  // The compare step was asked by stage, step and request number only: never a path.
  assert.deepEqual(Object.keys(asked[0] as object).sort(), ["seq", "stage", "step"]);
  off();
});

test("a difference is never offered as a deformed shape, and the window's own deformed shape returns with it off", async () => {
  const { h, off } = await compared();
  assert.equal(deformOf(h.store.get()).ok, true);
  h.store.dispatch({ type: "setDifference", on: true });
  const d = deformOf(h.store.get());
  assert.equal(d.ok, false);
  assert.match(!d.ok ? d.reason : "", /contour only, never as a deformed shape/);
  h.store.dispatch({ type: "setDifference", on: false });
  assert.equal(deformOf(h.store.get()).ok, true);
  off();
});

test("a combination across variants is refused: on a User Combination the difference draws nothing and says why", async () => {
  const { h, off } = await compared();
  h.store.dispatch({ type: "setDifference", on: true });
  // The phase as a User Combination would set it (the reducer checks the user file; the selector is judged here).
  const s: State = { ...h.store.get(), phase: { ...h.store.get().phase, at: { kind: "combination", name: "D+L" } } };
  const p = differencePlanOf(s);
  assert.equal(p?.ok, false);
  assert.match(p && !p.ok ? p.reason : "", /a combination across variants is refused/);
  const v = contourView(s);
  assert.equal(v.state, "refused");
  assert.match(v.state === "refused" ? v.reason : "", /a combination across variants is refused/);
  assert.ok(noticesOf(s).some((n) => /Contour not drawn: the difference is refused: a User Combination/.test(n)));
  off();
});

test("another mesh, a magnitude and an element field are refused, each with its reason", async () => {
  const { h, off } = await compared();
  h.store.dispatch({ type: "setDifference", on: true });
  const base = h.store.get();
  const c = base.variants.compare!;
  assert.equal(c.status, "open");
  const otherMesh: State = { ...base, variants: { ...base.variants, compare: { ...(c as Extract<typeof c, { status: "open" }>), femHash: "0".repeat(32) } } };
  assert.match((differencePlanOf(otherMesh) as { reason: string }).reason, /do not state one mesh/);
  const magnitude = reduce(base, { type: "setContourField", field: "DISPLACEMENT", component: MAGNITUDE });
  assert.match((differencePlanOf(magnitude) as { reason: string }).reason, /a magnitude is not subtracted/);
  const element: State = { ...base, contour: { ...base.contour, source: "element" } };
  assert.match((differencePlanOf(element) as { reason: string }).reason, /an element field is not subtracted/);
  off();
});

test("W4d #2843: the two variants' stages are one stage by their program names; a display name is a label only", async () => {
  const { h, off } = await compared();
  h.store.dispatch({ type: "setDifference", on: true });
  const base = h.store.get();
  const c = base.variants.compare as Extract<NonNullable<State["variants"]["compare"]>, { status: "open" }>;
  assert.equal(c.status, "open");
  assert.equal(differencePlanOf(base)?.ok, true, "precondition: the pair subtracts");
  // The window's stage shown by a display name ("Base load"), same program name: still subtracted.
  const shownAs = (name: string, programName: string): State => ({
    ...base,
    results: { ...base.results!, stages: base.results!.stages.map((st, i) => (i === 0 ? { ...st, name, programName } : st)) },
  });
  const prog = base.results!.stages[0]!.programName;
  const renamed = differencePlanOf(shownAs("Base load", prog));
  assert.equal(renamed?.ok, true, JSON.stringify(renamed));
  // Another program name is refused, even when the display name equals the compare stage's name.
  const other = differencePlanOf(shownAs(c.stages[0]!.name, "another stage"));
  assert.equal(other?.ok, false);
  assert.match(other && !other.ok ? other.reason : "", /is program stage "another stage" here and ".*" in "Cracked"/);
  off();
});

test("the compare set is never the window set: the reducer refuses to compare the window's own row", () => {
  const s = reduce({ ...harness().store.get() }, { type: "variantsRead", study: STUDY });
  assert.throws(() => reduce(s, { type: "variantCompare", row: { index: 0, label: "Uncracked" } }), /is the window set/);
  assert.throws(() => reduce(s, { type: "variantCompare", row: { index: 1, label: "Uncracked" } }), /no variant row 1 labelled "Uncracked"/);
  assert.throws(() => reduce(s, { type: "setDifference", on: true }), /no compare set is open/);
});

test("differenceColumn: a node either side lacks, or a value that is not finite, is no data and counted, never 0", () => {
  const r = differenceColumn(Float64Array.of(1, 2, 3, 4), Float64Array.of(1, 2, NaN, 4), Float64Array.of(4, 2, 3), Float64Array.of(0.5, 1, 3));
  assert.deepEqual(Array.from(r.values).map((v) => (Number.isNaN(v) ? "nd" : v)), ["nd", 1, "nd", 3.5]);
  assert.equal(r.missing, 2);
  assert.deepEqual(r.range, { min: 1, max: 3.5 });
  assert.throws(() => differenceColumn(Float64Array.of(1), Float64Array.of(1, 2), Float64Array.of(1), Float64Array.of(1)), /1 node ids and 2 values/);
});
