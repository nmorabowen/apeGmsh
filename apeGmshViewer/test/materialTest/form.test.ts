// ADR 0135 D6 and Amendment 1: the tester's form, the drawn curve, the axes, and
// the form's life in the store.

import assert from "node:assert/strict";
import { test } from "node:test";
import { strainPath } from "../../src/main/materialTestScript.ts";
import {
  materialTestAxesOf,
  materialTestFormFor,
  materialTestPointsOf,
  materialTestProtocolOf,
  materialTestStepsOf,
} from "../../src/state/materialTestForm.ts";
import { initialState, reduce } from "../../src/state/reduce.ts";
import type { MaterialTestForm } from "../../src/state/types.ts";

const form = (patch: Partial<MaterialTestForm>): MaterialTestForm => ({ ...materialTestFormFor("m"), ...patch });

test("each kind of form becomes its protocol", () => {
  assert.deepEqual(materialTestProtocolOf(form({ kind: "monotonic", peak: " -0.004 ", steps: "40" })), { protocol: { kind: "monotonic", peak: -0.004, steps: 40 } });
  assert.deepEqual(materialTestProtocolOf(form({ kind: "cyclic", amplitudes: "0.002, 0.005 1e-2", stepsPerRamp: "25" })), {
    protocol: { kind: "cyclic", amplitudes: [0.002, 0.005, 0.01], stepsPerRamp: 25 },
  });
  assert.deepEqual(materialTestProtocolOf(form({ kind: "values", values: "0.001\n0.002; -0.001" })), { protocol: { kind: "values", values: [0.001, 0.002, -0.001] } });
  assert.ok("protocol" in materialTestProtocolOf(materialTestFormFor("m")), "a new form runs as it is");
});

test("a wrong field says which, in the form's words and the material's axis", () => {
  const err = (f: MaterialTestForm, axis?: string) => {
    const a = materialTestProtocolOf(f, axis);
    return "error" in a ? a.error : "no error";
  };
  assert.equal(err(form({ kind: "monotonic", peak: "" }), "slip"), "the peak slip must be a number");
  assert.equal(err(form({ kind: "monotonic", steps: "2.5" })), "steps must be a whole number of at least 1");
  assert.equal(err(form({ kind: "monotonic", steps: "0" })), "steps must be a whole number of at least 1");
  assert.equal(err(form({ kind: "cyclic", amplitudes: "0.01, abc" })), 'amplitudes: "abc" is not a number');
  assert.equal(err(form({ kind: "cyclic", amplitudes: " , " })), "amplitudes: give at least one number");
  assert.equal(err(form({ kind: "cyclic", stepsPerRamp: "" })), "steps per ramp must be a whole number of at least 1");
  assert.equal(err(form({ kind: "values", values: "1, Infinity" }), "rotation jump"), 'rotation jump values: "Infinity" is not a number');
});

test("the step count shown is the number of steps main imposes after the starting zero", () => {
  for (const f of [form({ kind: "monotonic", steps: "7" }), form({ kind: "cyclic", amplitudes: "1 2 3", stepsPerRamp: "5" }), form({ kind: "values", values: "1 2 3 4" })]) {
    const a = materialTestProtocolOf(f);
    assert.ok("protocol" in a);
    assert.equal(materialTestStepsOf(a.protocol), strainPath(a.protocol).length - 1, f.kind);
  }
});

test("the axes: strain and stress, except the two materials ADR 0135 D6 relabels", () => {
  assert.deepEqual(materialTestAxesOf("Steel02"), ["strain", "stress"]);
  assert.deepEqual(materialTestAxesOf("LadrunoBondSlip"), ["slip", "bond stress"]);
  assert.deepEqual(materialTestAxesOf("LadrunoCohesiveHinge"), ["rotation jump", "moment"]);
});

test("a short curve is drawn whole", () => {
  assert.deepEqual(materialTestPointsOf([0, 1, 2], [0, 10, 20]), { strain: [0, 1, 2], stress: [0, 10, 20] });
});

test("a long curve is thinned to the cap, keeping every reversal and the last point", () => {
  // 0 -> 1 -> -1 -> 0.5 in about 40,000 steps, with ramp lengths prime to the stride: two reversals, off the stride.
  const strain: number[] = [];
  const ramp = (from: number, to: number, n: number) => {
    for (let i = strain.length ? 1 : 0; i <= n; i++) strain.push(from + ((to - from) * i) / n);
  };
  ramp(0, 1, 10007);
  ramp(1, -1, 20011);
  ramp(-1, 0.5, 9973);
  const stress = strain.map((e) => 2 * e);
  const p = materialTestPointsOf(strain, stress, 1000);
  assert.ok(p.strain.length <= 1000 + 3, `${p.strain.length} points`);
  assert.ok(p.strain.includes(1) && p.strain.includes(-1), "both reversal corners are kept");
  assert.equal(p.strain.at(-1), 0.5, "the last point is kept");
  assert.deepEqual(p.stress, p.strain.map((e) => 2 * e), "stress is taken at the same steps");
});

test("when reversals alone pass twice the cap, the stride alone decides", () => {
  const strain = Array.from({ length: 10_000 }, (_, i) => (i % 2 ? 1 : -1));
  const p = materialTestPointsOf(strain, strain, 100);
  assert.ok(p.strain.length <= 101, `${p.strain.length} points`);
});

test("the form lives in the store: opened, edited, closed; another model drops it", () => {
  let s = reduce(initialState, { type: "materialTestFormSet", form: materialTestFormFor("m") });
  assert.equal(s.materialTest.form?.decl, "m");
  const edited = { ...s.materialTest.form!, peak: "0.0" };
  s = reduce(s, { type: "materialTestFormSet", form: edited });
  assert.equal(s.materialTest.form, edited);
  assert.equal(reduce(s, { type: "materialTestFormSet", form: edited }), s, "the same form is no change");
  assert.equal(reduce(s, { type: "materialTestFormSet", form: null }).materialTest.form, null);
  assert.equal(reduce(s, { type: "fileOpened", artifact: "model", path: "other.h5" }).materialTest.form, null);
});
