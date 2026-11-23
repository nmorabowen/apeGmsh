// ADR 0135 D6 and Amendment 1: the tester's form, the drawn curve, the axes, and
// the form's life in the store.

import assert from "node:assert/strict";
import { test } from "node:test";
import { strainPath, stressPath } from "../../src/main/materialTestScript.ts";
import {
  materialTestAxesOf,
  materialTestFormFor,
  materialTestCsvName,
  materialTestCsvOf,
  materialTestPointsOf,
  materialTestProtocolOf,
  materialTestProtocolText,
  materialTestWorkOf,
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
  assert.deepEqual(materialTestPointsOf([0, 1, 2], [0, 10, 20], [10, 10, 10]), { step: [0, 1, 2], strain: [0, 1, 2], stress: [0, 10, 20], tangent: [10, 10, 10], energy: [0, 5, 20] });
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
  const p = materialTestPointsOf(strain, stress, strain.map(() => 2), 1000);
  assert.ok(p.strain.length <= 1000 + 3, `${p.strain.length} points`);
  assert.ok(p.strain.includes(1) && p.strain.includes(-1), "both reversal corners are kept");
  assert.equal(p.strain.at(-1), 0.5, "the last point is kept");
  assert.deepEqual(p.stress, p.strain.map((e) => 2 * e), "stress is taken at the same steps");
  assert.deepEqual(p.strain, p.step.map((i) => strain[i]), "each point names its step");
  const full = materialTestWorkOf(strain, stress);
  assert.deepEqual(p.energy, p.step.map((i) => full[i]), "the work is integrated on the full run, then thinned");
});

test("when reversals alone pass twice the cap, the stride alone decides", () => {
  const strain = Array.from({ length: 10_000 }, (_, i) => (i % 2 ? 1 : -1));
  const p = materialTestPointsOf(strain, strain, strain, 100);
  assert.ok(p.strain.length <= 101, `${p.strain.length} points`);
});

test("the work done: exact on a linear ramp, zero over an elastic cycle, the loop's area over a rigid-plastic one", () => {
  const ramp = Array.from({ length: 11 }, (_, i) => i / 10);
  assert.ok(Math.abs(materialTestWorkOf(ramp, ramp.map((e) => 200 * e)).at(-1)! - 100) < 1e-9, "E e^2 / 2 at e = 1, E = 200");
  const cycle = [0, 1, 0, -1, 0];
  assert.ok(Math.abs(materialTestWorkOf(cycle, cycle.map((e) => 3 * e)).at(-1)!) < 1e-12, "an elastic cycle stores and returns all of it");
  // rigid-plastic at +-1, with the reversals as their own steps: +1 to 2, unload at 2, -1 to -2, unload at -2, +1 back to 0
  const e = [0, 2, 2, -2, -2, 0];
  const s = [1, 1, -1, -1, 1, 1];
  assert.equal(materialTestWorkOf(e, s).at(-1), 2 + 4 + 2, "|stress| times the plastic path, 8");
});

test("the CSV: the material, build and protocol as # lines, then one row per step, in full precision", () => {
  const csv = materialTestCsvOf(
    { decl: "opensees/uniaxialMaterial/bond", type: "LadrunoBondSlip", build: "8b5e0593f8de", protocol: { kind: "monotonic", peak: 2, steps: 2 } },
    [0, 1, 2],
    [0, 0.1 + 0.2, 10],
    [0.3, 0.3, 0],
  );
  assert.equal(
    csv,
    [
      "# opensees/uniaxialMaterial/bond (LadrunoBondSlip)",
      "# fork build 8b5e0593f8de; monotonic to 2 in 2 steps; values in the model's units",
      "# work: the trapezoidal integral of bond stress over slip, computed by apeGmshViewer",
      "step,slip,bond stress,tangent,work",
      "0,0,0,0.3,0",
      "1,1,0.30000000000000004,0.3,0.15000000000000002",
      "2,2,10,0,5.300000000000001",
      "",
    ].join("\n"),
  );
  assert.equal(materialTestCsvName("opensees/uniaxialMaterial/buckled steel", 3), "buckled_steel-run3.csv");
  assert.equal(materialTestCsvName("", 1), "material-run1.csv");
  assert.equal(materialTestProtocolText({ kind: "cyclic", amplitudes: [0.01, 0.02], stepsPerRamp: 50 }), "cyclic 0.01, 0.02 at 50 steps per ramp");
  assert.equal(materialTestProtocolText({ kind: "values", values: [1, 2, 3] }), "3 pasted values");
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

test("S3: a stress-control form becomes its protocol; its step count is main's stress path", () => {
  const f = form({ kind: "stress", targets: "300; -300 400", stepsPerRamp: "5" });
  const a = materialTestProtocolOf(f, "strain");
  if (!("protocol" in a)) return assert.fail(a.error);
  assert.deepEqual(a.protocol, { kind: "stress", targets: [300, -300, 400], stepsPerRamp: 5 });
  assert.equal(materialTestStepsOf(a.protocol), stressPath(a.protocol).length);
  assert.deepEqual(materialTestProtocolOf(form({ kind: "stress", targets: "300, x" })), { error: 'targets: "x" is not a number' });
  assert.equal(materialTestProtocolText({ kind: "stress", targets: [300, -300], stepsPerRamp: 5 }), "stress control to 300, -300 at 5 steps per leg");
});
