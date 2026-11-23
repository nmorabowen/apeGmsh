// ADR 0135 D6 and D8 (Amendment 1 slice A3): overriding a material's numeric slots
// for one test. Oracle: fixtures/materials_tester.h5 and its generator's arguments
// (fixtures/README.md). Overrides reach the material under test only, never a
// dependency or a flag, and never the model.

import assert from "node:assert/strict";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { MaterialTestRunner } from "../../src/main/materialTest.ts";
import { scriptOf } from "../../src/main/materialTestScript.ts";
import { openModel } from "../../src/reader/node.ts";
import { BlobStore } from "../../src/state/blobs.ts";
import { loadModel } from "../../src/state/load.ts";
import { materialTestJobOf, materialTestSlotsOf } from "../../src/state/materialTest.ts";
import { materialTestFormFor, materialTestOverridesOf, materialTestOverrideText } from "../../src/state/materialTestForm.ts";
import { initialState, reduce } from "../../src/state/reduce.ts";
import type { MaterialTestOverride, MaterialTestProtocol } from "../../src/state/types.ts";

const model = await openModel(fileURLToPath(new URL("../../fixtures/materials_tester.h5", import.meta.url)));
const state = reduce(initialState, { type: "fileLoaded", artifact: "model", load: loadModel(model, new BlobStore()) });
const U = (name: string) => `opensees/uniaxialMaterial/${name}`;
const protocol: MaterialTestProtocol = { kind: "monotonic", peak: 0.01, steps: 20 };
const o = (slot: number, value: number, label = "x", declared = 0): MaterialTestOverride => ({ slot, label, declared, value });
const labels = (name: string) => materialTestSlotsOf(state, U(name)).map((x) => [x.slot, x.label, x.value, x.locked !== null]);

test("slots are labelled by params_names where the file has them", () => {
  assert.deepEqual(labels("steel"), [
    [0, "fy", 420, false], [1, "E", 200000, false], [2, "b", 0.01, false], [3, "R0", 18, false], [4, "cR1", 0.925, false], [5, "cR2", 0.15, false],
  ]);
  assert.deepEqual(labels("bond").map((x) => x[1]), ["tau_max", "s1", "s2", "s3", "tau_f", "alpha"]);
});

test("without params_names, by the preceding flag (and its words) and the position after it; flags are never slots", () => {
  assert.deepEqual(labels("j2"), [
    [0, "slot 1", 200000, false],
    [3, "-iso voce · 1", 420, false], [4, "-iso voce · 2", 100, false], [5, "-iso voce · 3", 10, false], [6, "-iso voce · 4", 0, false],
    [8, "-kin · 1", 1, false], [9, "-kin · 2", 5000, false], [10, "-kin · 3", 50, false],
  ]);
  assert.deepEqual(labels("hinge"), [[0, "slot 1", 100, false], [1, "slot 2", 1, false]]);
});

test("a wrapper's dependency tag is locked, and says why", () => {
  const slots = materialTestSlotsOf(state, U("buckled_steel"));
  assert.deepEqual(slots.map((x) => [x.slot, x.label, x.value]), [[0, "slot 1", 1], [2, "-lsr · 1", 8], [4, "-fy · 1", 420], [6, "-E · 1", 200000]]);
  assert.match(slots[0]!.locked!, /holds the tag of material 1/);
  assert.deepEqual(slots.slice(1).map((x) => x.locked), [null, null, null]);
  assert.match(materialTestSlotsOf(state, U("buckled_j2"))[0]!.locked!, /material 4/);
});

test("an override replaces the target's slot only; the dependency's line and the model are untouched", () => {
  const a = materialTestJobOf(state, U("buckled_steel"), protocol, [o(2, 12), o(4, 500)]);
  assert.ok("job" in a, JSON.stringify(a));
  assert.deepEqual(a.job.materials[0]!.args, [420, 200000, 0.01, 18, 0.925, 0.15], "Steel02 as declared");
  assert.deepEqual(a.job.materials[1]!.args, [1, "-lsr", 12, "-fy", 500, "-E", 200000]);
  assert.deepEqual(state.decls[U("buckled_steel")]!.params, [1, "-lsr", 8, "-fy", 420, "-E", 200000], "the model is not changed");
  assert.match(scriptOf(a.job).script, /uniaxialMaterial LadrunoRebarBuckling 2 1 -lsr 12 -fy 500 -E 200000\n/);
});

test("hostile overrides are refused by name: a locked slot, a flag, a slot out of range, a non-finite value", () => {
  const err = (overrides: MaterialTestOverride[]) => {
    const a = materialTestJobOf(state, U("buckled_steel"), protocol, overrides);
    return "error" in a ? a.error : "accepted";
  };
  assert.match(err([o(0, 99)]), /slot 1 is locked \(it holds the tag of material 1/);
  assert.match(err([o(1, 5)]), /argument 1 is not a numeric slot/, "the -lsr flag");
  assert.match(err([o(7, 5)]), /argument 7 is not a numeric slot/);
  assert.match(err([o(-1, 5)]), /argument -1 is not a numeric slot/);
  assert.match(err([o(2, Number.NaN)]), /-lsr · 1 must be a finite number/);
  assert.match(err([o(2, Infinity)]), /must be a finite number/);
  const s = reduce(state, { type: "materialTestRequested", decl: U("buckled_steel"), protocol, overrides: [o(0, 99)] });
  assert.equal(s.materialTest.runs.at(-1)!.status, "failed", "the reducer refuses the run at once");
});

test("the form: a blank field, or one equal to the declared value, is no override; a changed locked slot is an error", () => {
  const slots = materialTestSlotsOf(state, U("buckled_steel"));
  const form = (overrides: Record<string, string>) => ({ ...materialTestFormFor(U("buckled_steel")), overrides });
  assert.deepEqual(materialTestOverridesOf(form({ "2": " ", "4": "420", "6": "2e5" }), slots), { overrides: [] });
  assert.deepEqual(materialTestOverridesOf(form({ "4": "450", "2": "10" }), slots), {
    overrides: [{ slot: 2, label: "-lsr · 1", declared: 8, value: 10 }, { slot: 4, label: "-fy · 1", declared: 420, value: 450 }],
  });
  assert.deepEqual(materialTestOverridesOf(form({ "4": "4 50" }), slots), { error: "-fy · 1 must be a number" });
  assert.match((materialTestOverridesOf(form({ "0": "7" }), slots) as { error: string }).error, /slot 1 is locked/);
  assert.equal(
    materialTestOverrideText(U("buckled_steel"), "LadrunoRebarBuckling", [{ slot: 4, label: "-fy · 1", declared: 420, value: 450 }, { slot: 2, label: "-lsr · 1", declared: 8, value: 10 }]),
    "buckled_steel (LadrunoRebarBuckling): -fy · 1 420 -> 450; -lsr · 1 8 -> 10",
  );
});

const skip = process.env["APEGMSH_OPENSEES"] ? false : "set $APEGMSH_OPENSEES to the fork's OpenSees.exe";

test("on the fork: raising Steel02's fy raises the stress the material reaches", { skip }, async () => {
  const run = async (overrides: MaterialTestOverride[]) => {
    const a = materialTestJobOf(state, U("steel"), { kind: "monotonic", peak: 0.02, steps: 100 }, overrides);
    assert.ok("job" in a);
    const r = await new MaterialTestRunner().run(a.job, 1);
    assert.ok(r.ok, r.ok ? "" : r.error);
    return r.stress.at(-1)!;
  };
  const declared = await run([]);
  const raised = await run([o(0, 600, "fy", 420)]);
  assert.ok(raised > declared + 150, `declared ${declared}, fy 600 ${raised}`);
});
