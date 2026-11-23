// ADR 0135 S1's "done when", as a test: the six first materials, from a real
// model.h5 the bridge wrote (fixtures/materials_tester.h5; its generator is in
// fixtures/README.md), become jobs the script accepts (D3, D4), and, where
// $APEGMSH_OPENSEES names the fork, runs that return a curve.
//
// Oracle: the generator's own arguments, as the file stores them positionally.

import assert from "node:assert/strict";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { MaterialTestRunner } from "../../src/main/materialTest.ts";
import { scriptOf } from "../../src/main/materialTestScript.ts";
import { openModel } from "../../src/reader/node.ts";
import { BlobStore } from "../../src/state/blobs.ts";
import { loadModel } from "../../src/state/load.ts";
import { materialTestJobOf } from "../../src/state/materialTest.ts";
import { materialTestPointsOf } from "../../src/state/materialTestForm.ts";
import { initialState, reduce } from "../../src/state/reduce.ts";
import type { MaterialTestProtocol } from "../../src/state/types.ts";

const model = await openModel(fileURLToPath(new URL("../../fixtures/materials_tester.h5", import.meta.url)));
const state = reduce(initialState, { type: "fileLoaded", artifact: "model", load: loadModel(model, new BlobStore()) });
const U = (name: string) => `opensees/uniaxialMaterial/${name}`;
const cyclic: MaterialTestProtocol = { kind: "cyclic", amplitudes: [0.01, 0.03], stepsPerRamp: 50 };

/** name -> [type, the tags of the job's lines, dependencies first] */
const EXPECTED: Record<string, [string, number[]]> = {
  steel: ["Steel02", [1]],
  buckled_steel: ["LadrunoRebarBuckling", [1, 2]],
  concrete: ["Concrete02", [3]],
  j2: ["LadrunoUniaxialJ2", [4]],
  buckled_j2: ["LadrunoRebarBuckling", [4, 5]],
  bond: ["LadrunoBondSlip", [6]],
  hinge: ["LadrunoCohesiveHinge", [7]],
};

test("the file holds the seven declared uniaxial materials", () => {
  const found = Object.values(state.decls).filter((d) => d.kind === "uniaxialMaterial").map((d) => d.path).sort();
  assert.deepEqual(found, Object.keys(EXPECTED).map(U).sort());
});

test("each material's job carries its dependencies first, with the file's tags, and passes the script's token rules", () => {
  for (const [name, [type, tags]] of Object.entries(EXPECTED)) {
    const a = materialTestJobOf(state, U(name), cyclic);
    assert.ok("job" in a, `${name}: ${JSON.stringify(a)}`);
    assert.deepEqual(a.job.materials.map((m) => m.tag), tags, name);
    assert.equal(a.job.materials.at(-1)!.type, type, name);
    assert.equal(a.job.target, tags.at(-1), name);
    const { script } = scriptOf(a.job);
    for (const m of a.job.materials) assert.ok(script.includes(`uniaxialMaterial ${m.type} ${m.tag} `), `${name}: line for tag ${m.tag}`);
  }
  const j2 = materialTestJobOf(state, U("j2"), cyclic);
  assert.ok("job" in j2);
  assert.deepEqual(j2.job.materials[0]!.args, [200000, "-iso", "voce", 420, 100, 10, 0, "-kin", 1, 5000, 50], "flags and numbers as the bridge wrote them");
});

const skip = process.env["APEGMSH_OPENSEES"] ? false : "set $APEGMSH_OPENSEES to the fork's OpenSees.exe";

test("every material of the file runs on the fork and returns a curve", { skip }, async () => {
  const runner = new MaterialTestRunner();
  for (const name of Object.keys(EXPECTED)) {
    const a = materialTestJobOf(state, U(name), name === "bond" ? { kind: "monotonic", peak: 6, steps: 200 } : cyclic);
    assert.ok("job" in a);
    const r = await runner.run(a.job, 1);
    assert.ok(r.ok, `${name}: ${r.ok ? "" : r.error}`);
    assert.ok(r.stress.some((s) => s !== 0), `${name}: a curve, not zeros`);
    const p = materialTestPointsOf(r.strain, r.stress);
    assert.equal(p.strain.length, r.strain.length, `${name}: a short run is drawn whole`);
  }
});
