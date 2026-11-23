// ADR 0135 D9, slice S5: nD materials through the fork's `model testNDMaterial`
// (WP-248). The loading modes become `ndTest mixed` letters; the tangent shown is the
// fork's, condensed onto the driven component. The pure parts run anywhere; the
// acceptance runs on the fork where $APEGMSH_OPENSEES names a build with `ndTest`.
//
// Fixture: fixtures/materials_nd.h5 (generator in fixtures/README.md): ElasticIsotropic
// "solid" (E 30000, nu 0.2) and ASDConcrete3D "concrete" (E 30000, nu 0.2, fc 30), the
// owner's first nD material. Oracles: Hooke's law for the elastic one in every mode;
// for the concrete, its compressive strength fc, and a confined strength above it.

import assert from "node:assert/strict";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { MaterialTestRunner } from "../../src/main/materialTest.ts";
import { CONFINE_STEPS, NO_ND_TESTER, condensedTangent, ndLoadOf, parseNdRun, scriptOf } from "../../src/main/materialTestScript.ts";
import { openModel } from "../../src/reader/node.ts";
import { BlobStore } from "../../src/state/blobs.ts";
import { loadModel } from "../../src/state/load.ts";
import { materialTestJobOf, materialTestSlotsOf } from "../../src/state/materialTest.ts";
import { materialTestAxesOf, materialTestFormFor, materialTestProtocolOf, materialTestProtocolText } from "../../src/state/materialTestForm.ts";
import { initialState, reduce } from "../../src/state/reduce.ts";
import type { MaterialTestNd, MaterialTestProtocol } from "../../src/state/types.ts";

const model = await openModel(fileURLToPath(new URL("../../fixtures/materials_nd.h5", import.meta.url)));
const state = reduce(initialState, { type: "fileLoaded", artifact: "model", load: loadModel(model, new BlobStore()) });
const N = (name: string) => `opensees/nDMaterial/${name}`;
const nd = (mode: MaterialTestNd["mode"], confinement = 0, view: MaterialTestNd["view"] = "ThreeDimensional"): MaterialTestNd => ({ view, mode, confinement });

const E = 30000, nu = 0.2;
const G = E / (2 * (1 + nu)), lam = (E * nu) / ((1 + nu) * (1 - 2 * nu));
/** the isotropic elastic tangent, row by row, in OpenSees' 3D order (engineering shear) */
const C3 = (() => {
  const k = new Array<number>(36).fill(0);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) k[i * 6 + j] = lam + (i === j ? 2 * G : 0);
  for (let i = 3; i < 6; i++) k[i * 6 + i] = G;
  return k;
})();
const close = (a: number, b: number, rel = 1e-9) => Math.abs(a - b) <= rel * Math.max(1, Math.abs(b));

test("each loading mode is the fork's letters, the driven component, the held values and the confinement path", () => {
  assert.deepEqual(ndLoadOf(nd("uniaxialStress")), { view: "ThreeDimensional", kinds: "esssss", drive: 0, held: [0, 0, 0, 0, 0, 0], confine: [] });
  const tri = ndLoadOf(nd("triaxial", 10));
  assert.equal(tri.kinds, "esssss");
  assert.deepEqual(tri.held, [0, -10, -10, 0, 0, 0]);
  assert.equal(tri.confine.length, CONFINE_STEPS);
  assert.deepEqual(tri.confine.at(-1), [-10, -10, -10, 0, 0, 0]);
  assert.deepEqual(ndLoadOf(nd("oedometer")).kinds, "eeeeee");
  const shear = ndLoadOf(nd("simpleShear", 2));
  assert.deepEqual([shear.kinds, shear.drive, shear.held], ["sssess", 3, [-2, -2, -2, 0, 0, 0]]);
  assert.deepEqual(ndLoadOf(nd("simpleShear", 0, "PlaneStress")), { view: "PlaneStress", kinds: "sse", drive: 2, held: [0, 0, 0], confine: [] });
  assert.throws(() => ndLoadOf(nd("triaxial", 1, "PlaneStress")), /the PlaneStress view has no triaxial loading/);
  assert.throws(() => ndLoadOf(nd("uniaxialStress", 5)), /uniaxialStress holds no confinement: it must be 0/);
  assert.throws(() => ndLoadOf({ view: "AxiSymmetric", mode: "uniaxialStress", confinement: 0 }), /ThreeDimensional or PlaneStress/);
  assert.throws(() => ndLoadOf(nd("triaxial", Number.NaN)), /finite/);
});

test("the condensed tangent: E under uniaxial stress, lambda + 2G under uniaxial strain, G in shear; null when the held block is singular", () => {
  assert.ok(close(condensedTangent(C3, "esssss", 0)!, E));
  assert.ok(close(condensedTangent(C3, "eeeeee", 0)!, lam + 2 * G));
  assert.ok(close(condensedTangent(C3, "sssess", 3)!, G));
  // triaxial holds the same stresses as uniaxial stress: the same tangent
  assert.ok(close(condensedTangent(C3, "esssss", 0)!, E));
  assert.equal(condensedTangent(new Array<number>(36).fill(0), "esssss", 0), null);
});

test("an nD material's job carries its family; the script declares it with nDMaterial, drives ndTest and frames nrow lines", () => {
  const a = materialTestJobOf(state, N("concrete"), { kind: "monotonic", peak: -0.003, steps: 30, nd: nd("triaxial", 5) });
  if (!("job" in a)) return assert.fail(a.error);
  const line = a.job.materials.at(-1)!;
  assert.equal(line.family, "nDMaterial");
  assert.equal(line.type, "ASDConcrete3D");
  const built = scriptOf(a.job);
  assert.equal(built.strain, null);
  assert.equal(built.rows, 31);
  assert.deepEqual(built.nd, { kinds: "esssss", drive: 0 });
  assert.match(built.script, /^if \{\[catch \{model testNDMaterial\}\]\} \{puts "@@MT nond"; exit\}$/m);
  assert.match(built.script, /^nDMaterial ASDConcrete3D 2 30000 0\.2 -Te /m);
  assert.match(built.script, /^ndTest material 2 ThreeDimensional$/m);
  assert.equal((built.script.match(/^ndTest mixed ssssss /gm) ?? []).length, CONFINE_STEPS, "the confinement steps");
  assert.ok(built.script.indexOf("rename $mtCommand") < built.script.indexOf("ndTest"), "the dangerous commands are gone first");
});

test("the family and the loading must agree, and stress control stays uniaxial", () => {
  const values: MaterialTestProtocol = { kind: "values", values: [0.001] };
  assert.match(JSON.stringify(materialTestJobOf(state, N("solid"), values)), /needs a view and a loading mode/);
  assert.match(JSON.stringify(materialTestJobOf(state, N("solid"), { kind: "stress", targets: [1], stepsPerRamp: 1 })), /stress control runs uniaxial materials/);
  const ok = materialTestJobOf(state, N("solid"), { ...values, nd: nd("uniaxialStress") });
  if (!("job" in ok)) return assert.fail(ok.error);
  assert.throws(() => scriptOf({ ...ok.job, protocol: values }), /an nD material needs a view and a loading mode/);
  const uni = { target: 1, materials: [{ type: "Steel02", tag: 1, args: [420, 200000, 0.01] }], protocol: { ...values, nd: nd("uniaxialStress") } };
  assert.throws(() => scriptOf(uni), /a uniaxial material takes no nD loading/);
  // tags are per family: a uniaxial 1 and an nD 1 may sit in one job
  const both = { target: 1, materials: [{ type: "Steel02", tag: 1, args: [420] }, { family: "nDMaterial" as const, type: "ElasticIsotropic", tag: 1, args: [1, 0.2] }], protocol: { ...values, nd: nd("uniaxialStress") } };
  assert.match(scriptOf(both).script, /^uniaxialMaterial Steel02 1 420$\n^nDMaterial ElasticIsotropic 1 1 0\.2$/m);
  assert.throws(() => scriptOf({ ...both, materials: [both.materials[1]!, both.materials[1]!] }), /share a tag/);
  assert.throws(() => scriptOf({ ...both, materials: [{ ...both.materials[1]!, family: "section" }] }), /its family must be uniaxialMaterial or nDMaterial/);
});

const row = (i: number, e: number, s: number) => `@@MT nrow ${i} ${e} ${s} ${C3.join(" ")}`;
const good = ["@@MT build abc", row(0, 0, 0), row(1, 1e-4, 3), "@@MT end 2"];

test("an nD run is read with the driven strain and stress, and the condensed tangent", () => {
  const r = parseNdRun(good.join("\n"), 2, "esssss", 0);
  if (!r.ok) return assert.fail(r.error);
  assert.deepEqual([...r.strain], [0, 1e-4]);
  assert.deepEqual([...r.stress], [0, 3]);
  assert.ok(close(r.tangent[1]!, E));
  assert.equal(r.stopped, null);
  const stopped = parseNdRun(["@@MT build abc", row(0, 0, 0), "@@MT stop 1 ndTest: mixed control did not converge in 50 iterations", "@@MT end 1"].join("\n"), 5, "esssss", 0);
  if (!stopped.ok) return assert.fail(stopped.error);
  assert.equal(stopped.stopped, "step 1: ndTest: mixed control did not converge in 50 iterations");
  for (const [name, text, why] of [
    ["a row short of its tangent", good.join("\n").replace(` ${C3.join(" ")}\n@@MT nrow 1`, "\n@@MT nrow 1"), /row 0 is malformed/],
    ["a non-number", good.join("\n").replace("nrow 1 0.0001 3", "nrow 1 nan 3"), /returned nan/],
    ["a refused first step", "@@MT build abc\n@@MT stop 0 ndTest: the material refused the trial strain\n@@MT end 0", /the first step was refused: ndTest: the material refused/],
    ["a singular held block", ["@@MT build abc", `@@MT nrow 0 0 0 ${new Array(36).fill(0).join(" ")}`, "@@MT end 1"].join("\n"), /singular/],
    ["no closing line", good.slice(0, -1).join("\n"), /stopped after/],
  ] as const) {
    const r2 = parseNdRun(text, 2, "esssss", 0);
    assert.ok(!r2.ok && why.test(r2.error), `${name}: ${JSON.stringify(r2)}`);
  }
});

test("a fork build without the nD tester is told so by name, and a solver message carries no temporary path", () => {
  const old = parseNdRun("@@MT build 8b5e0593\n@@MT nond", 2, "esssss", 0);
  assert.deepEqual(old, { ok: false, build: "8b5e0593", error: NO_ND_TESTER });
  // the run's folder is random: two runs of one failure must read the same
  const trace = (dir: string) => `@@MT build abc\nWARNING bad input\n    (file "C:\\Users\\x\\AppData\\Local\\Temp\\${dir}\\test.tcl" line 5)`;
  const a = parseNdRun(trace("dlzOeZ"), 2, "esssss", 0);
  const b = parseNdRun(trace("FrQyP4"), 2, "esssss", 0);
  assert.ok(!a.ok && !b.ok);
  assert.equal(a.error, b.error);
  assert.match(a.error, /\(test script line 5\)/);
  assert.doesNotMatch(a.error, /AppData|test\.tcl/);
});

test("the form: nD needs a valid loading, carries it in the protocol, labels the driven component, and its slots are the nD command's", () => {
  const f = { ...materialTestFormFor(N("concrete")), kind: "monotonic" as const, peak: "-0.003", steps: "30", mode: "triaxial" as const, confinement: "5" };
  const a = materialTestProtocolOf(f, "ε11", true);
  if (!("protocol" in a)) return assert.fail(a.error);
  assert.deepEqual(a.protocol, { kind: "monotonic", peak: -0.003, steps: 30, nd: nd("triaxial", 5) });
  assert.equal(materialTestProtocolText(a.protocol), "triaxial at p = 5, 3D: monotonic to -0.003 in 30 steps");
  assert.deepEqual(materialTestProtocolOf({ ...f, view: "PlaneStress" }, "ε11", true), { error: "triaxial needs the 3D view" });
  assert.deepEqual(materialTestProtocolOf({ ...f, confinement: "x" }, "ε11", true), { error: "the confinement p must be a number" });
  assert.deepEqual(materialTestProtocolOf({ ...f, kind: "stress" }, "ε11", true), { error: "stress control runs uniaxial materials" });
  const plain = materialTestProtocolOf({ ...f, mode: "uniaxialStress", confinement: "x" }, "ε11", true);
  assert.ok("protocol" in plain && plain.protocol.kind !== "stress" && plain.protocol.nd?.confinement === 0, "an unconfined mode ignores the field");
  assert.deepEqual(materialTestAxesOf("ASDConcrete3D", nd("uniaxialStress")), ["ε11", "σ11"]);
  assert.deepEqual(materialTestAxesOf("ASDConcrete3D", nd("simpleShear")), ["γ12", "τ12"]);
  const slots = materialTestSlotsOf(state, N("solid"));
  assert.deepEqual(slots.map((x) => [x.slot, x.value]), [[0, 30000], [1, 0.2], [2, 0]]);
});

const skip = process.env["APEGMSH_OPENSEES"] ? false : "set $APEGMSH_OPENSEES to a fork OpenSees.exe with ndTest (WP-248)";
const runner = new MaterialTestRunner();
const run = async (name: string, protocol: MaterialTestProtocol) => {
  const a = materialTestJobOf(state, N(name), protocol);
  if (!("job" in a)) throw new Error(a.error);
  const r = await runner.run(a.job, 1);
  if (!r.ok) throw new Error(r.error);
  return r;
};
const peakOf = (xs: ArrayLike<number>) => Array.from(xs).reduce((p, x) => (Math.abs(x) > Math.abs(p) ? x : p), 0);

test("on the fork: the elastic material obeys Hooke in all four modes, with the tangent condensed right", { skip }, async () => {
  const ramp = (m: MaterialTestNd): MaterialTestProtocol => ({ kind: "monotonic", peak: 1e-3, steps: 4, nd: m });
  const uni = await run("solid", ramp(nd("uniaxialStress")));
  uni.stress.forEach((s, i) => assert.ok(close(s, E * uni.strain[i]!), `uniaxial step ${i}`));
  assert.ok(close(uni.tangent[4]!, E));
  const oed = await run("solid", ramp(nd("oedometer")));
  assert.ok(close(oed.stress[4]!, (lam + 2 * G) * 1e-3) && close(oed.tangent[4]!, lam + 2 * G));
  const tri = await run("solid", ramp(nd("triaxial", 5)));
  assert.ok(close(tri.stress[0]!, -5), "the drive starts from the confined state, s11 = -p");
  assert.ok(close(tri.stress[4]! - tri.stress[0]!, E * 1e-3), "then E along the drive");
  const shear = await run("solid", ramp(nd("simpleShear", 0, "PlaneStress")));
  assert.ok(close(shear.stress[4]!, G * 1e-3) && close(shear.tangent[4]!, G));
});

test("on the fork, S5 acceptance: ASDConcrete3D peaks at fc in uniaxial compression, higher when confined, in 3D and plane stress", { skip }, async () => {
  const comp = (m: MaterialTestNd): MaterialTestProtocol => ({ kind: "monotonic", peak: -0.004, steps: 80, nd: m });
  const uni = await run("concrete", comp(nd("uniaxialStress")));
  assert.equal(uni.stopped, undefined);
  assert.ok(close(peakOf(uni.stress), -30, 1e-3), `uniaxial peak ${peakOf(uni.stress)}`);
  assert.ok(close(uni.tangent[0]!, E, 1e-9), "the initial tangent is E");
  const ps = await run("concrete", comp(nd("uniaxialStress", 0, "PlaneStress")));
  assert.ok(close(peakOf(ps.stress), -30, 1e-3), `plane stress peak ${peakOf(ps.stress)}`);
  const tri = await run("concrete", comp(nd("triaxial", 5)));
  assert.equal(tri.stopped, undefined, "the confined run completes");
  assert.ok(peakOf(tri.stress) < -40, `confined peak ${peakOf(tri.stress)} above fc`);
  const cyc = await run("concrete", { kind: "cyclic", amplitudes: [0.0002, 0.001], stepsPerRamp: 20, nd: nd("uniaxialStress") });
  assert.ok(Math.max(...cyc.stress) < 3.5 && Math.max(...cyc.stress) > 1, `tension peak ${Math.max(...cyc.stress)} near ft`);
});

test("on the fork: ASDConcrete3D's -autoRegularization, with no element in the tester, stretches the softening; the slot at 1 gives the declared curve", { skip }, async () => {
  const slot = materialTestSlotsOf(state, N("concrete")).find((x) => x.label.startsWith("-autoRegularization"))!;
  assert.equal(slot.value, 100);
  const tension: MaterialTestProtocol = { kind: "monotonic", peak: 0.001, steps: 100, nd: nd("uniaxialStress") };
  const a = materialTestJobOf(state, N("concrete"), tension, [{ slot: slot.slot, label: slot.label, declared: 100, value: 1 }]);
  if (!("job" in a)) return assert.fail(a.error);
  const declared = await runner.run(a.job, 1);
  if (!declared.ok) return assert.fail(declared.error);
  const stretched = await run("concrete", tension);
  assert.ok(stretched.stress.at(-1)! > 2.5, `regularized to a 1-unit element: still ${stretched.stress.at(-1)} at 1e-3`);
  assert.ok(declared.stress.at(-1)! < 1, `as declared: softened to ${declared.stress.at(-1)} at 1e-3`);
});
