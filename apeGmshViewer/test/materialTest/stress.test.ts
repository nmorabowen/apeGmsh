// ADR 0135 D6, slice S3: stress control. The fork's tester is strain-driven, so a
// fixed driver searches each step's strain with exactly STRESS_TRIALS calls, the
// last of which commits. The pure parts run anywhere; the acceptance cases run on
// the fork where $APEGMSH_OPENSEES names it.
//
// Acceptance (ADR 0135 D6 and its probe table): on Concrete02 (peak -30) the target
// -29 is reached, which a doubling search misses, and -35 stops the run as beyond
// the peak; LadrunoUniaxialJ2 and Steel02 follow a cyclic stress path through yield
// and reversal.

import assert from "node:assert/strict";
import { test } from "node:test";
import { MaterialTestRunner } from "../../src/main/materialTest.ts";
import { MAX_STRESS_STEPS, MaterialTestRefused, parseStressRun, scriptOf, STRESS_TRIALS, stressPath } from "../../src/main/materialTestScript.ts";
import type { MaterialTestJob, MaterialTestLine, MaterialTestProtocol } from "../../src/state/types.ts";

const concrete: MaterialTestLine = { type: "Concrete02", tag: 1, args: [-30, -0.002, -6, -0.006, 0.1, 3, 300] };
const steel: MaterialTestLine = { type: "Steel02", tag: 1, args: [420, 200000, 0.01, 18, 0.925, 0.15] };
const j2: MaterialTestLine = { type: "LadrunoUniaxialJ2", tag: 1, args: [200000, "-iso", "voce", 420, 100, 10, 0, "-kin", 1, 5000, 50] };
const job = (m: MaterialTestLine, protocol: MaterialTestProtocol): MaterialTestJob => ({ target: m.tag, materials: [m], protocol });

test("the stress path: each leg from the previous target, stepsPerRamp steps, the zero not counted", () => {
  assert.deepEqual(stressPath({ kind: "stress", targets: [10, -10], stepsPerRamp: 2 }), [5, 10, 0, -10]);
  assert.throws(() => stressPath({ kind: "stress", targets: [], stepsPerRamp: 2 }), MaterialTestRefused);
  assert.throws(() => stressPath({ kind: "stress", targets: [Number.NaN], stepsPerRamp: 2 }), MaterialTestRefused);
  assert.throws(() => stressPath({ kind: "stress", targets: [1, -1], stepsPerRamp: MAX_STRESS_STEPS }), /at most 5000 steps/);
});

test("the script commits every STRESS_TRIALS calls, prints the zero row without a call, and carries only the app's numbers", () => {
  const { script, strain, rows } = scriptOf(job(concrete, { kind: "stress", targets: [-29], stepsPerRamp: 4 }));
  assert.equal(strain, null, "the fork finds the strains");
  assert.equal(rows, 5);
  assert.match(script, new RegExp(`^model testUniaxial ${STRESS_TRIALS}$`, "m"));
  assert.match(script, /^foreach mtTarget0 \{-7\.25 -14\.5 -21\.75 -29\} \{$/m);
  const beforeLoop = script.slice(0, script.indexOf("foreach mtTarget0"));
  assert.equal((beforeLoop.match(/^strainUniaxialTest /gm) ?? []).length, 0, "no strain call at the top level, outside the driver's procs");
  assert.ok(script.indexOf("rename $mtCommand") < script.indexOf("proc mtTry"), "the dangerous commands are gone before any driver code");
});

const good = ["@@MT build abc", "@@MT srow 0 0 0 2e5", "@@MT srow 1 -1e-4 -20 2e5", "@@MT srow 2 -2e-3 -29 0", "@@MT end 3"];

test("a complete stress run is read with the strains the driver found", () => {
  const r = parseStressRun(good.join("\r\n"), 3);
  assert.ok(r.ok);
  assert.deepEqual([...r.strain], [0, -1e-4, -2e-3]);
  assert.deepEqual([...r.stress], [0, -20, -29]);
  assert.equal(r.stopped, null);
});

test("a stop after the rows it reached keeps them, with the driver's reason", () => {
  const text = ["@@MT build abc", "@@MT srow 0 0 0 2e5", "@@MT srow 1 -1e-4 -20 2e5", "@@MT stop 2 the target -35 is beyond the peak: the most the material reaches on this path is -30.0", "@@MT end 2"].join("\n");
  const r = parseStressRun(text, 5);
  assert.ok(r.ok);
  assert.deepEqual([...r.stress], [0, -20]);
  assert.equal(r.stopped, "step 2: the target -35 is beyond the peak: the most the material reaches on this path is -30.0");
});

for (const [name, text] of [
  ["a missing row with no stop", good.filter((_, i) => i !== 3).join("\n").replace("end 3", "end 2")],
  ["a stop naming another step", good.join("\n").replace("@@MT srow 2 -2e-3 -29 0\n@@MT end 3", "@@MT stop 7 nope\n@@MT end 2")],
  ["a row without its strain", good.join("\n").replace("srow 1 -1e-4 -20 2e5", "srow 1 -20 2e5")],
  ["a non-number strain", good.join("\n").replace("-1e-4", "nan")],
  ["no closing line", good.slice(0, -1).join("\n")],
  ["lines after the end", `${good.join("\n")}\n@@MT srow 3 0 0 0`],
] as const) {
  test(`no curve from ${name}`, () => assert.equal(parseStressRun(text, 3).ok, false));
}

const skip = process.env["APEGMSH_OPENSEES"] ? false : "set $APEGMSH_OPENSEES to the fork's OpenSees.exe";
const run = (m: MaterialTestLine, targets: number[], stepsPerRamp: number) => new MaterialTestRunner().run(job(m, { kind: "stress", targets, stepsPerRamp }), 1);
const close = (a: number, b: number) => Math.abs(a - b) <= 1e-6 * Math.max(1, Math.abs(b));

test("D6 acceptance on the fork: Concrete02 reaches -29, which a doubling search steps over", { skip }, async () => {
  for (const steps of [1, 10]) {
    const r = await run(concrete, [-29], steps);
    assert.ok(r.ok, r.ok ? "" : r.error);
    assert.equal(r.stopped, undefined);
    assert.ok(close(r.stress.at(-1)!, -29), `stress ${r.stress.at(-1)}`);
    assert.ok(r.strain.at(-1)! > -0.002, "on the rising branch, before the peak strain");
  }
});

test("D6 acceptance on the fork: -35 on Concrete02 stops as beyond the peak, keeping the steps before it", { skip }, async () => {
  const r = await run(concrete, [-35], 10);
  assert.ok(r.ok, r.ok ? "" : r.error);
  assert.match(r.stopped ?? "", /^step 9: the target -31\.5 is beyond the peak: the most the material reaches on this path is -30/);
  assert.equal(r.stress.length, 9, "row 0 and steps 1 to 8");
  assert.ok(close(r.stress.at(-1)!, -28));
});

test("on the fork: J2 and Steel02 follow a cyclic stress path through yield and reversal, each step committed at its target", { skip }, async () => {
  for (const m of [j2, steel]) {
    const r = await run(m, [450, -450, 0], 10);
    assert.ok(r.ok, r.ok ? "" : r.error);
    assert.equal(r.stopped, undefined, m.type);
    const targets = [0, ...stressPath({ kind: "stress", targets: [450, -450, 0], stepsPerRamp: 10 })];
    r.stress.forEach((s, i) => assert.ok(close(s, targets[i]!), `${m.type} step ${i}: ${s} for ${targets[i]}`));
    // the first step after the peak unloads by 90: exactly at E for J2, whose unloading is
    // elastic; within a few percent for Steel02, whose Menegotto-Pinto curve bends at once
    const de = r.strain[10]! - r.strain[11]!;
    const rel = Math.abs(de - 90 / 200000) / (90 / 200000);
    assert.ok(rel < (m === j2 ? 1e-6 : 0.05), `${m.type} unloads at ${de}, ${rel} off E`);
  }
});
