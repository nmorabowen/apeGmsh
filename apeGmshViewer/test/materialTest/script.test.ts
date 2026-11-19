// ADR 0135 D3: what reaches the Tcl script, and what is believed of the answer.

import assert from "node:assert/strict";
import { test } from "node:test";
import { MAX_STEPS, MaterialTestRefused, parseRun, scriptOf, strainPath } from "../../src/main/materialTestScript.ts";
import type { MaterialTestProtocol } from "../../src/state/types.ts";

const protocol: MaterialTestProtocol = { kind: "monotonic", peak: 0.01, steps: 2 };
const steel = { type: "Steel02", tag: 1, args: [420, 200000, 0.01] };
const job = (over: object = {}) => ({ target: 1, materials: [steel], protocol, ...over });

test("the script declares the material, removes the dangerous commands first, and imposes the path", () => {
  const { script, strain } = scriptOf(job());
  assert.deepEqual(strain, [0, 0.005, 0.01]);
  assert.match(script, /^uniaxialMaterial Steel02 1 420 200000 0\.01$/m);
  assert.match(script, /^uniaxialTest 1$/m);
  assert.match(script, /^foreach mtStrain \{0 0\.005 0\.01\} \{$/m);
  const removed = script.indexOf("rename $mtCommand");
  assert.ok(removed > 0 && removed < script.indexOf("model testUniaxial"));
  for (const c of ["exec", "open", "socket", "source", "file", "load", "interp"]) assert.match(script.split("\n")[2]!, new RegExp(`\\b${c}\\b`));
});

test("flags and words pass; integers stay integers, so tag slots parse", () => {
  const wrap = { type: "LadrunoRebarBuckling", tag: 2, args: [1, "-lsr", 8.5, "-model", "dm"] };
  assert.match(scriptOf(job({ target: 2, materials: [steel, wrap] })).script, /^uniaxialMaterial LadrunoRebarBuckling 2 1 -lsr 8\.5 -model dm$/m);
});

for (const bad of ["[exec calc]", "$env(PATH)", "a;b", "a b", "{x}", '"x"', "a\nb", "..\\x", "C:/x", "", "-", "1e5x", "\\x"]) {
  test(`an argument ${JSON.stringify(bad)} refuses the run and reaches no script`, () => {
    assert.throws(() => scriptOf(job({ materials: [{ ...steel, args: [1, bad] }] })), (e: Error) => e instanceof MaterialTestRefused && /tag 1 .*argument 1/.test(e.message));
  });
  test(`a type ${JSON.stringify(bad)} refuses the run`, () => {
    assert.throws(() => scriptOf(job({ materials: [{ ...steel, type: bad }] })), MaterialTestRefused);
  });
}

for (const bad of [NaN, Infinity, -Infinity, null, {}, [1], true]) {
  test(`an argument ${String(bad)} refuses the run`, () => {
    assert.throws(() => scriptOf(job({ materials: [{ ...steel, args: [bad] }] })), MaterialTestRefused);
  });
}

test("a malformed job is refused, whatever the page sends", () => {
  for (const j of [null, 1, [], {}, job({ materials: [] }), job({ target: 9 }), job({ materials: [steel, steel] }), job({ protocol: null }), job({ protocol: { kind: "x" } }), job({ materials: [{ ...steel, tag: "1;exec" }] }), job({ materials: [{ ...steel, tag: 1.5 }] })]) {
    assert.throws(() => scriptOf(j), MaterialTestRefused, JSON.stringify(j));
  }
});

test("protocols: monotonic, a closed cycle per amplitude, pasted values, and the step limit", () => {
  assert.deepEqual(strainPath({ kind: "cyclic", amplitudes: [1, 2], stepsPerRamp: 1 }), [0, 1, 0, -1, 0, 2, 0, -2, 0]);
  assert.deepEqual(strainPath({ kind: "values", values: [0.1, -0.1] }), [0, 0.1, -0.1]);
  assert.throws(() => strainPath({ kind: "monotonic", peak: 1, steps: 0 }), MaterialTestRefused);
  assert.throws(() => strainPath({ kind: "monotonic", peak: NaN, steps: 1 }), MaterialTestRefused);
  assert.throws(() => strainPath({ kind: "cyclic", amplitudes: [1, 1], stepsPerRamp: MAX_STEPS }), MaterialTestRefused);
});

const good = ["banner", "@@MT build abc", "@@MT row 0 0.0e+00 2.0e+05", "@@MT row 1 1.0e+02 2.0e+05", "@@MT end 2"];

test("a complete answer is read, with CRLF too", () => {
  const r = parseRun(good.join("\r\n"), 2);
  assert.ok(r.ok);
  assert.equal(r.build, "abc");
  assert.deepEqual([...r.stress], [0, 100]);
  assert.deepEqual([...r.tangent], [200000, 200000]);
});

const without = (i: number) => good.filter((_, k) => k !== i).join("\n");
const cases: [string, string, RegExp][] = [
  ["no framing at all (exit 0 after a script error)", 'invalid command name "x"', /did not start.*invalid command/],
  ["no build line", without(1), /did not start/],
  ["a missing row", without(3), /stopped after 1 of 2/],
  ["no closing line", without(4), /stopped after 2 of 2/],
  ["a wrong closing count", good.join("\n").replace("end 2", "end 3"), /stopped after 2 of 2/],
  ["rows out of order", good.join("\n").replace("row 1", "row 5"), /row 1 is malformed/],
  ["a row after the end", `${good.join("\n")}\n@@MT row 2 0 0`, /stopped/],
  ["too many rows", good.join("\n").replace("@@MT end 2", "@@MT row 2 0 0\n@@MT end 3"), /row 2 is malformed/],
  ["a non-number from the material", good.join("\n").replace("1.0e+02", "-nan(ind)"), /step 1 .*-nan\(ind\)/],
  ["an infinite stress", good.join("\n").replace("1.0e+02", "inf"), /step 1/],
];
for (const [name, text, why] of cases) {
  test(`no curve from ${name}`, () => {
    const r = parseRun(text, 2);
    assert.equal(r.ok, false);
    if (!r.ok) assert.match(r.error, why);
  });
}

test("the solver's own words after the build line are the message, not its banner", () => {
  const r = parseRun("banner\n@@MT build abc\nWARNING bad material\n", 2);
  assert.ok(!r.ok);
  assert.match(r.error, /WARNING bad material/);
  assert.doesNotMatch(r.error, /banner/);
  assert.equal(r.build, "abc");
});

test("an executable without ladrunoBuild is named as such", () => {
  const r = parseRun("@@MT nofork\n", 2);
  assert.ok(!r.ok);
  assert.match(r.error, /not a Ladruno fork build/);
});
