// ADR 0135 S1 against the real fork. It runs only where $APEGMSH_OPENSEES names an
// executable: CI's app lane has none, so this is a local check, not a CI gate.

import assert from "node:assert/strict";
import { test } from "node:test";
import { MaterialTestRunner } from "../../src/main/materialTest.ts";
import type { MaterialTestLine, MaterialTestProtocol } from "../../src/state/types.ts";

const skip = process.env["APEGMSH_OPENSEES"] ? false : "set $APEGMSH_OPENSEES to the fork's OpenSees.exe";
const cyclic = (a: number): MaterialTestProtocol => ({ kind: "cyclic", amplitudes: [a / 2, a], stepsPerRamp: 100 });
const steel: MaterialTestLine = { type: "Steel02", tag: 1, args: [420, 200000, 0.01, 18, 0.925, 0.15] };
const j2: MaterialTestLine = { type: "LadrunoUniaxialJ2", tag: 1, args: [200000, "-iso", "voce", 420, 100, 10, 0, "-kin", 1, 5000, 50] };
const buckling: MaterialTestLine = { type: "LadrunoRebarBuckling", tag: 2, args: [1, "-lsr", 8, "-fy", 420, "-E", 200000] };

const cases: [string, MaterialTestLine[], MaterialTestProtocol][] = [
  ["Steel02", [steel], cyclic(0.03)],
  ["Concrete02", [{ type: "Concrete02", tag: 1, args: [-30, -0.002, -6, -0.006, 0.1, 3, 300] }], cyclic(0.004)],
  ["LadrunoUniaxialJ2", [j2], cyclic(0.03)],
  ["LadrunoRebarBuckling over Steel02", [steel, buckling], cyclic(0.03)],
  ["LadrunoRebarBuckling over LadrunoUniaxialJ2", [j2, buckling], cyclic(0.03)],
  ["LadrunoBondSlip", [{ type: "LadrunoBondSlip", tag: 1, args: [10, 1, 2, 5, 2, 0.4] }], { kind: "monotonic", peak: 6, steps: 200 }],
  ["LadrunoCohesiveHinge", [{ type: "LadrunoCohesiveHinge", tag: 1, args: [100, 1] }], cyclic(0.05)],
];

for (const [name, materials, protocol] of cases) {
  test(`the fork runs ${name}`, { skip }, async () => {
    const r = await new MaterialTestRunner().run({ target: materials.at(-1)!.tag, materials, protocol }, 1);
    assert.ok(r.ok, r.ok ? "" : r.error);
    assert.match(r.build, /^[0-9a-f]{7,}$/);
    assert.equal(r.stress.length, r.strain.length);
    assert.ok(r.stress.some((s) => s !== 0));
  });
}

test("Steel02 starts on its elastic line", { skip }, async () => {
  const r = await new MaterialTestRunner().run({ target: 1, materials: [steel], protocol: { kind: "monotonic", peak: 0.001, steps: 2 } }, 1);
  assert.ok(r.ok, r.ok ? "" : r.error);
  assert.ok(Math.abs(r.stress[2]! - 200) < 2, `stress ${r.stress[2]}`); // Menegotto-Pinto curves slightly before yield
  assert.ok(Math.abs(r.tangent[1]! - 200000) < 2000, `tangent ${r.tangent[1]}`);
});

test("a material the fork rejects gives its message and no curve", { skip }, async () => {
  const r = await new MaterialTestRunner().run({ target: 1, materials: [{ type: "NoSuchMaterial", tag: 1, args: [1] }], protocol: { kind: "values", values: [0.1] } }, 1);
  assert.ok(!r.ok);
  assert.match(r.error, /stopped after 0 of 2/);
});
