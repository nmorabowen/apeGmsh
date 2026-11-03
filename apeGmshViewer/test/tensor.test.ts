// Stress and strain tensors (state/tensor.ts): the component names that
// make a tensor, and the eigen decomposition against closed forms.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { assembleTensor, eigenSym3, tensorFields, tensorSlotOf, type Vec3 } from "../src/state/tensor.ts";

const close = (a: number, b: number, tol = 1e-12) => assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), `${a} vs ${b}`);
const closeVec = (a: Vec3, b: readonly number[], tol = 1e-12) => a.forEach((x, i) => close(x, b[i]!, tol));

test("the digit and axis spellings of the fork name their tensor slot; a plastic strain or a scalar is no slot", () => {
  assert.deepEqual(tensorSlotOf("stress/sigma11"), { family: "stress", voigt: 0 });
  assert.deepEqual(tensorSlotOf("stress/sigma12"), { family: "stress", voigt: 3 });
  assert.deepEqual(tensorSlotOf("stress/sigma31"), { family: "stress", voigt: 5 });
  assert.deepEqual(tensorSlotOf("strain/eta22"), { family: "strain", voigt: 1 });
  assert.deepEqual(tensorSlotOf("strain/eps23"), { family: "strain", voigt: 4 });
  assert.deepEqual(tensorSlotOf("strain/gamma_xy"), { family: "strain", voigt: 3 });
  assert.deepEqual(tensorSlotOf("stress/SIGMA_ZZ"), { family: "stress", voigt: 2 });
  assert.deepEqual(tensorSlotOf("stress/stress_xz"), { family: "stress", voigt: 5 });
  assert.equal(tensorSlotOf("strain/epsp11"), null);
  assert.equal(tensorSlotOf("material/eqpstrain"), null);
  assert.equal(tensorSlotOf("stress/sigma14"), null);
});

test("a RESULT's groups make one field; a group without the in-plane tensor is refused by name", () => {
  const g = (id: string, components: string[]) => ({ id, components, refused: [], nElements: 1, nGauss: 4, localAxes: false, naturalCoords: null });
  const fields = tensorFields([
    g("stress/31-FourNodeQuad[201:0:0]", ["stress/sigma11", "stress/sigma22", "stress/sigma12"]),
    g("strain/31-FourNodeQuad[201:0:0]", ["strain/eta11", "strain/eta22", "strain/eta12"]),
    g("stress/56-Brick[300:0:0]", ["stress/sigma11", "stress/sigma22", "stress/sigma33", "stress/sigma12", "stress/sigma23", "stress/sigma13"]),
    g("stress/99-Odd[1:0:0]", ["stress/sigma11", "stress/sigma22"]),
    g("material/1-X[1:0:0]", ["material/eqpstrain"]),
  ]);
  assert.deepEqual(fields.map((f) => [f.name, f.family, f.groups.map((x) => x.id)]), [
    ["stress", "stress", ["stress/31-FourNodeQuad[201:0:0]", "stress/56-Brick[300:0:0]"]],
    ["strain", "strain", ["strain/31-FourNodeQuad[201:0:0]"]],
  ]);
  assert.deepEqual(fields[0]!.groups[0]!.slots, ["stress/sigma11", "stress/sigma22", null, "stress/sigma12", null, null]);
  assert.equal(fields[0]!.components.length, 6);
  assert.match(fields[0]!.refused[0]!, /stress\/99-Odd.*no xy stress component/);
});

test("a group of both tensors gives one to each family's field; a field whose every group is refused is kept, for its notes", () => {
  const g = (id: string, components: string[]) => ({ id, components, refused: [], nElements: 1, nGauss: 1, localAxes: false, naturalCoords: null });
  const both = tensorFields([g("g0", ["stress_xx", "stress_yy", "stress_xy", "strain_xx", "strain_yy", "strain_xy"])]);
  assert.deepEqual(both.map((f) => [f.name, f.family, f.groups.length, f.refused]), [
    ["stress", "stress", 1, []],
    ["strain", "strain", 1, []],
  ]);
  const refusedOnly = tensorFields([g("stress/1-X[1:0:0]", ["stress/sigma11", "stress/sigma11"])]);
  assert.equal(refusedOnly.length, 1);
  assert.deepEqual(refusedOnly[0]!.groups, []);
  assert.match(refusedOnly[0]!.refused[0]!, /one stress slot twice/);
});

test("oracle: 300 random 3-D tensors give Python's principal_frame values and its signed (canonical) vectors", () => {
  const O = JSON.parse(readFileSync(fileURLToPath(new URL("../fixtures/principal_random.oracle.json", import.meta.url)), "utf8")) as {
    cases: { voigt: number[]; values: number[]; vectors: number[][] }[];
  };
  assert.equal(O.cases.length, 300);
  for (const c of O.cases) {
    const e = eigenSym3(c.voigt);
    const scale = Math.max(...c.values.map(Math.abs));
    c.values.forEach((v, k) => close(e.values[k]!, v, 1e-12 * Math.max(1, scale) / Math.max(1, Math.abs(v))));
    // Signed: the canonical sign is what the `arrow` head draws.
    c.vectors.forEach((v, k) => closeVec(e.vectors[k]!, v, 1e-9));
  }
});

test("a diagonal tensor: its values sorted and the axes as they are", () => {
  const e = eigenSym3([1, 3, -2, 0, 0, 0]);
  closeVec(e.values, [3, 1, -2]);
  closeVec(e.vectors[0], [0, 1, 0]);
  closeVec(e.vectors[1], [1, 0, 0]);
  closeVec(e.vectors[2], [0, 0, 1]);
});

test("plane stress: Mohr's circle values and the angle tan 2θ = 2τ / (σx − σy)", () => {
  const sx = 40, sy = -10, txy = 30;
  const e = eigenSym3([sx, sy, 0, txy, 0, 0]);
  const c = (sx + sy) / 2, r = Math.hypot((sx - sy) / 2, txy);
  closeVec(e.values, [c + r, 0, c - r]);
  const th = 0.5 * Math.atan2(2 * txy, sx - sy);
  closeVec(e.vectors[0], [Math.cos(th), Math.sin(th), 0]);
  closeVec(e.vectors[1], [0, 0, 1]);
  // The p3 axis is perpendicular; its sign is the canonical one (largest component positive).
  const v3 = [-Math.sin(th), Math.cos(th), 0];
  closeVec(e.vectors[2], Math.abs(v3[0]!) > Math.abs(v3[1]!) && v3[0]! < 0 ? v3.map((x) => -x) : v3);
});

test("pure shear: ±τ on the diagonals; the tie of |x| = |y| makes x positive", () => {
  const e = eigenSym3([0, 0, 0, 1, 0, 0]);
  closeVec(e.values, [1, 0, -1]);
  closeVec(e.vectors[0], [Math.SQRT1_2, Math.SQRT1_2, 0]);
  closeVec(e.vectors[2], [Math.SQRT1_2, -Math.SQRT1_2, 0]);
});

test("a rotated 3-D tensor: R diag(5, 2, −1) Rᵀ gives back the values and R's columns, up to sign", () => {
  // R from Euler angles, so no axis is aligned with x, y or z.
  const [a, b, c] = [0.7, -0.4, 1.9];
  const Rz = (t: number) => [[Math.cos(t), -Math.sin(t), 0], [Math.sin(t), Math.cos(t), 0], [0, 0, 1]];
  const Rx = (t: number) => [[1, 0, 0], [0, Math.cos(t), -Math.sin(t)], [0, Math.sin(t), Math.cos(t)]];
  const mul = (X: number[][], Y: number[][]) => X.map((r) => [0, 1, 2].map((j) => r.reduce((s, x, k) => s + x * Y[k]![j]!, 0)));
  const R = mul(mul(Rz(a), Rx(b)), Rz(c));
  const D = [5, 2, -1];
  const T = [0, 1, 2].map((i) => [0, 1, 2].map((j) => R[i]!.reduce((s, x, k) => s + x * D[k]! * R[j]![k]!, 0)));
  const e = eigenSym3([T[0]![0]!, T[1]![1]!, T[2]![2]!, T[0]![1]!, T[1]![2]!, T[0]![2]!]);
  closeVec(e.values, D, 1e-12);
  e.vectors.forEach((v, i) => {
    const col = [R[0]![i]!, R[1]![i]!, R[2]![i]!];
    const dot = v.reduce((s, x, k) => s + x * col[k]!, 0);
    close(Math.abs(dot), 1, 1e-12);
    // Canonical: the largest component is positive.
    const k = [0, 1, 2].reduce((m, j) => (Math.abs(v[j]!) > Math.abs(v[m]!) ? j : m), 0);
    assert.ok(v[k]! > 0);
  });
});

test("a hydrostatic tensor: three equal values and an orthonormal frame", () => {
  const e = eigenSym3([-7, -7, -7, 0, 0, 0]);
  closeVec(e.values, [-7, -7, -7]);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) close(e.vectors[i]!.reduce((s, x, k) => s + x * e.vectors[j]![k]!, 0), i === j ? 1 : 0);
});

test("a strain's engineering shear is halved; a slot the group does not record is 0; a non-finite value refuses the point", () => {
  const read = (vals: (number | null)[]) => (v: number) => vals[v] ?? null;
  assert.deepEqual([...assembleTensor("strain", read([1, 2, null, 4, null, null]))!], [1, 2, 0, 2, 0, 0]);
  assert.deepEqual([...assembleTensor("stress", read([1, 2, null, 4, null, null]))!], [1, 2, 0, 4, 0, 0]);
  assert.equal(assembleTensor("stress", read([1, NaN, null, 4, null, null])), null);
});
