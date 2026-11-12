// Stress and strain tensors (state/tensor.ts): the component names that
// make a tensor, and the eigen decomposition against closed forms.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { assembleTensor, derivedGroupValues, derivedTensorOf, eigenSym3, tensorFields, tensorSlotOf, vonMises, type TensorGroup, type Vec3 } from "../src/state/tensor.ts";

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

// -- VA6d (#2766): the derived von Mises scalars ---------------------------------------

test("von Mises, closed forms: uniaxial |s|, pure shear sqrt(3) t, hydrostatic 0; a strain's equivalent sqrt(2/3 e:e)", () => {
  close(vonMises([7, 0, 0, 0, 0, 0], "stress"), 7);
  close(vonMises([-7, 0, 0, 0, 0, 0], "stress"), 7);
  close(vonMises([0, 0, 0, 0, 0, 4], "stress"), Math.sqrt(3) * 4);
  close(vonMises([-3, -3, -3, 0, 0, 0], "stress"), 0);
  close(vonMises([5, -5, 0, 0, 0, 0], "stress"), Math.sqrt(3) * 5);
  // An isochoric uniaxial strain (e, -e/2, -e/2): its equivalent strain is e.
  close(vonMises([2, -1, -1, 0, 0, 0], "strain"), 2);
  // An engineering shear g is a tensor shear g/2: sqrt(2/3 (2 (g/2)^2)) = g / sqrt(3).
  const g = 0.6;
  close(vonMises(assembleTensor("strain", (v) => (v === 3 ? g : null))!, "strain"), g / Math.sqrt(3));
});

test("oracle: Python's _derived.compute on seeded random 3-D tensors, stress and strain (the strain's three engineering shears halved)", () => {
  // apeGmsh.results._derived.compute(name, columns, ndm=3) on numpy default_rng(2766).normal(scale=10, size=(6, 6)),
  // each row one case in Voigt order xx, yy, zz, xy, yz, xz (the VA6d oracle script, in the PR body).
  const VOIGT = [
    [15.300365287886482, -15.81199186730437, -14.834084971757486, 5.78283692651103, 1.106010674237234, 15.348598266026237],
    [5.7176423985416625, 5.5451254957350224, 8.42039667350909, 7.517374426467772, -4.110715109859714, -9.311655080458682],
    [-7.301970280030925, -0.1354031904100332, -0.46444731555406404, 1.3005125610504662, 0.796819125070053, -11.621399855591992],
    [1.5175006684504138, -5.392175961863524, 8.12010624141344, -3.4424915404369063, 3.651341101608456, 3.07580157753478],
    [1.6403244755088013, -10.358311027926579, 0.21518290197838574, 17.53161231886373, 4.6150643419658435, 17.844842250960724],
    [5.28707415759561, -14.442189258544802, 6.127732010746247, 0.0271385463882071, 4.046063817839199, -16.898001764113758],
  ];
  const PYTHON = {
    stress: [41.82393901625915, 22.0940798746608, 21.476950477599978, 15.520653545718789, 45.49925780922802, 36.22537230192857],
    strain: [22.521032436332263, 7.53916519218446, 8.223198303724505, 8.509931382600827, 16.522301580633236, 16.77261624329922],
  } as const;
  for (const family of ["stress", "strain"] as const) {
    VOIGT.forEach((t, i) => close(vonMises(assembleTensor(family, (v) => t[v]!)!, family), PYTHON[family][i]!));
  }
});

const gg = (id: string, components: string[], nGauss = 1) => ({ id, components, refused: [], nElements: 1, nGauss, localAxes: false, naturalCoords: null });

test("derivedTensorOf: a complete 2-D or 3-D tensor of the field's own components; null for a field with none of the family", () => {
  const plane = derivedTensorOf([gg("q", ["stress_xx", "stress_yy", "stress_xy"]), gg("t", ["stress_xx", "stress_yy", "stress_xy"])], ["stress_xx", "stress_yy", "stress_xy"], "stress");
  assert.ok(plane?.ok);
  assert.deepEqual(plane.groups.map((x) => x.id), ["q", "t"]);
  assert.deepEqual(plane.components, ["stress_xx", "stress_yy", "stress_xy"]);
  const solid = ["stress/sigma11", "stress/sigma22", "stress/sigma33", "stress/sigma12", "stress/sigma23", "stress/sigma13"];
  const s3 = derivedTensorOf([gg("stress/56-Brick[1:0:0]", solid)], solid, "stress");
  assert.ok(s3?.ok);
  assert.deepEqual(s3.groups[0]!.slots, solid);
  assert.equal(derivedTensorOf([gg("q", ["stress_xx", "stress_yy", "stress_xy"])], ["stress_xx", "stress_yy", "stress_xy"], "strain"), null);
  // Only the field's own names count: another field's strain in the same group is not this field's tensor.
  assert.equal(derivedTensorOf([gg("q", ["stress_xx", "strain_xx", "strain_yy", "strain_xy"])], ["stress_xx"], "strain"), null);
});

test("derivedTensorOf refuses, by name, an incomplete or ambiguous tensor: never a value from partial components", () => {
  const no = (groups: ReturnType<typeof gg>[], names: string[], family: "stress" | "strain") => {
    const t = derivedTensorOf(groups, names, family);
    assert.ok(t && !t.ok, JSON.stringify(t));
    return t.reason;
  };
  // The in-plane xy missing (elem_solid and extrap_solid record only xx and yy).
  assert.match(no([gg("hex_G8", ["stress_xx", "stress_yy"], 8)], ["stress_xx", "stress_yy"], "stress"), /group hex_G8: no xy stress component, so its tensor is not known/);
  // A 3-D tensor with some of the out-of-plane components and not the others.
  const five = ["stress_xx", "stress_yy", "stress_zz", "stress_xy", "stress_yz"];
  assert.match(no([gg("h", five)], five, "stress"), /group h records zz, yz but no xz stress component/);
  // One slot named twice.
  const twice = ["stress/sigma11", "stress/sigma22", "stress/sigma12", "stress/sigma21"];
  assert.match(no([gg("stress/1-Q[1:0:0]", twice)], twice, "stress"), /name one stress slot twice/);
  // Two tensors of one family in one field.
  const a = ["s/sigma11", "s/sigma22", "s/sigma12"];
  assert.match(no([gg("A/1", a), gg("B/1", a)], a, "stress"), /its groups make 2 stress tensors \(A, B\)/);
});

test("derivedGroupValues: von Mises at every point of a group; a non-finite component is no data; a missing or short column is an error", () => {
  const group: TensorGroup = { id: "g", slots: ["sx", "sy", null, "txy", null, null] };
  const values = { sx: { g: Float64Array.from([10, 0, NaN]) }, sy: { g: Float64Array.from([0, 0, 1]) }, txy: { g: Float64Array.from([0, 2, 1]) } };
  const v = derivedGroupValues("von_mises_stress", group, 3, values);
  close(v[0]!, 10);
  close(v[1]!, Math.sqrt(3) * 2);
  assert.ok(Number.isNaN(v[2]!));
  // The strain halves the engineering shear: a gamma of 2 is a tensor shear of 1, so 2 / sqrt(3).
  close(derivedGroupValues("von_mises_strain", group, 3, values)[1]!, 2 / Math.sqrt(3));
  assert.throws(() => derivedGroupValues("von_mises_stress", group, 3, { sx: values.sx, sy: values.sy }), /the step data has no txy for group g/);
  assert.throws(() => derivedGroupValues("von_mises_stress", group, 4, values), /sx of group g has 3 values, not 4/);
});
