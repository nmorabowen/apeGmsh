// Stress and strain tensors at the Gauss points: which `.ladruno` Gauss
// components make a tensor, and the eigen decomposition of a symmetric 3x3, as
// Python's `principal_frame` (src/apeGmsh/results/_derived.py) does it: the
// tensor is built from the Voigt components, a strain's shear terms halved (the
// file's shear strain is engineering, gamma = 2 eps), a component the file does
// not record taken as 0 (a 2-D element's zz: plane stress for a stress, plane
// strain for a strain, Python's `plane=None`); the values are sorted
// p1 >= p2 >= p3 and each eigenvector takes the sign that makes its largest
// component positive, so a direction is an axis, never a sense. Pure: tested
// without a window (state/principal.ts draws the arrows).

import type { GaussGroup } from "./types.ts";

export type Vec3 = [number, number, number];

/** The two tensors drawn: a stress and a (total) strain. */
export type TensorFamily = "stress" | "strain";

/** A component's place in the symmetric tensor, Voigt order xx, yy, zz, xy, yz, xz (0..5). */
export interface TensorSlot {
  family: TensorFamily;
  voigt: number;
}

// Python _ladruno_element_io `_CONTINUUM_DIGIT_RE` / `_CONTINUUM_AXIS_RE` and `_KIND_TO_ROOT`, the stress
// and strain roots only: a plastic strain (`epsp`, `pstrain`) is not one of the two tensors drawn.
const DIGIT = /^(sigma|eta|epsilon|eps)([123])([123])$/;
const AXIS = /^(sigma|epsilon|eps|gamma|stress|strain)_([xyz])([xyz])$/;
const FAMILY: Record<string, TensorFamily> = { sigma: "stress", stress: "stress", eta: "strain", eps: "strain", epsilon: "strain", gamma: "strain", strain: "strain" };
const AXIS_INDEX: Record<string, number> = { x: 0, y: 1, z: 2, "1": 0, "2": 1, "3": 2 };
/** Voigt index of the pair (i, j), either order. */
const VOIGT = [
  [0, 3, 5],
  [3, 1, 4],
  [5, 4, 2],
];

/** The tensor slot a Gauss component's label names (`stress/sigma12` -> stress xy), case-insensitive; null for any other. */
export function tensorSlotOf(component: string): TensorSlot | null {
  const label = component.slice(component.lastIndexOf("/") + 1).trim().toLowerCase();
  const m = DIGIT.exec(label) ?? AXIS.exec(label);
  if (!m) return null;
  return { family: FAMILY[m[1]!]!, voigt: VOIGT[AXIS_INDEX[m[2]!]!]![AXIS_INDEX[m[3]!]!]! };
}

/** One group of a tensor field: its components by Voigt slot (null where the group does not record it). */
export interface TensorGroup {
  id: string;
  slots: (string | null)[];
}

/**
 * A stress or strain field of the stage: the groups of one `.ladruno` RESULT
 * (`stress`, `strain`; each element class its own group) whose components make
 * a tensor. A group needs xx, yy and xy (the in-plane tensor); the others are 0
 * when absent. `refused` names a group of the field that cannot be drawn; a
 * field whose every group is refused is kept, so the card can say why.
 */
export interface TensorField {
  name: string;
  family: TensorFamily;
  groups: TensorGroup[];
  /** every component the step must read */
  components: string[];
  refused: string[];
}

/**
 * The stage's tensor fields, in the order of their first group. A group that
 * records both tensors gives one group to each (a field per family); a slot
 * named twice within one family refuses that family's group.
 */
export function tensorFields(groups: readonly GaussGroup[]): TensorField[] {
  const out = new Map<string, TensorField>();
  for (const g of groups) {
    const byFamily = new Map<TensorFamily, { slots: (string | null)[]; twice: boolean }>();
    for (const c of g.components) {
      const t = tensorSlotOf(c);
      if (!t) continue;
      let f = byFamily.get(t.family);
      if (!f) byFamily.set(t.family, (f = { slots: [null, null, null, null, null, null], twice: false }));
      if (f.slots[t.voigt] === null) f.slots[t.voigt] = c;
      else f.twice = true;
    }
    const base = g.id.includes("/") ? g.id.slice(0, g.id.indexOf("/")) : null;
    for (const [family, { slots, twice }] of byFamily) {
      // A `.ladruno` group is one RESULT (its id's prefix); a group of both tensors is named by each family.
      const name = base !== null && byFamily.size === 1 ? base : family;
      let f = out.get(name);
      if (!f) out.set(name, (f = { name, family, groups: [], components: [], refused: [] }));
      const missing = [0, 1, 3].filter((v) => slots[v] === null).map((v) => ["xx", "yy", "zz", "xy", "yz", "xz"][v]);
      if (twice) f.refused.push(`group ${g.id}: its components name one ${family} slot twice`);
      else if (family !== f.family) f.refused.push(`group ${g.id}: a ${family} in the ${f.family} field ${JSON.stringify(name)}`);
      else if (missing.length) f.refused.push(`group ${g.id}: no ${missing.join(", ")} ${family} component, so its tensor is not known`);
      else {
        f.groups.push({ id: g.id, slots });
        for (const c of slots) if (c !== null && !f.components.includes(c)) f.components.push(c);
      }
    }
  }
  return [...out.values()];
}

/** Eigen pairs of a symmetric 3x3: the values p1 >= p2 >= p3, and each unit vector with its largest component positive. */
export interface Eigen3 {
  values: Vec3;
  vectors: [Vec3, Vec3, Vec3];
}

/**
 * The eigen decomposition of the symmetric tensor `t` (Voigt xx, yy, zz, xy,
 * yz, xz) by cyclic Jacobi rotations, which keep the vectors orthonormal for
 * repeated values too.
 */
export function eigenSym3(t: ArrayLike<number>): Eigen3 {
  const A = [
    [t[0]!, t[3]!, t[5]!],
    [t[3]!, t[1]!, t[4]!],
    [t[5]!, t[4]!, t[2]!],
  ];
  const V = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  const norm2 = A.flat().reduce((s, x) => s + x * x, 0);
  for (let sweep = 0; sweep < 50; sweep++) {
    const off = A[0]![1]! ** 2 + A[0]![2]! ** 2 + A[1]![2]! ** 2;
    if (off <= 1e-32 * norm2 || off === 0) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]] as const) {
      const apq = A[p]![q]!;
      if (apq === 0) continue;
      const theta = (A[q]![q]! - A[p]![p]!) / (2 * apq);
      const tan = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(tan * tan + 1), s = tan * c;
      // A <- P^T A P with P the rotation in the (p, q) plane; V <- V P.
      for (let k = 0; k < 3; k++) {
        const akp = A[k]![p]!, akq = A[k]![q]!;
        A[k]![p] = c * akp - s * akq;
        A[k]![q] = s * akp + c * akq;
      }
      for (let k = 0; k < 3; k++) {
        const apk = A[p]![k]!, aqk = A[q]![k]!;
        A[p]![k] = c * apk - s * aqk;
        A[q]![k] = s * apk + c * aqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = V[k]![p]!, vkq = V[k]![q]!;
        V[k]![p] = c * vkp - s * vkq;
        V[k]![q] = s * vkp + c * vkq;
      }
    }
  }
  const order = [0, 1, 2].sort((a, b) => A[b]![b]! - A[a]![a]!);
  const vec = (j: number): Vec3 => {
    const v: Vec3 = [V[0]![j]!, V[1]![j]!, V[2]![j]!];
    // Python `_canonicalize_eigvec_sign`: the first largest-magnitude component is made non-negative.
    let k = 0;
    for (let i = 1; i < 3; i++) if (Math.abs(v[i]!) > Math.abs(v[k]!)) k = i;
    return v[k]! < 0 ? [-v[0], -v[1], -v[2]] : v;
  };
  return {
    values: [A[order[0]!]![order[0]!]!, A[order[1]!]![order[1]!]!, A[order[2]!]![order[2]!]!],
    vectors: [vec(order[0]!), vec(order[1]!), vec(order[2]!)],
  };
}

// ---- derived scalars (VA6d, #2766) ------------------------------------------------

/**
 * The derived scalars an element contour offers, each computed from one
 * family's tensor: Python's `_derived.compute` names (`_vocabulary`
 * `DERIVED_STRESS_SCALARS` / `DERIVED_STRAIN_SCALARS`). Only these two are
 * served; any other derived name is not offered, so a spec naming one is
 * refused as an unknown component.
 */
export const DERIVED_SCALARS = { von_mises_stress: "stress", von_mises_strain: "strain" } as const satisfies Record<string, TensorFamily>;
export type DerivedName = keyof typeof DERIVED_SCALARS;

export const isDerivedName = (name: string): name is DerivedName => Object.hasOwn(DERIVED_SCALARS, name);

const SLOT_NAMES = ["xx", "yy", "zz", "xy", "yz", "xz"] as const;

/**
 * The von Mises equivalent of an assembled Voigt tensor (`assembleTensor`: a
 * strain's shear already halved), as Python's `_von_mises`: J2 = s:s / 2 of
 * the deviator s; a stress gives sqrt(3 J2), a strain sqrt(4/3 J2), the
 * equivalent strain work-conjugate to it under J2 flow.
 */
export function vonMises(t: ArrayLike<number>, family: TensorFamily): number {
  const m = (t[0]! + t[1]! + t[2]!) / 3;
  const dx = t[0]! - m, dy = t[1]! - m, dz = t[2]! - m;
  const j2 = 0.5 * (dx * dx + dy * dy + dz * dz + 2 * (t[3]! * t[3]! + t[4]! * t[4]! + t[5]! * t[5]!));
  return Math.sqrt(Math.max((family === "strain" ? 4 / 3 : 3) * j2, 0));
}

/** One group a derived scalar reads: its tensor slots, as `tensorFields` found them. */
export type DerivedTensor =
  | { ok: true; family: TensorFamily; groups: TensorGroup[]; components: string[] }
  | { ok: false; reason: string };

/**
 * The `family` tensor that an element field's groups make, for its derived
 * scalars, through `tensorFields` (the Principal card's catalog) over the
 * field's own components; null when the field records no component of the
 * family. Refused, by name, when it is incomplete or ambiguous, so a derived
 * value is never drawn from partial components:
 * - a group without the in-plane xx, yy and xy (`tensorFields`' rule, Python's `available_derived`);
 * - a group that names one slot twice (`sigma12` and `sigma21`);
 * - a group with some, not all, of the out-of-plane zz, yz and xz: a 3-D tensor
 *   with a missing component. With none of them the group is a 2-D element,
 *   whose zz is 0 (Python's `plane=None`: exact for a plane-stress stress and a
 *   plane-strain strain);
 * - a field whose groups do not all make the tensor, or make two.
 */
export function derivedTensorOf(groups: readonly GaussGroup[], names: readonly string[], family: TensorFamily): DerivedTensor | null {
  const own = groups.map((g) => ({ ...g, components: g.components.filter((c) => names.includes(c)) }));
  const fields = tensorFields(own).filter((f) => f.family === family);
  if (fields.length === 0) return null;
  if (fields.length > 1) return { ok: false, reason: `its groups make ${fields.length} ${family} tensors (${fields.map((f) => f.name).join(", ")}), so which one is meant is not known` };
  const f = fields[0]!;
  if (f.refused.length) return { ok: false, reason: f.refused.join("; ") };
  for (const g of f.groups) {
    const out = [2, 4, 5].filter((v) => g.slots[v] !== null);
    if (out.length > 0 && out.length < 3) {
      const missing = [2, 4, 5].filter((v) => g.slots[v] === null).map((v) => SLOT_NAMES[v]);
      return { ok: false, reason: `group ${g.id} records ${out.map((v) => SLOT_NAMES[v]).join(", ")} but no ${missing.join(", ")} ${family} component, so its tensor is not known` };
    }
  }
  const missing = groups.filter((g) => !f.groups.some((x) => x.id === g.id)).map((g) => g.id);
  if (missing.length) return { ok: false, reason: `group${missing.length === 1 ? "" : "s"} ${missing.join(", ")} make${missing.length === 1 ? "s" : ""} no ${family} tensor` };
  return { ok: true, family, groups: f.groups, components: f.components };
}

/**
 * A derived scalar at every Gauss point of one group, (E * G,) element-major,
 * from the group's tensor read with the step (`values`: by component, then
 * group, as main's `tensor` read answers). A point with a non-finite
 * component is NaN (no data). A component the read lacks, or of another
 * length, is an error: never a value from partial components.
 */
export function derivedGroupValues(name: DerivedName, group: TensorGroup, n: number, values: Readonly<Record<string, Readonly<Record<string, Float64Array>>>>): Float64Array {
  const family = DERIVED_SCALARS[name];
  const cols = group.slots.map((c) => {
    if (c === null) return null;
    const col = values[c]?.[group.id];
    if (!(col instanceof Float64Array)) throw new Error(`${name}: the step data has no ${c} for group ${group.id}`);
    if (col.length !== n) throw new Error(`${name}: ${c} of group ${group.id} has ${col.length} values, not ${n}`);
    return col;
  });
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const t = assembleTensor(family, (v) => cols[v]?.[i] ?? null);
    out[i] = t ? vonMises(t, family) : NaN;
  }
  return out;
}

/** The Voigt tensor of one point: `read(slot)` is the file's value (NaN when not finite), 0 for a slot the group does not record; a strain's shear halved. */
export function assembleTensor(family: TensorFamily, read: (voigt: number) => number | null): Float64Array | null {
  const t = new Float64Array(6);
  for (let v = 0; v < 6; v++) {
    const x = read(v);
    if (x === null) continue;
    if (!Number.isFinite(x)) return null;
    t[v] = family === "strain" && v >= 3 ? 0.5 * x : x;
  }
  return t;
}

