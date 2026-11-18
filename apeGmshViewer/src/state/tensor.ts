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

import type { Param } from "../model/types.ts";
import type { Decl, DeclPath, GaussGroup } from "./types.ts";

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
 * - a group with one of the out-of-plane shears yz and xz and not the other: a
 *   3-D tensor with a missing component. With neither, the group is a 2-D
 *   element: a recorded zz is used (v1's `_assemble_tensor`), and a zz it does
 *   not record is recovered per element from its plane type and ν
 *   (`derivedGroupValues`, v1's `plane="auto"`);
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
    if ((g.slots[4] === null) !== (g.slots[5] === null)) {
      const [has, lacks] = g.slots[4] === null ? ["xz", "yz"] : ["yz", "xz"];
      return { ok: false, reason: `group ${g.id} records the ${has} but no ${lacks} ${family} component, so its tensor is not known` };
    }
  }
  const missing = groups.filter((g) => !f.groups.some((x) => x.id === g.id)).map((g) => g.id);
  if (missing.length) return { ok: false, reason: `group${missing.length === 1 ? "" : "s"} ${missing.join(", ")} make${missing.length === 1 ? "s" : ""} no ${family} tensor` };
  return { ok: true, family, groups: f.groups, components: f.components };
}

// ---- out-of-plane recovery (VA6d F1): v1's `_plane_recovery`, per element ----------------

/** A 2-D element's plane idealization and its material's Poisson's ratio (null: not read). */
export interface PlaneOf {
  plane: "PlaneStrain" | "PlaneStress";
  nu: number | null;
}

// `_plane_recovery`'s tables, restated (src/apeGmsh/results/_plane_recovery.py).
const NU_AT_PARAM1 = new Set(["ElasticIsotropic", "ElasticIsotropicPlaneStrain2D", "ElasticIsotropicPlaneStress2D", "ASDConcrete3D"]);
const KG_MATERIALS = new Set(["J2Plasticity", "DruckerPrager", "LadrunoJ2", "LadrunoJ2Finite"]);
const POSITIONAL_PLANE_ELEMENTS = new Set(["quad", "tri31", "tri6n", "BezierTri6"]);
const FLAG_PLANE_ELEMENTS = new Set(["LadrunoQuad", "LadrunoCST", "LadrunoLST"]);
const UP_PLANE_ELEMENT = "LadrunoUP";

/** Python's `float(value)`: a number, or a string that parses whole; else null. */
function asFloat(p: Param | undefined): number | null {
  if (typeof p === "number") return p;
  if (typeof p === "string" && p.trim() !== "") {
    const x = Number(p);
    return Number.isNaN(x) && p.trim().toLowerCase() !== "nan" ? null : x;
  }
  return null;
}

/** Python's `int(value)` on a tag: an integral number; else null. */
function asTag(p: Param | undefined): number | null {
  const x = asFloat(p);
  return x !== null && Number.isFinite(x) ? Math.trunc(x) : null;
}

/** `_normalize_plane`: "strain" anywhere in the text is plane strain, else "stress" plane stress. */
function normalizePlane(p: Param | undefined): PlaneOf["plane"] | null {
  const s = (typeof p === "string" ? p : String(p)).toLowerCase();
  return s.includes("strain") ? "PlaneStrain" : s.includes("stress") ? "PlaneStress" : null;
}

/** `_material_nu`: an nD material's Poisson's ratio, or null. */
export function materialNu(type: string, params: readonly Param[]): number | null {
  if (NU_AT_PARAM1.has(type) && params.length >= 2) return asFloat(params[1]);
  if (KG_MATERIALS.has(type) && params.length >= 2) {
    const k = asFloat(params[0]), g = asFloat(params[1]);
    if (k !== null && g !== null && 3 * k + g !== 0) return (3 * k - 2 * g) / (2 * (3 * k + g));
  }
  for (let i = 0; i < params.length - 1; i++) if (params[i] === "PoissonsRatio") return asFloat(params[i + 1]);
  return null;
}

/** `_element_plane_and_mat`: a 2-D element's plane type and nD material tag, or null. */
export function elementPlane(type: string, args: readonly Param[]): { plane: PlaneOf["plane"]; mat: number } | null {
  if (POSITIONAL_PLANE_ELEMENTS.has(type) && args.length >= 3) {
    const plane = normalizePlane(args[1]);
    const mat = asTag(args[2]);
    return plane !== null && mat !== null ? { plane, mat } : null;
  }
  if (FLAG_PLANE_ELEMENTS.has(type) && args.length >= 1) {
    const mat = asTag(args[0]);
    if (mat === null) return null;
    let plane: PlaneOf["plane"] = "PlaneStrain"; // the Ladruno default with no -type flag
    for (let i = 0; i < args.length - 1; i++) {
      if (args[i] === "-type") {
        plane = normalizePlane(args[i + 1]) ?? plane;
        break;
      }
    }
    return { plane, mat };
  }
  if (type === UP_PLANE_ELEMENT && args.length >= 1) {
    // `_up_perm_dim`: 2-D u-p (two permeabilities after -perm / -permH) is plane strain by construction.
    const at = args.findIndex((a) => a === "-perm" || a === "-permH");
    if (at < 0) return null;
    let n = 0;
    for (const a of args.slice(at + 1)) {
      if (typeof a === "string") break;
      n++;
    }
    const mat = asTag(args[0]);
    return n === 2 && mat !== null ? { plane: "PlaneStrain", mat } : null;
  }
  return null;
}

/** `planeRecoveryMap` and its digest, once per declarations object (a re-read model is a new object). */
const planeCache = new WeakMap<object, { map: Map<number, PlaneOf>; digest: string }>();
function planeEntry(decls: Readonly<Record<DeclPath, Decl>>): { map: Map<number, PlaneOf>; digest: string } {
  let e = planeCache.get(decls);
  if (!e) {
    const map = planeRecoveryMap(decls);
    // FNV-1a (32-bit) over the sorted entries: deterministic across processes, unlike a salted hash.
    let h = 0x811c9dc5;
    const text = [...map].sort((a, b) => a[0] - b[0]).map(([id, p]) => `${id}:${p.plane}:${p.nu ?? "-"}`).join(";");
    for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 0x01000193) >>> 0;
    planeCache.set(decls, (e = { map, digest: `${map.size}-${h.toString(16)}` }));
  }
  return e;
}

/** The model's plane map, cached per declarations object. */
export const planeMapOf = (decls: Readonly<Record<DeclPath, Decl>>): Map<number, PlaneOf> => planeEntry(decls).map;

/**
 * A digest of the model's plane map: it changes whenever the map does (a model
 * re-read with another plane type or ν), so a derived contour's step and scan
 * keys, which carry it, are fetched and scanned again (#2921 review).
 */
export const planeDigestOf = (decls: Readonly<Record<DeclPath, Decl>>): string => planeEntry(decls).digest;

/**
 * `plane_recovery_map`: `{FEM id: plane type and ν}` for the model's 2-D
 * elements, from the `/opensees` element rows (`element_meta/*` args) and the
 * nD materials' params. Empty for a model with no such records (a FEM-only
 * model.h5): every element's zz then stays 0, and the statement says so.
 */
export function planeRecoveryMap(decls: Readonly<Record<DeclPath, Decl>>): Map<number, PlaneOf> {
  const nuByTag = new Map<number, number>();
  const elements: Decl[] = [];
  for (const d of Object.values(decls)) {
    if (d.kind === "nDMaterial" && d.tag !== null) {
      const nu = materialNu(d.type, d.params);
      if (nu !== null) nuByTag.set(d.tag, nu);
    } else if (d.kind === "element") elements.push(d);
  }
  const out = new Map<number, PlaneOf>();
  for (const d of elements) {
    const f = d.element;
    if (!f || f.femId === null || f.femId < 0) continue;
    for (const m of f.metas) {
      const p = elementPlane(m.type, m.args);
      if (p) out.set(f.femId, { plane: p.plane, nu: nuByTag.get(p.mat) ?? null });
    }
  }
  return out;
}

/** What a group's derived values need beyond its tensor: its element ids (row order) and Gauss count, and the model's plane map. */
export interface PlaneRecovery {
  map: ReadonlyMap<number, PlaneOf>;
  eids: Float64Array;
  nGauss: number;
}

/**
 * The out-of-plane normal `inject_out_of_plane` gives a point whose zz is not
 * recorded (or recorded as NaN): σzz = ν(σxx + σyy) for a plane-strain element
 * (a stress), εzz = -ν/(1-ν)(εxx + εyy) for a plane-stress one (a strain); 0
 * for the other idealization (exact), an element the model does not classify
 * or one whose ν is not read (v1 warns; the manifest counts them).
 */
function recoveredZz(family: TensorFamily, info: PlaneOf | undefined, xx: number, yy: number): number {
  const nu = info?.nu;
  if (nu === null || nu === undefined || !Number.isFinite(nu)) return 0;
  if (family === "stress") return info!.plane === "PlaneStrain" ? nu * (xx + yy) : 0;
  return info!.plane === "PlaneStress" && nu !== 1 ? (-nu / (1 - nu)) * (xx + yy) : 0;
}

/**
 * A derived scalar at every Gauss point of one group, (E * G,) element-major,
 * from the group's tensor read with the step (`values`: by component, then
 * group, as main's `tensor` read answers), as v1's default read gives it
 * (`_composites._compute_derived`, `plane="auto"`): a recorded zz is used; a
 * zz the group does not record, or a recorded NaN, is recovered per element
 * (`recoveredZz`), unless the model classifies no element at all, where an
 * unrecorded zz is 0 and a NaN stays no data (`inject_out_of_plane` returns
 * early). A point with another non-finite component is NaN (no data). A
 * component the read lacks, or of another length, is an error: never a value
 * from partial components.
 */
export function derivedGroupValues(name: DerivedName, group: TensorGroup, n: number, values: Readonly<Record<string, Readonly<Record<string, Float64Array>>>>, plane: PlaneRecovery): Float64Array {
  const family = DERIVED_SCALARS[name];
  const cols = group.slots.map((c) => {
    if (c === null) return null;
    const col = values[c]?.[group.id];
    if (!(col instanceof Float64Array)) throw new Error(`${name}: the step data has no ${c} for group ${group.id}`);
    if (col.length !== n) throw new Error(`${name}: ${c} of group ${group.id} has ${col.length} values, not ${n}`);
    return col;
  });
  if (plane.eids.length * plane.nGauss !== n) throw new Error(`${name}: group ${group.id} has ${plane.eids.length} element ids at ${plane.nGauss} Gauss points for ${n} values`);
  const zzCol = cols[2] ?? null;
  const recover = plane.map.size > 0;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let zz: number | null = zzCol ? zzCol[i]! : null;
    if (recover && (zz === null || !Number.isFinite(zz))) {
      zz = recoveredZz(family, plane.map.get(plane.eids[Math.floor(i / plane.nGauss)]!), cols[0]![i]!, cols[1]![i]!);
    }
    const t = assembleTensor(family, (v) => (v === 2 ? zz : (cols[v]?.[i] ?? null)));
    out[i] = t ? vonMises(t, family) : NaN;
  }
  return out;
}

/**
 * What a derived contour's out-of-plane normal is, by element (v1 warns on
 * the last two): `recorded` (a group that records zz), `recovered` (the
 * idealization whose zz is not 0, with its ν), `zero_exact` (the idealization
 * whose zz is 0), `unclassified` (the model does not say plane stress or
 * plane strain: zz left 0), `no_nu` (it needs ν, which its material does not
 * give: zz left 0). Every element of the field's groups is counted once.
 */
export interface PlaneStatement {
  elements: number;
  recorded: number;
  recovered: number;
  zero_exact: number;
  unclassified: number;
  no_nu: number;
}

export function planeStatement(family: TensorFamily, groups: readonly { group: TensorGroup; eids: Float64Array }[], map: ReadonlyMap<number, PlaneOf>): PlaneStatement {
  const seen = new Set<number>();
  const st: PlaneStatement = { elements: 0, recorded: 0, recovered: 0, zero_exact: 0, unclassified: 0, no_nu: 0 };
  const needs = family === "stress" ? "PlaneStrain" : "PlaneStress";
  for (const { group, eids } of groups) {
    for (const e of eids) {
      if (seen.has(e)) continue;
      seen.add(e);
      st.elements++;
      if (group.slots[2] !== null) st.recorded++;
      else {
        const info = map.get(e);
        if (!info) st.unclassified++;
        else if (info.plane !== needs) st.zero_exact++;
        else if (info.nu === null || !Number.isFinite(info.nu) || (family === "strain" && info.nu === 1)) st.no_nu++;
        else st.recovered++;
      }
    }
  }
  return st;
}

/**
 * VF2 (#2924 item 6): the one line a derived contour's statement gives when
 * some element's zz is left at 0 without the model saying it is 0 (v1 warns on
 * these), else null. The manifest's `contour.derived.warning` and the banner
 * (selectors `derivedPlaneOf`) both say it, in these words.
 */
export function planeWarning(family: TensorFamily, st: PlaneStatement): string | null {
  const zz = `${family}_zz`;
  const parts = [
    ...(st.unclassified ? [`${st.unclassified} of ${st.elements} element(s) could not be classified as plane stress / plane strain`] : []),
    ...(st.no_nu ? [`${st.no_nu} element(s) need Poisson's ratio, which could not be read from their material`] : []),
  ];
  return parts.length ? `Out-of-plane ${zz} recovery is incomplete: ${parts.join("; ")}. ${zz} is left at 0 there.` : null;
}

/** `planeStatementOf`'s entries, weakly keyed on the first group's element-id array (a re-read or a new set reads new arrays). */
const statementCache = new WeakMap<Float64Array, { decls: object; family: TensorFamily; slots: string; eids: readonly Float64Array[]; st: Readonly<PlaneStatement> }[]>();

/**
 * VF2 review (#2952): `planeStatement` over the model's plane map, once per
 * (declarations object, family, the groups' slots and element-id arrays). The
 * step is not in the key: the banner asks on every step, and a 1M-element
 * statement is ~0.4 s on the main thread. A re-read model is a new
 * declarations object and a new set reads new element-id arrays, so either
 * computes it again. The statement is frozen: every caller shares it.
 */
export function planeStatementOf(family: TensorFamily, groups: readonly { group: TensorGroup; eids: Float64Array }[], decls: Readonly<Record<DeclPath, Decl>>): Readonly<PlaneStatement> {
  if (groups.length === 0) return Object.freeze(planeStatement(family, groups, planeMapOf(decls)));
  const slots = JSON.stringify(groups.map((g) => g.group.slots));
  const first = groups[0]!.eids;
  const list = statementCache.get(first) ?? [];
  const hit = list.find((x) => x.decls === decls && x.family === family && x.slots === slots && x.eids.length === groups.length && x.eids.every((a, i) => a === groups[i]!.eids));
  if (hit) return hit.st;
  const st = Object.freeze(planeStatement(family, groups, planeMapOf(decls)));
  list.push({ decls, family, slots, eids: groups.map((g) => g.eids), st });
  if (list.length > 8) list.shift(); // re-reads of the model over one results read: keep the latest few
  statementCache.set(first, list);
  return st;
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

