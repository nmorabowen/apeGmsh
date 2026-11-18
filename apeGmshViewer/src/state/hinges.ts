// The Hinges card's model: where the Ladruno beams' moment hinges are, what each
// one's `hingeSummary` columns are, and the colours and curves the card draws.
//
// Everything is read from the two files:
// - `model.h5` says which hinges exist and where: the element's emitted arguments
//   (`/opensees/element_meta/<type>/args`), decoded with the fork parsers' rules
//   (OPS_LadrunoForceBeamColumn, OPS_LadrunoDispBeamColumn, OPS_LadrunoIMKBeam).
//   A `.ladruno` holds neither fact, and an absent hinge reads as zeros there.
// - `.ladruno` says what each hinge did: `hingeSummary` (reader/ladruno.ts).
//
// Placement: a force or displacement beam's hinge sits at `-hingeLoc` (xi 0 and 1
// when the flag is absent; one value means xiJ = 1 - xiI). An IMK beam's sits at
// the rigid end zone's face (`-endZones dI dJ`). Shear hinges are not drawn.
//
// Pure: no three.js and no DOM, so it runs in Node tests.

import type { Param } from "../model/types.ts";
import type { BlobStore } from "./blobs.ts";
import { sampleCmap } from "./contour.ts";
import { buildNodeCoordsMap, type BuildLoads3DOptions } from "./loads.ts";
import type { Decl, DeclPath, HingeAxis, HingeEnd, HingeHistory, HingesState, State } from "./types.ts";

/** One step of `hingeSummary`, as main sends it (reader/ladruno.ts `LadrunoHingeStep`). */
export interface HingeStepData {
  readonly elementIds: Float64Array;
  /** (n * 8,) in `HINGE_COLUMN_NAMES` order; NaN for a column the file does not record */
  readonly values: Float64Array;
  /** the largest |rotation| and |moment| of the whole stage, or null when the stage is too large to scan */
  readonly range: { readonly rotation: number; readonly moment: number } | null;
}

/** The hinge material of one axis: the family and tag the element's arguments name. */
export interface HingeMaterial {
  readonly family: "uniaxialMaterial" | "nDMaterial";
  readonly tag: number;
}

export interface HingeSite {
  readonly decl: DeclPath;
  readonly type: string;
  /** the OpenSees element tag */
  readonly tag: number;
  readonly end: HingeEnd;
  /** position along the chord from node i, 0..1; null when `offset` places it */
  readonly xi: number | null;
  /** an IMK hinge's distance from its own end node (`-endZones`), in length units */
  readonly offset: number;
  readonly z: HingeMaterial | null;
  readonly y: HingeMaterial | null;
  readonly biaxial: boolean;
}

export interface HingeSites {
  readonly sites: readonly HingeSite[];
  /** elements whose hinge arguments could not be read, by name */
  readonly refused: readonly string[];
}

/** The `hingeSummary` columns, in the order every hinge read uses. */
export const HINGE_COLUMN_NAMES = ["thetaZ_i", "Mz_i", "thetaZ_j", "Mz_j", "thetaY_i", "My_i", "thetaY_j", "My_j"] as const;
export const HINGE_WIDTH = HINGE_COLUMN_NAMES.length;

/** The rotation column of a hinge; the moment is the next one. */
export const hingeColumn = (end: HingeEnd, axis: HingeAxis): number => (axis === "z" ? 0 : 4) + (end === "i" ? 0 : 2);

const BEAMS: ReadonlySet<string> = new Set(["LadrunoForceBeamColumn", "LadrunoDispBeamColumn"]);
const IMK = "LadrunoIMKBeam";

type Slot = { i: number | null; j: number | null };
const emptySlot = (): Slot => ({ i: null, j: null });

/** The numbers right after flag `at`, up to the next flag. */
function numbersAfter(args: readonly Param[], at: number): number[] {
  const out: number[] = [];
  for (let k = at + 1; k < args.length && typeof args[k] === "number"; k++) out.push(args[k] as number);
  return out;
}

type Assign = "first-second" | "i" | "j" | "both";
const BEAM_FLAGS: Readonly<Record<string, readonly ["z" | "y" | "c", Assign]>> = {
  "-hinge": ["z", "first-second"],
  "-hingeZ": ["z", "first-second"],
  "-hingeI": ["z", "i"],
  "-hinge1": ["z", "i"],
  "-hingeZI": ["z", "i"],
  "-hingeJ": ["z", "j"],
  "-hinge2": ["z", "j"],
  "-hingeZJ": ["z", "j"],
  "-hingeIJ": ["z", "both"],
  "-hingeZIJ": ["z", "both"],
  "-hingeY": ["y", "first-second"],
  "-hingeYI": ["y", "i"],
  "-hingeY1": ["y", "i"],
  "-hingeYJ": ["y", "j"],
  "-hingeY2": ["y", "j"],
  "-hingeYIJ": ["y", "both"],
  "-hingeBiaxial": ["c", "first-second"],
  "-hingeC": ["c", "first-second"],
  "-hingeBiaxialI": ["c", "i"],
  "-hingeBiaxial1": ["c", "i"],
  "-hingeCI": ["c", "i"],
  "-hingeBiaxialJ": ["c", "j"],
  "-hingeBiaxial2": ["c", "j"],
  "-hingeCJ": ["c", "j"],
  "-hingeBiaxialIJ": ["c", "both"],
  "-hingeCIJ": ["c", "both"],
};

function assign(slot: Slot, how: Assign, tags: readonly number[]): void {
  const [a, b] = tags;
  if (a === undefined) return;
  if (how === "i") slot.i = a;
  else if (how === "j") slot.j = a;
  else if (how === "both") {
    slot.i = a;
    slot.j = b ?? a;
  } else {
    slot.i = a;
    if (b !== undefined) slot.j = b;
  }
}

export interface HingeEndFacts {
  readonly end: HingeEnd;
  readonly xi: number | null;
  readonly offset: number;
  readonly z: HingeMaterial | null;
  readonly y: HingeMaterial | null;
  readonly biaxial: boolean;
}

function endsOf(z: Slot, y: Slot, c: Slot, place: (end: HingeEnd) => { xi: number | null; offset: number }): HingeEndFacts[] {
  const out: HingeEndFacts[] = [];
  const mat = (tag: number | null): HingeMaterial | null => (tag === null ? null : { family: "uniaxialMaterial", tag });
  for (const end of ["i", "j"] as const) {
    const bi = c[end];
    const uz = z[end];
    const uy = y[end];
    if (bi === null && uz === null && uy === null) continue;
    const nd: HingeMaterial | null = bi === null ? null : { family: "nDMaterial", tag: bi };
    out.push({ end, ...place(end), z: nd ?? mat(uz), y: nd ?? mat(uy), biaxial: bi !== null });
  }
  return out;
}

/** The hinges of one element's emitted arguments; [] for a beam without any. */
export function hingeEndsOf(type: string, args: readonly Param[]): HingeEndFacts[] {
  const z = emptySlot(), y = emptySlot(), c = emptySlot();
  if (BEAMS.has(type)) {
    let xiI = 0, xiJ = 1;
    args.forEach((a, at) => {
      if (typeof a !== "string") return;
      const nums = numbersAfter(args, at);
      if (a === "-hingeLoc") {
        if (nums[0] === undefined) throw new Error("-hingeLoc has no value");
        xiI = nums[0];
        xiJ = nums[1] ?? 1 - nums[0];
        return;
      }
      const flag = BEAM_FLAGS[a];
      if (flag) assign(flag[0] === "z" ? z : flag[0] === "y" ? y : c, flag[1], nums);
    });
    return endsOf(z, y, c, (end) => ({ xi: end === "i" ? xiI : xiJ, offset: 0 }));
  }
  if (type !== IMK) return [];
  let mode = "both";
  let dI = 0, dJ = 0;
  let matZ: number | null = null, matY: number | null = null, bi: number | null = null;
  args.forEach((a, at) => {
    if (typeof a !== "string") return;
    const nums = numbersAfter(args, at);
    const first = nums[0] ?? null;
    if (a === "-hinge") {
      const v = args[at + 1];
      if (typeof v !== "string") throw new Error("-hinge has no end selection");
      mode = v.toLowerCase();
    } else if (a === "-matZ") matZ = first;
    else if (a === "-matY") matY = first;
    else if (a === "-matZi") z.i = first;
    else if (a === "-matZj") z.j = first;
    else if (a === "-matYi") y.i = first;
    else if (a === "-matYj") y.j = first;
    else if (a === "-endZones" || a === "-endZone") {
      dI = nums[0] ?? 0;
      dJ = nums[1] ?? 0;
    } else if (a === "-hingeBiaxial" || a === "-hingeC") {
      // Two tags name the two ends outright; one is the symmetric law, gated by `-hinge`.
      if (nums.length >= 2) assign(c, "first-second", nums);
      else bi = first;
    } else {
      const flag = BEAM_FLAGS[a];
      if (flag?.[0] === "c") assign(c, flag[1], nums);
    }
  });
  if (!["both", "i", "j", "none"].includes(mode)) throw new Error(`-hinge ${mode} is not both, i, j or none`);
  const at = { i: mode === "both" || mode === "i", j: mode === "both" || mode === "j" };
  for (const end of ["i", "j"] as const) {
    if (!at[end]) continue;
    z[end] ??= matZ;
    y[end] ??= matY;
    c[end] ??= bi;
  }
  return endsOf(z, y, c, (end) => ({ xi: null, offset: end === "i" ? dI : dJ }));
}

const sitesMemo = new WeakMap<object, HingeSites>();

/** Every moment hinge the model declares, element by element. */
export function hingeSitesOf(decls: Readonly<Record<DeclPath, Decl>>): HingeSites {
  const memo = sitesMemo.get(decls);
  if (memo) return memo;
  const sites: HingeSite[] = [];
  const refused: string[] = [];
  for (const [decl, d] of Object.entries(decls)) {
    if (d.kind !== "element" || !d.element) continue;
    for (const m of d.element.metas) {
      if (!BEAMS.has(m.type) && m.type !== IMK) continue;
      try {
        for (const e of hingeEndsOf(m.type, m.args)) sites.push({ decl, type: m.type, tag: m.tag, ...e });
      } catch (err) {
        refused.push(`${m.type} ${m.tag}: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
  }
  const out: HingeSites = { sites, refused };
  sitesMemo.set(decls, out);
  return out;
}

const materialsMemo = new WeakMap<object, Map<string, Decl>>();

/** The declaration of a hinge's material, by family and tag. */
export function hingeMaterialDecl(decls: Readonly<Record<DeclPath, Decl>>, mat: HingeMaterial): Decl | undefined {
  let index = materialsMemo.get(decls);
  if (!index) {
    index = new Map();
    for (const d of Object.values(decls)) {
      const family = /^opensees\/(uniaxialMaterial|nDMaterial)\//.exec(d.path)?.[1];
      if (family && d.tag !== null) index.set(`${family}:${d.tag}`, d);
    }
    materialsMemo.set(decls, index);
  }
  return index.get(`${mat.family}:${mat.tag}`);
}

/** The moment capacity the model names for a hinge's axis, or why it names none. */
export function hingeCapacityOf(decls: Readonly<Record<DeclPath, Decl>>, site: HingeSite, axis: HingeAxis): { value: number } | { missing: string } {
  const mat = axis === "z" ? site.z : site.y;
  if (!mat) return { missing: `no ${axis === "z" ? "Mz" : "My"} hinge at end ${site.end}` };
  const d = hingeMaterialDecl(decls, mat);
  if (!d) return { missing: `${mat.family} ${mat.tag} is not in the model` };
  const index = d.type === "LadrunoCohesiveHinge" || d.type === "Steel01" ? 0 : d.type === "LadrunoCohesiveHingeBiaxial" ? (axis === "z" ? 0 : 2) : null;
  if (index === null) return { missing: `no moment capacity is read from ${d.type} ${mat.tag}` };
  const v = d.params[index];
  return typeof v === "number" && Number.isFinite(v) && v > 0 ? { value: v } : { missing: `${d.type} ${mat.tag} has no positive capacity in its arguments` };
}

/** A hinge's scalar for the colour: resultant rotation or moment, or the larger moment/capacity of its axes; null when not defined. */
export function hingeMetric(decls: Readonly<Record<DeclPath, Decl>>, site: HingeSite, row: ArrayLike<number>, by: HingesState["colorBy"]): number | null {
  const axes = (["z", "y"] as const).filter((a) => (a === "z" ? site.z : site.y) !== null);
  if (by === "capacity") {
    let best: number | null = null;
    for (const a of axes) {
      const cap = hingeCapacityOf(decls, site, a);
      const m = row[hingeColumn(site.end, a) + 1]!;
      if ("value" in cap && Number.isFinite(m)) best = Math.max(best ?? 0, Math.abs(m) / cap.value);
    }
    return best;
  }
  const shift = by === "moment" ? 1 : 0;
  let sum = 0;
  for (const a of axes) {
    const v = row[hingeColumn(site.end, a) + shift]!;
    if (!Number.isFinite(v)) return null;
    sum += v * v;
  }
  return axes.length ? Math.sqrt(sum) : null;
}

export interface Hinges3D {
  /** (count * 3,) marker centres */
  readonly positions: Float32Array;
  /** (count * 3,) marker colours */
  readonly colors: Float32Array;
  readonly count: number;
  /** hinges whose element's nodes are not on screen */
  readonly unplaced: number;
  /** hinges drawn in the no-data colour although a step is on screen */
  readonly noValue: number;
  /** the value at the top of the colour range; null with no step */
  readonly top: number | null;
}

/** A marker with no step to colour it: visible, and unlike any colour of the ramp. */
export const HINGE_NEUTRAL: readonly [number, number, number] = [0.93, 0.93, 0.9];
export const HINGE_NO_DATA: readonly [number, number, number] = [0.55, 0.55, 0.58];

function nodesOf(s: State, blobs: BlobStore, d: Decl): readonly number[] {
  const e = d.element;
  if (!e) return [];
  if (e.inlineNodes.length) return e.inlineNodes;
  const block = e.cell ? s.blocks[e.cell.block] : undefined;
  if (!e.cell || !block || !blobs.has(block.connectivity)) return [];
  const conn = blobs.get(block.connectivity) as Float64Array;
  return Array.from(conn.subarray(e.cell.row * block.npe, (e.cell.row + 1) * block.npe));
}

/**
 * The markers: one per hinge, on the element's chord between its (possibly
 * deformed) end nodes. With `step`, coloured by `s.hinges.colorBy`; `idOf` maps
 * an element to its row id in the file, null when the two cannot be joined.
 */
export function buildHinges3D(s: State, blobs: BlobStore, step: HingeStepData | null, idOf: ((decl: DeclPath) => number | undefined) | null, opts?: BuildLoads3DOptions): Hinges3D {
  const none: Hinges3D = { positions: new Float32Array(0), colors: new Float32Array(0), count: 0, unplaced: 0, noValue: 0, top: null };
  if (!s.mesh) return none;
  const { sites } = hingeSitesOf(s.decls);
  if (sites.length === 0) return none;
  const coords = buildNodeCoordsMap(s, blobs, opts);
  const base = opts?.drawnLinePositions ? buildNodeCoordsMap(s, blobs) : coords;
  const hidden = new Set(s.visibility.hidden);
  const rows = new Map<number, number>();
  if (step) step.elementIds.forEach((id, r) => rows.set(id, r));
  const by = s.hinges.colorBy;
  const placed: { p: [number, number, number]; v: number | null }[] = [];
  let unplaced = 0, noValue = 0, stepTop = 0;
  for (const site of sites) {
    if (hidden.has(site.decl)) continue;
    const nodes = nodesOf(s, blobs, s.decls[site.decl]!);
    const a = coords.get(nodes[0]!), b = coords.get(nodes[1]!);
    const a0 = base.get(nodes[0]!), b0 = base.get(nodes[1]!);
    if (!a || !b || !a0 || !b0) {
      unplaced++;
      continue;
    }
    let t = site.xi;
    if (t === null) {
      const len = Math.hypot(b0[0] - a0[0], b0[1] - a0[1], b0[2] - a0[2]);
      const f = len > 0 ? site.offset / len : 0;
      t = site.end === "i" ? f : 1 - f;
    }
    let v: number | null = null;
    if (step && idOf) {
      const id = idOf(site.decl);
      const r = id === undefined ? undefined : rows.get(id);
      if (r !== undefined) v = hingeMetric(s.decls, site, step.values.subarray(r * HINGE_WIDTH, (r + 1) * HINGE_WIDTH), by);
    }
    if (step) {
      if (v === null) noValue++;
      else stepTop = Math.max(stepTop, v);
    }
    placed.push({ p: [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])], v });
  }
  const top = !step ? null : by === "capacity" ? 1 : ((by === "rotation" ? step.range?.rotation : step.range?.moment) ?? stepTop);
  const positions = new Float32Array(placed.length * 3);
  const colors = new Float32Array(placed.length * 3);
  placed.forEach(({ p, v }, i) => {
    positions.set(p, i * 3);
    const rgb = !step ? HINGE_NEUTRAL : v === null ? HINGE_NO_DATA : sampleCmap("viridis", top! > 0 ? Math.min(1, v / top!) : 0);
    colors.set(rgb, i * 3);
  });
  return { positions, colors, count: placed.length, unplaced, noValue, top };
}

export interface HingeCurve {
  readonly rotationName: string;
  readonly momentName: string;
  readonly time: readonly number[];
  readonly rotation: readonly number[];
  readonly moment: readonly number[];
}

/** One hinge's rotation and moment over the stage, from its element's history. */
export function hingeCurveOf(h: HingeHistory, end: HingeEnd, axis: HingeAxis): HingeCurve {
  const c = hingeColumn(end, axis);
  const n = h.time.length;
  const rotation = new Array<number>(n), moment = new Array<number>(n);
  for (let k = 0; k < n; k++) {
    rotation[k] = h.values[k * HINGE_WIDTH + c]!;
    moment[k] = h.values[k * HINGE_WIDTH + c + 1]!;
  }
  return { rotationName: HINGE_COLUMN_NAMES[c]!, momentName: HINGE_COLUMN_NAMES[c + 1]!, time: h.time, rotation, moment };
}
