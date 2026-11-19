// The Cracks card's discs (renderer/cracks.ts `CracksLayer`): a smeared-crack
// material (ASDConcrete3D's `crackPattern`) records, at each Gauss point, up to
// three vectors along the crack normals in the global frame, each as long as that
// crack's opening. Each crack is drawn as a small disc lying in the crack plane
// (its normal the vector's direction) at the Gauss point, coloured by the opening.
// Seen edge-on (a side view along the plane), a disc is the segment across the
// projected normal that Python's `results.plot.cracks` draws
// (src/apeGmsh/results/plot/_cracks.py `crack_segments`): same keep rule (an
// opening > 0 and >= the threshold), same size rule (0.4 x the Gauss pitch).
// Pure: tested without a window.

import { rampT, sampleCmap } from "./contour.ts";
import type { ContourCmap, GaussGroup } from "./types.ts";

/** The two axes of a crack vector component: the crack `k` (1..3) and the global axis. */
const AXES = ["x", "y", "z"] as const;

/**
 * The nine components of one group's crack vectors, slot-major
 * (c1x, c1y, c1z, c2x, ...), in one of the spellings a results file gives them:
 * - `crack_c1_x` .. : Python's canonical names (a `.results.h5` apeGmsh wrote,
 *   `_ladruno_element_io._MATERIAL_BUCKET_TOKENS`);
 * - `<result>/C1x` .. : a `.ladruno` read as the file names it (ADR 0122 D7, no
 *   token table), `<result>` a RESULT whose last dotted token is `crackPattern`
 *   (`material.crackPattern`), the labels ASDConcrete3D's `lb_crackpattern`;
 * - `<result>/C1` .. `<result>/C9` : the same RESULT from an older fork build
 *   that writes generic labels, by position, as Python reads it.
 * Null when the group records none of them.
 */
export function crackSlotsOf(components: readonly string[]): { slots: string[] } | { partial: string } | null {
  const have = new Set(components);
  const canonical = [1, 2, 3].flatMap((k) => AXES.map((a) => `crack_c${k}_${a}`));
  const spellings: string[][] = [canonical];
  const results = new Set<string>();
  for (const c of components) {
    const slash = c.lastIndexOf("/");
    if (slash < 0) continue;
    const result = c.slice(0, slash);
    if (result.slice(result.lastIndexOf(".") + 1).toLowerCase() === "crackpattern") results.add(result);
  }
  for (const r of results) {
    spellings.push([1, 2, 3].flatMap((k) => AXES.map((a) => `${r}/C${k}${a}`)));
    spellings.push(Array.from({ length: 9 }, (_, i) => `${r}/C${i + 1}`));
  }
  for (const names of spellings) if (names.every((n) => have.has(n))) return { slots: names };
  const some = spellings.flat().filter((n) => have.has(n));
  if (some.length) return { partial: `it records ${some.length} of the 9 crack-vector components (${some.slice(0, 3).join(", ")}${some.length > 3 ? ", ..." : ""}), not all of one crack pattern` };
  return null;
}

/** One group of the stage's crack field: its nine components, slot-major. */
export interface CrackGroup {
  id: string;
  slots: string[];
}

/** The stage's crack pattern: the groups that record one, every component the step must read, and why a group is not drawn. */
export interface CrackField {
  groups: CrackGroup[];
  components: string[];
  refused: string[];
}

/** The crack field of a stage's Gauss groups; null when no group records a crack pattern. */
export function crackFieldOf(groups: readonly GaussGroup[]): CrackField | null {
  const out: CrackField = { groups: [], components: [], refused: [] };
  for (const g of groups) {
    const r = crackSlotsOf(g.components);
    if (r === null) {
      // A component refused at open (a partial write) is never in `components`: say so, never draw it short.
      const refused = g.refused.filter((x) => /crack/i.test(x.component));
      if (refused.length) out.refused.push(`${g.id}: ${refused[0]!.component} refused (${refused[0]!.reason})`);
      continue;
    }
    if ("partial" in r) {
      out.refused.push(`${g.id}: ${r.partial}`);
      continue;
    }
    out.groups.push({ id: g.id, slots: r.slots });
    for (const c of r.slots) if (!out.components.includes(c)) out.components.push(c);
  }
  return out.groups.length || out.refused.length ? out : null;
}

/**
 * Each point's nine crack-vector components (9 per point, slot-major) from the
 * step's components by group; NaN for a point whose values are missing.
 * `group[i]` indexes `field.groups`, `row[i]` the group's value (element-major).
 */
export function crackVectors(
  points: { layout: { count: number }; group: Int32Array; row: Int32Array },
  field: CrackField,
  values: Readonly<Record<string, Readonly<Record<string, Float64Array>>>>,
): Float64Array {
  const n = points.layout.count;
  const out = new Float64Array(9 * n).fill(NaN);
  for (let i = 0; i < n; i++) {
    const fg = field.groups[points.group[i]!]!;
    const r = points.row[i]!;
    const v = new Float64Array(9);
    let ok = true;
    for (let s = 0; s < 9 && ok; s++) {
      const col = values[fg.slots[s]!]?.[fg.id];
      if (!col || r >= col.length) ok = false;
      else v[s] = col[r]!;
    }
    if (ok) out.set(v, 9 * i);
  }
  return out;
}

/**
 * Half a disc's width: 0.4 x the typical Gauss spacing, Python's
 * `default_half_length` taken to 3-D. The spacing is the distinct points' share
 * of the span they fill: `span / (n - 1)` on a line, `sqrt(area / n)` in a
 * plane, `cbrt(volume / n)` in a volume, so a disc stays under half a pitch
 * across and neighbouring discs do not touch.
 */
export function crackHalfSize(points: Float32Array): number {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  const n = points.length / 3;
  let finite = 0;
  for (let i = 0; i < n; i++) {
    const x = points[3 * i]!, y = points[3 * i + 1]!, z = points[3 * i + 2]!;
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    finite++;
    for (const [a, v] of [x, y, z].entries()) {
      lo[a] = Math.min(lo[a]!, v);
      hi[a] = Math.max(hi[a]!, v);
    }
  }
  if (finite === 0) return 1;
  const span = [0, 1, 2].map((a) => hi[a]! - lo[a]!);
  const extent = Math.max(...span);
  if (!(extent > 0)) return 1;
  const tol = extent * 1e-9;
  const distinct = new Set<string>();
  for (let i = 0; i < n; i++) {
    const x = points[3 * i]!, y = points[3 * i + 1]!, z = points[3 * i + 2]!;
    if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) distinct.add(`${Math.round(x / tol)},${Math.round(y / tol)},${Math.round(z / tol)}`);
  }
  const d = distinct.size;
  if (d <= 1) return 0.4 * extent;
  const filled = span.filter((s) => s > tol);
  if (filled.length === 1) return (0.4 * filled[0]!) / (d - 1);
  const measure = filled.reduce((p, s) => p * s, 1);
  return 0.4 * (measure / d) ** (1 / filled.length);
}

export interface CrackOptions {
  /** an opening below this is not drawn (>= 0) */
  minOpening: number;
  cmap: ContourCmap;
  /** half a disc's width, in world units */
  halfSize: number;
  /** the polygon's sides */
  sides?: number;
}

export interface Cracks3D {
  /** non-indexed triangles, 3 floats per vertex */
  positions: Float32Array;
  /** sRGB 0..1, 3 per vertex */
  colours: Float32Array;
  /** the discs drawn */
  discs: number;
  /** the Gauss points with a disc */
  points: number;
  /** the largest opening drawn: the colour range is minOpening..maxOpening (Python's `clim=None`) */
  maxOpening: number;
}

/** Sides of a crack disc: enough to read as round, few enough for a beam's tens of thousands of points. */
export const CRACK_SIDES = 8;

/**
 * A disc per crack kept: opening `|v|` > 0 and >= `minOpening`, centred at its
 * point, in the plane normal to `v`, coloured by the opening on `cmap` over
 * `minOpening`..the largest opening drawn (Python's `clim=None`). A point that is not placed (NaN) or has no
 * values is skipped.
 */
export function buildCracks3D(base: Float32Array, vectors: Float64Array, opts: CrackOptions): Cracks3D {
  const n = base.length / 3;
  const sides = opts.sides ?? CRACK_SIDES;
  type Item = { point: number; open: number; nx: number; ny: number; nz: number };
  const items: Item[] = [];
  let maxOpening = 0;
  for (let i = 0; i < n; i++) {
    const x = base[3 * i]!, y = base[3 * i + 1]!, z = base[3 * i + 2]!;
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    for (let k = 0; k < 3; k++) {
      const vx = vectors[9 * i + 3 * k]!, vy = vectors[9 * i + 3 * k + 1]!, vz = vectors[9 * i + 3 * k + 2]!;
      const open = Math.hypot(vx, vy, vz);
      if (!(open > 0) || open < opts.minOpening) continue;
      items.push({ point: i, open, nx: vx / open, ny: vy / open, nz: vz / open });
      maxOpening = Math.max(maxOpening, open);
    }
  }
  const per = 3 * sides;
  const positions = new Float32Array(items.length * per * 3);
  const colours = new Float32Array(items.length * per * 3);
  const cos = Array.from({ length: sides + 1 }, (_, j) => Math.cos((2 * Math.PI * j) / sides));
  const sin = Array.from({ length: sides + 1 }, (_, j) => Math.sin((2 * Math.PI * j) / sides));
  const h = opts.halfSize;
  const pts = new Set<number>();
  items.forEach((it, m) => {
    pts.add(it.point);
    const { u, v } = planeBasis(it.nx, it.ny, it.nz);
    const cx = base[3 * it.point]!, cy = base[3 * it.point + 1]!, cz = base[3 * it.point + 2]!;
    const rgb = sampleCmap(opts.cmap, rampT(it.open, opts.minOpening, maxOpening));
    let o = m * per * 3;
    for (let j = 0; j < sides; j++) {
      for (const [c, s] of [[0, 0], [cos[j]!, sin[j]!], [cos[j + 1]!, sin[j + 1]!]] as const) {
        positions[o] = cx + h * (c * u[0] + s * v[0]);
        positions[o + 1] = cy + h * (c * u[1] + s * v[1]);
        positions[o + 2] = cz + h * (c * u[2] + s * v[2]);
        colours.set(rgb, o);
        o += 3;
      }
    }
  });
  return { positions, colours, discs: items.length, points: pts.size, maxOpening };
}

/** Two unit vectors spanning the plane normal to the unit `n`, the first across the axis `n` is least along. */
export function planeBasis(nx: number, ny: number, nz: number): { u: [number, number, number]; v: [number, number, number] } {
  const ax = Math.abs(nx), ay = Math.abs(ny), az = Math.abs(nz);
  const e: [number, number, number] = ax <= ay && ax <= az ? [1, 0, 0] : ay <= az ? [0, 1, 0] : [0, 0, 1];
  // u = n x e, normalised; v = n x u (unit, as n and u are orthonormal).
  let ux = ny * e[2] - nz * e[1], uy = nz * e[0] - nx * e[2], uz = nx * e[1] - ny * e[0];
  const l = Math.hypot(ux, uy, uz);
  ux /= l;
  uy /= l;
  uz /= l;
  return { u: [ux, uy, uz], v: [ny * uz - nz * uy, nz * ux - nx * uz, nx * uy - ny * ux] };
}
