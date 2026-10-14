// The cut face of a solid (V4e3, #1904; ADR 0083 S3): the volumetric cells
// cut by one plane, as triangles carrying the field interpolated onto them.
// Pure: no three.js and no DOM, so the tests run it in Node, and the
// PyVista oracle (fixtures/slice.oracle.json, written by
// scripts/make_viewer_results_fixtures.py --slice-oracle through the Qt
// backend's `PyVistaBackend.slice_layer`) checks it.
//
// Each cell is cut exactly: the plane meets the cell's edges, and those
// points, in order around their centroid, are the cell's polygon (a cell
// with planar faces is convex, so the polygon is too). Values are carried
// linearly along each cut edge, as VTK's cutter carries point data, so a
// cut point holds the value VTK puts there. The polygon is then trimmed by
// the other planes' kept half-spaces (Sutherland-Hodgman, values carried the
// same way) and split into a fan.
//
// A corner exactly on the plane counts as cut away (`d > 0` is kept). So a
// plane through a face two cells share makes that face once, from the kept
// cell, never twice; and a plane that only touches a cell (a corner or an
// edge) makes no polygon. A cut point at a corner takes the corner's own
// value, so a corner with no data (NaN) is never mixed into a neighbour's
// point, and a point between a valued corner and an empty one is no data.
// No data is never 0.

import { GMSH_SHAPE } from "../mesh/build.ts";
import type { Cut } from "./clip.ts";

export type SolidShape = "tet" | "hex" | "prism" | "pyramid";

/** The volumetric shapes, in the order `Solids.shape` indexes. */
export const SOLID_SHAPES: readonly SolidShape[] = ["tet", "hex", "prism", "pyramid"];

/** Corner count of each solid (gmsh corner order: corners come first). */
export const SOLID_CORNERS: Readonly<Record<SolidShape, number>> = { tet: 4, hex: 8, prism: 6, pyramid: 5 };

/** The edges of each solid, as corner pairs in gmsh ordering. */
export const SOLID_EDGES: Readonly<Record<SolidShape, readonly (readonly [number, number])[]>> = {
  tet: [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]],
  hex: [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]],
  prism: [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3], [0, 3], [1, 4], [2, 5]],
  pyramid: [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 4], [2, 4], [3, 4]],
};

/** The solid shape of a gmsh element-type code; null for a point, line or face cell (or an unknown code). */
export function solidShapeOf(code: number): SolidShape | null {
  const s = GMSH_SHAPE[code];
  return s === "tet" || s === "hex" || s === "prism" || s === "pyramid" ? s : null;
}

/** Volumetric cells over a table of points. */
export interface Solids {
  /** per cell, its shape (an index into SOLID_SHAPES) */
  shape: Uint8Array;
  /** cell `c`'s corners are `slot[start[c] .. start[c + 1])`; length cells + 1 */
  start: Int32Array;
  /** per corner, the row of its point in the xyz table */
  slot: Int32Array;
}

/** One plane's cut face. */
export interface Slice {
  /** 9 per triangle */
  positions: Float64Array;
  /** 3 per triangle (NaN: no data); null when sliced without values */
  values: Float64Array | null;
  /** the cell of each triangle */
  cell: Int32Array;
  /** the summed triangle area */
  area: number;
}

/** A polygon vertex: position and value. */
type Pt = [number, number, number, number];

const lerp = (a: Pt, b: Pt, t: number): Pt => {
  // At an end the end itself, value included: a NaN on the other end must not leak in.
  if (t === 0) return a;
  if (t === 1) return b;
  return [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2]), a[3] + t * (b[3] - a[3])];
};

const dist = (c: Cut, p: Pt | readonly number[]): number => c.normal[0] * p[0]! + c.normal[1] * p[1]! + c.normal[2] * p[2]! + c.constant;

/** Two unit axes spanning the plane of `n` (n unit). */
function planeAxes(n: readonly [number, number, number]): [[number, number, number], [number, number, number]] {
  // The world axis least along n, crossed with n.
  const ax = Math.abs(n[0]), ay = Math.abs(n[1]), az = Math.abs(n[2]);
  const w: [number, number, number] = ax <= ay && ax <= az ? [1, 0, 0] : ay <= az ? [0, 1, 0] : [0, 0, 1];
  const u: [number, number, number] = [n[1] * w[2] - n[2] * w[1], n[2] * w[0] - n[0] * w[2], n[0] * w[1] - n[1] * w[0]];
  const m = Math.hypot(u[0], u[1], u[2]);
  u[0] /= m; u[1] /= m; u[2] /= m;
  const v: [number, number, number] = [n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2], n[0] * u[1] - n[1] * u[0]];
  return [u, v];
}

/** Keep the part of a convex polygon in `c`'s kept half-space (`d >= 0`), values carried linearly. */
export function trimPolygon(poly: readonly Pt[], c: Cut): Pt[] {
  const out: Pt[] = [];
  const n = poly.length;
  for (let i = 0; i < n; i++) {
    const a = poly[i]!, b = poly[(i + 1) % n]!;
    const da = dist(c, a), db = dist(c, b);
    if (da >= 0) out.push(a);
    if ((da >= 0) !== (db >= 0)) out.push(lerp(a, b, da / (da - db)));
  }
  return out;
}

/**
 * The cut face of `solids` at the plane of `cut` (`n·p + constant = 0`),
 * trimmed by `trims` (the other planes' kept half-spaces). `xyz`: 3 per
 * point; `values`: one per corner (`Solids.slot` order: a cell's own value
 * at each of its corners, so a field drawn per cell or per element corner
 * is carried as drawn), or null for a cut with no field. Null when the
 * plane cuts no cell: never an empty or degenerate slice.
 */
export function sliceSolids(solids: Solids, xyz: Float64Array, values: Float64Array | null, cut: Cut, trims: readonly Cut[] = []): Slice | null {
  if (values && values.length !== solids.slot.length) throw new RangeError(`sliceSolids: ${values.length} values for ${solids.slot.length} corners`);
  const [u, v] = planeAxes(cut.normal);
  const pos: number[] = [], val: number[] = [], cells: number[] = [];
  let area = 0;
  const nCells = solids.shape.length;
  const d = new Float64Array(8);
  const pts: Pt[] = new Array(8);
  for (let c = 0; c < nCells; c++) {
    const shape = SOLID_SHAPES[solids.shape[c]!]!;
    const o = solids.start[c]!;
    const nc = solids.start[c + 1]! - o;
    if (nc !== SOLID_CORNERS[shape]) throw new RangeError(`sliceSolids: cell ${c} is a ${shape} with ${nc} corners`);
    let above = 0;
    for (let k = 0; k < nc; k++) {
      const s = solids.slot[o + k]!;
      const p: Pt = [xyz[3 * s]!, xyz[3 * s + 1]!, xyz[3 * s + 2]!, values ? values[o + k]! : 0];
      pts[k] = p;
      d[k] = dist(cut, p);
      if (d[k]! > 0) above++;
    }
    if (above === 0 || above === nc) continue;
    // The cut points, each once: a corner on the plane is met by several edges.
    const poly: Pt[] = [];
    const seen: number[] = [];
    for (const [a, b] of SOLID_EDGES[shape]) {
      const da = d[a]!, db = d[b]!;
      if (da > 0 === db > 0) continue;
      const t = da / (da - db);
      const key = t === 0 ? -1 - a : t === 1 ? -1 - b : (a + 1) * 16 + b;
      if (seen.includes(key)) continue;
      seen.push(key);
      poly.push(lerp(pts[a]!, pts[b]!, t));
    }
    if (poly.length < 3) continue;
    // In order around the centroid, in the plane's own axes.
    let cx = 0, cy = 0, cz = 0;
    for (const p of poly) { cx += p[0]; cy += p[1]; cz += p[2]; }
    cx /= poly.length; cy /= poly.length; cz /= poly.length;
    const angle = (p: Pt) => Math.atan2((p[0] - cx) * v[0] + (p[1] - cy) * v[1] + (p[2] - cz) * v[2], (p[0] - cx) * u[0] + (p[1] - cy) * u[1] + (p[2] - cz) * u[2]);
    const sorted = poly.map((p) => ({ p, a: angle(p) })).sort((x, y) => x.a - y.a).map((x) => x.p);
    let kept: Pt[] = sorted;
    for (const t of trims) {
      kept = trimPolygon(kept, t);
      if (kept.length < 3) break;
    }
    if (kept.length < 3) continue;
    const p0 = kept[0]!;
    for (let k = 1; k + 1 < kept.length; k++) {
      const p1 = kept[k]!, p2 = kept[k + 1]!;
      const e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]], e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
      const a = 0.5 * Math.hypot(e1[1]! * e2[2]! - e1[2]! * e2[1]!, e1[2]! * e2[0]! - e1[0]! * e2[2]!, e1[0]! * e2[1]! - e1[1]! * e2[0]!);
      if (!(a > 0)) continue; // a sliver of no area (collinear trim points) draws nothing
      area += a;
      pos.push(p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]);
      val.push(p0[3], p1[3], p2[3]);
      cells.push(c);
    }
  }
  if (!cells.length) return null;
  return { positions: Float64Array.from(pos), values: values ? Float64Array.from(val) : null, cell: Int32Array.from(cells), area };
}

/** The axis-aligned extent of a slice (its box's diagonal). */
export function sliceExtent(s: Slice): number {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < s.positions.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      const x = s.positions[i + k]!;
      if (x < lo[k]!) lo[k] = x;
      if (x > hi[k]!) hi[k] = x;
    }
  }
  return Math.hypot(hi[0]! - lo[0]!, hi[1]! - lo[1]!, hi[2]! - lo[2]!);
}
