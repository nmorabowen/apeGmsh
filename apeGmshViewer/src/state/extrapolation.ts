// Gauss-point to node extrapolation (V4c2b, #1875), as pure functions: the
// app's port of Python `results/_gauss_extrapolation.py` (read as context,
// never imported; ADR 0112 D4). No three.js and no DOM: the tests run it in
// Node, and main runs it for the history scan.
//
// Python's pipeline, restated:
// - each element's Gauss values are projected onto the LINEAR corner shape
//   functions of its type (`_LINEAR_COUNTERPART`: tri6 -> tri3, hex20/27 ->
//   hex8, ...), evaluated at the group's natural coordinates
//   (`_build_extrapolation_matrix`): A = N(nat[:, :pdim]) is (G, nc) and
//   M = pinv(A) is (nc, G); a corner's value is M @ gp;
// - an element with one Gauss point gives that value to every corner
//   (`extrapolate_gauss_slab_per_element`, the n_gp == 1 branch);
// - the averaged topologies sum each corner value into its node and divide by
//   the number of elements that gave one (`GaussToNodeAccumulator`).
//
// Where this port is stricter than Python (fail closed, the card's seams):
// - a gmsh type code with no shape functions here (Python's `None`), a row
//   shorter than its corners, or natural coordinates with fewer columns than
//   the parent dimension read "unsupported": never Python's fallback (the
//   element mean on every corner, or the whole connectivity row as corners);
// - an element with a non-finite Gauss value has no data: its corners are
//   NaN and it adds nothing to any node (the divisor counts only elements
//   that gave a value), never a value of the points that survive.

import type { BlobStore } from "./blobs.ts";
import type { BlobRef, BlockInfo, Decl, DeclPath, ExtrapolationVerdict } from "./types.ts";

export type { ExtrapolationVerdict };

// -- shape functions (Python `fem/_shape_functions.py`, the linear types) -------

type ShapeFn = (x: readonly number[]) => number[];

const HEX8_SIGNS = [
  [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
  [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
] as const;
const QUAD4_SIGNS = [[-1, -1], [1, -1], [1, 1], [-1, 1]] as const;

/** Linear shape functions by gmsh code, corner order as gmsh's (and Python's `*_N`). */
const SHAPES: Readonly<Record<number, { nc: number; N: ShapeFn }>> = {
  1: { nc: 2, N: ([x]) => [0.5 * (1 - x!), 0.5 * (1 + x!)] },
  2: { nc: 3, N: ([x, y]) => [1 - x! - y!, x!, y!] },
  3: { nc: 4, N: (p) => QUAD4_SIGNS.map(([a, b]) => 0.25 * (1 + a * p[0]!) * (1 + b * p[1]!)) },
  4: { nc: 4, N: ([x, y, z]) => [1 - x! - y! - z!, x!, y!, z!] },
  5: { nc: 8, N: (p) => HEX8_SIGNS.map(([a, b, c]) => 0.125 * (1 + a * p[0]!) * (1 + b * p[1]!) * (1 + c * p[2]!)) },
  6: {
    nc: 6,
    N: ([x, y, z]) => {
      const l = [1 - x! - y!, x!, y!];
      const lo = 0.5 * (1 - z!);
      const hi = 0.5 * (1 + z!);
      return [l[0]! * lo, l[1]! * lo, l[2]! * lo, l[0]! * hi, l[1]! * hi, l[2]! * hi];
    },
  },
};

/** Python `_LINEAR_COUNTERPART`: a higher-order type is projected onto its linear corners. */
const LINEAR_COUNTERPART: Readonly<Record<number, number>> = { 9: 2, 10: 3, 11: 4, 12: 5, 16: 3, 17: 5 };

/** Python `_PARENT_DIM`: the natural coordinates the shape functions read (a code absent here is unsupported). */
const PARENT_DIM: Readonly<Record<number, number>> = { 1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 3, 9: 2, 10: 2, 11: 3, 12: 3, 16: 2, 17: 3 };

/** The linear corner shape functions of a gmsh code, and its parent dimension; null when this port (and Python) has none. */
export function shapeOf(code: number): { nc: number; pdim: number; N: ShapeFn } | null {
  const s = SHAPES[LINEAR_COUNTERPART[code] ?? code];
  const pdim = PARENT_DIM[code];
  return s && pdim !== undefined ? { nc: s.nc, pdim, N: s.N } : null;
}

// -- the extrapolation matrix -------------------------------------------------------

const EPS = 2.220446049250313e-16;

/**
 * The Moore-Penrose pseudo-inverse of `a` (m x n, row-major), n x m, by a
 * one-sided Jacobi SVD (Hestenes). Singular values below
 * `max(m, n) * eps * sigma_max` count as zero, numpy 2's `pinv` default.
 */
export function pinv(a: readonly (readonly number[])[]): number[][] {
  const m = a.length;
  const n = m ? a[0]!.length : 0;
  const u = a.map((r) => [...r]);
  const v = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 60; sweep++) {
    let rotated = false;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        let al = 0, be = 0, ga = 0;
        for (let i = 0; i < m; i++) {
          al += u[i]![p]! * u[i]![p]!;
          be += u[i]![q]! * u[i]![q]!;
          ga += u[i]![p]! * u[i]![q]!;
        }
        if (Math.abs(ga) <= EPS * Math.sqrt(al * be) || ga === 0) continue;
        rotated = true;
        const zeta = (be - al) / (2 * ga);
        const t = (zeta >= 0 ? 1 : -1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = c * t;
        for (const w of [u, v]) {
          for (const row of w) {
            const x = row[p]!, y = row[q]!;
            row[p] = c * x - s * y;
            row[q] = s * x + c * y;
          }
        }
      }
    }
    if (!rotated) break;
  }
  const sigma = Array.from({ length: n }, (_, j) => Math.sqrt(u.reduce((acc, r) => acc + r[j]! * r[j]!, 0)));
  const cut = Math.max(m, n) * EPS * Math.max(0, ...sigma);
  // pinv = V diag(1/sigma^2) U~^T, with U~ = U sigma (the unnormalised columns).
  const out = Array.from({ length: n }, () => new Array<number>(m).fill(0));
  for (let j = 0; j < n; j++) {
    const sj = sigma[j]!;
    if (!(sj > cut)) continue;
    const w = 1 / (sj * sj);
    for (let i = 0; i < n; i++) {
      const vij = v[i]![j]! * w;
      if (vij === 0) continue;
      for (let k = 0; k < m; k++) out[i]![k]! += vij * u[k]![j]!;
    }
  }
  return out;
}

/**
 * Python `_build_extrapolation_matrix`: M = pinv(N(nat[:, :pdim])), (nc x G)
 * row-major, for an element of gmsh `code` with Gauss points at `natural`;
 * null when unsupported (no shape functions, or too few coordinates).
 */
export function extrapolationMatrix(code: number, natural: readonly (readonly number[])[]): Float64Array | null {
  const sh = shapeOf(code);
  if (!sh || natural.some((r) => r.length < sh.pdim)) return null;
  const a = natural.map((r) => sh.N(r.slice(0, sh.pdim)));
  const m = pinv(a);
  const G = natural.length;
  const out = new Float64Array(sh.nc * G);
  for (let c = 0; c < sh.nc; c++) for (let g = 0; g < G; g++) out[c * G + g] = m[c]![g]!;
  return out;
}

// -- the topologies ---------------------------------------------------------------------

/**
 * The node-located element topologies (V4c2b). The Qt names
 * (`viewers/diagrams/_contour.py`): `gauss_cell_averaged` (one-point cell
 * values averaged at the nodes), `gauss_node` (extrapolated and averaged),
 * `gauss_node_discrete` (extrapolated per element, no averaging).
 * `element_mean_averaged` is the app's `cell_averaged` mode on a field with
 * several Gauss points (each element's mean, averaged at the nodes): Qt has
 * no such presentation (it extrapolates instead), so it carries no Qt name.
 */
export type NodeTopology = "gauss_cell_averaged" | "element_mean_averaged" | "gauss_node" | "gauss_node_discrete";

export const usesMatrix = (t: NodeTopology): boolean => t === "gauss_node" || t === "gauss_node_discrete";

/** One gauss group as the plan reads it. */
export interface PlanGroupInput {
  id: string;
  nElements: number;
  nGauss: number;
  /** `_element_index` (FEM ids) */
  eids: Float64Array;
  naturalCoords: readonly (readonly number[])[] | null;
}

/** An element's cell as the model has it: its gmsh code and its connectivity row (node tags, corners first). */
export type CellOf = (femId: number) => { code: number; nodes: ArrayLike<number> } | null;

/** The verdict on a field's groups (each element's type through `cellOf`). */
export function extrapolationVerdict(groups: readonly PlanGroupInput[], cellOf: CellOf): ExtrapolationVerdict {
  const corners: string[] = [];
  let nCorners = 0;
  const matrix: string[] = [];
  const codesSeen = new Map<string, Set<number>>();
  for (const g of groups) {
    const codes = new Set<number>();
    for (let e = 0; e < g.eids.length; e++) {
      const id = g.eids[e]!;
      const cell = cellOf(id);
      const sh = cell ? shapeOf(cell.code) : null;
      let why: string | null = null;
      if (!cell) why = `element ${id} has no cell in the model`;
      else if (!sh) why = `element ${id} is gmsh type ${cell.code}, which has no corner shape functions (Python's _build_extrapolation_matrix returns None)`;
      else if (cell.nodes.length < sh.nc) why = `element ${id} lists ${cell.nodes.length} nodes, fewer than the ${sh.nc} corners of gmsh type ${cell.code}`;
      if (why) {
        nCorners++;
        if (corners.length < 3) corners.push(why);
        continue;
      }
      codes.add(cell!.code);
    }
    codesSeen.set(g.id, codes);
    if (g.nGauss > 1) {
      if (!g.naturalCoords) matrix.push(`group ${g.id} has no floating-point _natural_coords`);
      else {
        for (const code of codes) {
          if (!extrapolationMatrix(code, g.naturalCoords)) {
            matrix.push(`group ${g.id}: its _natural_coords have ${g.naturalCoords[0]?.length ?? 0} columns, fewer than gmsh type ${code}'s ${shapeOf(code)!.pdim}`);
          }
        }
      }
    }
  }
  const more = nCorners > corners.length ? ` (and ${nCorners - corners.length} more)` : "";
  return { corners: corners.length ? corners.join("; ") + more : null, matrix: matrix.length ? matrix.join("; ") : null };
}

// -- the plan ---------------------------------------------------------------------

/**
 * What one field's values are carried through, built once per join and
 * topology and reused every step (and sent to main for the history scan:
 * plain arrays and typed arrays only, structured-clonable).
 */
export interface ExtrapPlan {
  topology: NodeTopology;
  /** average the corner values at the nodes (every topology but `gauss_node_discrete`) */
  averaged: boolean;
  /** the node of each slot, ascending FEM id */
  nodeIds: Float64Array;
  /** the node slot of every corner, all groups end to end */
  cornerSlot: Int32Array;
  groups: {
    id: string;
    nElements: number;
    nGauss: number;
    /** element e's corners are `cornerSlot[start[e] .. start[e + 1])` (global positions) */
    start: Int32Array;
    /** the matrix of element e: an index into `mats` */
    kind: Int32Array;
    /** (nc x G) row-major, or null: the element's mean (its one value when G == 1) on every corner */
    mats: (Float64Array | null)[];
  }[];
}

/**
 * The plan of a field: each element's corners (the first `nc` nodes of its
 * row) and matrix. Throws when the verdict says it cannot be built: the
 * caller shows "unsupported" before ever asking for a plan.
 */
export function buildPlan(topology: NodeTopology, groups: readonly PlanGroupInput[], cellOf: CellOf): ExtrapPlan {
  const v = extrapolationVerdict(groups, cellOf);
  if (v.corners || (usesMatrix(topology) && v.matrix)) throw new Error(`buildPlan: ${v.corners ?? v.matrix}`);
  const corners: number[] = [];
  const out: ExtrapPlan["groups"] = [];
  for (const g of groups) {
    const start = new Int32Array(g.eids.length + 1);
    const kind = new Int32Array(g.eids.length);
    const mats: (Float64Array | null)[] = [];
    const matOf = new Map<number, number>();
    for (let e = 0; e < g.eids.length; e++) {
      const cell = cellOf(g.eids[e]!)!;
      const sh = shapeOf(cell.code)!;
      start[e] = corners.length;
      for (let c = 0; c < sh.nc; c++) corners.push(cell.nodes[c]!);
      let k = matOf.get(cell.code);
      if (k === undefined) {
        k = mats.length;
        mats.push(usesMatrix(topology) && g.nGauss > 1 ? extrapolationMatrix(cell.code, g.naturalCoords!)! : null);
        matOf.set(cell.code, k);
      }
      kind[e] = k;
    }
    start[g.eids.length] = corners.length;
    out.push({ id: g.id, nElements: g.nElements, nGauss: g.nGauss, start, kind, mats });
  }
  const nodeIds = Float64Array.from(new Set(corners)).sort();
  const slot = new Map<number, number>();
  nodeIds.forEach((id, i) => slot.set(id, i));
  return { topology, averaged: topology !== "gauss_node_discrete", nodeIds, cornerSlot: Int32Array.from(corners, (id) => slot.get(id)!), groups: out };
}

/** One step carried through a plan: every corner's value, and (averaged) every node's. */
export interface PlanValues {
  /** by `cornerSlot` position; NaN for an element with a non-finite Gauss value */
  corner: Float64Array;
  /** by node slot; NaN where no element gave a value; null for the discrete topology */
  nodal: Float64Array | null;
}

/**
 * Carry one step of a field through its plan. `values[g]` is group g's
 * (E * G,) element-major block. An element with any non-finite Gauss value
 * has NaN corners and adds nothing to its nodes (the divisor counts only
 * elements that gave a value).
 */
export function applyPlan(plan: ExtrapPlan, values: readonly Float64Array[]): PlanValues {
  if (values.length !== plan.groups.length) throw new RangeError(`applyPlan: ${values.length} value blocks for ${plan.groups.length} groups`);
  const corner = new Float64Array(plan.cornerSlot.length);
  const sum = plan.averaged ? new Float64Array(plan.nodeIds.length) : null;
  const count = plan.averaged ? new Uint32Array(plan.nodeIds.length) : null;
  plan.groups.forEach((g, gi) => {
    const vals = values[gi]!;
    const G = g.nGauss;
    if (vals.length !== g.nElements * G) throw new RangeError(`applyPlan: group ${g.id} has ${vals.length} values for ${g.nElements} elements x ${G} Gauss points`);
    for (let e = 0; e < g.nElements; e++) {
      const base = e * G;
      const o = g.start[e]!;
      const n = g.start[e + 1]! - o;
      let ok = true;
      let s = 0;
      for (let j = 0; j < G; j++) {
        const x = vals[base + j]!;
        if (!Number.isFinite(x)) {
          ok = false;
          break;
        }
        s += x;
      }
      if (!ok) {
        corner.fill(NaN, o, o + n);
        continue;
      }
      const M = g.mats[g.kind[e]!]!;
      if (M === null) corner.fill(s / G, o, o + n);
      else {
        for (let c = 0; c < n; c++) {
          let acc = 0;
          for (let j = 0; j < G; j++) acc += M[c * G + j]! * vals[base + j]!;
          corner[o + c] = acc;
        }
      }
      if (sum) {
        for (let c = o; c < o + n; c++) {
          const sl = plan.cornerSlot[c]!;
          sum[sl]! += corner[c]!;
          count![sl]!++;
        }
      }
    }
  });
  let nodal: Float64Array | null = null;
  if (sum) {
    nodal = new Float64Array(sum.length);
    for (let i = 0; i < sum.length; i++) nodal[i] = count![i]! > 0 ? sum[i]! / count![i]! : NaN;
  }
  return { corner, nodal };
}

/** The finite range a step colours with: the nodes' (averaged) or every element corner's (discrete). */
export function planRange(r: PlanValues): { min: number; max: number } | null {
  const a = r.nodal ?? r.corner;
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < a.length; i++) {
    const v = a[i]!;
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return lo <= hi ? { min: lo, max: hi } : null;
}

// -- the state's side -----------------------------------------------------------

/**
 * The cell of a FEM id from the state: its decl (`mesh/element/<id>`), its
 * block's gmsh code, and its row of the block's connectivity blob.
 */
export function cellLookup(decls: Readonly<Record<DeclPath, Decl>>, blocks: readonly BlockInfo[], blobs: BlobStore): CellOf {
  const conn = new Map<number, Float64Array>();
  return (id) => {
    const cell = decls[`mesh/element/${id}`]?.element?.cell;
    const b = cell ? blocks[cell.block] : undefined;
    if (!cell || !b) return null;
    let c = conn.get(cell.block);
    if (!c) {
      const x = blobs.get(b.connectivity);
      if (!(x instanceof Float64Array)) throw new TypeError(`blob ${b.connectivity.id} is not f64`);
      conn.set(cell.block, (c = x));
    }
    return { code: b.code, nodes: c.subarray(cell.row * b.npe, (cell.row + 1) * b.npe) };
  };
}

let planMemo: { key: string; plan: ExtrapPlan } | null = null;

/** What `planFor` reads of an element contour view (selectors `ContourView`, state `element`). */
export interface PlanView {
  joinKey: string;
  topology: NodeTopology;
  groups: readonly { id: string; nElements: number; nGauss: number; eids: BlobRef; naturalCoords: readonly (readonly number[])[] | null }[];
}

/**
 * The plan of the element field on screen, built once per key (the join key,
 * which names the read, the mesh and the rows, and the topology): the
 * effects (step range, scan) and the viewport (colours) share it.
 */
export function planFor(view: PlanView, s: { decls: Readonly<Record<DeclPath, Decl>>; blocks: readonly BlockInfo[] }, blobs: BlobStore): ExtrapPlan {
  const key = `${view.joinKey}|${view.topology}`;
  if (planMemo?.key === key) return planMemo.plan;
  const groups = view.groups.map((g) => {
    const eids = blobs.get(g.eids);
    if (!(eids instanceof Float64Array)) throw new TypeError(`element contour: ${g.eids.id} is not f64`);
    return { id: g.id, nElements: g.nElements, nGauss: g.nGauss, eids, naturalCoords: g.naturalCoords };
  });
  const plan = buildPlan(view.topology, groups, cellLookup(s.decls, s.blocks, blobs));
  planMemo = { key, plan };
  return plan;
}

// -- vertices ---------------------------------------------------------------------

/**
 * The corner (a `cornerSlot` position) of every drawn vertex: the corner of
 * the vertex's own element (`primElement` -> `meshFem` -> the join) whose
 * node is the vertex's node; -1 ("no data") when the element has no row or
 * the node is not one of its corners. Never a corner found by position.
 */
export function vertexCorners(
  plan: ExtrapPlan,
  at: ReadonlyMap<number, { group: number; row: number }>,
  meshFem: Float64Array,
  primElement: Int32Array,
  vertexNodes: Float64Array,
  perPrim: number,
): Int32Array {
  const out = new Int32Array(primElement.length * perPrim).fill(-1);
  for (let p = 0; p < primElement.length; p++) {
    const fem = meshFem[primElement[p]!]!;
    const hit = Number.isNaN(fem) ? undefined : at.get(fem);
    if (!hit) continue;
    const g = plan.groups[hit.group]!;
    const o = g.start[hit.row]!;
    const end = g.start[hit.row + 1]!;
    for (let v = 0; v < perPrim; v++) {
      const node = vertexNodes[p * perPrim + v]!;
      for (let c = o; c < end; c++) {
        if (plan.nodeIds[plan.cornerSlot[c]!] === node) {
          out[p * perPrim + v] = c;
          break;
        }
      }
    }
  }
  return out;
}

/**
 * The value at every vertex: its element's own corner value (discrete), or
 * the node's average (averaged) when its own element gave a value; NaN ("no
 * data") otherwise. An element with no data never shows a neighbour's value.
 */
export function cornerVertexValues(plan: ExtrapPlan, r: PlanValues, corners: Int32Array): Float64Array {
  const out = new Float64Array(corners.length);
  for (let i = 0; i < corners.length; i++) {
    const c = corners[i]!;
    const own = c < 0 ? NaN : r.corner[c]!;
    out[i] = !Number.isFinite(own) ? NaN : r.nodal ? r.nodal[plan.cornerSlot[c]!]! : own;
  }
  return out;
}

/** A plan as a plain value main can check (results:scan): the fields and their types, never trusted unchecked. */
export function checkPlan(p: unknown): p is ExtrapPlan {
  const x = p as ExtrapPlan | null;
  if (!x || typeof x !== "object") return false;
  if (!["gauss_cell_averaged", "element_mean_averaged", "gauss_node", "gauss_node_discrete"].includes(x.topology)) return false;
  if (typeof x.averaged !== "boolean" || !(x.nodeIds instanceof Float64Array) || !(x.cornerSlot instanceof Int32Array) || !Array.isArray(x.groups)) return false;
  for (let i = 0; i < x.cornerSlot.length; i++) if (!(x.cornerSlot[i]! >= 0 && x.cornerSlot[i]! < x.nodeIds.length)) return false;
  return x.groups.every((g) => {
    if (typeof g.id !== "string" || !Number.isInteger(g.nElements) || !Number.isInteger(g.nGauss) || g.nGauss < 1) return false;
    if (!(g.start instanceof Int32Array) || !(g.kind instanceof Int32Array) || !Array.isArray(g.mats)) return false;
    if (g.start.length !== g.nElements + 1 || g.kind.length !== g.nElements) return false;
    for (let e = 0; e < g.nElements; e++) {
      const o = g.start[e]!, n = g.start[e + 1]! - o;
      if (o < 0 || n < 1 || g.start[e + 1]! > x.cornerSlot.length) return false;
      const M = g.mats[g.kind[e]!];
      if (M === undefined || (M !== null && !(M instanceof Float64Array && M.length === n * g.nGauss))) return false;
    }
    return true;
  });
}
