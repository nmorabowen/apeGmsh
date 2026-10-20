// Turn a ModelFile into flat render buffers and a legend of the physical
// groups that colour them. Colours themselves come from the theme
// (src/theme/tokens.ts, through src/state/palette.ts); this module hands
// out the group of every primitive, never a colour.
//
// Pure: no three.js and no DOM, so the tests run it in Node. Cells are
// recognised by their gmsh element-type `code` (gmsh's own numbering,
// stored on every /elements block); only corner nodes are drawn. Solids
// draw their boundary faces only. An unknown code is a loud warning.
//
// No Map or Set here is sized by the node or element count (#2359): V8 caps
// both at 2^24 entries. Ids resolve through `IdLookup` (sorted typed arrays),
// and the solid-face and outline-edge deduplication sorts typed arrays.

import type { ElementRef } from "../chain/resolve.ts";
import type { ModelFile } from "../model/types.ts";

type Shape = "point" | "line" | "tri" | "quad" | "tet" | "hex" | "prism" | "pyramid";

/** gmsh element-type code -> base shape (corner nodes come first in gmsh order). */
export const GMSH_SHAPE: Readonly<Record<number, Shape>> = {
  15: "point",
  1: "line", 8: "line", 26: "line", 27: "line", 28: "line",
  2: "tri", 9: "tri", 20: "tri", 21: "tri", 22: "tri", 23: "tri", 24: "tri", 25: "tri",
  3: "quad", 10: "quad", 16: "quad", 36: "quad", 37: "quad", 38: "quad",
  4: "tet", 11: "tet", 29: "tet", 30: "tet", 31: "tet",
  5: "hex", 12: "hex", 17: "hex", 92: "hex", 93: "hex",
  6: "prism", 13: "prism", 18: "prism",
  7: "pyramid", 14: "pyramid", 19: "pyramid",
};

/** Faces of each solid, as corner-index lists in gmsh ordering, outward. */
const SOLID_FACES: Readonly<Record<"tet" | "hex" | "prism" | "pyramid", number[][]>> = {
  tet: [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
  hex: [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]],
  prism: [[0, 2, 1], [3, 4, 5], [0, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5]],
  pyramid: [[0, 3, 2, 1], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
};

const CORNERS: Readonly<Record<Shape, number>> = {
  point: 1, line: 2, tri: 3, quad: 4, tet: 4, hex: 8, prism: 6, pyramid: 5,
};

/** One legend row: a physical group (or one of the two synthetic rows) and how many cells it colours. */
export interface LegendEntry {
  name: string;
  elements: number;
}

export interface MeshBuffers {
  /** 6 floats per segment (two xyz endpoints). */
  linePositions: Float32Array;
  /** V4c1: the node tag of each segment endpoint (2 per segment), from the connectivity the position was read from. */
  lineNodes: Float64Array;
  lineRefs: ElementRef[];
  /** 9 floats per triangle. */
  triPositions: Float32Array;
  /** V4c1: the node tag of each triangle vertex (3 per triangle). */
  triNodes: Float64Array;
  triRefs: ElementRef[];
  /** 6 floats per outline edge of the drawn faces. */
  edgePositions: Float32Array;
  /** V4d: the node tag of each outline-edge vertex (2 per edge), from the polygon it outlines. */
  edgeNodes: Float64Array;
  legend: LegendEntry[];
  center: [number, number, number];
  radius: number;
  /** cells drawn, by kind */
  counts: { lineCells: number; faceCells: number; solidCells: number; opsOnly: number; points: number };
  warnings: string[];
}

/** Which value a repeated key keeps: its first (`if (!m.has(k)) m.set(k, v)`) or its last (`m.set(k, v)` every time). */
export type RepeatedKey = "first" | "last";

/** The earliest second listing of any key in an `IdLookup`'s list (`again`), and where that key was first listed (`first`). */
export interface RepeatedListing {
  first: number;
  again: number;
}

/**
 * A read-only lookup from a number (an id) to a value, with a Map's reading
 * API (`get`, `has`, `size`, iteration), held in sorted typed arrays (#2359).
 * V8 caps a `Map` and a `Set` at 2^24 entries (a RangeError past 16,777,216),
 * and a stage or a model can hold more nodes or elements than that; a binary
 * search over a sorted copy has no such cap.
 *
 * It answers what the Map it replaces answered: keys compare as a Map's do
 * (SameValueZero: NaN equals NaN, -0 equals 0 and is stored as 0), a repeated
 * key keeps the value `repeated` names, and iteration runs in the order each
 * key was first listed. It is read-only: `set`, `delete` and `clear` throw.
 * `value(i)` is called on each `get`, with `i` the position of the listing
 * the key keeps.
 */
export class IdLookup<V = number> implements Map<number, V> {
  /** the distinct keys, ascending, NaN excluded */
  readonly #keys: Float64Array;
  /** the position of the listing each key of `#keys` keeps */
  readonly #at: Float64Array;
  /** where each key of `#keys` was first listed: the iteration order */
  readonly #first: Float64Array;
  /** NaN's kept and first positions, when a NaN key was listed */
  readonly #nan: { at: number; first: number } | null;
  readonly #value: (i: number) => V;
  /** The earliest second listing of a key, or null when every key is listed once. */
  readonly repeat: RepeatedListing | null;
  /** rows of `#keys` in first-listed order, built on the first iteration */
  #order: Uint32Array | null = null;

  private constructor(
    keys: Float64Array,
    at: Float64Array,
    first: Float64Array,
    nan: { at: number; first: number } | null,
    value: (i: number) => V,
    repeat: RepeatedListing | null,
  ) {
    this.#keys = keys;
    this.#at = at;
    this.#first = first;
    this.#nan = nan;
    this.#value = value;
    this.repeat = repeat;
  }

  /** The lookup `keys[i] -> value(i)`, where a repeated key keeps its `repeated` listing. */
  static of<V>(keys: ArrayLike<number>, value: (i: number) => V, repeated: RepeatedKey): IdLookup<V> {
    const n = keys.length;
    let ascending = true;
    for (let i = 0; i < n && ascending; i++) ascending = !Number.isNaN(keys[i]!) && (i === 0 || keys[i]! > keys[i - 1]!);
    if (ascending) {
      const at = Float64Array.from({ length: n }, (_, i) => i);
      const lookup = new IdLookup(Float64Array.from({ length: n }, (_, i) => (keys[i] === 0 ? 0 : keys[i]!)), at, at, null, value, null);
      lookup.#order = Uint32Array.from({ length: n }, (_, i) => i);
      return lookup;
    }
    // The positions of the non-NaN keys, sorted by key and then by position:
    // each run of one key starts at its first listing and ends at its last.
    let nan: { at: number; first: number } | null = null;
    let repeat: RepeatedListing | null = null;
    const repeatAt = (first: number, again: number) => {
      if (repeat === null || again < repeat.again) repeat = { first, again };
    };
    let m = 0;
    for (let i = 0; i < n; i++) {
      if (!Number.isNaN(keys[i]!)) m++;
      else if (nan === null) nan = { at: i, first: i };
      else {
        if (nan.at === nan.first) repeatAt(nan.first, i);
        if (repeated === "last") nan.at = i;
      }
    }
    const perm = new Uint32Array(m);
    for (let i = 0, j = 0; i < n; i++) if (!Number.isNaN(keys[i]!)) perm[j++] = i;
    perm.sort((a, b) => keys[a]! - keys[b]! || a - b);
    let u = 0;
    for (let r = 0; r < m; r++) if (r === 0 || keys[perm[r]!] !== keys[perm[r - 1]!]) u++;
    const k = new Float64Array(u), at = new Float64Array(u), first = new Float64Array(u);
    for (let r = 0, w = -1; r < m; r++) {
      const i = perm[r]!;
      if (r === 0 || keys[i] !== keys[perm[r - 1]!]) {
        w++;
        k[w] = keys[i] === 0 ? 0 : keys[i]!;
        first[w] = i;
        at[w] = i;
        continue;
      }
      if (perm[r - 1] === first[w]) repeatAt(first[w]!, i);
      if (repeated === "last") at[w] = i;
    }
    return new IdLookup(k, at, first, nan, value, repeat);
  }

  /** The row of `key` in `#keys`, or -1 (a binary search). */
  #row(key: number): number {
    const keys = this.#keys;
    let lo = 0, hi = keys.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >>> 1;
      const x = keys[mid]!;
      if (x === key) return mid;
      if (x < key) lo = mid + 1;
      else hi = mid - 1;
    }
    return -1;
  }

  get(key: number): V | undefined {
    if (Number.isNaN(key)) return this.#nan === null ? undefined : this.#value(this.#nan.at);
    const r = this.#row(key);
    return r < 0 ? undefined : this.#value(this.#at[r]!);
  }

  has(key: number): boolean {
    return Number.isNaN(key) ? this.#nan !== null : this.#row(key) >= 0;
  }

  get size(): number {
    return this.#keys.length + (this.#nan === null ? 0 : 1);
  }

  *entries(): MapIterator<[number, V]> {
    if (this.#order === null) {
      const first = this.#first;
      this.#order = Uint32Array.from({ length: first.length }, (_, i) => i).sort((a, b) => first[a]! - first[b]!);
    }
    const order = this.#order;
    const nan = this.#nan;
    let nanDue = nan !== null;
    for (let i = 0; i < order.length; i++) {
      const r = order[i]!;
      if (nanDue && nan!.first < this.#first[r]!) {
        nanDue = false;
        yield [NaN, this.#value(nan!.at)];
      }
      yield [this.#keys[r]!, this.#value(this.#at[r]!)];
    }
    if (nanDue) yield [NaN, this.#value(nan!.at)];
  }

  *keys(): MapIterator<number> {
    for (const [k] of this.entries()) yield k;
  }

  *values(): MapIterator<V> {
    for (const [, v] of this.entries()) yield v;
  }

  [Symbol.iterator](): MapIterator<[number, V]> {
    return this.entries();
  }

  forEach(callbackfn: (value: V, key: number, map: Map<number, V>) => void, thisArg?: unknown): void {
    for (const [k, v] of this.entries()) callbackfn.call(thisArg, v, k, this);
  }

  get [Symbol.toStringTag](): string {
    return "IdLookup";
  }

  set(key: number, value: V): this {
    throw new TypeError(`IdLookup is read-only: set(${key}, ${String(value)}) refused`);
  }

  delete(key: number): boolean {
    throw new TypeError(`IdLookup is read-only: delete(${key}) refused`);
  }

  clear(): void {
    throw new TypeError("IdLookup is read-only: clear() refused");
  }
}

/** A growable Float64Array: a plain `number[]` stops at 2^27 entries in V8. */
class F64List {
  data = new Float64Array(1024);
  length = 0;

  push(x: number): void {
    if (this.length === this.data.length) {
      const grown = new Float64Array(2 * this.data.length);
      grown.set(this.data);
      this.data = grown;
    }
    this.data[this.length++] = x;
  }
}

/**
 * The colour group of each FEM element: the smallest element-side physical
 * group that contains it (the most specific), ties broken by name.
 */
export function colourGroups(model: ModelFile): { byElement: IdLookup; order: string[] } {
  const groups = model.physicalGroups
    .filter((g) => g.elementIds.length > 0)
    .slice()
    .sort((a, b) => a.elementIds.length - b.elementIds.length || a.name.localeCompare(b.name));
  // Every group's element ids in group order, each beside its group: an
  // element in two groups keeps the first, the most specific.
  let n = 0;
  for (const g of groups) n += g.elementIds.length;
  const ids = new Float64Array(n);
  const group = new Uint32Array(n);
  let o = 0;
  groups.forEach((g, gi) => {
    ids.set(g.elementIds, o);
    group.fill(gi, o, o + g.elementIds.length);
    o += g.elementIds.length;
  });
  return { byElement: IdLookup.of(ids, (i) => group[i]!, "first"), order: groups.map((g) => g.name) };
}

/**
 * The faces (of `n`, 4 corner indices each, -1 padding a triangle) whose
 * corner set no other face shares, ascending: the boundary of the solids.
 */
function boundaryFaces(corners: Float64Array, n: number): Uint32Array {
  // Each face's corners sorted: two faces are one face when these agree.
  const key = corners.slice(0, 4 * n);
  for (let f = 0; f < n; f++) {
    for (let i = 4 * f + 1; i < 4 * f + 4; i++) {
      const x = key[i]!;
      let j = i - 1;
      for (; j >= 4 * f && key[j]! > x; j--) key[j + 1] = key[j]!;
      key[j + 1] = x;
    }
  }
  const perm = Uint32Array.from({ length: n }, (_, f) => f);
  perm.sort((a, b) => key[4 * a]! - key[4 * b]! || key[4 * a + 1]! - key[4 * b + 1]! || key[4 * a + 2]! - key[4 * b + 2]! || key[4 * a + 3]! - key[4 * b + 3]! || a - b);
  const same = (a: number, b: number) => key[4 * a] === key[4 * b] && key[4 * a + 1] === key[4 * b + 1] && key[4 * a + 2] === key[4 * b + 2] && key[4 * a + 3] === key[4 * b + 3];
  const once = new Uint8Array(n);
  let count = 0;
  for (let r = 0; r < n; r++) {
    const f = perm[r]!;
    if ((r === 0 || !same(f, perm[r - 1]!)) && (r === n - 1 || !same(f, perm[r + 1]!))) {
      once[f] = 1;
      count++;
    }
  }
  const out = new Uint32Array(count);
  for (let f = 0, w = 0; f < n; f++) if (once[f]) out[w++] = f;
  return out;
}

/** Whether each of the `n` pairs `(lo[i], hi[i])` is the first listing of that pair. */
function firstOfEach(lo: Float64Array, hi: Float64Array, n: number): Uint8Array {
  const perm = Uint32Array.from({ length: n }, (_, i) => i);
  perm.sort((a, b) => lo[a]! - lo[b]! || hi[a]! - hi[b]! || a - b);
  const first = new Uint8Array(n);
  for (let r = 0; r < n; r++) {
    const i = perm[r]!, p = perm[r - 1]!;
    if (r === 0 || lo[i] !== lo[p] || hi[i] !== hi[p]) first[i] = 1;
  }
  return first;
}

/** How many distinct values `values` holds, compared as a Set compares them (NaN equals NaN). */
function distinctCount(values: Float64Array): number {
  const sorted = values.slice().sort();
  let u = 0;
  for (let i = 0; i < sorted.length; i++) {
    const x = sorted[i]!, p = sorted[i - 1]!;
    if (i === 0 || !(x === p || (Number.isNaN(x) && Number.isNaN(p)))) u++;
  }
  return u;
}

export function buildMesh(model: ModelFile): MeshBuffers {
  const warnings: string[] = [];
  // A node id listed twice resolves to its last row, as the Map this replaced did.
  const nodeIndex = IdLookup.of(model.nodeIds, (i) => i, "last");
  const xyz = model.nodeCoords;

  const { byElement, order } = colourGroups(model);
  const legendCount = new Array<number>(order.length).fill(0);
  let noGroup = 0;
  /** Count the cell in its colour group's legend row (or in the no-group row). */
  const count = (femId: number): void => {
    const gi = byElement.get(femId);
    if (gi === undefined) noGroup++;
    else legendCount[gi]!++;
  };

  const lp: number[] = [], lr: ElementRef[] = [];
  const tp: number[] = [], tr: ElementRef[] = [];
  const counts = { lineCells: 0, faceCells: 0, solidCells: 0, opsOnly: 0, points: 0 };
  /** every reference to a node tag not in /nodes/ids, counted distinct at the end */
  const missingNodes = new F64List();

  const at = (nodeId: number): number => {
    const i = nodeIndex.get(nodeId);
    if (i === undefined) {
      missingNodes.push(nodeId);
      return -1;
    }
    return i;
  };
  // V4c1: the node tag of every drawn vertex, beside its position, from the
  // same node index the position is read from (never matched by position).
  const ln: number[] = [], tn: number[] = [];
  const push3 = (arr: number[], i: number) => arr.push(xyz[3 * i]!, xyz[3 * i + 1]!, xyz[3 * i + 2]!);
  const pushSeg = (a: number, b: number, ref: ElementRef) => {
    push3(lp, a); push3(lp, b);
    ln.push(model.nodeIds[a]!, model.nodeIds[b]!);
    lr.push(ref);
  };
  const polygons: number[][] = [];
  const pushFace = (ids: number[], ref: ElementRef) => {
    polygons.push(ids);
    for (let k = 1; k + 1 < ids.length; k++) {
      push3(tp, ids[0]!); push3(tp, ids[k]!); push3(tp, ids[k + 1]!);
      tn.push(model.nodeIds[ids[0]!]!, model.nodeIds[ids[k]!]!, model.nodeIds[ids[k + 1]!]!);
      tr.push(ref);
    }
  };

  // Solid boundary faces: a face seen once is on the boundary. Each solid
  // face, in the order seen: its corners in drawing order (a triangle's
  // fourth slot -1) and the solid it bounds (an index into `solidRefs`).
  const faceCorners = new F64List();
  const faceSolid = new F64List();
  const solidRefs: ElementRef[] = [];

  model.blocks.forEach((b, blockIndex) => {
    const shape = GMSH_SHAPE[b.code];
    if (shape === undefined) {
      warnings.push(`/elements/${b.alias}: gmsh type code ${b.code} is not drawn (${b.ids.length} cells)`);
      return;
    }
    const nc = CORNERS[shape];
    if (b.npe < nc) {
      warnings.push(`/elements/${b.alias}: npe ${b.npe} < ${nc} corners of a ${shape}; not drawn`);
      return;
    }
    for (let row = 0; row < b.ids.length; row++) {
      const ref: ElementRef = { kind: "fem", blockIndex, row };
      const corner: number[] = [];
      let ok = true;
      for (let k = 0; k < nc; k++) {
        const i = at(b.connectivity[row * b.npe + k]!);
        if (i < 0) ok = false;
        corner.push(i);
      }
      if (!ok) continue;
      if (shape === "point") {
        counts.points++;
        continue;
      }
      count(b.ids[row]!);
      if (shape === "line") {
        counts.lineCells++;
        pushSeg(corner[0]!, corner[1]!, ref);
      } else if (shape === "tri" || shape === "quad") {
        counts.faceCells++;
        pushFace(corner, ref);
      } else {
        counts.solidCells++;
        const solid = solidRefs.length;
        solidRefs.push(ref);
        for (const f of SOLID_FACES[shape]) {
          for (let k = 0; k < 4; k++) faceCorners.push(k < f.length ? corner[f[k]!]! : -1);
          faceSolid.push(solid);
        }
      }
    }
  });
  for (const f of boundaryFaces(faceCorners.data, faceSolid.length)) {
    const ids: number[] = [];
    for (let k = 0; k < 4; k++) if (faceCorners.data[4 * f + k]! >= 0) ids.push(faceCorners.data[4 * f + k]!);
    pushFace(ids, solidRefs[faceSolid.data[f]!]!);
  }

  // OpenSees-only elements (no neutral-zone cell): drawn from inline connectivity.
  model.opensees?.elementMeta.forEach((meta, metaIndex) => {
    if (!meta.inlineConnectivity) return;
    meta.inlineConnectivity.forEach((conn, row) => {
      if (meta.femEids[row]! >= 0 || conn.length < 2) return;
      const idx = conn.map(at);
      if (idx.some((i) => i < 0)) return;
      counts.opsOnly++;
      for (let k = 0; k + 1 < idx.length; k++) pushSeg(idx[k]!, idx[k + 1]!, { kind: "ops", metaIndex, row });
    });
  });

  const missing = distinctCount(missingNodes.data.subarray(0, missingNodes.length));
  if (missing) {
    warnings.push(`${missing} node tags referenced by cells are not in /nodes/ids; those cells are not drawn`);
  }

  // Outline: the perimeter edges of every drawn polygon, each edge once
  // (so a split quad shows no diagonal), in the order first seen.
  const edgeLo = new F64List(), edgeHi = new F64List();
  for (const poly of polygons) {
    for (let k = 0; k < poly.length; k++) {
      const a = poly[k]!, b = poly[(k + 1) % poly.length]!;
      edgeLo.push(Math.min(a, b));
      edgeHi.push(Math.max(a, b));
    }
  }
  const firstSeen = firstOfEach(edgeLo.data, edgeHi.data, edgeLo.length);
  const ep: number[] = [], en: number[] = [];
  let e = 0;
  for (const poly of polygons) {
    for (let k = 0; k < poly.length; k++, e++) {
      if (!firstSeen[e]) continue;
      const a = poly[k]!, b = poly[(k + 1) % poly.length]!;
      push3(ep, a); push3(ep, b);
      en.push(model.nodeIds[a]!, model.nodeIds[b]!);
    }
  }

  // Bounds over all nodes.
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < xyz.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      lo[k] = Math.min(lo[k]!, xyz[i + k]!);
      hi[k] = Math.max(hi[k]!, xyz[i + k]!);
    }
  }
  const center: [number, number, number] = [0, 1, 2].map((k) => (lo[k]! + hi[k]!) / 2) as [number, number, number];
  const radius = Math.max(1e-9, Math.hypot(hi[0]! - lo[0]!, hi[1]! - lo[1]!, hi[2]! - lo[2]!) / 2);

  const legend: LegendEntry[] = order
    .map((name, gi) => ({ name, elements: legendCount[gi]! }))
    .filter((e) => e.elements > 0)
    .sort((a, b) => b.elements - a.elements || a.name.localeCompare(b.name));
  if (noGroup) legend.push({ name: "(no physical group)", elements: noGroup });
  if (counts.opsOnly) legend.push({ name: "(OpenSees-only, no cell)", elements: counts.opsOnly });

  return {
    linePositions: new Float32Array(lp),
    lineNodes: new Float64Array(ln),
    lineRefs: lr,
    triPositions: new Float32Array(tp),
    triNodes: new Float64Array(tn),
    triRefs: tr,
    edgePositions: new Float32Array(ep),
    edgeNodes: new Float64Array(en),
    legend,
    center,
    radius,
    counts,
    warnings,
  };
}
