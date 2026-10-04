// Which drawn primitives the hidden set leaves visible: pure, so the
// viewport's rebuild decision is testable without a GPU.

import type { DeclPath, LegendEntry } from "../state/types.ts";

/** Original indices of the segments, triangles and outline edges to draw, in order. */
export interface Drawn {
  lineMap: Int32Array;
  triMap: Int32Array;
  /** null: every outline edge */
  edgeMask: Uint8Array | null;
}

/** The maps of the primitives whose legend row is not hidden. */
export function visibleMaps(legend: readonly LegendEntry[], hidden: readonly DeclPath[], lineGroup: Int32Array, triGroup: Int32Array): { lineMap: Int32Array; triMap: Int32Array } {
  const set = new Set(hidden);
  const rowHidden = legend.map((e) => e.decl !== null && set.has(e.decl));
  const keep = (group: Int32Array): Int32Array => {
    if (!rowHidden.some(Boolean)) return Int32Array.from(group.keys());
    const out: number[] = [];
    group.forEach((g, i) => {
      if (!rowHidden[g]) out.push(i);
    });
    return Int32Array.from(out);
  };
  return { lineMap: keep(lineGroup), triMap: keep(triGroup) };
}

/** Whether two draws show the same primitives (the contents, not the counts: two groups of equal size differ). */
export function sameDrawn(a: { lineMap: Int32Array; triMap: Int32Array }, b: { lineMap: Int32Array; triMap: Int32Array }): boolean {
  const same = (x: Int32Array, y: Int32Array) => x.length === y.length && x.every((v, i) => v === y[i]);
  return same(a.lineMap, b.lineMap) && same(a.triMap, b.triMap);
}

/**
 * Which outline edges to keep when faces are hidden: an outline edge is a
 * side of some polygon, and a polygon's sides are sides of its fan
 * triangles, so an edge stays iff it is a side of a visible triangle.
 * Vertices are matched by their float32 coordinates, which both buffers
 * share.
 */
export function edgeMask(triPositions: Float32Array, edgePositions: Float32Array, triMap: Int32Array): Uint8Array {
  const vid = new Map<string, number>();
  const id = (a: Float32Array, o: number) => {
    const k = `${a[o]},${a[o + 1]},${a[o + 2]}`;
    let v = vid.get(k);
    if (v === undefined) vid.set(k, (v = vid.size));
    return v;
  };
  const sides = new Set<number>();
  const key = (a: number, b: number) => (a < b ? a * 4294967296 + b : b * 4294967296 + a);
  for (const t of triMap) {
    const v0 = id(triPositions, 9 * t), v1 = id(triPositions, 9 * t + 3), v2 = id(triPositions, 9 * t + 6);
    sides.add(key(v0, v1)).add(key(v1, v2)).add(key(v2, v0));
  }
  const n = edgePositions.length / 6;
  const mask = new Uint8Array(n);
  for (let e = 0; e < n; e++) mask[e] = sides.has(key(id(edgePositions, 6 * e), id(edgePositions, 6 * e + 3))) ? 1 : 0;
  return mask;
}
