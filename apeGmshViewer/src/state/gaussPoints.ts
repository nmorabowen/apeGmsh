// Where the Gauss points of a results stage sit: each point is the element's
// corners weighted by the corner shape functions at the point's natural
// coordinates (`_natural_coords`, or a beam station's xi). The weights are kept,
// so a deformed step moves each point by the same weights of its corners'
// displacements. Pure, so the placement is tested without a window.

import type { BlobStore } from "./blobs.ts";
import { columnOf } from "./contour.ts";
import { shapeOf } from "./extrapolation.ts";
import { joinLadrunoElements } from "./lineDiagram.ts";
import { hiddenElementsOf, inactiveElements, resultsPairing } from "./selectors.ts";
import type { BlockInfo, Decl, DeclPath, State } from "./types.ts";

/** Every placed point: its corner node ids and their weights, `offsets[i]..offsets[i+1]` for point `i`. */
export interface GaussLayout {
  count: number;
  offsets: Int32Array;
  nodes: Float64Array;
  weights: Float64Array;
  /** why points were not placed, with how many elements each reason cost */
  refused: { reason: string; elements: number }[];
}

type Cell = { code: number; nodes: ArrayLike<number> };

/** An element's cell by its declaration path: its gmsh code and its connectivity row, corners first. */
export function cellOfDecl(decls: Readonly<Record<DeclPath, Decl>>, blocks: readonly BlockInfo[], blobs: BlobStore): (path: DeclPath) => Cell | null {
  const conn = new Map<number, Float64Array>();
  return (path) => {
    const cell = decls[path]?.element?.cell;
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

/** Collects points one element at a time; a point that cannot be placed is counted against its reason, never guessed. */
export class GaussLayoutBuilder {
  private readonly offsets: number[] = [0];
  private readonly nodes: number[] = [];
  private readonly weights: number[] = [];
  private readonly refusals = new Map<string, number>();
  private readonly cellOf: (path: DeclPath) => Cell | null;
  private readonly shown: (path: DeclPath) => boolean;

  constructor(cellOf: (path: DeclPath) => Cell | null, shown: (path: DeclPath) => boolean) {
    this.cellOf = cellOf;
    this.shown = shown;
  }

  refuse(reason: string, elements: number): void {
    if (elements > 0) this.refusals.set(reason, (this.refusals.get(reason) ?? 0) + elements);
  }

  /** The element's points at the natural coordinates `rows`. */
  add(path: DeclPath, rows: readonly (readonly number[])[]): void {
    if (!this.shown(path)) return;
    const cell = this.cellOf(path);
    if (!cell) return this.refuse("not an element of the model's mesh", 1);
    const shape = shapeOf(cell.code);
    if (!shape) return this.refuse(`no shape functions for gmsh type ${cell.code}`, 1);
    if (rows.some((r) => r.length < shape.pdim)) return this.refuse(`natural coordinates with fewer than ${shape.pdim} components`, 1);
    for (const r of rows) {
      const N = shape.N(r);
      for (let k = 0; k < shape.nc; k++) {
        this.nodes.push(cell.nodes[k]!);
        this.weights.push(N[k]!);
      }
      this.offsets.push(this.nodes.length);
    }
  }

  build(): GaussLayout {
    return {
      count: this.offsets.length - 1,
      offsets: Int32Array.from(this.offsets),
      nodes: Float64Array.from(this.nodes),
      weights: Float64Array.from(this.weights),
      refused: [...this.refusals].map(([reason, elements]) => ({ reason, elements })),
    };
  }
}

/** One step's beam stations, as the results reader gives them: element id and xi in [-1, 1] per station. */
export interface StationPoints {
  elementIndex: Float64Array;
  stationNaturalCoord: Float64Array;
}

/**
 * The points of the stage shown: every Gauss group the file places (a verified
 * file with `_natural_coords`), and the beam stations when `stations` is given.
 * Hidden and inactive elements are left out.
 */
export function gaussLayoutOf(s: State, blobs: BlobStore, stations: StationPoints | null): GaussLayout | null {
  const at = s.phase.at;
  const stage = at?.kind === "results" ? s.results?.stages[at.stage] : undefined;
  if (!stage || !s.mesh) return null;
  const hidden = hiddenElementsOf(s);
  const inactive = inactiveElements(s);
  let inactiveSet: Set<DeclPath> | null = null;
  if (inactive && inactive.hidden > 0) {
    inactiveSet = new Set();
    s.mesh.elements.forEach((p, i) => {
      if (inactive.mask[i]) inactiveSet!.add(p);
    });
  }
  const shown = (p: DeclPath) => !hidden.has(p) && !inactiveSet?.has(p);
  const b = new GaussLayoutBuilder(cellOfDecl(s.decls, s.blocks, blobs), shown);
  for (const g of stage.gauss) {
    if (!g.elementIds) b.refuse("element ids not verified as the model's (the file's index space is not stamped)", g.nElements);
    else if (!g.naturalCoords) b.refuse("no Gauss point coordinates in the file", g.nElements);
    else {
      const eids = blobs.get(g.elementIds);
      if (!(eids instanceof Float64Array)) throw new TypeError(`gauss points: ${g.elementIds.id} is not f64`);
      for (const id of eids) b.add(`mesh/element/${id}`, g.naturalCoords);
    }
  }
  if (stations && s.results) {
    const join = joinLadrunoElements(s.decls, resultsPairing(s), s.results.elementIndexSpace);
    if (!join.ok) b.refuse(`beam stations: ${join.reason}`, new Set(stations.elementIndex).size);
    else {
      const byElement = new Map<number, number[][]>();
      stations.elementIndex.forEach((id, i) => {
        const xi = stations.stationNaturalCoord[i]!;
        if (!Number.isFinite(xi)) return;
        let rows = byElement.get(id);
        if (!rows) byElement.set(id, (rows = []));
        rows.push([xi]);
      });
      for (const [id, rows] of byElement) {
        const path = join.byLadrunoId.get(id);
        if (path) b.add(path, rows);
        else b.refuse("beam stations of an element the model does not have", 1);
      }
    }
  }
  return b.build();
}

/** Each point's position from the node coordinates (`ids`, and `xyz` 3 per node). */
export function gaussBasePositions(layout: GaussLayout, ids: Float64Array, xyz: Float64Array, column: ReadonlyMap<number, number> = columnOf(ids)): Float32Array {
  const out = new Float32Array(layout.count * 3);
  for (let i = 0; i < layout.count; i++) {
    let x = 0, y = 0, z = 0;
    for (let k = layout.offsets[i]!; k < layout.offsets[i + 1]!; k++) {
      const j = column.get(layout.nodes[k]!);
      if (j === undefined) {
        x = y = z = NaN;
        break;
      }
      const w = layout.weights[k]!;
      x += w * xyz[3 * j]!;
      y += w * xyz[3 * j + 1]!;
      z += w * xyz[3 * j + 2]!;
    }
    out[3 * i] = x;
    out[3 * i + 1] = y;
    out[3 * i + 2] = z;
  }
  return out;
}

/**
 * Each point's displacement: the same weights over its corners' displacements.
 * A point with a corner that has none is left in place (0), as the mesh leaves
 * such a vertex (never moved by a displacement of 0 it does not have).
 */
export function gaussDisplacements(
  layout: GaussLayout,
  column: ReadonlyMap<number, number>,
  rows: readonly [Float64Array, Float64Array, Float64Array | null],
): Float64Array {
  const d = new Float64Array(layout.count * 3);
  const [rx, ry, rz] = rows;
  for (let i = 0; i < layout.count; i++) {
    let x = 0, y = 0, z = 0;
    let ok = true;
    for (let k = layout.offsets[i]!; k < layout.offsets[i + 1]!; k++) {
      const j = column.get(layout.nodes[k]!);
      const ux = j === undefined ? NaN : rx[j]!, uy = j === undefined ? NaN : ry[j]!, uz = j === undefined ? NaN : rz ? rz[j]! : 0;
      if (!Number.isFinite(ux) || !Number.isFinite(uy) || !Number.isFinite(uz)) {
        ok = false;
        break;
      }
      const w = layout.weights[k]!;
      x += w * ux;
      y += w * uy;
      z += w * uz;
    }
    if (!ok) continue;
    d[3 * i] = x;
    d[3 * i + 1] = y;
    d[3 * i + 2] = z;
  }
  return d;
}
