// Where the Gauss points of a results stage sit: each point is the element's
// corners weighted by the corner shape functions at the point's natural
// coordinates (`_natural_coords`, or a beam station's xi). The weights are kept,
// so a deformed step moves each point by the same weights of its corners'
// displacements. A beam station also carries a fixed shift to the clear span
// between rigid joint offsets, as the line diagram draws it. Pure, so the
// placement is tested without a window.

import type { BlobStore } from "./blobs.ts";
import { columnOf } from "./contour.ts";
import { shapeOf } from "./extrapolation.ts";
import { collectTransformJntOffsets, joinLadrunoElements, resolveElementJointOffset } from "./lineDiagram.ts";
import { activationNow, hiddenElementsOf, resultsPairing } from "./selectors.ts";
import type { BlockInfo, Decl, DeclPath, State } from "./types.ts";

/** Every placed point: its corner node ids and their weights, `offsets[i]..offsets[i+1]` for point `i`. */
export interface GaussLayout {
  count: number;
  offsets: Int32Array;
  nodes: Float64Array;
  weights: Float64Array;
  /** a fixed shift per point (3 each), 0 except at beam stations with rigid joint offsets */
  shift: Float64Array;
  /** why points were not placed, with how many elements each reason cost */
  refused: { reason: string; elements: number }[];
  /** what could not be counted in elements (a step whose beam stations were not read) */
  notes: string[];
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

type Vec3 = readonly [number, number, number];

/** Collects points one element at a time; a point that cannot be placed is counted against its reason, never guessed. */
export class GaussLayoutBuilder {
  private readonly offsets: number[] = [0];
  private readonly nodes: number[] = [];
  private readonly weights: number[] = [];
  private readonly shifts: number[] = [];
  private readonly refusals = new Map<string, number>();
  private readonly notes: string[] = [];
  readonly cellOf: (path: DeclPath) => Cell | null;
  private readonly shown: (path: DeclPath) => boolean;

  constructor(cellOf: (path: DeclPath) => Cell | null, shown: (path: DeclPath) => boolean) {
    this.cellOf = cellOf;
    this.shown = shown;
  }

  refuse(reason: string, elements: number): void {
    if (elements > 0) this.refusals.set(reason, (this.refusals.get(reason) ?? 0) + elements);
  }

  note(text: string): void {
    this.notes.push(text);
  }

  /**
   * The element's points at the natural coordinates `rows`. `ends` shifts a
   * 2-node element's points to its clear span: point `i` moves by
   * `N0 * dI + N1 * dJ`, so xi = -1 sits at node I + dI and xi = 1 at node J + dJ.
   */
  add(path: DeclPath, rows: readonly (readonly number[])[], ends: { dI: Vec3; dJ: Vec3 } | null = null): void {
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
      if (ends && shape.nc === 2) {
        this.shifts.push(N[0]! * ends.dI[0] + N[1]! * ends.dJ[0], N[0]! * ends.dI[1] + N[1]! * ends.dJ[1], N[0]! * ends.dI[2] + N[1]! * ends.dJ[2]);
      } else this.shifts.push(0, 0, 0);
    }
  }

  build(): GaussLayout {
    return {
      count: this.offsets.length - 1,
      offsets: Int32Array.from(this.offsets),
      nodes: Float64Array.from(this.nodes),
      weights: Float64Array.from(this.weights),
      shift: Float64Array.from(this.shifts),
      refused: [...this.refusals].map(([reason, elements]) => ({ reason, elements })),
      notes: [...this.notes],
    };
  }
}

/**
 * One stage's beam stations, as the results reader gives them: element id and
 * xi in [-1, 1] per station, and each station's element end nodes `[i, j]`
 * (2 per station) when the file records them. `error` when the step's stations
 * could not be read.
 */
export interface StationPoints {
  elementIndex: Float64Array;
  stationNaturalCoord: Float64Array;
  elementNodes?: Float64Array | null;
  error?: string | null;
}

/**
 * A beam's station rows in the cell's corner order: xi is measured from the
 * element's node I, so a cell stored J to I takes -xi (the line diagram's P0-2
 * rule). Null when the element's nodes contradict the cell's (neither order).
 */
export function orientStations(cellNodes: ArrayLike<number>, elementNodes: readonly [number, number] | null, xis: readonly number[]): number[][] | null {
  if (!elementNodes) return xis.map((x) => [x]);
  const [i, j] = elementNodes;
  const a = cellNodes[0], b = cellNodes[1];
  if (i === j) return null;
  if (i === a && j === b) return xis.map((x) => [x]);
  if (i === b && j === a) return xis.map((x) => [-x]);
  return null;
}

/**
 * The points of the stage shown: every Gauss group the file places (a verified
 * file with `_natural_coords`), and the beam stations when `stations` is given.
 * Hidden elements and those the stage does not have are left out, interior
 * solid elements included.
 */
export function gaussLayoutOf(s: State, blobs: BlobStore, stations: StationPoints | null): GaussLayout | null {
  const at = s.phase.at;
  const stage = at?.kind === "results" ? s.results?.stages[at.stage] : undefined;
  if (!stage || !s.mesh) return null;
  const hidden = hiddenElementsOf(s);
  // By the activation's own tags, so an interior element (no drawn face, absent from `mesh.elements`) is judged too.
  const a = activationNow(s);
  const inactiveTags = a && a.activation.state === "filtered" ? a.activation.inactiveTags : null;
  const inactive = (p: DeclPath) => (inactiveTags ? (s.decls[p]?.element?.metas ?? []).some((m) => inactiveTags.has(m.tag)) : false);
  const shown = (p: DeclPath) => !hidden.has(p) && !inactive(p);
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
  if (stations?.error) b.note(`Beam stations not read: ${stations.error}`);
  else if (stations && s.results) addStations(s, blobs, b, stations);
  return b.build();
}

function addStations(s: State, blobs: BlobStore, b: GaussLayoutBuilder, stations: StationPoints): void {
  const join = joinLadrunoElements(s.decls, resultsPairing(s), s.results!.elementIndexSpace);
  if (!join.ok) return b.refuse(`beam stations: ${join.reason}`, new Set(stations.elementIndex).size);
  const byElement = new Map<number, { xis: number[]; nodes: [number, number] | null }>();
  stations.elementIndex.forEach((id, i) => {
    const xi = stations.stationNaturalCoord[i]!;
    let e = byElement.get(id);
    if (!e) {
      const en = stations.elementNodes;
      const nodes: [number, number] | null = en && Number.isFinite(en[2 * i]!) && Number.isFinite(en[2 * i + 1]!) ? [en[2 * i]!, en[2 * i + 1]!] : null;
      byElement.set(id, (e = { xis: [], nodes }));
    }
    e.xis.push(xi);
  });
  const nodes = s.mesh!.nodes;
  const column = nodes ? columnOf(blobs.get(nodes.ids) as Float64Array) : null;
  const xyz = nodes ? (blobs.get(nodes.xyz) as Float64Array) : null;
  const at = (id: number): Vec3 | null => {
    const j = column?.get(id);
    return j === undefined || !xyz ? null : [xyz[3 * j]!, xyz[3 * j + 1]!, xyz[3 * j + 2]!];
  };
  const jntByTransfTag = collectTransformJntOffsets(s.decls);
  for (const [id, e] of byElement) {
    const path = join.byLadrunoId.get(id);
    if (!path) {
      b.refuse("beam stations of an element the model does not have", 1);
      continue;
    }
    const cell = b.cellOf(path);
    if (!cell) {
      b.refuse("not an element of the model's mesh", 1);
      continue;
    }
    // The line diagram refuses the same: a station off the element would be drawn past its end.
    if (e.xis.some((x) => !Number.isFinite(x) || x < -1 - 1e-9 || x > 1 + 1e-9)) {
      b.refuse("beam stations with a natural coordinate that is not finite or is outside [-1, 1]", 1);
      continue;
    }
    // The element's own I and J: the step's, else the model's connectivity, as the line diagram takes them.
    const inline = s.decls[path]?.element?.inlineNodes;
    const ends = e.nodes ?? (inline && inline.length >= 2 ? ([inline[0]!, inline[inline.length - 1]!] as [number, number]) : null);
    const rows = orientStations(cell.nodes, ends, e.xis);
    if (!rows) {
      b.refuse("beam stations whose element nodes contradict the model's connectivity", 1);
      continue;
    }
    // The clear span between rigid joint offsets. The offsets are the element's, from I to J; the rows follow
    // the cell's corners, so on a cell stored J to I the offset of I goes to its second corner.
    const reversed = ends !== null && ends[0] === cell.nodes[1];
    const meta0 = s.decls[path]?.element?.metas[0];
    const pI = at(cell.nodes[reversed ? 1 : 0]!), pJ = at(cell.nodes[reversed ? 0 : 1]!);
    const jnt = meta0 && pI && pJ ? resolveElementJointOffset(meta0.type, meta0.args, pI, pJ, jntByTransfTag) : null;
    b.add(path, rows, jnt ? (reversed ? { dI: jnt.dJ, dJ: jnt.dI } : { dI: jnt.dI, dJ: jnt.dJ }) : null);
  }
}

/** Each point's position from the node coordinates (`ids`, and `xyz` 3 per node), plus its fixed shift. NaN when a corner has none. */
export function gaussBasePositions(layout: GaussLayout, ids: Float64Array, xyz: Float64Array, column: ReadonlyMap<number, number> = columnOf(ids)): Float32Array {
  const out = new Float32Array(layout.count * 3);
  for (let i = 0; i < layout.count; i++) {
    let x = layout.shift[3 * i]!, y = layout.shift[3 * i + 1]!, z = layout.shift[3 * i + 2]!;
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

/** The layout and positions without the points a corner node could not place (NaN), counted as refused points. */
export function dropUnplaced(layout: GaussLayout, base: Float32Array): { layout: GaussLayout; base: Float32Array } {
  const keep: number[] = [];
  for (let i = 0; i < layout.count; i++) if (Number.isFinite(base[3 * i]!) && Number.isFinite(base[3 * i + 1]!) && Number.isFinite(base[3 * i + 2]!)) keep.push(i);
  if (keep.length === layout.count) return { layout, base };
  const offsets = [0], nodes: number[] = [], weights: number[] = [], shift: number[] = [], pos: number[] = [];
  for (const i of keep) {
    for (let k = layout.offsets[i]!; k < layout.offsets[i + 1]!; k++) {
      nodes.push(layout.nodes[k]!);
      weights.push(layout.weights[k]!);
    }
    offsets.push(nodes.length);
    shift.push(layout.shift[3 * i]!, layout.shift[3 * i + 1]!, layout.shift[3 * i + 2]!);
    pos.push(base[3 * i]!, base[3 * i + 1]!, base[3 * i + 2]!);
  }
  const dropped = layout.count - keep.length;
  return {
    layout: {
      count: keep.length,
      offsets: Int32Array.from(offsets),
      nodes: Float64Array.from(nodes),
      weights: Float64Array.from(weights),
      shift: Float64Array.from(shift),
      refused: layout.refused,
      notes: [...layout.notes, `${dropped.toLocaleString()} ${dropped === 1 ? "point" : "points"} not placed: a corner node is not in the model's node table.`],
    },
    base: Float32Array.from(pos),
  };
}

/**
 * Each point's displacement: the same weights over its corners' displacements.
 * A point with a corner that has none is left in place (0), as the mesh leaves
 * such a vertex (never moved by a displacement of 0 it does not have). The
 * joint-offset shift is rigid, so it does not change.
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
