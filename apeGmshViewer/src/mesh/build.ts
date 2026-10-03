// Turn a ModelFile into flat render buffers, coloured by physical group.
//
// Pure: no three.js and no DOM, so the tests run it in Node. Cells are
// recognised by their gmsh element-type `code` (gmsh's own numbering,
// stored on every /elements block); only corner nodes are drawn. Solids
// draw their boundary faces only. An unknown code is a loud warning.

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

/** Categorical palette (sRGB 0..1); groups cycle through it. */
export const PALETTE: readonly (readonly [number, number, number])[] = [
  [0.29, 0.56, 0.89], [0.95, 0.55, 0.22], [0.36, 0.73, 0.42], [0.86, 0.33, 0.38],
  [0.62, 0.47, 0.84], [0.25, 0.75, 0.78], [0.89, 0.75, 0.25], [0.84, 0.45, 0.70],
  [0.55, 0.62, 0.30], [0.50, 0.58, 0.68],
];
export const NO_GROUP: readonly [number, number, number] = [0.62, 0.64, 0.68];
export const OPS_ONLY: readonly [number, number, number] = [0.93, 0.93, 0.93];

export interface LegendEntry {
  name: string;
  color: readonly [number, number, number];
  elements: number;
}

export interface MeshBuffers {
  /** 6 floats per segment (two xyz endpoints). */
  linePositions: Float32Array;
  lineColors: Float32Array;
  lineRefs: ElementRef[];
  /** 9 floats per triangle. */
  triPositions: Float32Array;
  triColors: Float32Array;
  triRefs: ElementRef[];
  /** 6 floats per outline edge of the drawn faces. */
  edgePositions: Float32Array;
  legend: LegendEntry[];
  center: [number, number, number];
  radius: number;
  /** cells drawn, by kind */
  counts: { lineCells: number; faceCells: number; solidCells: number; opsOnly: number; points: number };
  warnings: string[];
}

/**
 * The colour group of each FEM element: the smallest element-side physical
 * group that contains it (the most specific), ties broken by name.
 */
export function colourGroups(model: ModelFile): { byElement: Map<number, number>; order: string[] } {
  const groups = model.physicalGroups
    .filter((g) => g.elementIds.length > 0)
    .slice()
    .sort((a, b) => a.elementIds.length - b.elementIds.length || a.name.localeCompare(b.name));
  const byElement = new Map<number, number>();
  groups.forEach((g, gi) => {
    for (const e of g.elementIds) if (!byElement.has(e)) byElement.set(e, gi);
  });
  return { byElement, order: groups.map((g) => g.name) };
}

export function buildMesh(model: ModelFile): MeshBuffers {
  const warnings: string[] = [];
  const nodeIndex = new Map<number, number>();
  model.nodeIds.forEach((id, i) => nodeIndex.set(id, i));
  const xyz = model.nodeCoords;

  const { byElement, order } = colourGroups(model);
  const legendCount = new Array<number>(order.length).fill(0);
  let noGroup = 0;
  const colourOf = (femId: number): readonly [number, number, number] => {
    const gi = byElement.get(femId);
    if (gi === undefined) {
      noGroup++;
      return NO_GROUP;
    }
    legendCount[gi]!++;
    return PALETTE[gi % PALETTE.length]!;
  };

  const lp: number[] = [], lc: number[] = [], lr: ElementRef[] = [];
  const tp: number[] = [], tc: number[] = [], tr: ElementRef[] = [];
  const counts = { lineCells: 0, faceCells: 0, solidCells: 0, opsOnly: 0, points: 0 };
  const missingNodes = new Set<number>();

  const at = (nodeId: number): number => {
    const i = nodeIndex.get(nodeId);
    if (i === undefined) {
      missingNodes.add(nodeId);
      return -1;
    }
    return i;
  };
  const push3 = (arr: number[], i: number) => arr.push(xyz[3 * i]!, xyz[3 * i + 1]!, xyz[3 * i + 2]!);
  const pushSeg = (a: number, b: number, c: readonly number[], ref: ElementRef) => {
    push3(lp, a); push3(lp, b);
    lc.push(c[0]!, c[1]!, c[2]!, c[0]!, c[1]!, c[2]!);
    lr.push(ref);
  };
  const polygons: number[][] = [];
  const pushFace = (ids: number[], c: readonly number[], ref: ElementRef) => {
    polygons.push(ids);
    for (let k = 1; k + 1 < ids.length; k++) {
      push3(tp, ids[0]!); push3(tp, ids[k]!); push3(tp, ids[k + 1]!);
      for (let m = 0; m < 3; m++) tc.push(c[0]!, c[1]!, c[2]!);
      tr.push(ref);
    }
  };

  // Solid boundary faces: a face seen once is on the boundary.
  const solidFaces = new Map<string, { ids: number[]; colour: readonly number[]; ref: ElementRef; n: number }>();

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
      const colour = colourOf(b.ids[row]!);
      if (shape === "line") {
        counts.lineCells++;
        pushSeg(corner[0]!, corner[1]!, colour, ref);
      } else if (shape === "tri" || shape === "quad") {
        counts.faceCells++;
        pushFace(corner, colour, ref);
      } else {
        counts.solidCells++;
        for (const f of SOLID_FACES[shape]) {
          const ids = f.map((k) => corner[k]!);
          const key = ids.slice().sort((x, y) => x - y).join(",");
          const seen = solidFaces.get(key);
          if (seen) seen.n++;
          else solidFaces.set(key, { ids, colour, ref, n: 1 });
        }
      }
    }
  });
  for (const f of solidFaces.values()) if (f.n === 1) pushFace(f.ids, f.colour, f.ref);

  // OpenSees-only elements (no neutral-zone cell): drawn from inline connectivity.
  model.opensees?.elementMeta.forEach((meta, metaIndex) => {
    if (!meta.inlineConnectivity) return;
    meta.inlineConnectivity.forEach((conn, row) => {
      if (meta.femEids[row]! >= 0 || conn.length < 2) return;
      const idx = conn.map(at);
      if (idx.some((i) => i < 0)) return;
      counts.opsOnly++;
      for (let k = 0; k + 1 < idx.length; k++) pushSeg(idx[k]!, idx[k + 1]!, OPS_ONLY, { kind: "ops", metaIndex, row });
    });
  });

  if (missingNodes.size) {
    warnings.push(`${missingNodes.size} node tags referenced by cells are not in /nodes/ids; those cells are not drawn`);
  }

  // Outline: the perimeter edges of every drawn polygon, each edge once
  // (so a split quad shows no diagonal).
  const seenEdge = new Set<number>();
  const ep: number[] = [];
  const nNodes = model.nodeIds.length;
  for (const poly of polygons) {
    for (let k = 0; k < poly.length; k++) {
      const a = poly[k]!, b = poly[(k + 1) % poly.length]!;
      const key = a < b ? a * nNodes + b : b * nNodes + a;
      if (seenEdge.has(key)) continue;
      seenEdge.add(key);
      push3(ep, a); push3(ep, b);
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
    .map((name, gi) => ({ name, color: PALETTE[gi % PALETTE.length]!, elements: legendCount[gi]! }))
    .filter((e) => e.elements > 0)
    .sort((a, b) => b.elements - a.elements || a.name.localeCompare(b.name));
  if (noGroup) legend.push({ name: "(no physical group)", color: NO_GROUP, elements: noGroup });
  if (counts.opsOnly) legend.push({ name: "(OpenSees-only, no cell)", color: OPS_ONLY, elements: counts.opsOnly });

  return {
    linePositions: new Float32Array(lp),
    lineColors: new Float32Array(lc),
    lineRefs: lr,
    triPositions: new Float32Array(tp),
    triColors: new Float32Array(tc),
    triRefs: tr,
    edgePositions: new Float32Array(ep),
    legend,
    center,
    radius,
    counts,
    warnings,
  };
}
