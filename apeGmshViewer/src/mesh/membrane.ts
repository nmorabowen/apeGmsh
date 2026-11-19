// A `LadrunoMembrane` is one polygon macro-element. The bridge mints it with
// `fem_eid = -1` and no `inline_connectivity`; its nodes are in the archived
// command, `-nodes n1 n2 ... [-hole h1 h2 ...]...` (h5-schema.md "Bridge-minted
// rows"; `LadrunoMembrane._emit`). Pure: no three.js and no DOM.

import type { ElementBlock, ElementMeta, Group, ModelFile, Param } from "../model/types.ts";
import { triangulateSectionProfile } from "./polygon.ts";

export const MEMBRANE_TYPE = "LadrunoMembrane";

const NONE: readonly number[] = Object.freeze([]);

export interface MembraneLoops {
  /** the boundary node tags, in polygon order */
  outer: number[];
  /** one node-tag loop per `-hole` */
  holes: number[][];
}

/** The tags after flag `i`, up to the next string token. */
function tagsAfter(args: readonly Param[], i: number): number[] | null {
  const tags: number[] = [];
  for (let k = i + 1; k < args.length && typeof args[k] !== "string"; k++) {
    const v = args[k] as number;
    if (!Number.isInteger(v)) return null;
    tags.push(v);
  }
  return tags;
}

/** The polygon of a membrane row; null when its args hold no usable `-nodes` loop. */
export function membraneLoops(args: readonly Param[]): MembraneLoops | null {
  let outer: number[] | null = null;
  const holes: number[][] = [];
  for (let i = 0; i < args.length; i++) {
    if (args[i] !== "-nodes" && args[i] !== "-hole") continue;
    const tags = tagsAfter(args, i);
    if (!tags || tags.length < 3) return null;
    if (args[i] === "-nodes") outer = tags;
    else holes.push(tags);
  }
  return outer ? { outer, holes } : null;
}

/** The nodes an OpenSees-only row is drawn from: its inline connectivity, or a membrane's boundary. */
export function opsOnlyNodes(meta: ElementMeta, row: number): readonly number[] {
  const inline = meta.inlineConnectivity?.[row];
  if (inline?.length) return inline;
  if (meta.type !== MEMBRANE_TYPE) return NONE;
  return membraneLoops(meta.args[row] ?? [])?.outer ?? NONE;
}

/**
 * Triangulate a polygon with holes, in the plane of its outer loop (best-fit
 * by Newell's method, so a slightly warped panel still projects cleanly).
 * Each loop lists indices into `xyz` (3 per point). Triangles wind as the
 * outer loop does. Empty when the outer loop encloses no area.
 */
export function triangulateLoops(xyz: ArrayLike<number>, outer: readonly number[], holes: readonly (readonly number[])[]): [number, number, number][] {
  const p = (i: number, k: number): number => xyz[3 * i + k]!;
  const n = [0, 0, 0];
  outer.forEach((a, s) => {
    const b = outer[(s + 1) % outer.length]!;
    for (let k = 0; k < 3; k++) {
      const u = (k + 1) % 3, v = (k + 2) % 3;
      n[k]! += (p(a, u) - p(b, u)) * (p(a, v) + p(b, v));
    }
  });
  const len = Math.hypot(n[0]!, n[1]!, n[2]!);
  if (!(len > 0)) return [];
  const w = n.map((c) => c / len) as [number, number, number];
  const r = Math.abs(w[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const cross = (a: readonly number[], b: readonly number[]): number[] => [
    a[1]! * b[2]! - a[2]! * b[1]!,
    a[2]! * b[0]! - a[0]! * b[2]!,
    a[0]! * b[1]! - a[1]! * b[0]!,
  ];
  const e1 = cross(r, w);
  const l1 = Math.hypot(e1[0]!, e1[1]!, e1[2]!);
  const u = e1.map((c) => c / l1);
  const v = cross(w, u);
  // Scale by the panel size: the triangulator's tolerances are absolute.
  const size = Math.sqrt(len / 2);
  const o = outer[0]!;
  const plane = (i: number): [number, number] => {
    const d = [p(i, 0) - p(o, 0), p(i, 1) - p(o, 1), p(i, 2) - p(o, 2)];
    return [
      (d[0]! * u[0]! + d[1]! * u[1]! + d[2]! * u[2]!) / size,
      (d[0]! * v[0]! + d[1]! * v[1]! + d[2]! * v[2]!) / size,
    ];
  };
  const area = (ring: readonly (readonly [number, number])[]): number =>
    ring.reduce((a, [y, z], i) => {
      const [y1, z1] = ring[(i + 1) % ring.length]!;
      return a + y * z1 - y1 * z;
    }, 0);
  // Outer counter-clockwise about its own normal; each hole clockwise.
  const orderedHoles = holes.map((h) => (area(h.map(plane)) > 0 ? [...h].reverse() : [...h]));
  const order = [...outer, ...orderedHoles.flat()];
  const loops = [outer, ...orderedHoles].map((l) => l.map(plane));
  return triangulateSectionProfile({ loops, source: "computed" }).triangles.map(
    ([a, b, c]) => [order[a]!, order[b]!, order[c]!] as [number, number, number],
  );
}

/**
 * The node sets of the surface groups a `LadrunoMembrane(pg=...)` was declared
 * on. Those panels' mesh cells stay in `/elements` and are drawn already, so a
 * membrane whose boundary lies in one of these sets needs no fill of its own.
 */
export function meshedPanelNodeSets(model: ModelFile): Set<number>[] {
  const names = new Set<string>();
  for (const rec of model.opensees?.definitions.decls?.records ?? []) {
    for (const p of rec.params) {
      if (p.type === MEMBRANE_TYPE && typeof p.params["pg"] === "string") names.add(p.params["pg"]);
    }
  }
  if (!names.size) return [];
  const surfaces = model.blocks.filter((b) => b.dim === 2);
  const groups: Group[] = [...model.physicalGroups, ...model.labels].filter((g) => g.dim === 2 && names.has(g.name));
  return groups.map((g) => cellNodes(g.elementIds, surfaces));
}

function cellNodes(ids: Float64Array, blocks: readonly ElementBlock[]): Set<number> {
  const wanted = new Set(ids);
  const nodes = new Set<number>();
  for (const b of blocks) {
    b.ids.forEach((id, row) => {
      if (!wanted.has(id)) return;
      for (let k = 0; k < b.npe; k++) nodes.add(b.connectivity[row * b.npe + k]!);
    });
  }
  return nodes;
}
