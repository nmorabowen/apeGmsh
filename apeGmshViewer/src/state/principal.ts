// The Principal card's arrows (renderer/loads.ts `PrincipalLayer`): at each
// Gauss point of a stress or strain field (state/tensor.ts), an arrow along
// each principal direction, the length from |p| and the colour from the signed
// p on a diverging map, so tension and compression read apart. The viewer twin
// of Qt's PrincipalDirectionDiagram (src/apeGmsh/viewers/diagrams/_principal_glyph.py).
// Pure: tested without a window.

import type { BlobStore } from "./blobs.ts";
import { rampT, sampleCmap } from "./contour.ts";
import { cellOfDecl, GaussLayoutBuilder, shownElementOf, type GaussLayout } from "./gaussPoints.ts";
import { ANCHOR, ARROW, ARROW_STRIDE, BAR, thinningLevels } from "./loads.ts";
import { assembleTensor, eigenSym3, type TensorField, type Vec3 } from "./tensor.ts";
import type { ContourCmap, PrincipalHead, State } from "./types.ts";

// ---- the points and their tensors ------------------------------------------------

/** The field's points on screen: their placement, and for each its group (in `field.groups`) and value index (element-major, `e * G + k`). */
export interface PrincipalPoints {
  layout: GaussLayout;
  group: Int32Array;
  row: Int32Array;
}

/**
 * Where the field's Gauss points sit (state/gaussPoints.ts, as the Gauss
 * markers place them), hidden and inactive elements left out, and which value
 * each point reads. A group the file does not place is refused by name.
 */
export function principalPointsOf(s: State, blobs: BlobStore, field: TensorField): PrincipalPoints | null {
  const at = s.phase.at;
  const stage = at?.kind === "results" ? s.results?.stages[at.stage] : undefined;
  if (!stage || !s.mesh) return null;
  const b = new GaussLayoutBuilder(cellOfDecl(s.decls, s.blocks, blobs), shownElementOf(s));
  const group: number[] = [], row: number[] = [];
  field.groups.forEach((fg, gi) => {
    const g = stage.gauss.find((x) => x.id === fg.id);
    if (!g) return;
    if (!g.elementIds) return b.refuse("element ids not verified as the model's (the file's index space is not stamped)", g.nElements);
    if (!g.naturalCoords) return b.refuse("no Gauss point coordinates in the file", g.nElements);
    const G = g.nGauss;
    if (g.naturalCoords.length !== G) return b.refuse(`${g.naturalCoords.length} Gauss point coordinates for ${G} points`, g.nElements);
    const eids = blobs.get(g.elementIds);
    if (!(eids instanceof Float64Array)) throw new TypeError(`principal: ${g.elementIds.id} is not f64`);
    eids.forEach((id, e) => {
      const n0 = b.count;
      b.add(`mesh/element/${id}`, g.naturalCoords!);
      for (let k = 0; k < b.count - n0; k++) {
        group.push(gi);
        row.push(e * G + k);
      }
    });
  });
  return { layout: b.build(), group: Int32Array.from(group), row: Int32Array.from(row) };
}

/** Each point's Voigt tensor (6 per point) from the step's components by group, NaN for a point whose values are missing or not finite. */
export function principalTensors(points: PrincipalPoints, field: TensorField, values: Readonly<Record<string, Readonly<Record<string, Float64Array>>>>): Float64Array {
  const n = points.layout.count;
  const out = new Float64Array(6 * n).fill(NaN);
  for (let i = 0; i < n; i++) {
    const fg = field.groups[points.group[i]!]!;
    const r = points.row[i]!;
    let missing = false;
    const t = assembleTensor(field.family, (v) => {
      const c = fg.slots[v];
      if (c === null || c === undefined) return null;
      const col = values[c]?.[fg.id];
      if (!col || r >= col.length) {
        missing = true;
        return null;
      }
      return col[r]!;
    });
    if (t && !missing) out.set(t, 6 * i);
  }
  return out;
}

// ---- the arrows ----------------------------------------------------------------------

export interface PrincipalOptions {
  /** p1, p2, p3 */
  show: readonly [boolean, boolean, boolean];
  head: PrincipalHead;
  cmap: ContourCmap;
  /** the length multiplier (> 0) */
  scale: number;
  /** the model's bounding radius: the reference length and the thinning distances */
  radius: number;
}

export interface Principal3D {
  /** `ARROW_STRIDE` records, tail-anchored at the points */
  arrows: Float32Array;
  /** the points drawn (any principal) */
  points: number;
  /** the largest |p| drawn: the colour range is -maxAbs..maxAbs */
  maxAbs: number;
}

/** Below this fraction of the largest |p|, a principal is noise (a plane element's zz), not an arrow: the reactions' floor. */
export const PRINCIPAL_FLOOR = 1e-6;

/**
 * Arrows from each point along the principal directions shown, the length
 * |p| over the largest |p| times the reference length, the colour the signed
 * value on `cmap` over -max..max (a diverging map reads tension against
 * compression). Thinning is per point: its arrows show and hide together.
 */
export function buildPrincipal3D(base: Float32Array, tensors: Float64Array, opts: PrincipalOptions): Principal3D {
  const n = base.length / 3;
  type Item = { point: number; p: number; v: Vec3 };
  const items: Item[] = [];
  let maxAbs = 0;
  for (let i = 0; i < n; i++) {
    const x = base[3 * i]!, y = base[3 * i + 1]!, z = base[3 * i + 2]!;
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z) || !Number.isFinite(tensors[6 * i]!)) continue;
    const e = eigenSym3(tensors.subarray(6 * i, 6 * i + 6));
    for (let k = 0; k < 3; k++) {
      if (!opts.show[k]) continue;
      const p = e.values[k]!;
      items.push({ point: i, p, v: e.vectors[k]! });
      maxAbs = Math.max(maxAbs, Math.abs(p));
    }
  }
  const kept = maxAbs > 0 ? items.filter((it) => Math.abs(it.p) >= PRINCIPAL_FLOOR * maxAbs) : [];
  const radius = opts.radius > 0 && Number.isFinite(opts.radius) ? opts.radius : 1;
  const baseLen = Math.max(radius * 0.1, 1e-4) * opts.scale;
  const per = opts.head === "axis" ? 2 : 1;
  const arrows = new Float32Array(kept.length * per * ARROW_STRIDE);
  let r = 0;
  const push = (it: Item, sign: number) => {
    const o = r++ * ARROW_STRIDE;
    const ratio = Math.abs(it.p) / maxAbs;
    const rgb = sampleCmap(opts.cmap, rampT(it.p, -maxAbs, maxAbs));
    arrows.set([base[3 * it.point]!, base[3 * it.point + 1]!, base[3 * it.point + 2]!], o + ARROW.tip);
    arrows.set([sign * it.v[0], sign * it.v[1], sign * it.v[2]], o + ARROW.dir);
    arrows[o + ARROW.len] = baseLen * ratio;
    arrows.set(rgb, o + ARROW.rgb);
    arrows[o + ARROW.heads] = 1;
    arrows[o + ARROW.bar] = BAR.none;
    arrows[o + ARROW.priority] = ratio;
    arrows[o + ARROW.anchor] = ANCHOR.tail;
  };
  for (const it of kept) {
    push(it, 1);
    if (per === 2) push(it, -1);
  }
  // Per point: its strongest arrow stands for it in the thinning, and every arrow of it takes that level.
  const rep = new Map<number, number>();
  for (let j = 0; j < r; j++) {
    const pt = kept[Math.floor(j / per)]!.point;
    const k = rep.get(pt);
    if (k === undefined || arrows[j * ARROW_STRIDE + ARROW.priority]! > arrows[k * ARROW_STRIDE + ARROW.priority]!) rep.set(pt, j);
  }
  const reps = [...rep.values()];
  const perPoint = new Float32Array(reps.length * ARROW_STRIDE);
  reps.forEach((j, m) => perPoint.set(arrows.subarray(j * ARROW_STRIDE, (j + 1) * ARROW_STRIDE), m * ARROW_STRIDE));
  thinningLevels(perPoint, radius);
  const level = new Map<number, number>();
  rep.forEach((_, pt) => level.set(pt, 0));
  [...rep.keys()].forEach((pt, m) => level.set(pt, perPoint[m * ARROW_STRIDE + ARROW.level]!));
  for (let j = 0; j < r; j++) arrows[j * ARROW_STRIDE + ARROW.level] = level.get(kept[Math.floor(j / per)]!.point)!;
  return { arrows, points: rep.size, maxAbs };
}
