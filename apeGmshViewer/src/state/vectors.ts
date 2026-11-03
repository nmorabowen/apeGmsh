// The Vectors card's field and glyphs: a nodal vector field of the results step
// on screen (displacement, velocity, acceleration, displacement increment, and
// the `.ladruno` groups of the same fields) as arrows starting at the nodes, and
// a rotational field (rotation, angular velocity) as half circles about each
// node's axis (renderer/loads.ts `VectorsLayer`). The viewer twin of the Qt
// `viewers/diagrams/_vector_glyph.py`.
//
// Pure: no three.js and no DOM, so every function runs in Node tests.

import type { BlobStore } from "./blobs.ts";
import { fieldsOf, LADRUNO_ROTATION, sampleCmap } from "./contour.ts";
import {
  ANCHOR,
  ARROW,
  ARROW_STRIDE,
  buildNodeCoordsMap,
  loadsColourPalette,
  pushArrowToPoint,
  thinningLevelsByNode,
  type BuildLoads3DOptions,
  type GlyphBuffers,
  type Vec3,
} from "./loads.ts";
import type { State } from "./types.ts";

const AXES = ["x", "y", "z"] as const;

/** The canonical rotational fields; with the `.ladruno` rotational groups, a field drawn as half circles. */
const ROTATIONAL = new Set(["rotation", "angular_velocity", "angular_acceleration", "reaction_moment"]);

/** A nodal vector field a results stage records: its axis components (`x`, `y`, `z`, null where not recorded). */
export interface VectorField {
  /** The nodal field name (`DISPLACEMENT`, `velocity`), as the Contour card names it. */
  readonly name: string;
  /** The dataset of each axis, in x, y, z order; null for an axis the stage does not record. */
  readonly axes: readonly (string | null)[];
  /** Drawn as half circles about the axis (right-hand rule), not arrows. */
  readonly rotational: boolean;
}

/**
 * The vector fields of the results stage on screen: each nodal field with two
 * or more axis components (`state/contour.ts` `nodalFields`, so `<field>_x` and
 * the `.ladruno` `<RESULT>/Ux` labels alike), none of them refused (a vector
 * missing an axis it records is not the vector). A 2-D model's lone `Rz` is no
 * vector (`nodalFields` keeps it a component of its own: it may be a pore
 * pressure), so a 2-D model offers its translations only. Empty off a results step.
 */
export function vectorFieldsOf(s: State): readonly VectorField[] {
  const at = s.phase.at;
  if (at?.kind !== "results" || !s.results) return [];
  const stage = s.results.stages[at.stage];
  if (!stage) return [];
  const out: VectorField[] = [];
  for (const f of fieldsOf(stage)) {
    const axes = AXES.map((a) => f.components.find((c) => !c.magnitude && c.label === a)?.key ?? null);
    if (axes.filter((a) => a !== null).length < 2) continue;
    const name = f.name;
    if (stage.refusedNodal.some((r) => r.component === name || r.component.startsWith(`${name}/`) || AXES.some((a) => r.component === `${name}_${a}`))) continue;
    out.push({ name, axes, rotational: ROTATIONAL.has(name) || LADRUNO_ROTATION.has(name) });
  }
  return out;
}

/** The field the Vectors card draws: the one it names, or the stage's first vector field; null when the stage has none. */
export function vectorFieldOf(s: State): VectorField | null {
  const fields = vectorFieldsOf(s);
  return fields.find((f) => f.name === s.vectors.field) ?? fields[0] ?? null;
}

/** A vector below this fraction of the step's largest is solver noise (a fixed node's 1e-18), not drawn. */
export const VECTOR_FLOOR = 1e-6;

export interface Vectors3D {
  /** One record per arrow, `ARROW_STRIDE` floats, anchored at its tail (`ANCHOR.tail`): the point is the node. */
  readonly arrows: Float32Array;
  /** One record per half circle (a rotational field), `ARROW_STRIDE` floats: the point is the node, the direction the axis. */
  readonly arcs: Float32Array;
  /** Glyphs drawn: one per node for the resultant or one axis, up to three for `xyz`. */
  readonly count: number;
  /** The step's largest glyph (resultant or component), which sets the longest one. */
  readonly maxMag: number;
  /** The field drawn, or null. */
  readonly field: VectorField | null;
}

/**
 * The glyphs of the Vectors card's field at one results step, at each node with
 * a finite, non-zero vector above the noise floor: an arrow starting AT the node
 * and pointing along the vector (a load or a reaction points at its node; a
 * displacement leaves it), or for a rotational field a half circle about the
 * vector's axis. Along the resultant or along the global axes as
 * `s.vectors.component` asks (all three, or one; an axis the field does not
 * record draws nothing). Lengths follow the step's largest glyph, as the
 * reaction arrows do, and thinning is decided per node. `nodal` and `nodeIds`
 * are the step's rows and the stage's node ids; the node positions are the
 * drawn (deformed) ones when `opts` carries them.
 */
export function buildVectors3D(
  s: State,
  blobs: BlobStore,
  nodal: Readonly<Record<string, Float64Array>>,
  nodeIds: Float64Array,
  opts?: BuildLoads3DOptions,
): Vectors3D {
  const empty = new Float32Array(0);
  const field = vectorFieldOf(s);
  const none: Vectors3D = { arrows: empty, arcs: empty, count: 0, maxMag: 0, field };
  if (!s.mesh || !field) return none;
  const rows = field.axes.map((k) => (k ? nodal[k] : undefined));
  if (!rows.some((r) => r)) return none;
  const want = s.vectors.component;
  const picked = want === "xyz" ? [0, 1, 2] : want === "x" ? [0] : want === "y" ? [1] : want === "z" ? [2] : null;
  const nodeCoords = buildNodeCoordsMap(s, blobs, opts);
  type Item = { at: Vec3; vec: Vec3; mag: number; node: number };
  const items: Item[] = [];
  let max = 0;
  for (let i = 0; i < nodeIds.length; i++) {
    const at = nodeCoords.get(nodeIds[i]!);
    if (!at) continue;
    const v = rows.map((r) => (r ? r[i]! : 0));
    if (!v.every(Number.isFinite)) continue;
    const vecs: Vec3[] = picked ? picked.filter((a) => rows[a]).map((a) => [a === 0 ? v[0]! : 0, a === 1 ? v[1]! : 0, a === 2 ? v[2]! : 0]) : [[v[0]!, v[1]!, v[2]!]];
    for (const vec of vecs) {
      const mag = Math.hypot(...vec);
      if (mag <= 1e-30) continue;
      items.push({ at, vec, mag, node: i });
      if (mag > max) max = mag;
    }
  }
  const kept = items.filter((it) => it.mag >= VECTOR_FLOOR * max);
  if (kept.length === 0) return none;
  const radius = s.mesh.radius > 0 && Number.isFinite(s.mesh.radius) ? s.mesh.radius : 1;
  const baseLen = Math.max(radius * 0.22, 1e-4) * s.vectors.scale;
  const palette = loadsColourPalette(opts?.theme ?? "dark");
  const g: GlyphBuffers = { arrows: [], combs: [] };
  for (const it of kept) {
    const ratio = it.mag / max;
    const rgb = s.vectors.colorBy === "type" ? (field.rotational ? palette.moment : palette.force) : s.vectors.colorBy === "magnitude" ? sampleCmap("jet", ratio) : palette.mono;
    pushArrowToPoint(g, it.at, [it.vec[0] / it.mag, it.vec[1] / it.mag, it.vec[2] / it.mag], baseLen * (0.35 + 0.65 * ratio), rgb, ratio);
    // An arc is centred on its node either way; an arrow starts there.
    if (!field.rotational) g.arrows[g.arrows.length - ARROW_STRIDE + ARROW.anchor] = ANCHOR.tail;
  }
  const records = new Float32Array(g.arrows);
  thinningLevelsByNode(
    kept.map((_, i) => records.subarray(i * ARROW_STRIDE, (i + 1) * ARROW_STRIDE)),
    kept.map((it) => it.node),
    radius,
  );
  return { arrows: field.rotational ? empty : records, arcs: field.rotational ? records : empty, count: kept.length, maxMag: max, field };
}
