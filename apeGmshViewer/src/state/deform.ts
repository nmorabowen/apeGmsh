// The deformed shape (V4d, #1765), as pure functions: which nodal datasets
// are the translational displacement of a results stage (or why none can be
// used), the model's spatial dimension, the displacement at every drawn
// vertex, and the deformed positions. No three.js and no DOM: the tests run
// all of it in Node. The effects compute the displacement of the step shown;
// the viewport (renderer/deform.ts) writes the positions.
//
// Parity target: the Qt deformed geometry (`viewers/diagrams/_geometries.py`,
// `deform_scale`, `add_reference_ghost`, `GHOST_OPACITY`; read as context,
// never imported): positions are `x + scale * u`, translations only, scale 1
// by default, and the reference ghost is the undeformed mesh at 0.3 opacity.
//
// The rules this file keeps:
// - a vertex's node is the one mesh/build.ts read its position from
//   (`MeshInfo.vertexNodes`, V4c1), never a node found by position;
// - a node with no displacement (absent from `_ids`, or NaN on any axis
//   used) is "missing": it is drawn undeformed and counted, never moved by a
//   displacement of 0 taken as a value;
// - a displacement axis the reader refused (a partial write) refuses the
//   whole deformed shape: it is never drawn from the axes that survived;
// - an axis is never assumed: unnamed components (`displacement_`, `[k]`)
//   refuse, and a missing `displacement_z` is 0 only in a model declared
//   2-D (`ndm` 2), where z is not a degree of freedom.

import { fieldsOf } from "./contour.ts";
import type { Param } from "../model/types.ts";
import type { ResultsStage, SpatialNdm } from "./types.ts";

/** The reference ghost's opacity: the Qt `GHOST_OPACITY`. */
export const GHOST_OPACITY = 0.3;

/** The translational displacement field of the canonical vocabulary (h5-schema.md "Component names and units"). */
export const DISPLACEMENT = "displacement";

/** The first neutral version whose `/meta/ndm` is the declared spatial dimension (h5-schema.md: `2.34.0`, #1291). */
export const NDM_IS_SPATIAL_FROM = [2, 34, 0] as const;

/** The datasets of the x, y and z displacement; z is null in a 2-D model. */
export type DisplacementAxes =
  | { ok: true; datasets: readonly [string, string, string | null]; note: string | null }
  /** `absent`: the stage records no displacement; `refused`: it does, and it cannot be used (the reason says why) */
  | { ok: false; kind: "absent" | "refused"; reason: string };

const DISP_NAME = /^displacement_([^_]*)$/;

/**
 * The displacement datasets of a stage, or why the deformed shape cannot use
 * them. `stage.nodal` holds what the reader read; `stage.refusedNodal` what
 * it refused one by one (V4c1's partial-write refusal), which never appears
 * in `nodal`, so it is checked first: a refused z must not read as "a 2-D
 * file has no z".
 */
export function displacementAxes(stage: ResultsStage, ndm: SpatialNdm): DisplacementAxes {
  const label = stage.name || stage.id;
  const refused = stage.refusedNodal.filter((r) => DISP_NAME.test(r.component));
  if (refused.length) {
    return {
      ok: false,
      kind: "refused",
      reason:
        `${refused.map((r) => r.component).join(", ")} not read (${refused.map((r) => r.reason).join("; ")}); ` +
        "the deformed shape is never drawn from the axes that survive",
    };
  }
  const field = fieldsOf(stage).find((f) => f.name === DISPLACEMENT);
  if (!field) return { ok: false, kind: "absent", reason: `stage ${label} records no nodal displacement` };
  const odd = field.components.filter((c) => !c.magnitude && !["x", "y", "z"].includes(c.label));
  if (odd.length) {
    return {
      ok: false,
      kind: "refused",
      reason: `the displacement components ${odd.map((c) => `${c.key} (${c.label})`).join(", ")} name no axis, so which is x, y or z is ambiguous`,
    };
  }
  const axis = (a: string) => field.components.find((c) => !c.magnitude && c.label === a)?.key ?? null;
  const x = axis("x"), y = axis("y"), z = axis("z");
  if (x && y && z) return { ok: true, datasets: [x, y, z], note: null };
  if (x && y && !z) {
    if (ndm.ndm === 2) {
      return {
        ok: true,
        datasets: [x, y, null],
        // h5-schema.md, "Nodal results": a shorthand request is clipped to the model's ndm, so a 2-D file has no displacement_z.
        note: `2-D model (ndm 2, ${ndm.source}): z is not a degree of freedom, so it stays at the node's z`,
      };
    }
    return {
      ok: false,
      kind: "refused",
      reason:
        `stage ${label} records no displacement_z and the model is not declared 2-D ` +
        `(${ndm.ndm === null ? ndm.reason : `ndm ${ndm.ndm}`}): a z displacement of 0 would be a guess`,
    };
  }
  const missing = (["x", "y"] as const).filter((a) => !axis(a)).map((a) => `displacement_${a}`);
  return { ok: false, kind: "refused", reason: `stage ${label} records no ${missing.join(" or ")}: the deformed shape needs every translational axis` };
}

const parseVersion = (v: string): number[] | null => {
  const m = /^(\d+)\.(\d+)\.(\d+)$/.exec(v);
  return m ? [Number(m[1]), Number(m[2]), Number(m[3])] : null;
};

/**
 * The model's spatial dimension from its `/meta` attrs and neutral version.
 * Before neutral 2.34.0 `/meta/ndm` was the mesh dimension (a shell-only 3-D
 * model stamped 2), and `0` means the file declares none; neither is taken
 * to be 2-D.
 */
export function spatialNdm(meta: Readonly<Record<string, Param | Param[]>>, neutralVersion: string): SpatialNdm {
  const raw = meta["ndm"];
  if (typeof raw !== "number") return { ndm: null, reason: "the model file has no /meta/ndm" };
  const v = parseVersion(neutralVersion);
  const from = NDM_IS_SPATIAL_FROM;
  const spatial = v !== null && (v[0]! !== from[0] ? v[0]! > from[0] : v[1]! !== from[1] ? v[1]! > from[1] : v[2]! >= from[2]);
  if (!spatial) return { ndm: null, reason: `/meta/ndm of a neutral ${neutralVersion} file is the mesh dimension, not the declared one` };
  if (raw === 0) return { ndm: null, reason: "the model declares no ndm (/meta/ndm 0, a broker-only file)" };
  if (raw === 2 || raw === 3) return { ndm: raw, source: `/meta/ndm, neutral ${neutralVersion}` };
  return { ndm: null, reason: `/meta/ndm is ${raw}` };
}

/** The displacement at every vertex (3 per vertex) and which vertices have none. */
export interface VertexDisplacement {
  /** f64, 3 per vertex; 0 on a missing vertex (which is drawn undeformed, and flagged) */
  d: Float64Array;
  /** 1 where the vertex's node has no displacement: absent from `_ids`, or not finite on an axis used */
  missing: Uint8Array;
  /** how many vertices are missing */
  count: number;
}

/**
 * The displacement of every vertex, through the vertex's node id and the
 * column of that node in the stage's `_ids`. `rows` are the step's x, y and
 * z datasets; z is null only in a 2-D model, where it is 0 by definition.
 */
export function vertexDisplacements(
  vertexNodes: Float64Array,
  column: ReadonlyMap<number, number>,
  rows: readonly [Float64Array, Float64Array, Float64Array | null],
): VertexDisplacement {
  const n = vertexNodes.length;
  const d = new Float64Array(3 * n);
  const missing = new Uint8Array(n);
  let count = 0;
  const [rx, ry, rz] = rows;
  for (let i = 0; i < n; i++) {
    const j = column.get(vertexNodes[i]!);
    if (j === undefined) {
      missing[i] = 1;
      count++;
      continue;
    }
    const ux = rx[j]!, uy = ry[j]!, uz = rz ? rz[j]! : 0;
    if (!Number.isFinite(ux) || !Number.isFinite(uy) || !Number.isFinite(uz)) {
      missing[i] = 1;
      count++;
      continue;
    }
    d[3 * i] = ux;
    d[3 * i + 1] = uy;
    d[3 * i + 2] = uz;
  }
  return { d, missing, count };
}

/** `base + scale * d`, per coordinate, into `out` (allocated when absent). */
export function deformedPositions(base: Float32Array, d: Float64Array, scale: number, out: Float32Array = new Float32Array(base.length)): Float32Array {
  if (d.length !== base.length) throw new RangeError(`deformedPositions: ${d.length} displacements for ${base.length} coordinates`);
  for (let i = 0; i < base.length; i++) out[i] = base[i]! + scale * d[i]!;
  return out;
}

/**
 * The drawn nodes and those of them with no displacement: each node once,
 * over the vertices of the drawn primitives (`lineMap`, `triMap`: original
 * indices, as renderer/visible.ts gives them), so a hidden or inactive
 * element's nodes do not count.
 */
export function missingDrawnNodes(
  lineNodes: Float64Array,
  triNodes: Float64Array,
  lineMissing: Uint8Array,
  triMissing: Uint8Array,
  lineMap: Int32Array,
  triMap: Int32Array,
): { drawn: number; missing: number } {
  const drawn = new Set<number>();
  const missing = new Set<number>();
  const take = (nodes: Float64Array, miss: Uint8Array, map: Int32Array, per: number) => {
    for (let k = 0; k < map.length; k++) {
      const o = map[k]! * per;
      for (let v = 0; v < per; v++) {
        const id = nodes[o + v]!;
        drawn.add(id);
        if (miss[o + v]) missing.add(id);
      }
    }
  };
  take(lineNodes, lineMissing, lineMap, 2);
  take(triNodes, triMissing, triMap, 3);
  return { drawn: drawn.size, missing: missing.size };
}

/**
 * The positions of the drawn vertices whose node has no displacement, each
 * node once (3 floats each): where the viewport marks them.
 */
export function missingMarkers(base: Float32Array, nodes: Float64Array, missing: Uint8Array, map: Int32Array, per: number, seen: Set<number> = new Set()): number[] {
  const out: number[] = [];
  for (let k = 0; k < map.length; k++) {
    const o = map[k]! * per;
    for (let v = 0; v < per; v++) {
      if (!missing[o + v]) continue;
      const id = nodes[o + v]!;
      if (seen.has(id)) continue;
      seen.add(id);
      out.push(base[3 * (o + v)]!, base[3 * (o + v) + 1]!, base[3 * (o + v) + 2]!);
    }
  }
  return out;
}

/** The scale as the legend states it. */
export const scaleLabel = (scale: number): string => (scale === 1 ? "true scale (×1)" : `scale ×${Number(scale.toPrecision(4))} (not true scale)`);
