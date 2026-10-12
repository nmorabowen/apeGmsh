// The nodal contour (V4c1, #1756), as pure functions: the fields a results
// stage offers, the colormaps, the colour range and what the legend says
// about it, the node at every drawn vertex, and the value at every vertex.
// No three.js and no DOM: the tests run all of it in Node. The viewport
// (renderer/contour.ts) turns these values into vertex colours; the effects
// measure the step range and run the history scan.
//
// Parity target: the Qt `viewers/diagrams/_contour.py::ContourDiagram`,
// `topology="nodes"` (read as context, never imported): a value per node,
// coloured per vertex and interpolated by the GPU inside each primitive;
// the colour range defaults to the whole history (`_compute_initial_clim`),
// and non-finite values are not values.
//
// Two rules this file keeps:
// - a node with no value (absent from the field's `_ids`, or NaN in it) is
//   "no data": never zero, never a value taken from a neighbour. A primitive
//   that touches such a node is drawn whole in the no-data colour, so no
//   colour is interpolated toward it;
// - a vertex's node is the one mesh/build.ts read its position from (the
//   element connectivity, `MeshInfo.vertexNodes`), never a node found by
//   position: coincident nodes (interfaces, ties) keep their own values.

import { hexToRgb, OFFICE, type RGB } from "../theme/tokens.ts";
import type { ContourCmap, ContourRange, ContourScan, ContourState, ResultsStage, State } from "./types.ts";

// -- fields -------------------------------------------------------------------

const AXES = ["x", "y", "z"] as const;

/**
 * The vector fields whose magnitude is offered: the translational families
 * of the canonical vocabulary (`src/apeGmsh/_vocabulary.py`, the names
 * h5-schema.md "Component names and units" points to). Rotations and moments
 * are not: a magnitude of rotation components is not a rotation.
 */
const TRANSLATIONAL = new Set(["displacement", "velocity", "acceleration", "displacement_increment", "reaction_force"]);

/** The key of the magnitude component. A dataset name never has this form (the `|` cannot appear in a vocabulary name). */
export const MAGNITUDE = "|magnitude|";

export interface FieldComponent {
  /** a dataset name, or MAGNITUDE */
  key: string;
  /** what the picker and the legend show: the axis (`x`), `magnitude`, the dataset name of a scalar field, or `[k]` for an unnamed one */
  label: string;
  /** the datasets read for it: one, or every axis component for the magnitude */
  datasets: string[];
  magnitude: boolean;
}

export interface NodalField {
  name: string;
  components: FieldComponent[];
  /**
   * Why the magnitude is not offered although the field is a translational
   * vector: an axis of it was refused (a partial write). A magnitude of the
   * axes that survived would be a guessed value. Null otherwise.
   */
  magnitudeRefused: string | null;
}

/**
 * The nodal fields of a stage's component names, in first-seen order. A name
 * `<field>_<x|y|z>` is an axis component of `<field>`; a name ending in `_`
 * is an unnamed component of the name before it, labelled `[k]` (its
 * position in that field), never a guessed axis; any other name is a scalar
 * field of its own (`pore_pressure`). The magnitude comes first for a
 * translational field with two or more axis components, none of them
 * `refused` (the reader's per-component refusals: a magnitude without a
 * refused axis is not the magnitude).
 */
export function nodalFields(names: readonly string[], refused: readonly { component: string; reason: string }[] = []): NodalField[] {
  const order: string[] = [];
  const parts = new Map<string, { axis: Map<string, string>; other: string[] }>();
  const at = (f: string) => {
    let p = parts.get(f);
    if (!p) {
      p = { axis: new Map(), other: [] };
      parts.set(f, p);
      order.push(f);
    }
    return p;
  };
  for (const n of names) {
    const m = /^(.+)_([^_]*)$/.exec(n);
    if (m && (AXES as readonly string[]).includes(m[2]!)) at(m[1]!).axis.set(m[2]!, n);
    else if (m && m[2] === "") at(m[1]!).other.push(n);
    else at(n).other.push(n);
  }
  return order.map((name) => {
    const p = parts.get(name)!;
    const comps: FieldComponent[] = [];
    const axis = AXES.filter((a) => p.axis.has(a)).map((a) => p.axis.get(a)!);
    const lost = TRANSLATIONAL.has(name) ? refused.filter((r) => AXES.some((a) => r.component === `${name}_${a}`)) : [];
    const magnitudeRefused = lost.length ? `${lost.map((r) => r.component).join(", ")} not read (${lost.map((r) => r.reason).join("; ")})` : null;
    if (TRANSLATIONAL.has(name) && axis.length >= 2 && !magnitudeRefused) comps.push({ key: MAGNITUDE, label: "magnitude", datasets: axis, magnitude: true });
    for (const a of AXES) if (p.axis.has(a)) comps.push({ key: p.axis.get(a)!, label: a, datasets: [p.axis.get(a)!], magnitude: false });
    let k = comps.filter((c) => !c.magnitude).length;
    for (const o of p.other) {
      const label = o === name ? o : `[${k}]`;
      comps.push({ key: o, label, datasets: [o], magnitude: false });
      k++;
    }
    return { name, components: comps, magnitudeRefused };
  });
}

/** A stage's nodal fields, with its refused components taken into account. */
export const fieldsOf = (stage: ResultsStage): NodalField[] => nodalFields(stage.nodal, stage.refusedNodal);

/** The field and component of a stage that a selection names, or null when the stage does not have it. */
export function componentOf(stage: ResultsStage, field: string, component: string): { field: NodalField; component: FieldComponent } | null {
  const f = fieldsOf(stage).find((x) => x.name === field);
  const c = f?.components.find((x) => x.key === component);
  return f && c ? { field: f, component: c } : null;
}

/** The default pick on a stage: the first field, its first component (the magnitude when offered). */
export function defaultPick(stage: ResultsStage | undefined): { field: string; component: string } | null {
  const f = stage ? fieldsOf(stage)[0] : undefined;
  const c = f?.components[0];
  return f && c ? { field: f.name, component: c.key } : null;
}

// -- keys ---------------------------------------------------------------------

/** The step key (the range of one step of one component) and the scan key (its whole history). */
export function contourKeys(path: string, stageId: string, step: number, field: string, component: string): { step: string; scan: string } {
  const scan = `${path}|${stageId}|${field}|${component}`;
  return { scan, step: `${scan}|${step}` };
}

/** The keys of what the state shows: a results step and a contour pick; null otherwise. */
export function contourKeysOf(s: State): { step: string; scan: string } | null {
  const at = s.phase.at;
  const c = s.contour;
  if (at?.kind !== "results" || !s.results || c.field === null || c.component === null) return null;
  const st = s.results.stages[at.stage];
  return st ? contourKeys(s.results.path, st.id, at.step, c.field, c.component) : null;
}

/** A component key as a sentence names it: `magnitude`, or the dataset name. */
export const componentName = (key: string): string => (key === MAGNITUDE ? "magnitude" : key);

// -- colormaps ----------------------------------------------------------------

type Stops = readonly (readonly [number, number, number, number])[];
type Segments = readonly (readonly [number, number])[];

/** Stops `[t, r, g, b]` from per-channel piecewise-linear segments: at the union of their breakpoints, so the result is exact. */
function fromSegments(r: Segments, g: Segments, b: Segments): Stops {
  const at = (seg: Segments, t: number): number => {
    for (let i = 1; i < seg.length; i++) {
      const [t1, v1] = seg[i]!;
      if (t <= t1) {
        const [t0, v0] = seg[i - 1]!;
        return v0 + ((t - t0) / (t1 - t0)) * (v1 - v0);
      }
    }
    return seg[seg.length - 1]![1];
  };
  const ts = [...new Set([...r, ...g, ...b].map(([t]) => t))].sort((x, y) => x - y);
  return ts.map((t) => [t, at(r, t), at(g, t), at(b, t)] as const);
}

/**
 * The colormaps, as sRGB stops `[t, r, g, b]` interpolated linearly. Jet is
 * matplotlib's piecewise-linear `jet`, built from its segment data, the
 * maintainer's stated preference for contours; viridis and coolwarm are the
 * perceptual alternatives, sampled from matplotlib's `viridis` and
 * Moreland's `coolwarm`.
 */
export const CMAPS: Readonly<Record<ContourCmap, Stops>> = {
  jet: fromSegments(
    [[0, 0], [0.35, 0], [0.66, 1], [0.89, 1], [1, 0.5]],
    [[0, 0], [0.125, 0], [0.375, 1], [0.64, 1], [0.91, 0], [1, 0]],
    [[0, 0.5], [0.11, 1], [0.34, 1], [0.65, 0], [1, 0]],
  ),
  viridis: [
    [0, 0.267, 0.005, 0.329],
    [0.125, 0.283, 0.141, 0.458],
    [0.25, 0.254, 0.265, 0.53],
    [0.375, 0.207, 0.372, 0.553],
    [0.5, 0.164, 0.471, 0.558],
    [0.625, 0.128, 0.567, 0.551],
    [0.75, 0.135, 0.659, 0.518],
    [0.875, 0.478, 0.821, 0.318],
    [1, 0.993, 0.906, 0.144],
  ],
  coolwarm: [
    [0, 0.23, 0.299, 0.754],
    [0.25, 0.552, 0.69, 0.996],
    [0.5, 0.865, 0.865, 0.865],
    [0.75, 0.958, 0.603, 0.482],
    [1, 0.706, 0.016, 0.15],
  ],
};

export const CMAP_NAMES: readonly ContourCmap[] = ["jet", "viridis", "coolwarm"];

/**
 * The "no data" colour (sRGB): the office's muted grey (theme/tokens.ts),
 * away from every ramp above (test/contour.test.ts checks its distance to
 * each sample of each ramp), and named in the legend.
 */
export const NO_DATA_RGB: RGB = hexToRgb(OFFICE.muted);

/** The ramp colour (sRGB, 0..1) at `t` in [0, 1] (clamped). */
export function sampleCmap(cmap: ContourCmap, t: number): [number, number, number] {
  const stops = CMAPS[cmap];
  if (!stops) throw new RangeError(`unknown colormap ${JSON.stringify(cmap)}`);
  const x = Math.min(1, Math.max(0, t));
  for (let i = 1; i < stops.length; i++) {
    const b = stops[i]!;
    if (x <= b[0]) {
      const a = stops[i - 1]!;
      const u = b[0] === a[0] ? 0 : (x - a[0]) / (b[0] - a[0]);
      return [a[1] + u * (b[1] - a[1]), a[2] + u * (b[2] - a[2]), a[3] + u * (b[3] - a[3])];
    }
  }
  const last = stops[stops.length - 1]!;
  return [last[1], last[2], last[3]];
}

/** Where a value falls on the ramp: 0 at min, 1 at max; a constant field (min = max) sits at the middle. */
export function rampT(v: number, min: number, max: number): number {
  return max > min ? (v - min) / (max - min) : 0.5;
}

// -- ranges ---------------------------------------------------------------------

/** The finite min and max over every value of `rows` (one component), or of their magnitude; null when none is finite. */
export function finiteRange(rows: readonly Float64Array[], magnitude: boolean): { min: number; max: number } | null {
  if (!rows.length) return null;
  let lo = Infinity;
  let hi = -Infinity;
  const n = rows[0]!.length;
  for (let j = 0; j < n; j++) {
    let v: number;
    if (magnitude) {
      let sq = 0;
      for (const r of rows) sq += r[j]! * r[j]!;
      v = Math.sqrt(sq);
    } else v = rows[0]![j]!;
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return lo <= hi ? { min: lo, max: hi } : null;
}

/** The colour range in use and the sentence the legend states for it. */
export interface Clim {
  /** null: the range in use has no finite value (everything is no data) */
  min: number | null;
  max: number | null;
  /** which range is in use: the legend always says it */
  mode: "history" | "step" | "fixed";
  label: string;
}

/**
 * The range a contour colours with. `history` uses the scan's range once it
 * is done; until then the step's range, labelled with the scan's progress
 * (or its failure). `step` is the step's range; `fixed` the user's.
 */
export function climOf(c: Pick<ContourState, "range" | "fixed">, step: { min: number | null; max: number | null } | null, scan: ContourScan | null, scanKey: string): Clim {
  const stepClim = (label: string): Clim => ({ min: step?.min ?? null, max: step?.max ?? null, mode: "step", label });
  switch (c.range) {
    case "fixed":
      if (!c.fixed) throw new Error("contour range fixed with no min and max");
      return { min: c.fixed.min, max: c.fixed.max, mode: "fixed", label: "range: fixed (user)" };
    case "step":
      return stepClim("range: this step");
    case "history": {
      const sc = scan && scan.key === scanKey ? scan : null;
      if (sc?.status === "done") return { min: sc.min, max: sc.max, mode: "history", label: `range: whole history (${sc.total} steps)` };
      if (sc?.status === "failed") return stepClim(`range: this step (history scan failed: ${sc.error ?? "unknown error"})`);
      const pct = sc && sc.total > 0 ? Math.floor((100 * sc.done) / sc.total) : 0;
      return stepClim(`range: this step (history scan ${pct}%)`);
    }
    default: {
      const unknown: never = c.range;
      throw new Error(`unknown contour range ${JSON.stringify(unknown)}`);
    }
  }
}

export const RANGE_MODES: readonly { range: ContourRange; label: string }[] = [
  { range: "history", label: "history" },
  { range: "step", label: "this step" },
  { range: "fixed", label: "fixed" },
];

// -- vertices -------------------------------------------------------------------

/** The column of each node id in a field's `_ids`. */
export function columnOf(ids: Float64Array): Map<number, number> {
  const m = new Map<number, number>();
  ids.forEach((id, j) => {
    if (!m.has(id)) m.set(id, j);
  });
  return m;
}

/**
 * The value at every vertex: the component (or the magnitude of `rows`) at
 * the vertex's node. NaN ("no data") when the node is not in the field's
 * `_ids`, or its value is not finite. Never 0.
 */
export function vertexValues(vertexNodes: Float64Array, column: ReadonlyMap<number, number>, rows: readonly Float64Array[], magnitude: boolean): Float64Array {
  const out = new Float64Array(vertexNodes.length);
  for (let i = 0; i < vertexNodes.length; i++) {
    const j = column.get(vertexNodes[i]!);
    if (j === undefined) {
      out[i] = NaN;
      continue;
    }
    let v: number;
    if (magnitude) {
      let sq = 0;
      for (const r of rows) sq += r[j]! * r[j]!;
      v = Math.sqrt(sq);
    } else v = rows[0]![j]!;
    out[i] = Number.isFinite(v) ? v : NaN;
  }
  return out;
}

/**
 * Per-vertex sRGB colours of the drawn primitives (`map`: original primitive
 * indices, `perPrim` vertices each). A primitive with any no-data vertex, or
 * any primitive when the range has no finite value, is no-data whole.
 * Returns the colours and how many drawn primitives are no data.
 */
export function primitiveColours(
  values: Float64Array,
  map: Int32Array,
  perPrim: number,
  clim: { min: number | null; max: number | null },
  cmap: ContourCmap,
): { rgb: Float32Array; noData: number } {
  const rgb = new Float32Array(map.length * perPrim * 3);
  let noData = 0;
  const ok = clim.min !== null && clim.max !== null;
  for (let d = 0; d < map.length; d++) {
    const o = map[d]! * perPrim;
    let missing = !ok;
    for (let v = 0; v < perPrim && !missing; v++) if (Number.isNaN(values[o + v]!)) missing = true;
    if (missing) noData++;
    for (let v = 0; v < perPrim; v++) {
      const c = missing ? NO_DATA_RGB : sampleCmap(cmap, rampT(values[o + v]!, clim.min!, clim.max!));
      rgb.set(c, (d * perPrim + v) * 3);
    }
  }
  return { rgb, noData };
}
