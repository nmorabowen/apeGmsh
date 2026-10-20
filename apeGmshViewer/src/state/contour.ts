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
import type { NodeTopology } from "./extrapolation.ts";
import { CMAP_NAMES, type ContourCmap, type ContourRange, type ContourScan, type ContourSource, type ContourState, type DeclPath, type Decl, type GaussGroup, type GaussMode, type ResultsStage, type State } from "./types.ts";
export { CMAP_NAMES };

// -- fields -------------------------------------------------------------------

const AXES = ["x", "y", "z"] as const;

/**
 * The vector fields whose magnitude is offered: the translational families
 * of the canonical vocabulary (`src/apeGmsh/_vocabulary.py`, the names
 * h5-schema.md "Component names and units" points to). Rotations and moments
 * are not: a magnitude of rotation components is not a rotation.
 */
const TRANSLATIONAL = new Set(["displacement", "velocity", "acceleration", "displacement_increment", "reaction_force"]);

/**
 * The `.ladruno` ON_NODES groups whose vector magnitude is offered: the
 * translational groups of fork schema `ladruno_schema_v1.md` (§7.1.1 and the
 * §7.1.2 "Translational groups" table, fork ≥ d370ed8b). Each holds the
 * node's first `ndm` DOFs. The `…_INCLUDING_INERTIA` and Rayleigh variants
 * share their base group's labels (§7.1.2).
 */
const LADRUNO_TRANSLATIONAL = new Set([
  // §7.1.1
  "DISPLACEMENT",
  "MODES_OF_VIBRATION(U)", // a mode stage's translation, as a native mode stage's `displacement`
  // §7.1.2
  "VELOCITY",
  "ACCELERATION",
  "REACTION_FORCE",
  "REACTION_FORCE_INCLUDING_INERTIA",
  "RAYLEIGH_FORCE",
  "UNBALANCED_FORCE",
  "UNBALANCED_FORCE_INCLUDING_INERTIA",
  "CONSTRAINT_TIE_FORCE",
]);

/**
 * The axis of each `.ladruno` ON_NODES `<RESULT>/<label>` label: exactly the
 * labels fork schema `ladruno_schema_v1.md` §7.1.1 and §7.1.2 document (fork
 * ≥ d370ed8b), and no other (ADR 0122 D7). A label not here is drawn as its
 * own component, never as a guessed axis: `X`/`Y`/`Z` (no writer emits them),
 * the scalar `p` (PRESSURE, `overlayPressure_<tag>`) and the sensitivity
 * labels `d_<sym>/d_q<g>` (§7.1.2).
 */
export const LADRUNO_AXIS_MAP: Readonly<Record<string, (typeof AXES)[number]>> = {
  // §7.1.1: DISPLACEMENT, MODES_OF_VIBRATION(U)
  Ux: "x", Uy: "y", Uz: "z",
  // §7.1.1: ROTATION, MODES_OF_VIBRATION(R)
  Rx: "x", Ry: "y", Rz: "z",
  // §7.1.2 translational: VELOCITY
  Vx: "x", Vy: "y", Vz: "z",
  // §7.1.2 translational: ACCELERATION
  Ax: "x", Ay: "y", Az: "z",
  // §7.1.2 translational: REACTION_FORCE, REACTION_FORCE_INCLUDING_INERTIA, RAYLEIGH_FORCE
  RFx: "x", RFy: "y", RFz: "z",
  // §7.1.2 translational: UNBALANCED_FORCE, UNBALANCED_FORCE_INCLUDING_INERTIA
  Fx: "x", Fy: "y", Fz: "z",
  // §7.1.2 translational: CONSTRAINT_TIE_FORCE
  TFx: "x", TFy: "y", TFz: "z",
  // §7.1.2 rotational: ANGULAR_VELOCITY
  RVx: "x", RVy: "y", RVz: "z",
  // §7.1.2 rotational: ANGULAR_ACCELERATION
  RAx: "x", RAy: "y", RAz: "z",
  // §7.1.2 rotational: REACTION_MOMENT, REACTION_MOMENT_INCLUDING_INERTIA, RAYLEIGH_MOMENT
  RMx: "x", RMy: "y", RMz: "z",
  // §7.1.2 rotational: UNBALANCED_MOMENT, UNBALANCED_MOMENT_INCLUDING_INERTIA
  Mx: "x", My: "y", Mz: "z",
};

/**
 * The fork's rotational groups (`ladruno_schema_v1.md` §7.1.1, and the
 * §7.1.2 "Rotational groups" table). At ndm 2 each holds one `z` label, a
 * copy of DOF index 2: on a u-p node (ndf 3, the `*UP` / pressure-DOF
 * elements) that is the pore pressure, not a rotation, and the file does not
 * say which (the §7.1.1 caveat, which §7.1.2 applies to every rotational
 * group). So a rotational group whose only label is its `z` label is drawn as
 * its own component, labelled as the file names it (`Rz`, `RMz`, ...), never
 * as the z of a rotation vector.
 */
const LADRUNO_ROTATION = new Set([
  // §7.1.1
  "ROTATION",
  "MODES_OF_VIBRATION(R)",
  // §7.1.2
  "ANGULAR_VELOCITY",
  "ANGULAR_ACCELERATION",
  "REACTION_MOMENT",
  "REACTION_MOMENT_INCLUDING_INERTIA",
  "RAYLEIGH_MOMENT",
  "UNBALANCED_MOMENT",
  "UNBALANCED_MOMENT_INCLUDING_INERTIA",
]);

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
 * `<field>_<x|y|z>` is an axis component of `<field>`; a `.ladruno` name
 * `<RESULT>/<label>` is grouped under `<RESULT>` with its axis label mapped
 * (`Ux -> x`, `Uy -> y`, `Uz -> z`, etc.); a name ending in `_` is an unnamed
 * component of the name before it, labelled `[k]` (its position in that
 * field), never a guessed axis; any other name is a scalar field of its own
 * (`pore_pressure`). The magnitude comes first for a translational field with
 * two or more axis components, none of them `refused` (the reader's
 * per-component refusals: a magnitude without a refused axis is not the
 * magnitude).
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
  // §7.1.1/§7.1.2 trap: a rotational group whose only label is its `z` label (ndm 2) may hold a pore pressure.
  const rotationLabels = new Map<string, string[]>();
  for (const n of names) {
    const slash = /^([^/]+)\/(.+)$/.exec(n);
    if (slash && LADRUNO_ROTATION.has(slash[1]!)) rotationLabels.set(slash[1]!, [...(rotationLabels.get(slash[1]!) ?? []), slash[2]!]);
  }
  const loneZ = new Set([...rotationLabels].filter(([, l]) => l.length === 1 && LADRUNO_AXIS_MAP[l[0]!] === "z").map(([f]) => f));
  for (const n of names) {
    const slash = /^([^/]+)\/(.+)$/.exec(n);
    if (slash) {
      const field = slash[1]!;
      const sub = slash[2]!;
      const axis = loneZ.has(field) || !Object.hasOwn(LADRUNO_AXIS_MAP, sub) ? undefined : LADRUNO_AXIS_MAP[sub];
      if (axis && !at(field).axis.has(axis)) at(field).axis.set(axis, n);
      else at(field).other.push(n);
      continue;
    }
    const m = /^(.+)_([^_]*)$/.exec(n);
    if (m && (AXES as readonly string[]).includes(m[2]!)) at(m[1]!).axis.set(m[2]!, n);
    else if (m && m[2] === "") at(m[1]!).other.push(n);
    else at(n).other.push(n);
  }
  return order.map((name) => {
    const p = parts.get(name)!;
    const comps: FieldComponent[] = [];
    const axis = AXES.filter((a) => p.axis.has(a)).map((a) => p.axis.get(a)!);
    const isTranslational = TRANSLATIONAL.has(name) || LADRUNO_TRANSLATIONAL.has(name);
    const lost = isTranslational
      ? refused.filter(
          (r) =>
            AXES.some((a) => r.component === `${name}_${a}`) ||
            r.component === name ||
            r.component.startsWith(`${name}/`),
        )
      : [];
    const magnitudeRefused = lost.length ? `${lost.map((r) => r.component).join(", ")} not read (${lost.map((r) => r.reason).join("; ")})` : null;
    if (isTranslational && axis.length >= 2 && !magnitudeRefused) comps.push({ key: MAGNITUDE, label: "magnitude", datasets: axis, magnitude: true });
    for (const a of AXES) if (p.axis.has(a)) comps.push({ key: p.axis.get(a)!, label: a, datasets: [p.axis.get(a)!], magnitude: false });
    let k = comps.filter((c) => !c.magnitude).length;
    for (const o of p.other) {
      const slash = /^([^/]+)\/(.+)$/.exec(o);
      const label = slash ? slash[2]! : o === name ? o : `[${k}]`;
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

/**
 * The default pick on a stage: the first nodal field, its first component (the
 * magnitude when offered); V4c2a: on a stage with no nodal field, the first
 * element field that is not refused.
 */
export function defaultPick(stage: ResultsStage | undefined): { field: string; component: string; source: ContourSource } | null {
  const f = stage ? fieldsOf(stage)[0] : undefined;
  const c = f?.components[0];
  if (f && c) return { field: f.name, component: c.key, source: "nodal" };
  const e = stage ? gaussFields(stage.gauss).find((x) => x.refused === null) : undefined;
  const ec = e?.components[0];
  return e && ec ? { field: e.name, component: ec.key, source: "element" } : null;
}

/** Whether a stage has the pick (`componentOf` for nodal, `elementComponentOf` for element). */
export function hasPick(stage: ResultsStage, source: ContourSource, field: string, component: string): boolean {
  return source === "element" ? elementComponentOf(stage, field, component) !== null : componentOf(stage, field, component) !== null;
}

// -- keys ---------------------------------------------------------------------

/** The step key (the range of one step of one component) and the scan key (its whole history). */
export function contourKeys(path: string, stageId: string, step: number, field: string, component: string): { step: string; scan: string } {
  const scan = `${path}|${stageId}|${field}|${component}`;
  return { scan, step: `${scan}|${step}` };
}

/**
 * The keys of what the state shows: a results step and a contour pick; null
 * otherwise. An element field's keys name it `elem:<field>` (V4c2a), so a
 * nodal and an element field of one name never share a range or a scan.
 */
export function contourKeysOf(s: State): { step: string; scan: string } | null {
  const at = s.phase.at;
  const c = s.contour;
  if (at?.kind !== "results" || !s.results || c.field === null || c.component === null) return null;
  const st = s.results.stages[at.stage];
  // V4c2b: a node-located mode names itself too (its range is not the flat cells' range).
  const field = c.source !== "element" ? c.field : c.gauss === "cell" ? `elem:${c.field}` : `elem:${c.field}@${c.gauss}`;
  return st ? contourKeys(s.results.path, st.id, at.step, field, c.component) : null;
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
 * maintainer's stated preference for contours; turbo, viridis, plasma,
 * inferno and magma are the sequential alternatives; coolwarm, RdBu_r, RdBu,
 * bwr, seismic, BrBG and Spectral are the diverging maps (RdBu_r, bwr and
 * seismic are the symmetric blue-white-red wavefield palettes: negative blue,
 * zero white, positive red).
 */
export const CMAPS: Readonly<Record<ContourCmap, Stops>> = {
  jet: fromSegments(
    [[0, 0], [0.35, 0], [0.66, 1], [0.89, 1], [1, 0.5]],
    [[0, 0], [0.125, 0], [0.375, 1], [0.64, 1], [0.91, 0], [1, 0]],
    [[0, 0.5], [0.11, 1], [0.34, 1], [0.65, 0], [1, 0]],
  ),
  turbo: [
    [0, 0.19, 0.072, 0.232],
    [0.125, 0.276, 0.421, 0.891],
    [0.25, 0.158, 0.736, 0.923],
    [0.375, 0.197, 0.949, 0.595],
    [0.5, 0.644, 0.99, 0.234],
    [0.625, 0.933, 0.812, 0.227],
    [0.75, 0.984, 0.493, 0.128],
    [0.875, 0.816, 0.185, 0.018],
    [1, 0.48, 0.016, 0.011],
  ],
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
  plasma: [
    [0, 0.05, 0.03, 0.528],
    [0.125, 0.3, 0.01, 0.632],
    [0.25, 0.495, 0.012, 0.658],
    [0.375, 0.665, 0.139, 0.586],
    [0.5, 0.798, 0.28, 0.47],
    [0.625, 0.902, 0.425, 0.36],
    [0.75, 0.973, 0.586, 0.252],
    [0.875, 0.993, 0.772, 0.155],
    [1, 0.94, 0.975, 0.131],
  ],
  inferno: [
    [0, 0.001, 0, 0.014],
    [0.125, 0.129, 0.047, 0.291],
    [0.25, 0.342, 0.062, 0.429],
    [0.375, 0.541, 0.135, 0.415],
    [0.5, 0.736, 0.216, 0.33],
    [0.625, 0.894, 0.353, 0.194],
    [0.75, 0.978, 0.558, 0.035],
    [0.875, 0.975, 0.798, 0.206],
    [1, 0.988, 0.998, 0.645],
  ],
  magma: [
    [0, 0.001, 0, 0.014],
    [0.125, 0.113, 0.065, 0.277],
    [0.25, 0.317, 0.072, 0.485],
    [0.375, 0.513, 0.148, 0.508],
    [0.5, 0.716, 0.215, 0.475],
    [0.625, 0.904, 0.32, 0.388],
    [0.75, 0.987, 0.536, 0.382],
    [0.875, 0.997, 0.77, 0.535],
    [1, 0.987, 0.991, 0.75],
  ],
  coolwarm: [
    [0, 0.23, 0.299, 0.754],
    [0.25, 0.552, 0.69, 0.996],
    [0.5, 0.865, 0.865, 0.865],
    [0.75, 0.958, 0.603, 0.482],
    [1, 0.706, 0.016, 0.15],
  ],
  RdBu_r: [
    [0, 0.02, 0.188, 0.38],
    [0.1, 0.129, 0.4, 0.675],
    [0.2, 0.263, 0.576, 0.765],
    [0.3, 0.573, 0.773, 0.871],
    [0.4, 0.82, 0.898, 0.941],
    [0.5, 0.969, 0.969, 0.969],
    [0.6, 0.992, 0.859, 0.78],
    [0.7, 0.957, 0.647, 0.51],
    [0.8, 0.839, 0.376, 0.302],
    [0.9, 0.698, 0.094, 0.169],
    [1, 0.404, 0, 0.122],
  ],
  RdBu: [
    [0, 0.404, 0, 0.122],
    [0.1, 0.698, 0.094, 0.169],
    [0.2, 0.839, 0.376, 0.302],
    [0.3, 0.957, 0.647, 0.51],
    [0.4, 0.992, 0.859, 0.78],
    [0.5, 0.969, 0.969, 0.969],
    [0.6, 0.82, 0.898, 0.941],
    [0.7, 0.573, 0.773, 0.871],
    [0.8, 0.263, 0.576, 0.765],
    [0.9, 0.129, 0.4, 0.675],
    [1, 0.02, 0.188, 0.38],
  ],
  bwr: [
    [0, 0, 0, 1],
    [0.5, 1, 1, 1],
    [1, 1, 0, 0],
  ],
  seismic: [
    [0, 0, 0, 0.3],
    [0.25, 0, 0, 1],
    [0.5, 1, 1, 1],
    [0.75, 1, 0, 0],
    [1, 0.5, 0, 0],
  ],
  BrBG: [
    [0, 0.329, 0.188, 0.02],
    [0.1, 0.549, 0.318, 0.039],
    [0.2, 0.749, 0.506, 0.176],
    [0.3, 0.875, 0.761, 0.49],
    [0.4, 0.965, 0.91, 0.765],
    [0.5, 0.961, 0.961, 0.961],
    [0.6, 0.78, 0.918, 0.898],
    [0.7, 0.502, 0.804, 0.757],
    [0.8, 0.208, 0.592, 0.561],
    [0.9, 0.004, 0.4, 0.369],
    [1, 0, 0.235, 0.188],
  ],
  Spectral: [
    [0, 0.62, 0.004, 0.259],
    [0.1, 0.835, 0.243, 0.31],
    [0.2, 0.957, 0.427, 0.263],
    [0.3, 0.992, 0.682, 0.38],
    [0.4, 0.996, 0.878, 0.545],
    [0.5, 1, 1, 0.749],
    [0.6, 0.902, 0.961, 0.596],
    [0.7, 0.671, 0.867, 0.643],
    [0.8, 0.4, 0.761, 0.647],
    [0.9, 0.196, 0.533, 0.741],
    [1, 0.369, 0.31, 0.635],
  ],
};

const cssRgb = (rgb: readonly number[]): string => `rgb(${rgb.map((c) => Math.round(c * 255)).join(",")})`;

/** A CSS gradient through a colormap's stops (the same table the mesh colours from). */
export const rampCss = (cmap: ContourCmap): string =>
  `linear-gradient(to right, ${CMAPS[cmap].map(([t, r, g, b]) => `${cssRgb([r, g, b])} ${(100 * t).toFixed(1)}%`).join(", ")})`;

/** A legend number: four significant digits, exponent form outside 1e-3..1e4. */
export function fmtValue(v: number): string {
  if (v === 0) return "0";
  const a = Math.abs(v);
  return a >= 1e-3 && a < 1e4 ? String(Number(v.toPrecision(4))) : v.toExponential(3);
}
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

// -- V4c2a: element (Gauss) fields --------------------------------------------------

/** The axis and Voigt suffixes of the canonical vocabulary (h5-schema.md "Component names and units"). */
const ELEMENT_SUFFIXES = new Set(["x", "y", "z", "xx", "yy", "zz", "xy", "yz", "xz"]);

export interface ElementComponent extends FieldComponent {
  /** the name says its axes are local (`_local_` before the axis, h5-schema.md's component naming rule) */
  local: boolean;
}

export interface ElementField {
  name: string;
  components: ElementComponent[];
  /** the gauss groups that carry the field (each colours its own cells) */
  groups: string[];
  /** the distinct Gauss-point counts of those groups, ascending */
  nGauss: number[];
  /**
   * Why the field cannot be drawn: its groups disagree on its component set
   * across element types, or a group refused a component (a partial write, a
   * group with no Gauss point). A component a group lacks is never taken
   * from the others. Null when it can.
   */
  refused: string | null;
}

/**
 * `<field>_<suffix>` with a vocabulary suffix, `<field>_` (an unnamed component), or a scalar of its own.
 * #2187: a `.ladruno` Gauss component `<RESULT>/<label>` is the component `<label>` of the field `<RESULT>`,
 * as `nodalFields` groups a nodal one. A native dataset name never holds a `/`, so its fields do not change.
 */
function splitName(n: string): { field: string; suffix: string | null } {
  const slash = /^([^/]+)\/(.+)$/.exec(n);
  if (slash) return { field: slash[1]!, suffix: slash[2]! };
  const m = /^(.+)_([^_]*)$/.exec(n);
  if (m && ELEMENT_SUFFIXES.has(m[2]!)) return { field: m[1]!, suffix: m[2]! };
  if (m && m[2] === "") return { field: m[1]!, suffix: "" };
  return { field: n, suffix: null };
}

/**
 * The element fields of a stage's gauss groups, in first-seen order, with
 * component names taken from the file (`[k]` for an unnamed one). One field
 * across several groups (quads and triangles) is one field when every group
 * that carries it has the same component set, none refused; otherwise it is
 * listed with `refused` and never drawn.
 */
export function gaussFields(groups: readonly GaussGroup[]): ElementField[] {
  const order: string[] = [];
  const byField = new Map<string, { names: string[]; groups: string[] }>();
  for (const g of groups) {
    for (const n of [...g.components, ...g.refused.map((r) => r.component)]) {
      const { field } = splitName(n);
      let f = byField.get(field);
      if (!f) {
        f = { names: [], groups: [] };
        byField.set(field, f);
        order.push(field);
      }
      if (!f.names.includes(n)) f.names.push(n);
      if (!f.groups.includes(g.id)) f.groups.push(g.id);
    }
  }
  return order.map((name) => {
    const f = byField.get(name)!;
    const problems: string[] = [];
    for (const gid of f.groups) {
      const g = groups.find((x) => x.id === gid)!;
      for (const n of f.names) {
        if (g.components.includes(n)) continue;
        const r = g.refused.find((x) => x.component === n);
        problems.push(r ? `group ${gid} refused ${n} (${r.reason})` : `group ${gid} has no ${n}`);
      }
    }
    let k = 0;
    const components: ElementComponent[] = f.names.map((n) => {
      const { suffix } = splitName(n);
      const local = /(^|_)local(_|$)/.test(name);
      const base = suffix === null ? n : suffix === "" ? `[${k}]` : suffix;
      k++;
      return { key: n, label: local ? `${base} (local)` : base, datasets: [n], magnitude: false, local };
    });
    const nGauss = [...new Set(f.groups.map((gid) => groups.find((x) => x.id === gid)!.nGauss))].sort((a, b) => a - b);
    const refused = problems.length ? `its groups disagree on its components: ${problems.join("; ")}` : null;
    return { name, components, groups: f.groups, nGauss, refused };
  });
}

/** The element field and component of a stage that a pick names, or null when the stage does not have it. */
export function elementComponentOf(stage: ResultsStage, field: string, component: string): { field: ElementField; component: ElementComponent } | null {
  const f = gaussFields(stage.gauss).find((x) => x.name === field);
  const c = f?.components.find((x) => x.key === component);
  return f && c ? { field: f, component: c } : null;
}

/**
 * The topology an element field is drawn with, from the mode and its
 * Gauss-point counts. `gauss_cell`, `gauss_cell_averaged`, `gauss_node` and
 * `gauss_node_discrete` are the Qt `ContourDiagram` names (`_contour.py`);
 * `element_mean` (the flat mean of an element's several Gauss points) and
 * `element_mean_averaged` (those means averaged at the nodes) are app
 * presentations with no Qt equivalent, so they carry no Qt name.
 */
export type ElementTopology = FlatTopology | NodeTopology;

/** The flat-per-cell topologies (V4c2a); the node-located ones are state/extrapolation.ts (V4c2b). */
export type FlatTopology = "gauss_cell" | "element_mean";

export const isNodeTopology = (t: ElementTopology): t is NodeTopology => t !== "gauss_cell" && t !== "element_mean";

export function elementTopology(mode: GaussMode, nGauss: readonly number[]): ElementTopology {
  switch (mode) {
    case "cell":
      return nGauss.every((g) => g === 1) ? "gauss_cell" : "element_mean";
    case "cell_averaged":
      return nGauss.every((g) => g === 1) ? "gauss_cell_averaged" : "element_mean_averaged";
    case "node":
      return "gauss_node";
    case "node_discrete":
      return "gauss_node_discrete";
    default: {
      const unknown: never = mode;
      throw new Error(`unknown gauss mode ${JSON.stringify(unknown)}`);
    }
  }
}

/** What the legend states for a topology (the averaging, in words). */
export function topologyLabel(t: ElementTopology, nGauss: readonly number[]): string {
  switch (t) {
    case "gauss_cell":
      return "gauss_cell: each element's one Gauss-point value, flat per cell";
    case "element_mean":
      return `element mean of ${nGauss.join(" or ")} Gauss points, flat per cell`;
    case "gauss_cell_averaged":
      return "gauss_cell_averaged: each element's one Gauss-point value, averaged at the nodes across elements";
    case "element_mean_averaged":
      return `element mean of ${nGauss.join(" or ")} Gauss points, averaged at the nodes across elements`;
    case "gauss_node":
      return `gauss_node: ${nGauss.join(" or ")} Gauss points extrapolated to the corners (least squares), averaged at the nodes across elements`;
    case "gauss_node_discrete":
      return `gauss_node_discrete: ${nGauss.join(" or ")} Gauss points extrapolated to each element's corners, not averaged (jumps between elements show)`;
    default: {
      const unknown: never = t;
      throw new Error(`unknown element topology ${JSON.stringify(unknown)}`);
    }
  }
}

/**
 * One value per element from a gauss block, (E * G,) element-major: the
 * element's one value (G == 1), else the mean of its G values. An element
 * with any non-finite Gauss value is NaN ("no data"): a mean of the points
 * that survive would be a guessed value.
 */
export function gaussCellValues(values: Float64Array, E: number, G: number): Float64Array {
  if (!Number.isInteger(G) || G < 1) throw new RangeError(`gaussCellValues: ${G} Gauss points per element`);
  if (values.length !== E * G) throw new RangeError(`gaussCellValues: ${values.length} values for ${E} elements x ${G} Gauss points`);
  const out = new Float64Array(E);
  for (let e = 0; e < E; e++) {
    let sum = 0;
    let ok = true;
    for (let g = 0; g < G; g++) {
      const v = values[e * G + g]!;
      if (!Number.isFinite(v)) {
        ok = false;
        break;
      }
      sum += v;
    }
    out[e] = ok ? sum / G : NaN;
  }
  return out;
}

/** The finite min and max over several value arrays; null when none is finite. */
export function finiteRangeOf(arrays: readonly Float64Array[]): { min: number; max: number } | null {
  let lo = Infinity;
  let hi = -Infinity;
  for (const a of arrays) {
    for (let i = 0; i < a.length; i++) {
      const v = a[i]!;
      if (!Number.isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  return lo <= hi ? { min: lo, max: hi } : null;
}

/** The decl path of a model cell (state/decls.ts `cellPath`, restated: this module imports no loader). */
const cellDecl = (femId: number): DeclPath => `mesh/element/${femId}`;

/** The FEM id of every drawn element (`MeshInfo.elements`); NaN for an OpenSees-only row, which has no cell and no element row. */
export function meshFemIds(elements: readonly DeclPath[], decls: Readonly<Record<DeclPath, Decl>>): Float64Array {
  const out = new Float64Array(elements.length);
  elements.forEach((p, i) => {
    const id = decls[p]?.element?.femId;
    out[i] = typeof id === "number" ? id : NaN;
  });
  return out;
}

export type ElementJoin =
  | { ok: true; rows: number; at: Map<number, { group: number; row: number }> }
  | { ok: false; code: "unknown-id" | "duplicate-id"; reason: string };

/**
 * The join of element rows to the model's cells, by FEM id only (never by
 * position or row order): row `e` of group `g` is the cell whose FEM id is
 * `eids[e]`. An id the model has no cell for, or an id listed twice (in one
 * group or two), refuses the join.
 */
export function joinElements(decls: Readonly<Record<DeclPath, Decl>>, groups: readonly { id: string; eids: Float64Array }[]): ElementJoin {
  const at = new Map<number, { group: number; row: number }>();
  const unknown: number[] = [];
  let nUnknown = 0;
  let rows = 0;
  for (let gi = 0; gi < groups.length; gi++) {
    const g = groups[gi]!;
    for (let e = 0; e < g.eids.length; e++) {
      const id = g.eids[e]!;
      rows++;
      if (decls[cellDecl(id)]?.element?.femId !== id) {
        nUnknown++;
        if (unknown.length < 5) unknown.push(id);
        continue;
      }
      const prev = at.get(id);
      if (prev) {
        return { ok: false, code: "duplicate-id", reason: `element ${id} has two rows (group ${groups[prev.group]!.id} row ${prev.row}, group ${g.id} row ${e}): its value would be a guess` };
      }
      at.set(id, { group: gi, row: e });
    }
  }
  if (nUnknown) {
    const more = nUnknown > unknown.length ? ", ..." : "";
    return {
      ok: false,
      code: "unknown-id",
      reason: `_element_index names ${nUnknown} element id${nUnknown === 1 ? "" : "s"} the model has no cell for (${unknown.join(", ")}${more}): the rows cannot be joined to the mesh`,
    };
  }
  return { ok: true, rows, at };
}

/** The value of every drawn element (`MeshInfo.elements` order): its own row's value, or NaN ("no data") when no group has it. */
export function elementCellValues(meshFem: Float64Array, at: ReadonlyMap<number, { group: number; row: number }>, perGroup: readonly Float64Array[]): Float64Array {
  const out = new Float64Array(meshFem.length);
  for (let i = 0; i < meshFem.length; i++) {
    const hit = Number.isNaN(meshFem[i]!) ? undefined : at.get(meshFem[i]!);
    out[i] = hit ? perGroup[hit.group]![hit.row]! : NaN;
  }
  return out;
}

/** Per-vertex values of the primitives (`primElement`: the element of each primitive), every vertex its element's value: flat per cell. */
export function cellVertexValues(elementValues: Float64Array, primElement: Int32Array, perPrim: number): Float64Array {
  const out = new Float64Array(primElement.length * perPrim);
  for (let p = 0; p < primElement.length; p++) out.fill(elementValues[primElement[p]!]!, p * perPrim, (p + 1) * perPrim);
  return out;
}
