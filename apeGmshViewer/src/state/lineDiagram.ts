// Pure state, orientation, geometry, hover-needle, and multi-beam chain helpers
// for 1-D beam internal-force and section-deformation diagrams (Escalón 2).
//
// Pure: no three.js and no DOM, so every function runs in Node tests.
// Parity targets:
// - `src/apeGmsh/viewers/diagrams/_beam_geometry.py` (`compute_beam_local_axes`,
//   `default_vecxz`, `build_beam_diagram_geometry`),
// - `src/apeGmsh/viewers/diagrams/_line_force.py` (`LineForceDiagram`),
// - `src/apeGmsh/opensees/_response_catalog.py` (`local_force_station_signs`).

import { ELEMENT_SYNTAX } from "../chain/signatures.ts";
import type { Param } from "../model/types.ts";
import {
  LINE_STATION_COMPONENTS,
  normalizeLineStationComponent,
  quaternionToZRec,
  type LadrunoLineStationStep,
  type LineStationComponent,
} from "../reader/ladruno.ts";
import { DARK, hexToRgb, themeTokens, type ThemeMode } from "../theme/tokens.ts";
import { CMAP_NAMES, climOf, rampT, sampleCmap, type Clim } from "./contour.ts";
import {
  isBeamForceComponent,
  joinLadrunoElements,
  reconstructElementSpanDiagram,
  reconstructSpanDiagramsForStep,
  type BeamClosureCheckOptions,
  type LadrunoElementJoin,
} from "./spanLoads.ts";
import type {
  BeamChain,
  BeamChainJoint,
  BeamChainSpan,
  BeamChainStationPoint,
  ContourCmap,
  ContourRange,
  ContourScan,
  Decl,
  DeclPath,
  Definitions,
  LineDiagramBeamSummary,
  LineDiagramCmap,
  LineDiagramHoverReadout,
  LineDiagramPlane,
  LineDiagramState,
  ResultsStage,
  SequenceZone,
  State,
} from "./types.ts";

export {
  isBeamForceComponent,
  joinLadrunoElements,
  reconstructElementSpanDiagram,
  reconstructSpanDiagramsForStep,
  type BeamClosureCheckOptions,
  type LadrunoElementJoin,
};

export {
  LINE_STATION_COMPONENTS,
  normalizeLineStationComponent,
  quaternionToZRec,
  type BeamChain,
  type BeamChainJoint,
  type BeamChainSpan,
  type BeamChainStationPoint,
  type LineDiagramBeamSummary,
  type LineDiagramCmap,
  type LineDiagramHoverReadout,
  type LineDiagramPlane,
  type LineDiagramState,
  type LineStationComponent,
};

export type LineDiagramFieldName = "section_force" | "section_deformation";

export const LINE_DIAGRAM_CMAP_NAMES: readonly LineDiagramCmap[] = ["sign", ...CMAP_NAMES];

export const LINE_DIAGRAM_PREFERRED_COMPONENTS: readonly LineStationComponent[] = [
  "Mz",
  "My",
  "Vy",
  "Vz",
  "N",
  "T",
  "kappaZ",
  "kappaY",
  "eps",
  "gammaY",
  "gammaZ",
  "theta",
];

export interface LineDiagramComponentMeta {
  readonly key: LineStationComponent;
  readonly canonical: string;
  readonly label: string;
  readonly field: LineDiagramFieldName;
  /** Local beam axis along which the diagram is extruded when `plane === "local"`. */
  readonly axis: "y" | "z";
  /** Default tension-side sign convention (`true` for bending moments `My, Mz` and curvatures `kappaY, kappaZ`, `false` otherwise). */
  readonly defaultFlipSign: boolean;
  /** Whether a 3-D beam requires an explicit local frame (`LOCAL_AXES` or `geomTransf.vecxz`) when `plane === "local"`. */
  readonly requires3dFrame: boolean;
}

export interface LineDiagramField {
  readonly name: LineDiagramFieldName;
  readonly label: string;
  readonly components: readonly LineDiagramComponentMeta[];
}

export const LINE_DIAGRAM_COMPONENT_META: Readonly<Record<LineStationComponent, LineDiagramComponentMeta>> = {
  N: {
    key: "N",
    canonical: "axial_force",
    label: "N (axial)",
    field: "section_force",
    axis: "z",
    defaultFlipSign: false,
    requires3dFrame: false,
  },
  Vy: {
    key: "Vy",
    canonical: "shear_y",
    label: "Vy",
    field: "section_force",
    axis: "y",
    defaultFlipSign: false,
    requires3dFrame: true,
  },
  Vz: {
    key: "Vz",
    canonical: "shear_z",
    label: "Vz",
    field: "section_force",
    axis: "z",
    defaultFlipSign: false,
    requires3dFrame: true,
  },
  T: {
    key: "T",
    canonical: "torsion",
    label: "T",
    field: "section_force",
    axis: "y",
    defaultFlipSign: false,
    requires3dFrame: false,
  },
  My: {
    key: "My",
    canonical: "bending_moment_y",
    label: "My",
    field: "section_force",
    axis: "z",
    defaultFlipSign: true,
    requires3dFrame: true,
  },
  Mz: {
    key: "Mz",
    canonical: "bending_moment_z",
    label: "Mz",
    field: "section_force",
    axis: "y",
    defaultFlipSign: true,
    requires3dFrame: true,
  },
  eps: {
    key: "eps",
    canonical: "axial_strain",
    label: "eps",
    field: "section_deformation",
    axis: "z",
    defaultFlipSign: false,
    requires3dFrame: false,
  },
  gammaY: {
    key: "gammaY",
    canonical: "shear_strain_y",
    label: "gammaY",
    field: "section_deformation",
    axis: "y",
    defaultFlipSign: false,
    requires3dFrame: true,
  },
  gammaZ: {
    key: "gammaZ",
    canonical: "shear_strain_z",
    label: "gammaZ",
    field: "section_deformation",
    axis: "z",
    defaultFlipSign: false,
    requires3dFrame: true,
  },
  theta: {
    key: "theta",
    canonical: "torsional_strain",
    label: "theta",
    field: "section_deformation",
    axis: "y",
    defaultFlipSign: false,
    requires3dFrame: false,
  },
  kappaY: {
    key: "kappaY",
    canonical: "curvature_y",
    label: "kappaY",
    field: "section_deformation",
    axis: "z",
    defaultFlipSign: true,
    requires3dFrame: true,
  },
  kappaZ: {
    key: "kappaZ",
    canonical: "curvature_z",
    label: "kappaZ",
    field: "section_deformation",
    axis: "y",
    defaultFlipSign: true,
    requires3dFrame: true,
  },
};

export const LINE_DIAGRAM_FIELDS: readonly LineDiagramField[] = [
  {
    name: "section_force",
    label: "section_force",
    components: [
      LINE_DIAGRAM_COMPONENT_META.N,
      LINE_DIAGRAM_COMPONENT_META.Vy,
      LINE_DIAGRAM_COMPONENT_META.Vz,
      LINE_DIAGRAM_COMPONENT_META.T,
      LINE_DIAGRAM_COMPONENT_META.My,
      LINE_DIAGRAM_COMPONENT_META.Mz,
    ],
  },
  {
    name: "section_deformation",
    label: "section_deformation",
    components: [
      LINE_DIAGRAM_COMPONENT_META.eps,
      LINE_DIAGRAM_COMPONENT_META.gammaY,
      LINE_DIAGRAM_COMPONENT_META.gammaZ,
      LINE_DIAGRAM_COMPONENT_META.theta,
      LINE_DIAGRAM_COMPONENT_META.kappaY,
      LINE_DIAGRAM_COMPONENT_META.kappaZ,
    ],
  },
];

export const initialLineDiagramState: LineDiagramState = {
  on: false,
  component: null,
  scale: 1,
  plane: "local",
  cmap: "sign",
  fill: true,
  flipSign: null,
  range: "history",
  fixed: null,
  stepRange: null,
  scan: null,
  notice: null,
  drawnBeams: [],
  chains: [],
  hover: null,
};

/** Default auto-picked component for a stage's `lineStations` (`Mz, My, Vy, Vz, N, T, ...`), or null when none is available. */
export function defaultLineDiagramComponent(available?: readonly string[]): LineStationComponent | null {
  if (!available || available.length === 0) return null;
  const set = new Set<LineStationComponent>();
  for (const raw of available) {
    const c = normalizeLineStationComponent(raw);
    if (c) set.add(c);
  }
  for (const pref of LINE_DIAGRAM_PREFERRED_COMPONENTS) {
    if (set.has(pref)) return pref;
  }
  return null;
}

/** Filter `LINE_DIAGRAM_FIELDS` to the components available in a `.ladruno` stage. */
export function lineDiagramFieldsOf(available: readonly string[]): LineDiagramField[] {
  const set = new Set<LineStationComponent>();
  for (const raw of available) {
    const c = normalizeLineStationComponent(raw);
    if (c) set.add(c);
  }
  const out: LineDiagramField[] = [];
  for (const f of LINE_DIAGRAM_FIELDS) {
    const comps = f.components.filter((c) => set.has(c.key));
    if (comps.length > 0) out.push({ ...f, components: comps });
  }
  return out;
}

/** Default `flipSign` for `component`: `true` for bending moments (`My, Mz`) and curvatures (`kappaY, kappaZ`), `false` otherwise. */
export function defaultFlipSignFor(component: string): boolean {
  const c = normalizeLineStationComponent(component);
  return c ? LINE_DIAGRAM_COMPONENT_META[c].defaultFlipSign : false;
}

/** Effective boolean `flipSign` from a component and a nullable user override (`null` uses component default), or from `LineDiagramState`. */
export function effectiveFlipSign(
  componentOrState: string | null | { component: string | null; flipSign: boolean | null },
  flipSign?: boolean | null,
): boolean {
  if (componentOrState !== null && typeof componentOrState === "object") {
    return effectiveFlipSign(componentOrState.component, componentOrState.flipSign);
  }
  if (typeof flipSign === "boolean") return flipSign;
  return componentOrState ? defaultFlipSignFor(componentOrState) : false;
}

/** Step and history-scan keys for a line-diagram component on a results stage. */
export function lineDiagramKeys(
  path: string,
  stageId: string,
  step: number,
  component: LineStationComponent,
): { step: string; scan: string } {
  const scan = `${path}|${stageId}|line:${component}`;
  return { scan, step: `${scan}|${step}` };
}

/** Current state's line-diagram step and scan keys, or null when not on a results step with a picked component. */
export function lineDiagramKeysOf(s: State): { step: string; scan: string } | null {
  const at = s.phase.at;
  const ld = s.lineDiagram;
  if (at?.kind !== "results" || !s.results || ld.component === null) return null;
  const st = s.results.stages[at.stage];
  if (!st || !st.lineStations?.includes(ld.component)) return null;
  return lineDiagramKeys(s.results.path, st.id, at.step, ld.component);
}

/** Current state's line-diagram colour range (`Clim`). */
export function lineDiagramClimOf(s: State): Clim {
  const keys = lineDiagramKeysOf(s);
  const ld = s.lineDiagram;
  if (!keys) return climOf(ld, null, null, "");
  const stepRange = ld.stepRange !== null && ld.stepRange.key === keys.step ? ld.stepRange : null;
  const clim = climOf(ld, stepRange, ld.scan, keys.scan);
  if (
    clim.mode === "history" &&
    stepRange !== null &&
    clim.min !== null &&
    clim.max !== null &&
    stepRange.min !== null &&
    stepRange.max !== null &&
    Number.isFinite(stepRange.min) &&
    Number.isFinite(stepRange.max)
  ) {
    return { ...clim, min: Math.min(clim.min, stepRange.min), max: Math.max(clim.max, stepRange.max) };
  }
  return clim;
}

/** Local beam axis (`"y"` or `"z"`) along which `component` is extruded when `plane === "local"`. */
export function localAxisForComponent(component: string): "y" | "z" {
  const c = normalizeLineStationComponent(component);
  if (!c) throw new RangeError(`unknown line-diagram component ${JSON.stringify(component)}`);
  return LINE_DIAGRAM_COMPONENT_META[c].axis;
}

/** Whether `component` requires an explicit 3-D beam frame (`LOCAL_AXES` or `geomTransf.vecxz`) on a 3-D model when `plane === "local"`. */
export function requires3dFrame(component: string): boolean {
  const c = normalizeLineStationComponent(component);
  if (!c) throw new RangeError(`unknown line-diagram component ${JSON.stringify(component)}`);
  return LINE_DIAGRAM_COMPONENT_META[c].requires3dFrame;
}

// -- local axes & orientation math --------------------------------------------

const VERTICAL_COS_THRESHOLD = 0.99;

/**
 * Default OpenSees-style `vecxz` reference vector for a unit beam chord `xLocal`
 * (`src/apeGmsh/viewers/diagrams/_beam_geometry.py::default_vecxz`):
 * - non-vertical (`|dot(xLocal, +Z)| <= 0.99`): `[0, 0, 1]`
 * - vertical (`|dot(xLocal, +Z)| > 0.99`): `[1, 0, 0]`
 */
export function defaultVecxz(xLocal: readonly [number, number, number]): [number, number, number] {
  return Math.abs(xLocal[2]) > VERTICAL_COS_THRESHOLD ? [1, 0, 0] : [0, 0, 1];
}

/**
 * Compute right-handed local axes `(xLocal, yLocal, zLocal)` from unit chord
 * `xLocal` and reference vector `vecxz` via Gram-Schmidt orthogonalization
 * (`src/apeGmsh/viewers/diagrams/_beam_geometry.py::compute_beam_local_axes`).
 *
 * `zLocal = normalize(vecxz - (vecxz . xLocal) xLocal)`
 * `yLocal = normalize(zLocal x xLocal)`
 *
 * Returns `null` when the frame is degenerate (`vecxz` parallel to the chord,
 * zero or non-finite). The caller decides whether a substitute frame is
 * acceptable: a 3-D bending frame is never guessed (#2473).
 */
export function beamLocalAxes(
  xLocal: readonly [number, number, number],
  vecxz: readonly [number, number, number],
): {
  xLocal: [number, number, number];
  yLocal: [number, number, number];
  zLocal: [number, number, number];
} | null {
  const [xx, xy, xz] = xLocal;
  const [vx, vy, vz] = vecxz;
  const dot = vx * xx + vy * xy + vz * xz;
  const zx = vx - dot * xx;
  const zy = vy - dot * xy;
  const zz = vz - dot * xz;
  const zNorm = Math.hypot(zx, zy, zz);
  if (!Number.isFinite(zNorm) || zNorm < 1e-6) return null;
  const zLocal: [number, number, number] = [zx / zNorm, zy / zNorm, zz / zNorm];
  // yLocal = zLocal x xLocal
  const yx = zLocal[1] * xz - zLocal[2] * xy;
  const yy = zLocal[2] * xx - zLocal[0] * xz;
  const yz = zLocal[0] * xy - zLocal[1] * xx;
  const yNorm = Math.hypot(yx, yy, yz) || 1;
  const yLocal: [number, number, number] = [yx / yNorm, yy / yNorm, yz / yNorm];
  return { xLocal: [xx, xy, xz], yLocal, zLocal };
}

/**
 * Screen-plane extrusion direction perpendicular to `xLocal` and `cameraViewDir`:
 * `d = normalize(xLocal x cameraViewDir)`.
 */
export function screenExtrusionDirection(
  xLocal: readonly [number, number, number],
  cameraViewDir: readonly [number, number, number] = [0, 0, -1],
): [number, number, number] {
  const [xx, xy, xz] = xLocal;
  const [vx, vy, vz] = cameraViewDir;
  let px = xy * vz - xz * vy;
  let py = xz * vx - xx * vz;
  let pz = xx * vy - xy * vx;
  let norm = Math.hypot(px, py, pz);
  if (!Number.isFinite(norm) || norm < 1e-6) {
    const fb = defaultVecxz(xLocal);
    px = xy * fb[2] - xz * fb[1];
    py = xz * fb[0] - xx * fb[2];
    pz = xx * fb[1] - xy * fb[0];
    norm = Math.hypot(px, py, pz) || 1;
  }
  return [px / norm, py / norm, pz / norm];
}

// -- model.h5 transform vecxz extraction from State.decls ---------------------

const BRACKET_TRIPLE_RE = /\[\s*([^\[\],;\s]+)\s*,\s*([^\[\],;\s]+)\s*,\s*([^\[\],;\s]+)\s*\]/g;
const BRACKET_NUMS_RE = /[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?/g;

/**
 * Extract `(emittedTag, vecxz)` pairs from a `geomTransf` `Decl` in `State.decls`
 * (`decl.vecxz`, or `per_element_vecxz` and `per_element_emitted_tag` fields, or 3-D `params`).
 * Returns `[]` for a 2-D transform (`per_element_vecxz` is `"empty (no values)"`).
 */
export function declTransformVecxz(decl: Decl): { tag: number; vecxz: [number, number, number] }[] {
  if (decl.kind !== "geomTransf") return [];
  if (decl.vecxz !== undefined) {
    return decl.vecxz.map((p) => ({
      tag: p.tag,
      vecxz: [p.vecxz[0], p.vecxz[1], p.vecxz[2]],
    }));
  }
  const vecField = decl.fields.find((f) => f.label === "per_element_vecxz");
  if (vecField !== undefined) {
    if (vecField.value.startsWith("empty")) return [];
    const triples: [number, number, number][] = [];
    for (const m of vecField.value.matchAll(BRACKET_TRIPLE_RE)) {
      const vx = Number(m[1]);
      const vy = Number(m[2]);
      const vz = Number(m[3]);
      if (Number.isFinite(vx) && Number.isFinite(vy) && Number.isFinite(vz)) {
        triples.push([vx, vy, vz]);
      }
    }
    if (triples.length === 0) return [];
    const tagField = decl.fields.find((f) => f.label === "per_element_emitted_tag");
    const tags: number[] = [];
    if (tagField && !tagField.value.startsWith("empty")) {
      for (const m of tagField.value.matchAll(BRACKET_NUMS_RE)) {
        const t = Number(m[0]);
        if (Number.isInteger(t)) tags.push(t);
      }
    }
    if (tags.length === 0 && decl.tag !== null) tags.push(decl.tag);
    const n = Math.min(triples.length, tags.length);
    const out: { tag: number; vecxz: [number, number, number] }[] = [];
    for (let i = 0; i < n; i++) {
      out.push({ tag: tags[i]!, vecxz: triples[i]! });
    }
    return out;
  }
  if (
    (decl.params.length === 3 || decl.params.length === 9) &&
    typeof decl.params[0] === "number" &&
    typeof decl.params[1] === "number" &&
    typeof decl.params[2] === "number" &&
    Number.isFinite(decl.params[0]) &&
    Number.isFinite(decl.params[1]) &&
    Number.isFinite(decl.params[2]) &&
    decl.tag !== null
  ) {
    return [{ tag: decl.tag, vecxz: [decl.params[0], decl.params[1], decl.params[2]] }];
  }
  return [];
}

/**
 * Extract candidate `geomTransf` integer tags from an element declaration's
 * `metas`: when `ELEMENT_SYNTAX[m.type]` decodes the element, only the decoded
 * `geomTransf` slot is returned (preventing `secTag` or `matTag` from colliding
 * with a transform tag). An element type not in `ELEMENT_SYNTAX`, or one whose
 * args the decoder refuses, yields no candidate: any integer arg (a node tag, a
 * material tag) could equal a transform tag, so none is guessed (#2473).
 */
function candidateElementTransfTags(elementDecl: Decl): number[] {
  const out: number[] = [];
  for (const m of elementDecl.element?.metas ?? []) {
    const decoder = ELEMENT_SYNTAX[m.type];
    if (decoder) {
      const dec = decoder(m.args as Param[]);
      if (!("reason" in dec)) {
        for (const s of dec.slots) {
          if (s.family === "geomTransf") {
            const tag = m.args[s.slot];
            if (typeof tag === "number" && Number.isInteger(tag)) out.push(tag);
          }
        }
      }
    }
  }
  return out;
}

/**
 * Resolve a 1-D element declaration's `vecxz` `(3,)` from `State.decls` (via its
 * `transfTag` reference or emitted transform tag). Returns `null` when the
 * element has no 3-D transform `vecxz`.
 */
export function elementTransformVecxzFromState(
  decls: Readonly<Record<DeclPath, Decl>>,
  elementDecl: Decl,
  explicitByTag?: ReadonlyMap<number, readonly [number, number, number]>,
): [number, number, number] | null {
  const transfTags = candidateElementTransfTags(elementDecl);
  const transfPath = elementDecl.refs?.["transfTag"];
  if (transfPath) {
    const transfDecl = decls[transfPath];
    if (transfDecl) {
      const pairs = declTransformVecxz(transfDecl);
      if (pairs.length === 1) return pairs[0]!.vecxz;
      if (pairs.length > 1) {
        for (const tag of transfTags) {
          const byArg = pairs.find((p) => p.tag === tag);
          if (byArg) return byArg.vecxz;
        }
        if (transfDecl.tag !== null) {
          const hit = pairs.find((p) => p.tag === transfDecl.tag);
          if (hit) return hit.vecxz;
        }
      }
    }
  }
  if (explicitByTag) {
    for (const tag of transfTags) {
      const v = explicitByTag.get(tag);
      if (v) return [v[0], v[1], v[2]];
    }
    const transfPathDecl = transfPath ? decls[transfPath] : undefined;
    if (transfPathDecl?.tag !== null && transfPathDecl?.tag !== undefined) {
      const v = explicitByTag.get(transfPathDecl.tag);
      if (v) return [v[0], v[1], v[2]];
    }
  }
  if (!transfPath && transfTags.length > 0) {
    // Fan-out `geomTransf` models carry no `refs.transfTag` (#2142): match the
    // decoded transform tag across every transform, but only when exactly one
    // vecxz answers. Two transforms that claim the tag with different vecxz are
    // ambiguous, and the beam is left unframed (#2473).
    let found: [number, number, number] | null = null;
    for (const d of Object.values(decls)) {
      if (d.kind !== "geomTransf") continue;
      for (const p of declTransformVecxz(d)) {
        if (!transfTags.includes(p.tag)) continue;
        if (found && (found[0] !== p.vecxz[0] || found[1] !== p.vecxz[1] || found[2] !== p.vecxz[2])) return null;
        found = p.vecxz;
      }
    }
    return found;
  }
  return null;
}

// -- 3-D Quad-Strip, Profile & Riser Builder ----------------------------------

export type ZRecSource = "step_local_axes" | "model_local_axes" | "model_transform" | "default_vecxz" | "screen";

export interface BeamDiagramSpan3D {
  decl: DeclPath;
  elementId: number;
  segmentIndex: number;
  /** Last drawn segment index for this element (equals `segmentIndex` for 2-node elements; `> segmentIndex` for multi-segment 1-D elements). */
  endSegmentIndex?: number;
  /** Start node tag (`iNode`, at `xi = -1`), when resolved from `lineNodes` / `inlineNodes`. */
  nodeStart?: number;
  /** End node tag (`jNode`, at `xi = +1`), when resolved from `lineNodes` / `inlineNodes`. */
  nodeEnd?: number;
  pStart: [number, number, number];
  pEnd: [number, number, number];
  length: number;
  xLocal: [number, number, number];
  extrudeDir: [number, number, number];
  zRecSource: ZRecSource;
  /** Expanded station natural coordinates in [-1, +1] (at least 2 stations). */
  xi: Float64Array;
  /** Raw physical values at `xi`. */
  values: Float64Array;
  /** Whether the span curve was reconstructed analytically (`"exact"`) or station-interpolated (`"interpolated"`). */
  spanMode?: "exact" | "interpolated";
  /** Fallback reason when `spanMode === "interpolated"` (`null` when `"exact"`). */
  spanReason?: string | null;
}

export interface DiagramLabelAnchor {
  decl: DeclPath;
  elementId: number;
  value: number;
  position: [number, number, number];
}

export interface LineDiagram3DGeometry {
  component: LineStationComponent;
  /** 9 floats per triangle (3 xyz vertices). */
  triPositions: Float32Array;
  /** 3 floats per triangle (raw physical scalar per vertex). */
  triValues: Float32Array;
  /** 3 floats per triangle (normalized [0, 1] ramp position per vertex). */
  triUVs: Float32Array;
  /** 9 floats per triangle (3 sRGB floats per vertex). */
  triColors: Float32Array;
  /** Index into `beams` for each triangle. */
  triBeamIndex: Int32Array;
  /** 6 floats per top-profile line segment. */
  profilePositions: Float32Array;
  /** 2 floats per top-profile line segment (raw physical scalar at segment endpoints). */
  profileValues: Float32Array;
  /** Index into `beams` for each top-profile line segment. */
  profileBeamIndex: Int32Array;
  /** 6 floats per station riser tick segment (chord -> profile). */
  riserPositions: Float32Array;
  /** 2 floats per station riser tick segment (raw physical scalar at both endpoints). */
  riserValues: Float32Array;
  /** Index into `beams` for each station riser tick segment. */
  riserBeamIndex: Int32Array;
  /** Per-beam metadata and local axes for hover needle and inspection. */
  beams: BeamDiagramSpan3D[];
  /** Peak/station label anchors for `showLabels`. */
  labelAnchors: DiagramLabelAnchor[];
  /** Maximum absolute station value across drawn beams (`0` when empty). */
  peakAbsValue: number;
  /** Auto-scale factor (`0.15 * L_char / v_peak`). */
  autoScale: number;
  /** Effective world scale (`autoScale * userScale`). */
  effectiveScale: number;
  /** Colour limits used for vertex colouring. */
  clim: [number, number];
  /** Count of 3-D beams refused by the fail-closed local-frame gate. */
  skippedUnorientedBeams: number;
  /** Count of beams refused because element connectivity (iNode, jNode) contradicts cell corner nodes. */
  skippedOrientationMismatchBeams: number;
  /** `DeclPath`s of 3-D beams refused by the fail-closed local-frame gate. */
  skippedBeamDecls: DeclPath[];
  /** User-facing notice when beams were skipped or nothing could be drawn. */
  notice: string | null;
  /** Count of drawn beams whose span curve was reconstructed analytically (`spanMode === "exact"`). */
  exactSpanCount: number;
  /** Count of drawn beams using station interpolation (`spanMode === "interpolated"`). */
  interpolatedSpanCount: number;
  /** Distinct non-empty fallback reasons across station-interpolated beams. */
  spanReasons: readonly string[];
}

export interface BuildLineDiagram3DOptions {
  /** Current state (or minimal subset: `decls`, `mesh`, `visibility`, `ndm`). */
  decls: Readonly<Record<DeclPath, Decl>>;
  /** Optional `/opensees` model definitions used for exact span-load reconstruction (`E2b`). */
  definitions?: Definitions | null;
  /** Optional `/sequence` zone used for staged load activation during exact span-load reconstruction (`E2b`). */
  sequence?: SequenceZone | null;
  /** Optional active results stage metadata used to match stage index for exact span-load reconstruction (`E2b`). */
  resultsStage?: Pick<ResultsStage, "id" | "name" | "kind"> | null;
  /** Optional fallback stage index into `definitions.stages` when `resultsStage` is omitted. */
  stageIndex?: number;
  /** Optional step pseudo-time `t` for evaluating time series (defaults to `stepData.time`). */
  stepTime?: number;
  /** Optional map of line-station steps by component (`Vy`/`Vz` companion shear for `Mz`/`My`). */
  stepByComponent?: Readonly<Partial<Record<LineStationComponent, LadrunoLineStationStep>>>;
  /** Optional closure tolerance overrides for exact span-load reconstruction (`E2b`). */
  closureTolerance?: BeamClosureCheckOptions;
  /** Optional number of sample stations per sub-interval for exact span-load reconstruction (`E2b`). */
  samplesPerSegment?: number;
  /** `MeshInfo.elements` (decl path per drawn primitive). */
  meshElements: readonly DeclPath[];
  /** `lineElement` blob (`Int32Array` index into `meshElements` per drawn line segment). */
  lineElement: Int32Array;
  /** Current drawn line segment positions (`Float32Array`, 6 floats per segment — deformed when `deform.on` is true). */
  drawnLinePositions: Float32Array;
  /** Optional node tags per drawn line segment (2 per segment, `MeshInfo.vertexNodes.line`), used for the P0-2 element vs. cell orientation check. */
  lineNodes?: ArrayLike<number>;
  /** Optional OpenSees element connectivity `[iNode, jNode]` by `.ladruno` element ID, used for the P0-2 orientation check when `Decl.element.inlineNodes` is empty. */
  elementNodesById?: ReadonlyMap<number, readonly [number, number]>;
  /** Optional indices of visible line segments (from `renderer/visible.ts`); if omitted, all segments in `lineElement` are considered. */
  visibleLineMap?: Int32Array;
  /** `.ladruno` single-step line-station data. */
  stepData: LadrunoLineStationStep;
  /** Element-ID join from `.ladruno` IDs to `DeclPath`. */
  join: Extract<LadrunoElementJoin, { ok: true }>;
  /** Optional per-step `RESULTS/ON_ELEMENTS/LOCAL_AXES` `z_rec` map (`elementId -> [zx, zy, zz]`). */
  stepLocalAxesZRec?: ReadonlyMap<number, readonly [number, number, number]>;
  /** Optional `MODEL/LOCAL_AXES` `z_rec` map (`elementId -> [zx, zy, zz]`). */
  modelLocalAxesZRec?: ReadonlyMap<number, readonly [number, number, number]>;
  /** Optional `model.h5` transform `vecxz` map by emitted tag (`tag -> [vx, vy, vz]`). */
  modelTransformVecxzByTag?: ReadonlyMap<number, readonly [number, number, number]>;
  /** Spatial dimension (`2` or `3`; `null` is treated fail-closed as 3-D). */
  ndm: 2 | 3 | null;
  /** Optional characteristic length (`2 * mesh.radius` in the 3-D layer; defaults to median beam length). */
  charLength?: number;
  /** Whether `drawnLinePositions` is a deformed baseline (`s.deform.on === true`). */
  deformed?: boolean;
  /** Diagram display options. */
  scale?: number;
  flipSign?: boolean | null;
  cmap?: LineDiagramCmap;
  plane?: LineDiagramPlane;
  cameraViewDir?: readonly [number, number, number];
  clim?: readonly [number, number] | null;
  theme?: ThemeMode;
}

/**
 * Deterministic digest of `visibleLineMap` so cache keys in `Effects.syncLineDiagram`,
 * `LineDiagramLayer.sync`, and `buildLineDiagram3D` reflect `hidden`, `isolate`, and
 * stage element activation (`elementMaskOf`).
 */
export function visibleLineDigest(visibleLineMap: Int32Array | undefined, totalSegments?: number): string {
  if (!visibleLineMap) return totalSegments !== undefined ? `all:${totalSegments}` : "all";
  const n = visibleLineMap.length;
  if (totalSegments !== undefined && n === totalSegments) {
    let identity = true;
    for (let i = 0; i < n; i++) {
      if (visibleLineMap[i] !== i) {
        identity = false;
        break;
      }
    }
    if (identity) return `all:${totalSegments}`;
  }
  let h1 = 0x811c9dc5;
  let h2 = 0x01000193;
  for (let i = 0; i < n; i++) {
    const v = visibleLineMap[i]! | 0;
    h1 = Math.imul(h1 ^ v, 0x01000193);
    h2 = Math.imul(h2 ^ (v + i), 0x811c9dc5);
  }
  return `${n}:${(h1 >>> 0).toString(16)}:${(h2 >>> 0).toString(16)}`;
}

function arrayDigest(arr: ArrayLike<number> | undefined | null): string {
  if (!arr) return "null";
  const n = arr.length;
  let h1 = 0x811c9dc5 ^ n;
  let h2 = 0x01000193 ^ n;
  if (ArrayBuffer.isView(arr) && arr.byteLength >= 4) {
    const u32 = new Uint32Array(arr.buffer, arr.byteOffset, Math.floor(arr.byteLength / 4));
    const step = u32.length > 16384 ? Math.floor(u32.length / 8192) : 1;
    for (let i = 0; i < u32.length; i += step) {
      const v = u32[i]! | 0;
      h1 = Math.imul(h1 ^ v, 0x01000193);
      h2 = Math.imul(h2 ^ (v + i), 0x811c9dc5);
    }
  } else {
    for (let i = 0; i < n; i++) {
      const v = Number(arr[i]) || 0;
      const bits = Math.imul((v * 1e6) | 0, 2654435761);
      h1 = Math.imul(h1 ^ bits, 0x01000193);
      h2 = Math.imul(h2 ^ (bits + i), 0x811c9dc5);
    }
  }
  return `${n}:${(h1 >>> 0).toString(16)}:${(h2 >>> 0).toString(16)}`;
}

const objIds = new WeakMap<object, number>();
let nextObjId = 1;

export function stateObjectDigest(...objs: readonly (object | null | undefined)[]): string {
  return objs
    .map((o) => {
      if (!o) return "0";
      let id = objIds.get(o);
      if (id === undefined) {
        id = nextObjId++;
        objIds.set(o, id);
      }
      return String(id);
    })
    .join(":");
}

function sameStepByComponent(
  a: Readonly<Partial<Record<LineStationComponent, LadrunoLineStationStep>>> | undefined,
  b: Readonly<Partial<Record<LineStationComponent, LadrunoLineStationStep>>> | undefined,
): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  const ka = Object.keys(a) as LineStationComponent[];
  const kb = Object.keys(b) as LineStationComponent[];
  return ka.length === kb.length && ka.every((k) => a[k] === b[k]);
}

interface LineDiagram3DCacheEntry {
  key: string;
  decls: Readonly<Record<DeclPath, Decl>>;
  definitions: Definitions | null;
  sequence: SequenceZone | null;
  stepLocalAxesZRec: ReadonlyMap<number, readonly [number, number, number]> | undefined;
  modelLocalAxesZRec: ReadonlyMap<number, readonly [number, number, number]> | undefined;
  modelTransformVecxzByTag: ReadonlyMap<number, readonly [number, number, number]> | undefined;
  elementNodesById: ReadonlyMap<number, readonly [number, number]> | undefined;
  stepByComponent: Readonly<Partial<Record<LineStationComponent, LadrunoLineStationStep>>> | undefined;
  autoClim: boolean;
  geom: LineDiagram3DGeometry;
}

let lastLineDiagram3DCache: LineDiagram3DCacheEntry | null = null;
let lineDiagram3DBuildCount = 0;

/** Total number of non-cached `buildLineDiagram3D` geometry builds executed. */
export function getLineDiagram3DBuildCount(): number {
  return lineDiagram3DBuildCount;
}

/** Reset the `buildLineDiagram3D` build counter and memoization cache (for tests). */
export function resetLineDiagram3DBuildCount(): void {
  lineDiagram3DBuildCount = 0;
  lastLineDiagram3DCache = null;
}

export const SIGN_POS_RGB: readonly [number, number, number] = hexToRgb(DARK.ok);
export const SIGN_NEG_RGB: readonly [number, number, number] = hexToRgb(DARK.warn);
export const SIGN_ZERO_RGB: readonly [number, number, number] = hexToRgb(DARK.muted);

/** Two-colour CSS gradient for the `"sign"` ramp (`themeTokens(mode).warn` on the negative half, `themeTokens(mode).ok` on the positive half). */
export function signRampCss(mode: ThemeMode = "dark"): string {
  const t = themeTokens(mode);
  return `linear-gradient(to right, ${t.warn} 0%, ${t.warn} 50%, ${t.ok} 50%, ${t.ok} 100%)`;
}

export function colorForValue(
  v: number,
  cmap: LineDiagramCmap,
  cmin: number,
  cmax: number,
  signHint = 0,
  mode: ThemeMode = "dark",
): [number, number, number] {
  if (cmap === "sign") {
    const s = v !== 0 ? Math.sign(v) : signHint;
    if (mode === "dark") {
      if (s > 0) return [SIGN_POS_RGB[0], SIGN_POS_RGB[1], SIGN_POS_RGB[2]];
      if (s < 0) return [SIGN_NEG_RGB[0], SIGN_NEG_RGB[1], SIGN_NEG_RGB[2]];
      return [SIGN_ZERO_RGB[0], SIGN_ZERO_RGB[1], SIGN_ZERO_RGB[2]];
    }
    const t = themeTokens(mode);
    if (s > 0) return hexToRgb(t.ok);
    if (s < 0) return hexToRgb(t.warn);
    return hexToRgb(t.muted);
  }
  return sampleCmap(cmap, rampT(v, cmin, cmax));
}

function medianOf(values: readonly number[]): number {
  if (values.length === 0) return 1;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0 ? sorted[mid]! : 0.5 * (sorted[mid - 1]! + sorted[mid]!);
}

/**
 * Build the 3-D quad-strip triangles (splitting across zero crossings), top
 * profile segments, and station riser segments for a `.ladruno` line-station step.
 */
export function buildLineDiagram3D(opts: BuildLineDiagram3DOptions): LineDiagram3DGeometry {
  const component = opts.stepData.component;
  const userScale = opts.scale ?? 1;
  const flipSign = effectiveFlipSign(component, opts.flipSign);
  const signMult = flipSign ? -1 : 1;
  const cmap = opts.cmap ?? "sign";
  const plane = opts.plane ?? "local";
  const theme = opts.theme ?? "dark";
  const axisChoice = localAxisForComponent(component);
  const needs3dFrame = requires3dFrame(component);
  const is2D = opts.ndm === 2;

  const viewDirKey =
    plane === "screen" && opts.cameraViewDir
      ? `${opts.cameraViewDir[0].toFixed(5)},${opts.cameraViewDir[1].toFixed(5)},${opts.cameraViewDir[2].toFixed(5)}`
      : "local";
  const cacheKey = [
    component,
    opts.stepTime ?? opts.stepData.time,
    opts.resultsStage?.id ?? "",
    opts.stageIndex ?? -1,
    opts.ndm ?? "null",
    opts.charLength ?? "auto",
    Boolean(opts.deformed),
    userScale,
    flipSign,
    cmap,
    plane,
    viewDirKey,
    theme,
    opts.samplesPerSegment ?? 16,
    opts.closureTolerance?.rtol ?? "def",
    opts.closureTolerance?.atol ?? "def",
    opts.join.space,
    opts.join.byDeclPath.size,
    opts.meshElements.length,
    visibleLineDigest(opts.visibleLineMap, opts.lineElement.length),
    arrayDigest(opts.lineElement),
    arrayDigest(opts.drawnLinePositions),
    arrayDigest(opts.lineNodes),
    arrayDigest(opts.stepData.elementIndex),
    arrayDigest(opts.stepData.stationNaturalCoord),
    arrayDigest(opts.stepData.values),
    arrayDigest(opts.stepData.localAxesZRec),
    arrayDigest(opts.stepData.localAxesQuaternion),
    opts.stepData.localAxesSources ? opts.stepData.localAxesSources.join(",") : "null",
    arrayDigest(opts.stepData.elementNodes),
    opts.stepData.companionStep?.component ?? "null",
    arrayDigest(opts.stepData.companionStep?.elementIndex),
    arrayDigest(opts.stepData.companionStep?.stationNaturalCoord),
    arrayDigest(opts.stepData.companionStep?.values),
  ].join("|");

  const cached = lastLineDiagram3DCache;
  if (
    cached &&
    cached.key === cacheKey &&
    cached.decls === opts.decls &&
    cached.definitions === (opts.definitions ?? null) &&
    cached.sequence === (opts.sequence ?? null) &&
    cached.stepLocalAxesZRec === opts.stepLocalAxesZRec &&
    cached.modelLocalAxesZRec === opts.modelLocalAxesZRec &&
    cached.modelTransformVecxzByTag === opts.modelTransformVecxzByTag &&
    cached.elementNodesById === opts.elementNodesById &&
    sameStepByComponent(cached.stepByComponent, opts.stepByComponent)
  ) {
    const climMatches =
      opts.clim == null
        ? cached.autoClim
        : opts.clim[0] === cached.geom.clim[0] && opts.clim[1] === cached.geom.clim[1];
    if (climMatches) {
      return cached.geom;
    }
  }

  lineDiagram3DBuildCount++;

  // Group station rows in `stepData` by element ID.
  const stationsByEid = new Map<
    number,
    {
      xi: number[];
      values: number[];
      zRec: [number, number, number] | null;
      zRecSource: "step_local_axes" | "model_local_axes" | null;
      nodes: [number, number] | null;
    }
  >();
  const { elementIndex, stationNaturalCoord, values, localAxesZRec, localAxesQuaternion, localAxesSources, elementNodes } = opts.stepData;
  for (let i = 0; i < elementIndex.length; i++) {
    const eid = elementIndex[i]!;
    const xi = stationNaturalCoord[i]!;
    const val = values[i]!;
    let entry = stationsByEid.get(eid);
    if (!entry) {
      let zRec: [number, number, number] | null = null;
      if (localAxesZRec) {
        const zx = localAxesZRec[i * 3]!;
        const zy = localAxesZRec[i * 3 + 1]!;
        const zz = localAxesZRec[i * 3 + 2]!;
        if (Number.isFinite(zx) && Number.isFinite(zy) && Number.isFinite(zz)) {
          zRec = [zx, zy, zz];
        }
      }
      if (!zRec && localAxesQuaternion) {
        const z = quaternionToZRec(
          localAxesQuaternion[i * 4]!,
          localAxesQuaternion[i * 4 + 1]!,
          localAxesQuaternion[i * 4 + 2]!,
          localAxesQuaternion[i * 4 + 3]!,
        );
        if (Number.isFinite(z[0])) zRec = z;
      }
      const zRecSource = zRec ? (localAxesSources?.[i] ?? "model_local_axes") : null;
      let nodes: [number, number] | null = null;
      if (elementNodes && i * 2 + 1 < elementNodes.length) {
        const ni = elementNodes[i * 2]!;
        const nj = elementNodes[i * 2 + 1]!;
        if (Number.isFinite(ni) && Number.isFinite(nj)) {
          nodes = [ni, nj];
        }
      }
      entry = { xi: [], values: [], zRec, zRecSource, nodes };
      stationsByEid.set(eid, entry);
    }
    entry.xi.push(xi);
    entry.values.push(val);
  }

  // All .ladruno element IDs that join any 1-D beam in the model mesh (`opts.lineElement`),
  // regardless of current group/element visibility (`opts.visibleLineMap`).
  const allLineEids = new Set<number>();
  for (let segIdx = 0; segIdx < opts.lineElement.length; segIdx++) {
    const elemIdx = opts.lineElement[segIdx]!;
    const decl = opts.meshElements[elemIdx];
    if (!decl) continue;
    const eid = opts.join.byDeclPath.get(decl);
    if (eid !== undefined) allLineEids.add(eid);
  }
  let unjoinedRows = 0;
  for (let i = 0; i < elementIndex.length; i++) {
    if (!allLineEids.has(elementIndex[i]!)) unjoinedRows++;
  }

  // Map DeclPath -> first and last drawn line segment index (supporting multi-segment 1-D elements).
  const nSegs = opts.visibleLineMap ? opts.visibleLineMap.length : opts.lineElement.length;
  const candidateSegments: { segIdx: number; endSegIdx: number; decl: DeclPath; eid: number }[] = [];
  const byDeclCand = new Map<DeclPath, { segIdx: number; endSegIdx: number; decl: DeclPath; eid: number }>();
  for (let k = 0; k < nSegs; k++) {
    const segIdx = opts.visibleLineMap ? opts.visibleLineMap[k]! : k;
    const elemIdx = opts.lineElement[segIdx]!;
    const decl = opts.meshElements[elemIdx];
    if (!decl) continue;
    const existing = byDeclCand.get(decl);
    if (existing) {
      existing.endSegIdx = segIdx;
      continue;
    }
    const eid = opts.join.byDeclPath.get(decl);
    if (eid === undefined || !stationsByEid.has(eid)) continue;
    const cand = { segIdx, endSegIdx: segIdx, decl, eid };
    byDeclCand.set(decl, cand);
    candidateSegments.push(cand);
  }

  const beams: BeamDiagramSpan3D[] = [];
  const skippedBeamDecls: DeclPath[] = [];
  let skippedNonFiniteBeams = 0;
  let skippedZeroLengthBeams = 0;
  let skippedOrientationBeams = 0;
  const lengths: number[] = [];
  const elementLengths = new Map<number, number>();
  let maxAbsVal = 0;
  let minVal = Infinity;
  let maxVal = -Infinity;

  for (const { segIdx, endSegIdx, decl, eid } of candidateSegments) {
    const st = stationsByEid.get(eid)!;
    const badStation =
      st.values.length === 0 ||
      st.xi.length !== st.values.length ||
      st.values.some((v) => !Number.isFinite(v)) ||
      st.xi.some((x) => !Number.isFinite(x) || x < -1 - 1e-9 || x > 1 + 1e-9);
    if (badStation) {
      skippedNonFiniteBeams++;
      continue;
    }

    const oStart = segIdx * 6;
    const oEnd = endSegIdx * 6;
    let pStart: [number, number, number] = [
      opts.drawnLinePositions[oStart]!,
      opts.drawnLinePositions[oStart + 1]!,
      opts.drawnLinePositions[oStart + 2]!,
    ];
    let pEnd: [number, number, number] = [
      opts.drawnLinePositions[oEnd + 3]!,
      opts.drawnLinePositions[oEnd + 4]!,
      opts.drawnLinePositions[oEnd + 5]!,
    ];

    // P0-2: Cross-check element connectivity (iNode, jNode) against cell corner nodes (nStart, nEnd).
    const declObj = opts.decls[decl];
    const inlineNodes = declObj?.element?.inlineNodes;
    const elemNodes: readonly [number, number] | null =
      opts.elementNodesById?.get(eid) ??
      st.nodes ??
      (inlineNodes && inlineNodes.length >= 2
        ? [inlineNodes[0]!, inlineNodes[inlineNodes.length - 1]!]
        : null);
    let nodeStart: number | undefined;
    let nodeEnd: number | undefined;
    if (opts.lineNodes && endSegIdx * 2 + 1 < opts.lineNodes.length) {
      const nStart = opts.lineNodes[segIdx * 2]!;
      const nEnd = opts.lineNodes[endSegIdx * 2 + 1]!;
      nodeStart = nStart;
      nodeEnd = nEnd;
      if (elemNodes) {
        const [iNode, jNode] = elemNodes;
        if (iNode === nStart && jNode === nEnd && iNode !== jNode) {
          // Forward orientation: cell corner order matches element (iNode -> jNode).
        } else if (iNode === nEnd && jNode === nStart && iNode !== jNode) {
          // Reversed orientation: flip pStart <-> pEnd so xi = -1 sits at iNode and xLocal runs iNode -> jNode.
          [pStart, pEnd] = [pEnd, pStart];
          nodeStart = iNode;
          nodeEnd = jNode;
        } else {
          skippedOrientationBeams++;
          continue;
        }
      }
    } else if (elemNodes) {
      [nodeStart, nodeEnd] = elemNodes;
    }

    const dx = pEnd[0] - pStart[0];
    const dy = pEnd[1] - pStart[1];
    const dz = pEnd[2] - pStart[2];
    const length = Math.hypot(dx, dy, dz);
    if (!Number.isFinite(length) || length < 1e-12) {
      skippedZeroLengthBeams++;
      continue;
    }
    const xLocal: [number, number, number] = [dx / length, dy / length, dz / length];

    let extrudeDir: [number, number, number];
    let zRecSource: ZRecSource;

    if (plane === "screen") {
      extrudeDir = screenExtrusionDirection(xLocal, opts.cameraViewDir);
      zRecSource = "screen";
    } else {
      // Priority 1: per-step RESULTS/ON_ELEMENTS/LOCAL_AXES (via opts.stepLocalAxesZRec or stepData with source "step_local_axes")
      // Priority 2: MODEL/LOCAL_AXES/<class>/FRAME (via opts.modelLocalAxesZRec or stepData with source "model_local_axes")
      // Priority 3: model.h5 /opensees/transforms/<name>/per_element_vecxz + per_element_emitted_tag
      // Priority 4: defaultVecxz(xLocal) in 2-D or for axial/torsion (N, T, eps, theta)
      let zRec: readonly [number, number, number] | null = null;
      let source: ZRecSource | null = null;

      const stepZ = opts.stepLocalAxesZRec?.get(eid) ?? (st.zRecSource === "step_local_axes" ? st.zRec : null);
      if (stepZ && Number.isFinite(stepZ[0]) && Number.isFinite(stepZ[1]) && Number.isFinite(stepZ[2])) {
        zRec = stepZ;
        source = "step_local_axes";
      } else {
        const modelZ = opts.modelLocalAxesZRec?.get(eid) ?? st.zRec;
        if (modelZ && Number.isFinite(modelZ[0]) && Number.isFinite(modelZ[1]) && Number.isFinite(modelZ[2])) {
          zRec = modelZ;
          source = "model_local_axes";
        } else {
          const transfZ = declObj ? elementTransformVecxzFromState(opts.decls, declObj, opts.modelTransformVecxzByTag) : null;
          if (transfZ) {
            zRec = transfZ;
            source = "model_transform";
          } else if (is2D || !needs3dFrame) {
            zRec = defaultVecxz(xLocal);
            source = "default_vecxz";
          }
        }
      }

      if (!zRec || !source) {
        // Fail-closed on 3-D shear/moment/curvature when no local frame is recorded.
        skippedBeamDecls.push(decl);
        continue;
      }

      let axes = beamLocalAxes(xLocal, zRec);
      if (!axes && (is2D || !needs3dFrame)) {
        // A degenerate frame (vecxz parallel to the chord) is replaced by the
        // default only where no 3-D bending frame is needed (#2473).
        axes = beamLocalAxes(xLocal, defaultVecxz(xLocal));
        source = "default_vecxz";
      }
      if (!axes) {
        // Fail-closed: a degenerate recorded frame in 3-D is never guessed.
        skippedBeamDecls.push(decl);
        continue;
      }
      if (axisChoice === "y") {
        extrudeDir = axes.yLocal;
      } else if (component === "My" || component === "kappaY") {
        // Right-hand rule about +yLocal = zLocal x xLocal puts tension on +zLocal for My > 0
        // and on -zLocal for My < 0 (opposite to Mz about +zLocal = xLocal x yLocal).
        // Using -zLocal as the unflipped (compression-side) baseline makes flipSign=true
        // (signMult = -1) extrude on the tension side (+My * zLocal), matching Mz.
        extrudeDir = [-axes.zLocal[0], -axes.zLocal[1], -axes.zLocal[2]];
      } else {
        extrudeDir = axes.zLocal;
      }
      zRecSource = source;
    }

    // Sort stations by ascending xi; expand single-station (xi = 0) to [-1, +1].
    const pairs = st.xi.map((xi, idx) => ({ xi, v: st.values[idx]! })).sort((a, b) => a.xi - b.xi);
    let xiArr: Float64Array;
    let valArr: Float64Array;
    if (pairs.length === 1) {
      xiArr = new Float64Array([-1, 1]);
      valArr = new Float64Array([pairs[0]!.v, pairs[0]!.v]);
    } else {
      xiArr = Float64Array.from(pairs.map((p) => p.xi));
      valArr = Float64Array.from(pairs.map((p) => p.v));
    }

    lengths.push(length);
    elementLengths.set(eid, length);
    beams.push({
      decl,
      elementId: eid,
      segmentIndex: segIdx,
      endSegmentIndex: endSegIdx,
      ...(nodeStart !== undefined ? { nodeStart } : {}),
      ...(nodeEnd !== undefined ? { nodeEnd } : {}),
      pStart,
      pEnd,
      length,
      xLocal,
      extrudeDir,
      zRecSource,
      xi: xiArr,
      values: valArr,
    });
  }

  // E2b: Exact span-load reconstruction from `/opensees/patterns/*/loads/elements`
  // when `component` is a force/moment (`N`, `Vy`, `Vz`, `T`, `My`, `Mz`).
  let exactSpanCount = 0;
  let interpolatedSpanCount = 0;
  const spanReasonSet = new Set<string>();
  if (beams.length > 0) {
    if (isBeamForceComponent(component) && opts.definitions) {
      const stepByComp: Partial<Record<LineStationComponent, LadrunoLineStationStep>> = {
        ...opts.stepByComponent,
        [component]: opts.stepData,
      };
      if (opts.stepData.companionStep && !stepByComp[opts.stepData.companionStep.component]) {
        stepByComp[opts.stepData.companionStep.component] = opts.stepData.companionStep;
      }
      const reconstructions = reconstructSpanDiagramsForStep({
        component,
        stepByComponent: stepByComp,
        elementLengths,
        join: opts.join,
        decls: opts.decls,
        definitions: opts.definitions,
        sequence: opts.sequence ?? null,
        ndm: opts.ndm,
        resultsStage: opts.resultsStage ?? null,
        stageIndex: opts.stageIndex,
        time: opts.stepTime ?? opts.stepData.time,
        nSub: opts.samplesPerSegment,
        closureTolerance: opts.closureTolerance,
      });
      for (const b of beams) {
        const rec = reconstructions.get(b.elementId);
        if (rec && rec.mode === "exact") {
          b.xi = rec.xi;
          b.values = rec.values;
          b.spanMode = "exact";
          b.spanReason = null;
          exactSpanCount++;
        } else {
          const reason = rec && rec.mode === "interpolated" ? rec.reason : "end stations do not span [-1, +1]";
          b.spanMode = "interpolated";
          b.spanReason = reason;
          interpolatedSpanCount++;
          if (reason) spanReasonSet.add(reason);
        }
      }
    } else {
      for (const b of beams) {
        b.spanMode = "interpolated";
        const reason = !isBeamForceComponent(component)
          ? `section deformation (${component}) uses recorded stations`
          : b.xi.length > 2 && b.xi.some((x) => Math.abs(x) < 1 - 1e-3)
            ? "interior Gauss stations recorded"
            : "no /opensees load definitions";
        b.spanReason = reason;
        interpolatedSpanCount++;
        spanReasonSet.add(reason);
      }
    }
  }

  for (const b of beams) {
    for (let i = 0; i < b.values.length; i++) {
      const v = b.values[i]!;
      const av = Math.abs(v);
      if (av > maxAbsVal) maxAbsVal = av;
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    }
  }

  const clim: [number, number] = opts.clim
    ? [opts.clim[0], opts.clim[1]]
    : minVal <= maxVal
      ? [minVal, maxVal]
      : [-1, 1];
  const rawClimPeak = opts.clim ? Math.max(Math.abs(opts.clim[0]), Math.abs(opts.clim[1])) : maxAbsVal;
  const peakRef = rawClimPeak > 0 ? rawClimPeak : maxAbsVal;
  const lChar = opts.charLength ?? medianOf(lengths);
  const autoScale = peakRef > 1e-30 ? (0.15 * lChar) / peakRef : opts.charLength !== undefined ? 0 : 1;
  const effectiveScale = autoScale * userScale;

  const triPos: number[] = [];
  const triVal: number[] = [];
  const triUV: number[] = [];
  const triCol: number[] = [];
  const triBeam: number[] = [];
  const profilePos: number[] = [];
  const profileVal: number[] = [];
  const profileBeam: number[] = [];
  const riserPos: number[] = [];
  const riserVal: number[] = [];
  const riserBeam: number[] = [];
  const labelAnchors: DiagramLabelAnchor[] = [];

  const pushTri = (
    beamIdx: number,
    a: readonly [number, number, number],
    b: readonly [number, number, number],
    c: readonly [number, number, number],
    va: number,
    vb: number,
    vc: number,
    signHint = 0,
  ) => {
    triPos.push(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
    triVal.push(va, vb, vc);
    triUV.push(rampT(va, clim[0], clim[1]), rampT(vb, clim[0], clim[1]), rampT(vc, clim[0], clim[1]));
    const ca = colorForValue(va, cmap, clim[0], clim[1], signHint, theme);
    const cb = colorForValue(vb, cmap, clim[0], clim[1], signHint, theme);
    const cc = colorForValue(vc, cmap, clim[0], clim[1], signHint, theme);
    triCol.push(ca[0], ca[1], ca[2], cb[0], cb[1], cb[2], cc[0], cc[1], cc[2]);
    triBeam.push(beamIdx);
  };

  beams.forEach((beam, beamIdx) => {
    const nSt = beam.xi.length;
    const basePts: [number, number, number][] = [];
    const topPts: [number, number, number][] = [];
    const chord = [beam.pEnd[0] - beam.pStart[0], beam.pEnd[1] - beam.pStart[1], beam.pEnd[2] - beam.pStart[2]] as const;

    let peakIdx = 0;
    let peakAbs = -1;

    for (let s = 0; s < nSt; s++) {
      const u = (beam.xi[s]! + 1) * 0.5;
      const bx = beam.pStart[0] + u * chord[0];
      const by = beam.pStart[1] + u * chord[1];
      const bz = beam.pStart[2] + u * chord[2];
      const vRaw = beam.values[s]!;
      const offset = signMult * vRaw * effectiveScale;
      const tx = bx + offset * beam.extrudeDir[0];
      const ty = by + offset * beam.extrudeDir[1];
      const tz = bz + offset * beam.extrudeDir[2];
      basePts.push([bx, by, bz]);
      topPts.push([tx, ty, tz]);

      // Station riser tick: base -> top
      riserPos.push(bx, by, bz, tx, ty, tz);
      riserVal.push(vRaw, vRaw);
      riserBeam.push(beamIdx);

      const av = Math.abs(vRaw);
      if (av > peakAbs) {
        peakAbs = av;
        peakIdx = s;
      }
    }

    labelAnchors.push({
      decl: beam.decl,
      elementId: beam.elementId,
      value: beam.values[peakIdx]!,
      position: topPts[peakIdx]!,
    });

    for (let s = 0; s + 1 < nSt; s++) {
      const p1Base = basePts[s]!;
      const p2Base = basePts[s + 1]!;
      const p1Top = topPts[s]!;
      const p2Top = topPts[s + 1]!;
      const v1Raw = beam.values[s]!;
      const v2Raw = beam.values[s + 1]!;

      // Top profile segment
      profilePos.push(p1Top[0], p1Top[1], p1Top[2], p2Top[0], p2Top[1], p2Top[2]);
      profileVal.push(v1Raw, v2Raw);
      profileBeam.push(beamIdx);

      if (v1Raw * v2Raw < 0) {
        // Zero crossing between station s and station s + 1: split at pZero on the chord into two non-degenerate triangles.
        const t = Math.abs(v1Raw) / (Math.abs(v1Raw) + Math.abs(v2Raw));
        const pZero: [number, number, number] = [
          p1Base[0] + t * (p2Base[0] - p1Base[0]),
          p1Base[1] + t * (p2Base[1] - p1Base[1]),
          p1Base[2] + t * (p2Base[2] - p1Base[2]),
        ];
        pushTri(beamIdx, p1Base, pZero, p1Top, v1Raw, 0, v1Raw, Math.sign(v1Raw));
        pushTri(beamIdx, pZero, p2Base, p2Top, 0, v2Raw, v2Raw, Math.sign(v2Raw));
      } else {
        const signHint = Math.sign(v1Raw || v2Raw);
        pushTri(beamIdx, p1Base, p2Base, p2Top, v1Raw, v2Raw, v2Raw, signHint);
        pushTri(beamIdx, p1Base, p2Top, p1Top, v1Raw, v2Raw, v1Raw, signHint);
      }
    }
  });

  const skippedCount = skippedBeamDecls.length;
  const reasons: string[] = [];
  if (skippedCount > 0) {
    reasons.push(
      `${skippedCount} 3-D beam${skippedCount === 1 ? "" : "s"} skipped: ${component} in local plane requires LOCAL_AXES or geomTransf vecxz that is unambiguous and not parallel to the beam (never guessed in 3-D).`,
    );
  }
  if (skippedOrientationBeams > 0) {
    reasons.push(
      `${skippedOrientationBeams} beam${skippedOrientationBeams === 1 ? "" : "s"} skipped: element connectivity (iNode, jNode) contradicts mesh cell corner nodes.`,
    );
  }
  if (skippedNonFiniteBeams > 0) {
    reasons.push(
      `${skippedNonFiniteBeams} beam${skippedNonFiniteBeams === 1 ? "" : "s"} skipped: non-finite station value or natural coordinate xi outside [-1, 1].`,
    );
  }
  if (skippedZeroLengthBeams > 0) {
    reasons.push(
      `${skippedZeroLengthBeams} beam${skippedZeroLengthBeams === 1 ? "" : "s"} skipped: zero or non-finite chord length.`,
    );
  }
  const totalRows = elementIndex.length;
  if (unjoinedRows > 0) {
    if (unjoinedRows === totalRows) {
      reasons.push(
        `No line-station row joins a drawn 1-D beam (${unjoinedRows} of ${totalRows} station row${totalRows === 1 ? "" : "s"} unjoined).`,
      );
    } else {
      reasons.push(
        `${unjoinedRows} of ${totalRows} station row${totalRows === 1 ? "" : "s"} unjoined (no matching 1-D beam in mesh).`,
      );
    }
  } else if (beams.length === 0 && reasons.length === 0) {
    reasons.push("No line-station row joins a drawn 1-D beam.");
  }
  if (
    opts.deformed &&
    beams.some((b) => b.zRecSource !== "step_local_axes" && b.zRecSource !== "screen")
  ) {
    reasons.push(
      "Deformed baseline uses reference local frame (no per-step RESULTS/ON_ELEMENTS/LOCAL_AXES recorded).",
    );
  }
  const notice = reasons.length > 0 ? reasons.join(" ") : null;

  const geom: LineDiagram3DGeometry = {
    component,
    triPositions: new Float32Array(triPos),
    triValues: new Float32Array(triVal),
    triUVs: new Float32Array(triUV),
    triColors: new Float32Array(triCol),
    triBeamIndex: new Int32Array(triBeam),
    profilePositions: new Float32Array(profilePos),
    profileValues: new Float32Array(profileVal),
    profileBeamIndex: new Int32Array(profileBeam),
    riserPositions: new Float32Array(riserPos),
    riserValues: new Float32Array(riserVal),
    riserBeamIndex: new Int32Array(riserBeam),
    beams,
    labelAnchors,
    peakAbsValue: maxAbsVal,
    autoScale,
    effectiveScale,
    clim,
    skippedUnorientedBeams: skippedCount,
    skippedOrientationMismatchBeams: skippedOrientationBeams,
    skippedBeamDecls,
    notice,
    exactSpanCount,
    interpolatedSpanCount,
    spanReasons: [...spanReasonSet],
  };

  lastLineDiagram3DCache = {
    key: cacheKey,
    decls: opts.decls,
    definitions: opts.definitions ?? null,
    sequence: opts.sequence ?? null,
    stepLocalAxesZRec: opts.stepLocalAxesZRec,
    modelLocalAxesZRec: opts.modelLocalAxesZRec,
    modelTransformVecxzByTag: opts.modelTransformVecxzByTag,
    elementNodesById: opts.elementNodesById,
    stepByComponent: opts.stepByComponent,
    autoClim: opts.clim == null,
    geom,
  };

  return geom;
}

// -- 3-D Hover Needle Helper --------------------------------------------------

export interface HoverNeedleResult {
  decl: DeclPath;
  elementId: number;
  /** Relative distance along beam chord x/L in [0, 1]. */
  xRel: number;
  /** Distance along beam chord x in [0, L]. */
  x: number;
  /** Total chord length L. */
  length: number;
  /** Natural coordinate xi = 2*(x/L) - 1 in [-1, +1]. */
  xi: number;
  /** Linearly interpolated station value at xi. */
  value: number;
  /** True when xi lies within the recorded station range [xiMin, xiMax] (or single-station constant span); false in unmeasured Gauss end zones. */
  withinStationSpan: boolean;
  /** Projected point on the beam chord. */
  pBase: [number, number, number];
  /** Corresponding point on the diagram profile. */
  pTop: [number, number, number];
  /** 6 floats: `[pBase.x, pBase.y, pBase.z, pTop.x, pTop.y, pTop.z]`. */
  needlePositions: Float32Array;
}

export type HoverableBeamSpan = {
  readonly decl: DeclPath;
  readonly elementId: number;
  readonly pStart: readonly [number, number, number];
  readonly pEnd: readonly [number, number, number];
  readonly length: number;
  readonly extrudeDir: readonly [number, number, number];
  readonly xi: ArrayLike<number>;
  readonly values: ArrayLike<number>;
};

const ENDPOINT_XI_TOL = 1e-6;

/** Piecewise-linear interpolation of `value(xi)` across sorted stations `(xiArr, valArr)`. */
export function interpolateStationValue(xiArr: ArrayLike<number>, valArr: ArrayLike<number>, xi: number): number {
  const n = xiArr.length;
  if (n === 0) return Number.NaN;
  if (n === 1 || xi <= xiArr[0]!) return valArr[0]!;
  if (xi >= xiArr[n - 1]!) return valArr[n - 1]!;
  for (let i = 0; i + 1 < n; i++) {
    const x0 = xiArr[i]!;
    const x1 = xiArr[i + 1]!;
    if (xi >= x0 && xi <= x1) {
      const denom = x1 - x0;
      const alpha = denom > 1e-15 ? (xi - x0) / denom : 0;
      return valArr[i]! + alpha * (valArr[i + 1]! - valArr[i]!);
    }
  }
  return valArr[n - 1]!;
}

/**
 * Project a 3-D `hoverPoint` onto `beam`'s chord `pStart -> pEnd`, returning
 * `xRel = x / L` in `[0, 1]`, `xi = 2 * xRel - 1`, the linearly interpolated
 * value `v(xi)`, `withinStationSpan`, and the 3-D needle segment `pBase -> pTop`.
 */
export function projectHoverNeedle(
  beam: HoverableBeamSpan,
  hoverPoint: readonly [number, number, number],
  effectiveScale: number,
  flipSign: boolean,
): HoverNeedleResult {
  const cx = beam.pEnd[0] - beam.pStart[0];
  const cy = beam.pEnd[1] - beam.pStart[1];
  const cz = beam.pEnd[2] - beam.pStart[2];
  const lSq = cx * cx + cy * cy + cz * cz;
  const rawT =
    lSq > 1e-24
      ? ((hoverPoint[0] - beam.pStart[0]) * cx +
          (hoverPoint[1] - beam.pStart[1]) * cy +
          (hoverPoint[2] - beam.pStart[2]) * cz) /
        lSq
      : 0;
  const xRel = Math.min(1, Math.max(0, rawT));
  const x = xRel * beam.length;
  const xi = 2 * xRel - 1;
  const nStations = beam.xi.length;
  let xiMin = -1;
  let xiMax = 1;
  if (nStations >= 2) {
    xiMin = Number.POSITIVE_INFINITY;
    xiMax = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < nStations; i++) {
      const v = beam.xi[i]!;
      if (v < xiMin) xiMin = v;
      if (v > xiMax) xiMax = v;
    }
  } else if (nStations === 0) {
    xiMin = Number.POSITIVE_INFINITY;
    xiMax = Number.NEGATIVE_INFINITY;
  }
  const withinStationSpan =
    nStations === 1 ||
    (nStations >= 2 && xi >= xiMin - ENDPOINT_XI_TOL && xi <= xiMax + ENDPOINT_XI_TOL);
  const value = interpolateStationValue(beam.xi, beam.values, xi);
  const pBase: [number, number, number] = [
    beam.pStart[0] + xRel * cx,
    beam.pStart[1] + xRel * cy,
    beam.pStart[2] + xRel * cz,
  ];
  const signMult = flipSign ? -1 : 1;
  const offset = signMult * value * effectiveScale;
  const pTop: [number, number, number] = [
    pBase[0] + offset * beam.extrudeDir[0],
    pBase[1] + offset * beam.extrudeDir[1],
    pBase[2] + offset * beam.extrudeDir[2],
  ];
  return {
    decl: beam.decl,
    elementId: beam.elementId,
    xRel,
    x,
    length: beam.length,
    xi,
    value,
    withinStationSpan,
    pBase,
    pTop,
    needlePositions: new Float32Array([pBase[0], pBase[1], pBase[2], pTop[0], pTop[1], pTop[2]]),
  };
}

// -- Multi-Beam Chain Builder (2-D Plot Span Ordering) ------------------------

export interface BeamChainInputSpan {
  decl: DeclPath;
  elementId: number;
  nodeStart: number;
  nodeEnd: number;
  length: number;
  xi: ArrayLike<number>;
  values: ArrayLike<number>;
  /** Optional 3-D start/end coordinates or unit chord direction (`nodeStart -> nodeEnd`) for colinear tie-breaking at junctions. */
  pStart?: readonly [number, number, number];
  pEnd?: readonly [number, number, number];
  xLocal?: readonly [number, number, number];
  dir?: readonly [number, number, number];
}

function spanUnitDirection(span: BeamChainInputSpan, reversed: boolean): [number, number, number] | null {
  let dx: number;
  let dy: number;
  let dz: number;
  if (span.xLocal) {
    [dx, dy, dz] = span.xLocal;
  } else if (span.dir) {
    [dx, dy, dz] = span.dir;
  } else if (span.pStart && span.pEnd) {
    dx = span.pEnd[0] - span.pStart[0];
    dy = span.pEnd[1] - span.pStart[1];
    dz = span.pEnd[2] - span.pStart[2];
  } else {
    return null;
  }
  const norm = Math.hypot(dx, dy, dz);
  if (!Number.isFinite(norm) || norm < 1e-12) return null;
  const s = reversed ? -1 / norm : 1 / norm;
  return [dx * s, dy * s, dz * s];
}

const COLINEAR_DOT_THRESHOLD = 0.95;

/**
 * Order selected 1-D beams (`inputSpans`, e.g. from `s.selection.decls`)
 * node-to-node into continuous chains `[0, sum(L)]` (Robot style) with joint
 * markers and cumulative station coordinates for the 2-D SVG plot.
 *
 * Returns one `BeamChain` per connected component (when all selected beams
 * are contiguous, returns `[chain]` with `chain.contiguous === true`).
 */
export function buildBeamChains(inputSpans: readonly BeamChainInputSpan[]): BeamChain[] {
  if (inputSpans.length === 0) return [];

  const n = inputSpans.length;
  const used = new Uint8Array(n);
  const adj = new Map<number, number[]>();
  const outDeg = new Map<number, number>();
  const inDeg = new Map<number, number>();

  inputSpans.forEach((s, idx) => {
    (adj.get(s.nodeStart) ?? adj.set(s.nodeStart, []).get(s.nodeStart)!).push(idx);
    (adj.get(s.nodeEnd) ?? adj.set(s.nodeEnd, []).get(s.nodeEnd)!).push(idx);
    outDeg.set(s.nodeStart, (outDeg.get(s.nodeStart) ?? 0) + 1);
    inDeg.set(s.nodeEnd, (inDeg.get(s.nodeEnd) ?? 0) + 1);
  });

  const chains: BeamChain[] = [];
  const totalComponentsSuggested = () => chains.length === 1 && chains[0]!.spans.length === n;

  while (true) {
    let firstUnused = -1;
    for (let i = 0; i < n; i++) {
      if (!used[i]) {
        firstUnused = i;
        break;
      }
    }
    if (firstUnused < 0) break;

    // Collect all spans in the connected component of `firstUnused`.
    const compIndices: number[] = [];
    const compNodes = new Set<number>();
    const queueNodes = [inputSpans[firstUnused]!.nodeStart, inputSpans[firstUnused]!.nodeEnd];
    const visitedNodes = new Set<number>(queueNodes);
    const inComp = new Set<number>([firstUnused]);

    while (queueNodes.length > 0) {
      const u = queueNodes.shift()!;
      compNodes.add(u);
      for (const edgeIdx of adj.get(u) ?? []) {
        if (used[edgeIdx]) continue;
        inComp.add(edgeIdx);
        const sp = inputSpans[edgeIdx]!;
        const other = sp.nodeStart === u ? sp.nodeEnd : sp.nodeStart;
        if (!visitedNodes.has(other)) {
          visitedNodes.add(other);
          queueNodes.push(other);
        }
      }
    }
    for (const idx of inComp) compIndices.push(idx);
    compIndices.sort((a, b) => a - b);

    // Degree of each node within this component:
    const deg = new Map<number, number>();
    const compOut = new Map<number, number>();
    const compIn = new Map<number, number>();
    for (const idx of compIndices) {
      const sp = inputSpans[idx]!;
      deg.set(sp.nodeStart, (deg.get(sp.nodeStart) ?? 0) + 1);
      deg.set(sp.nodeEnd, (deg.get(sp.nodeEnd) ?? 0) + 1);
      compOut.set(sp.nodeStart, (compOut.get(sp.nodeStart) ?? 0) + 1);
      compIn.set(sp.nodeEnd, (compIn.get(sp.nodeEnd) ?? 0) + 1);
    }

    // Whether an endpoint's path (walking through any intermediate degree-2 nodes)
    // continues colinearly through the first degree >= 3 junction it reaches:
    const hasColinearThroughJunction = (epNode: number): boolean => {
      let curr = epNode;
      let prevEdge = -1;
      for (let step = 0; step < compIndices.length; step++) {
        const inc = (adj.get(curr) ?? []).filter((idx) => idx !== prevEdge && !used[idx] && inComp.has(idx));
        if (inc.length !== 1) return false;
        const e0 = inc[0]!;
        const sp0 = inputSpans[e0]!;
        const rev0 = sp0.nodeStart !== curr;
        const nextNode = rev0 ? sp0.nodeStart : sp0.nodeEnd;
        const nextDeg = deg.get(nextNode) ?? 0;
        if (nextDeg >= 3) {
          const dir0 = spanUnitDirection(sp0, rev0);
          if (!dir0) return false;
          for (const e1 of adj.get(nextNode) ?? []) {
            if (e1 === e0 || used[e1] || !inComp.has(e1)) continue;
            const sp1 = inputSpans[e1]!;
            const rev1 = sp1.nodeStart !== nextNode;
            const dir1 = spanUnitDirection(sp1, rev1);
            if (!dir1) continue;
            const dot = dir0[0] * dir1[0] + dir0[1] * dir1[1] + dir0[2] * dir1[2];
            if (dot > COLINEAR_DOT_THRESHOLD) return true;
          }
          return false;
        }
        if (nextDeg === 2) {
          prevEdge = e0;
          curr = nextNode;
          continue;
        }
        return false;
      }
      return false;
    };

    // Pick start node for the chain walk:
    // 1. Endpoint (degree === 1) with out-degree 1 and in-degree 0 that continues colinearly through a degree >= 3 junction
    // 2. Any endpoint (degree === 1) that continues colinearly through a degree >= 3 junction
    // 3. Endpoint (degree === 1) with out-degree 1 and in-degree 0 (natural tail of directed chain)
    // 4. Any endpoint (degree === 1), earliest in selection order
    // 5. `nodeStart` of the earliest span in the component (closed loop)
    const endpoints = [...deg.entries()].filter(([, d]) => d === 1).map(([node]) => node);
    const directedStarts = endpoints.filter((node) => (compOut.get(node) ?? 0) === 1 && (compIn.get(node) ?? 0) === 0);
    const currentNodeCandidate =
      directedStarts.find((node) => hasColinearThroughJunction(node)) ??
      endpoints.find((node) => hasColinearThroughJunction(node)) ??
      directedStarts[0] ??
      endpoints[0] ??
      inputSpans[compIndices[0]!]!.nodeStart;
    let currentNode: number = currentNodeCandidate;

    const orderedSpans: BeamChainSpan[] = [];
    const joints: BeamChainJoint[] = [{ nodeId: currentNode, x: 0 }];
    let xCursor = 0;
    let prevDir: [number, number, number] | null = null;

    while (true) {
      const candidates = (adj.get(currentNode) ?? []).filter((idx) => !used[idx] && inComp.has(idx));
      if (candidates.length === 0) break;

      const dotForCandidate = (idx: number): number => {
        if (!prevDir) return Number.NEGATIVE_INFINITY;
        const cand = inputSpans[idx]!;
        const candRev = cand.nodeStart !== currentNode;
        const nextDir = spanUnitDirection(cand, candRev);
        if (!nextDir) return Number.NEGATIVE_INFINITY;
        return prevDir[0] * nextDir[0] + prevDir[1] * nextDir[1] + prevDir[2] * nextDir[2];
      };

      // When continuing a chain (`orderedSpans.length > 0` and `prevDir !== null`),
      // prefer a nearly colinear candidate (`dot(prevDir, nextDir) > 0.95`) before
      // falling back to forward orientation, lowest `elementId`, and input order.
      candidates.sort((a, b) => {
        if (orderedSpans.length > 0 && prevDir !== null) {
          const dotA = dotForCandidate(a);
          const dotB = dotForCandidate(b);
          const colA = dotA > COLINEAR_DOT_THRESHOLD;
          const colB = dotB > COLINEAR_DOT_THRESHOLD;
          if (colA !== colB) return colA ? -1 : 1;
          if (colA && colB && dotA !== dotB) return dotB - dotA;
        }
        const aFwd = inputSpans[a]!.nodeStart === currentNode ? 0 : 1;
        const bFwd = inputSpans[b]!.nodeStart === currentNode ? 0 : 1;
        return aFwd - bFwd || inputSpans[a]!.elementId - inputSpans[b]!.elementId || a - b;
      });
      const chosenIdx = candidates[0]!;
      used[chosenIdx] = 1;
      const raw = inputSpans[chosenIdx]!;
      const reversed = raw.nodeStart !== currentNode;
      const nodeStart = reversed ? raw.nodeEnd : raw.nodeStart;
      const nodeEnd = reversed ? raw.nodeStart : raw.nodeEnd;
      const xStart = xCursor;
      const xEnd = xStart + raw.length;
      prevDir = spanUnitDirection(raw, reversed);

      // Expand single-station (xi = 0) to [-1, +1] so a flat span covers [xStart, xEnd].
      // For multi-station spans (e.g. Legendre or Radau interior Gauss points), record
      // `xiMin`, `xiMax`, `startEndKnown`, and `endEndKnown` without clamping interior
      // stations onto the unmeasured beam ends.
      const rawPairs: { xi: number; value: number }[] = [];
      let xiMin = -1;
      let xiMax = 1;
      let startEndKnown = true;
      let endEndKnown = true;
      if (raw.xi.length === 1) {
        const v0 = raw.values[0]!;
        rawPairs.push({ xi: -1, value: v0 }, { xi: 1, value: v0 });
      } else if (raw.xi.length >= 2) {
        xiMin = Number.POSITIVE_INFINITY;
        xiMax = Number.NEGATIVE_INFINITY;
        for (let i = 0; i < raw.xi.length; i++) {
          const xiVal = raw.xi[i]!;
          if (xiVal < xiMin) xiMin = xiVal;
          if (xiVal > xiMax) xiMax = xiVal;
          rawPairs.push({ xi: xiVal, value: raw.values[i]! });
        }
        const elemStartKnown = xiMin <= -1 + ENDPOINT_XI_TOL;
        const elemEndKnown = xiMax >= 1 - ENDPOINT_XI_TOL;
        startEndKnown = reversed ? elemEndKnown : elemStartKnown;
        endEndKnown = reversed ? elemStartKnown : elemEndKnown;
      } else {
        xiMin = Number.NaN;
        xiMax = Number.NaN;
        startEndKnown = false;
        endEndKnown = false;
      }

      const stations: BeamChainStationPoint[] = rawPairs
        .map(({ xi, value }) => {
          const u = reversed ? (1 - xi) * 0.5 : (xi + 1) * 0.5;
          return {
            xi,
            x: xStart + u * raw.length,
            value,
          };
        })
        .sort((a, b) => a.x - b.x);

      orderedSpans.push({
        decl: raw.decl,
        elementId: raw.elementId,
        nodeStart,
        nodeEnd,
        reversed,
        xStart,
        xEnd,
        length: raw.length,
        xiMin,
        xiMax,
        startEndKnown,
        endEndKnown,
        stations,
      });
      xCursor = xEnd;
      currentNode = nodeEnd;
      joints.push({ nodeId: currentNode, x: xCursor });
    }

    chains.push({
      spans: orderedSpans,
      joints,
      totalLength: xCursor,
      contiguous: false,
    });
  }

  const isSingleContiguous = totalComponentsSuggested();
  if (isSingleContiguous && chains[0]) {
    chains[0].contiguous = true;
  }
  return chains;
}

export type ChainableBeamSpan = {
  readonly decl: DeclPath;
  readonly elementId: number;
  readonly segmentIndex?: number;
  readonly endSegmentIndex?: number;
  readonly nodeStart?: number;
  readonly nodeEnd?: number;
  readonly pStart: readonly [number, number, number];
  readonly pEnd: readonly [number, number, number];
  readonly length: number;
  readonly xLocal: readonly [number, number, number];
  readonly xi: ArrayLike<number>;
  readonly values: ArrayLike<number>;
};

/**
 * Build the selected `BeamChain` list from `State.selection.decls` and `diagram3D.beams`.
 * Returns `[]` when no drawn 1-D beam is selected (never falls back to every beam in the model).
 */
export function buildSelectedBeamChainsFromState(
  s: Pick<State, "selection">,
  diagram3D: { readonly beams: readonly ChainableBeamSpan[] },
  lineNodes?: ArrayLike<number>,
): BeamChain[] {
  const byDecl = new Map<DeclPath, ChainableBeamSpan>();
  for (const b of diagram3D.beams) byDecl.set(b.decl, b);
  const selected = s.selection.decls.filter((d) => byDecl.has(d));
  if (selected.length === 0) return [];
  const spans: BeamChainInputSpan[] = [];
  for (const decl of selected) {
    const b = byDecl.get(decl);
    if (!b) continue;
    const segIdx = b.segmentIndex ?? 0;
    const endSeg = b.endSegmentIndex ?? segIdx;
    const nodeStart = b.nodeStart ?? (lineNodes ? lineNodes[segIdx * 2]! : 0);
    const nodeEnd = b.nodeEnd ?? (lineNodes ? lineNodes[endSeg * 2 + 1]! : 0);
    spans.push({
      decl: b.decl,
      elementId: b.elementId,
      nodeStart,
      nodeEnd,
      length: b.length,
      xi: b.xi,
      values: b.values,
      pStart: b.pStart,
      pEnd: b.pEnd,
      xLocal: b.xLocal,
    });
  }
  return buildBeamChains(spans);
}
