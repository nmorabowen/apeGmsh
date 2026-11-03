// The D3 manifest (ADR 0128 D3, INV-6): the numbers behind every still. An
// agent reports a value from here, never by reading pixels.
//
// `manifestEntry` builds one shot's entry from the state and the shot. It is
// pure, knows nothing of Electron or of the render job, and reads the step's
// values and the mesh's arrays through what it is handed, so the batch stills
// (renderer/agentRender.ts) and a snapshot of the human's window (D6) state a
// view the same way. It throws when it cannot state what the view shows: a
// value with no position, step data of another step. The render mode builds
// the entry before it writes the PNG and refuses the shot on a throw, so no
// still ever stands without its entry.
//
// What the numbers mean:
// - `extrema.recorded` is over every value the file records for the field at
//   the step, drawn or not: the values the `step` range is measured on, and
//   what Python's `Results` reads (tests/viewers/test_agent_manifest_oracle.py).
// - `extrema.drawn` keeps only what the visible primitives show: a solid's
//   interior node, a hidden group and an inactive element are left out.
//   With a section plane cutting it is null, and `drawn_reason` says why: what
//   a cut shows (its caps, the coordinates they cut at) is not computed here,
//   and a number from a part the still has cut away must not stand as drawn.
// - `no_data` counts the visible nodes (a nodal or an averaged field) or cells
//   (a flat or a discrete field) the field gives no finite value. A deformed
//   shape's counts are of drawn nodes too. With a plane cutting, each carries
//   a `limit`: they are counted before the cut.
// - A tie takes the lowest id, so the answer never depends on row order.
// - `position` is the reference position, in the model's coordinates, also
//   under a deformed shape: a node's position, or the mean of an element's
//   nodes. A discrete field names the element and its corner's node, and
//   gives that node's position.
//
// Only `timing` varies between two runs of one spec on one machine: paths of
// stills are relative to the manifest, and every number comes from the files.

import type { CombinationColumnOf, StepValues } from "../renderer/contour.ts";
import { elementMaskOf, visibleMaps } from "../renderer/visible.ts";
import type { BlobStore } from "../state/blobs.ts";
import { columnOf, gaussCellValues, isNodeTopology, meshFemIds, vertexValues } from "../state/contour.ts";
import { applyPlan, cellLookup, planFor } from "../state/extrapolation.ts";
import { activeCount, contourView, deformView, type ContourView } from "../state/selectors.ts";
import type { ArtifactInfo, CameraPose, ContourScan, MeshInfo, State } from "../state/types.ts";
import { bannerLines, describeSet } from "./describe.ts";
import type { SpecShot, ViewSpec } from "./spec.ts";

export const MANIFEST_VERSION = "apegmsh-manifest/1";
export const MANIFEST_FILE = "manifest.json";

type Vec3 = [number, number, number];

/** One end of a field's range: the value, and what holds it. */
export interface ManifestPoint {
  value: number;
  /** the node holding the value; null when an element does (a flat element field) */
  node: number | null;
  /** the element holding the value; null when a node does (a nodal or an averaged field) */
  element: number | null;
  /** the reference position (see the header) */
  position: Vec3;
}

export interface ManifestExtrema {
  min: ManifestPoint;
  max: ManifestPoint;
}

/** A contour as the entry states it; `drawn: false` is the sentence the legend gives instead. */
export type ManifestContour =
  | { drawn: false; reason: string }
  | {
      drawn: true;
      field: string;
      /** the dataset name, or the app's magnitude key */
      component: string;
      /** what the legend shows for the component */
      label: string;
      /** VU7b-2: `combination` is a User Combination's nodal value (`combination` names it) */
      source: "nodal" | "element" | "combination";
      /** VU7b-2, a combination's contour only: which, and whether it may draw as a deformed shape */
      combination?: { name: string; signed: boolean };
      /** element fields: how the values reach the screen (`element_mean`, `gauss_node`, ...); nodal: `nodal` */
      topology: string;
      cmap: string;
      range: {
        min: number | null;
        max: number | null;
        /** VU7b-2: `state` is a User Combination's own range (one state: no step, no history) */
        mode: "history" | "step" | "fixed" | "state";
        label: string;
        /** the whole-history scan of this field, and how far it got; `none` when none ran */
        history_scan: { status: ContourScan["status"] | "none"; done: number; total: number };
      };
      /** `recorded` null: no value is finite; `drawn` null: `drawn_reason` says why */
      extrema: { recorded: ManifestExtrema | null; drawn: ManifestExtrema | null; drawn_reason: string | null };
      /** `limit`: why the count is not of what the still shows, or null when it is */
      no_data: { kind: "nodes" | "cells"; count: number; of: number; limit: string | null };
    };

export type ManifestDeform =
  | { state: "unavailable" | "refused"; reason: string }
  | { state: "loading" | "ready" | "partial"; scale: number; ghost: boolean; drawn: number | null; undeformed: number | null; limit: string | null };

export interface ManifestFile {
  path: string;
  status: ArtifactInfo["status"];
  size: number;
}

export interface ManifestFiles {
  model: (ManifestFile & { model_hash: string | null; fem_hash: string | null }) | null;
  geometry: ManifestFile | null;
  results: (ManifestFile & { model_hash: string | null; fem_hash: string | null; results_hash: string | null }) | null;
}

export interface ManifestCamera extends CameraPose {
  /** the vertical field of view, in degrees; null for an orthographic projection */
  fov: number | null;
}

export interface ManifestEntry {
  name: string;
  status: "ok" | "refused";
  reason: string | null;
  /** the still, relative to the manifest; null for a refused shot */
  png: string | null;
  /** the still's size in pixels; null for a refused shot */
  image: [number, number] | null;
  theme: string;
  /** the shot as applied, every default resolved */
  shot: SpecShot;
  /** as resolved after the view was drawn; null for a refused shot */
  camera: ManifestCamera | null;
  files: ManifestFiles;
  pairing: { state: string; reason: string | null };
  stage: { index: number; id: string; name: string } | null;
  step: number | null;
  time: number | null;
  contour: ManifestContour | null;
  deform: ManifestDeform | null;
  /** every banner line on screen, shown or hidden by a layer (D9) */
  banner: string[];
  /** the DOM check (D9): the viewport's box and the panels left in the layout; null when no still was laid out */
  layout: { viewport: [number, number, number, number]; panels: string[] } | null;
}

export interface Manifest {
  manifest: typeof MANIFEST_VERSION;
  spec: string;
  size: [number, number];
  theme: string;
  shots: ManifestEntry[];
  /** the only part that varies between two runs; main adds `written_at` */
  timing: { shots: Record<string, { ms: number; shown_ms: number | null }> };
}

/** What an entry states that the state does not hold: the still and the camera, and the arrays to read. */
export interface EntryInput {
  shot: SpecShot;
  /** the still's file name, relative to the manifest */
  png: string;
  image: [number, number];
  camera: ManifestCamera;
  layout: ManifestEntry["layout"];
  blobs: BlobStore;
  /** the step data on screen (`effects.currentStep()`); null on the mesh */
  step: StepValues | null;
  /** VU7b-2: the User Combination column on screen (renderer/contour.ts `CombinationFeed.held`); a combination contour without it cannot be stated */
  combination?: CombinationColumnOf | null;
}

function filesOf(s: State): ManifestFiles {
  const a = s.artifacts;
  const file = (x: ArtifactInfo): ManifestFile => ({ path: x.path, status: x.status, size: x.sizeBytes });
  return {
    model: a.model ? { ...file(a.model), model_hash: s.lineage?.modelHash ?? null, fem_hash: s.lineage?.femHash ?? null } : null,
    geometry: a.geometry ? file(a.geometry) : null,
    results: a.results
      ? { ...file(a.results), model_hash: s.results?.lineage?.modelHash ?? null, fem_hash: s.results?.lineage?.femHash ?? null, results_hash: s.results?.lineage?.resultsHash ?? null }
      : null,
  };
}

// -- extrema ---------------------------------------------------------------------

/** One candidate of a field: its value (NaN: no data), and what holds it. */
interface Candidates {
  count: number;
  value(i: number): number;
  node(i: number): number | null;
  element(i: number): number | null;
  /** on a visible primitive */
  visible(i: number): boolean;
}

/** Lower than `b` in the tie order: the element's id, then the node's. */
function ranksFirst(c: Candidates, a: number, b: number): boolean {
  const ea = c.element(a) ?? 0, eb = c.element(b) ?? 0;
  return ea !== eb ? ea < eb : (c.node(a) ?? 0) < (c.node(b) ?? 0);
}

function extremaOf(c: Candidates, keep: (i: number) => boolean, position: (i: number) => Vec3): ManifestExtrema | null {
  let lo = -1;
  let hi = -1;
  for (let i = 0; i < c.count; i++) {
    const v = c.value(i);
    if (!Number.isFinite(v) || !keep(i)) continue;
    if (lo < 0 || v < c.value(lo) || (v === c.value(lo) && ranksFirst(c, i, lo))) lo = i;
    if (hi < 0 || v > c.value(hi) || (v === c.value(hi) && ranksFirst(c, i, hi))) hi = i;
  }
  if (lo < 0) return null;
  const point = (i: number): ManifestPoint => ({ value: c.value(i), node: c.node(i), element: c.element(i), position: position(i) });
  return { min: point(lo), max: point(hi) };
}

/** The reference position of a model node; a node the model does not have is an error, never a guess. */
function nodePosition(mesh: MeshInfo, blobs: BlobStore): (id: number) => Vec3 {
  if (!mesh.nodes) throw new Error("manifest: the mesh was built without its node table, so an extremum has no position");
  const ids = blobs.get(mesh.nodes.ids);
  const xyz = blobs.get(mesh.nodes.xyz);
  if (!(ids instanceof Float64Array)) throw new TypeError(`manifest: ${mesh.nodes.ids.id} is not f64`);
  return (id) => {
    const i = ids.indexOf(id);
    if (i < 0) throw new Error(`manifest: node ${id} has a value, and the model has no such node`);
    return [xyz[3 * i]!, xyz[3 * i + 1]!, xyz[3 * i + 2]!];
  };
}

/** What the visible primitives show: their nodes, and their elements as indices into `mesh.elements` with those elements' FEM ids. */
function visibleOf(s: State, mesh: MeshInfo, blobs: BlobStore): { nodes: Set<number>; elements: Set<number>; femIds: Float64Array } {
  const { lineMap, triMap } = visibleMaps(mesh.legend, s.visibility.hidden, blobs.i32(mesh.lineGroup), blobs.i32(mesh.triGroup), elementMaskOf(s, mesh, blobs));
  const nodes = new Set<number>();
  const elements = new Set<number>();
  const take = (map: Int32Array, per: number, vertexNodes: ArrayLike<number>, primElement: Int32Array) => {
    for (let k = 0; k < map.length; k++) {
      const p = map[k]!;
      elements.add(primElement[p]!);
      for (let v = 0; v < per; v++) nodes.add(vertexNodes[p * per + v]!);
    }
  };
  take(lineMap, 2, blobs.get(mesh.vertexNodes.line), blobs.i32(mesh.lineElement));
  take(triMap, 3, blobs.get(mesh.vertexNodes.tri), blobs.i32(mesh.triElement));
  return { nodes, elements, femIds: meshFemIds(mesh.elements, s.decls) };
}

type DrawnContour = Extract<ContourView, { state: "ready" | "element" }>;
/** A field's numbers before the section planes (`drawnContour` applies them). */
interface Evidence {
  extrema: { recorded: ManifestExtrema | null; drawn: ManifestExtrema | null };
  no_data: { kind: "nodes" | "cells"; count: number; of: number };
}

/** Whether a section plane is cutting the view, as the renderer's cuts read it (state/clip.ts). */
const cutting = (s: State): boolean => s.clip.applyCuts && activeCount(s.clip) > 0;
const CUT_EXTREMA = "section planes are cutting this view, and the manifest does not compute what a cut shows; `extrema.recorded` stands, and `shot.clip` names the planes";
const CUT_COUNT = "counted before the section planes: a node or cell they cut away is still counted";

function row(step: StepValues, key: string): Float64Array {
  const r = step.nodal[key];
  if (!(r instanceof Float64Array)) throw new Error(`manifest: the step data has no ${key}`);
  return r;
}

function nodalEvidence(v: Extract<ContourView, { state: "ready" }>, mesh: MeshInfo, blobs: BlobStore, step: StepValues, seen: ReturnType<typeof visibleOf>): Evidence {
  const ids = blobs.get(v.nodeIds);
  if (!(ids instanceof Float64Array)) throw new TypeError(`manifest: ${v.nodeIds.id} is not f64`);
  const column = columnOf(ids);
  const values = vertexValues(
    ids,
    column,
    v.component.datasets.map((d) => row(step, d)),
    v.component.magnitude,
  );
  const c: Candidates = { count: ids.length, value: (i) => values[i]!, node: (i) => ids[i]!, element: () => null, visible: (i) => seen.nodes.has(ids[i]!) };
  const at = nodePosition(mesh, blobs);
  let missing = 0;
  for (const id of seen.nodes) {
    const j = column.get(id);
    if (j === undefined || !Number.isFinite(values[j]!)) missing++;
  }
  return {
    extrema: { recorded: extremaOf(c, () => true, (i) => at(ids[i]!)), drawn: extremaOf(c, c.visible, (i) => at(ids[i]!)) },
    no_data: { kind: "nodes", count: missing, of: seen.nodes.size },
  };
}

function gaussBlocks(v: Extract<ContourView, { state: "element" }>, step: StepValues): Float64Array[] {
  if (step.element !== v.component.key || !step.gauss) throw new Error(`manifest: the step data holds ${step.element ?? "no element component"}, not ${v.component.key}`);
  return v.groups.map((g) => {
    const vals = step.gauss![g.id];
    if (!(vals instanceof Float64Array)) throw new Error(`manifest: the step data has no ${v.component.key} for group ${g.id}`);
    return vals;
  });
}

/** How many visible cells have no finite value, given the value of each FEM id that has one. */
function missingCells(seen: ReturnType<typeof visibleOf>, hasValue: ReadonlySet<number>): number {
  let missing = 0;
  for (const e of seen.elements) if (!hasValue.has(seen.femIds[e]!)) missing++;
  return missing;
}

function elementEvidence(s: State, v: Extract<ContourView, { state: "element" }>, mesh: MeshInfo, blobs: BlobStore, step: StepValues, seen: ReturnType<typeof visibleOf>): Evidence {
  const blocks = gaussBlocks(v, step);
  const at = nodePosition(mesh, blobs);
  const cells = new Set<number>();
  for (const e of seen.elements) cells.add(seen.femIds[e]!);
  const eids = v.groups.map((g) => {
    const x = blobs.get(g.eids);
    if (!(x instanceof Float64Array)) throw new TypeError(`manifest: ${g.eids.id} is not f64`);
    return x;
  });
  if (isNodeTopology(v.topology)) {
    const plan = planFor({ ...v, topology: v.topology }, s, blobs);
    const r = applyPlan(plan, blocks);
    if (r.nodal) {
      const nodal = r.nodal;
      const slot = plan.nodeIds;
      const c: Candidates = { count: slot.length, value: (i) => nodal[i]!, node: (i) => slot[i]!, element: () => null, visible: (i) => seen.nodes.has(slot[i]!) };
      const column = columnOf(slot);
      let missing = 0;
      for (const id of seen.nodes) {
        const j = column.get(id);
        if (j === undefined || !Number.isFinite(nodal[j]!)) missing++;
      }
      return {
        extrema: { recorded: extremaOf(c, () => true, (i) => at(slot[i]!)), drawn: extremaOf(c, c.visible, (i) => at(slot[i]!)) },
        no_data: { kind: "nodes", count: missing, of: seen.nodes.size },
      };
    }
    // Discrete: every corner of every element keeps its own value.
    const corner = r.corner;
    const owner = new Float64Array(corner.length);
    const withValue = new Set<number>();
    plan.groups.forEach((g, gi) => {
      for (let e = 0; e < g.nElements; e++) {
        owner.fill(eids[gi]![e]!, g.start[e]!, g.start[e + 1]!);
        if (g.start[e + 1]! > g.start[e]! && Number.isFinite(corner[g.start[e]!]!)) withValue.add(eids[gi]![e]!);
      }
    });
    const nodeAt = (i: number) => plan.nodeIds[plan.cornerSlot[i]!]!;
    const c: Candidates = { count: corner.length, value: (i) => corner[i]!, node: nodeAt, element: (i) => owner[i]!, visible: (i) => cells.has(owner[i]!) && seen.nodes.has(nodeAt(i)) };
    return {
      extrema: { recorded: extremaOf(c, () => true, (i) => at(nodeAt(i))), drawn: extremaOf(c, c.visible, (i) => at(nodeAt(i))) },
      no_data: { kind: "cells", count: missingCells(seen, withValue), of: seen.elements.size },
    };
  }
  // Flat: one value per element, the row's own or the mean of its Gauss points.
  const owner: number[] = [];
  const values: number[] = [];
  v.groups.forEach((g, gi) => {
    const cellValues = gaussCellValues(blocks[gi]!, g.nElements, g.nGauss);
    for (let e = 0; e < g.nElements; e++) {
      owner.push(eids[gi]![e]!);
      values.push(cellValues[e]!);
    }
  });
  const withValue = new Set(owner.filter((_, i) => Number.isFinite(values[i]!)));
  const cell = cellLookup(s.decls, s.blocks, blobs);
  const centroid = (i: number): Vec3 => {
    const hit = cell(owner[i]!);
    if (!hit) throw new Error(`manifest: element ${owner[i]} has a value, and the model has no such cell`);
    const sum: Vec3 = [0, 0, 0];
    for (let n = 0; n < hit.nodes.length; n++) {
      const p = at(hit.nodes[n]!);
      sum[0] += p[0], sum[1] += p[1], sum[2] += p[2];
    }
    return [sum[0] / hit.nodes.length, sum[1] / hit.nodes.length, sum[2] / hit.nodes.length];
  };
  const c: Candidates = { count: owner.length, value: (i) => values[i]!, node: () => null, element: (i) => owner[i]!, visible: (i) => cells.has(owner[i]!) };
  return {
    extrema: { recorded: extremaOf(c, () => true, centroid), drawn: extremaOf(c, c.visible, centroid) },
    no_data: { kind: "cells", count: missingCells(seen, withValue), of: seen.elements.size },
  };
}

function contourOf(s: State, blobs: BlobStore, step: StepValues | null, combination: CombinationColumnOf | null): ManifestContour | null {
  const v = contourView(s);
  if (v.state === "off") return null;
  if (v.state === "unavailable" || v.state === "refused" || v.state === "not-yet") return { drawn: false, reason: v.reason };
  if (v.state === "combination") return combinationContour(s, v, blobs, combination);
  return drawnContour(s, v, blobs, step);
}

/** VU7b-2: a User Combination's contour, from the column on screen (one state: no step, no history scan). */
function combinationContour(s: State, v: Extract<ContourView, { state: "combination" }>, blobs: BlobStore, held: CombinationColumnOf | null): ManifestContour {
  if (!held || held.key !== v.key) throw new Error(`manifest: the combination column on hand is not the read the view shows (${held ? held.key : "none"})`);
  const mesh = s.mesh;
  if (!mesh) throw new Error("manifest: a contour is drawn with no mesh");
  const seen = visibleOf(s, mesh, blobs);
  const { ids, values } = held.column;
  const column = columnOf(ids);
  const c: Candidates = { count: ids.length, value: (i) => values[i]!, node: (i) => ids[i]!, element: () => null, visible: (i) => seen.nodes.has(ids[i]!) };
  const at = nodePosition(mesh, blobs);
  let missing = 0;
  for (const id of seen.nodes) {
    const j = column.get(id);
    if (j === undefined || !Number.isFinite(values[j]!)) missing++;
  }
  const drawn = extremaOf(c, c.visible, (i) => at(ids[i]!));
  const cut = cutting(s);
  return {
    drawn: true,
    field: v.field,
    component: v.component.key,
    label: v.component.label,
    source: "combination",
    combination: { name: v.name, signed: v.signed },
    topology: "nodal",
    cmap: v.cmap,
    // One state: the range is the user's (fixed) or the combination's own, never a step's or a history's.
    range: { min: v.clim.min, max: v.clim.max, mode: v.clim.mode === "fixed" ? "fixed" : "state", label: v.clim.label, history_scan: { status: "none", done: 0, total: 0 } },
    extrema: {
      recorded: extremaOf(c, () => true, (i) => at(ids[i]!)),
      drawn: cut ? null : drawn,
      drawn_reason: cut ? CUT_EXTREMA : drawn ? null : "no drawn value is finite",
    },
    no_data: { kind: "nodes", count: missing, of: seen.nodes.size, limit: cut ? CUT_COUNT : null },
  };
}

function drawnContour(s: State, v: DrawnContour, blobs: BlobStore, step: StepValues | null): ManifestContour {
  const at = s.phase.at;
  if (at?.kind !== "results" || !step || step.stage !== at.stage || step.step !== at.step) {
    throw new Error(`manifest: the step data on hand is not the step the view shows (${step ? `stage ${step.stage} step ${step.step}` : "none"})`);
  }
  const mesh = s.mesh;
  if (!mesh) throw new Error("manifest: a contour is drawn with no mesh");
  const seen = visibleOf(s, mesh, blobs);
  const evidence = v.state === "ready" ? nodalEvidence(v, mesh, blobs, step, seen) : elementEvidence(s, v, mesh, blobs, step, seen);
  const scan = s.contour.scan?.key === v.keys.scan ? s.contour.scan : null;
  const cut = cutting(s);
  return {
    drawn: true,
    field: v.field,
    component: v.component.key,
    label: v.component.label,
    source: v.state === "ready" ? "nodal" : "element",
    topology: v.state === "ready" ? "nodal" : v.topology,
    cmap: v.cmap,
    range: {
      min: v.clim.min,
      max: v.clim.max,
      mode: v.clim.mode,
      label: v.clim.label,
      history_scan: { status: scan?.status ?? "none", done: scan?.done ?? 0, total: scan?.total ?? 0 },
    },
    extrema: {
      recorded: evidence.extrema.recorded,
      drawn: cut ? null : evidence.extrema.drawn,
      drawn_reason: cut ? CUT_EXTREMA : evidence.extrema.drawn ? null : "no drawn value is finite",
    },
    no_data: { ...evidence.no_data, limit: cut ? CUT_COUNT : null },
  };
}

function deformOf(s: State): ManifestDeform | null {
  const v = deformView(s);
  const limit = cutting(s) ? CUT_COUNT : null;
  switch (v.state) {
    case "off":
      return null;
    case "unavailable":
    case "refused":
      return { state: v.state, reason: v.reason };
    case "loading":
      return { state: v.state, scale: v.scale, ghost: v.ghost, drawn: null, undeformed: null, limit };
    case "ready":
      return { state: v.state, scale: v.scale, ghost: v.ghost, drawn: v.drawn, undeformed: 0, limit };
    case "partial":
      return { state: v.state, scale: v.scale, ghost: v.ghost, drawn: v.drawn, undeformed: v.missing, limit };
    default: {
      const unknown: never = v;
      throw new Error(`manifest: unknown deformed view ${JSON.stringify(unknown)}`);
    }
  }
}

function placeOf(s: State): Pick<ManifestEntry, "stage" | "step" | "time"> {
  const at = s.phase.at;
  if (at?.kind !== "results") return { stage: null, step: null, time: null };
  const st = s.results?.stages[at.stage];
  if (!st) throw new Error(`manifest: results stage ${at.stage} is not in the file`);
  const time = st.time[at.step];
  if (time === undefined) throw new Error(`manifest: stage ${st.name || st.id} has no step ${at.step}`);
  return { stage: { index: at.stage, id: st.id, name: st.name }, step: at.step, time };
}

/**
 * One drawn view's D3 entry, from the state and the shot (ADR 0128 D3). The
 * state must be the view being captured: `step` is the step on screen, and
 * the shot says what was asked for. Throws when it cannot state what the view
 * shows, before any still is written (see the header).
 */
export function manifestEntry(s: State, input: EntryInput): ManifestEntry {
  return {
    name: input.shot.name,
    status: "ok",
    reason: null,
    png: input.png,
    image: input.image,
    theme: s.theme,
    shot: input.shot,
    camera: input.camera,
    files: filesOf(s),
    pairing: describeSet(s, []).pairing,
    ...placeOf(s),
    contour: contourOf(s, input.blobs, input.step, input.combination ?? null),
    deform: deformOf(s),
    banner: bannerLines(s),
    layout: input.layout,
  };
}

/**
 * A refused shot's entry (D2): the app's reason and no still. A shot that
 * compiled names the stage and step it asked for, and the banner of its view;
 * one that did not (`at` null) applied nothing, so it has no view to report.
 */
export function refusedEntry(
  s: State,
  shot: SpecShot,
  reason: string,
  at: { stage: ManifestEntry["stage"]; step: number | null; time: number | null } | null,
): ManifestEntry {
  return {
    name: shot.name,
    status: "refused",
    reason,
    png: null,
    image: null,
    theme: s.theme,
    shot,
    camera: null,
    files: filesOf(s),
    pairing: describeSet(s, []).pairing,
    stage: at?.stage ?? null,
    step: at?.step ?? null,
    time: at?.time ?? null,
    contour: null,
    deform: null,
    banner: at ? bannerLines(s) : [],
    layout: null,
  };
}

/** The run's manifest; main adds the app's version and the time it was written. */
export function manifestOf(spec: ViewSpec, shots: readonly { entry: ManifestEntry; ms: number; shownMs: number | null }[]): Manifest {
  return {
    manifest: MANIFEST_VERSION,
    spec: spec.spec,
    size: spec.size,
    theme: spec.theme,
    shots: shots.map((x) => x.entry),
    timing: { shots: Object.fromEntries(shots.map((x) => [x.entry.name, { ms: x.ms, shown_ms: x.shownMs }])) },
  };
}

/**
 * The manifest of a spec refused whole (D2): the key and the reason, and no
 * shot. `spec` is null: the document was not a valid spec. `app` is what
 * `appVersion` (./job.ts) read, at startup, before any window.
 */
export function refusedManifest(specPath: string, key: string, reason: string, app: { version: string | null; version_reason: string | null }, writtenAt: string): string {
  return JSON.stringify({ manifest: MANIFEST_VERSION, spec: null, spec_file: specPath, refused: { key, reason }, shots: [], app, timing: { shots: {}, written_at: writtenAt } }, null, 2);
}
