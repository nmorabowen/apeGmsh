// Selectors: pure reads of the state that panels and the viewport render
// from. A panel imports this module and store.ts, nothing else of the app.

import type { Chain, ChainNode, Field } from "../chain/resolve.ts";
import type { OpsFamily, Param } from "../model/types.ts";
import { elementFields, elementLinks, type Lookup } from "./decls.ts";
import { pairSessions } from "../reader/geometry.ts";
import { PROVENANCE_ORIGIN_FROM } from "../reader/provenance.ts";
import { roleColouring } from "./roles.ts";
import { activationNotice, activationOf, type Activation } from "./activation.ts";
import { pairingNotice, pairResults, type ResultsPairing } from "./results.ts";
import {
  climOf,
  componentName,
  componentOf,
  contourKeysOf,
  elementComponentOf,
  gaussFields,
  elementTopology,
  fieldsOf,
  isNodeTopology,
  topologyLabel,
  type Clim,
  type ElementComponent,
  type ElementField,
  type ElementTopology,
  type FieldComponent,
  type NodalField,
} from "./contour.ts";
import { usesMatrix } from "./extrapolation.ts";
import type { BlobRef, ContourCmap } from "./types.ts";
import { displacementAxes } from "./deform.ts";

export type { Clim, ElementComponent, ElementField, FieldComponent, NodalField } from "./contour.ts";
export { GHOST_OPACITY, scaleLabel } from "./deform.ts";
export { CMAP_NAMES, CMAPS, MAGNITUDE, NO_DATA_RGB, RANGE_MODES, sampleCmap } from "./contour.ts";

export type { Activation } from "./activation.ts";
export type { ResultsPairing } from "./results.ts";
import type {
  AnalysisDef,
  ArtifactInfo,
  Decl,
  DeclPath,
  DeclRecord,
  DeclSource,
  DeclValue,
  Definitions,
  DefObject,
  LegendEntry,
  Origin,
  Pick,
  ResultsInfo,
  ResultsStage,
  SequenceZone,
  StageDef,
  State,
  WindowContent,
  WindowKind,
  WindowRow,
  ZoneStatus,
} from "./types.ts";

function lookupOf(s: State): Lookup {
  // Tags are not keys of the state; an element's links are followed by the
  // same table the loader used, over the objects the state holds.
  let byTag: Map<string, DeclPath[]> | null = null;
  let byH5: Map<string, DeclPath> | null = null;
  const build = () => {
    byTag = new Map();
    byH5 = new Map();
    for (const d of Object.values(s.decls)) {
      if (d.tag === null) continue;
      const key = `${d.kind}:${d.tag}`;
      (byTag.get(key) ?? byTag.set(key, []).get(key)!).push(d.path);
      byH5.set(d.h5, d.path);
    }
  };
  return {
    byTag: (family: OpsFamily, tag: number) => {
      if (!byTag) build();
      return byTag!.get(`${family}:${tag}`) ?? [];
    },
    byH5: (path: string) => {
      if (!byH5) build();
      return byH5!.get(path) ?? null;
    },
  };
}

/** A declaration as a chain node, with the links that could be followed as children. */
function nodeOf(s: State, d: Decl, via: Field | null, problems: string[], seen: Set<DeclPath>, nodes: readonly number[] | null): ChainNode {
  let fields: readonly Field[] = d.fields, links = d.links, refs = d.refs;
  if (d.element) {
    const hasOpensees = s.artifacts.model?.zones["opensees"]?.status === "ready";
    const l = elementLinks(d.element, hasOpensees, lookupOf(s));
    const npe = d.element.cell ? (s.blocks[d.element.cell.block]?.npe ?? null) : null;
    fields = elementFields(d.element, d.element.groups.map((g) => s.decls[g]?.name ?? g), nodes, npe);
    links = l.links;
    refs = l.refs;
    problems.push(...l.problems);
  } else {
    problems.push(...d.problems);
  }
  const children: ChainNode[] = [];
  for (const [key, to] of Object.entries(refs)) {
    const child = s.decls[to];
    if (!child) throw new Error(`${d.path}: ref ${key} points at ${to}, which is not a declaration`);
    const link = links[key];
    if (!link) throw new Error(`${d.path}: ref ${key} has no link field`);
    if (seen.has(to)) {
      problems.push(`${d.path}: ${key} -> ${to} closes a cycle`);
      continue;
    }
    children.push(nodeOf(s, child, link, problems, new Set([...seen, to]), null));
  }
  if (d.kind === "physical_group" || d.kind === "label") throw new Error(`${d.path}: a ${d.kind} has no definition chain`);
  return {
    role: d.kind,
    name: d.name,
    nameSource: d.nameSource,
    type: d.type,
    path: d.h5,
    via,
    fields: [...fields],
    children,
  };
}

/**
 * The definition chain of an element or an OpenSees object, or null when the
 * path is not declared or is a group (a group has members, not a chain).
 */
export function chainOf(s: State, path: DeclPath, nodes: readonly number[] | null = null): Chain | null {
  const d = s.decls[path];
  if (!d || d.kind === "physical_group" || d.kind === "label") return null;
  const problems: string[] = [];
  const root = nodeOf(s, d, null, problems, new Set([path]), nodes);
  return { root, problems };
}

/** The pick of a selected declaration, if it is selected (its node tags come from it). */
const pickOf = (s: State, decl: DeclPath): Pick | undefined => s.selection.picks.find((p) => p.decl === decl);

/** What the inspector shows: the pinned declarations, then the selection. */
export function inspected(s: State): { decl: DeclPath; pinned: boolean; chain: Chain }[] {
  const out: { decl: DeclPath; pinned: boolean; chain: Chain }[] = [];
  for (const p of s.inspector.pinned) {
    const chain = chainOf(s, p, pickOf(s, p)?.nodes ?? null);
    if (chain) out.push({ decl: p, pinned: true, chain });
  }
  for (const p of s.selection.decls) {
    if (s.inspector.pinned.includes(p)) continue;
    const chain = chainOf(s, p, pickOf(s, p)?.nodes ?? null);
    if (chain) out.push({ decl: p, pinned: false, chain });
  }
  return out;
}

export const legendOf = (s: State): LegendEntry[] => s.mesh?.legend ?? [];

/** The colour-by-role legend: the roles the file records, the unassigned row, and whether it records any. */
export const roleLegendOf = (s: State): { legend: LegendEntry[]; fileHasRoles: boolean } | null => {
  const r = roleColouring(s);
  return r ? { legend: r.legend, fileHasRoles: r.fileHasRoles } : null;
};

export const isHidden = (s: State, decl: DeclPath | null): boolean => decl !== null && s.visibility.hidden.includes(decl);

/** The header's summary of the model artifact, or null when none is loaded. */
export function modelSummary(s: State): { name: string; stats: string } | null {
  const a = s.artifacts.model;
  if (!a || a.status !== "ready") return null;
  const v = (z: string) => {
    const st = a.zones[z];
    return st?.status === "ready" ? st.version : null;
  };
  const ops = v("opensees");
  return {
    name: a.path.split(/[\\/]/).pop() ?? a.path,
    stats:
      `${a.counts.nodes.toLocaleString()} nodes · ${a.counts.cells.toLocaleString()} cells` +
      (a.counts.opsOnly ? ` (+${a.counts.opsOnly.toLocaleString()} OpenSees-only)` : "") +
      ` · neutral ${v("neutral") ?? "?"}` +
      (ops ? ` · opensees ${ops}` : " · no /opensees"),
  };
}

/** Loud reader and mesh warnings of every artifact. */
export function warningsOf(s: State): string[] {
  const out: string[] = [];
  for (const a of Object.values(s.artifacts)) if (a) out.push(...a.warnings);
  if (s.mesh) out.push(...s.mesh.warnings);
  return out;
}

/** The zones a reader refused, as one sentence each (never a blank view). */
export function refusalsOf(s: State): string[] {
  const out: string[] = [];
  for (const [kind, a] of Object.entries(s.artifacts) as [string, ArtifactInfo | null][]) {
    if (!a) continue;
    for (const [zone, st] of Object.entries(a.zones) as [string, ZoneStatus][]) {
      if (st.status !== "refused") continue;
      out.push(
        `${kind} file written by ${st.newer ? "a newer" : "an older"} apeGmsh: its ${zone} zone is version ${st.version}, ` +
          `this app reads ${st.accepted} (${st.reason})`,
      );
    }
  }
  return out;
}

const baseName = (p: string): string => p.split(/[\\/]/).pop() ?? p;

/**
 * Whether the geometry sibling may be drawn, and the notice when it may not
 * (h5-schema.md, the pairing rule): it pairs with the model only when both
 * carry an equal `/meta/session_id`. With no model open it is drawn alone;
 * while the model is still loading it waits, with no notice.
 */
export function geometryPairing(s: State): { draw: boolean; notice: string | null } {
  const g = s.geometry;
  if (!g) return { draw: false, notice: null };
  const m = s.artifacts.model;
  if (!m) return { draw: true, notice: null };
  if (m.status !== "ready") return { draw: false, notice: null };
  const p = pairSessions(m.sessionId, g.sessionId, baseName(g.path));
  return p.paired ? { draw: true, notice: null } : { draw: false, notice: `Geometry not drawn: ${p.reason}.` };
}

/** The rows of `/sequence/stages`: what `State.stage` indexes (0 when the zone is not ready). */
export const stageCount = (seq: SequenceZone): number => (seq.status === "ready" ? seq.stages.rows.length : 0);

/**
 * What a `/sequence/stages` row is, to tell whether a re-read still has the
 * same stage at that index: its `order`, `name` and `source` cells (the
 * whole row when a column is missing). Null when the index is not a row.
 */
export function stageIdentity(seq: SequenceZone, index: number): string | null {
  if (seq.status !== "ready" || !Number.isInteger(index) || index < 0 || index >= seq.stages.rows.length) return null;
  const row = seq.stages.rows[index]!;
  const at = ["order", "name", "source"].map((c) => seq.stages.columns.indexOf(c));
  return JSON.stringify(at.includes(-1) ? row : at.map((i) => row[i]));
}

/** Notices that are not warnings: the stale-geometry notice, and (V4b) why results or their stage activation are not drawn. */
export function noticesOf(s: State): string[] {
  const out: string[] = [];
  const n = geometryPairing(s).notice;
  if (n) out.push(n);
  // While the model of the set is still being read, "no model is open" would only flash.
  if (s.results && s.artifacts.model?.status !== "opened") {
    const p = pairingNotice(resultsPairing(s), baseName(s.results.path));
    if (p) out.push(p);
  }
  const a = activationNow(s);
  if (a) {
    const t = activationNotice(a.activation, a.stageName);
    if (t) out.push(t);
  }
  // V4c1: nodal components refused one by one on the stage shown, and a contour pick a re-read cleared.
  const at = s.phase.at;
  const st = at?.kind === "results" ? s.results?.stages[at.stage] : undefined;
  for (const r of st?.refusedNodal ?? []) out.push(`Nodal component ${r.component} not read: ${r.reason}.`);
  // V4c2a: element components refused one by one on the stage shown, and why a picked element field is not drawn.
  for (const g of st?.gauss ?? []) for (const r of g.refused) out.push(`Element component ${r.component} of group ${g.id} not read: ${r.reason}.`);
  const ev = contourView(s);
  if (ev.state === "refused") out.push(`Element contour refused: ${ev.reason}.`);
  if (s.contour.notice) out.push(s.contour.notice);
  // V4d: drawn nodes the deformed shape leaves undeformed, or why it is not drawn.
  const d = deformNotice(s);
  if (d) out.push(d);
  return out;
}

// -- V4c1: the nodal contour ----------------------------------------------------

/**
 * Why an element contour is refused (V4c2a): each is drawn as no data, and
 * the banner says why. V4c2b: `unsupported`, a node-located topology the
 * field's element types or natural coordinates cannot be carried through.
 */
export type ElementRefusal = "unverified" | "components" | "unknown-id" | "duplicate-id" | "unsupported";

/** One gauss group an element contour reads: its id, size, and its element ids (blob). */
export interface ElementGroupDraw {
  id: string;
  nElements: number;
  nGauss: number;
  eids: BlobRef;
  /** V4c2b: the group's `_natural_coords` (null: not floating point) */
  naturalCoords: number[][] | null;
}

export type ContourView =
  | { state: "off" }
  /** colouring by contour, and nothing can be coloured: the sentence says why (the mesh is drawn as no data) */
  | { state: "unavailable"; reason: string }
  /**
   * V4c2a: an element field that must not be drawn: an unverified file (its
   * rows are not proved to be FEM ids), groups that disagree on the field's
   * components, or rows that do not join the mesh. Drawn as no data, never
   * "with a warning"; the banner says why.
   */
  | { state: "refused"; code: ElementRefusal; reason: string }
  /**
   * V4c2a: an element topology whose drawing was not built yet. V4c2b draws
   * every topology, so no view is in this state any more; the variant stays
   * because renderer/viewport.ts (outside this slice) still narrows on it.
   */
  | { state: "not-yet"; topology: "gauss_cell_averaged" | "gauss_node" | "gauss_node_discrete"; reason: string }
  /** V4c2a: an element field, flat per cell; V4c2b: or carried to the nodes */
  | {
      state: "element";
      field: string;
      component: ElementComponent;
      topology: ElementTopology;
      /** what the legend states for the topology (the averaging, in words) */
      topologyLabel: string;
      nGauss: number[];
      groups: ElementGroupDraw[];
      stageName: string;
      keys: { step: string; scan: string };
      /** selectors `elementJoinKey`: the join the colours go through */
      joinKey: string;
      clim: Clim;
      cmap: ContourCmap;
      stepMeasured: boolean;
      reads: number;
    }
  | {
      state: "ready";
      field: string;
      component: FieldComponent;
      stageName: string;
      keys: { step: string; scan: string };
      clim: Clim;
      cmap: ContourCmap;
      /** the stage's `_ids` blob */
      nodeIds: BlobRef;
      /** the measured range is for the step shown (else it is still loading) */
      stepMeasured: boolean;
      /** how many times the results were read (each read can carry other data at the same path, so the colour key names it) */
      reads: number;
    };

/** A results file's element-id verdict; a results info without one is unverified (fail closed). */
export function elementVerdictOf(info: ResultsInfo): { verified: true } | { verified: false; reason: string } {
  return info.elementIndex ?? { verified: false, reason: "this results info carries no element-index verdict" };
}

/**
 * The key of an element field's join to the mesh (V4c2a): the results read
 * (a re-read is joined again), the file, the stage, the field, the mesh and
 * the groups' element-id blobs. Null when no verified element field is picked.
 */
export function elementJoinKey(s: State): string | null {
  const at = s.phase.at;
  const c = s.contour;
  if (c.source !== "element" || c.field === null || at?.kind !== "results" || !s.results || !s.mesh || !elementVerdictOf(s.results).verified) return null;
  const st = s.results.stages[at.stage];
  const f = st ? gaussFields(st.gauss).find((x) => x.name === c.field) : undefined;
  if (!st || !f || f.refused !== null) return null;
  const ids = f.groups.map((gid) => st.gauss.find((g) => g.id === gid)?.elementIds?.id ?? "?");
  return `read ${c.reads}|${s.results.path}|${st.id}|elem:${f.name}|mesh ${s.mesh.triElement.id}|${ids.join(",")}`;
}

/** What an element field colours (V4c2a), or why it cannot; `contourView` calls it once the stage is known. */
function elementView(s: State, st: ResultsStage, label: string): ContourView {
  const c = s.contour;
  if (c.field === null || c.component === null) return { state: "unavailable", reason: "pick a field" };
  const hit = elementComponentOf(st, c.field, c.component);
  if (!hit) return { state: "unavailable", reason: `stage ${label} has no element field ${c.field} ${componentName(c.component)}` };
  const v = elementVerdictOf(s.results!);
  if (!v.verified) {
    return {
      state: "refused",
      code: "unverified",
      reason: `${baseName(s.results!.path)} does not prove which elements its rows are, so its element results are not drawn (${v.reason})`,
    };
  }
  if (hit.field.refused !== null) return { state: "refused", code: "components", reason: `element field ${c.field}: ${hit.field.refused}` };
  const topology = elementTopology(c.gauss, hit.field.nGauss);
  if (!s.mesh) return { state: "unavailable", reason: "no mesh is loaded" };
  const groups: ElementGroupDraw[] = [];
  for (const gid of hit.field.groups) {
    const g = st.gauss.find((x) => x.id === gid)!;
    if (!g.elementIds) return { state: "unavailable", reason: `the element ids of group ${gid} are not loaded` };
    groups.push({ id: g.id, nElements: g.nElements, nGauss: g.nGauss, eids: g.elementIds, naturalCoords: g.naturalCoords });
  }
  const joinKey = elementJoinKey(s)!;
  const j = c.join;
  if (!j || j.key !== joinKey) return { state: "unavailable", reason: "the element rows are being joined to the mesh" };
  if (!j.ok) return { state: "refused", code: j.code, reason: `element field ${c.field}: ${j.reason}` };
  // V4c2b: a node-located topology the field's types or natural coordinates cannot carry reads "unsupported", never a guess.
  if (isNodeTopology(topology)) {
    const why = j.extrapolation.corners ?? (usesMatrix(topology) ? j.extrapolation.matrix : null);
    if (why !== null) return { state: "refused", code: "unsupported", reason: `element field ${c.field}: ${topology} is unsupported: ${why}` };
  }
  const keys = contourKeysOf(s)!;
  const measured = c.stepRange !== null && c.stepRange.key === keys.step;
  return {
    state: "element",
    field: c.field,
    component: hit.component,
    topology,
    topologyLabel: topologyLabel(topology, hit.field.nGauss),
    nGauss: hit.field.nGauss,
    groups,
    stageName: label,
    keys,
    joinKey,
    clim: climOf(c, measured ? c.stepRange : null, c.scan, keys.scan),
    cmap: c.cmap,
    stepMeasured: measured,
    reads: c.reads,
  };
}

/** What the contour colours, or why it cannot (V4c1; V4c2a adds element fields). */
export function contourView(s: State): ContourView {
  if (s.visibility.colourBy !== "contour") return { state: "off" };
  const p = resultsPairing(s);
  if (!s.results || p.state !== "paired") return { state: "unavailable", reason: "no results file pairs with this model, so there is no field to colour by" };
  const at = s.phase.at;
  if (at?.kind !== "results") return { state: "unavailable", reason: "contours colour a results step: choose results in the phase bar" };
  const st = s.results.stages[at.stage];
  if (!st) return { state: "unavailable", reason: `results stage ${at.stage} is not in the file` };
  const label = st.name || st.id;
  if (s.contour.source === "element") return elementView(s, st, label);
  if (fieldsOf(st).length === 0) return { state: "unavailable", reason: `stage ${label} records no nodal results` };
  const c = s.contour;
  if (c.field === null || c.component === null) return { state: "unavailable", reason: "pick a field" };
  const hit = componentOf(st, c.field, c.component);
  if (!hit) return { state: "unavailable", reason: `stage ${label} has no ${c.field} ${componentName(c.component)}` };
  if (!st.nodeIds) return { state: "unavailable", reason: `the node ids of stage ${label} are not loaded` };
  if (!s.mesh) return { state: "unavailable", reason: "no mesh is loaded" };
  const keys = contourKeysOf(s)!;
  const measured = c.stepRange !== null && c.stepRange.key === keys.step;
  const clim = climOf(c, measured ? c.stepRange : null, c.scan, keys.scan);
  return {
    state: "ready",
    field: c.field,
    component: hit.component,
    stageName: label,
    keys,
    clim,
    cmap: c.cmap,
    nodeIds: st.nodeIds,
    stepMeasured: measured,
    reads: c.reads,
  };
}

/**
 * The field picker's rows: the nodal fields of the results stage shown, and
 * the refused components; V4c2a: its element fields (a refused one says why),
 * and whether the file's element rows are verified.
 */
export function contourFieldsOf(s: State): {
  fields: NodalField[];
  refused: { component: string; reason: string }[];
  elements: ElementField[];
  elementsVerified: boolean;
} | null {
  const at = s.phase.at;
  const st = at?.kind === "results" ? s.results?.stages[at.stage] : undefined;
  return st ? { fields: fieldsOf(st), refused: st.refusedNodal, elements: gaussFields(st.gauss), elementsVerified: elementVerdictOf(s.results!).verified } : null;
}

// -- V4d: the deformed shape ----------------------------------------------------

/** What the deformed shape draws, besides its state: the datasets, the scale, the ghost, and the keys of the step shown. */
export interface DeformDraw {
  /** the x, y and z displacement datasets; z null in a 2-D model */
  datasets: readonly [string, string, string | null];
  /** a sentence about the axes (the 2-D case), or null */
  note: string | null;
  scale: number;
  ghost: boolean;
  stageName: string;
  /** the stage's `_ids` blob */
  nodeIds: BlobRef;
  /**
   * `disp` names the displacement of the step shown: the results read
   * (V4c1's count: a re-read at the same path may hold other values), the
   * file, the stage, the step, the mesh and the datasets. `count` adds what
   * is drawn (the hidden set), which the missing-node count depends on.
   */
  keys: { disp: string; count: string };
}

export type DeformView =
  | { state: "off" }
  /** on, and there is nothing to deform by (no results step, no displacement): the mesh is drawn undeformed */
  | { state: "unavailable"; reason: string }
  /** on, and the displacement cannot be used (a refused axis, unnamed components, no z in a model not declared 2-D) */
  | { state: "refused"; reason: string }
  /** the drawn nodes are not counted yet for the step shown */
  | ({ state: "loading" } & DeformDraw)
  /** every drawn node has a displacement */
  | ({ state: "ready"; drawn: number } & DeformDraw)
  /** `missing` drawn nodes have none: drawn undeformed and marked, and the banner says how many */
  | ({ state: "partial"; drawn: number; missing: number } & DeformDraw);

/**
 * Whether the stage shown can be deformed: the displacement to use, or why
 * not (`unavailable` / `refused`). The legend disables the toggle with the
 * reason; `deformView` adds the user's toggle and the effects' count.
 */
export function deformOf(s: State): { ok: true; draw: DeformDraw } | { ok: false; state: "unavailable" | "refused"; reason: string } {
  const p = resultsPairing(s);
  if (!s.results || p.state !== "paired") return { ok: false, state: "unavailable", reason: "no results file pairs with this model, so there is no displacement" };
  const at = s.phase.at;
  if (at?.kind !== "results") return { ok: false, state: "unavailable", reason: "the deformed shape shows a results step: choose results in the phase bar" };
  const st = s.results.stages[at.stage];
  if (!st) return { ok: false, state: "unavailable", reason: `results stage ${at.stage} is not in the file` };
  if (!s.mesh) return { ok: false, state: "unavailable", reason: "no mesh is loaded" };
  const axes = displacementAxes(st, s.ndm);
  if (!axes.ok) return { ok: false, state: axes.kind === "absent" ? "unavailable" : "refused", reason: axes.reason };
  if (!st.nodeIds) return { ok: false, state: "unavailable", reason: `the node ids of stage ${st.name || st.id} are not loaded` };
  const disp = `read ${s.contour.reads}|${s.results.path}|${st.id}|${at.step}|mesh ${s.mesh.vertexNodes.tri.id}|${axes.datasets.join(",")}`;
  return {
    ok: true,
    draw: {
      datasets: axes.datasets,
      note: axes.note,
      scale: s.deform.scale,
      ghost: s.deform.ghost,
      stageName: st.name || st.id,
      nodeIds: st.nodeIds,
      keys: { disp, count: `${disp}|hidden ${s.visibility.hidden.join(",")}` },
    },
  };
}

/** What the deformed shape draws, or why it does not (V4d). */
export function deformView(s: State): DeformView {
  if (!s.deform.on) return { state: "off" };
  const d = deformOf(s);
  if (!d.ok) return { state: d.state, reason: d.reason };
  const m = s.deform.measured;
  if (!m || m.key !== d.draw.keys.count) return { state: "loading", ...d.draw };
  return m.missing > 0 ? { state: "partial", drawn: m.drawn, missing: m.missing, ...d.draw } : { state: "ready", drawn: m.drawn, ...d.draw };
}

/** The banner's line for a deformed shape that leaves drawn nodes undeformed, or that cannot be drawn; null otherwise. */
export function deformNotice(s: State): string | null {
  const v = deformView(s);
  if (v.state === "partial") {
    return (
      `Deformed shape: ${v.missing.toLocaleString()} of ${v.drawn.toLocaleString()} drawn nodes have no displacement at this step ` +
      "(not recorded, or NaN): they are drawn undeformed and marked, never moved by a displacement of 0."
    );
  }
  if (v.state === "refused") return `Deformed shape not drawn: ${v.reason}.`;
  return null;
}

// -- V4b: results pairing, the time axis and stage activation ---------------------

/** Whether the read results file pairs with the loaded model by content (state/results.ts). */
export function resultsPairing(s: State): ResultsPairing {
  const m = s.artifacts.model;
  const ops = m?.zones["opensees"];
  const zone = ops?.status === "ready";
  // No decls group in a ready opensees zone reads as "absent" (h5-schema.md): the joins are not recorded.
  const programDecl = zone ? (s.definitions?.decls ? (s.definitions.decls.programDecl ?? "absent") : "absent") : null;
  return pairResults(m?.status === "ready" ? { lineage: s.lineage, openseesVersion: zone ? ops.version : null, programDecl } : null, s.results);
}

/**
 * Whether the phase bar carries the results scrubber (results are on the axis).
 * Then it takes its own row under the header, and the banner moves below it
 * (panels/phase.ts, panels/banner.ts): the scrubber is too wide to sit beside
 * the header without covering it.
 */
export const scrubberRow = (s: State): boolean => s.phase.axis.some((k) => k.kind === "results");
/** The scrubber row's top (the header row is 14 px to about 50 px), and the banner's top below it. */
export const SCRUBBER_TOP_PX = 58;
export const BANNER_BELOW_SCRUBBER_PX = 106;

/** What the time scrubber shows:the results stage on screen, its step and time, or null off the results phases. */
export interface Timeline {
  /** the results stages, in reader order, with the index `setPhase` names */
  stages: { index: number; label: string }[];
  stage: number;
  step: number;
  steps: number;
  time: number;
  kind: "static" | "transient" | "mode";
  /** the current step's fetch: `loading`, `ready`, `failed: <why>` */
  fetch: string;
}

export function timelineOf(s: State): Timeline | null {
  const at = s.phase.at;
  if (at?.kind !== "results" || !s.results) return null;
  const st = s.results.stages[at.stage];
  if (!st) return null;
  const r = s.resultStep;
  const fetch = !r ? "loading" : r.status === "failed" ? `failed: ${r.error ?? "unknown error"}` : r.status;
  return {
    stages: s.results.stages.map((x, i) => ({ index: i, label: x.name === "" ? `${x.id} (unnamed)` : x.name })),
    stage: at.stage,
    step: at.step,
    steps: st.time.length,
    time: st.time[at.step]!,
    kind: st.kind,
    fetch,
  };
}

let activationMemo: { key: readonly unknown[]; value: { activation: Activation; stageName: string } | null } | null = null;

/** The stage activation of the results stage on screen (state/activation.ts), or null off the results phases. */
export function activationNow(s: State): { activation: Activation; stageName: string } | null {
  const at = s.phase.at;
  if (at?.kind !== "results" || !s.results) return null;
  const st = s.results.stages[at.stage];
  if (!st) return null;
  // A fem_hash pairing proves the mesh, not the program: joining /sequence
  // stage names to these results would be a guess, so it is not made.
  const p = resultsPairing(s);
  const byFem = p.state === "paired" && p.by === "fem_hash";
  const key = [s.sequence, s.stageElements, s.definitions, st.name, byFem];
  if (activationMemo && activationMemo.key.every((k, i) => k === key[i])) return activationMemo.value;
  const activation: Activation = byFem
    ? {
        state: "unbound",
        reason: "the results pair with this model by fem_hash only (the mesh matches; they carry no model_hash), so they are not tied to this model's /sequence program",
      }
    : activationOf(s.sequence, s.stageElements, s.definitions, st.name);
  const value = { activation, stageName: st.name };
  activationMemo = { key, value };
  return value;
}

let inactiveMemo: { key: readonly unknown[]; value: { mask: Uint8Array; hidden: number; inactive: number } } | null = null;

/**
 * The elements a `filtered` activation hides, as a mask over `mesh.elements`
 * (1 = not drawn); `hidden` counts the drawn elements it masks and
 * `inactive` the elements the stage does not have (a solid's interior
 * elements draw no face, so `hidden` can be the smaller). Null when the
 * phase shown has no `filtered` activation. The viewport draws through
 * renderer/visible.ts with it, the one visibility path.
 */
export function inactiveElements(s: State): { mask: Uint8Array; hidden: number; inactive: number } | null {
  const a = activationNow(s);
  if (!a || a.activation.state !== "filtered" || !s.mesh) return null;
  const key = [a.activation, s.mesh, s.decls];
  if (inactiveMemo && inactiveMemo.key.every((k, i) => k === key[i])) return inactiveMemo.value;
  const tags = a.activation.inactiveTags;
  const mask = new Uint8Array(s.mesh.elements.length);
  s.mesh.elements.forEach((p, i) => {
    for (const m of s.decls[p]?.element?.metas ?? []) if (tags.has(m.tag)) mask[i] = 1;
  });
  const value = { mask, hidden: mask.reduce((n, v) => n + v, 0), inactive: tags.size };
  inactiveMemo = { key, value };
  return value;
}

/**
 * The jump to a recorded source. A source with no digest (a pseudo-file, or a
 * file the writer could not read) opens only when main found its path on
 * disk; otherwise it is "source not recorded" and the button is off. There is
 * no edit check: with no digest there is nothing to compare.
 */
function jumpOf(src: DeclSource): { ok: true; source: DeclSource; label: string } | { ok: false; reason: string } {
  if (!src.recorded) return { ok: false, reason: `source not recorded (${src.file})` };
  return { ok: true, source: src, label: `${baseName(src.file)}:${src.line}` };
}

/**
 * Where go-to-source would jump for `decl`, or why it cannot. An unnamed
 * object's path (`#k`) is never joined: the app numbers unnamed objects in
 * the reader's listing order and /provenance in declaration order, so one
 * `#k` can name two different objects.
 */
export function sourceFor(s: State, decl: DeclPath): { ok: true; source: DeclSource; label: string } | { ok: false; reason: string } {
  const d = s.decls[decl];
  if (!d) {
    // A /provenance record with no declaration in the app (a series, a
    // pattern, a synthesised object): the listing's go-to-source.
    const r = s.provenance.find((p) => p.key === decl);
    if (!r) return { ok: false, reason: `${decl} is not a declaration of the loaded model` };
    if (!r.source) return { ok: false, reason: `${decl} has no source frame in /provenance` };
    return jumpOf(r.source);
  }
  if (d.provenance) return jumpOf(d.provenance);
  const zone = s.artifacts.model?.zones["provenance"];
  if (!zone || zone.status === "absent") return { ok: false, reason: "the model file has no /provenance zone (written before apeGmsh recorded sources)" };
  if (zone.status === "refused") return { ok: false, reason: `the /provenance zone was refused: ${zone.reason}` };
  if (zone.status === "malformed") return { ok: false, reason: `the /provenance zone is malformed: ${zone.reason}` };
  if (d.kind === "element") {
    return { ok: false, reason: "an element has no provenance key yet: /provenance records the declaration, and the file does not join it to its elements" };
  }
  if (decl.includes("/#")) {
    return { ok: false, reason: "an unnamed declaration is not joined to /provenance (the app's #k is the reader's listing order, not the declaration order); give it a name= to reach its source" };
  }
  return { ok: false, reason: `no /provenance record for ${decl}` };
}

/**
 * The declaration an OpenSees object of the chain was read from, by its HDF5
 * group (a chain node names its group, not its decl path). Objects have one
 * group each; elements share their block's and are never looked up here.
 */
export function declOfH5(s: State, h5: string): DeclPath | null {
  for (const d of Object.values(s.decls)) if (d.tag !== null && d.h5 === h5) return d.path;
  return null;
}

/** One row of the sources listing (panels/sources.ts). */
export interface SourceRow {
  key: DeclPath;
  origin: Origin;
  /** `<verb>` of a synthesised key (`support` in `support:<stage>/hold`); null for the user's own declarations */
  verb: string | null;
  /**
   * The key cut after each `/`, the only places it may wrap: joined they are
   * the key; the first is the zone prefix (`opensees/`).
   */
  segments: string[];
  /** `file:line` of the go-to-source target, or null when the record has no frame */
  label: string | null;
  /** the button's tooltip: the absolute file, the line and the function; or the reason it is off */
  title: string;
  /** why go-to-source is off for this row (shown on a second line), or null when it can jump */
  off: string | null;
}

/** `a/b:c/d` -> [`a/`, `b:c/`, `d`]: the pieces between which a path may wrap. */
export function pathSegments(key: string): string[] {
  return key.match(/[^/]*\/|[^/]+$/g) ?? [];
}

/**
 * The sources listing: every /provenance record of the model, in capture
 * order. Synthesised objects are listed by default, marked by `origin`
 * (maintainer ruling on #1378); there is no filter that hides them.
 */
export function sourcesOf(s: State): SourceRow[] {
  return s.provenance.map((p) => {
    const where = sourceFor(s, p.key);
    const name = p.key.split("/").slice(2).join("/");
    const colon = name.indexOf(":");
    return {
      key: p.key,
      origin: p.origin,
      verb: p.origin === "synthesised" && colon > 0 ? name.slice(0, colon) : null,
      segments: pathSegments(p.key),
      label: where.ok ? where.label : null,
      title: where.ok ? `open ${where.source.file}:${where.source.line} (${where.source.function}) in the editor` : where.reason,
      off: where.ok ? null : where.reason,
    };
  });
}

/**
 * The sources panel's header: the record count, how many are synthesised,
 * and a notice when the records carry no `origin` column (a file below
 * 1.1.0), so none is marked: the app does not guess one from the key. The
 * notice follows the column, not the version: a 1.0.x file that carries the
 * column is read as written and needs none. `null` when there is nothing to
 * list: no model, no /provenance, or no record (the panel hides).
 */
export function sourcesHeaderOf(s: State): { count: number; synthesised: number; notice: string | null } | null {
  if (s.provenance.length === 0) return null;
  const zone = s.artifacts.model?.zones["provenance"];
  const version = zone?.status === "ready" ? zone.version.split(".").slice(0, 2).join(".") : "?";
  const { major, minor } = PROVENANCE_ORIGIN_FROM;
  return {
    count: s.provenance.length,
    synthesised: s.provenance.filter((p) => p.origin === "synthesised").length,
    notice: s.provenanceOrigin ? null : `provenance ${version}: synthesised records are not marked (origin is recorded from ${major}.${minor})`,
  };
}

/** The latest go-to-source answer for `decl`, when it failed. */
export function sourceFailure(s: State, decl: DeclPath): string | null {
  const l = s.source.last;
  return l && l.decl === decl && !l.ok ? (l.reason ?? "go-to-source failed") : null;
}

/** The load failure to show instead of the model, or null. */
export function failureOf(s: State): string | null {
  const a = s.artifacts.model;
  return a && a.status === "failed" ? `${a.path}: ${a.error ?? "failed"}` : null;
}

// ---- model-definition windows (V3c0) ----------------------------------------

/** One value, short: a `$decl` ref as `-> key`, a `$struct` as `Class{...}`, an `$opaque` as its class. */
export function shortValue(v: DeclValue | Param[] | undefined): string {
  if (v === undefined) return "";
  if (v === null) return "null";
  if (Array.isArray(v)) {
    const head = (v as DeclValue[]).slice(0, 6).map(shortValue).join(", ");
    return v.length > 6 ? `[${head}, ... ${v.length}]` : `[${head}]`;
  }
  if (typeof v === "object") {
    if (typeof v["$decl"] === "string") return `-> ${v["$decl"]}`;
    if (typeof v["$struct"] === "string") return `${v["$struct"]}{...}`;
    if (typeof v["$opaque"] === "string") return v["$opaque"];
    return `{${Object.keys(v).join(", ")}}`;
  }
  return String(v);
}

/** A declaration's parameters by field name (its first owner), `field=value` each. */
export function namedParams(d: DeclRecord | null): string {
  const p = d?.params[0];
  if (!p) return "";
  return Object.entries(p.params).map(([k, v]) => `${k}=${shortValue(v)}`).join(" · ");
}

/** The `/opensees/decls` row that owns tag `tag` of allocator kind `kind`, or null. */
export function declOfTag(defs: Definitions, kind: string, tag: number | null): DeclRecord | null {
  if (!defs.decls || tag === null) return null;
  const run = defs.decls.tags.find((t) => t.kind === kind && tag >= t.first && tag < t.first + t.count);
  return run ? (defs.decls.records[run.decl] ?? null) : null;
}

/**
 * The declaration of each recorder group, from `rows/<store>`: "one per
 * group, in index order" (h5-schema.md), the index being the trailing `_<i>`
 * of the group name. No join (an empty map) when the counts differ.
 */
export function recorderDecls(defs: Definitions, recorders: readonly DefObject[], store: string): Map<string, DeclRecord> {
  const out = new Map<string, DeclRecord>();
  const rows = defs.decls?.rows[store];
  if (!defs.decls || !rows || rows.length !== recorders.length) return out;
  const idx = recorders.map((r) => Number(/_(\d+)$/.exec(r.groupName)?.[1] ?? Number.NaN));
  if (idx.some((i) => !Number.isInteger(i))) return out;
  const order = recorders.map((r, i) => ({ r, i: idx[i]! })).sort((a, b) => a.i - b.i);
  order.forEach(({ r }, k) => {
    const d = defs.decls!.records[rows[k]!];
    if (d) out.set(r.path, d);
  });
  return out;
}

const stageSection = (st: StageDef): string => `stage ${st.index} · ${st.name}`;
const countOf = (o: DefObject, table: string): number => o.tables[table]?.rows.length ?? 0;
const CHAIN = ["analysis", "integrator", "algorithm", "test", "system", "numberer", "handler"] as const;

function analysisRow(a: AnalysisDef, section: string): WindowRow {
  const t = a.attrs;
  const chain = CHAIN.filter((k) => t[k] !== undefined).map((k) => {
    const args = t[`${k}_args`];
    return `${k}=${shortValue(t[k])}` + (args === undefined ? "" : `(${(Array.isArray(args) ? args : [args]).map((x) => shortValue(x)).join(", ")})`);
  });
  const title = (["analysis", "integrator", "algorithm"] as const).filter((k) => t[k] !== undefined).map((k) => shortValue(t[k])).join(" · ");
  return { key: a.path, section, title: title || "(no chain tokens)", detail: chain.join(" · "), h5: a.path, declKey: null, select: null };
}

function patternRow(defs: Definitions, p: DefObject, section: string): WindowRow {
  const d = declOfTag(defs, "pattern", p.tag);
  const ref = p.attrs["series_ref"];
  const parts = [
    `${p.type} tag ${p.tag ?? "?"}`,
    typeof ref === "string" ? `series ${ref.slice(ref.lastIndexOf("/") + 1)}` : "",
    countOf(p, "loads") ? `${countOf(p, "loads")} loads` : "",
    countOf(p, "sps") ? `${countOf(p, "sps")} sps` : "",
    countOf(p, "element_loads") ? `${countOf(p, "element_loads")} element loads` : "",
    countOf(p, "sp_holds") ? `${countOf(p, "sp_holds")} holds` : "",
  ];
  return { key: p.path, section, title: d?.name || p.groupName, detail: parts.filter(Boolean).join(" · "), h5: p.path, declKey: d?.key ?? null, select: null };
}

function recorderRow(r: DefObject, section: string, d: DeclRecord | undefined): WindowRow {
  const file = r.attrs["file"], kind = r.attrs["kind"];
  const parts = [r.type, typeof file === "string" ? file : "", typeof kind === "string" ? kind : ""];
  return { key: r.path, section, title: d?.name || r.groupName, detail: parts.filter(Boolean).join(" · "), h5: r.path, declKey: d?.key ?? null, select: null };
}

const MATERIAL_KINDS: ReadonlySet<string> = new Set(["uniaxialMaterial", "nDMaterial", "section"]);

/**
 * What each model-definition window lists, from the store alone (ADR 0112
 * D6): the five V3c0 placeholders show these rows, and the V3 slice that owns
 * a window renders them in full. A row that is a declaration of the store (a
 * material or a section) carries it in `select`, which a click dispatches.
 */
export function windowContentOf(s: State, kind: WindowKind): WindowContent {
  const model = s.artifacts.model;
  if (!model || model.status !== "ready") return { rows: [], note: "no model is open" };
  const defs = s.definitions;
  if (kind === "materials") {
    const rows: WindowRow[] = Object.values(s.decls)
      .filter((d) => MATERIAL_KINDS.has(d.kind))
      .map((d) => {
        const rec = defs ? declOfTag(defs, d.kind, d.tag) : null;
        const named = namedParams(rec);
        const positional = d.params.map((x) => shortValue(x)).join(" ");
        return {
          key: d.h5,
          section: d.kind,
          title: d.name,
          detail: `${d.type} tag ${d.tag ?? "?"}` + (named ? ` · ${named}` : positional ? ` · ${positional}` : ""),
          h5: d.h5,
          declKey: rec?.key ?? null,
          select: d.path,
        };
      });
    return { rows, note: rows.length ? null : "the model declares no material or section" };
  }
  if (!defs) return { rows: [], note: "the model has no /opensees zone" };
  const rows: WindowRow[] = [];
  switch (kind) {
    case "stages": {
      for (const st of defs.stages) {
        const steps = st.attrs["analyze_steps"], dt = st.attrs["analyze_dt"];
        const parts = [
          steps !== undefined ? `${shortValue(steps)} steps` : "",
          dt !== undefined ? `dt ${shortValue(dt)}` : "static",
          `${st.patterns.length} patterns`,
          `${st.recorders.length} recorders`,
          st.bcs.fix ? `${st.bcs.fix.rows.length} fix` : "",
          ...st.other,
        ];
        rows.push({ key: st.path, section: "stages", title: `${st.index} · ${st.name}`, detail: parts.filter(Boolean).join(" · "), h5: st.path, declKey: null, select: null });
      }
      const seq = s.sequence;
      const seqNote =
        seq.status === "ready" ? `/sequence ${seq.version}: ${seq.stages.rows.length} stages, ${seq.changes.rows.length} changes` :
        seq.status === "refused" ? `/sequence ${seq.version} was not read (${seq.reason})` :
        "this file has no /sequence zone; the stages are read from /opensees/stages";
      const none = defs.analysis ? "unstaged model: one implicit stage, from /opensees/analysis" : "the model declares no stage";
      return { rows, note: rows.length ? seqNote : `${none}; ${seqNote}` };
    }
    case "analysis":
      if (defs.analysis) rows.push(analysisRow(defs.analysis, "global"));
      for (const st of defs.stages) if (st.analysis) rows.push(analysisRow(st.analysis, stageSection(st)));
      return { rows, note: rows.length ? null : "the model declares no analysis" };
    case "patterns": {
      for (const p of defs.patterns) rows.push(patternRow(defs, p, "global"));
      for (const st of defs.stages) for (const p of st.patterns) rows.push(patternRow(defs, p, stageSection(st)));
      for (const t of defs.timeSeries) {
        const d = declOfTag(defs, "timeSeries", t.tag);
        const named = namedParams(d);
        rows.push({ key: t.path, section: "time series", title: d?.name || t.groupName, detail: `${t.type} tag ${t.tag ?? "?"}` + (named ? ` · ${named}` : ""), h5: t.path, declKey: d?.key ?? null, select: null });
      }
      return { rows, note: rows.length ? null : "the model declares no pattern or time series" };
    }
    case "recorders": {
      const global = recorderDecls(defs, defs.recorders, "recorders");
      for (const r of defs.recorders) rows.push(recorderRow(r, "global", global.get(r.path)));
      for (const st of defs.stages) {
        const staged = recorderDecls(defs, st.recorders, `stages/stage_${String(st.index).padStart(3, "0")}/recorders`);
        for (const r of st.recorders) rows.push(recorderRow(r, stageSection(st), staged.get(r.path)));
      }
      return { rows, note: rows.length ? null : "the model declares no recorder" };
    }
    default: {
      const unknown: never = kind;
      throw new Error(`windowContentOf: unknown window ${JSON.stringify(unknown)}`);
    }
  }
}

// -- V4e1: section planes (ADR 0083 Part 1) --------------------------------------

export { activeCount, gizmoPlanes, MAX_ACTIVE_PLANES, offsetRange, orientationOf } from "./clip.ts";
export type { Box, Vec3 } from "./clip.ts";
