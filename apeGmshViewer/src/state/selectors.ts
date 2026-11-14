// Selectors: pure reads of the state that panels and the viewport render
// from. A panel imports this module and store.ts, nothing else of the app.

import type { Chain, ChainNode, Field } from "../chain/resolve.ts";
import { judgeDraft } from "../combination/author.ts";
import { CombinationRefused, CombinationTable, isSigned } from "../combination/evaluate.ts";
import { REASONS, type Refusal as RuleRefusal } from "../reader/userRules.ts";
import type { OpsFamily, Param } from "../model/types.ts";
import { elementFields, elementLinks, type Lookup } from "./decls.ts";
import { pairSessions } from "../reader/geometry.ts";
import { PROVENANCE_ORIGIN_FROM } from "../reader/provenance.ts";
import { roleColouring } from "./roles.ts";
import { activationNotice, activationOf, type Activation } from "./activation.ts";
import { pairingNotice, pairResults, particleNotices, type ResultsPairing } from "./results.ts";
import {
  climOf,
  componentName,
  componentOf,
  contourKeysOf,
  elementComponentOf,
  elementContourFields,
  gaussFields,
  elementTopology,
  fieldsOf,
  isNodeTopology,
  nodalFields,
  topologyLabel,
  type Clim,
  type ElementComponent,
  type ElementField,
  type ElementTopology,
  type FieldComponent,
  type NodalField,
} from "./contour.ts";
import { meanFallbackGroups, usesMatrix } from "./extrapolation.ts";
import { planeStatementOf, planeWarning, tensorFields, type PlaneStatement, type TensorField } from "./tensor.ts";
import type { BlobRef, ContourCmap, ContourState, StaleMark, UserRefusalIn, UserResult } from "./types.ts";
import { displacementAxes } from "./deform.ts";
import {
  beamLocalAxes,
  buildBeamChains,
  buildLineDiagram3D,
  buildSelectedBeamChainsFromState,
  colorForValue,
  declTransformVecxz,
  defaultFlipSignFor,
  defaultLineDiagramComponent,
  defaultVecxz,
  effectiveFlipSign,
  elementTransformVecxzFromState,
  interpolateStationValue,
  joinLadrunoElements,
  LINE_DIAGRAM_CMAP_NAMES,
  LINE_DIAGRAM_COMPONENT_META,
  LINE_DIAGRAM_FIELDS,
  LINE_DIAGRAM_PREFERRED_COMPONENTS,
  lineDiagramClimOf,
  lineDiagramFieldsOf,
  lineDiagramKeys,
  lineDiagramKeysOf,
  localAxisForComponent,
  projectHoverNeedle,
  requires3dFrame,
  screenExtrusionDirection,
  SIGN_NEG_RGB,
  SIGN_POS_RGB,
  SIGN_ZERO_RGB,
  signRampCss,
  type BeamChain,
  type BeamChainInputSpan,
  type BeamChainJoint,
  type BeamChainSpan,
  type BeamChainStationPoint,
  type BeamDiagramSpan3D,
  type BuildLineDiagram3DOptions,
  type DiagramLabelAnchor,
  type HoverNeedleResult,
  type LadrunoElementJoin,
  type LineDiagram3DGeometry,
  type LineDiagramComponentMeta,
  type LineDiagramField,
  type LineDiagramFieldName,
  type ZRecSource,
} from "./lineDiagram.ts";

export type { Clim, ElementComponent, ElementField, FieldComponent, NodalField } from "./contour.ts";
export { GHOST_OPACITY, scaleLabel } from "./deform.ts";
export { CMAP_NAMES, CMAPS, fmtValue, MAGNITUDE, NO_DATA_RGB, rampCss, RANGE_MODES, sampleCmap } from "./contour.ts";
// VF2-d (#2956): the one range rule the legend's scale and the command palette both offer.
export { rangeKindOf, rangeOptionsOf, type RangeKind, type RangeOption } from "./contour.ts";
export { NODE_SIZE_MAX, NODE_SIZE_MIN } from "./nodes.ts";
export { gaussNotesOf, gaussUnavailable } from "./gaussNotes.ts";
export { legendEntryColour } from "./palette.ts";
export { classifyFieldDimension, resolveFieldUnits, UNIT_DISPLAY_OPTIONS, UNIT_SYSTEM_IDS, UNIT_SYSTEMS } from "./units.ts";
export {
  beamLocalAxes,
  buildBeamChains,
  buildLineDiagram3D,
  buildSelectedBeamChainsFromState,
  colorForValue,
  declTransformVecxz,
  defaultFlipSignFor,
  defaultLineDiagramComponent,
  defaultVecxz,
  effectiveFlipSign,
  elementTransformVecxzFromState,
  interpolateStationValue,
  joinLadrunoElements,
  LINE_DIAGRAM_CMAP_NAMES,
  LINE_DIAGRAM_COMPONENT_META,
  LINE_DIAGRAM_FIELDS,
  LINE_DIAGRAM_PREFERRED_COMPONENTS,
  lineDiagramClimOf,
  lineDiagramFieldsOf,
  lineDiagramKeys,
  lineDiagramKeysOf,
  localAxisForComponent,
  projectHoverNeedle,
  requires3dFrame,
  screenExtrusionDirection,
  SIGN_NEG_RGB,
  SIGN_POS_RGB,
  SIGN_ZERO_RGB,
  signRampCss,
  type BeamChain,
  type BeamChainInputSpan,
  type BeamChainJoint,
  type BeamChainSpan,
  type BeamChainStationPoint,
  type BeamDiagramSpan3D,
  type BuildLineDiagram3DOptions,
  type DiagramLabelAnchor,
  type HoverNeedleResult,
  type LadrunoElementJoin,
  type LineDiagram3DGeometry,
  type LineDiagramComponentMeta,
  type LineDiagramField,
  type LineDiagramFieldName,
  type ZRecSource,
};
export {
  buildBeamChainPlotModel,
  computeLineDiagramHover,
  formatHoverReadout,
  pickChainPlotAtS,
  summarizeDrawnBeams,
  syncLineDiagramDerived,
  type BeamChainExtremaPlot,
  type BeamChainHoverPlot,
  type BeamChainJointPlot,
  type BeamChainPlotModel,
  type BeamChainSpanPlotModel,
  type BeamChainStationDotPlot,
  type BeamChainUnknownGapPlot,
  type BuildBeamChainPlotOptions,
} from "./lineDiagramChainSvg.ts";

export type { Activation } from "./activation.ts";
export type { ResultsPairing } from "./results.ts";
export {
  buildLoads3D,
  evaluatePatternSeriesFactor,
  loadsColourPalette,
  loadsSummaryOf,
  loadsUnitsOf,
  reactionComponentsOf,
  reactionMomentComponentsOf,
  type BuildLoads3DOptions,
  type LoadPatternCounts,
  type LoadPatternEntry,
  type Loads3DGeometry,
  type LoadsPeaks,
  type LoadsSummary,
  type LoadsUnitLabels,
} from "./loads.ts";
export { springCountOf } from "./springs.ts";
// VU9a (ADR 0132): what the sets panel reads of the workspace slice.
export { pickSummaryOf } from "./workspace.ts";
export { cannotSave as workspaceCannotSave } from "./reduce/workspace.ts";
export { vectorComponentOf, vectorFieldOf, vectorFieldsOf, type VectorField } from "./vectors.ts";
import type {
  AnalysisDef,
  ArtifactInfo,
  CombinationDraft,
  CombineState,
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

import {
  clauseLabel,
  clauseToSearchClause,
  elementsOfDecl,
  hiddenElementsOf,
  indexOf,
  parseSearchText,
  runSearchTerms,
  searchResultsOf,
  selectedElementsOf,
  type ModelIndex,
  type SearchSummary,
  type SearchTerm,
} from "./search.ts";
import {
  initialSearch,
  initialSpotlight,
  SPOTLIGHT_EVENT_LEDGER,
  spotlightViewOf as spotlightViewWith,
  type EventLedgerEntry,
  type SpotlightItem,
  type SpotlightView,
} from "./spotlight.ts";

/**
 * What Spotlight (`Ctrl+K`) renders for `s` (state/spotlight.ts), handed the
 * range a pick of "fixed" starts from (`fixedRangeFrom`): the palette cannot
 * read the contour view itself, which this module holds (VF2-d, #2956).
 */
export function spotlightViewOf(s: State): SpotlightView {
  return spotlightViewWith(s, fixedRangeFrom);
}

export {
  clauseLabel,
  clauseToSearchClause,
  elementsOfDecl,
  hiddenElementsOf,
  indexOf,
  initialSearch,
  initialSpotlight,
  parseSearchText,
  runSearchTerms,
  searchResultsOf,
  selectedElementsOf,
  SPOTLIGHT_EVENT_LEDGER,
  type EventLedgerEntry,
  type ModelIndex,
  type SearchSummary,
  type SearchTerm,
  type SpotlightItem,
  type SpotlightView,
};

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

/** Maximum number of individual unpinned definition chains rendered before switching to a batch summary card. */
export const MAX_INSPECTED_CHAINS = 4;

/** What the inspector shows: the pinned declarations, then the selection (up to MAX_INSPECTED_CHAINS unpinned chains). */
export function inspected(s: State): { decl: DeclPath; pinned: boolean; chain: Chain }[] {
  const out: { decl: DeclPath; pinned: boolean; chain: Chain }[] = [];
  for (const p of s.inspector.pinned) {
    const chain = chainOf(s, p, pickOf(s, p)?.nodes ?? null);
    if (chain) out.push({ decl: p, pinned: true, chain });
  }
  const unpinned = s.selection.decls.filter(
    (p) => !s.inspector.pinned.includes(p) && s.decls[p] !== undefined && s.decls[p]!.kind !== "physical_group" && s.decls[p]!.kind !== "label",
  );
  if (unpinned.length <= MAX_INSPECTED_CHAINS) {
    for (const p of unpinned) {
      const chain = chainOf(s, p, pickOf(s, p)?.nodes ?? null);
      if (chain) out.push({ decl: p, pinned: false, chain });
    }
  }
  return out;
}

/** Summary card data for any pinned or selected physical_group or label declaration. */
export interface InspectedGroup {
  readonly decl: DeclPath;
  readonly pinned: boolean;
  readonly kind: "physical_group" | "label";
  readonly name: string;
  readonly h5: string;
  readonly dim: number;
  readonly tag: number;
  readonly count: number;
  readonly hidden: boolean;
}

export function inspectedGroupsOf(s: State): InspectedGroup[] {
  const out: InspectedGroup[] = [];
  const seen = new Set<DeclPath>();
  const add = (p: DeclPath, pinned: boolean) => {
    if (seen.has(p)) return;
    const d = s.decls[p];
    if (!d || (d.kind !== "physical_group" && d.kind !== "label") || !d.group) return;
    seen.add(p);
    out.push({
      decl: p,
      pinned,
      kind: d.kind,
      name: d.name,
      h5: d.h5,
      dim: d.group.dim,
      tag: d.group.tag,
      count: d.group.count,
      hidden: s.visibility.hidden.includes(p),
    });
  };
  for (const p of s.inspector.pinned) add(p, true);
  for (const p of s.selection.decls) add(p, false);
  return out;
}

/** Summary card data when a batch selection holds more than MAX_INSPECTED_CHAINS declarations. */
export interface InspectedBatch {
  readonly count: number;
  readonly elementCount: number;
  readonly kinds: readonly string[];
  readonly types: readonly string[];
  readonly groups: readonly string[];
  readonly decls: readonly DeclPath[];
}

export function inspectedBatchOf(s: State): InspectedBatch | null {
  const unpinned = s.selection.decls.filter(
    (p) => !s.inspector.pinned.includes(p) && s.decls[p] !== undefined && s.decls[p]!.kind !== "physical_group" && s.decls[p]!.kind !== "label",
  );
  if (unpinned.length <= MAX_INSPECTED_CHAINS) return null;
  const kinds = new Set<string>();
  const types = new Set<string>();
  const groups = new Set<string>();
  const elems = new Set<DeclPath>();
  for (const p of unpinned) {
    const d = s.decls[p];
    if (!d) continue;
    kinds.add(d.kind);
    types.add(d.type);
    for (const g of d.element?.groups ?? []) groups.add(s.decls[g]?.name ?? g);
    for (const e of elementsOfDecl(s, p)) elems.add(e);
  }
  return {
    count: unpinned.length,
    elementCount: elems.size,
    kinds: [...kinds],
    types: [...types].slice(0, 8),
    groups: [...groups].slice(0, 8),
    decls: unpinned,
  };
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
/** VU9a (ADR 0132 D6, OQ5): the banner line naming the file an unreadable workspace was renamed aside to. */
export function workspaceAsideNotice(aside: string): string {
  const name = aside.split(/[\\/]/).pop() ?? aside;
  const file = name.replace(/\.unreadable-[0-9TZ-]+$/, "");
  return `${file} was not a readable workspace: it was moved aside to ${name}, unchanged, before the save wrote a new ${file}. Nothing was overwritten.`;
}

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
  if (ev.state === "refused") out.push(ev.code === "combination" || ev.code === "difference" || ev.code === "user" ? `Contour not drawn: ${ev.reason}.` : `Element contour refused: ${ev.reason}.`);
  // VF (#2940): a gauss_node pick drawn cell_averaged says so.
  if (ev.state === "element" && ev.fallback) out.push(fallbackNotice(ev));
  // VU10: a newer manifest's banner (ADR 0132 D2), and the nodes a difference cannot give a value.
  if (s.variants.study?.newer) out.push(s.variants.study.newer);
  // VU9a (ADR 0132 D2, D6): a newer workspace is shown read-only, and an unreadable one moved aside is named, once per rename.
  const ws = s.workspace;
  if (ws.read?.status === "ready" && ws.read.readOnly) out.push(ws.read.readOnly);
  if (ws.aside !== null) out.push(workspaceAsideNotice(ws.aside));
  if (ev.state === "difference" && ev.missing > 0) {
    out.push(`Difference: ${ev.missing.toLocaleString()} of ${ev.nodes.toLocaleString()} nodes have no value in one of the two variants; they are drawn as no data.`);
  }
  if (s.contour.notice) out.push(s.contour.notice);
  // V4d: drawn nodes the deformed shape leaves undeformed, or why it is not drawn.
  const d = deformNotice(s);
  if (d) out.push(d);
  // ADR 0134 D5: the particles drawn, how they are coloured, and what of them is not drawn.
  out.push(...particleNotices(s));
  return out;
}

/** VF (#2940): the banner's line for the groups of a `gauss_node` contour drawn `cell_averaged` (their element means). */
export function fallbackNotice(v: Extract<ContourView, { state: "element" }>): string {
  const f = v.fallback;
  if (!f) throw new Error(`fallbackNotice: element field ${v.field} is drawn as asked (${v.topology})`);
  const extrapolated = v.groups.filter((g) => g.nGauss > 1 && !f.groups.includes(g.id)).map((g) => g.id);
  const rest = extrapolated.length ? ` Group(s) ${extrapolated.join(", ")} are extrapolated to the nodes (gauss_node) as asked.` : "";
  return (
    `Element contour ${v.field}: group(s) ${f.groups.join(", ")} cannot be extrapolated to the nodes (gauss_node), so each of their elements ` +
    `gives its mean to its corners (cell_averaged), averaged at the nodes.${rest} Why: ${f.reason}.`
  );
}

/** VA6d / VF2: a derived contour's stored components, its out-of-plane statement by element, and the line it warns with (null: none). */
export interface DerivedPlane {
  from: string[];
  /** shared and frozen: tensor.ts `planeStatementOf` computes it once per model, set and family, not per step */
  plane: Readonly<PlaneStatement>;
  warning: string | null;
}

/**
 * VF2 (#2924 item 6): what the derived contour on screen (`von_mises_*`) says
 * about its out-of-plane normal, built once for the manifest's
 * `contour.derived` and the banner's warning line (tensor.ts `planeWarning`
 * words it). Null when the contour is not a drawn derived element contour.
 * The groups' element ids are blobs, so the caller hands the store's lookup
 * (a panel never holds the BlobStore: renderer/app.ts closes over it). A
 * drawn element view has its element ids on hand (its join read them), so one
 * missing here throws rather than counting fewer elements.
 */
export function derivedPlaneOf(s: State, blobs: { get(ref: BlobRef): unknown }): DerivedPlane | null {
  const v = contourView(s);
  if (v.state !== "element" || !v.component.derived) return null;
  const t = v.component.derived.tensor;
  const groups = t.groups.map((group) => {
    const g = v.groups.find((x) => x.id === group.id);
    const eids = g ? blobs.get(g.eids) : undefined;
    if (!(eids instanceof Float64Array)) throw new Error(`derivedPlaneOf: the element ids of group ${group.id} are not on hand`);
    return { group, eids };
  });
  const plane = planeStatementOf(t.family, groups, s.decls);
  return { from: [...t.components], plane, warning: planeWarning(t.family, plane) };
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
  | { state: "refused"; code: ElementRefusal | "combination" | "difference" | "user"; reason: string }
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
      /**
       * VF (#2940): `topology` is `gauss_node`, and `groups` have no natural
       * coordinates, so each of their elements gives its mean to its corners
       * (`cell_averaged`) while every other group is extrapolated; `reason` is
       * the verdict that refused them. Null when every group is drawn as asked.
       * The banner and the manifest state it.
       */
      fallback: { asked: "gauss_node"; drawn: "cell_averaged"; groups: string[]; reason: string } | null;
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
    }
  /**
   * VU7b-2 (ADR 0130 D4): a User Combination's value for one nodal quantity,
   * one state with no step. The colours come from the read under `key`
   * (renderer/contour.ts `combinationColours`).
   */
  | {
      state: "combination";
      /** the combination */
      name: string;
      field: string;
      component: FieldComponent;
      /** `combinationRequestOf(s).key`: the read whose values are drawn */
      key: string;
      clim: Clim;
      cmap: ContourCmap;
      /** whether it may draw as a deformed shape (an unsigned one is a contour only) */
      signed: boolean;
    }
  /**
   * VU16 (ADR 0130 D7): one component of a state-free `ON_NODES` user result of
   * the set's `.user.h5`, by node id; it has no state, so it is shown at any
   * phase. The colours come from the read under `key` and `read`
   * (renderer/contour.ts `userResultColours`); a node it gives no finite value
   * is no data. Never a deformed shape.
   */
  | {
      state: "user";
      /** the user result (`field` too: the legend names it as a field); a compare one's is `compareUserName`'s display string */
      name: string;
      field: string;
      /**
       * VF2-d (#2956): a compare variant's user result (VU19) names its variant and
       * its entry here, apart, since the display string `name` can equal a window
       * entry of that literal name; null for the window set's own user result
       */
      compare: { variant: string; entry: string } | null;
      component: FieldComponent;
      /** `userResultRequestOf(s).key`, and the feed's read under it */
      key: string;
      read: number;
      basis: UserResult["basis"];
      clim: Clim;
      cmap: ContourCmap;
      /** the node ids the column lists, and how many hold a finite value */
      nodes: number;
      finite: number;
    }
  /**
   * VU10 (ADR 0132 D9.5): the window variant minus the compare variant for one
   * nodal component, by node id, at the stage and step shown. The colours come
   * from the read under `key` (renderer/contour.ts `differenceColours`); a node
   * either side lacks is no data. Never a deformed shape.
   */
  | {
      state: "difference";
      /** the window variant's label (or the model's file name when the window set is no row) and the compare variant's */
      window: string;
      compare: string;
      field: string;
      component: FieldComponent;
      stageName: string;
      /** `differencePlanOf(s).key`: the read whose values are drawn */
      key: string;
      clim: Clim;
      cmap: ContourCmap;
      /** the window's nodes, and how many of them have no value on one side */
      nodes: number;
      missing: number;
    };

/** A results file's element-id verdict; a results info without one is unverified (fail closed). */
export function elementVerdictOf(info: ResultsInfo): { verified: true } | { verified: false; reason: string } {
  return info.elementIndex ?? { verified: false, reason: "this results info carries no element-index verdict" };
}

/** The stress and strain fields of a verified file's results stage on screen; a field with no group that makes a tensor is kept, for its refusals. */
export function principalFieldsOf(s: State): TensorField[] {
  const at = s.phase.at;
  if (at?.kind !== "results" || !s.results || !s.mesh || !elementVerdictOf(s.results).verified) return [];
  const st = s.results.stages[at.stage];
  return st ? tensorFields(st.gauss) : [];
}

/** The field the Principal card draws: the one picked, else the first stress that can draw, then the first field that can; null when the stage has none. */
export function principalFieldOf(s: State): TensorField | null {
  const fields = principalFieldsOf(s);
  const drawable = fields.filter((f) => f.groups.length > 0);
  return fields.find((f) => f.name === s.principal.field) ?? drawable.find((f) => f.family === "stress") ?? drawable[0] ?? fields[0] ?? null;
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
  // VF (#2940, the owner's rulings on #2924 item 2): the groups of a gauss_node field refused only because they have no
  // natural coordinates (a BezierTet10's barycentric points) draw their element means (cell_averaged), per group; every
  // other group is still extrapolated, and the banner and the manifest name the groups that were not.
  const meanGroups = topology === "gauss_node" ? meanFallbackGroups(groups, j.extrapolation) : null;
  const fallback = meanGroups ? { asked: "gauss_node" as const, drawn: "cell_averaged" as const, groups: meanGroups, reason: j.extrapolation.matrix! } : null;
  if (isNodeTopology(topology) && !fallback) {
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
    topologyLabel: topologyLabel(topology, hit.field.nGauss) + (fallback ? `; group(s) ${fallback.groups.join(", ")}: each element's mean (cell_averaged), no natural coordinates` : ""),
    fallback,
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
  if (s.variants.difference) return differenceContour(s);
  // VU19: a compare user result has no state either, so it is shown whatever phase is on screen.
  if (s.variants.shown !== null) return compareUserContour(s, s.variants.shown);
  if (s.phase.at?.kind === "combination") return combinationContour(s, s.phase.at.name);
  // VU16: a user result has no state, so it is shown whatever phase is on screen, with or without results.
  if (s.contour.source === "user") return userResultContour(s);
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
 * The range a pick of "fixed" starts from, on the legend's scale and in the
 * command palette alike (VF2-d, #2956): the stored fixed range, else the range
 * the contour shows when it is a finite, non-empty one, else null (there is
 * nothing to start from, and the pick changes nothing).
 */
export function fixedRangeFrom(s: State): { min: number; max: number } | null {
  if (s.contour.fixed) return s.contour.fixed;
  const c = contourView(s);
  if (c.state !== "ready" && c.state !== "element" && c.state !== "combination" && c.state !== "difference" && c.state !== "user") return null;
  const { min, max } = c.clim;
  return min !== null && max !== null && min < max ? { min, max } : null;
}

/**
 * The field picker's rows: the nodal fields of the results stage shown, and
 * the refused components; V4c2a: its element fields (a refused one says why;
 * VA6d: with the derived scalars of each field's tensor, `elementContourFields`),
 * and whether the file's element rows are verified.
 */
export function contourFieldsOf(s: State): {
  fields: NodalField[];
  refused: { component: string; reason: string }[];
  elements: ElementField[];
  elementsVerified: boolean;
} | null {
  const at = s.phase.at;
  // VU7b-2: a User Combination offers the nodal quantities of the cases it reaches, each component alone.
  if (at?.kind === "combination") {
    const f = combinationFieldsOf(s, at.name);
    return f.ok ? { fields: f.fields, refused: [], elements: [], elementsVerified: false } : null;
  }
  const st = at?.kind === "results" ? s.results?.stages[at.stage] : undefined;
  return st ? { fields: fieldsOf(st), refused: st.refusedNodal, elements: elementContourFields(st.gauss), elementsVerified: elementVerdictOf(s.results!).verified } : null;
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
  // VU10: a difference of two variants is a contour only; it is never offered as a shape, nor is either variant's under it.
  if (s.variants.difference) return { ok: false, state: "refused", reason: "the difference of two variants is shown as a contour only, never as a deformed shape" };
  // VU19: so is a compare user result, and nothing of the window set's is offered as a shape under it.
  if (s.visibility.colourBy === "contour" && s.variants.shown !== null) {
    return { ok: false, state: "refused", reason: "a compare variant's user result is shown as a contour only, never as a deformed shape" };
  }
  if (s.phase.at?.kind === "combination") return combinationDeform(s, s.phase.at.name);
  // VU16: a user result is a contour only; it is never offered as a shape, nor is a results step's under it.
  if (s.visibility.colourBy === "contour" && s.contour.source === "user") {
    return { ok: false, state: "refused", reason: "a user result is shown as a contour only, never as a deformed shape" };
  }
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
  const program = zone ? (s.definitions?.program ?? null) : null;
  return pairResults(m?.status === "ready" ? { lineage: s.lineage, openseesVersion: zone ? ops.version : null, programDecl, program } : null, s.results);
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
  // W4d #2843: matched by the program name; `stageName` is the label shown (a display name may differ).
  const key = [s.sequence, s.stageElements, s.definitions, st.programName, st.name, byFem];
  if (activationMemo && activationMemo.key.every((k, i) => k === key[i])) return activationMemo.value;
  const activation: Activation = byFem
    ? {
        state: "unbound",
        reason: "the results pair with this model by fem_hash only (the mesh matches; they carry no model_hash), so they are not tied to this model's /sequence program",
      }
    : activationOf(s.sequence, s.stageElements, s.definitions, st.programName);
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
  if (kind === "search") {
    const res = searchResultsOf(s);
    const rows: WindowRow[] = res.matched.slice(0, 200).map((d) => ({
      key: d.path,
      section: d.kind,
      title: d.name,
      detail: `${d.type}` + (d.tag !== null ? ` tag ${d.tag}` : d.element?.femId !== null && d.element?.femId !== undefined ? ` #${d.element.femId}` : ""),
      h5: d.h5,
      declKey: d.path,
      select: d.path,
    }));
    return {
      rows,
      note: res.total === 0 ? "no declarations match the current search" : `${res.total} matching declarations (${res.elementCount} elements)`,
    };
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

/** V4e3: why a mesh gets no cut-face cap: it has no volumetric cell. */
export const NO_SOLID_CELLS = "No cap: this mesh has no volumetric cells (shells, beams or 2-D), so a cut has no inside to fill.";

/** V4e3: what the planes panel says under "Contour on cut face": why no cap can be drawn, or null when one can (or no mesh is loaded). */
export function cutFaceNote(s: State): string | null {
  return s.mesh && s.mesh.counts.solidCells === 0 ? NO_SOLID_CELLS : null;
}

// -- VU7b-1: a User Combination over the /spectra cases (ADR 0130 D4, D5) --------

/**
 * The `/spectra` cases combination `name` reads, directly or through the
 * combinations it names, in the order first met: the cases to fetch
 * (`user:readCase`) before the evaluator runs, since its `Source` is
 * synchronous. A name the table does not hold, or refused, throws the
 * evaluator's `CombinationRefused` (`unresolved`, `refused`).
 */
export function spectrumCasesOf(table: CombinationTable, name: string): string[] {
  const cases: string[] = [];
  const seen = new Set<string>();
  const stack = [name];
  while (stack.length > 0) {
    const current = stack.pop()!;
    if (seen.has(current)) continue;
    seen.add(current);
    for (const t of table.row(current, name).terms) {
      if (t.kind === "combination") stack.push(t.source);
      else if ((t.kind === "spectrum" || t.kind === "reference") && !cases.includes(t.source)) cases.push(t.source);
    }
  }
  return cases;
}

// -- VU7b-2: the User Combination shown (ADR 0130 D1, D4, A3) ---------------------

const quoted = (name: string): string => JSON.stringify(name);

/** A refused row as the evaluator's table takes it; a reason or origin the reader never gives is a bug, and raises. */
function ruleRefusal(r: UserRefusalIn): RuleRefusal {
  if (!(REASONS as readonly string[]).includes(r.reason)) throw new RangeError(`combinationTableOf: refusal reason ${quoted(r.reason)} of ${quoted(r.name)} is not one of ${REASONS.join(", ")}`);
  if (r.origin !== "user" && r.origin !== "model") throw new RangeError(`combinationTableOf: refusal origin ${quoted(r.origin)} of ${quoted(r.name)} is not user or model`);
  return { name: r.name, reason: r.reason as RuleRefusal["reason"], detail: r.detail, origin: r.origin, row: r.row };
}

let tableMemo: { of: object; table: CombinationTable } | null = null;

/** The user file's combination rows as VU4's evaluator takes them, or null when it holds none (or `/combinations` is refused). */
export function combinationTableOf(s: State): CombinationTable | null {
  const t = s.user?.status === "ready" ? s.user.info.table : null;
  if (!t) return null;
  if (tableMemo?.of !== t) tableMemo = { of: t, table: new CombinationTable({ rows: t.rows, modelRows: t.modelRows, refused: t.refused.map(ruleRefusal) }) };
  return tableMemo.table;
}

/**
 * What a combination can colour by: the `ON_NODES` columns of the `/spectra`
 * cases it reaches (`DISPLACEMENT/Ux`, grouped as a `.ladruno` stage's are),
 * each component alone. A combination is evaluated per component (D4), so no
 * magnitude is offered: the CQC of a magnitude is not the magnitude of the
 * CQCs. When there is nothing, the reason says why; a combination of stage or
 * mode terms only is read from the results file, which this app does not
 * combine.
 */
export function combinationFieldsOf(s: State, name: string, over: CombinationTable | null = null): { ok: true; fields: NodalField[]; cases: string[] } | { ok: false; reason: string } {
  const u = s.user;
  if (u?.status !== "ready") return { ok: false, reason: "the user file is not read" };
  const file = baseName(u.path);
  // VU9b: a draft is judged over the table it would make (`over`), not the file's.
  const table = over ?? combinationTableOf(s);
  if (!table) return { ok: false, reason: `${file} holds no /combinations row` };
  let cases: string[];
  try {
    cases = spectrumCasesOf(table, name);
  } catch (err) {
    if (err instanceof CombinationRefused) return { ok: false, reason: err.message };
    throw err;
  }
  if (!cases.length) {
    return {
      ok: false,
      reason:
        `User Combination ${quoted(name)} reaches no /spectra case: its stage and mode terms are read from the results file, ` +
        "and this app combines /spectra cases only (Python's Results also combines stages and modes)",
    };
  }
  const sp = u.info.spectra;
  if (sp === null) return { ok: false, reason: `${file} holds no /spectra, so no case ${cases.map(quoted).join(", ")}` };
  if (!sp.ok) return { ok: false, reason: `/spectra of ${file} is refused: ${sp.reason}` };
  const names: string[] = [];
  for (const c of cases) {
    const found = sp.items.find((x) => x.name === c);
    for (const q of found?.nodal ?? []) for (const comp of q.components) if (!names.includes(`${q.label}/${comp}`)) names.push(`${q.label}/${comp}`);
  }
  if (!names.length) return { ok: false, reason: `the /spectra cases ${quoted(name)} reaches (${cases.map(quoted).join(", ")}) hold no nodal result` };
  const fields = nodalFields(names)
    .map((f) => ({
      name: f.name,
      components: f.components.filter((c) => !c.magnitude),
      magnitudeRefused: f.components.some((c) => c.magnitude)
        ? "a combination is evaluated per component (ADR 0130 D4): the combination of a magnitude is not the magnitude of the combinations"
        : null,
    }))
    .filter((f) => f.components.length > 0);
  return { ok: true, fields, cases };
}

/**
 * What the app reads for the combination shown, or null when nothing is to be
 * read (not a combination, not colouring by contour, nothing picked that the
 * combination offers). `quantity` is the picked column as the case spells it;
 * a result's label never holds a `/` (reader/userSpectra.ts refuses one), so
 * the component key splits at its first. `key` names the read: the user file,
 * the combination, the column, and whether `/spectra` is stale (A6).
 */
export function combinationRequestOf(s: State): { key: string; path: string; name: string; quantity: { label: string; component: string }; staleness: StaleMark } | null {
  const at = s.phase.at;
  const u = s.user;
  const c = s.contour;
  if (at?.kind !== "combination" || s.visibility.colourBy !== "contour" || u?.status !== "ready" || c.source !== "nodal" || c.field === null || c.component === null) return null;
  const f = combinationFieldsOf(s, at.name);
  if (!f.ok || !f.fields.some((x) => x.name === c.field && x.components.some((y) => y.key === c.component))) return null;
  const sp = u.info.spectra;
  if (!sp?.ok) return null;
  const slash = c.component.indexOf("/");
  const quantity = { label: c.component.slice(0, slash), component: c.component.slice(slash + 1) };
  return { key: JSON.stringify([u.path, at.name, c.component, sp.stale.stale]), path: u.path, name: at.name, quantity, staleness: sp.stale };
}

/**
 * The User Combination shown, as the phase bar states it: its rule, whether it
 * is stale (A6), and its sign from the descriptors (A3; null with why, when the
 * table cannot say). Null off a combination.
 */
export function combinationShownOf(s: State): { name: string; rule: string; stale: boolean; signed: boolean | null; why: string | null } | null {
  const at = s.phase.at;
  if (at?.kind !== "combination" || s.user?.status !== "ready") return null;
  const c = s.user.info.combinations;
  const item = c?.ok ? c.items.find((x) => x.name === at.name) : undefined;
  if (!item) return null;
  const sign = combinationSign(s, at.name);
  return { name: at.name, rule: item.rule, stale: item.stale, signed: sign.signed, why: combinationDeform(s, at.name).reason };
}

// -- VU9b: authoring a User Combination (ADR 0130 A3, A4; HG9 on #2693) ----------------------

/** The key of a draft's check: the user file, the read's token and the draft. Null with no file read. */
export function draftKeyOf(s: State, draft: CombinationDraft): string | null {
  const u = s.user;
  if (u?.status !== "ready") return null;
  return JSON.stringify([u.path, u.token ?? null, draft]);
}

/** VU9b: what the authoring panel offers (reader/userRules.ts: the rules, the envelopes and the term kinds A3 names). */
export { RULES as COMBINATION_RULES, ENVELOPES as COMBINATION_ENVELOPES, KINDS as TERM_KINDS } from "../reader/userRules.ts";

/** VU9b: an empty draft, a linear combination of one stage term at the last step. */
export const EMPTY_DRAFT: CombinationDraft = { name: "", rule: "linear", envelope: "", fraction: null, damping: null, terms: [{ kind: "stage", source: "", step: -1, mode: 0, factor: 1 }] };

/**
 * VU9b: the sources a term of each kind can name in this set, for the panel's
 * suggestions: the results stages (their stamped names, else `stage_<i>`), the
 * mode hosts, the `/spectra` cases and the combinations in force. A source the
 * set does not have can still be typed; the readers and the evaluator judge it.
 */
export function draftSourcesOf(s: State): Record<string, string[]> {
  const stages = (s.results?.stages ?? []).map((st, i) => ({ name: st.programName || `stage_${i}`, mode: st.kind === "mode" }));
  const u = s.user?.status === "ready" ? s.user.info : null;
  const cases = u?.spectra?.ok ? u.spectra.items.map((c) => c.name) : [];
  const combos = u?.table ? [...u.table.rows, ...u.table.modelRows].map((r) => r.name) : [];
  return { stage: stages.filter((x) => !x.mode).map((x) => x.name), mode: stages.filter((x) => x.mode).map((x) => x.name), spectrum: cases, reference: cases, combination: combos };
}

/** VU9b: the panel's entry in the phase bar, or null when there is no user file to add to (the app never creates one). */
export function combineEntryOf(s: State): { title: string } | null {
  const u = s.user;
  if (u?.status !== "ready") return null;
  const c = u.info.combinations;
  if (c !== null && !c.ok) return null;
  return { title: `Write a User Combination and save it into ${baseName(u.path)} (ADR 0130 A4: a copy of the file, then a replace)` };
}

/**
 * The draft judged by the readers' rules beside the file's rows (combination/
 * author.ts `judgeDraft`), and what VU4's evaluator reads for it: the `/spectra`
 * cases it reaches and the first nodal quantity they hold (null when none).
 * Null with no draft to judge. The rows on screen are the ones that stand; main
 * judges again against every stored row before it writes.
 */
export function draftJudgedOf(
  s: State,
  draft: CombinationDraft,
): { ok: true; key: string; path: string; table: CombinationTable; name: string; cases: string[]; quantity: { label: string; component: string } | null; staleness: StaleMark } | { ok: false; key: string; reason: string; detail: string } | null {
  const key = draftKeyOf(s, draft);
  const u = s.user;
  if (key === null || u?.status !== "ready") return null;
  const t = u.info.table;
  const c = u.info.combinations;
  if (c !== null && !c.ok) return { ok: false, key, reason: "refused", detail: `/combinations of ${baseName(u.path)} is refused (${c.reason}), so nothing is added to it` };
  const j = judgeDraft(t?.rows ?? [], t?.modelRows ?? [], draft);
  if (!j.ok) return { ok: false, key, reason: j.reason, detail: j.detail };
  let cases: string[];
  try {
    cases = spectrumCasesOf(j.table, j.name);
  } catch (err) {
    if (err instanceof CombinationRefused) return { ok: false, key, reason: err.reason, detail: err.detail };
    throw err;
  }
  const f = cases.length ? combinationFieldsOf(s, j.name, j.table) : null;
  const comp = f?.ok ? f.fields[0]?.components[0]?.key ?? null : null;
  const slash = comp === null ? -1 : comp.indexOf("/");
  const quantity = comp === null || slash < 0 ? null : { label: comp.slice(0, slash), component: comp.slice(slash + 1) };
  const sp = u.info.spectra;
  const staleness = sp === null ? { stale: false, reason: "" } : sp.stale;
  return { ok: true, key, path: u.path, table: j.table, name: j.name, cases, quantity, staleness };
}

/** Why the panel may not ask main for `op` now, or null. The save token is the read's: a save is never made on another. */
export function combineCannotSave(s: State, c: CombineState, op: "add" | "revert"): string | null {
  const u = s.user;
  if (u?.status !== "ready") return u === null ? "the set has no .user.h5 (write one with Python first)" : "the user file is not read";
  if ((u.token ?? null) === null) return "this window does not save: only the human's window writes the user file";
  if (c.saving !== null) return "a save is running";
  if (op === "add") {
    const key = draftKeyOf(s, c.draft);
    if (c.check === null || c.check.key !== key || c.check.status === "checking") return "the draft is being checked";
    if (c.check.status === "refused") return `the draft is refused (${c.check.reason}): ${c.check.detail}`;
    return null;
  }
  // A7.3, one revert: it undoes the save made last in this window, while the file is as that save left it.
  const last = c.last;
  if (last === null || last.file !== u.path) return "Revert undoes a save made in this window, and none was made on this file";
  if (last.op === "revert") return "the last save was reverted already: the app gives one revert";
  if (last.target === null) return "the last save created /combinations: there is no generation before it";
  const g = u.info.generations ?? null;
  if (u.token !== last.token || g?.current !== last.generation) return "the file changed since the last save made here, so that save is not undone";
  return null;
}

/** Whether combination `name` is signed, from the descriptors (A3), or null with why the table cannot say. */
function combinationSign(s: State, name: string): { signed: boolean } | { signed: null; why: string } {
  const table = combinationTableOf(s);
  if (!table) return { signed: null, why: `User Combination ${quoted(name)} is not in the user file` };
  try {
    return { signed: isSigned(table, name) };
  } catch (err) {
    if (err instanceof CombinationRefused) return { signed: null, why: err.message };
    throw err;
  }
}

/** A combination's colour range: the user's when fixed, else its own values (one state: there is no history to scan). */
function combinationClim(c: ContourState, range: { min: number; max: number } | null): Clim {
  if (c.range === "fixed") return climOf(c, null, null, "");
  return { min: range?.min ?? null, max: range?.max ?? null, mode: c.range, label: "range: this combination (one state, no history)" };
}

/** `contourView` on a User Combination. */
function combinationContour(s: State, name: string): ContourView {
  if (!s.mesh) return { state: "unavailable", reason: "no mesh is loaded" };
  const f = combinationFieldsOf(s, name);
  if (!f.ok) return { state: "unavailable", reason: f.reason };
  const c = s.contour;
  if (c.source !== "nodal" || c.field === null || c.component === null) return { state: "unavailable", reason: "pick a field" };
  const hit = f.fields.find((x) => x.name === c.field)?.components.find((x) => x.key === c.component);
  if (!hit) return { state: "unavailable", reason: `User Combination ${quoted(name)} has no ${c.field} ${componentName(c.component)}` };
  const want = combinationRequestOf(s)!;
  const r = s.combination;
  if (!r || r.key !== want.key || r.status === "loading") return { state: "unavailable", reason: `User Combination ${quoted(name)} is being evaluated` };
  if (r.status === "refused") return { state: "refused", code: "combination", reason: `User Combination ${quoted(name)} is refused (${r.reason}): ${r.detail}` };
  if (r.status === "failed") return { state: "refused", code: "combination", reason: `User Combination ${quoted(name)} could not be evaluated: ${r.error}` };
  return { state: "combination", name, field: c.field, component: hit, key: r.key, clim: combinationClim(c, r.range), cmap: c.cmap, signed: r.signed };
}

/**
 * `deformOf` on a User Combination. An unsigned one (a modal rule, an
 * envelope, or a `linear` over one) is refused with the reason: it has lost
 * the signs a shape needs (ADR 0130 D4, A3). A signed one is not drawn yet,
 * because the deformed shape reads a results step's displacement.
 */
function combinationDeform(s: State, name: string): { ok: false; state: "unavailable" | "refused"; reason: string } {
  const sign = combinationSign(s, name);
  if (sign.signed === null) return { ok: false, state: "unavailable", reason: sign.why };
  if (!sign.signed) {
    const rule = combinationTableOf(s)!.row(name, name).rule;
    return { ok: false, state: "refused", reason: `User Combination ${quoted(name)} (${rule}) is unsigned, so it is shown as a contour only, never as a deformed shape (ADR 0130 D4, A3)` };
  }
  return { ok: false, state: "unavailable", reason: `User Combination ${quoted(name)} is signed; this app draws the deformed shape of a results step only` };
}

// -- VU16: a state-free user result as a contour (ADR 0130 D7) ---------------------

/**
 * Whether user result `name` of the set's `.user.h5` can be drawn as a
 * contour: its entry, or why not. Only a state-free `ON_NODES` entry the file
 * lists (a refused one never) can; staleness (A6) is judged by the caller, so a
 * stale entry is picked and refused with its reason, never silently dropped.
 */
export function userResultPickOf(s: State, name: string): { ok: true; entry: UserResult } | { ok: false; reason: string } {
  const u = s.user;
  if (u === null) return { ok: false, reason: "the set has no .user.h5" };
  if (u.status === "reading") return { ok: false, reason: `${baseName(u.path)} is being read` };
  if (u.status === "failed") return { ok: false, reason: `the user file is refused: ${u.error}` };
  const file = baseName(u.path);
  const r = u.info.results;
  if (r === null) return { ok: false, reason: `${file} holds no /results entry, so it has no user result ${quoted(name)}` };
  if (!r.ok) return { ok: false, reason: `/results of ${file} is refused: ${r.reason}` };
  const entry = r.items.find((x) => x.name === name);
  if (!entry) {
    const why = r.refused.find((x) => x.name === name);
    return { ok: false, reason: why ? `user result ${quoted(name)} is refused: ${why.reason} (${why.detail})` : `${file} lists no user result ${quoted(name)}` };
  }
  if (entry.target !== "nodes") return { ok: false, reason: `user result ${quoted(name)} is on ${entry.target}; this app draws a user result on nodes only` };
  if (entry.perState) return { ok: false, reason: `user result ${quoted(name)} is per state; this app draws a state-free one only` };
  return { ok: true, entry };
}

/**
 * What the app reads for the user result picked, or null when nothing is to
 * be read: not colouring by a user result, a difference or a User Combination
 * shown instead, a pick the file cannot give, or a stale entry (A6: refused,
 * never read). `key` names the read: the user file, the entry, the component.
 */
export function userResultRequestOf(s: State): { key: string; path: string; name: string; component: string } | null {
  const c = s.contour;
  if (s.visibility.colourBy !== "contour" || c.source !== "user" || c.field === null || c.component === null) return null;
  // VU19: a compare user result drawn is read by its own door; the window's file is not read under it.
  if (s.variants.difference || s.variants.shown !== null || s.phase.at?.kind === "combination") return null;
  const p = userResultPickOf(s, c.field);
  if (!p.ok || p.entry.stale.stale || !p.entry.components.includes(c.component)) return null;
  const path = s.user!.path;
  return { key: JSON.stringify([path, c.field, c.component]), path, name: c.field, component: c.component };
}

/** `contourView` on a user result: what its read gave, or why it draws nothing. */
function userResultContour(s: State): ContourView {
  if (!s.mesh) return { state: "unavailable", reason: "no mesh is loaded" };
  const c = s.contour;
  if (c.field === null || c.component === null) return { state: "unavailable", reason: "pick a field" };
  const label = `user result ${quoted(c.field)}`;
  if (s.user?.status === "reading") return { state: "unavailable", reason: `${label}: ${baseName(s.user.path)} is being read` };
  const p = userResultPickOf(s, c.field);
  if (!p.ok) return { state: "refused", code: "user", reason: p.reason };
  if (p.entry.stale.stale) return { state: "refused", code: "user", reason: `${label} is stale (A6): ${p.entry.stale.reason || "no reason given"}` };
  if (!p.entry.components.includes(c.component)) {
    return { state: "refused", code: "user", reason: `${label} has no component ${quoted(c.component)}; its components are ${p.entry.components.join(", ")}` };
  }
  const want = userResultRequestOf(s)!;
  const r = s.userResult;
  if (!r || r.key !== want.key || r.status === "loading") return { state: "unavailable", reason: `${label} is being read` };
  if (r.status === "refused") return { state: "refused", code: "user", reason: `${label} is refused: ${r.detail}` };
  if (r.status === "failed") return { state: "refused", code: "user", reason: `${label} could not be read: ${r.error}` };
  // One state: the user's range when fixed, else the column's own (there is no history to scan).
  const clim: Clim =
    c.range === "fixed" ? climOf(c, null, null, "") : { min: r.range?.min ?? null, max: r.range?.max ?? null, mode: c.range, label: "range: this user result (one state, no history)" };
  return {
    state: "user",
    name: c.field,
    field: c.field,
    compare: null,
    component: { key: c.component, label: c.component, datasets: [c.component], magnitude: false },
    key: r.key,
    read: r.read,
    basis: p.entry.basis,
    clim,
    cmap: c.cmap,
    nodes: r.nodes,
    finite: r.finite,
  };
}

// -- VU10: the difference of two variants (ADR 0132 D9.5) -------------------------

/** What the difference shown reads, or why there is none to read (a refusal); null when no difference is asked for. */
export type DifferencePlan =
  | {
      ok: true;
      /** names the read: both results files and reads, the compare row, the stage, the step and the dataset */
      key: string;
      window: string;
      compare: string;
      /** the compare row (the runner asks main for its step by row; never by path) */
      row: { index: number; label: string };
      stageId: string;
      stageName: string;
      step: number;
      field: string;
      component: FieldComponent;
      /** the one dataset subtracted (a component, never a magnitude) */
      dataset: string;
      /** the window stage's `_ids` blob */
      nodeIds: BlobRef;
    }
  | { ok: false; reason: string };

/** Two times are one step's when they agree to 1e-12 of their size. */
const sameTime = (a: number, b: number): boolean => Math.abs(a - b) <= 1e-12 * Math.max(1, Math.abs(a), Math.abs(b));

/**
 * VU10: the difference the state asks for. It is the window variant minus the
 * compare variant, at the stage and step shown, of one nodal component, joined
 * by node id. It is refused, with the reason, on a User Combination (a
 * combination across variants is refused: ADR 0130 A7, ADR 0132 D8), when the
 * two results do not state one mesh (`fem_hash`), when the compare set has no
 * such stage, step time or dataset, and for an element field or a magnitude.
 */
export function differencePlanOf(s: State): DifferencePlan | null {
  const v = s.variants;
  if (!v.difference || s.visibility.colourBy !== "contour") return null;
  const c = v.compare;
  if (c?.status !== "open") return { ok: false, reason: "no compare set is open" };
  const at = s.phase.at;
  if (at?.kind === "combination") {
    return { ok: false, reason: `a User Combination is of the window variant's own states, so a combination across variants is refused: ${quoted(at.name)} is not subtracted from ${quoted(c.label)}` };
  }
  // VU16: so is a user result, of the window variant's own .user.h5.
  if (s.contour.source === "user") return { ok: false, reason: "a user result is of the window variant's own file, so it is not subtracted across variants" };
  if (!s.results || resultsPairing(s).state !== "paired") return { ok: false, reason: "no results file pairs with this model, so there is nothing to subtract from" };
  if (at?.kind !== "results") return { ok: false, reason: "a difference is of a results step: choose results in the phase bar" };
  const st = s.results.stages[at.stage];
  if (!st) return { ok: false, reason: `results stage ${at.stage} is not in the file` };
  const mine = s.results.lineage?.femHash ?? null;
  if (mine === null || c.femHash === null || mine !== c.femHash) {
    return { ok: false, reason: `the two variants' results do not state one mesh (fem_hash ${mine ?? "not stamped"} and ${c.femHash ?? "not stamped"}), so their nodes are not subtracted` };
  }
  const label = st.name || st.id;
  const other = c.stages.find((x) => x.id === st.id);
  if (!other) return { ok: false, reason: `the compare variant ${quoted(c.label)} has no stage ${quoted(st.id)}` };
  // W4d #2843: the two stages are one program stage by their program names; a display name is a label only.
  if (other.programName !== st.programName) {
    return { ok: false, reason: `stage ${quoted(st.id)} is program stage ${quoted(st.programName)} here and ${quoted(other.programName)} in ${quoted(c.label)}` };
  }
  const k = at.step;
  const t = st.time[k];
  const u = other.time[k];
  if (t === undefined || u === undefined) return { ok: false, reason: `the compare variant ${quoted(c.label)} has no step ${k} of stage ${quoted(label)}` };
  if (!sameTime(t, u)) return { ok: false, reason: `step ${k} of stage ${quoted(label)} is at time ${t} here and ${u} in ${quoted(c.label)}` };
  const pick = s.contour;
  if (pick.source !== "nodal") return { ok: false, reason: "a difference is of a nodal component; an element field is not subtracted" };
  if (pick.field === null || pick.component === null) return { ok: false, reason: "pick a field" };
  const hit = componentOf(st, pick.field, pick.component);
  if (!hit) return { ok: false, reason: `stage ${label} has no ${pick.field} ${componentName(pick.component)}` };
  if (hit.component.magnitude || hit.component.datasets.length !== 1) return { ok: false, reason: "a difference is of one component; a magnitude is not subtracted" };
  const dataset = hit.component.datasets[0]!;
  if (!other.nodal.includes(dataset)) return { ok: false, reason: `the compare variant ${quoted(c.label)} records no ${dataset} in stage ${quoted(label)}` };
  if (!st.nodeIds) return { ok: false, reason: `the node ids of stage ${label} are not loaded` };
  const study = v.study;
  const windowRow = study?.rows.find((r) => r.index === study.active);
  const window = windowRow?.label ?? baseName(s.artifacts.model?.path ?? s.results.path);
  return {
    ok: true,
    key: JSON.stringify([s.results.path, pick.reads, c.results, c.index, st.id, k, dataset]),
    window,
    compare: c.label,
    row: { index: c.index, label: c.label },
    stageId: st.id,
    stageName: label,
    step: k,
    field: pick.field,
    component: hit.component,
    dataset,
    nodeIds: st.nodeIds,
  };
}

/**
 * VU10: the study's row labelled `label`, as a spec's `compare` names it
 * (spec/compile.ts `VariantOf`), and whether it is the compare set already;
 * refused when no study is open, no passing row has the label, or it is the
 * window set (INV-6).
 */
export function variantRowOf(s: State, label: string): { ok: true; row: { index: number; label: string }; open: boolean } | { ok: false; reason: string } {
  const study = s.variants.study;
  if (!study) return { ok: false, reason: `no variants manifest is open, so there is no variant ${quoted(label)}` };
  const row = study.rows.find((r) => r.label === label);
  if (!row) {
    const refused = study.refused.find((r) => r.label === label);
    return { ok: false, reason: refused ? `variant ${quoted(label)} is refused (${refused.code}: ${refused.reason})` : `the manifest has no variant ${quoted(label)}; its variants are ${study.rows.map((r) => quoted(r.label)).join(", ") || "none"}` };
  }
  if (study.active === row.index) return { ok: false, reason: `variant ${quoted(label)} is the window set; a compare set is another variant` };
  const c = s.variants.compare;
  return { ok: true, row: { index: row.index, label: row.label }, open: c !== null && c.status !== "failed" && c.index === row.index && c.label === row.label };
}

// -- VU19: the compare set's user results (ADR 0132 D9.5) ----------------------------

/**
 * The compare set's user result `name`, as the compare set's own `.user.h5`
 * lists it with A6's verdict against the compare set's own model and results
 * (main's `compare:userRead`), or why there is none. Only a state-free
 * `ON_NODES` entry can be drawn; staleness is judged by the caller, so a stale
 * entry is picked and refused with its reason, never silently dropped.
 */
export function compareUserPickOf(s: State, name: string): { ok: true; entry: UserResult; label: string } | { ok: false; reason: string } {
  const c = s.variants.compare;
  if (c?.status !== "open") return { ok: false, reason: "no compare set is open" };
  const u = s.variants.compareUser;
  const of = `the compare variant ${quoted(c.label)}`;
  if (u === null || u.status === "reading") return { ok: false, reason: `the user file of ${of} is being read` };
  if (u.status === "none") return { ok: false, reason: `${of} has no .user.h5 (${baseName(u.path)})` };
  if (u.status === "failed") return { ok: false, reason: `the user file of ${of} is refused: ${u.error}` };
  const file = baseName(u.path);
  const r = u.info.results;
  if (r === null) return { ok: false, reason: `${file} of ${of} holds no /results entry, so it has no user result ${quoted(name)}` };
  if (!r.ok) return { ok: false, reason: `/results of ${file} of ${of} is refused: ${r.reason}` };
  const entry = r.items.find((x) => x.name === name);
  if (!entry) {
    const why = r.refused.find((x) => x.name === name);
    // VF2-d (#2956): a spec names the entry, so an unknown one lists the entries the file offers.
    const offers = r.items.map((x) => quoted(x.name)).join(", ") || "none";
    return { ok: false, reason: why ? `user result ${quoted(name)} of ${of} is refused: ${why.reason} (${why.detail})` : `${file} of ${of} lists no user result ${quoted(name)}; its user results are ${offers}` };
  }
  if (entry.target !== "nodes") return { ok: false, reason: `user result ${quoted(name)} of ${of} is on ${entry.target}; this app draws a user result on nodes only` };
  if (entry.perState) return { ok: false, reason: `user result ${quoted(name)} of ${of} is per state; this app draws a state-free one only` };
  return { ok: true, entry, label: c.label };
}

/**
 * Whether the compare set's values can be placed on the window's mesh: only
 * when the compare set's model (main's `compare:model`) and the window's model
 * state one mesh by `fem_hash`, so a node id names one node on both. Two meshes
 * that do not pair are refused with the reason, never interpolated.
 */
export function compareMeshOf(s: State): { ok: true } | { ok: false; reading: boolean; reason: string } {
  const c = s.variants.compare;
  if (c?.status !== "open") return { ok: false, reading: false, reason: "no compare set is open" };
  const m = s.variants.compareModel;
  if (m === null || m.status === "reading") return { ok: false, reading: true, reason: `the model of the compare variant ${quoted(c.label)} is being read` };
  if (m.status === "failed") return { ok: false, reading: false, reason: `the model of the compare variant ${quoted(c.label)} is refused: ${m.reason}` };
  const mine = s.lineage?.femHash ?? null;
  if (mine === null) return { ok: false, reading: false, reason: `the window's model stamps no fem_hash, so the nodes of ${quoted(c.label)} cannot be placed on it` };
  if (mine !== m.femHash) {
    return { ok: false, reading: false, reason: `the two variants' models are not one mesh (fem_hash ${mine} here, ${m.femHash} in ${quoted(c.label)}), so its values are not placed on this mesh; nodes are never interpolated` };
  }
  return { ok: true };
}

/**
 * What the app reads for the compare user result shown, or null when nothing
 * is to be read: not colouring by a contour, nothing picked, the difference on,
 * the two meshes not one, a pick the file cannot give, or a stale entry (A6 by
 * its own verdict: refused, never read). `key` names the read: the compare row,
 * its user file, the entry and the component.
 */
export function compareUserRequestOf(s: State): { key: string; name: string; component: string } | null {
  const v = s.variants;
  const pick = v.shown;
  if (s.visibility.colourBy !== "contour" || pick === null || v.difference) return null;
  if (!compareMeshOf(s).ok) return null;
  const p = compareUserPickOf(s, pick.name);
  if (!p.ok || p.entry.stale.stale || !p.entry.components.includes(pick.component)) return null;
  const c = v.compare as Extract<NonNullable<typeof v.compare>, { status: "open" }>;
  const u = v.compareUser as Extract<NonNullable<typeof v.compareUser>, { status: "ready" }>;
  return { key: JSON.stringify(["compare", c.index, c.label, u.path, pick.name, pick.component]), name: pick.name, component: pick.component };
}

/**
 * The display string of compare variant `variant`'s user result `entry`, as
 * the legend, the banner, a still's scale and the manifest's `user.name` show
 * it. It can equal a window entry of that literal name, so it is never read
 * back as a name: the manifest states `user.compare` apart (VF2-d, #2956).
 */
export function compareUserName(variant: string, entry: string): string {
  return `${variant}: ${entry}`;
}

/** `contourView` on a compare user result: what its read gave, or why it draws nothing. */
function compareUserContour(s: State, pick: { name: string; component: string }): ContourView {
  if (!s.mesh) return { state: "unavailable", reason: "no mesh is loaded" };
  const cmp = s.variants.compare;
  if (cmp?.status === "opening") return { state: "unavailable", reason: `the compare variant ${quoted(cmp.label)} is being opened` };
  const label = `user result ${quoted(pick.name)} of the compare variant ${quoted(cmp?.label ?? "")}`;
  const mesh = compareMeshOf(s);
  if (!mesh.ok) return mesh.reading ? { state: "unavailable", reason: mesh.reason } : { state: "refused", code: "user", reason: `${label} is not drawn: ${mesh.reason}` };
  if (s.variants.compareUser?.status === "reading") return { state: "unavailable", reason: `the user file of the compare variant ${quoted(cmp!.label)} is being read` };
  const p = compareUserPickOf(s, pick.name);
  if (!p.ok) return { state: "refused", code: "user", reason: p.reason };
  if (p.entry.stale.stale) return { state: "refused", code: "user", reason: `${label} is stale (A6, against its own model and results): ${p.entry.stale.reason || "no reason given"}` };
  if (!p.entry.components.includes(pick.component)) {
    return { state: "refused", code: "user", reason: `${label} has no component ${quoted(pick.component)}; its components are ${p.entry.components.join(", ")}` };
  }
  const want = compareUserRequestOf(s)!;
  const r = s.variants.shownRead;
  if (!r || r.key !== want.key || r.status === "loading") return { state: "unavailable", reason: `${label} is being read` };
  if (r.status === "refused") return { state: "refused", code: "user", reason: `${label} is refused: ${r.detail}` };
  if (r.status === "failed") return { state: "refused", code: "user", reason: `${label} could not be read: ${r.error}` };
  const c = s.contour;
  const clim: Clim =
    c.range === "fixed" ? climOf(c, null, null, "") : { min: r.range?.min ?? null, max: r.range?.max ?? null, mode: c.range, label: "range: this user result (one state, no history)" };
  // The legend and the manifest name the compare variant with the entry: it is not the window set's file.
  // VF2-d (#2956): the variant and the entry also stand apart, since the display string can equal a window entry's name.
  const shownName = compareUserName(p.label, pick.name);
  return {
    state: "user",
    name: shownName,
    field: shownName,
    compare: { variant: p.label, entry: pick.name },
    component: { key: pick.component, label: pick.component, datasets: [pick.component], magnitude: false },
    key: r.key,
    read: r.read,
    basis: p.entry.basis,
    clim,
    cmap: c.cmap,
    nodes: r.nodes,
    finite: r.finite,
  };
}

/** `contourView` with the difference on: what its read gave, or why it draws nothing. */
function differenceContour(s: State): ContourView {
  if (!s.mesh) return { state: "unavailable", reason: "no mesh is loaded" };
  const cmp = s.variants.compare;
  if (cmp?.status === "opening") return { state: "unavailable", reason: `the compare variant ${quoted(cmp.label)} is being opened` };
  const p = differencePlanOf(s)!;
  if (!p.ok) return { state: "refused", code: "difference", reason: `the difference is refused: ${p.reason}` };
  const r = s.variants.read;
  if (!r || r.key !== p.key || r.status === "loading") return { state: "unavailable", reason: `the difference ${p.window} − ${p.compare} is being read` };
  if (r.status === "refused") return { state: "refused", code: "difference", reason: `the difference is refused: ${r.reason}` };
  const c = s.contour;
  const clim: Clim =
    c.range === "fixed" ? climOf(c, null, null, "") : { min: r.range?.min ?? null, max: r.range?.max ?? null, mode: c.range, label: "range: this difference (the step shown)" };
  return { state: "difference", window: p.window, compare: p.compare, field: p.field, component: p.component, stageName: p.stageName, key: p.key, clim, cmap: c.cmap, nodes: r.nodes, missing: r.missing };
}
