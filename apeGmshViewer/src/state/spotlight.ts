// Spotlight (`Ctrl+K` / `Cmd+K`) state, exhaustive EventType ledger, and pure
// command & declaration palette selector (ADR 0112 D6).
//
// Architectural guarantees:
// 1. Exhaustive EventType ledger (`SPOTLIGHT_EVENT_LEDGER`): every member of
//    `EventType` must be classified as `"expose"`, `"gesture"`, or `"system"`.
//    Adding a new `Event` to `types.ts` without classifying it here is a
//    TypeScript compile error, and `test/spotlight.test.ts` checks that every
//    entry carries a non-empty reason/command.
// 2. One search engine: Spotlight's declaration & filter queries delegate
//    directly to `src/state/search.ts` (`parseSearchText`, `runSearchTerms`,
//    `indexOfModel`), so Spotlight and the Search window never drift.

import { CMAP_NAMES, fieldsOf, gaussFields, RANGE_MODES } from "./contour.ts";
import { clauseLabel, indexOfModel, parseSearchText, runSearchTerms, type ParsedTerm } from "./search.ts";
import {
  WINDOW_KINDS,
  type ColourBy,
  type DeclKind,
  type DeclPath,
  type Event,
  type EventType,
  type GaussMode,
  type SearchClause,
  type SearchState,
  type SpotlightMode,
  type SpotlightState,
  type State,
} from "./types.ts";
import { UNIT_DISPLAY_OPTIONS, UNIT_SYSTEM_IDS, UNIT_SYSTEMS } from "./units.ts";

export const initialSpotlight: SpotlightState = {
  open: false,
  mode: "root",
  query: "",
  clauses: [],
  seq: 0,
};

export const initialSearch: SearchState = {
  text: "",
  clauses: [],
  seq: 0,
};

/**
 * Classification of every `EventType` in the store:
 * - `"expose"`: user-facing action or property exposed in Spotlight (`command` names the Spotlight entry id)
 * - `"gesture"`: direct pointer/scrub/spotlight-input gesture with a stated reason why it is not a standalone command
 * - `"system"`: file lifecycle or async effect reply with a stated reason why it is not user-dispatched
 */
export type EventLedgerEntry =
  | { readonly kind: "expose"; readonly command: string; readonly description: string }
  | { readonly kind: "gesture"; readonly reason: string }
  | { readonly kind: "system"; readonly reason: string };

export const SPOTLIGHT_EVENT_LEDGER: { readonly [K in EventType]: EventLedgerEntry } = {
  // System / Effect lifecycle events
  fileOpened: { kind: "system", reason: "dispatched by the file loader when an artifact path is opened" },
  setLoaded: { kind: "system", reason: "dispatched by the loader when model and geometry reload together" },
  fileLoaded: { kind: "system", reason: "dispatched by the loader when an artifact finishes reading" },
  fileFailed: { kind: "system", reason: "dispatched by the loader when reading an artifact fails" },
  fileClosed: { kind: "system", reason: "dispatched by the loader when an open set closes an artifact" },
  fileChanged: { kind: "system", reason: "dispatched by the file watcher when an artifact changes on disk" },
  zoneRefused: { kind: "system", reason: "dispatched by the loader when a schema zone version is refused" },
  sourceResult: { kind: "system", reason: "dispatched by effects with the outcome of a go-to-source request" },
  resultsLoaded: { kind: "system", reason: "dispatched by effects when results metadata finishes reading" },
  resultStep: { kind: "system", reason: "dispatched by effects to report step fetch status" },
  contourElementJoin: { kind: "system", reason: "dispatched by effects after verifying element contour join" },
  contourStepRange: { kind: "system", reason: "dispatched by effects after measuring step min/max" },
  contourScan: { kind: "system", reason: "dispatched by effects during history-range scan" },
  deformMeasured: { kind: "system", reason: "dispatched by effects after counting drawn nodes without displacement" },
  lineDiagramStepRange: { kind: "system", reason: "dispatched by effects after measuring line-diagram step min/max" },
  lineDiagramScan: { kind: "system", reason: "dispatched by effects during line-diagram history-range scan" },

  // Pointer / Input gestures
  hover: { kind: "gesture", reason: "continuous pointer hover over a declaration or 3D primitive" },
  setClipPose: { kind: "gesture", reason: "continuous 3D gizmo rotation drag; Spotlight exposes discrete plane normal/offset controls instead" },
  openSpotlight: { kind: "gesture", reason: "Ctrl+K / header trigger that opens the Spotlight palette itself" },
  closeSpotlight: { kind: "gesture", reason: "Escape / backdrop gesture that closes the Spotlight palette" },
  setSpotlightMode: { kind: "gesture", reason: "internal drill-down navigation within the Spotlight palette" },
  setSpotlightQuery: { kind: "gesture", reason: "typing into the Spotlight search input" },
  addSpotlightClause: { kind: "gesture", reason: "Tab gesture in Spotlight to pin a filter chip" },
  removeSpotlightClause: { kind: "gesture", reason: "Backspace / chip click to remove a pinned filter chip in Spotlight" },
  clearSpotlightClauses: { kind: "gesture", reason: "clearing pinned filter chips in Spotlight" },
  setSearchText: { kind: "gesture", reason: "typing into the Search window text input" },
  addSearchClause: { kind: "gesture", reason: "adding a filter chip in the Search window" },
  removeSearchClause: { kind: "gesture", reason: "removing a filter chip in the Search window" },
  clearSearch: { kind: "gesture", reason: "resetting the Search window filters" },

  // Exposed user commands & viewer properties
  select: { kind: "expose", command: "select-decl", description: "Select a declaration in the 3D viewport and inspector" },
  selectAdd: { kind: "expose", command: "select-add-decl", description: "Add a declaration to the current selection (Shift+Enter)" },
  selectDecls: { kind: "expose", command: "select-batch", description: "Select all declarations matching a filter or group" },
  clearSelection: { kind: "expose", command: "clear-selection", description: "Clear the current selection" },
  setHidden: { kind: "expose", command: "toggle-hidden", description: "Hide or show declarations or physical groups" },
  isolate: { kind: "expose", command: "isolate-decls", description: "Isolate declarations or physical groups in the 3D viewport" },
  showAll: { kind: "expose", command: "show-all", description: "Show all hidden groups and elements" },
  setEdges: { kind: "expose", command: "toggle-edges", description: "Toggle mesh outline edges on or off" },
  setOpacity: { kind: "expose", command: "set-opacity", description: "Set mesh surface opacity (0..1)" },
  setColourBy: { kind: "expose", command: "set-colour-by", description: "Set mesh colouring mode (by group, by role, or contour)" },
  setPhase: { kind: "expose", command: "set-phase", description: "Switch phase on the phase axis (geometry, mesh, or results stage)" },
  setResultStep: { kind: "expose", command: "set-result-step", description: "Jump to a results time step" },
  inspectorPin: { kind: "expose", command: "inspector-pin", description: "Pin a declaration to the inspector" },
  inspectorUnpin: { kind: "expose", command: "inspector-unpin", description: "Unpin a declaration from the inspector" },
  openWindow: { kind: "expose", command: "open-window", description: "Open a model-definition or search window" },
  closeWindow: { kind: "expose", command: "close-window", description: "Close an open window" },
  requestSource: { kind: "expose", command: "request-source", description: "Open a declaration's Python source location in the editor" },
  frameSelection: { kind: "expose", command: "frame-selection", description: "Frame the current selection or fit the whole model (F)" },
  selectStage: { kind: "expose", command: "select-stage", description: "Select a construction stage in the sequence" },
  setContourField: { kind: "expose", command: "set-contour-field", description: "Pick the nodal or element contour field and component" },
  setContourGauss: { kind: "expose", command: "set-contour-gauss", description: "Set the element (Gauss) contour topology mode" },
  setContourRange: { kind: "expose", command: "set-contour-range", description: "Set the contour colour range mode (history, step, or fixed)" },
  setContourCmap: { kind: "expose", command: "set-contour-cmap", description: "Set the contour colormap (viridis, coolwarm, turbo, etc.)" },
  setContourUnits: { kind: "expose", command: "set-contour-units", description: "Set the contour native or display unit system (baseUnits)" },
  setDeform: { kind: "expose", command: "toggle-deform", description: "Toggle the deformed shape on or off" },
  setDeformScale: { kind: "expose", command: "set-deform-scale", description: "Set the deformation scale factor" },
  setReferenceGhost: { kind: "expose", command: "toggle-ghost", description: "Toggle the undeformed reference ghost mesh" },
  addClipPlane: { kind: "expose", command: "add-clip-plane", description: "Add a section plane along X, Y, Z" },
  removeClipPlane: { kind: "expose", command: "remove-clip-plane", description: "Remove a section plane" },
  setClipActive: { kind: "expose", command: "toggle-clip-active", description: "Toggle a section plane active or inactive" },
  setClipFlipped: { kind: "expose", command: "flip-clip-plane", description: "Flip a section plane's normal side" },
  setClipOffset: { kind: "expose", command: "set-clip-offset", description: "Set a section plane's offset along its normal" },
  setClipNormal: { kind: "expose", command: "set-clip-normal", description: "Set a section plane's normal axis" },
  setApplyCuts: { kind: "expose", command: "toggle-apply-cuts", description: "Toggle section plane cuts on or off globally" },
  setShowGizmos: { kind: "expose", command: "toggle-show-gizmos", description: "Toggle 3D section plane gizmos on or off" },
  setClipGizmo: { kind: "expose", command: "toggle-plane-gizmo", description: "Toggle one section plane's 3D gizmo eye" },
  setCutFace: { kind: "expose", command: "toggle-cut-face", description: "Toggle solid cut-face cap fill on cutting planes" },
  setLineDiagram: { kind: "expose", command: "toggle-line-diagram", description: "Toggle 3-D beam/frame line diagrams on or off" },
  setLineDiagramComponent: { kind: "expose", command: "set-line-diagram-component", description: "Select 3-D line diagram force/moment component" },
  setLineDiagramScale: { kind: "expose", command: "set-line-diagram-scale", description: "Set 3-D line diagram amplitude scale factor" },
  setLineDiagramPlane: { kind: "expose", command: "set-line-diagram-plane", description: "Set 3-D line diagram projection plane (auto, local, global, screen)" },
  setLineDiagramCmap: { kind: "expose", command: "set-line-diagram-cmap", description: "Set 3-D line diagram colour mode (sign or colormap)" },
  setLineDiagramFill: { kind: "expose", command: "toggle-line-diagram-fill", description: "Toggle 3-D line diagram ribbon fill" },
  setLineDiagramFlipSign: { kind: "expose", command: "toggle-line-diagram-flip-sign", description: "Toggle 3-D line diagram tension-side / sign convention" },
  setLineDiagramRange: { kind: "expose", command: "set-line-diagram-range", description: "Set 3-D line diagram colour range mode (history, step, or fixed)" },
  setLoads: { kind: "expose", command: "toggle-loads", description: "Toggle or configure 3-D structural load and prescribed-SP glyphs" },
};

// ---- Spotlight Item Model ---------------------------------------------------

export interface SpotlightAction {
  readonly events: readonly Event[];
  readonly close: boolean;
  readonly nextMode?: SpotlightMode;
  readonly tabClause?: Omit<SearchClause, "id">;
  readonly tabClauses?: readonly Omit<SearchClause, "id">[];
  readonly shiftEvents?: readonly Event[];
  readonly shiftMode?: SpotlightMode;
  readonly altEvents?: readonly Event[];
  readonly ctrlEvents?: readonly Event[];
}

export interface SpotlightItem {
  readonly id: string;
  readonly category: "command" | "search" | "filter" | "group" | "declaration" | "element" | "value";
  readonly section: string;
  readonly badge: string;
  readonly title: string;
  readonly detail: string;
  readonly subtitle: string;
  readonly active?: boolean;
  readonly decl?: DeclPath;
  readonly events: readonly Event[];
  readonly keepOpen?: boolean;
  readonly tabClause?: Omit<SearchClause, "id">;
  readonly tabClauses?: readonly Omit<SearchClause, "id">[];
  readonly shiftEvents?: readonly Event[];
  readonly altEvents?: readonly Event[];
  readonly ctrlEvents?: readonly Event[];
  readonly action: SpotlightAction;
}

export interface SpotlightClauseView extends SearchClause {
  readonly label: string;
}

export interface SpotlightView {
  readonly open: boolean;
  readonly mode: SpotlightMode;
  readonly modeLabel: string | null;
  readonly modeTitle: string;
  readonly placeholder: string;
  readonly query: string;
  readonly clauses: readonly SpotlightClauseView[];
  readonly summary: string | null;
  readonly items: readonly SpotlightItem[];
  readonly hint: string;
}

export const SPOTLIGHT_ITEM_CAP = 32;

const MODE_META: Readonly<Record<SpotlightMode, { label: string | null; placeholder: string }>> = {
  root: {
    label: null,
    placeholder: "Type a group/type (e.g. beam_V50x70 and beam_V95x70), a command (set opacity), or ? for help…",
  },
  help: {
    label: "Help & Examples",
    placeholder: "Filter help topics or press Enter on any example to try it…",
  },
  opacity: {
    label: "Set Opacity",
    placeholder: "Pick a preset opacity or type a number (0..1 or 0..100%)…",
  },
  colourBy: {
    label: "Colour By",
    placeholder: "Pick how the mesh is coloured…",
  },
  phase: {
    label: "Phase / Stage",
    placeholder: "Pick a phase or construction stage…",
  },
  step: {
    label: "Result Step",
    placeholder: "Pick a result step or type a 1-based step number…",
  },
  contourField: {
    label: "Contour Field",
    placeholder: "Filter nodal or element (Gauss) contour fields…",
  },
  contourGauss: {
    label: "Element Contour Mode",
    placeholder: "Pick how element (Gauss) fields are drawn…",
  },
  contourRange: {
    label: "Contour Range",
    placeholder: "Pick history, step, or type `min max` for a fixed range…",
  },
  contourCmap: {
    label: "Contour Colormap",
    placeholder: "Pick a colormap…",
  },
  contourUnits: {
    label: "Contour Units",
    placeholder: "Pick display units or native file units (baseUnits)…",
  },
  deformScale: {
    label: "Deform Scale",
    placeholder: "Pick a preset scale or type a positive number…",
  },
  windows: {
    label: "Windows",
    placeholder: "Open or close a window…",
  },
  planes: {
    label: "Section Planes",
    placeholder: "Add, flip, toggle, or set offset of section planes…",
  },
  selectionActions: {
    label: "Action on Matched Set",
    placeholder: "Type hide, isolate, show, or add — then press Enter…",
  },
};

function normText(s: string): string {
  return s
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase();
}

function makeItem(spec: {
  id: string;
  category: SpotlightItem["category"];
  section: string;
  badge: string;
  title: string;
  detail: string;
  active?: boolean;
  decl?: DeclPath;
  action: SpotlightAction;
}): SpotlightItem {
  const tabClause = spec.action.tabClause ?? spec.action.tabClauses?.[0];
  const tabClauses = spec.action.tabClauses ?? (spec.action.tabClause ? [spec.action.tabClause] : undefined);
  return {
    id: spec.id,
    category: spec.category,
    section: spec.section,
    badge: spec.badge,
    title: spec.title,
    detail: spec.detail,
    subtitle: spec.detail,
    ...(spec.active !== undefined ? { active: spec.active } : {}),
    ...(spec.decl !== undefined ? { decl: spec.decl } : {}),
    events: spec.action.events,
    keepOpen: !spec.action.close,
    ...(tabClause !== undefined ? { tabClause } : {}),
    ...(tabClauses !== undefined ? { tabClauses } : {}),
    ...(spec.action.shiftEvents !== undefined ? { shiftEvents: spec.action.shiftEvents } : {}),
    ...(spec.action.altEvents !== undefined ? { altEvents: spec.action.altEvents } : {}),
    ...(spec.action.ctrlEvents !== undefined ? { ctrlEvents: spec.action.ctrlEvents } : {}),
    action: spec.action,
  };
}

function wordStartsWith(haystack: string, tok: string): boolean {
  for (const w of haystack.split(/[^a-z0-9_.%?+-]+/)) {
    if (w && w.startsWith(tok)) return true;
  }
  return false;
}

function scoreMatch(query: string, title: string, detail: string, keywords = ""): number {
  const q = normText(query).trim();
  if (!q) return 1;
  const t = normText(title);
  const d = normText(detail);
  const k = normText(keywords);

  const tokens = q.split(/\s+/).filter(Boolean);
  let total = 0;
  for (const tok of tokens) {
    if (t === tok) total += 120;
    else if (t.startsWith(tok)) total += 85;
    else if (wordStartsWith(t, tok)) total += 65;
    else if (wordStartsWith(k, tok)) total += 45;
    else if (t.includes(tok)) total += 35;
    else if (k.includes(tok)) total += 25;
    else if (d.includes(tok)) total += 20;
    else return 0;
  }
  return total;
}

function drillDownItem(id: string, badge: string, title: string, detail: string, mode: SpotlightMode, section = "Commands"): SpotlightItem {
  return makeItem({
    id,
    category: "command",
    section,
    badge,
    title,
    detail,
    action: { events: [{ type: "setSpotlightMode", mode }], close: false, nextMode: mode },
  });
}

function commandItem(
  id: string,
  badge: string,
  title: string,
  detail: string,
  events: readonly Event[],
  active = false,
  altEvents?: readonly Event[],
  section = "Commands",
): SpotlightItem {
  return makeItem({
    id,
    category: "command",
    section,
    badge,
    title,
    detail,
    active,
    action: { events, close: true, ...(altEvents ? { altEvents } : {}) },
  });
}

function activeResultsStage(s: State) {
  const at = s.phase.at;
  if (at?.kind !== "results" || !s.results) return null;
  const st = s.results.stages[at.stage];
  if (!st || st.time.length === 0) return null;
  return { stageIndex: at.stage, step: at.step, steps: st.time.length, stage: st };
}

const SEARCH_VERB_PREFIX_RE = /^(?:select|find|filter|hide|isolate|search)\b\s*/i;

function buildRootCommands(s: State): { item: SpotlightItem; keywords: string }[] {
  const out: { item: SpotlightItem; keywords: string }[] = [];
  const hasModel = s.artifacts.model?.status === "ready";
  const hasMesh = s.mesh !== null;
  const resStage = activeResultsStage(s);

  // 1. Open Search Window (or transfer current Spotlight query/clauses to the Search Window)
  const transferEvents: Event[] = [{ type: "openWindow", window: "search" }];
  const cleanedRootQ = s.spotlight.query.trim().replace(SEARCH_VERB_PREFIX_RE, "");
  if (cleanedRootQ || s.spotlight.clauses.length > 0) {
    transferEvents.push({ type: "clearSearch" });
    if (cleanedRootQ) {
      transferEvents.push({ type: "setSearchText", text: cleanedRootQ });
    }
    for (const c of s.spotlight.clauses) {
      transferEvents.push({
        type: "addSearchClause",
        clause: {
          join: c.join,
          negate: c.negate,
          field: c.field,
          op: c.op,
          ...(c.key !== undefined ? { key: c.key } : {}),
          value: c.value,
        },
      });
    }
  }
  out.push({
    item: commandItem(
      "cmd:open-search-window",
      "search",
      s.windows.open.includes("search") ? "Focus Search Window" : "Open Search Window…",
      s.spotlight.query.trim() || s.spotlight.clauses.length
        ? "Open the visual query builder with the current query and filter chips"
        : "Visual query builder for intricate AND / OR / NOT searches and batch actions",
      transferEvents,
      s.windows.open.includes("search"),
    ),
    keywords: "search find query filter builder window",
  });
  out.push({
    item: drillDownItem(
      "cmd:help",
      "help",
      "How Spotlight Works (Help & Examples)…",
      "Step-by-step guide: selecting multiple groups, Tab filter chips, Shift+Enter actions (hide/isolate), and set commands",
      "help",
    ),
    keywords: "help ? how guide tutorial examples syntax procedure instructions",
  });

  if (hasMesh) {
    out.push({
      item: commandItem(
        "cmd:frame-selection",
        "view",
        s.selection.decls.length ? `Frame Selection (${s.selection.decls.length})` : "Fit Whole Model (F)",
        "Move camera to fit the current selection or the entire model",
        [{ type: "frameSelection" }],
      ),
      keywords: "frame fit zoom center camera f",
    });

    if (s.selection.decls.length > 0) {
      out.push({
        item: commandItem(
          "cmd:clear-selection",
          "select",
          `Clear Selection (${s.selection.decls.length} selected)`,
          "Deselect all declarations and restore full model brightness",
          [{ type: "clearSelection" }],
        ),
        keywords: "clear deselect none selection",
      });
      out.push({
        item: commandItem(
          "cmd:isolate-selection",
          "visibility",
          `Isolate Current Selection (${s.selection.decls.length})`,
          "Show only the currently selected declarations and hide the rest",
          [{ type: "isolate", decls: [...s.selection.decls] }],
        ),
        keywords: "isolate selection only solo",
      });
      out.push({
        item: commandItem(
          "cmd:hide-selection",
          "visibility",
          `Hide Current Selection (${s.selection.decls.length})`,
          "Hide the currently selected declarations from the 3D viewport",
          [{ type: "setHidden", decls: [...s.selection.decls], hidden: true }],
        ),
        keywords: "hide off turn off selection",
      });
    }

    if (s.visibility.hidden.length > 0) {
      out.push({
        item: commandItem(
          "cmd:show-all",
          "visibility",
          `Show All Hidden (${s.visibility.hidden.length} hidden)`,
          "Restore visibility of all hidden physical groups and elements",
          [{ type: "showAll" }],
        ),
        keywords: "show all unhide reset visibility restore",
      });
    }

    out.push({
      item: commandItem(
        "cmd:toggle-edges",
        "display",
        s.visibility.edges ? "Turn Off Mesh Edges" : "Turn On Mesh Edges",
        `Outline edges are currently ${s.visibility.edges ? "ON" : "OFF"}`,
        [{ type: "setEdges", on: !s.visibility.edges }],
        s.visibility.edges,
        [{ type: "setEdges", on: !s.visibility.edges }],
      ),
      keywords: "set edges wireframe outline mesh lines toggle",
    });

    out.push({
      item: drillDownItem(
        "cmd:set-opacity",
        "display",
        `Set Surface Opacity… (currently ${Math.round(s.visibility.opacity * 100)}%)`,
        "Change mesh face transparency (presets or custom 0..1)",
        "opacity",
      ),
      keywords: "set opacity transparency alpha surface ghost",
    });

    out.push({
      item: drillDownItem(
        "cmd:set-colour-by",
        "display",
        `Set Colour By… (currently ${s.visibility.colourBy})`,
        "Colour mesh by physical group, structural role, or results contour",
        "colourBy",
      ),
      keywords: "set colour color by group role contour mode legend",
    });

    for (const by of ["group", "role", "contour"] as const) {
      out.push({
        item: commandItem(
          `cmd:colour-by-${by}`,
          "display",
          `Colour by ${by === "group" ? "Physical Group" : by === "role" ? "Structural Role" : "Results Contour"}`,
          s.visibility.colourBy === by ? "Currently active" : `Switch mesh colouring to ${by}`,
          [{ type: "setColourBy", by }],
          s.visibility.colourBy === by,
        ),
        keywords: `set colour color by ${by}`,
      });
    }
  }

  if (hasModel) {
    out.push({
      item: drillDownItem(
        "cmd:windows",
        "window",
        `Toggle Windows… (${s.windows.open.length} open)`,
        "Open or close Search, Stages, Analysis, Patterns, Recorders, or Materials",
        "windows",
      ),
      keywords: "window windows open close stages analysis patterns recorders materials search",
    });

    for (const w of WINDOW_KINDS) {
      const open = s.windows.open.includes(w);
      out.push({
        item: commandItem(
          `cmd:window-${w}`,
          "window",
          `${open ? "Close" : "Open"} ${w[0]!.toUpperCase() + w.slice(1)} Window`,
          open ? "Window is currently open" : `Open the ${w} window card`,
          [{ type: open ? "closeWindow" : "openWindow", window: w }],
          open,
          [{ type: open ? "closeWindow" : "openWindow", window: w }],
        ),
        keywords: `window open close ${w}`,
      });
    }
  }

  const seqStageCount = s.sequence.status === "ready" ? s.sequence.stages.rows.length : 0;
  if (s.phase.axis.length > 1 || seqStageCount > 0) {
    out.push({
      item: drillDownItem(
        "cmd:phase",
        "phase",
        "Switch Phase or Construction Stage…",
        "Switch between geometry, mesh, sequence stages, and results stages",
        "phase",
      ),
      keywords: "set phase stage geometry mesh results sequence",
    });
  }

  if (hasMesh || s.phase.axis.some((k) => k.kind === "results")) {
    out.push({
      item: drillDownItem(
        "cmd:contour-units",
        "units",
        `Set Units… (${s.contour.targetUnits === "as_stored" ? "as stored" : s.contour.targetUnits})`,
        "Convert load and contour values between consistent baseUnits systems (kN-m, tf-m, N-mm, kip-in…)",
        "contourUnits",
      ),
      keywords: "set contour loads units baseunits kn tf kgf mm kip mpa kpa",
    });
  }

  if (hasMesh) {
    out.push({
      item: commandItem(
        "cmd:toggle-loads",
        "loads",
        s.loads.on ? "Turn Off 3-D Structural Loads" : "Turn On 3-D Structural Loads",
        `3-D forces, moments, distributed loads, and prescribed SPs are currently ${s.loads.on ? `ON (scale ${s.loads.scale}×)` : "OFF"}`,
        [{ type: "setLoads", on: !s.loads.on }],
        s.loads.on,
        [{ type: "setLoads", on: !s.loads.on }],
      ),
      keywords: "set loads forces moments pressure distributed beam sp prescribed toggle",
    });
  }

  if (s.phase.axis.some((k) => k.kind === "results")) {
    out.push({
      item: drillDownItem(
        "cmd:contour-field",
        "contour",
        `Set Contour Field… (${s.contour.field ? `${s.contour.field} · ${s.contour.component}` : "none"})`,
        "Pick a nodal or element (Gauss) field and component to colour by",
        "contourField",
      ),
      keywords: "set contour field component nodal element gauss displacement stress force",
    });
    out.push({
      item: drillDownItem(
        "cmd:contour-gauss",
        "contour",
        `Set Element Contour Topology… (currently ${s.contour.gauss})`,
        "Pick per-cell, cell averaged at nodes, extrapolated at nodes, or discrete",
        "contourGauss",
      ),
      keywords: "set contour gauss topology mode cell node averaged discrete extrapolation",
    });
    out.push({
      item: drillDownItem(
        "cmd:contour-range",
        "contour",
        `Set Contour Range… (currently ${s.contour.range})`,
        "Switch between whole-history, current step, or custom fixed min/max range",
        "contourRange",
      ),
      keywords: "set contour range clim history step fixed min max",
    });
    out.push({
      item: drillDownItem(
        "cmd:contour-cmap",
        "contour",
        `Set Contour Colormap… (currently ${s.contour.cmap})`,
        `Switch between ${CMAP_NAMES.length} colormaps (viridis, coolwarm, turbo, inferno, magma, plasma, jet…)`,
        "contourCmap",
      ),
      keywords: `set contour cmap colormap palette ${CMAP_NAMES.join(" ")}`,
    });
    out.push({
      item: commandItem(
        "cmd:toggle-deform",
        "deform",
        s.deform.on ? "Turn Off Deformed Shape" : "Turn On Deformed Shape",
        `Deformed shape is currently ${s.deform.on ? `ON (scale ${s.deform.scale}×)` : "OFF"}`,
        [{ type: "setDeform", on: !s.deform.on }],
        s.deform.on,
        [{ type: "setDeform", on: !s.deform.on }],
      ),
      keywords: "set deform deformed shape displacement toggle",
    });
    out.push({
      item: drillDownItem(
        "cmd:deform-scale",
        "deform",
        `Set Deform Scale… (currently ${s.deform.scale}×)`,
        "Set the displacement magnification factor",
        "deformScale",
      ),
      keywords: "set deform scale factor magnification",
    });
    out.push({
      item: commandItem(
        "cmd:toggle-ghost",
        "deform",
        s.deform.ghost ? "Hide Undeformed Reference Ghost" : "Show Undeformed Reference Ghost",
        `Reference ghost mesh is currently ${s.deform.ghost ? "ON" : "OFF"}`,
        [{ type: "setReferenceGhost", on: !s.deform.ghost }],
        s.deform.ghost,
        [{ type: "setReferenceGhost", on: !s.deform.ghost }],
      ),
      keywords: "set deform reference ghost undeformed toggle",
    });
    out.push({
      item: commandItem(
        "cmd:toggle-line-diagram",
        "diagram",
        s.lineDiagram.on ? "Turn Off 3-D Line Diagrams" : "Turn On 3-D Line Diagrams",
        `1-D beam/frame diagrams are currently ${s.lineDiagram.on ? `ON (${s.lineDiagram.component ?? "auto"})` : "OFF"}`,
        [{ type: "setLineDiagram", on: !s.lineDiagram.on }],
        s.lineDiagram.on,
        [{ type: "setLineDiagram", on: !s.lineDiagram.on }],
      ),
      keywords: "set line diagram beam moment shear axial ribbon mvy mz vy n toggle",
    });
  }

  if (resStage) {
    out.push({
      item: drillDownItem(
        "cmd:result-step",
        "results",
        `Go to Result Step… (step ${resStage.step + 1}/${resStage.steps})`,
        "Jump to a specific time step or mode shape",
        "step",
      ),
      keywords: "set step time result scrub jump",
    });
  }

  if (hasMesh) {
    out.push({
      item: drillDownItem(
        "cmd:planes",
        "clip",
        `Section Planes… (${s.clip.planes.length} planes)`,
        "Add X / Y / Z section planes, flip, toggle cuts, gizmos, or cut-face caps",
        "planes",
      ),
      keywords: "set section plane planes clip cut gizmo cap x y z",
    });
    for (const [axis, normal] of [["X", [1, 0, 0]], ["Y", [0, 1, 0]], ["Z", [0, 0, 1]]] as const) {
      out.push({
        item: commandItem(
          `cmd:add-plane-${axis.toLowerCase()}`,
          "clip",
          `Add Section Plane Normal to ${axis}`,
          `Cut the model with a plane normal to ${axis} through the centre`,
          [{ type: "addClipPlane", normal }],
        ),
        keywords: `add section plane clip cut ${axis.toLowerCase()}`,
      });
    }
  }

  if (s.inspector.pinned.length > 0) {
    out.push({
      item: commandItem(
        "cmd:unpin-all",
        "inspector",
        `Unpin All Inspector Cards (${s.inspector.pinned.length})`,
        "Remove all pinned declarations from the inspector",
        s.inspector.pinned.map((decl) => ({ type: "inspectorUnpin" as const, decl })),
      ),
      keywords: "unpin clear inspector pin",
    });
  }

  return out;
}

function declSpotlightItem(
  s: State,
  d: { path: DeclPath; kind: DeclKind; name: string; type: string; detail: string; elementCount: number },
  mods: { join: "and" | "or"; negate: boolean } = { join: "and", negate: false },
  sectionOverride?: string,
): SpotlightItem {
  const isGroup = d.kind === "physical_group" || d.kind === "label";
  const hidden = s.visibility.hidden.includes(d.path);
  const pinned = s.inspector.pinned.includes(d.path);
  const selected = s.selection.decls.includes(d.path);
  const hasSource = Boolean(s.decls[d.path]?.provenance?.recorded);

  const tabClause: Omit<SearchClause, "id"> | undefined = isGroup
    ? { join: mods.join, negate: mods.negate, field: "group", op: ":", value: d.name }
    : d.kind !== "element"
      ? { join: mods.join, negate: mods.negate, field: "uses", op: ":", value: d.name }
      : undefined;

  return makeItem({
    id: `decl:${d.path}`,
    category: isGroup ? "group" : d.kind === "element" ? "element" : "declaration",
    section: sectionOverride ?? (isGroup ? "Groups & Labels" : d.kind === "element" ? "Elements" : "Materials & Sections"),
    badge: isGroup ? (hidden ? `${d.kind} · hidden` : d.kind) : d.kind,
    title: `${d.name}${d.type && d.type !== d.kind ? ` · ${d.type}` : ""}`,
    detail: d.detail,
    active: selected || pinned,
    decl: d.path,
    action: {
      events: [
        { type: "select", pick: { decl: d.path, at: null, nodes: null } },
        { type: "frameSelection" },
      ],
      close: true,
      ...(tabClause ? { tabClause } : {}),
      shiftEvents: [
        { type: "selectAdd", pick: { decl: d.path, at: null, nodes: null } },
        { type: "frameSelection" },
      ],
      altEvents: isGroup
        ? [{ type: "setHidden", decls: [d.path], hidden: !hidden }]
        : [pinned ? { type: "inspectorUnpin", decl: d.path } : { type: "inspectorPin", decl: d.path }],
      ...(hasSource ? { ctrlEvents: [{ type: "requestSource", decl: d.path }] } : {}),
    },
  });
}

function buildDrillDownItems(s: State, mode: Exclude<SpotlightMode, "root">, q: string): SpotlightItem[] {
  const items: SpotlightItem[] = [];
  const low = q.trim().toLowerCase();
  const section = MODE_META[mode].label ?? "Options";
  const filter = (it: SpotlightItem) => !low || scoreMatch(low, it.title, it.detail, it.badge) > 0;

  switch (mode) {
    case "help": {
      const sampleGroups = Object.values(s.decls)
        .filter((d) => d.kind === "physical_group")
        .map((d) => d.name)
        .slice(0, 2);
      const ex1 = sampleGroups[0] ?? "BeamX_V50x70";
      const ex2 = sampleGroups[1] ?? "BeamY_V50x70";
      const helpEntries: { id: string; badge: string; title: string; detail: string; events: readonly Event[]; close: boolean }[] = [
        {
          id: "help:multi-select",
          badge: "step 1 · select",
          title: `Combine Groups with 'and' / 'or' — e.g. "select ${ex1} and ${ex2}"`,
          detail: "Type one or more group/section names separated by 'and', 'or', or ',' → press Enter to select & frame all of them",
          events: [
            { type: "setSpotlightMode", mode: "root" },
            { type: "setSpotlightQuery", query: `select ${ex1} and ${ex2}` },
          ],
          close: false,
        },
        {
          id: "help:partial-match",
          badge: "step 2 · wildcards",
          title: `Partial & Pattern Names — e.g. "beam_V50x70" matches BeamO/X/Y_V50x70`,
          detail: "Underscore patterns match across prefixes: typing 'beam_V50x70' matches every Beam*_V50x70 group at once",
          events: [
            { type: "setSpotlightMode", mode: "root" },
            { type: "setSpotlightQuery", query: ex1 },
          ],
          close: false,
        },
        {
          id: "help:tab-chips",
          badge: "step 3 · Tab key",
          title: `Pin Filter Chips with Tab (⇥) — Build Multi-Chip Queries Step by Step`,
          detail: `Type "${ex1}" → press Tab to pin it as a chip → type "${ex2}" → press Tab → press Enter to select or Shift+Enter for actions`,
          events: [
            { type: "setSpotlightMode", mode: "root" },
            { type: "addSpotlightClause", clause: { join: "and", negate: false, field: "group", op: ":", value: ex1 } },
          ],
          close: false,
        },
        {
          id: "help:shift-enter",
          badge: "step 4 · Shift+Enter",
          title: `Hide, Isolate, Show, or Add to Selection (Shift+Enter)`,
          detail: "Once your query matches elements, press Shift+Enter to open batch actions → type 'hide' or 'isolate' → press Enter",
          events: [
            { type: "setSpotlightMode", mode: "root" },
            { type: "setSpotlightQuery", query: ex1 },
          ],
          close: false,
        },
        {
          id: "help:set-props",
          badge: "step 5 · viewer",
          title: `Change Viewer Settings — Type "set", "opacity", "colour", "edges", "planes"`,
          detail: "Press Enter to try typing 'set' and browse all viewer display, contour, deform, and section-plane controls",
          events: [
            { type: "setSpotlightMode", mode: "root" },
            { type: "setSpotlightQuery", query: "set" },
          ],
          close: false,
        },
        {
          id: "help:search-win",
          badge: "step 6 · window",
          title: `Open the Full Visual Search Window — Type "search" + Enter`,
          detail: "Opens the dedicated Search Window with dropdown facets (group, type, uses, parameter comparisons) and batch buttons",
          events: [{ type: "openWindow", window: "search" }],
          close: true,
        },
      ];
      for (const h of helpEntries) {
        const it = makeItem({
          id: h.id,
          category: "command",
          section: "Spotlight Guide (Press Enter on any row to try it)",
          badge: h.badge,
          title: h.title,
          detail: h.detail,
          action: { events: h.events, close: h.close },
        });
        if (filter(it)) items.push(it);
      }
      return items;
    }

    case "opacity": {
      const num = Number(q.trim().replace(/%$/, ""));
      if (q.trim() !== "" && Number.isFinite(num)) {
        const val = q.trim().endsWith("%") || num > 1 ? Math.min(1, Math.max(0, num / 100)) : Math.min(1, Math.max(0, num));
        items.push(
          commandItem(
            `val:opacity-custom`,
            "custom",
            `Set Opacity to ${Math.round(val * 100)}% (${val.toFixed(2)})`,
            "Apply custom surface opacity",
            [{ type: "setOpacity", value: val }],
            false,
            [{ type: "setOpacity", value: val }],
            section,
          ),
        );
      }
      for (const [pct, desc] of [
        [100, "Opaque (default)"],
        [75, "Light transparency"],
        [50, "Half transparent"],
        [35, "See-through solids (rebar visibility)"],
        [20, "Faint surfaces"],
        [10, "Ghost-like surfaces"],
      ] as const) {
        const val = pct / 100;
        const active = Math.abs(s.visibility.opacity - val) < 1e-3;
        const it = commandItem(
          `val:opacity-${pct}`,
          "opacity",
          `${pct}% (${val.toFixed(2)})`,
          desc,
          [{ type: "setOpacity", value: val }],
          active,
          [{ type: "setOpacity", value: val }],
          section,
        );
        if (filter(it)) items.push(it);
      }
      return items;
    }

    case "colourBy": {
      const specs: readonly { by: ColourBy; title: string; detail: string }[] = [
        { by: "group", title: "Physical Groups", detail: "Colour each element by its most specific element-side physical group" },
        { by: "role", title: "Structural Roles", detail: "Colour elements by their group's @role attribute (column, wall, beam, concrete, void)" },
        { by: "contour", title: "Results Contour", detail: "Colour vertices or cells by a nodal or element (Gauss) results field" },
      ];
      for (const sp of specs) {
        const it = commandItem(
          `val:colour-${sp.by}`,
          "colour",
          sp.title,
          sp.detail,
          [{ type: "setColourBy", by: sp.by }],
          s.visibility.colourBy === sp.by,
          [{ type: "setColourBy", by: sp.by }],
          section,
        );
        if (filter(it)) items.push(it);
      }
      return items;
    }

    case "phase": {
      for (const k of s.phase.axis) {
        const label =
          k.kind === "geometry"
            ? "Geometry (/geometry sibling)"
            : k.kind === "mesh"
              ? "Mesh (undeformed model)"
              : k.kind === "stage"
                ? `Stage ${k.index + 1}`
                : `Results Stage ${k.stage + 1}: ${s.results?.stages[k.stage]?.name || s.results?.stages[k.stage]?.id || k.stage}`;
        const active =
          s.phase.at?.kind === k.kind &&
          (k.kind !== "results" || (s.phase.at.kind === "results" && s.phase.at.stage === k.stage));
        const it = commandItem(
          `val:phase-${JSON.stringify(k)}`,
          "phase",
          label,
          active ? "Currently shown on viewport" : "Switch viewport phase",
          [{ type: "setPhase", at: k }],
          active,
          [{ type: "setPhase", at: k }],
          section,
        );
        if (filter(it)) items.push(it);
      }
      if (s.sequence.status === "ready") {
        const nameIdx = s.sequence.stages.columns.indexOf("name");
        s.sequence.stages.rows.forEach((row, idx) => {
          const name = nameIdx >= 0 ? String(row[nameIdx] ?? "") : "";
          const active = s.stage === idx;
          const it = commandItem(
            `val:seq-stage-${idx}`,
            "sequence",
            `Select Sequence Stage ${idx}${name ? ` · ${name}` : ""}`,
            active ? "Currently selected in Stages window" : "Select this stage row in the sequence",
            [{ type: "selectStage", index: idx }],
            active,
            [{ type: "selectStage", index: idx }],
            section,
          );
          if (filter(it)) items.push(it);
        });
      }
      return items;
    }

    case "step": {
      const t = activeResultsStage(s);
      if (!t) return [];
      const num = Number(q.trim());
      if (q.trim() !== "" && Number.isInteger(num) && num >= 1 && num <= t.steps) {
        items.push(
          commandItem(
            `val:step-custom-${num}`,
            "step",
            `Jump to Step ${num} / ${t.steps}`,
            `Set result step index to ${num - 1}`,
            [{ type: "setResultStep", step: num - 1 }],
            t.step === num - 1,
            [{ type: "setResultStep", step: num - 1 }],
            section,
          ),
        );
      }
      const times = t.stage.time;
      for (let i = 0; i < t.steps; i++) {
        const timeVal = times[i];
        const title = `Step ${i + 1} / ${t.steps}`;
        const detail = timeVal !== undefined ? (t.stage.kind === "mode" ? `mode shape · f/λ = ${timeVal}` : `t = ${timeVal}`) : "";
        const it = commandItem(
          `val:step-${i}`,
          "step",
          title,
          detail,
          [{ type: "setResultStep", step: i }],
          i === t.step,
          [{ type: "setResultStep", step: i }],
          section,
        );
        if (filter(it)) items.push(it);
        if (items.length >= SPOTLIGHT_ITEM_CAP) break;
      }
      return items;
    }

    case "contourField": {
      const at = s.phase.at;
      const st = at?.kind === "results" ? s.results?.stages[at.stage] : undefined;
      if (!st) return [];
      for (const f of fieldsOf(st)) {
        for (const c of f.components) {
          const active = s.contour.source === "nodal" && s.contour.field === f.name && s.contour.component === c.key;
          const it = commandItem(
            `val:cfield-nodal-${f.name}-${c.key}`,
            "nodal",
            `${f.name} · ${c.label}`,
            `Nodal contour (${c.datasets.join(", ")})`,
            [
              { type: "setColourBy", by: "contour" },
              { type: "setContourField", field: f.name, component: c.key, source: "nodal" },
            ],
            active,
            [
              { type: "setColourBy", by: "contour" },
              { type: "setContourField", field: f.name, component: c.key, source: "nodal" },
            ],
            section,
          );
          if (filter(it)) items.push(it);
        }
      }
      for (const f of gaussFields(st.gauss)) {
        for (const c of f.components) {
          const active = s.contour.source === "element" && s.contour.field === f.name && s.contour.component === c.key;
          const it = commandItem(
            `val:cfield-elem-${f.name}-${c.key}`,
            "element",
            `${f.name} · ${c.label}`,
            `Element Gauss contour (${c.datasets.join(", ")})`,
            [
              { type: "setColourBy", by: "contour" },
              { type: "setContourField", field: f.name, component: c.key, source: "element" },
            ],
            active,
            [
              { type: "setColourBy", by: "contour" },
              { type: "setContourField", field: f.name, component: c.key, source: "element" },
            ],
            section,
          );
          if (filter(it)) items.push(it);
        }
      }
      return items.slice(0, SPOTLIGHT_ITEM_CAP);
    }

    case "contourGauss": {
      const modes: readonly { mode: GaussMode; title: string; detail: string }[] = [
        { mode: "cell", title: "Per Cell (flat)", detail: "Flat per cell: one Gauss point or element mean" },
        { mode: "cell_averaged", title: "Cell, Averaged at Nodes", detail: "Cell values averaged to shared nodes across elements" },
        { mode: "node", title: "Extrapolated at Nodes (averaged)", detail: "Gauss points extrapolated to corners and averaged at shared nodes" },
        { mode: "node_discrete", title: "Extrapolated at Nodes, Per Element (discrete)", detail: "Extrapolated to element corners without averaging across elements" },
      ];
      for (const m of modes) {
        const it = commandItem(
          `val:gauss-${m.mode}`,
          "topology",
          m.title,
          m.detail,
          [{ type: "setContourGauss", mode: m.mode }],
          s.contour.gauss === m.mode,
          [{ type: "setContourGauss", mode: m.mode }],
          section,
        );
        if (filter(it)) items.push(it);
      }
      return items;
    }

    case "contourRange": {
      const parts = q.trim().split(/[\s,]+/).map(Number);
      if (parts.length === 2 && Number.isFinite(parts[0]) && Number.isFinite(parts[1]) && parts[0]! < parts[1]!) {
        items.push(
          commandItem(
            "val:range-fixed-custom",
            "fixed",
            `Set Fixed Range [${parts[0]}, ${parts[1]}]`,
            "Apply custom min and max colour limits",
            [{ type: "setContourRange", range: "fixed", fixed: { min: parts[0]!, max: parts[1]! } }],
            false,
            [{ type: "setContourRange", range: "fixed", fixed: { min: parts[0]!, max: parts[1]! } }],
            section,
          ),
        );
      }
      for (const r of RANGE_MODES) {
        if (r.range === "fixed" && !s.contour.fixed) continue;
        const it = commandItem(
          `val:range-${r.range}`,
          "range",
          `Range: ${r.label}`,
          r.range === "history"
            ? "Scale colours across all steps of the stage"
            : r.range === "step"
              ? "Scale colours to the min/max of the current step"
              : `Fixed range [${s.contour.fixed?.min}, ${s.contour.fixed?.max}]`,
          [{ type: "setContourRange", range: r.range }],
          s.contour.range === r.range,
          [{ type: "setContourRange", range: r.range }],
          section,
        );
        if (filter(it)) items.push(it);
      }
      return items;
    }

    case "contourCmap": {
      for (const cmap of CMAP_NAMES) {
        const it = commandItem(
          `val:cmap-${cmap}`,
          "cmap",
          `Colormap: ${cmap}`,
          `Use the ${cmap} colour ramp`,
          [{ type: "setContourCmap", cmap }],
          s.contour.cmap === cmap,
          [{ type: "setContourCmap", cmap }],
          section,
        );
        if (filter(it)) items.push(it);
      }
      return items;
    }

    case "contourUnits": {
      for (const opt of UNIT_DISPLAY_OPTIONS) {
        const active = s.contour.targetUnits === opt.value;
        const it = commandItem(
          `val:unit-target-${opt.value}`,
          "display units",
          `Display in ${opt.label}`,
          active ? "Currently active display unit system" : `Convert contour values to ${opt.label}`,
          [{ type: "setContourUnits", target: opt.value }],
          active,
          [{ type: "setContourUnits", target: opt.value }],
          section,
        );
        if (filter(it)) items.push(it);
      }
      for (const src of UNIT_SYSTEM_IDS) {
        const spec = UNIT_SYSTEMS[src];
        const active = s.contour.sourceUnits === src;
        const it = commandItem(
          `val:unit-source-${src}`,
          "native units",
          `Native File Units: ${spec.label}`,
          active ? "Currently assumed native unit system" : `Assume file values are stored in ${spec.label}`,
          [{ type: "setContourUnits", source: src }],
          active,
          [{ type: "setContourUnits", source: src }],
          section,
        );
        if (filter(it)) items.push(it);
      }
      return items;
    }

    case "deformScale": {
      const num = Number(q.trim().replace(/x$/i, ""));
      if (q.trim() !== "" && Number.isFinite(num) && num > 0) {
        items.push(
          commandItem(
            "val:deform-custom",
            "custom",
            `Set Deform Scale to ${num}×`,
            "Turn on deformed shape with custom magnification",
            [
              { type: "setDeform", on: true },
              { type: "setDeformScale", scale: num },
            ],
            false,
            [
              { type: "setDeform", on: true },
              { type: "setDeformScale", scale: num },
            ],
            section,
          ),
        );
      }
      for (const sc of [1, 2, 5, 10, 20, 50, 100, 200, 500] as const) {
        const active = s.deform.on && Math.abs(s.deform.scale - sc) < 1e-6;
        const it = commandItem(
          `val:deform-${sc}`,
          "scale",
          `${sc}× (${sc === 1 ? "true scale" : `magnified ${sc}×`})`,
          "Enable deformed shape at this scale factor",
          [
            { type: "setDeform", on: true },
            { type: "setDeformScale", scale: sc },
          ],
          active,
          [
            { type: "setDeform", on: true },
            { type: "setDeformScale", scale: sc },
          ],
          section,
        );
        if (filter(it)) items.push(it);
      }
      return items;
    }

    case "windows": {
      for (const w of WINDOW_KINDS) {
        const open = s.windows.open.includes(w);
        const it = commandItem(
          `val:win-${w}`,
          open ? "open" : "closed",
          `${open ? "Close" : "Open"} ${w[0]!.toUpperCase() + w.slice(1)} Window`,
          `Toggle the ${w} window`,
          [{ type: open ? "closeWindow" : "openWindow", window: w }],
          open,
          [{ type: open ? "closeWindow" : "openWindow", window: w }],
          section,
        );
        if (filter(it)) items.push(it);
      }
      return items;
    }

    case "planes": {
      for (const [axis, normal] of [["X", [1, 0, 0]], ["Y", [0, 1, 0]], ["Z", [0, 0, 1]]] as const) {
        const it = commandItem(
          `val:plane-add-${axis}`,
          "add plane",
          `Add Plane Normal to ${axis}`,
          `Add a section plane normal to ${axis}`,
          [{ type: "addClipPlane", normal }],
          false,
          [{ type: "addClipPlane", normal }],
          section,
        );
        if (filter(it)) items.push(it);
      }
      const toggleCuts = commandItem(
        "val:plane-apply-cuts",
        "clip",
        s.clip.applyCuts ? "Disable Section Cuts (Keep Planes)" : "Enable Section Cuts",
        `Apply cuts is currently ${s.clip.applyCuts ? "ON" : "OFF"}`,
        [{ type: "setApplyCuts", on: !s.clip.applyCuts }],
        s.clip.applyCuts,
        [{ type: "setApplyCuts", on: !s.clip.applyCuts }],
        section,
      );
      if (filter(toggleCuts)) items.push(toggleCuts);

      const toggleGizmos = commandItem(
        "val:plane-gizmos",
        "clip",
        s.clip.showGizmos ? "Hide Section Plane Gizmos" : "Show Section Plane Gizmos",
        `Plane gizmos are currently ${s.clip.showGizmos ? "ON" : "OFF"}`,
        [{ type: "setShowGizmos", on: !s.clip.showGizmos }],
        s.clip.showGizmos,
        [{ type: "setShowGizmos", on: !s.clip.showGizmos }],
        section,
      );
      if (filter(toggleGizmos)) items.push(toggleGizmos);

      const toggleCap = commandItem(
        "val:plane-cut-face",
        "clip",
        s.clip.cutFace ? "Disable Contour on Cut Face" : "Enable Contour on Cut Face",
        `Solid cut-face cap is currently ${s.clip.cutFace ? "ON" : "OFF"}`,
        [{ type: "setCutFace", on: !s.clip.cutFace }],
        s.clip.cutFace,
        [{ type: "setCutFace", on: !s.clip.cutFace }],
        section,
      );
      if (filter(toggleCap)) items.push(toggleCap);

      for (const p of s.clip.planes) {
        const num = Number(q.trim());
        if (q.trim() !== "" && Number.isFinite(num)) {
          items.push(
            commandItem(
              `val:plane-offset-${p.id}`,
              p.id,
              `Set ${p.id} Offset to ${num}`,
              `Current offset: ${p.offset}`,
              [{ type: "setClipOffset", id: p.id, offset: num }],
              false,
              [{ type: "setClipOffset", id: p.id, offset: num }],
              section,
            ),
          );
        }
        const flip = commandItem(
          `val:plane-flip-${p.id}`,
          p.id,
          `Flip ${p.id}`,
          `Currently ${p.flipped ? "flipped" : "normal"}`,
          [{ type: "setClipFlipped", id: p.id, flipped: !p.flipped }],
          p.flipped,
          [{ type: "setClipFlipped", id: p.id, flipped: !p.flipped }],
          section,
        );
        if (filter(flip)) items.push(flip);
        const act = commandItem(
          `val:plane-act-${p.id}`,
          p.id,
          `${p.active ? "Deactivate" : "Activate"} ${p.id}`,
          `Cutting is ${p.active ? "ON" : "OFF"}`,
          [{ type: "setClipActive", id: p.id, active: !p.active }],
          p.active,
          [{ type: "setClipActive", id: p.id, active: !p.active }],
          section,
        );
        if (filter(act)) items.push(act);
        const giz = commandItem(
          `val:plane-giz-${p.id}`,
          p.id,
          `${p.gizmo ? "Hide" : "Show"} ${p.id} Gizmo`,
          `Gizmo eye is ${p.gizmo ? "ON" : "OFF"}`,
          [{ type: "setClipGizmo", id: p.id, visible: !p.gizmo }],
          p.gizmo,
          [{ type: "setClipGizmo", id: p.id, visible: !p.gizmo }],
          section,
        );
        if (filter(giz)) items.push(giz);
        const del = commandItem(
          `val:plane-del-${p.id}`,
          p.id,
          `Remove ${p.id}`,
          "Delete this section plane",
          [{ type: "removeClipPlane", id: p.id }],
          false,
          undefined,
          section,
        );
        if (filter(del)) items.push(del);
      }
      return items;
    }

    case "selectionActions": {
      const terms = activeSpotlightTerms(s);
      const summary = runSearchTerms(s, terms, 10);
      const decls = summary.elementDecls.length > 0 ? summary.elementDecls : summary.rows.map((r) => r.path);
      if (decls.length === 0) return [];
      const countLabel = `${decls.length.toLocaleString()} matched ${summary.elementDecls.length ? "elements" : "declarations"}`;
      const actions: { item: SpotlightItem; keywords: string }[] = [
        {
          item: commandItem(
            "act:select",
            "select",
            `Select & Frame (${countLabel})`,
            "Replace selection with all matched elements and frame them in 3D",
            [{ type: "selectDecls", decls }, { type: "frameSelection" }],
            false,
            undefined,
            section,
          ),
          keywords: "select frame focus",
        },
        {
          item: commandItem(
            "act:select-add",
            "select",
            `Add to Selection (${countLabel})`,
            "Add all matched elements to the current selection and frame",
            [{ type: "selectDecls", decls, add: true }, { type: "frameSelection" }],
            false,
            undefined,
            section,
          ),
          keywords: "add append selection",
        },
        {
          item: commandItem(
            "act:hide",
            "visibility",
            `Hide / Turn Off (${countLabel})`,
            "Hide all matched elements from the 3D viewport",
            [{ type: "setHidden", decls, hidden: true }],
            false,
            [{ type: "setHidden", decls, hidden: true }],
            section,
          ),
          keywords: "hide off turn off",
        },
        {
          item: commandItem(
            "act:show",
            "visibility",
            `Show / Unhide (${countLabel})`,
            "Unhide all matched elements in the 3D viewport",
            [{ type: "setHidden", decls, hidden: false }],
            false,
            [{ type: "setHidden", decls, hidden: false }],
            section,
          ),
          keywords: "show unhide on",
        },
        {
          item: commandItem(
            "act:isolate",
            "visibility",
            `Isolate Only These (${countLabel})`,
            "Show only the matched elements and hide the rest of the model",
            [{ type: "isolate", decls }],
            false,
            [{ type: "isolate", decls }],
            section,
          ),
          keywords: "isolate solo only",
        },
      ];
      if (s.spotlight.clauses.length > 0 && low) {
        const scored = actions
          .map((a) => ({ item: a.item, score: scoreMatch(low, a.item.title, a.item.detail, a.keywords) }))
          .filter((x) => x.score >= 25)
          .sort((a, b) => b.score - a.score);
        if (scored.length > 0) return scored.map((x) => x.item);
      }
      return actions.map((a) => a.item);
    }
  }
}

function queryTermsOfSpotlight(s: State): ParsedTerm[] {
  const cleaned = s.spotlight.query
    .trim()
    .replace(/^(?:select|find|filter|hide|isolate)\s+(?!(?:stage|selection|all)\b)/i, "");
  return parseSearchText(cleaned);
}

function activeSpotlightTerms(s: State): ParsedTerm[] {
  const fromClauses: ParsedTerm[] = s.spotlight.clauses.map((c) => ({
    join: c.join,
    negate: c.negate,
    field: c.field,
    op: c.op,
    ...(c.key !== undefined ? { key: c.key } : {}),
    value: c.value,
  }));
  if (s.spotlight.mode === "selectionActions" && fromClauses.length > 0) {
    return fromClauses;
  }
  return [...fromClauses, ...queryTermsOfSpotlight(s)];
}

function pendingClauseMods(raw: string, queryTerms: readonly ParsedTerm[]): { join: "and" | "or"; negate: boolean } {
  if (queryTerms.length > 0) {
    const last = queryTerms[queryTerms.length - 1]!;
    return { join: last.join, negate: last.negate };
  }
  let join: "and" | "or" = "and";
  let negate = false;
  for (const tok of raw.trim().split(/\s+/).filter(Boolean)) {
    const low = normText(tok);
    if (low === "or" || low === "|" || low === "||") join = "or";
    else if (low === "and" || low === "&" || low === "&&") join = "and";
    else if (low === "not" || low === "!") negate = !negate;
  }
  return { join, negate };
}

function isDuplicateLabelRow(s: State, r: { kind: DeclKind; name: string }): boolean {
  return r.kind === "label" && s.decls[`mesh/physical_group/${r.name}`] !== undefined;
}

/**
 * Pure selector returning what Spotlight (`Ctrl+K`) renders for `s`.
 */
export function spotlightViewOf(s: State): SpotlightView {
  const { open, mode, query, clauses } = s.spotlight;
  const meta = MODE_META[mode];
  const clauseViews: SpotlightClauseView[] = clauses.map((c, idx) => ({
    ...c,
    label: `${idx > 0 ? `${c.join.toUpperCase()} ` : ""}${clauseLabel(c)}`,
  }));
  const hint =
    mode === "root"
      ? "Enter: run / select+frame · Shift+Enter: actions (hide/isolate/add) · Tab: pin filter chip · Alt+Enter: toggle hide/pin · Ctrl+Enter: source"
      : "Enter: apply · Alt+Enter: apply & keep open · Backspace (empty) / Esc: back";

  if (mode !== "root") {
    return {
      open,
      mode,
      modeLabel: meta.label,
      modeTitle: meta.label ?? mode,
      placeholder: meta.placeholder,
      query,
      clauses: clauseViews,
      summary: null,
      items: buildDrillDownItems(s, mode, query),
      hint,
    };
  }

  const q = query.trim();
  const rootCommands = buildRootCommands(s);

  // When no query and no clauses are typed yet: show top commands + physical groups & materials.
  if (!q && clauses.length === 0) {
    const items: SpotlightItem[] = rootCommands.slice(0, 12).map((c) => c.item);
    if (s.artifacts.model?.status === "ready") {
      const summary = runSearchTerms(s, [], 40);
      for (const r of summary.rows) {
        if (r.kind === "element" || isDuplicateLabelRow(s, r)) continue;
        items.push(declSpotlightItem(s, r));
        if (items.length >= SPOTLIGHT_ITEM_CAP) break;
      }
    }
    return {
      open,
      mode,
      modeLabel: meta.label,
      modeTitle: "Spotlight",
      placeholder: meta.placeholder,
      query,
      clauses: clauseViews,
      summary: null,
      items,
      hint,
    };
  }

  const items: SpotlightItem[] = [];
  let summaryText: string | null = null;
  const isSearchPrefix = /^search\b/i.test(q);
  const isHelpPrefix = q === "?" || /^help\b/i.test(q);
  const isBareVerb = /^(?:select|find|filter|hide|isolate|show)$/i.test(q);

  // 1. Score commands when no filter clauses are pinned (or when "search"/"help" is explicitly typed)
  let topCmdScore = 0;
  if ((!isBareVerb && clauses.length === 0) || isSearchPrefix || isHelpPrefix) {
    const candidateCmds =
      clauses.length === 0
        ? rootCommands
        : rootCommands.filter((c) => c.item.id === "cmd:open-search-window" || c.item.id === "cmd:help");
    const scoredCmds = candidateCmds
      .map((c) => {
        const base = scoreMatch(q, c.item.title, c.item.detail, c.keywords);
        const boost =
          (isSearchPrefix && c.item.id === "cmd:open-search-window") || (isHelpPrefix && c.item.id === "cmd:help") ? 200 : 0;
        return { item: c.item, score: base + boost };
      })
      .filter((x) => x.score > 0)
      .sort((a, b) => b.score - a.score);
    topCmdScore = scoredCmds[0]?.score ?? 0;
    for (const c of scoredCmds.slice(0, 16)) items.push(c.item);
  }

  // 2. If a model is loaded, evaluate the Search Engine on `(clauses + query)`
  if (s.artifacts.model?.status === "ready") {
    const queryTerms = queryTermsOfSpotlight(s);
    const terms = activeSpotlightTerms(s);
    const mods = pendingClauseMods(q, queryTerms);
    if (terms.length === 0 && isBareVerb) {
      summaryText = "Type group/type names (e.g. beam_V50x70 and beam_V95x70), or press ↓ then Tab to pin chips";
      const allDecls = runSearchTerms(s, [], 48);
      for (const r of allDecls.rows) {
        if (r.kind === "element" || isDuplicateLabelRow(s, r)) continue;
        items.push(declSpotlightItem(s, r, mods));
        if (items.length >= SPOTLIGHT_ITEM_CAP) break;
      }
    } else if (terms.length > 0) {
      const idx = indexOfModel(s);
      if (q) {
        const lastTok = normText(q.split(/\s+/).pop()!);
        let typeFacetCount = 0;
        for (const t of idx.types) {
          if (normText(t).includes(lastTok) && !clauses.some((c) => c.field === "type" && c.value === t)) {
            const typeClause: Omit<SearchClause, "id"> = { join: mods.join, negate: mods.negate, field: "type", op: ":", value: t };
            items.push(
              makeItem({
                id: `facet:type:${t}`,
                category: "filter",
                section: "Filters",
                badge: "type filter",
                title: `Filter by Type: ${t}`,
                detail: "Press Tab to add as a filter chip, or Enter to filter by this type",
                action: {
                  events: [
                    { type: "addSpotlightClause", clause: typeClause },
                    { type: "setSpotlightQuery", query: "" },
                  ],
                  close: false,
                  tabClause: typeClause,
                },
              }),
            );
            typeFacetCount++;
            if (typeFacetCount >= 4) break;
          }
        }
      }

      const summary = runSearchTerms(s, terms, SPOTLIGHT_ITEM_CAP);
      summaryText = `${summary.totalDecls.toLocaleString()} matching declarations · ${summary.elementDecls.length.toLocaleString()} elements`;
      if (summary.elementDecls.length > 0) {
        const desc = terms
          .map((t, i) => `${i > 0 && t.join === "or" ? "OR " : ""}${clauseLabel(t)}`)
          .join(" ");
        const queryClauses: Omit<SearchClause, "id">[] = queryTerms.map((t) => ({
          join: t.join,
          negate: t.negate,
          field: t.field,
          op: t.op,
          ...(t.key !== undefined ? { key: t.key } : {}),
          value: t.value,
        }));
        const toSelectionActionsEvents: Event[] = [
          ...queryClauses.map((clause): Event => ({ type: "addSpotlightClause", clause })),
          { type: "setSpotlightMode", mode: "selectionActions" },
        ];
        const transferEvents: Event[] = [{ type: "openWindow", window: "search" }, { type: "clearSearch" }];
        const cleanedQuery = s.spotlight.query.trim().replace(SEARCH_VERB_PREFIX_RE, "");
        if (cleanedQuery) {
          transferEvents.push({ type: "setSearchText", text: cleanedQuery });
        }
        for (const c of s.spotlight.clauses) {
          transferEvents.push({
            type: "addSearchClause",
            clause: {
              join: c.join,
              negate: c.negate,
              field: c.field,
              op: c.op,
              ...(c.key !== undefined ? { key: c.key } : {}),
              value: c.value,
            },
          });
        }
        const batchItems: SpotlightItem[] = [
          makeItem({
            id: "batch:select-matched",
            category: "search",
            section: "Matched Elements",
            badge: `${summary.elementDecls.length.toLocaleString()} elements`,
            title: `Select ${summary.elementDecls.length.toLocaleString()} Matching Elements (${desc})`,
            detail: "Enter: select & frame · Shift+Enter: choose action (Hide / Isolate / Add) · Tab: pin filter · Alt+Enter: hide",
            action: {
              events: [{ type: "selectDecls", decls: summary.elementDecls }, { type: "frameSelection" }],
              close: true,
              ...(queryClauses.length > 0 ? { tabClauses: queryClauses } : {}),
              shiftEvents: toSelectionActionsEvents,
              shiftMode: "selectionActions",
              altEvents: [{ type: "setHidden", decls: summary.elementDecls, hidden: true }],
            },
          }),
          makeItem({
            id: "batch:actions-matched",
            category: "search",
            section: "Matched Elements",
            badge: "actions…",
            title: `Actions on ${summary.elementDecls.length.toLocaleString()} Matching Elements (Hide, Isolate, Add…)`,
            detail: `Apply a batch visibility or selection action to (${desc})`,
            action: {
              events: toSelectionActionsEvents,
              close: false,
              nextMode: "selectionActions",
              ...(queryClauses.length > 0 ? { tabClauses: queryClauses } : {}),
            },
          }),
          makeItem({
            id: "batch:open-in-search",
            category: "search",
            section: "Matched Elements",
            badge: "search window",
            title: `Open in Search Window (${summary.totalDecls.toLocaleString()} matches)…`,
            detail: "Transfer query and filter chips to the visual Search window builder",
            action: {
              events: transferEvents,
              close: true,
            },
          }),
        ];
        const hasStructuredTerm =
          clauses.length > 0 ||
          terms.length > 1 ||
          terms.some((t) => t.field !== "any" || t.negate || t.join === "or");
        if (!isSearchPrefix && !isHelpPrefix && (hasStructuredTerm || topCmdScore < 85)) {
          items.unshift(...batchItems);
        } else {
          items.push(...batchItems);
        }
      }

      const wantsElements =
        clauses.length > 0 ||
        /\d|#|^el/i.test(q) ||
        terms.some(
          (t) =>
            t.field !== "any" ||
            /beam|column|shell|wall|slab|solid|brick|truss|disp|displacement|force|element|quad|tri|hex|tet|line/i.test(
              normText(t.value),
            ),
        );
      const shownPaths = new Set<DeclPath>();
      for (const r of summary.rows) {
        if (r.kind === "element" && !wantsElements) continue;
        if (isDuplicateLabelRow(s, r)) continue;
        shownPaths.add(r.path);
        items.push(declSpotlightItem(s, r, mods));
        if (items.length >= SPOTLIGHT_ITEM_CAP) break;
      }

      // When chips are pinned and the query box is empty (right after pressing Tab),
      // also list remaining unpinned groups & sections so the user can add another with Tab.
      if (!q && clauses.length > 0 && items.length < SPOTLIGHT_ITEM_CAP) {
        const allDecls = runSearchTerms(s, [], 48);
        for (const r of allDecls.rows) {
          if (r.kind === "element" || isDuplicateLabelRow(s, r) || shownPaths.has(r.path)) continue;
          if (clauses.some((c) => c.value === r.name)) continue;
          items.push(declSpotlightItem(s, r, { join: "and", negate: false }, "Add Another Group / Section (Tab to add)"));
          if (items.length >= SPOTLIGHT_ITEM_CAP) break;
        }
      }
    }
  }

  return {
    open,
    mode,
    modeLabel: meta.label,
    modeTitle: "Spotlight",
    placeholder: meta.placeholder,
    query,
    clauses: clauseViews,
    summary: summaryText,
    items: items.slice(0, SPOTLIGHT_ITEM_CAP),
    hint,
  };
}
