// The state before any event (state/reduce.ts re-exports all of it). A slice
// with a sub-reducer under reduce/ keeps its own initial value there; this
// file composes them with the rest.

import { initialClip } from "../clip.ts";
import { initialLineDiagramState } from "../lineDiagram.ts";
import { initialSearch, initialSpotlight } from "../spotlight.ts";
import type { ContourState, DeformState, LineDiagramState, SpatialNdm, State } from "../types.ts";
import { initialCamera } from "./camera.ts";
import { initialOffers } from "./offers.ts";
import { initialLoads, initialPrincipal, initialReactions, initialSprings, initialSupports, initialVectors } from "./overlays.ts";
import { initialVariants } from "./variants.ts";

/** The deformed shape before the user turns it on: true scale, no ghost (V4d; the Qt defaults). */
export const initialDeform: DeformState = { on: false, scale: 1, ghost: false, measured: null };

export const initialLineDiagram: LineDiagramState = initialLineDiagramState;

/** No model, no dimension. */
const NO_NDM: SpatialNdm = { ndm: null, reason: "no model is loaded" };

/** A record with no prototype: a key named `constructor` or `__proto__` is just a key. */
export const emptyRecord = <T>(): Record<string, T> => Object.create(null) as Record<string, T>;

/** The contour before anything is picked: whole-history range, jet (V4c1). */
export const initialContour: ContourState = {
  source: "nodal",
  gauss: "cell",
  join: null,
  field: null,
  component: null,
  range: "history",
  fixed: null,
  cmap: "jet",
  declaredUnits: null,
  sourceUnits: null,
  targetUnits: "as_stored",
  stepRange: null,
  scan: null,
  notice: null,
  reads: 0,
};

export const initialState: State = {
  artifacts: { geometry: null, model: null, results: null },
  decls: emptyRecord(),
  names: emptyRecord(),
  blocks: [],
  mesh: null,
  phase: { axis: [], at: null },
  selection: { decls: [], picks: [] },
  hover: null,
  visibility: { hidden: [], edges: true, opacity: 1, colourBy: "group", extrudeFrames: false, showOffsets: false },
  nodes: { on: false, size: null, auto: true },
  gaussPoints: { on: false, size: null, auto: true, measured: null },
  inspector: { pinned: [] },
  windows: { open: [] },
  overrides: {},
  geometry: null,
  provenance: [],
  provenanceOrigin: false,
  source: { seq: 0, request: null, last: null },
  view: { frameSeq: 0 },
  definitions: null,
  neutralLoads: null,
  sequence: { status: "absent" },
  stage: null,
  lineage: null,
  stageElements: { status: "none" },
  results: null,
  resultStep: null,
  user: null,
  combination: null,
  userResult: null,
  variants: initialVariants,
  contour: initialContour,
  ndm: NO_NDM,
  units: null,
  deform: initialDeform,
  clip: initialClip,
  lineDiagram: initialLineDiagram,
  loads: initialLoads,
  reactions: initialReactions,
  springs: initialSprings,
  principal: initialPrincipal,
  vectors: initialVectors,
  supports: initialSupports,
  spotlight: initialSpotlight,
  search: initialSearch,
  camera: initialCamera,
  theme: "dark",
  offers: initialOffers,
};
