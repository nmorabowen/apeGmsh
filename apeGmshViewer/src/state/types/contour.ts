// The contour slice: the field, colour range, colour map and units of the contour shown.
// One slice of the store's types; src/state/types.ts re-exports it, and the
// panels import it from there. A slice file imports sibling files, never the barrel.

import type { UnitSystemId } from "./model.ts";

/** How a contour's colour range is chosen (V4c1): the whole history of the stage (the default), the step shown, or a user's min and max. */
export type ContourRange = "history" | "step" | "fixed";

/** The contour colormaps (state/contour.ts `CMAPS`); jet is the default. */
export const CMAP_NAMES = [
  "jet",
  "turbo",
  "viridis",
  "plasma",
  "inferno",
  "magma",
  "coolwarm",
  "RdBu_r",
  "RdBu",
  "bwr",
  "seismic",
  "BrBG",
  "Spectral",
] as const;
export type ContourCmap = (typeof CMAP_NAMES)[number];

/** The history-range scan of the contour's field (main reads every step; the effects report it). */
export interface ContourScan {
  /** `<results path>|<stage id>|<field>|<component>` (state/contour.ts `contourKeys`) */
  key: string;
  status: "running" | "done" | "failed";
  done: number;
  total: number;
  /** null until done, or when the history holds no finite value */
  min: number | null;
  max: number | null;
  error: string | null;
}

/**
 * What a contour colours by (V4c2a): a nodal field, or an element (Gauss)
 * field. VU16: or a state-free `ON_NODES` user result of the set's `.user.h5`;
 * then `field` names the entry and `component` one of its columns, and it is
 * shown at whatever phase is on screen (it has no state).
 */
export type ContourSource = "nodal" | "element" | "user";

/**
 * How an element (Gauss) field is drawn (V4c2a). `cell` is flat per cell: the
 * element's one Gauss value (Qt `gauss_cell`), or, with several points, the
 * element mean of its Gauss values (`element_mean`, an app presentation with
 * no Qt equivalent). V4c2b draws the node-located modes: `cell_averaged` (Qt
 * `gauss_cell_averaged`: one-point cell values averaged to the nodes across
 * elements; each element's mean on a field with several points), `node`
 * (`gauss_node`: extrapolated to the nodes and averaged) and `node_discrete`
 * (`gauss_node_discrete`: extrapolated per element, no averaging).
 */
export type GaussMode = "cell" | "cell_averaged" | "node" | "node_discrete";

/**
 * Why a joined element field cannot be carried to the nodes (V4c2b,
 * state/extrapolation.ts `extrapolationVerdict`), null where it can.
 * `corners` blocks every node-located topology (an element type with no
 * corner shape functions, a short row); `matrix` blocks the extrapolating
 * ones on groups with several Gauss points (unusable `_natural_coords`).
 */
export interface ExtrapolationVerdict {
  corners: string | null;
  matrix: string | null;
}

/**
 * The join of an element field's rows to the mesh (V4c2a), as the effects
 * found it: every `_element_index` id is a cell of the model, or the field is
 * refused (an id the mesh does not have, or one listed twice). `key` is
 * selectors `elementJoinKey`.
 */
export type ElementJoinVerdict =
  | { key: string; ok: true; rows: number; extrapolation: ExtrapolationVerdict }
  | { key: string; ok: false; code: "unknown-id" | "duplicate-id"; reason: string };

/** What unit system the contour legend displays values in: `as_stored` (raw file values, default) or a `baseUnits` system. */
export type UnitDisplay = "as_stored" | UnitSystemId;

/** The contour colouring (V4c1, #1756): what is picked, and the ranges the effects measured. */
export interface ContourState {
  /** V4c2a: a nodal field (V4c1) or an element field; `field` and `component` name one of that source */
  source: ContourSource;
  /** V4c2a: how an element field is drawn */
  gauss: GaussMode;
  /** V4c2a: the join of the picked element field to the mesh; null until the effects checked it */
  join: ElementJoinVerdict | null;
  /** the field (`displacement`) and its component (a dataset name such as `displacement_x`, or `magnitude`); null until picked */
  field: string | null;
  component: string | null;
  range: ContourRange;
  /** the user's min and max (`range: fixed`) */
  fixed: { min: number; max: number } | null;
  cmap: ContourCmap;
  /** the consistent unit system declared in the model or results file (`/meta/@units` or `INFO/PROVENANCE@units`), or null when undeclared */
  declaredUnits: UnitSystemId | null;
  /** the file's native `baseUnits` system (`declaredUnits` when declared; null when undeclared until the user selects `from <system>`) */
  sourceUnits: UnitSystemId | null;
  /** the target `baseUnits` display system (`as_stored` keeps raw numbers; defaults to `declaredUnits` when declared) */
  targetUnits: UnitDisplay;
  /** the finite min and max of the step shown, over every recorded node; `key` names the step and component it is for */
  stepRange: { key: string; min: number | null; max: number | null } | null;
  scan: ContourScan | null;
  /** a typed notice, such as a selection cleared by a re-read that no longer has the field */
  notice: string | null;
  /** results reads so far (`resultsLoaded`): a re-read at the same path may hold other values, so what the contour shows is keyed by it */
  reads: number;
}
