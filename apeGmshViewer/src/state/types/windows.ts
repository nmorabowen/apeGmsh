// The windows slice: the model-definition windows, the Search window and Spotlight.
// One slice of the store's types; src/state/types.ts re-exports it, and the
// panels import it from there. A slice file imports sibling files, never the barrel.

import type { DeclPath } from "./model.ts";

export type WindowKind = "stages" | "analysis" | "patterns" | "recorders" | "materials" | "search";

/** Every window kind, in menu order (the registry in panels/windows/ has one view per kind). */
export const WINDOW_KINDS: readonly WindowKind[] = ["stages", "analysis", "patterns", "recorders", "materials", "search"];

/** Fields the pure search engine (`state/search.ts`) filters declarations by. */
export type SearchField =
  | "any"
  | "kind"
  | "type"
  | "name"
  | "group"
  | "role"
  | "uses"
  | "id"
  | "param"
  | "stage"
  | "visibility";

/** Comparison operators supported by a `SearchClause`. */
export type SearchOp = ":" | "contains" | "=" | "!=" | ">" | ">=" | "<" | "<=";

/** One structured filter clause in the Search window or Spotlight filter bar. */
export interface SearchClause {
  readonly id: string;
  readonly join: "and" | "or";
  readonly negate: boolean;
  readonly field: SearchField;
  readonly op: SearchOp;
  readonly key?: string;
  readonly value: string;
}

/** The Search window's state: structured filter chips and free-text query. */
export interface SearchState {
  readonly text: string;
  readonly clauses: readonly SearchClause[];
  readonly seq: number;
}

/** Which drill-down mode Spotlight (`Ctrl+K`) is currently in. */
export type SpotlightMode =
  | "root"
  | "help"
  | "opacity"
  | "colourBy"
  | "phase"
  | "step"
  | "contourField"
  | "contourGauss"
  | "contourRange"
  | "contourCmap"
  | "contourUnits"
  | "deformScale"
  | "windows"
  | "planes"
  | "selectionActions";

/** Spotlight (`Ctrl+K`) state: open status, drill-down mode, query string, and interactive filter chips. */
export interface SpotlightState {
  readonly open: boolean;
  readonly mode: SpotlightMode;
  readonly query: string;
  readonly clauses: readonly SearchClause[];
  readonly seq: number;
}

/**
 * One row of a model-definition window: what the reader read, where (`h5`),
 * the `/opensees/decls` key it joins when it joins one, and the declaration a
 * click selects (null when the row is not a declaration of the store).
 */
export interface WindowRow {
  /** the row's HDF5 path (unique within a window) */
  key: string;
  /** the group the row is listed under (`global`, `stage 0 · gravity`, `time series`, ...) */
  section: string;
  title: string;
  detail: string;
  h5: string;
  declKey: string | null;
  select: DeclPath | null;
}

/** What a window lists: its rows, and a sentence when there is nothing (or something missing) to say why. */
export interface WindowContent {
  rows: WindowRow[];
  note: string | null;
}

export interface Layout {
  open: WindowKind[];
}
