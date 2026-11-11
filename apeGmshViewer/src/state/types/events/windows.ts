// The windows slice's events: its part of the store's Event union (types/event.ts).
// Not re-exported by src/state/types.ts, which exports only the whole union.

import type { SearchClause, SpotlightMode, WindowKind } from "../windows.ts";

export type WindowsEvent =
  | { type: "openWindow"; window: WindowKind }
  | { type: "closeWindow"; window: WindowKind }
  /** Spotlight (`Ctrl+K`): open, close, navigate drill-down modes, edit query or filter chips */
  | { type: "openSpotlight"; mode?: SpotlightMode; query?: string }
  | { type: "closeSpotlight" }
  | { type: "setSpotlightMode"; mode: SpotlightMode; query?: string }
  | { type: "setSpotlightQuery"; query: string }
  | { type: "addSpotlightClause"; clause: Omit<SearchClause, "id"> }
  | { type: "removeSpotlightClause"; id: string }
  | { type: "clearSpotlightClauses" }
  /** Search window: edit free-text query or structured filter clauses */
  | { type: "setSearchText"; text: string }
  | { type: "addSearchClause"; clause: Omit<SearchClause, "id"> }
  | { type: "removeSearchClause"; id: string }
  | { type: "clearSearch" };
