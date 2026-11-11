// The offers slice (ADR 0128 D5): the views agents offered, and the way back from one.
// One slice of the store's types; src/state/types.ts re-exports it, and the
// panels import it from there. A slice file imports sibling files, never the barrel.

import type { ThemeMode } from "../../theme/tokens.ts";
import type { CameraPose } from "./camera.ts";
import type { ClipPlane } from "./clip.ts";
import type { ContourCmap, ContourRange, ContourSource, GaussMode, UnitDisplay } from "./contour.ts";
import type { PhaseKey, Pick, UnitSystemId } from "./model.ts";
import type { Visibility } from "./visibility.ts";

/**
 * VA5 (ADR 0128 D5): where an offer stands. `pending` can be applied;
 * `applying` is the click being carried out; `applied` is a click that drew
 * the view; `refused` names why it cannot be (D2) and has no Apply.
 */
export type OfferStatus = "pending" | "applying" | "applied" | "refused";

/** What `--present` hands the window for one shot of its spec (src/spec/snapshot.ts `offersOfDocument`). */
export interface OfferIn {
  name: string;
  caption: string | null;
  /** D9: says the still-only keys the shot carries are ignored here; null when it carries none */
  ignored: string | null;
  /** the spec's side of the offer: read only by src/spec/snapshot.ts (the store and the panels never look inside it) */
  payload: unknown;
  /** already refused when it arrived (the window could not read its spec): listed with this reason, and never applied */
  refused?: string;
}

export interface Offer extends OfferIn {
  /** unique for the window session */
  id: number;
  /** when it reached the window (ms since the epoch) */
  arrivedAt: number;
  status: OfferStatus;
  /** why it is refused; null otherwise */
  reason: string | null;
  /** D5: the files outside the open set a click opens first */
  opens: readonly string[];
  /** refused by the drawn view at a click, which reading the set again does not undo */
  viewRefused: boolean;
}

/** VA5: the view before an offer was applied, for Back (D5). Everything an offer changes, and the set it was on. */
export interface ViewMemo {
  files: { model: string; results: string | null };
  at: PhaseKey | null;
  picks: readonly Pick[];
  visibility: Visibility;
  contour: {
    source: ContourSource;
    field: string | null;
    component: string | null;
    gauss: GaussMode;
    range: ContourRange;
    fixed: { min: number; max: number } | null;
    cmap: ContourCmap;
    sourceUnits: UnitSystemId | null;
    targetUnits: UnitDisplay;
  };
  deform: { on: boolean; scale: number; ghost: boolean };
  clip: { planes: readonly ClipPlane[]; applyCuts: boolean; showGizmos: boolean; cutFace: boolean };
  lineDiagram: boolean;
  loads: boolean;
  camera: CameraPose | null;
  theme: ThemeMode;
}

/** VA5: offers last for the window session (D5), through any open or failed read. */
export interface OffersState {
  /** newest first, at most MAX_OFFERS (state/reduce.ts) */
  list: readonly Offer[];
  /** the last offer id handed out */
  seq: number;
  /** the human's click being carried out; the offer runner acts on each new `seq`. Null at rest: a click while one runs is dropped. */
  request: { kind: "apply"; id: number; seq: number } | { kind: "back"; seq: number } | null;
  /** counts requests for the whole session */
  requestSeq: number;
  /** the views to go back to, last applied first; `offer` is the id it was applied from */
  back: readonly { offer: number; memo: ViewMemo }[];
  /** why the last Back could not restore the view; cleared by the next click */
  notice: string | null;
}
