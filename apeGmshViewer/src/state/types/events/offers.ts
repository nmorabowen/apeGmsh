// The offers slice's events: its part of the store's Event union (types/event.ts).
// Not re-exported by src/state/types.ts, which exports only the whole union.

import type { OfferIn, ViewMemo } from "../offers.ts";

export type OffersEvent =
  /** VA5 (ADR 0128 D5): `--present` reached the window. It lists offers and changes nothing else (INV-5). */
  | { type: "offerReceived"; offers: readonly OfferIn[]; at: number }
  /** VA5: the offer runner's judgement of a listed offer against the open set: applicable, or refused with the reason */
  | { type: "offerAssessed"; id: number; refused: string | null; opens: readonly string[] }
  /** VA5: the human clicked Apply; the only way an offer reaches the view (INV-5) */
  | { type: "applyOffer"; id: number }
  /** VA5: the human clicked Back */
  | { type: "backOffer" }
  /** VA5 (the runner): the click drew the view; `memo` is the view before it, null when no model was open to go back to */
  | { type: "offerApplied"; id: number; memo: ViewMemo | null }
  /** VA5 (the runner): the click was refused and the view put back; the offer stays listed as refused */
  | { type: "offerRefused"; id: number; reason: string }
  /** VA5 (the runner): Back finished; `notice` says why the view could not be restored */
  | { type: "offerBackDone"; notice: string | null };
