// The variants slice's events: its part of the store's Event union (types/event.ts).
// Not re-exported by src/state/types.ts, which exports only the whole union.

import type { UserResultRead } from "../user.ts";
import type { CompareIn, CompareModelAnswer, CompareUserAnswer, CompareUserPick, DifferenceRead, VariantsIn } from "../variants.ts";

export type VariantsEvent =
  /** VU10 (renderer/contour.ts `VariantRunner`): main's study, at start and after every change; null with no manifest */
  | { type: "variantsRead"; study: VariantsIn | null }
  /** VU10: the variants panel asks main to open row `index` as the window set (D9.6) */
  | { type: "variantSwitch"; index: number; label: string }
  /** VU10 (the runner): main answered the switch; `reason` when it refused */
  | { type: "variantSwitched"; reason: string | null }
  /** VU10: the variants panel asks main to open row `index` as the compare set (D9.5), or with null to close it */
  | { type: "variantCompare"; row: { index: number; label: string } | null }
  /** VU10 (the runner): main answered the compare */
  | { type: "compareOpened"; result: CompareIn | { index: number; label: string; reason: string } }
  /** VU10: the contour shows the window variant minus the compare variant, or not */
  | { type: "setDifference"; on: boolean }
  /** VU10 (the runner): the difference shown is being read, or its read answered; one for another key is dropped */
  | { type: "differenceRead"; read: DifferenceRead }
  /** VU19 (the runner): main answered `compare:model` for the slot it names; one for another slot is dropped */
  | { type: "compareModelRead"; answer: CompareModelAnswer }
  /** VU19 (the runner): main answered `compare:userRead` for the slot it names; one for another slot is dropped */
  | { type: "compareUserRead"; answer: CompareUserAnswer }
  /** VU19: the variants panel draws one component of a compare user result as the contour, or with null stops */
  | { type: "showCompareUser"; pick: CompareUserPick | null }
  /** VU19 (the runner): the compare user result shown is being read, or its read answered; one for another key is dropped */
  | { type: "compareUserResultRead"; read: UserResultRead };
