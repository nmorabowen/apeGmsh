// The user slice's events: its part of the store's Event union (types/event.ts).
// Not re-exported by src/state/types.ts, which exports only the whole union.

import type { CombinationRead, UserReadIn, UserResultRead } from "../user.ts";

export type UserEvent =
  /** ADR 0130 (renderer/app.ts): the set's `.user.h5` is being read; the same path again is pending too, so a verdict from before a change is never shown as current */
  | { type: "userOpened"; path: string }
  /** ADR 0130: `user:read` answered for the open `.user.h5`; VU9b: also the file as main read it again after a combination save (panels/combine.ts) */
  | { type: "userRead"; path: string; read: UserReadIn }
  /** ADR 0130: the set has no `.user.h5` (none beside it, or it was deleted) */
  | { type: "userClosed" }
  /** VU7b-2 (renderer/app.ts): the User Combination shown is being read, or its read answered; one for another key is dropped */
  | { type: "combinationRead"; read: CombinationRead }
  /** VU16 (renderer/app.ts): the user result picked is being read, or its read answered; one for another key is dropped */
  | { type: "userResultRead"; read: UserResultRead };
