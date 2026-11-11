// The store's Event union, composed from each slice's events (types/events/<slice>.ts).
// src/state/types.ts re-exports `Event` and `EventType`, not the slice unions.

import type { CameraEvent } from "./events/camera.ts";
import type { ClipEvent } from "./events/clip.ts";
import type { ContourEvent } from "./events/contour.ts";
import type { DeformEvent } from "./events/deform.ts";
import type { LineDiagramEvent } from "./events/lineDiagram.ts";
import type { ModelEvent } from "./events/model.ts";
import type { OffersEvent } from "./events/offers.ts";
import type { OverlaysEvent } from "./events/overlays.ts";
import type { ResultsEvent } from "./events/results.ts";
import type { UserEvent } from "./events/user.ts";
import type { VariantsEvent } from "./events/variants.ts";
import type { WindowsEvent } from "./events/windows.ts";

/**
 * Decision 17's twenty events, plus `fileClosed` (added by V2e review: an
 * `onOpen` set that names no results file closes the results artifact), and
 * V2f's go-to-source pair: `requestSource` (a panel asks; the effects act)
 * and `sourceResult` (the effects answer), and `setLoaded`: the model and its
 * geometry sibling re-read together after a D1 re-run, applied as one step so
 * no state pairs a new file with an old one; and V4f's `selectStage` (a
 * stage card click); and V4b's `resultsLoaded` and `resultStep`.
 */
export type Event =
  | ModelEvent
  | ResultsEvent
  | UserEvent
  | VariantsEvent
  | ContourEvent
  | DeformEvent
  | ClipEvent
  | LineDiagramEvent
  | OverlaysEvent
  | WindowsEvent
  | OffersEvent
  | CameraEvent;

export type EventType = Event["type"];
