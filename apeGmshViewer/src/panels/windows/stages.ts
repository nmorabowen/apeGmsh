// The stages window (V3c0 placeholder): one row per /opensees/stages/stage_NNN,
// and whether the file carries a /sequence zone. The stage-diagram slice
// replaces this view; the frame and the registry stay as they are.

import { windowContentOf } from "../../state/selectors.ts";
import { placeholderList, type WindowView } from "./view.ts";

export const view: WindowView = (s, act) => placeholderList(windowContentOf(s, "stages"), act);
