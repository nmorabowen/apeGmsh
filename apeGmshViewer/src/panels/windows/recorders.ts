// The recorders window (V3c0 placeholder): the recorders, global and per
// stage. The recorders slice replaces this view.

import { windowContentOf } from "../../state/selectors.ts";
import { placeholderList, type WindowView } from "./view.ts";

export const view: WindowView = (s, act) => placeholderList(windowContentOf(s, "recorders"), act);
