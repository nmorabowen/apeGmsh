// The analysis window (V3c0 placeholder): the analysis chain of
// /opensees/analysis and of each stage. The analysis slice replaces this view.

import { windowContentOf } from "../../state/selectors.ts";
import { placeholderList, type WindowView } from "./view.ts";

export const view: WindowView = (s, act) => placeholderList(windowContentOf(s, "analysis"), act);
