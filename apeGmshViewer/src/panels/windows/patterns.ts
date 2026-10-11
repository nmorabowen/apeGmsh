// The patterns window (V3c0 placeholder): the load patterns, global and per
// stage, and the time series. The patterns / time-series slice replaces this view.

import { windowContentOf } from "../../state/selectors.ts";
import { placeholderList, type WindowView } from "./view.ts";

export const view: WindowView = (s, act) => placeholderList(windowContentOf(s, "patterns"), act);
