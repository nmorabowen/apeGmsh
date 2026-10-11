// The materials window (V3c0 placeholder): the materials and sections, with
// their parameters by field name where /opensees/decl_params names them. A
// click selects the declaration (a store event; the inspector shows it). The
// materials / sections slice replaces this view.

import { windowContentOf } from "../../state/selectors.ts";
import { placeholderList, type WindowView } from "./view.ts";

export const view: WindowView = (s, act) => placeholderList(windowContentOf(s, "materials"), act);
