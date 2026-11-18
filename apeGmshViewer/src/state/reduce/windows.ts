// The windows slice of the reducer (state/reduce.ts delegates here): the
// open windows, the spotlight palette and the search window's query.

import type { WindowsEvent } from "../types/events/windows.ts";
import { WINDOW_KINDS, type State } from "../types.ts";

/** The open windows, the spotlight and the search, each its own field of the state. */
export type Windows = Pick<State, "windows" | "spotlight" | "search">;

/** The windows slice after `e`: the same object when nothing changed. */
export function reduceWindows(w: Windows, e: WindowsEvent): Windows {
  switch (e.type) {
    case "openWindow":
      if (!WINDOW_KINDS.includes(e.window)) throw new RangeError(`openWindow: ${JSON.stringify(e.window)} is not a window kind`);
      if (w.windows.open.includes(e.window)) return w;
      return { ...w, windows: { open: [...w.windows.open, e.window] } };
    case "closeWindow":
      return { ...w, windows: { open: w.windows.open.filter((x) => x !== e.window) } };
    case "openSpotlight": {
      const mode = e.mode ?? "root";
      const query = e.query ?? w.spotlight.query;
      if (w.spotlight.open && w.spotlight.mode === mode && w.spotlight.query === query) return w;
      return { ...w, spotlight: { ...w.spotlight, open: true, mode, query } };
    }
    case "closeSpotlight":
      return !w.spotlight.open && w.spotlight.mode === "root" && w.spotlight.query === "" && w.spotlight.clauses.length === 0
        ? w
        : { ...w, spotlight: { ...w.spotlight, open: false, mode: "root", query: "", clauses: [] } };
    case "setSpotlightMode": {
      const query = e.query ?? "";
      if (w.spotlight.mode === e.mode && w.spotlight.query === query) return w;
      return { ...w, spotlight: { ...w.spotlight, mode: e.mode, query } };
    }
    case "setSpotlightQuery":
      return w.spotlight.query === e.query ? w : { ...w, spotlight: { ...w.spotlight, query: e.query } };
    case "addSpotlightClause": {
      const seq = w.spotlight.seq + 1;
      const clause = { ...e.clause, id: `sp-${seq}` };
      return { ...w, spotlight: { ...w.spotlight, clauses: [...w.spotlight.clauses, clause], seq } };
    }
    case "removeSpotlightClause": {
      const clauses = w.spotlight.clauses.filter((c) => c.id !== e.id);
      return clauses.length === w.spotlight.clauses.length ? w : { ...w, spotlight: { ...w.spotlight, clauses } };
    }
    case "clearSpotlightClauses":
      return w.spotlight.clauses.length === 0 ? w : { ...w, spotlight: { ...w.spotlight, clauses: [] } };
    case "setSearchText":
      return w.search.text === e.text ? w : { ...w, search: { ...w.search, text: e.text } };
    case "addSearchClause": {
      const seq = w.search.seq + 1;
      const clause = { ...e.clause, id: `q-${seq}` };
      return { ...w, search: { ...w.search, clauses: [...w.search.clauses, clause], seq } };
    }
    case "removeSearchClause": {
      const clauses = w.search.clauses.filter((c) => c.id !== e.id);
      return clauses.length === w.search.clauses.length ? w : { ...w, search: { ...w.search, clauses } };
    }
    case "clearSearch":
      return w.search.text === "" && w.search.clauses.length === 0 ? w : { ...w, search: { ...w.search, text: "", clauses: [] } };
    default: {
      const unknown: never = e;
      throw new Error(`reduceWindows: unknown event ${JSON.stringify(unknown)}`);
    }
  }
}
