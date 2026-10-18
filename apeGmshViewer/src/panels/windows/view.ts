// The shape of a model-definition window's view, and the V3c0 placeholder
// list the five views share until the V3 slice that owns each replaces it.
//
// A view is pure: `(state, act) -> DOM`. It reads the state and selectors,
// and a click dispatches an event through `act`; it never calls another
// panel or window (ADR 0112 D6). The frame (frame.ts) mounts it and calls it
// again whenever the state it reads changes.

import type { Event, State } from "../../state/store.ts";
import type { WindowContent } from "../../state/types.ts";
import { el, isSelectAdd } from "../../ui/dom.ts";

export type WindowView = (s: State, act: (e: Event) => void) => HTMLElement;

/** The rows of a window, grouped by section; a row that is a declaration of the store selects it on click. */
export function placeholderList(content: WindowContent, act: (e: Event) => void): HTMLElement {
  const body = el("div", "window-list");
  if (content.note) body.append(el("div", "window-note", content.note));
  let section: string | null = null;
  for (const r of content.rows) {
    if (r.section !== section) {
      section = r.section;
      body.append(el("div", "window-section", section));
    }
    const row = el("div", "window-row" + (r.select ? " clickable" : ""));
    const head = el("div", "window-row-head");
    head.append(el("span", "window-row-title", r.title));
    if (r.declKey) head.append(el("span", "window-row-decl", r.declKey));
    row.append(head);
    if (r.detail) row.append(el("div", "window-row-detail", r.detail));
    row.append(el("div", "f-src", r.h5));
    const decl = r.select;
    if (decl) {
      row.title = `select ${decl}`;
      row.addEventListener("click", (e) => act({ type: isSelectAdd(e) ? "selectAdd" : "select", pick: { decl, at: null, nodes: null } }));
    }
    body.append(row);
  }
  return body;
}
