// The model-definition window frame (ADR 0112 D6): a menu with one button per
// window kind, and the open windows (`state.windows.open`), each a card with
// its title, a close button and the body its registry view builds. Opening and
// closing are the store's `openWindow` / `closeWindow` events; the frame keeps
// no state of its own and calls no other panel. It creates its two elements,
// so the page needs no markup for it.

import type { State, Store } from "../../state/store.ts";
import { WINDOW_KINDS, type DeclPath } from "../../state/types.ts";
import { el, listen } from "../../ui/dom.ts";
import { WINDOWS } from "./registry.ts";

/**
 * The parts of the state a window view reads; the frame re-renders when one
 * changes. `selection` (the selected-row highlight), `stage` (the selected
 * stage card) and `source` (a go-to-source answer) are among them, so a click
 * in a window comes back through the store as a redraw (D6: one-way flow).
 */
export const changed = (s: State, p: State): boolean =>
  s.windows !== p.windows ||
  s.artifacts !== p.artifacts ||
  s.definitions !== p.definitions ||
  s.sequence !== p.sequence ||
  s.decls !== p.decls ||
  s.provenance !== p.provenance ||
  s.selection !== p.selection ||
  s.stage !== p.stage ||
  s.source !== p.source;

/** The highlight of a selected row: the office accent, as the selection is drawn everywhere else. */
export const SELECTED_STYLE = { background: "var(--accent-wash)", boxShadow: "inset 3px 0 0 var(--accent)" } as const;

interface Markable {
  readonly children: ArrayLike<unknown>;
  className: string;
  dataset: Record<string, string | undefined>;
  style: unknown;
}

/**
 * Mark every row (`.window-row`) of a window body whose `data-decl` is selected: the
 * `selected` class and the accent highlight. Rows carry `data-decl` when a
 * click on them selects that declaration, so the highlight is the store's
 * selection read back, never a panel's own memory of a click. Returns the
 * count marked.
 */
export function markSelected(root: HTMLElement, selected: readonly DeclPath[]): number {
  if (!selected.length) return 0;
  const want = new Set(selected);
  let n = 0;
  const walk = (node: Markable) => {
    const d = node.dataset?.["decl"];
    // Rows only (`.window-row`): an inline link to a declaration carries `data-decl` too, and stays a link.
    if (d !== undefined && want.has(d) && node.className.split(" ").includes("window-row")) {
      node.className += " selected";
      Object.assign(node.style as object, SELECTED_STYLE);
      n++;
    }
    for (const c of Array.from(node.children ?? [])) walk(c as Markable);
  };
  walk(root as unknown as Markable);
  return n;
}

export function mountWindows(store: Store): () => void {
  const menu = el("nav");
  menu.id = "windows-menu";
  const host = el("div");
  host.id = "windows";
  document.body.append(menu, host);

  const act = store.dispatch.bind(store);
  const buttons = WINDOW_KINDS.map((kind) => {
    const b = el("button", "window-toggle", WINDOWS[kind].title);
    b.dataset["window"] = kind;
    return { kind, b, off: listen(b, "click", () => store.dispatch({ type: store.get().windows.open.includes(kind) ? "closeWindow" : "openWindow", window: kind })) };
  });
  menu.append(el("span", "window-menu-label", "Windows"), ...buttons.map((x) => x.b));

  const render = (s: State, prev: State | null) => {
    if (prev && !changed(s, prev)) return;
    const ready = s.artifacts.model?.status === "ready";
    menu.hidden = !ready;
    for (const { kind, b } of buttons) b.classList.toggle("on", s.windows.open.includes(kind));
    const cards = ready
      ? s.windows.open.map((kind) => {
          const entry = WINDOWS[kind];
          const card = el("section", "window");
          card.dataset["window"] = kind;
          const head = el("h3", undefined, entry.title);
          const close = el("button", "window-close", "×");
          close.title = `close ${entry.title}`;
          close.addEventListener("click", () => store.dispatch({ type: "closeWindow", window: kind }));
          head.append(close);
          const body = entry.view(s, act);
          markSelected(body, s.selection.decls);
          card.append(head, body);
          return card;
        })
      : [];
    // A re-render replaces the cards; keep where the user had scrolled (the
    // host's list, and each stage timeline's horizontal position), so a click
    // that selects a row deep in a window does not jump it back to the top.
    const top = host.scrollTop;
    const lefts = new Map<string, number>();
    for (const t of host.querySelectorAll<HTMLElement>("[data-window] .stage-timeline")) {
      const kind = t.closest<HTMLElement>("[data-window]")?.dataset["window"];
      if (kind) lefts.set(kind, t.scrollLeft);
    }
    host.hidden = cards.length === 0;
    host.replaceChildren(...cards);
    host.scrollTop = top;
    for (const card of cards) {
      const left = lefts.get(card.dataset["window"] ?? "");
      const t = card.querySelector<HTMLElement>(".stage-timeline");
      if (left !== undefined && t) t.scrollLeft = left;
    }
  };
  render(store.get(), null);
  const unsubscribe = store.subscribe(render);
  return () => {
    unsubscribe();
    for (const x of buttons) x.off();
    menu.remove();
    host.remove();
  };
}
