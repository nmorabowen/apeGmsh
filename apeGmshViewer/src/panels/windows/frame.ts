// The model-definition window frame (ADR 0112 D6): a menu with one button per
// window kind, and the open windows (`state.windows.open`), each a card with
// its title, a close button and the body its registry view builds. Opening and
// closing are the store's `openWindow` / `closeWindow` events; the frame keeps
// no state of its own and calls no other panel. It creates its two elements,
// so the page needs no markup for it.

import type { State, Store } from "../../state/store.ts";
import { WINDOW_KINDS } from "../../state/types.ts";
import { el, listen } from "../../ui/dom.ts";
import { WINDOWS } from "./registry.ts";

/** The parts of the state a window view reads; the frame re-renders when one changes. */
const changed = (s: State, p: State): boolean =>
  s.windows !== p.windows ||
  s.artifacts !== p.artifacts ||
  s.definitions !== p.definitions ||
  s.sequence !== p.sequence ||
  s.decls !== p.decls;

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
          card.append(head, entry.view(s, act));
          return card;
        })
      : [];
    host.hidden = cards.length === 0;
    host.replaceChildren(...cards);
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
