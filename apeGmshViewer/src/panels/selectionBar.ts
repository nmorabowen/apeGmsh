// The selection bar (ADR 0138 D2, D6, D7): which kind a click and a box act
// on, how many of each are selected, and a way to clear the active one. It
// sits bottom-centre over the Windows menu and shows only with a mesh. The
// kind buttons dispatch the same event as the Sets panel's toggle and the 1
// and 2 keys, so all three stay one value.

import type { State, Store } from "../state/store.ts";
import { selectedElementsOf } from "../state/selectors.ts";
import { el, listen } from "../ui/dom.ts";

/** What the bar shows, as plain values (tested in Node; the DOM below draws them). */
export function selectionBarLines(s: State): { kinds: { kind: "element" | "node"; label: string; count: number; active: boolean }[]; hint: string; canClear: boolean; canUndo: boolean } | null {
  if (!s.mesh) return null;
  const elements = selectedElementsOf(s).length, nodes = s.selection.nodes.length;
  const active = s.selection.active;
  return {
    kinds: [
      { kind: "element", label: "Elements", count: elements, active: active === "element" },
      { kind: "node", label: "Nodes", count: nodes, active: active === "node" },
    ],
    hint: "Drag: left to right inside, right to left touching · Shift adds · Ctrl removes",
    canClear: (active === "node" ? nodes : elements) > 0,
    canUndo: s.selectionUndo.past.length > 0,
  };
}

export function mountSelectionBar(store: Store): () => void {
  const bar = el("div");
  bar.id = "selection-bar";
  bar.hidden = true;
  document.body.append(bar);
  let unlisten: (() => void)[] = [];
  const render = (s: State, prev: State | null) => {
    if (prev && s.selection === prev.selection && s.selectionUndo === prev.selectionUndo && s.mesh === prev.mesh && s.decls === prev.decls) return;
    for (const u of unlisten) u();
    unlisten = [];
    const lines = selectionBarLines(s);
    bar.hidden = lines === null;
    if (lines === null) return bar.replaceChildren();
    const parts: HTMLElement[] = [el("span", "window-menu-label", "Select")];
    for (const k of lines.kinds) {
      const b = el("button", `window-toggle selection-kind${k.active ? " on" : ""}`, `${k.label} ${k.count.toLocaleString()}`);
      b.type = "button";
      b.dataset["kind"] = k.kind;
      b.title = `${k.kind === "node" ? "Clicks and boxes select nodes" : "Clicks and boxes select elements"} (key ${k.kind === "node" ? "2" : "1"})`;
      b.setAttribute("aria-pressed", String(k.active));
      unlisten.push(listen(b, "click", () => store.dispatch({ type: "setWorkspaceKind", kind: k.kind === "node" ? "nodes" : "elements" })));
      parts.push(b);
    }
    parts.push(el("span", "selection-hint", lines.hint));
    const undo = el("button", "window-toggle", "Undo");
    undo.type = "button";
    undo.title = "Undo the last selection change (Ctrl+Z); Ctrl+Y redoes it";
    undo.disabled = !lines.canUndo;
    unlisten.push(listen(undo, "click", () => store.dispatch({ type: "undoSelection" })));
    parts.push(undo);
    const clear = el("button", "window-toggle", "Clear");
    clear.type = "button";
    clear.title = "Clear the active kind's selection (Esc)";
    clear.disabled = !lines.canClear;
    unlisten.push(listen(clear, "click", () => store.dispatch({ type: "clearSelection" })));
    parts.push(clear);
    bar.replaceChildren(...parts);
  };
  render(store.get(), null);
  const unsubscribe = store.subscribe(render);
  return () => {
    unsubscribe();
    for (const u of unlisten) u();
    bar.remove();
  };
}
