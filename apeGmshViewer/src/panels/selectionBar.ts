// The selection bar (ADR 0138 D2, D6, D7, D12): which kind a click and a box
// act on, how many of each are selected, undo, and a way to clear the active
// one. It sits bottom-centre over the Windows menu and shows only with a mesh.
// The kind buttons dispatch the same event as the Sets panel's toggle and the
// 1 and 2 keys, so all three stay one value. Its Sets list holds the saved sets
// of the active kind: each can be selected, added, subtracted, hidden or
// isolated.

import type { State, Store } from "../state/store.ts";
import { savedSets, selectedElementsOf, setTargets } from "../state/selectors.ts";
import { el, listen } from "../ui/dom.ts";

/** One saved set in the bar's list: what it holds, what selecting it takes, and whether Hide and Isolate apply. */
export interface SetRow {
  row: number;
  name: string;
  /** "12", or "9 of 12" when some members no longer match the model, with the verdict when it is not current */
  detail: string;
  /** the element declarations it hides or isolates; empty for a node set */
  decls: readonly string[];
  /** why Hide and Isolate are off, or null */
  noHide: string | null;
}

/** The saved sets of the active kind, for the bar's list (tested in Node); null without a mesh. */
export function setListLines(s: State): { kind: "element" | "node"; rows: SetRow[]; empty: string } | null {
  if (!s.mesh) return null;
  const kind = s.selection.active, setKind = kind === "node" ? "nodes" : "elements";
  const rows = savedSets(s)
    .filter((x) => x.kind === setKind)
    .map((x): SetRow => {
      const t = setTargets(s, x);
      const verdict = x.verdict && x.verdict.verdict !== "current" ? ` · ${x.verdict.verdict}` : "";
      return {
        row: x.row,
        name: x.name,
        detail: `${t.taken < t.total ? `${t.taken.toLocaleString()} of ` : ""}${t.total.toLocaleString()}${verdict}`,
        decls: t.decls,
        noHide: kind === "node" ? "Hiding a node set hides the elements its nodes belong to: a later slice (ADR 0138 D4)" : t.decls.length === 0 ? "none of its members is in this model" : null,
      };
    });
  const empty = s.workspace.read?.status === "ready"
    ? `No ${setKind.slice(0, -1)} sets saved: select ${setKind} and save a set in the Sets panel.`
    : "No workspace sets are read for this model.";
  return { kind, rows, empty };
}

/** What the bar shows, as plain values (tested in Node; the DOM below draws them). */
export function selectionBarLines(s: State): { kinds: { kind: "element" | "node"; label: string; count: number; active: boolean }[]; hint: string; hintDetail: string; canClear: boolean; canUndo: boolean; through: boolean } | null {
  if (!s.mesh) return null;
  const elements = selectedElementsOf(s).length, nodes = s.selection.nodes.length;
  const active = s.selection.active;
  return {
    kinds: [
      { kind: "element", label: "Elements", count: elements, active: active === "element" },
      { kind: "node", label: "Nodes", count: nodes, active: active === "node" },
    ],
    // Short, so the bar fits beside the right rail; the drag rule is in the tooltip, and the box's own count says "inside" or "touching".
    hint: "Shift adds · Ctrl removes",
    hintDetail: "Click or drag a box. A box drawn left to right takes what is fully inside it; right to left, what it touches. Shift adds, Ctrl removes, Ctrl+Z undoes.",
    canClear: (active === "node" ? nodes : elements) > 0,
    canUndo: s.selectionUndo.past.length > 0,
    through: s.selection.through,
  };
}

export function mountSelectionBar(store: Store): () => void {
  const bar = el("div");
  bar.id = "selection-bar";
  bar.hidden = true;
  document.body.append(bar);
  let unlisten: (() => void)[] = [];
  let listOpen = false;
  const button = (label: string, title: string, disabled: string | null, on: () => void) => {
    const b = el("button", "window-toggle", label);
    b.type = "button";
    b.title = disabled ?? title;
    b.disabled = disabled !== null;
    unlisten.push(listen(b, "click", on));
    return b;
  };
  const setList = (sets: NonNullable<ReturnType<typeof setListLines>>): HTMLElement => {
    const box = el("div", "selection-sets");
    if (sets.rows.length === 0) box.append(el("div", "selection-sets-empty", sets.empty));
    for (const r of sets.rows) {
      const row = el("div", "selection-set");
      row.append(
        el("span", "selection-set-name", r.name),
        el("span", "selection-set-detail", r.detail),
        button("Select", `Select the set's ${sets.kind}s, replacing the selection`, null, () => store.dispatch({ type: "selectSet", row: r.row, mode: "replace" })),
        button("+", "Add the set's members to the selection", null, () => store.dispatch({ type: "selectSet", row: r.row, mode: "add" })),
        button("−", "Take the set's members out of the selection", null, () => store.dispatch({ type: "selectSet", row: r.row, mode: "remove" })),
        button("Hide", "Hide the set's elements", r.noHide, () => store.dispatch({ type: "setHidden", decls: r.decls, hidden: true })),
        button("Isolate", "Show only the set's elements", r.noHide, () => store.dispatch({ type: "isolate", decls: r.decls })),
      );
      box.append(row);
    }
    return box;
  };
  // A press outside the bar closes the list.
  const outside = (ev: PointerEvent) => {
    if (listOpen && !bar.contains(ev.target as Node)) {
      listOpen = false;
      render(store.get(), null);
    }
  };
  document.addEventListener("pointerdown", outside);
  const render = (s: State, prev: State | null) => {
    if (prev && s.selection === prev.selection && s.selectionUndo === prev.selectionUndo && s.mesh === prev.mesh && s.decls === prev.decls && s.workspace === prev.workspace) return;
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
    const hint = el("span", "selection-hint", lines.hint);
    hint.title = lines.hintDetail;
    parts.push(hint);
    const undo = el("button", "window-toggle", "Undo");
    undo.type = "button";
    undo.title = "Undo the last selection change (Ctrl+Z); Ctrl+Y redoes it";
    undo.disabled = !lines.canUndo;
    unlisten.push(listen(undo, "click", () => store.dispatch({ type: "undoSelection" })));
    parts.push(undo);
    const through = el("button", `window-toggle${lines.through ? " on" : ""}`, "Through");
    through.type = "button";
    through.title = lines.through
      ? "A box selects through the model: every drawn target in it, hidden behind others or not (section planes and hidden elements still leave theirs out)"
      : "A box selects what is visible; turn on to select through the model";
    through.setAttribute("aria-pressed", String(lines.through));
    unlisten.push(listen(through, "click", () => store.dispatch({ type: "setSelectThrough", on: !lines.through })));
    parts.push(through);
    const sets = setListLines(s)!;
    const setsBtn = el("button", `window-toggle${listOpen ? " on" : ""}`, `Sets ${sets.rows.length}`);
    setsBtn.type = "button";
    setsBtn.title = `The saved ${sets.kind} sets: select, add, subtract, hide or isolate one`;
    setsBtn.setAttribute("aria-expanded", String(listOpen));
    unlisten.push(listen(setsBtn, "click", () => {
      listOpen = !listOpen;
      render(store.get(), null);
    }));
    parts.push(setsBtn);
    if (listOpen) parts.push(setList(sets));
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
    document.removeEventListener("pointerdown", outside);
    unsubscribe();
    for (const u of unlisten) u();
    bar.remove();
  };
}
